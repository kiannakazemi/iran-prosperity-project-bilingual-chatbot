"""
Production retrieval pipeline for the bilingual RAG chatbot.

Dense vector search (Cohere ``embed-multilingual-v3.0`` + Qdrant), optionally
narrowed by Cohere reranking, then two context-repair passes so downstream
generation sees whole sections rather than fragments:

1. **Split-sibling expansion** — re-attaches the other parts of a section the
   chunker split for size (``split_part`` metadata).
2. **Section-continuation stitching** — rejoins a section the source document
   continued across a page break with a "(cont.)" heading, where the two halves
   share no linking metadata.

Reranking is *section-aware*: candidates are sent to Cohere with their
``header_path`` prepended, so the reranker can distinguish identically-named
sections ("Key Priorities") repeated under different phases.

``query(rerank=False)`` is dense-only ("Test 1"); ``query(rerank=True)`` is the
production path ("Test 2"). See ``eval/EVALUATION.md``.

Usage:
    from indexing.retrieval import ReliableRetriever
"""

from __future__ import annotations

import io
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace"
    )

from dotenv import find_dotenv, load_dotenv
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters
from llama_index.vector_stores.qdrant import QdrantVectorStore

from indexing.embedding.cohere_embed import COHERE_MODEL, make_cohere_embed_model
from qdrant_client import QdrantClient

# Load project-root .env so COHERE_API_KEY etc. are available when this
# module is imported from any working directory.
load_dotenv(find_dotenv(usecwd=True), override=False)

# ── Paths & constants ────────────────────────────────────────────────
# This file lives at indexing/retrieval/, so two levels up is indexing/.
INDEXING_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_DIR = INDEXING_ROOT / "chunking" / "by_white_paper_validated" / "chunks"
QDRANT_PATH = INDEXING_ROOT / "embedding" / "qdrant_db"
COLLECTION_NAME = "emergency_phase_cohere_v3_validated"

EMBED_MODEL = COHERE_MODEL

DENSE_TOP_K = 15
# Candidate pool pulled from dense before reranking down to final_top_k.
RERANK_CANDIDATE_POOL = 40

# ── Header sibling routing (LLM-based) ───────────────────────────────
# When enabled, after split-sibling expansion the retriever asks the LLM
# which sibling subsections under the same parent as the ranked chunks are
# also relevant to the query, and pulls those chunks. Targets the "we
# retrieved sibling A of a section but missed sibling D" failure mode.
HEADER_ROUTER_MAX_CANDIDATES = 8      # candidate siblings shown to the LLM
HEADER_ROUTER_MAX_SELECTIONS = 3      # max subsections the LLM may pick

# ── Section-continuation stitching ───────────────────────────────────
# The source document continues a section across a page break with a "(cont.)"
# heading (English) or "(ادامه)" (Persian). The chunker cuts at that break and
# the two halves end up in chunks with no linking metadata — no shared
# split_part, no shared header parent — so a list like ADVISORS can be
# retrieved half-complete. Chunks linked this way are stitched back together.
CONTINUATION_RE = re.compile(
    r"\(\s*(?:cont\.?|cont'd|continued|ادامه)\s*\)", re.IGNORECASE
)
MAX_CONTINUATION_HOPS = 3   # supports a section spanning up to 4 chunks

TEXT_PATTERN = re.compile(r"^--- text \(\d+ chars\) ---$", re.MULTILINE)


# ── Data classes ─────────────────────────────────────────────────────
@dataclass
class RetrievalResult:
    """Single ranked result returned by the pipeline."""

    text: str
    metadata: dict
    rerank_score: float  # dense similarity, or rerank relevance when rerank=True
    source: str  # "dense", "rerank", or with "+sibling" suffix


@dataclass
class _Hit:
    """Internal intermediate hit (dense or reranked)."""

    chunk_id: str
    text: str
    metadata: dict
    score: float
    source: str


# ── Main retriever ───────────────────────────────────────────────────
class ReliableRetriever:
    """Dense retriever with optional rerank, then sibling expansion."""

    def __init__(
        self,
        *,
        index: VectorStoreIndex | None = None,
        qdrant_path: Path = QDRANT_PATH,
        chunks_dir: Path = CHUNKS_DIR,
        embed_model: str = EMBED_MODEL,
    ):
        print(f"  Loading embedding model ({embed_model}) …")
        Settings.embed_model = make_cohere_embed_model(embed_model)
        print("  Embedding model ready.")

        if index is not None:
            self._index = index
        else:
            last_err: Exception | None = None
            for attempt in range(10):
                lock_file = qdrant_path / ".lock"
                if lock_file.exists():
                    try:
                        lock_file.unlink()
                    except OSError:
                        pass
                try:
                    client = QdrantClient(path=str(qdrant_path))
                    vector_store = QdrantVectorStore(
                        client=client, collection_name=COLLECTION_NAME
                    )
                    self._index = VectorStoreIndex.from_vector_store(vector_store)
                    break
                except Exception as e:
                    last_err = e
                    es = str(e).lower()
                    recoverable = any(
                        w in es
                        for w in (
                            "already accessed",
                            "alreadylocked",
                            "permission denied",
                            "locked",
                        )
                    )
                    if attempt < 9 and recoverable:
                        time.sleep(0.5 * (attempt + 1))
                        continue
                    break
            if not hasattr(self, "_index"):
                raise RuntimeError(
                    "Could not open Qdrant local storage after retries"
                ) from last_err
        print("  Dense index loaded.")

        # Reranker is built lazily on first use so the dense-only path needs no
        # rerank client.
        self._reranker = None

        # Load chunk records by id so sibling expansion can fetch adjacent parts.
        self._by_chunk_id: dict[str, dict] = {}
        self._load_chunks(chunks_dir)

        # Header index (built from _by_chunk_id): {(lang, wp): {parent_path: {subsection: [chunk_ids]}}}
        # Used by the LLM-based header sibling router.
        self._header_index: dict[tuple[str, str], dict[str, dict[str, list[str]]]] = {}
        self._build_header_index()

    # ── Initialisation helpers ───────────────────────────────────────

    def _load_chunks(self, chunks_dir: Path) -> None:
        count = 0
        for path in sorted(chunks_dir.rglob("chunk_*.txt")):
            raw = path.read_text(encoding="utf-8")
            metadata, text = _parse_chunk(raw)
            if not text.strip():
                continue
            if metadata.get("language", "") in ("en", "fa"):
                cid = metadata.get("chunk_id", "")
                self._by_chunk_id[cid] = {
                    "chunk_id": cid, "text": text, "metadata": metadata
                }
                count += 1
        print(f"  Loaded {count} chunks for sibling expansion.")

    def _build_header_index(self) -> None:
        """Group chunks by (language, white_paper) → parent → subsection.

        For a chunk with header_path ``/III. EXAMPLES/D. BRITAIN/`` we
        record parent = ``/III. EXAMPLES/`` and subsection = ``D. BRITAIN``,
        pointing at that chunk_id (plus any siblings from the same subsection).
        Chunks with fewer than two heading levels are skipped — they have no
        sibling structure to route over.
        """
        n_subs = 0
        for cid, rec in self._by_chunk_id.items():
            md = rec["metadata"]
            lang = md.get("language", "")
            wp = md.get("white_paper", "")
            hp = md.get("header_path", "")
            if not hp:
                continue
            parts = [p for p in hp.strip("/").split("/") if p]
            if len(parts) < 2:
                continue
            parent = "/" + "/".join(parts[:-1]) + "/"
            subsection = parts[-1]
            key = (lang, wp)
            self._header_index.setdefault(key, {})
            self._header_index[key].setdefault(parent, {})
            bucket = self._header_index[key][parent]
            if subsection not in bucket:
                bucket[subsection] = []
                n_subs += 1
            bucket[subsection].append(cid)
        print(f"  Header index built: {n_subs} subsections across {len(self._header_index)} (lang, white_paper) buckets.")

    # ── Public API ───────────────────────────────────────────────────

    def query(
        self,
        query_text: str,
        language: str,
        *,
        final_top_k: int = DENSE_TOP_K,
        rerank: bool = False,
        candidate_pool: int = RERANK_CANDIDATE_POOL,
    ) -> list[RetrievalResult]:
        """Retrieve passages, then expand split-chunk siblings.

        Two variants, matching the evaluation arms:

        ==================  ==============================================
        Flags               Variant
        ==================  ==============================================
        (defaults)          Test 1 — dense only
        ``rerank=True``     Test 2 — dense + rerank  (production)
        ==================  ==============================================

        With ``rerank=False`` this is dense-only: dense top ``final_top_k`` →
        expansion. With ``rerank=True`` a wider dense ``candidate_pool`` is
        pulled and Cohere reranks it down to ``final_top_k`` first;
        ``rerank_score`` carries the dense similarity or the rerank relevance
        accordingly.

        Both paths then run split-sibling expansion and section-continuation
        stitching.
        """
        if rerank:
            hits = self.rerank_search(
                query_text, language, candidate_pool, final_top_k
            )
        else:
            hits = self.dense_search(query_text, language, final_top_k)

        results = [
            RetrievalResult(
                text=h.text,
                metadata=h.metadata,
                rerank_score=h.score,
                source=h.source,
            )
            for h in hits
        ]
        results = self._expand_split_siblings(results)
        # Always on: a half-retrieved list is worse than a slightly larger
        # context, and this only fires when a "(cont.)" link actually exists.
        results = self._expand_continuations(results)
        return results

    # ── Dense retrieval ──────────────────────────────────────────────

    def dense_search(self, query: str, language: str, top_k: int) -> list[_Hit]:
        """Dense vector search, language-filtered, returning ``top_k`` hits."""
        lang_filter = MetadataFilters(
            filters=[MetadataFilter(key="language", value=language)]
        )
        retriever = self._index.as_retriever(
            similarity_top_k=top_k, filters=lang_filter
        )
        nodes = retriever.retrieve(query)
        return [
            _Hit(
                chunk_id=n.node.metadata.get("chunk_id", ""),
                text=n.node.text,
                metadata=dict(n.node.metadata),
                score=n.score if n.score is not None else 0.0,
                source="dense",
            )
            for n in nodes
        ]

    # Backwards-compatible private alias.
    def _dense(self, query: str, language: str, top_k: int) -> list[_Hit]:
        return self.dense_search(query, language, top_k)

    # ── Rerank retrieval ─────────────────────────────────────────────

    @staticmethod
    def rerank_document(hit: _Hit) -> str:
        """Build the string handed to the reranker for one candidate.

        The stored chunk text already carries ``[Summary: …]`` and — except in
        Front Matter — ``[Topic: <white paper>]``. What it never carries is the
        ``header_path``, so the reranker cannot see which section or phase a
        passage belongs to. That is invisible for most questions and decisive for
        a few: the Healthcare paper repeats "Objectives" and "Key Priorities"
        under every phase, so "priorities for the first 30 days" gives the
        reranker several identical-looking candidates with nothing to choose
        between them.

        This prepends the heading trail, and the white paper when the chunker
        omitted it, so the reranker scores against section context as well as
        body text.
        """
        md = hit.metadata
        lines: list[str] = []

        wp = (md.get("white_paper") or "").strip()
        if wp and f"[Topic: {wp}]" not in hit.text:
            lines.append(f"[Topic: {wp}]")

        header = (md.get("header_path") or "").strip("/")
        if header:
            lines.append(f"[Section: {header.replace('/', ' > ')}]")

        return ("\n".join(lines) + "\n" + hit.text) if lines else hit.text

    def rerank_search(
        self, query: str, language: str, candidate_pool: int, top_n: int
    ) -> list[_Hit]:
        """Dense pool of ``candidate_pool`` → Cohere rerank → top ``top_n``."""
        pool = self.dense_search(query, language, candidate_pool)
        return self.rerank_pool(query, pool, top_n)

    def rerank_pool(
        self, query: str, pool: list[_Hit], top_n: int
    ) -> list[_Hit]:
        """Rerank an existing dense candidate pool, returning the top ``top_n``
        in reranked order. Lets a caller reuse one pool for both its dense view
        and its reranked view without a second dense query.

        Candidates are always sent as section-aware documents built by
        :meth:`rerank_document`. Sending the bare chunk text hid ``header_path``
        from the reranker, which could not then distinguish same-named sections
        ("Key Priorities") repeated under different phases.
        """
        if not pool:
            return []

        if self._reranker is None:
            from indexing.retrieval.cohere_rerank import make_cohere_reranker

            self._reranker = make_cohere_reranker()

        docs = [self.rerank_document(h) for h in pool]
        ranked = self._reranker.rerank(query, docs, top_n)
        out: list[_Hit] = []
        for orig_idx, relevance in ranked:
            h = pool[orig_idx]
            out.append(
                _Hit(
                    chunk_id=h.chunk_id,
                    text=h.text,
                    metadata=h.metadata,
                    score=relevance,
                    source="rerank",
                )
            )
        return out

    # ── Sibling expansion for split chunks ────────────────────────────

    def _expand_split_siblings(
        self, results: list[RetrievalResult]
    ) -> list[RetrievalResult]:
        """If a result has split_part (e.g. '1/2'), fetch immediately adjacent
        sibling parts (distance ≤ MAX_SIBLINGS_PER_PARENT) so the LLM sees
        the surrounding context without ballooning the passage count."""
        MAX_SIBLINGS_PER_PARENT = 2  # only fetch parts within distance 2

        seen_ids: set[str] = {r.metadata.get("chunk_id", "") for r in results}
        extras: list[RetrievalResult] = []

        for r in results:
            split = r.metadata.get("split_part", "")
            if "/" not in split:
                continue
            _cur, total_str = split.split("/", 1)
            total = int(total_str)
            chunk_id = r.metadata.get("chunk_id", "")
            chunk_idx = int(r.metadata.get("chunk_index", 0))

            base_prefix = re.sub(r"\d+$", "", chunk_id)
            cur_part = int(_cur)
            offset = chunk_idx - cur_part

            for part_num in range(1, total + 1):
                if part_num == cur_part:
                    continue
                # Only immediately adjacent siblings (distance 1-2)
                if abs(part_num - cur_part) > MAX_SIBLINGS_PER_PARENT:
                    continue
                sibling_idx = offset + part_num
                sibling_id = f"{base_prefix}{sibling_idx:05d}"
                if sibling_id in seen_ids:
                    continue

                rec = self._by_chunk_id.get(sibling_id)
                if rec is None:
                    continue

                extras.append(
                    RetrievalResult(
                        text=rec["text"],
                        metadata=rec["metadata"],
                        rerank_score=r.rerank_score,
                        source=r.source + "+sibling",
                    )
                )
                seen_ids.add(sibling_id)

        if extras:
            results = list(results) + extras

        return results

    # ── Section-continuation stitching ───────────────────────────────

    def _expand_continuations(
        self,
        results: list[RetrievalResult],
        *,
        max_hops: int = MAX_CONTINUATION_HOPS,
    ) -> list[RetrievalResult]:
        """Stitch back together a section the chunker split across chunks.

        The source document marks a section running over a page break with a
        "(cont.)" heading — ``# ADVISORS`` then ``# ADVISORS (cont.)``. When the
        chunker cuts at that break the two halves end up in separate chunks with
        **no linking metadata**: they get different ``header_path`` values and no
        ``split_part``, so neither :meth:`_expand_split_siblings` (which needs
        ``split_part``) nor :meth:`_expand_header_siblings` (which needs a shared
        parent path) can connect them.

        The observed failure: asking for the advisors retrieves
        ``/ADVISORS (cont.)/`` — whose heading matches the query — but not the
        preceding chunk holding the *first half* of the same list, because that
        chunk is labelled ``/AUTHORS (cont.)/``. The user gets half the list with
        no indication anything is missing.

        Rule applied here, in both directions and keyed on the document's own
        convention:

        - a retrieved chunk whose heading says "(cont.)" continues something, so
          pull the chunk before it;
        - a retrieved chunk whose *next* chunk says "(cont.)" is continued there,
          so pull the chunk after it.

        Followed transitively up to ``max_hops`` so a list spanning three or more
        chunks arrives whole.
        """
        if not results:
            return results

        by_index: dict[tuple[str, str, int], dict] = {}
        for rec in self._by_chunk_id.values():
            md = rec["metadata"]
            key = (
                md.get("language", ""),
                md.get("white_paper", ""),
                int(md.get("chunk_index", -1)),
            )
            by_index[key] = rec

        def is_continuation(rec: dict | None) -> bool:
            if not rec:
                return False
            return bool(CONTINUATION_RE.search(rec["metadata"].get("header_path", "")))

        seen_ids: set[str] = {r.metadata.get("chunk_id", "") for r in results}
        extras: list[RetrievalResult] = []
        frontier = list(results)

        for _hop in range(max_hops):
            next_frontier: list[RetrievalResult] = []
            for r in frontier:
                md = r.metadata
                lang = md.get("language", "")
                wp = md.get("white_paper", "")
                try:
                    idx = int(md.get("chunk_index", -1))
                except (TypeError, ValueError):
                    continue

                wanted: list[int] = []
                # This chunk continues the previous one.
                if CONTINUATION_RE.search(md.get("header_path", "")):
                    wanted.append(idx - 1)
                # The next chunk continues this one.
                if is_continuation(by_index.get((lang, wp, idx + 1))):
                    wanted.append(idx + 1)

                for nidx in wanted:
                    rec = by_index.get((lang, wp, nidx))
                    if rec is None:
                        continue
                    cid = rec["metadata"].get("chunk_id", "")
                    if not cid or cid in seen_ids:
                        continue
                    new = RetrievalResult(
                        text=rec["text"],
                        metadata=rec["metadata"],
                        rerank_score=r.rerank_score,
                        source=r.source + "+continuation",
                    )
                    extras.append(new)
                    next_frontier.append(new)
                    seen_ids.add(cid)

            if not next_frontier:
                break
            frontier = next_frontier

        if extras:
            results = list(results) + extras
        return results

    # ── Bulk load: fetch every chunk for a (language, white_paper) ────

    def get_white_paper_chunks(
        self, language: str, white_paper: str
    ) -> list[RetrievalResult]:
        """Return every chunk belonging to ``(language, white_paper)``, sorted
        by ``chunk_index`` in ascending order. Used by the project-overview
        route to bulk-load the Front Matter into context.

        The white_paper argument is matched exactly against the metadata
        field of the same name (e.g. ``"Front Matter"``, ``"LEGAL"``).
        """
        matching = [
            rec for rec in self._by_chunk_id.values()
            if rec["metadata"].get("language") == language
            and rec["metadata"].get("white_paper") == white_paper
        ]
        matching.sort(key=lambda r: int(r["metadata"].get("chunk_index", 0)))
        return [
            RetrievalResult(
                text=rec["text"],
                metadata=rec["metadata"],
                rerank_score=1.0,
                source="front_matter_dump" if white_paper.lower() == "front matter" else "whitepaper_dump",
            )
            for rec in matching
        ]

    # ── LLM-based header sibling routing ──────────────────────────────

    # ── PARKED: LLM header-sibling router (not in the active pipeline) ──
    # Removed from production and from the evaluation after measurement showed
    # it changed the retrieved set on only 5/47 English and 9/47 Persian
    # questions — meaning most observed "router effects" were model
    # nondeterminism, not retrieval. On the rows where it did fire, every chunk
    # it added was already inside the dense candidate pool, just ranked below
    # top-5, which is what section-aware reranking now handles directly.
    #
    # Kept rather than deleted because this project is not under version
    # control, so a deletion would be unrecoverable. Nothing calls it; the
    # header index it relies on is still built in __init__ and is cheap.
    def _expand_header_siblings(
        self,
        results: list[RetrievalResult],
        query_text: str,
        *,
        max_candidates: int = HEADER_ROUTER_MAX_CANDIDATES,
        max_selections: int = HEADER_ROUTER_MAX_SELECTIONS,
    ) -> tuple[list[RetrievalResult], list[RetrievalResult]]:
        """Ask the LLM which sibling subsections (same parent, different last
        heading level) are relevant to the query, and pull those chunks.

        Returns ``(all_results, header_only_additions)``. If no candidates
        exist or the LLM selects none, ``header_only_additions`` is empty.
        """
        if not results:
            return results, []

        # Chunks + subsections we already have covered.
        seen_ids: set[str] = {r.metadata.get("chunk_id", "") for r in results}
        covered_subsections: set[tuple[str, str, str, str]] = set()  # (lang, wp, parent, subsection)
        for r in results:
            md = r.metadata
            hp = md.get("header_path", "")
            parts = [p for p in hp.strip("/").split("/") if p]
            if len(parts) < 2:
                continue
            parent = "/" + "/".join(parts[:-1]) + "/"
            subsection = parts[-1]
            covered_subsections.add((md.get("language", ""), md.get("white_paper", ""), parent, subsection))

        # Gather candidate siblings (same parent, different subsection, not covered).
        candidates: list[dict] = []
        for (lang, wp, parent, _sub) in covered_subsections:
            parent_map = self._header_index.get((lang, wp), {}).get(parent, {})
            for sib_sub, chunk_ids in parent_map.items():
                if (lang, wp, parent, sib_sub) in covered_subsections:
                    continue
                if any((lang, wp, parent, sib_sub) == (c["lang"], c["wp"], c["parent"], c["subsection"]) for c in candidates):
                    continue  # dedupe candidates across covered parents
                # Sample summary from the first chunk in the subsection.
                sample = self._by_chunk_id.get(chunk_ids[0], {})
                sample_md = sample.get("metadata", {})
                candidates.append({
                    "lang": lang,
                    "wp": wp,
                    "parent": parent,
                    "subsection": sib_sub,
                    "chunk_ids": chunk_ids,
                    "summary": sample_md.get("summary", "") or "(no summary)",
                    "header_label": f"{parent.strip('/')} > {sib_sub}",
                })

        if not candidates:
            return results, []

        # Cap candidates presented to the LLM.
        if len(candidates) > max_candidates:
            candidates = candidates[:max_candidates]

        # Build the prompt.
        retrieved_lines = []
        for r in results[:5]:
            hp = r.metadata.get("header_path", "").strip("/")
            summary = r.metadata.get("summary", "")
            if hp:
                retrieved_lines.append(f"- {hp}  ({summary})")
        candidate_lines = []
        for i, c in enumerate(candidates, 1):
            candidate_lines.append(f"{i}. {c['header_label']}\n   Summary: {c['summary']}")

        system = (
            "You are a section router for a policy-document Q&A system. "
            "Given the user's question and a list of candidate document subsections, "
            "identify which subsections likely contain information relevant to answering the question. "
            "Reply with the comma-separated numbers of relevant candidates (e.g. '1, 3'), "
            "or 'none' if none are relevant. "
            f"Select at most {max_selections} candidates. Numbers only, no explanation."
        )
        user = (
            f"Question: {query_text}\n\n"
            f"Already retrieved subsections:\n" + "\n".join(retrieved_lines) + "\n\n"
            f"Candidate sibling subsections (in the same document, not yet retrieved):\n"
            + "\n".join(candidate_lines) + "\n\n"
            "Which candidates likely contain relevant material? Numbers only, or 'none'."
        )

        # Call the LLM. Imported lazily to avoid a circular import at module load.
        try:
            from chatbot.engine import _gemini_generate
            response = _gemini_generate(system, user, max_tokens=48, temperature=0.0)
        except Exception:
            # If the router call fails, degrade gracefully: no header expansion.
            return results, []

        # Parse response — extract numbers.
        response_lower = (response or "").strip().lower()
        if "none" in response_lower and not any(ch.isdigit() for ch in response_lower):
            return results, []
        picks: list[int] = []
        for tok in re.findall(r"\d+", response_lower):
            n = int(tok)
            if 1 <= n <= len(candidates) and n not in picks:
                picks.append(n)
            if len(picks) >= max_selections:
                break

        # Fetch chunks from selected subsections.
        added: list[RetrievalResult] = []
        for n in picks:
            c = candidates[n - 1]
            for cid in c["chunk_ids"]:
                if cid in seen_ids:
                    continue
                rec = self._by_chunk_id.get(cid)
                if rec is None:
                    continue
                added.append(RetrievalResult(
                    text=rec["text"],
                    metadata=rec["metadata"],
                    rerank_score=0.0,
                    source="header_sibling",
                ))
                seen_ids.add(cid)

        return list(results) + added, added


# ── Utility functions ────────────────────────────────────────────────

def _parse_chunk(raw: str) -> tuple[dict, str]:
    """Parse a chunk_*.txt file into (metadata_dict, text_body)."""
    m = TEXT_PATTERN.search(raw)
    if not m:
        return {}, raw

    meta_block = raw[: m.start()].strip()
    text_body = raw[m.end() :].lstrip("\n")

    metadata: dict[str, str] = {}
    for line in meta_block.splitlines():
        if line.startswith("---"):
            continue
        key, _, value = line.partition(":")
        if key.strip():
            metadata[key.strip()] = value.strip()

    return metadata, text_body
