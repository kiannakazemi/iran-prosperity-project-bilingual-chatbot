"""
Convert Marker-produced Markdown into structured, RAG-ready text chunks.

The script loads Markdown files, removes pagination artifacts, parses the
content by headings with LlamaIndex, enriches chunks with metadata such as
language, topic/section, and page range, and applies post-processing to
improve retrieval quality, including merging small chunks, splitting large
ones, and adding brief summaries.

Processed chunks are saved as text files with metadata headers in
language-specific output directories.

Usage (run as a script — no package wrapper required):
    python indexing/chunking/chunking.py     # from rag_pipeline/
    python chunking.py                       # from indexing/chunking/
"""

from __future__ import annotations

import io
import re
import shutil
import sys
from pathlib import Path

# Make rag_pipeline/ importable so 'from chatbot.engine import ...' resolves
# when this file is executed directly (not as a package module).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from llama_index.core import Document
from llama_index.core.node_parser import MarkdownNodeParser

# This module lives at rag_pipeline/indexing/chunking/, so go up TWO levels
# (chunking/ → indexing/ → rag_pipeline/) to reach REPO_ROOT.
CHUNKING_PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = CHUNKING_PKG_DIR.parent.parent
MD_DIR = REPO_ROOT / "document_preprocessing" / "iran_prosperity_project_md"
CHUNKS_DIR = CHUNKING_PKG_DIR / "chunks_raw"

MIN_CHUNK_CHARS = 300
MAX_CHUNK_CHARS = 1500
OVERLAP_CHARS = 100

PAGE_BREAK_RE = re.compile(r"\{(\d+)\}-{10,}\n?")
FOOTNOTE_LINE_RE = re.compile(
    r"^\d{1,3}\s+(?:See |The |This |In |For |Under |As |Note )",
    re.MULTILINE,
)

# Canonical white paper topic prefixes (the part before ":").
# These are the ONLY strings accepted as white paper boundaries.
_EN_TOPICS = [
    "LEGAL", "POLITICAL", "MILITARY AND SECURITY", "FOREIGN POLICY",
    "GOVERNMENT ESSENTIAL FUNCTIONS", "MACROECONOMIC GOVERNANCE",
    "NATIONAL ASSETS", "ENERGY", "INDUSTRY", "CYBERSECURITY",
    "ENVIRONMENT", "WATER", "HEALTHCARE", "EDUCATIONAL SYSTEM",
]
_FA_TOPICS = [
    "حقوقی", "سیاسی", "نظامی و امنیتی", "سیاست خارجی",
    "کارکردهای اصلی و ضروری دولت", "اقتصاد کلان",
    "داراییهای ملی", "انرژی", "صنعت", "امنیت سایبری",
    "محیط زیست", "آب", "بهداشت و درمان", "نظام آموزشی",
]
_ALL_TOPICS = set(_EN_TOPICS + _FA_TOPICS)


# ── Language helpers ─────────────────────────────────────────────────

def _language_from_filename(name: str) -> str:
    upper = name.upper()
    if "PERSIAN" in upper:
        return "fa"
    if "ENGLISH" in upper:
        return "en"
    return "unknown"


def _make_chunk_id(*, language: str, source_stem: str, chunk_index: int) -> str:
    safe = source_stem.replace(" ", "_")
    return f"{language}__{safe}__{chunk_index:05d}"


# ── Step 1: Pre-process markdown ────────────────────────────────────

def _build_page_map(raw_text: str) -> list[tuple[int, int]]:
    """Return [(char_offset_in_cleaned, page_number), ...].

    For each page marker in the raw text, find the first substantive line
    of that page's content and locate it in the cleaned text via sequential
    search.  This avoids position-drift issues from footnote removal and
    newline collapsing.
    """
    cleaned = _clean_markdown(raw_text)
    entries: list[tuple[int, int]] = []
    search_start = 0

    for m in PAGE_BREAK_RE.finditer(raw_text):
        page_num = int(m.group(1))

        # Grab the first ~500 chars after this page marker in the raw text.
        after = raw_text[m.end() : m.end() + 500]

        # Find the first non-empty, non-marker line to use as anchor text.
        anchor = ""
        for line in after.split("\n"):
            stripped = line.strip()
            if stripped and not PAGE_BREAK_RE.match(stripped):
                anchor = stripped[:60]
                break

        if anchor:
            pos = cleaned.find(anchor, search_start)
            if pos >= 0:
                entries.append((pos, page_num))
                search_start = pos
                continue

        # Fallback: no anchor found (e.g. blank page) — carry forward.
        entries.append((search_start, page_num))

    return entries


def _page_for_position(page_map: list[tuple[int, int]], pos: int) -> int:
    """Given a character offset in cleaned text, return its page number."""
    page = 0
    for map_pos, map_page in page_map:
        if map_pos <= pos:
            page = map_page
        else:
            break
    return page


def _clean_markdown(text: str) -> str:
    """Strip page-break markers and inline footnote lines."""
    text = PAGE_BREAK_RE.sub("", text)
    text = FOOTNOTE_LINE_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


# ── Step 2: White paper map ─────────────────────────────────────────

def _build_white_paper_map(text: str) -> list[tuple[int, str]]:
    """Return [(char_offset, "Topic Name"), ...] for each white paper title.

    Scans both ``# `` and ``## `` headings and only accepts lines whose
    topic prefix (the part before ``:``) is in the known white paper list.
    """
    heading_re = re.compile(
        r"^#{1,2}\s+(?:\*\*)?(.+?)(?:\*\*)?$",
        re.MULTILINE,
    )
    entries: list[tuple[int, str]] = []
    for m in heading_re.finditer(text):
        full = m.group(1).strip().strip("*").strip()
        if ":" not in full:
            continue
        topic_prefix = full.split(":")[0].strip().strip("*").strip()
        if topic_prefix in _ALL_TOPICS:
            entries.append((m.start(), topic_prefix))
    return entries


def _assign_white_paper(
    node_text: str,
    wp_map: list[tuple[int, str]],
    full_text: str,
) -> str:
    """Find which white paper a chunk belongs to by locating its text
    in the original markdown and checking which white paper boundary
    it falls under."""
    snippet = node_text[:120].strip()
    if not snippet:
        return "Front Matter"

    pos = full_text.find(snippet)
    if pos == -1:
        normalized = re.sub(r"\s+", " ", snippet[:80])
        normalized_full = re.sub(r"\s+", " ", full_text)
        pos = normalized_full.find(normalized)
    if pos == -1:
        return "Front Matter"

    current = "Front Matter"
    for wp_pos, wp_name in wp_map:
        if wp_pos <= pos:
            current = wp_name
        else:
            break
    return current


# ── Step 3: Load & parse ────────────────────────────────────────────

def load_documents(md_dir: Path = MD_DIR) -> list[Document]:
    paths = sorted(md_dir.glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"No .md files found in {md_dir}")

    documents: list[Document] = []
    for path in paths:
        raw = path.read_text(encoding="utf-8")
        page_map = _build_page_map(raw)
        cleaned = _clean_markdown(raw)
        source_pdf = path.stem + ".pdf"
        documents.append(
            Document(
                text=cleaned,
                metadata={
                    "language": _language_from_filename(path.name),
                    "source_file": path.name,
                    "source_pdf": source_pdf,
                    "_raw_text": raw,
                    "_page_map": page_map,
                },
            )
        )
    return documents


def _parse_nodes(documents: list[Document]) -> list:
    parser = MarkdownNodeParser()
    return parser.get_nodes_from_documents(documents)


# ── Step 4a: Assign white paper metadata ─────────────────────────────

def _enrich_with_white_paper(nodes: list, documents: list[Document]) -> None:
    doc_raw: dict[str, str] = {}
    doc_wp_map: dict[str, list[tuple[int, str]]] = {}
    for doc in documents:
        src = doc.metadata["source_file"]
        raw = doc.metadata.get("_raw_text", doc.text)
        cleaned = _clean_markdown(raw)
        doc_raw[src] = cleaned
        doc_wp_map[src] = _build_white_paper_map(cleaned)

    for node in nodes:
        src = node.metadata.get("source_file", "")
        ft = doc_raw.get(src, "")
        wp_map = doc_wp_map.get(src, [])
        wp = _assign_white_paper(node.text, wp_map, ft)
        node.metadata["white_paper"] = wp


# ── Step 4a-bis: Assign page numbers ─────────────────────────────────

def _find_text_position(text: str, cleaned: str) -> int:
    """Find where a chunk's text starts in the cleaned document.
    Tries exact match on first 120 chars, then normalized fallback."""
    snippet = text[:120].strip()
    if not snippet:
        return -1
    pos = cleaned.find(snippet)
    if pos != -1:
        return pos
    normalized = re.sub(r"\s+", " ", snippet[:80])
    normalized_full = re.sub(r"\s+", " ", cleaned)
    pos = normalized_full.find(normalized)
    return pos


def _enrich_with_page_numbers(nodes: list, documents: list[Document]) -> None:
    """Assign page_start and page_end metadata to each chunk.

    Must be called AFTER merge/split so that each final chunk gets the
    correct page range based on its actual text content.
    """
    doc_cleaned: dict[str, str] = {}
    doc_page_map: dict[str, list[tuple[int, int]]] = {}

    for doc in documents:
        src = doc.metadata["source_file"]
        doc_cleaned[src] = doc.text
        doc_page_map[src] = doc.metadata.get("_page_map", [])

    for node in nodes:
        src = node.metadata.get("source_file", "")
        cleaned = doc_cleaned.get(src, "")
        page_map = doc_page_map.get(src, [])

        # Strip [Summary: ...] and [Topic: ...] prefixes if those steps already ran
        raw_text = re.sub(r"^(\[(?:Summary|Topic): [^\]]+\]\s*)+", "", node.text)

        if not page_map:
            node.metadata["page_start"] = "0"
            node.metadata["page_end"] = "0"
            continue

        # Find start position from beginning of chunk
        start_pos = _find_text_position(raw_text, cleaned)
        if start_pos == -1:
            start_pos = 0

        # Find end position from last 120 chars independently
        end_snippet = raw_text[-120:].strip() if len(raw_text) > 120 else ""
        if end_snippet:
            end_match = cleaned.rfind(end_snippet)
            if end_match != -1:
                end_pos = end_match + len(end_snippet)
            else:
                end_pos = start_pos + len(raw_text)
        else:
            end_pos = start_pos + len(raw_text)

        page_start = _page_for_position(page_map, start_pos)
        page_end = _page_for_position(page_map, end_pos)

        node.metadata["page_start"] = str(page_start)
        node.metadata["page_end"] = str(page_end)


# ── Step 4b: Merge small consecutive chunks ──────────────────────────

def _merge_small_chunks(nodes: list) -> list:
    """Merge consecutive chunks from the same document that are below
    MIN_CHUNK_CHARS, accumulating until the merged text >= threshold."""
    if not nodes:
        return nodes

    merged: list = []
    buffer_node = None

    def _same_group(a, b) -> bool:
        return (
            a.metadata.get("source_file") == b.metadata.get("source_file")
            and a.metadata.get("white_paper") == b.metadata.get("white_paper")
        )

    def _flush():
        nonlocal buffer_node
        if buffer_node is not None:
            merged.append(buffer_node)
            buffer_node = None

    for node in nodes:
        if buffer_node is None:
            buffer_node = node
            continue

        if _same_group(buffer_node, node) and len(buffer_node.text) < MIN_CHUNK_CHARS:
            buffer_node.text = buffer_node.text.rstrip() + "\n\n" + node.text.lstrip()
            if node.metadata.get("header_path", "/") != "/":
                buffer_node.metadata["header_path"] = node.metadata["header_path"]
        else:
            _flush()
            buffer_node = node

    _flush()
    return merged


# ── Step 4c: Split oversized chunks ─────────────────────────────────

def _split_large_chunks(nodes: list) -> list:
    """Split chunks exceeding MAX_CHUNK_CHARS on sentence boundaries
    with OVERLAP_CHARS of overlap between pieces."""
    result: list = []
    for node in nodes:
        if len(node.text) <= MAX_CHUNK_CHARS:
            result.append(node)
            continue

        pieces = _split_text(node.text, MAX_CHUNK_CHARS, OVERLAP_CHARS)
        for i, piece in enumerate(pieces):
            new_node = node.model_copy(deep=True)
            new_node.text = piece
            if len(pieces) > 1:
                new_node.metadata["split_part"] = f"{i + 1}/{len(pieces)}"
            result.append(new_node)

    return result


def _split_text(text: str, max_chars: int, overlap: int) -> list[str]:
    """Split text into pieces of ~max_chars, breaking at sentence
    boundaries where possible, with overlap between pieces."""
    if len(text) <= max_chars:
        return [text]

    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = start + max_chars

        if end >= len(text):
            pieces.append(text[start:])
            break

        break_point = _find_sentence_break(text, start, end)
        pieces.append(text[start:break_point])
        raw_next = max(start + 1, break_point - overlap)
        snapped = _snap_to_word_start(text, raw_next, floor=start + 1)
        start = snapped

    return pieces


def _snap_to_word_start(text: str, pos: int, floor: int = 0) -> int:
    """Move *pos* so the next chunk starts on a complete word.

    Strategy:
      1. Try snapping **backward** to the start of the current word.
      2. If that would land before *floor* (which would stall the loop),
         snap **forward** to the next whitespace boundary instead.
    This guarantees the chunk never starts mid-word AND never goes backward.
    """
    if pos <= 0 or pos >= len(text):
        return max(pos, floor)

    if text[pos].isspace() or text[pos] == "\n":
        return max(pos, floor)

    # Try backward
    back = pos
    while back > 0 and not text[back - 1].isspace():
        back -= 1

    if back >= floor:
        return back

    # Backward would stall; snap forward to next whitespace instead
    fwd = pos
    while fwd < len(text) and not text[fwd].isspace():
        fwd += 1
    return min(fwd, len(text))


def _find_sentence_break(text: str, start: int, end: int) -> int:
    """Find the best sentence break between start and end."""
    search_start = start + (end - start) // 2
    for sep in [". ", ".\n", "؟ ", ".\u200c", "\n\n", "\n- ", "\n"]:
        pos = text.rfind(sep, search_start, end)
        if pos != -1:
            return pos + len(sep)
    return end


# ── Step 4d: Prepend topic + fix header_path ────────────────────────

def _finalize_chunks(nodes: list) -> list:
    """Prepend [Topic: ...] to chunk text; fix empty header paths."""
    for node in nodes:
        wp = node.metadata.get("white_paper", "")
        lang = node.metadata.get("language", "")

        if wp and wp != "Front Matter":
            prefix = f"[Topic: {wp}]\n\n"
            if not node.text.startswith(prefix):
                node.text = prefix + node.text

        hp = node.metadata.get("header_path", "/")
        if hp == "/" and wp:
            node.metadata["header_path"] = f"/{wp}/"

    return nodes


# ── Step 4e: Per-chunk LLM summaries ────────────────────────────────

def _add_chunk_summaries(nodes: list) -> list:
    """Prepend a one-line LLM-generated summary to each chunk.

    The summary is prepended as ``[Summary: ...]`` so the embedding sees
    the topical signal in the first tokens. Re-run ``indexing.embedding.embed_and_store``
    after changing summaries to rebuild the vector index.
    """
    from chatbot.engine import _llm_complete

    _SYSTEM_EN = (
        "You write a single-line summary that describes WHAT TYPE of content "
        "a passage contains — not what the content says. Think of it as a "
        "label that tells a reader what category of material is in this "
        "chunk, so they can decide whether to read it.\n\n"
        "Rules:\n"
        "1. Output ONLY one line.\n"
        "2. Include the key words / category terms.\n"
        "3. If the chunk contains multiple types of content, mention each "
        "type briefly, joined with commas and 'and' for the last one."
    )
    _SYSTEM_FA = (
        "شما یک خلاصه تک‌خطی می‌نویسید که توصیف می‌کند این متن چه نوع "
        "محتوایی را در بر دارد — نه اینکه محتوا چه می‌گوید. این خلاصه مانند "
        "یک برچسب است که به خواننده می‌گوید چه دسته‌ای از مطالب در این تکه "
        "وجود دارد تا بتواند تصمیم بگیرد آن را بخواند یا نه.\n\n"
        "قواعد:\n"
        "۱. فقط یک خط بنویسید.\n"
        "۲. کلمات کلیدی / اصطلاحات دسته را در خلاصه بگنجانید.\n"
        "۳. اگر تکه شامل چندین نوع محتواست، هر نوع را به‌اختصار با ویرگول "
        "و «و» جدا کنید."
    )

    total = len(nodes)
    for i, node in enumerate(nodes, 1):
        lang = node.metadata.get("language", "en")
        system = _SYSTEM_FA if lang == "fa" else _SYSTEM_EN

        # Strip [Topic:] / [Summary:] prefixes before sending to the model.
        clean_text = re.sub(r"^\[(?:Topic|Summary): [^\]]+\]\s*", "", node.text).strip()

        # Heading-only / ghost chunks have almost no body once prefixes are
        # stripped. DashScope rejects empty or near-empty user content with
        # HTTP 400, so skip the API call entirely for these.
        if len(clean_text) < 30:
            if i % 50 == 0 or i == total:
                print(f"  Summarized {i}/{total} chunks …")
            continue

        try:
            summary = _llm_complete(
                system,
                "Describe what TYPE of content this passage contains, in one "
                "line, following the format and rules in the system "
                f"prompt:\n\n{clean_text[:1600]}",
                max_tokens=80,
            )
        except Exception as exc:
            print(f"  [summary] chunk {i}/{total} error: {exc}")
            summary = ""

        if summary:
            node.metadata["summary"] = summary
            node.text = f"[Summary: {summary}]\n\n{node.text}"

        if i % 50 == 0 or i == total:
            print(f"  Summarized {i}/{total} chunks …")

    return nodes


# ── Step 5: Filter + Save ───────────────────────────────────────────

LANG_FOLDER_MAP = {
    "en": "emergency_phase_english",
    "fa": "emergency_phase_persian",
}


def _write_chunk_file(path: Path, node) -> None:
    skip_keys = {"_raw_text", "_page_map"}
    # Fixed metadata field order (matches the project's expected layout).
    # Any extra metadata keys not listed here are appended at the end so
    # nothing is silently dropped if the schema grows in the future.
    field_order = [
        "chunk_id",
        "chunk_index",
        "source_file",
        "source_pdf",
        "language",
        "white_paper",
        "header_path",
        "page_start",
        "page_end",
        "split_part",
        "summary",
    ]
    content = "--- metadata ---\n"
    written: set[str] = set()
    for k in field_order:
        if k in node.metadata and k not in skip_keys:
            content += f"{k}: {node.metadata[k]}\n"
            written.add(k)
    for k, v in sorted(node.metadata.items()):
        if k in skip_keys or k in written:
            continue
        content += f"{k}: {v}\n"
    content += f"--- text ({len(node.text)} chars) ---\n"
    content += node.text
    path.write_text(content, encoding="utf-8")


def save_chunks(nodes: list, out_dir: Path = CHUNKS_DIR) -> dict[str, int]:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    per_doc: dict[tuple[str, str], int] = {}
    per_lang: dict[str, int] = {}
    for node in nodes:
        lang = node.metadata.get("language", "unknown")
        source_file = node.metadata.get("source_file", "unknown.md")
        source_stem = Path(source_file).stem

        folder_name = LANG_FOLDER_MAP.get(lang, f"emergency_phase_{lang}")
        doc_dir = out_dir / folder_name / source_stem
        doc_dir.mkdir(parents=True, exist_ok=True)

        key = (lang, source_stem)
        per_doc[key] = per_doc.get(key, 0) + 1
        idx = per_doc[key]
        chunk_id = _make_chunk_id(language=lang, source_stem=source_stem, chunk_index=idx)
        node.metadata["chunk_index"] = idx
        node.metadata["chunk_id"] = chunk_id

        _write_chunk_file(doc_dir / f"chunk_{idx}.txt", node)
        per_lang[lang] = per_lang.get(lang, 0) + 1

    return per_lang


# ── Inspect ──────────────────────────────────────────────────────────

def inspect_chunks(nodes: list) -> None:
    print(f"Total chunks: {len(nodes)}\n")

    en_count = sum(1 for n in nodes if n.metadata.get("language") == "en")
    fa_count = sum(1 for n in nodes if n.metadata.get("language") == "fa")
    other = len(nodes) - en_count - fa_count
    print(f"  English (en): {en_count}")
    print(f"  Persian (fa): {fa_count}")
    if other:
        print(f"  Other:        {other}")
    print()

    lengths = [len(n.text) for n in nodes]
    if lengths:
        avg = sum(lengths) / len(lengths)
        mn, mx = min(lengths), max(lengths)
        under_300 = sum(1 for l in lengths if l < 300)
        over_1500 = sum(1 for l in lengths if l > 1500)
        print(f"  Chunk sizes:  min={mn}  avg={avg:.0f}  max={mx}")
        print(f"  Under 300 chars: {under_300}  |  Over 1500 chars: {over_1500}")
    print()

    wp_counts: dict[str, int] = {}
    for n in nodes:
        wp = n.metadata.get("white_paper", "?")
        wp_counts[wp] = wp_counts.get(wp, 0) + 1
    print("  White papers detected:")
    for wp, c in sorted(wp_counts.items(), key=lambda x: -x[1]):
        print(f"    {wp}: {c} chunks")
    print()

    for i, node in enumerate(nodes[:3], start=1):
        lang = node.metadata.get("language", "?")
        cid = node.metadata.get("chunk_id", "?")
        wp = node.metadata.get("white_paper", "?")
        preview = node.text[:150].replace("\n", " ")
        print(f"  Sample {i} [{lang}] wp={wp} ({len(node.text)} chars): {preview}...")
    print()


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading markdown files …")
    documents = load_documents()
    print(f"Loaded {len(documents)} document(s)\n")

    print("Parsing with MarkdownNodeParser …")
    nodes = _parse_nodes(documents)
    print(f"  Raw nodes from parser: {len(nodes)}")

    print("Assigning white paper metadata …")
    _enrich_with_white_paper(nodes, documents)

    print("Merging small chunks …")
    nodes = _merge_small_chunks(nodes)
    print(f"  After merge: {len(nodes)}")

    print("Splitting oversized chunks …")
    nodes = _split_large_chunks(nodes)
    print(f"  After split: {len(nodes)}")

    print("Finalizing (topic prefix, header_path fix) …")
    nodes = _finalize_chunks(nodes)

    print("Generating per-chunk summaries …")
    nodes = _add_chunk_summaries(nodes)

    print("Assigning page numbers …")
    _enrich_with_page_numbers(nodes, documents)

    min_chars = 80
    before = len(nodes)
    nodes = [n for n in nodes if len(n.text.strip()) >= min_chars]
    print(f"  Filtered out {before - len(nodes)} stub chunks under {min_chars} chars\n")

    print(f"Saving chunks to {CHUNKS_DIR} …")
    counters = save_chunks(nodes)
    inspect_chunks(nodes)

    for lang, count in sorted(counters.items()):
        folder = LANG_FOLDER_MAP.get(lang, f"emergency_phase_{lang}")
        print(f"  {CHUNKS_DIR / folder}: {count} chunks")
    print(f"Done — {len(nodes)} chunk files written.\n")


if __name__ == "__main__":
    main()