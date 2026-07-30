# RAG Pipeline — Technical Reference

Bilingual (EN/FA) retrieval-augmented generation over the IPP *Emergency Phase Booklet*: 178 pages, Front Matter + 14 white papers, two parallel language editions.

Corpus: **503 EN + 456 FA validated chunks → 959 vectors**, Cohere `embed-multilingual-v3.0` (1024-dim), local Qdrant.

---

## Contents

- [RAG Pipeline — Technical Reference](#rag-pipeline--technical-reference)
  - [Contents](#contents)
  - [Stage graph](#stage-graph)
  - [Directory map](#directory-map)
  - [Setup](#setup)
  - [Stage 1 — PDF → Markdown](#stage-1--pdf--markdown)
  - [Stage 2 — Chunking](#stage-2--chunking)
    - [Chunk metadata schema](#chunk-metadata-schema)
  - [Stage 3 — Per-white-paper split](#stage-3--per-white-paper-split)
  - [Stage 4 — Agentic validation](#stage-4--agentic-validation)
  - [Stage 5 — Embedding + vector store](#stage-5--embedding--vector-store)
  - [Stage 6 — Retrieval](#stage-6--retrieval)
    - [6.1 Dense search](#61-dense-search)
    - [6.2 Rerank (section-aware)](#62-rerank-section-aware)
    - [6.3 Split-sibling expansion](#63-split-sibling-expansion)
    - [6.4 Section-continuation stitching](#64-section-continuation-stitching)
    - [6.5 Parked](#65-parked)
  - [Stage 7 — Chat engine](#stage-7--chat-engine)
    - [Language detection](#language-detection)
    - [Context format](#context-format)
    - [System prompt](#system-prompt)
    - [Source resolution](#source-resolution)
    - [Models](#models)
  - [Stage 8 — API](#stage-8--api)
  - [Configuration](#configuration)
  - [Rebuild](#rebuild)

---

## Stage graph

<img width="1880" height="430" alt="pipeline" src="https://github.com/user-attachments/assets/c3f650df-78a0-4d28-a100-afea6556182b" />

---

## Directory map

```
rag_pipeline/
├── document_preprocessing/
│   ├── marker_pdf_to_json_and_md.py        # stage 1
│   ├── iran_prosperity_project_pdf/
│   └── iran_prosperity_project_md/         # *.md + *.json + *_meta.json
├── indexing/
│   ├── chunking/
│   │   ├── chunking.py                     # stage 2
│   │   ├── organize_by_white_paper.py      # stage 3
│   │   ├── AGENTIC_VALIDATION_README.md    # stage 4 protocol
│   │   ├── chunks_raw/<lang>/
│   │   ├── by_white_paper/{md,chunks}/
│   │   └── by_white_paper_validated/
│   │       ├── chunks/<lang>/<NN_slug>/chunk_*.txt
│   │       └── validation_changelog.xlsx
│   ├── embedding/
│   │   ├── cohere_embed.py                 # BaseEmbedding adapter
│   │   ├── embed_and_store.py              # stage 5
│   │   └── qdrant_db/
│   └── retrieval/
│       ├── pipeline.py                     # stage 6 — ReliableRetriever
│       ├── cohere_rerank.py                # rerank-v3.5 adapter
│       ├── __init__.py                     # public re-exports
│       └── eval/                           # EVALUATION.md, run_eval.py
├── chatbot/
│   ├── engine.py                           # stage 7 — ChatEngine
│   └── cli.py                              # terminal client
└── api/
    ├── main.py                             # stage 8 — FastAPI
    ├── database.py                         # SQLite history (stdlib only)
    └── chat_history.db                     # created at runtime
```

---

## Setup

```powershell
# from project root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip wheel
pip install -r requirements.txt
```

`.env` at project root, loaded via `dotenv.find_dotenv(usecwd=True)` so any CWD works:

| Key | Consumed by |
|---|---|
| `COHERE_API_KEY` | `cohere_embed.py`, `cohere_rerank.py` |
| `GEMINI_API_KEY` | `engine.py` (generation + helpers) |
| `QWEN_API_KEY` \| `DASHSCOPE_API_KEY` | `chunking.py` summaries, eval Qwen arm |

Serving queries requires only `COHERE_API_KEY` + `GEMINI_API_KEY`.

---

## Stage 1 — PDF → Markdown

`document_preprocessing/marker_pdf_to_json_and_md.py` — Marker conversion, emits paginated Markdown plus structured JSON.

Pagination markers are the load-bearing output:

```
{57}------------------------------------------------
```

Stage 2 consumes marker offsets to derive `page_start`/`page_end`. Do not strip them from committed Markdown.

---

## Stage 2 — Chunking

`indexing/chunking/chunking.py` → `chunks_raw/<lang>/chunk_N.txt`

Sequence:

1. Strip pagination markers, retaining byte offsets for page attribution.
2. `MarkdownNodeParser` split on `#`/`##`/`###`.
3. Size normalisation — merge `< MIN_CHUNK_CHARS`, split `> MAX_CHUNK_CHARS` with `OVERLAP_CHARS`.
4. Metadata attach (schema below).
5. Categorical summary per chunk via Qwen-Plus, Gemini fallback on moderation refusal. Summaries state content *type* only (`"This chunk contains a priorities list."`) — no facts, names or figures, so they discriminate chunks without paraphrasing policy text.
6. Prefix body with `[Summary: …]` + `[Topic: <white_paper>]` (contextual retrieval).

### Chunk metadata schema

| Key | Type | Role downstream |
|---|---|---|
| `chunk_id` | `str` | `<lang>__<source>__<NNNNN>`; sibling lookup key |
| `language` | `en`\|`fa` | hard Qdrant filter |
| `white_paper` | `str` | source display, header index key |
| `header_path` | `str` | `/A/B/C/` heading trail — see note |
| `page_start`, `page_end` | `int` | source display, ground-truth mapping |
| `chunk_index` | `int` | document order; continuation adjacency |
| `split_part` | `str` | `"2/3"` when split for size; sibling expansion |
| `summary` | `str` | embedded signal |

---

## Stage 3 — Per-white-paper split

`indexing/chunking/organize_by_white_paper.py`

Regroups flat chunk output into 15 per-white-paper folders and slices each source MD into `00_front_matter.md … 14_educational_system.md`. Exists to bound Stage 4 context.

---

## Stage 4 — Agentic validation

Protocol: `indexing/chunking/AGENTIC_VALIDATION_README.md`. One Claude session per (white paper × language).

Corrects: malformed/empty `header_path`, summaries leaking specifics, OCR artefacts, table headers detached from rows. Diffs logged to `validation_changelog.xlsx`.

Invariant: **chunk body text is never paraphrased.** Sole permitted body mutation is prepending verbatim table column headers from the source MD.

Output `by_white_paper_validated/chunks/` is the indexing input. EN/FA chunk-count asymmetry (503/456) is structural — FA sections divide differently, producing larger chunks. Consequence: EN splits the 39-entry advisor list across a `(cont.)` boundary; FA does not.

---

## Stage 5 — Embedding + vector store

`indexing/embedding/cohere_embed.py` — `BaseEmbedding` subclass wrapping `embed-multilingual-v3.0`, correct `input_type` for `search_document` vs `search_query`. Single multilingual model ⇒ shared EN/FA vector space.

`indexing/embedding/embed_and_store.py`:

```
collection : emergency_phase_cohere_v3_validated
vectors    : 959   (503 en + 456 fa)
dim        : 1024
path       : indexing/embedding/qdrant_db/
```

Terminates with EN + FA sanity queries printing top-3.

> **Embedded text ≠ chunk body.** `TextNode` is constructed without `excluded_embed_metadata_keys`, so LlamaIndex's `MetadataMode.EMBED` prepends *all* metadata as `key: value` lines before embedding. Each vector therefore encodes `header_path`, `white_paper`, `summary`, `page_*` and `chunk_id` alongside the body. Dense search can consequently match on phase names. Any metadata schema change alters the embedding surface.

> Qdrant local storage is single-writer. Stop the API before re-embedding or running the eval; `ReliableRetriever.__init__` retries lock acquisition 10× with backoff.

---

## Stage 6 — Retrieval

`indexing/retrieval/pipeline.py` — `ReliableRetriever`

```python
query(query_text, language, *, final_top_k=DENSE_TOP_K,
      rerank=False, candidate_pool=RERANK_CANDIDATE_POOL) -> list[RetrievalResult]
```

| Flags | Variant |
|---|---|
| defaults | Test 1 — dense only |
| `rerank=True` | Test 2 — **production** |

### 6.1 Dense search

`dense_search(query, language, top_k)` — Qdrant `MetadataFilter(key="language", value=language)`. Cross-language retrieval is structurally impossible. `rerank=True` pulls `candidate_pool=40` instead of `top_k`.

### 6.2 Rerank (section-aware)

`rerank_pool(query, pool, top_n)` → `rerank-v3.5`, calibrated 0–1 relevance. Documents are built by `rerank_document(hit)`, not `hit.text`:

```
[Topic: HEALTHCARE]                                     # if absent from body
[Section: HEALTHCARE SPECIFIC OPERATIONS > Phase 1 (Days 1‑30) > Key Priorities]
[Summary: This chunk contains a priorities list.]
<body>
```

Rationale: stored body carries `[Summary:]` and (outside Front Matter) `[Topic:]` but never `header_path`. Without the `[Section:]` line the reranker cannot separate `Key Priorities` under Phase 1 from the identically-headed list under Phase 2.

### 6.3 Split-sibling expansion

`_expand_split_siblings(results)` — for hits with `split_part = "k/n"`, reconstructs sibling `chunk_id`s by offset arithmetic (`chunk_index - k`), fetches parts within `MAX_SIBLINGS_PER_PARENT = 2`. Tags `source += "+sibling"`.

### 6.4 Section-continuation stitching

`_expand_continuations(results, max_hops=MAX_CONTINUATION_HOPS)`

Target defect: source continues a section across a page break with a `(cont.)` heading; the chunker cuts there and the halves share **no** linking metadata — divergent `header_path`, no `split_part`. Neither 6.3 (needs `split_part`) nor header-parent logic (needs shared parent) connects them.

Rule, bidirectional, keyed on `CONTINUATION_RE` (`(cont.)` / `(ادامه)`), transitive to `max_hops=3`:

- hit whose `header_path` matches → fetch `chunk_index - 1`
- hit whose successor's `header_path` matches → fetch `chunk_index + 1`

Concrete case: EN advisors span `/AUTHORS (cont.)/` (Aghakouchak → Ardavan Khoshnood) and `/ADVISORS (cont.)/` (Arvin Khoshnood → Yassini). A query for advisors ranks the latter high, the former low ⇒ 21 of 39 entries returned with no gap signal. Tags `source += "+continuation"`.

### 6.5 Parked

`_expand_header_siblings` — LLM header-sibling router, evaluated and withdrawn (see `eval/EVALUATION.md` §variants). Retained under a `# ── PARKED` banner; no call sites. `_build_header_index()` still runs in `__init__` and is O(chunks).

---

## Stage 7 — Chat engine

`chatbot/engine.py` — `ChatEngine`. Single answer path for CLI and API.

| Method | Contract |
|---|---|
| `plan(question, history_text="") -> StreamPlan` | blocking; language detect → history rewrite → retrieve → context build |
| `stream_answer(plan, provider_out=None) -> Generator[str]` | token stream |
| `finalize(full_text, plan) -> (clean_text, sources)` | strip markers, resolve citations |

`StreamPlan.kind ∈ {greeting, lowconf, answer}`.

### Language detection

`PERSIAN_RE` script match → `fa`, else `en`. Selects system prompt, Qdrant filter, and `MAX_OUTPUT_TOKENS` (1536 EN / 2048 FA — FA needs more tokens per unit content).

### Context format

```
[Passage 1 — Source: HEALTHCARE, p.167]
[Section: HEALTHCARE SPECIFIC OPERATIONS > Phase 1 (Days 1‑30) > Key Priorities]
[Summary: …]
[Topic: …]
<body>

---

[Passage 2 — …]
```

Wrapped as `CONTEXT:\n…\n\nQUESTION:\n…`. `run_eval._build_context` must mirror this exactly or the eval measures a different pipeline.

### System prompt

Nine numbered rules per language, plus a static `DOCUMENT IDENTITY` block:

| # | Rule |
|---|---|
| 1 | Answer only from CONTEXT; else emit the exact refusal sentinel |
| 2 | Project questions — treat `DOCUMENT IDENTITY` as valid context for whole-document queries |
| 3 | Partial coverage — answer covered parts, explicitly name the uncovered part |
| 4 | No inline `Passage N` references in body |
| 5 | No visible reasoning or self-correction |
| 6 | Section scope — for phase/day-range queries use only passages whose `[Section:]` matches |
| 7 | Counts — emit number **and** enumerate items |
| 8 | Complete every list; no mid-item truncation |
| 9 | Answer in the query language |

`DOCUMENT IDENTITY` supplies producer, backer, Project Director, NUFDI President/CEO, foreword contributor, contributor counts and the 15-paper list, so whole-document questions do not depend on Front Matter being retrieved.

Hidden trailer `USED_PASSAGES: 1, 3` is parsed by `_extract_used_passages` (tolerates Western/Arabic-Indic/Persian digits) and stripped.

### Source resolution

`USED_PASSAGES` is a *candidate set*, not ground truth — the model over-reports (single-fact answers claiming 3 passages). `_grounded_passage_idxs(answer, sources, lang)` verifies each candidate:

- primary signal: shared adjacent content-token pairs (`_phrases`), threshold `_MIN_SHARED_PHRASES = 2`
- fallback: distinctive unigram overlap, `doc_freq <= ceil(n/2)`, threshold `_MIN_FALLBACK_TERMS = 2`
- final fallback: top-1 by score, so a substantive answer never renders zero sources

Unigram-only matching was rejected: an answer enumerating the booklet's topics contains "cybersecurity", which would wrongly credit that white paper. Persian is normalised for ZWNJ (`U+200C`) before tokenisation.

`_is_refusal` requires the answer be *nothing but* a refusal — sentinel sentences are removed and the remainder tested for list markers / length ≥ `_REFUSAL_REMAINDER_CHARS`. A partial-coverage answer terminating in the sentinel retains citations for the answered half.

### Models

| Constant | Value | Use |
|---|---|---|
| `ANSWER_MODEL` | `gemini-3.5-flash-lite` | user-facing generation |
| `GEMINI_MODEL` | `gemini-2.5-flash` | helper completions (titles, pronoun rewrites) |
| `QWEN_MODEL` | `qwen-plus` | Stage 2 summaries |

Plain REST, no vendor SDK. Gemini 3.x rejects `thinkingConfig`; both `_gemini_generate` and `_gemini_generate_stream` retry without it on HTTP 400.

> **SSE charset.** Gemini's stream endpoint returns `text/event-stream` with no charset. Per RFC, `requests` defaults `text/*` to ISO-8859-1, decoding UTF-8 as latin-1 — marginal in EN, total corruption in FA. `_gemini_generate_stream` pins `resp.encoding = "utf-8"` before `iter_lines`. Do not remove.

---

## Stage 8 — API

`api/main.py`, `api/database.py`

```powershell
cd rag_pipeline
..\.venv\Scripts\python.exe -m api.main     # or: uvicorn api.main:app --reload
```

Engine constructed once in the `lifespan` handler (embedding model + Qdrant + chunk store load ≈ seconds).

| Method | Path | Notes |
|---|---|---|
| POST | `/api/chat` | SSE stream |
| POST | `/api/conversations` | create |
| GET | `/api/conversations` | list (sidebar) |
| GET | `/api/conversations/{id}` | full history |
| DELETE | `/api/conversations/{id}` | delete |
| POST | `/api/generate-title` | auto-title from first exchange |
| GET | `/api/health` | status + active provider |

SSE event sequence: `token*` → `replace?` (if `finalize` altered the text) → `sources` → `done`. All payloads `json.dumps(..., ensure_ascii=False)`.

`database.py` — `sqlite3` stdlib, `chat_history.db`, tables `conversations` / `messages`. No driver dependency.

CORS is unrestricted for local dev. **Restrict `allow_origins` before deployment.**

---

## Configuration

| Constant | Value | File |
|---|---|---|
| `MIN_CHUNK_CHARS` | 300 | `chunking/chunking.py` |
| `MAX_CHUNK_CHARS` | 1500 | `chunking/chunking.py` |
| `OVERLAP_CHARS` | 100 | `chunking/chunking.py` |
| `COLLECTION_NAME` | `emergency_phase_cohere_v3_validated` | `retrieval/pipeline.py` |
| `DENSE_TOP_K` | 15 | `retrieval/pipeline.py` |
| `RERANK_CANDIDATE_POOL` | 40 | `retrieval/pipeline.py` |
| `MAX_SIBLINGS_PER_PARENT` | 2 | `retrieval/pipeline.py` |
| `MAX_CONTINUATION_HOPS` | 3 | `retrieval/pipeline.py` |
| `TOP_K` | 5 | `chatbot/engine.py` |
| `USE_RERANK` | `True` | `chatbot/engine.py` |
| `ANSWER_MODEL` | `gemini-3.5-flash-lite` | `chatbot/engine.py` |
| `MAX_OUTPUT_TOKENS` | `{en: 1536, fa: 2048}` | `chatbot/engine.py` |

Mutating any Stage 2 constant invalidates the index — re-run Stages 2–5.

---

## Rebuild

```powershell
python -m document_preprocessing.marker_pdf_to_json_and_md   # 1 (needs marker-pdf)
python indexing/chunking/chunking.py                         # 2
python indexing/chunking/organize_by_white_paper.py          # 3
#                                                              4 — manual, see protocol
python -m indexing.embedding.embed_and_store                 # 5
python -m indexing.retrieval.eval.run_eval                   # re-measure
```

Stage 4 is intentionally not automated: it catches OCR damage, leaked summaries and detached table headers — defects that degrade retrieval silently and are hardest to detect downstream.

Quality methodology and current numbers: [`indexing/retrieval/eval/README.md`](indexing/retrieval/eval/README.md).
