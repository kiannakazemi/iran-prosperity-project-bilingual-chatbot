# RAG Pipeline

Bilingual (EN/FA) retrieval-augmented generation over the IPP *Emergency Phase Booklet*: 178 pages, Front Matter + 14 white papers, two parallel language editions.

## Layout

```
rag_pipeline/
├── document_preprocessing/     # 1. PDF → paginated Markdown + JSON (Marker)
├── indexing/
│   ├── chunking/               # 2. Markdown → chunks   3. group by white paper   4. agentic validation
│   ├── embedding/              # 5. embed chunks → local Qdrant index
│   └── retrieval/              # 6. dense search + section-aware rerank + expansion
│       └── eval/               #    retrieval/answer evaluation (see eval/README.md)
├── chatbot/                    # 7. answer generation (engine) + CLI
└── api/                        # 8. FastAPI serving + SQLite chat history
```

Stages 1–5 are **build-time** (run once to produce the index, which ships in the repo). Stages 6–8 are **serve-time** (run on every question).

## Pipeline at a glance

```
PDF ─▶ Markdown ─▶ chunks ─▶ grouped ─▶ validated ─▶ embeddings ─▶ Qdrant
                                                                      │
question ─▶ language detect ─▶ dense search ─▶ rerank ─▶ expand ─▶ context ─▶ Gemini ─▶ answer + sources
```

---



## 1. Document preprocessing

`document_preprocessing/marker_pdf_to_json_and_md.py`

Each language edition is a separate PDF. [Marker](https://github.com/VikParuchuri/marker) converts each to a structured JSON representation and a **paginated** Markdown file, where every page boundary is written as a `{N}----------` marker. Those markers are what later lets a chunk be traced back to a specific PDF page.

```bash
python document_preprocessing/marker_pdf_to_json_and_md.py
# input:  document_preprocessing/iran_prosperity_project_pdf/
# output: document_preprocessing/iran_prosperity_project_md/   (.md + .json + _meta.json)
```

The English and Persian booklets produce parallel Markdown files that all downstream stages treat as the ground truth.

## 2. Chunking

`indexing/chunking/chunking.py`

Turns each Markdown file into retrieval-ready chunks, carrying the metadata retrieval and citation depend on. The steps, in order:

1. **Clean.** Strip page-break markers and inline footnote lines, collapse blank runs — but first build a *page map* `(char offset → page number)` so page numbers survive the cleaning.
2. **Parse.** LlamaIndex `MarkdownNodeParser` splits on the heading hierarchy, giving each node a `header_path`.
3. **Assign white paper.** Each chunk is located in the source text and tagged with the white paper whose boundary it falls under. Boundaries are matched only against the 14 canonical topic titles (English and Persian lists), so stray headings can't create phantom sections. Everything before the first boundary is Front Matter.
4. **Merge / split.** Consecutive small chunks from the same white paper merge until they clear the minimum; oversized chunks split on sentence boundaries with overlap. Split pieces are marked `split_part = N/M`.
5. **Finalize.** Prepend `[Topic: <white paper>]` to the body and repair empty header paths.
6. **Summarize.** An LLM writes a one-line *category* summary per chunk ("what type of content this is"), prepended as `[Summary: …]` so the embedding sees a topical signal in its first tokens.
7. **Page numbers.** Using the page map, assign `page_start` / `page_end` from the chunk's actual final text.
8. **Filter.** Drop stubs under 80 characters.

Size controls:


| Constant          | Value |
| ----------------- | ----- |
| `MIN_CHUNK_CHARS` | 300   |
| `MAX_CHUNK_CHARS` | 1500  |
| `OVERLAP_CHARS`   | 100   |


Each chunk is written as a `.txt` file with a metadata header followed by the body. The metadata schema, in fixed order:

```
chunk_id, chunk_index, source_file, source_pdf, language,
white_paper, header_path, page_start, page_end, split_part, summary
```

`chunk_id` is `"{lang}__{source_stem}__{index:05d}"` (e.g. `en__Emergency_Phase_ENGLISH_20260301_1440__00042`). Output lands in `chunking/chunks_raw/` (regenerable; git-ignored).

## 3. Organize by white paper

`indexing/chunking/organize_by_white_paper.py`

Reshapes the flat output into `by_white_paper/`, with `md/<lang>/NN_slug.md` (the source Markdown sliced per white paper) beside `chunks/<lang>/NN_slug/` (the chunks grouped to match). Both languages use identical `NN_slug` names (`00_front_matter`, `01_legal`, …) so a white paper's two editions pair at a glance. This is the layout the validation step audits, one white paper at a time.

## 4. Agentic validation

`indexing/chunking/AGENTIC_VALIDATION_README.md` (workflow + exact prompts)

The chunker is deterministic and fast but leaves structural defects an LLM reading the source can fix. Run once per `(language, white_paper)`, an agent audits every chunk against its source Markdown slice and:

- corrects and de-noises `header_path` (wrong/off-by-one section, stray `**`/bullet characters);
- rewrites each `summary` into a consistent category-only form;
- re-attaches column headers to table chunks that lost them in a split;
- (Persian only) repairs recurring OCR artifacts — lam-alef letter swaps and missing ZWNJ in compounds.

Hard constraint: **the chunk's policy content is never paraphrased or invented.** The only body edits allowed are prepending verbatim table headers and the bounded Persian OCR fixes. Every chunk is written to `by_white_paper_validated/chunks/<lang>/…` (modified or not), and each change is logged to `validation_changelog.xlsx`.

The validated folder is the single source of truth for embedding — **959 chunks** (503 English, 456 Persian).

## 5. Embedding and indexing

`indexing/embedding/cohere_embed.py` · `indexing/embedding/embed_and_store.py`

Each validated chunk is embedded with **Cohere** `embed-multilingual-v3.0` (1024-dim) and stored in a local **Qdrant** collection, `emergency_phase_cohere_v3_validated`. Documents are embedded with Cohere's `search_document` input type and queries later with `search_query`, so cosine similarity at query time is well-calibrated. The run ends with an English and a Persian sanity query.

```bash
python -m indexing.embedding.embed_and_store
# reads: indexing/chunking/by_white_paper_validated/chunks/
# writes: indexing/embedding/qdrant_db/   (ships in the repo — no need to re-embed)
```

The built index is committed, so a fresh clone can answer questions immediately without paying to re-embed.

## 6. Retrieval

`indexing/retrieval/pipeline.py` · `indexing/retrieval/cohere_rerank.py`

`ReliableRetriever.query()` is the serve-time entry point. Two variants share one code path:

- **Test 1 — dense only** (`rerank=False`): top dense matches, then expansion.
- **Test 2 — dense + rerank** (`rerank=True`, production): a wider dense candidate pool is reranked by **Cohere** `rerank-v3.5` down to the final top-k, then expansion.

The rerank is **section-aware**: each candidate is sent to Cohere with its `header_path` (and white paper) prepended, so the reranker can tell apart identically-named sections — the Healthcare paper repeats "Key Priorities" under every phase, and without the heading trail "priorities for the first 30 days" is a coin-flip between phases.

Two expansion passes then repair fragmented context:

- **Split-sibling expansion** re-attaches adjacent `split_part` pieces of a section the chunker cut for size.
- **Section-continuation stitching** rejoins a section the document continued across a page break with a "(cont.)" / "(ادامه)" heading, where the two halves share no linking metadata — the failure that once returned a half-complete advisor list.

Key constants:


| Constant                | Value | Meaning                           |
| ----------------------- | ----- | --------------------------------- |
| `DENSE_TOP_K`           | 15    | default dense hits                |
| `RERANK_CANDIDATE_POOL` | 40    | dense pool sent to the reranker   |
| `TOP_K` (engine)        | 5     | passages kept after rerank        |
| `MAX_CONTINUATION_HOPS` | 3     | a section may span up to 4 chunks |


An earlier third variant — an extra LLM call to route in "header siblings" — was evaluated and removed; it changed retrieval on only a handful of questions and added nothing section-aware reranking doesn't already recover. Its code is parked (unused) in `pipeline.py`. Full write-up in `[eval/README.md](indexing/retrieval/eval/README.md)`.

## 7. Answer generation

`chatbot/engine.py`

`ChatEngine` is the single source of truth for the answer path, exposed as three steps the CLI and API both call: `plan()` (blocking: detect → retrieve → build context), `stream_answer()` (stream tokens), and `finalize()` (clean up + resolve sources).

- **Language detection** compares Persian vs Latin character counts; a Persian question is always answered from Persian text, English from English.
- **Greetings** are detected and answered without retrieval; a greeting *followed by* a real question keeps a short prefix and proceeds.
- **Follow-ups** are rewritten into standalone search queries using recent history, so pronouns resolve before retrieval.
- **Context** is built by numbering the retrieved passages and prefixing each with its source, page range, and `[Section: …]` heading trail — the trail is what makes phase-scoped questions answerable.
- **Generation** streams from **Gemini 3.5 Flash-Lite** (`ANSWER_MODEL`), chosen by evaluation; short helper calls (titles, query rewrites) use `gemini-2.5-flash` (`GEMINI_MODEL`), and Qwen-plus is used for some build-time helper completions with a Gemini fallback on content-moderation refusals.

The system prompt enforces the behaviour the assistant is graded on: answer only from context, decline with a fixed sentence when the answer isn't there, handle multi-part questions part by part, obey section scope, enumerate on counts, and emit a hidden `USED_PASSAGES:` line.

Because the model's `USED_PASSAGES` self-report tends to over-claim, `finalize()` treats it as a *candidate* list and verifies each passage against the answer text — a source is shown only if enough of its distinctive phrasing actually surfaces in the answer, so citations match what the reader just saw. Refusal answers show no sources.

Two robustness details worth knowing: the streaming call pins `resp.encoding = "utf-8"` (Gemini sends `text/event-stream` with no charset, which would otherwise be decoded as Latin-1 and mangle all Persian), and Gemini 3.x rejects the `thinkingConfig` block, so both call paths retry without it on a 400.

Answer length caps: `MAX_OUTPUT_TOKENS` = 1536 (English), 2048 (Persian).

`chatbot/cli.py` — an interactive terminal client for the same engine:

```bash
python -m chatbot.cli
```



## 8. Serving

`api/main.py` · `api/database.py`

A FastAPI app loads one `ChatEngine` at startup and streams answers over **Server-Sent Events**. `POST /api/chat` emits `token` events as the answer streams, an optional `replace` (the cleaned text with `USED_PASSAGES` stripped), a `sources` event, and a final `done`. Retrieval runs in a thread pool with a timeout so the event loop is never blocked. Conversation CRUD, a title generator, and a health check round out the API; chat history persists in a local SQLite database (`chat_history.db`, git-ignored).

```bash
python -m api.main          # or: uvicorn api.main:app --reload
```

Endpoints: `POST /api/chat`, `POST /api/conversations`, `GET /api/conversations[/{id}]`, `PATCH`/`DELETE /api/conversations/{id}`, `POST /api/generate-title`, `GET /api/health`.

## Evaluation

Retrieval and answer quality are measured end to end on a fixed bilingual question set, comparing the two retrieval variants and the answer models. Method, metrics, and results: `[indexing/retrieval/eval/README.md](indexing/retrieval/eval/README.md)`.

## Configuration

Keys are read from the project-root `.env` (see `.env.example`):


| Variable                             | Used for                        | Required |
| ------------------------------------ | ------------------------------- | -------- |
| `COHERE_API_KEY`                     | embeddings + reranking          | yes      |
| `GEMINI_API_KEY`                     | answer generation + helpers     | yes      |
| `QWEN_API_KEY` / `DASHSCOPE_API_KEY` | build-time chunk summaries only | no       |




## Rebuilding from scratch

The index ships in the repo, so this is only needed if the source or chunking changes. Run from `rag_pipeline/`, with the API stopped so Qdrant is not locked:

```bash
python document_preprocessing/marker_pdf_to_json_and_md.py   # 1. PDF → Markdown
python indexing/chunking/chunking.py                          # 2. Markdown → chunks
python -m indexing.chunking.organize_by_white_paper           # 3. group by white paper
#                                                               4. agentic validation (see its README)
python -m indexing.embedding.embed_and_store                  # 5. embed → Qdrant
```

Then serve with `python -m api.main` (or query directly with `python -m chatbot.cli`).
