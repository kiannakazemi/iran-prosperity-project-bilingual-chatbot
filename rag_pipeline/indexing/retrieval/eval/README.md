# Retrieval & Answer-Quality Evaluation

Empirical selection of the production RAG configuration. 2 retrieval variants × 3 answer models × 47 questions × 2 languages = **564 graded answers**.

## Dataset

data/golden_dataset_with_chunks.xlsx — 47 EN + 47 FA questions, hand-authored against the source Markdown.

Ground-truth columns: `Source Section`, `MD Source Page(s)`, `Expected Chunks`, `Question Type`.

### Behavioural sub-types

**Out-of-scope (8).** Unanswerable by construction (Iran's GDP, today's date, arrest counts). Correct response is the refusal sentinel. Measures hallucination rate on unanswerable input.

**Partial coverage (2).** One answerable clause + one unanswerable clause, e.g. *"TECU responsibilities, and how many personnel are assigned?"* — responsibilities are in-document, headcount is not. `Complete` requires both answering the covered clause **and** explicitly naming the uncovered one. Silent omission fails: the user cannot detect partial service.

**False premise (≥1 per language).** EN Q35 asserts that *enforcing* Iranian Medical Council price controls reduces cost; the source recommends *halting* enforcement. `Complete` requires premise correction. Refusal fails — the content exists and the user leaves misinformed.

> FA Q35 uses «توقف» (halting), matching the source, so no false premise exists in the FA phrasing. That row is not cross-language comparable.

---

## Variants under test

Shared substrate: `embed-multilingual-v3.0` (1024-dim) → Qdrant, `language` metadata filter enforced.

### Test 1 — dense only

```
dense top-5 → split-sibling expansion → continuation stitching
```

### Test 2 — dense + rerank  *(production)*

```
dense top-40 → rerank-v3.5 → top-5 → split-sibling expansion → continuation stitching
```

Reranking is **section-aware**: candidates are submitted with `header_path` prepended via `rerank_document()`. Absent that, the reranker receives only the chunk body — which carries `[Summary:]` and `[Topic:]` but not the heading trail.

Both variants run identical context repair (see pipeline reference §6.3–6.4).

---

## Models under test

All three receive identical context, system prompt and `MAX_OUTPUT_TOKENS`. Model is the only free variable.

| `MODEL_KINDS` | Endpoint |
|---|---|
| `gemini_flash` | `gemini-2.5-flash` |
| `gemini_flash_lite` | `gemini-3.5-flash-lite` |
| `qwen` | `qwen-plus` (DashScope) |

---

## Grading rubric

Graded on **answer text against source Markdown**, not chunk-ID overlap. Chunk overlap proved near-orthogonal to answer quality — Test 2 retrieved more expected chunks on 13 EN rows, Test 1 on 11, while answer quality diverged clearly.

Ground truth = pages named in `MD Source Page(s)`, sliced from the language-matched source MD. No external knowledge admitted; a claim absent from those pages is wrong even if true.

| Verdict | Criterion |
|---|---|
| **Complete** | All ground-truth key items present, correctly scoped. Out-of-scope row: correct refusal. |
| **Partial** | ≥ ⌈n/2⌉ key items, no fabrication. Detectable omission: dropped list entry, unanswered clause, headings without discriminating detail. |
| **Incomplete** | False refusal (content present in cited pages) ∨ hallucination (claim absent from cited pages) ∨ wrong-section answer ∨ provider moderation block. |

Multi-part rows graded on all clauses: 3-of-4 ⇒ `Partial`.

Accuracy measures:

```
strict   = Complete / 47
weighted = (Complete + 0.5·Partial) / 47
```

---

## Grading procedure

Claude Opus 5 via Cowork sessions, filesystem access to results workbook + source MD.

1. Derive a per-question **key-item checklist** from the source MD.
2. Score each of the 564 answers on checklist coverage; flag refusals and provider blocks.
3. Apply the verdict rule uniformly.
4. Read full text for any contestable cell; correct.

Two artefacts of the checklist method worth recording:

- **FA matching required normalisation.** Persian applies ZWNJ (`U+200C`) inconsistently and mixes Arabic/Persian letterforms (`ي/ی`, `ك/ک`). The first pass scored several rows 0/n across *all six* configurations — six models do not fail identically, so the checklist was at fault. Adding normalisation and re-deriving terms from observed vocabulary reduced FA problem cells from >100 to 28 (EN: 41).
- **Two apparent content misses were moderation blocks.** EN Q19 and Q44 under Qwen returned DashScope HTTP 400. Qwen blocks on EN as well as FA — 2 cells each.

Per-cell verdicts and justifications: 12 appended columns per sheet in `results/retrieval_eval_results.xlsx`, colour-coded.

---

## Results

### English

| Test | Model | C | P | I | strict | weighted |
|---|---|---:|---:|---:|---:|---:|
| 1 | Flash | 39 | 4 | 4 | 83.0% | 87.2% |
| 1 | Flash-Lite | 41 | 2 | 4 | 87.2% | 89.4% |
| 1 | Qwen | 38 | 5 | 4 | 80.9% | 86.2% |
| 2 | Flash | 40 | 5 | 2 | 85.1% | 90.4% |
| **2** | **Flash-Lite** | **42** | 5 | **0** | **89.4%** | **94.7%** |
| 2 | Qwen | 41 | 6 | 0 | 87.2% | 93.6% |

### Persian

| Test | Model | C | P | I | strict | weighted |
|---|---|---:|---:|---:|---:|---:|
| 1 | Flash | 43 | 0 | 4 | 91.5% | 91.5% |
| 1 | Flash-Lite | 42 | 2 | 3 | 89.4% | 91.5% |
| 1 | Qwen | 40 | 1 | 6 | 85.1% | 86.2% |
| 2 | Flash | 44 | 1 | 2 | 93.6% | 94.7% |
| **2** | **Flash-Lite** | **45** | 1 | **1** | **95.7%** | **96.8%** |
| 2 | Qwen | 42 | 2 | 3 | 89.4% | 91.5% |

### Median end-to-end latency (s, incl. retrieval)

| | Flash | Flash-Lite | Qwen |
|---|---:|---:|---:|
| EN T1 | 1.62 | 1.66 | 4.39 |
| EN T2 | 1.97 | 2.17 | 5.11 |
| FA T1 | 1.65 | 1.71 | 6.51 |
| FA T2 | 2.02 | 2.17 | 7.05 |

### Observations

- **Test 2 > Test 1 in all six pairings.** Test 1 false-refuses wherever dense retrieval misses the target chunks entirely (EN Q15/Q31/Q35/Q36; FA Q14/Q15/Q36 — FA Q14 refuses under all three models at T1, answers under all three at T2).
- **Flash-Lite takes the top cell in both languages**; sole configuration with ≈zero hard failures (0 EN, 1 FA Incomplete).
- **Flash wins no cell.** Failure mode is confabulation — on Q15 it emits a plausible Phase A/B4 action list sourced from an adjacent phase. Flash-Lite's failure mode is over-caution. For a quotable policy document, over-caution is the cheaper error class.
- **Qwen** narrows the gap at T2 but incurred **4 moderation blocks** (2 EN, 2 FA), including on *"what day is today"*. Non-deterministic availability disqualifies it from the user-facing path irrespective of score.
- **FA > EN throughout**, primarily a chunking artefact: FA retains the 39-entry advisor list in one chunk; EN splits it across a `(cont.)` boundary.

---

## Selected configuration

> **Test 2 (dense + section-aware rerank) + `gemini-3.5-flash-lite`, both languages.**

```python
# chatbot/engine.py
ANSWER_MODEL = "gemini-3.5-flash-lite"
USE_RERANK   = True
TOP_K        = 5
```

No per-language divergence. An earlier EN-T2 / FA-T3 split was retired once measurement showed 2 of its 3 supporting FA rows had context identical to T2 — i.e. nondeterminism, not a router effect.

---

## Defects surfaced

| Defect | Consequence |
|---|---|
| `plan()` called `query()` with default `rerank=False` | production ran Test 1 — the weakest measured configuration |
| SSE stream decoded as ISO-8859-1 (`text/event-stream`, no charset ⇒ `requests` latin-1 default) | marginal EN mojibake; total FA corruption. Not caught by this harness — it uses the non-streaming path |
| Reranker received body without `header_path` | phase-scoped answers merged items across phases |
| `(cont.)` splits unrecoverable — divergent `header_path`, no `split_part` | advisor query returned 21 of 39 entries, no gap signal |
| `USED_PASSAGES` trusted verbatim | single-fact answers citing 3 passages; Front-Matter answers citing unrelated white papers |
| `_is_refusal` was a substring test | partial-coverage answers ending in the sentinel lost **all** citations |
| Counts returned without enumeration | "how many authors" → bare integer |

---

## Limitations

**n = 47 per language.** Deltas of 1–2 rows are within noise.

**Single run per cell.** LLM nondeterminism is not separated from configuration effect. This is precisely what invalidated the withdrawn third variant. **Future variant comparisons must first verify the retrieved context differs before attributing a delta to retrieval.**

**Cross-language grading asymmetry.** The FA checklist was re-derived post-normalisation with looser alternation (`A|B`) than the EN checklist. Intra-language ranking is sound; the FA−EN margin is not precisely 6 points.

**Three production fixes unexercised.** `generate_answer()` uses the non-streaming API and never invokes `ChatEngine.finalize()`, so the UTF-8 fix, source-grounding verification and partial-coverage refusal logic are covered only by unit-level tests and manual web-app testing.

**Four rows fail under every configuration:**

| Row | Diagnosis |
|---|---|
| Q31 Industry phases | EN T1 Flash is the *only* run reaching *Phase 0: Pre-Entry Coordination*. Dense finds early-phase chunks; rerank evicts them ⇒ `rerank-v3.5` appears to score introductory/setup sections low. Investigate rerank-query framing. |
| Q39 white-paper summary | top-5 budget cannot summarise a 9-chunk paper. Requires bulk white-paper load, not better ranking. |
| Q15 Phase A/B4 | rare-token: literal `A/B4` occurs once, embeds poorly. |
| Q12 dissolution list | answer spans a wide MD table whose rows land in distinct chunks; no top-5 selection covers it. |

**Two quality failures the rubric does not capture** — framing-level, every constituent fact true:

- *List item rendered as specific.* Asked why Khuzestan is named, an answer fused the passage where Khuzestan is the exemplar of a water grievance with a separate passage listing it among four "vulnerable regions", implying it is singled out for infrastructure security.
- *Summary imbalance.* An Education white-paper summary expanded the gender-integration prose (longest section) and under-covered the structural/legal interventions and the four enumerated risks.

---

## Running

Stop the API first — Qdrant local storage is single-writer.

```powershell
cd rag_pipeline
..\.venv\Scripts\python.exe -m indexing.retrieval.eval.run_eval
```

| Flag | Effect |
|---|---|
| `--tests 2` | single variant; skips the unselected variant's rerank + generation calls |
| `--models gemini_flash_lite` | single model |
| `--sheet "English Golden Dataset"` | single language |
| `--file <path>` | alternate dataset |

Output overwrites `results/retrieval_eval_results.xlsx`. Archive first to compare; prior runs in `results_v1/`, `results_v2/`.

> `run_eval._build_context` must remain byte-identical to `ChatEngine._build_context` plus the `CONTEXT:/QUESTION:` wrapper. It has diverged once (omitted `[Section:]`), understating results on phase-scoped rows.

---

## Artefacts

```
eval/
├── EVALUATION.md
├── run_eval.py
├── data/golden_dataset_with_chunks.xlsx
├── results/retrieval_eval_results.xlsx      # current, graded
├── results_v1/
└── results_v2/
    ├── retrieval_eval_results.xlsx
    └── answer_quality_assessment/           # per-variant slim graded workbooks
```

Per question, each sheet holds: ground-truth columns, full retrieval trace per variant (dense pool → reranked order → top-5 → siblings added → final set), per-model answer + latency, and 12 grading columns.
