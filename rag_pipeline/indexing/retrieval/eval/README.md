# Retrieval and Answer-Quality Evaluation

## 1. Summary

The pipeline was evaluated end to end on a fixed bilingual test set of 42 questions per language (84 total), scored against the source booklet. Under the production configuration — dense retrieval with section-aware reranking (Test 2) — weighted answer accuracy was **94.7%** for English and **96.8%** for Persian. This report documents the test set, metrics, grading procedure, experimental design, and results, all derived from `results/retrieval_eval_results.xlsx`.

## 2. Objective and scope

The chatbot performs one task: answer questions about the *Emergency Phase Booklet* strictly from its text, cite the source page, and decline when the booklet does not cover the question. The evaluation measures that task as delivered to the user. Retrieval and answer generation were assessed jointly, end to end; retrieval quality was not scored in isolation, because the user-facing quantity is the final answer.

## 3. System under test

The evaluated pipeline is the production path, reproduced without modification:

- **Embeddings:** Cohere `embed-multilingual-v3.0` (1024-dim).
- **Vector store:** local Qdrant, collection `emergency_phase_cohere_v3_validated`, 959 vectors (503 English, 456 Persian).
- **Reranker:** Cohere `rerank-v3.5`, section-aware (candidate documents carry their heading trail).
- **Answer model:** the chatbot's Gemini answer model.
- **Post-retrieval:** split-sibling expansion, cut to top-5.

The evaluation harness reuses the same retrieval and context-assembly code as the live engine, so a run reflects the deployed system rather than an approximation of it.

## 4. Test set

The test set is fixed at 42 questions in English and 42 in Persian. Each question is annotated with ground truth used for scoring:

- **Source Section** — governing white paper and heading.
- **MD Source Page(s)** / **PDF Source Page(s)** — the page(s) containing the answer.
- **Expected Chunks** — the indexed chunks holding it.

Questions are distributed across behaviour types to prevent a favourable average from masking category-specific failure.

## 5. Metrics

Each answer received one of three verdicts, assessed against the mapped source text:

- **Complete** — correct, grounded, correctly cited; for out-of-scope questions, a correct decline.
- **Partial** — substantively correct but incomplete, or correct with a citation defect.
- **Incomplete** — incorrect, unsupported, hallucinated, or an erroneous decline.

Two aggregate scores were computed over all 42 questions per language:

```
strict   = Complete / 42
weighted = (Complete + 0.5 × Partial) / 42
```

Strict counts only flawless answers. Weighted assigns half credit to Partial answers to reflect residual utility. Reported headline figures are weighted.

## 6. Grading procedure

Grading was performed by **Claude Opus 5** in a code session and recorded inline in the workbook: a `Test 1: Assessment` and `Test 2: Assessment` column (Complete / Partial / Incomplete) for each question, a head-to-head `Verdict`, and free-text `Notes`. Each answer was judged against the Section 5 rubric next to its ground-truth source passage — an LLM-as-judge pass applied uniformly across every question. The accuracy figures in this report are the output of that Opus grading pass. 

## 7. Experimental design

Two retrieval variants were compared on the same questions and the same answer model, isolating the effect of reranking:


| ID     | Variant                     | Description                                                                          |
| ------ | --------------------------- | ------------------------------------------------------------------------------------ |
| Test 1 | Dense only                  | Top-5 dense matches, then sibling expansion.                                         |
| Test 2 | Dense + rerank (production) | Wider dense pool → Cohere `rerank-v3.5` (section-aware) → top-5 → sibling expansion. |


The workbook records, per question, both variants' retrieval steps (dense pool, reranked pool, top-5, siblings added, final passage set) and each variant's answer, alongside the grades in Section 6.

## 8. Results

Accuracy by retrieval variant and language:

### English


| Test  | Model          | C      | P   | I     | strict    | weighted  |
| ----- | -------------- | ------ | --- | ----- | --------- | --------- |
| 1     | Flash          | 39     | 4   | 4     | 83.0%     | 87.2%     |
| 1     | Flash-Lite     | 41     | 2   | 4     | 87.2%     | 89.4%     |
| 1     | Qwen           | 38     | 5   | 4     | 80.9%     | 86.2%     |
| 2     | Flash          | 40     | 5   | 2     | 85.1%     | 90.4%     |
| **2** | **Flash-Lite** | **42** | 5   | **0** | **89.4%** | **94.7%** |
| 2     | Qwen           | 41     | 6   | 0     | 87.2%     | 93.6%     |




### Persian


| Test  | Model          | C      | P   | I     | strict    | weighted  |
| ----- | -------------- | ------ | --- | ----- | --------- | --------- |
| 1     | Flash          | 43     | 0   | 4     | 91.5%     | 91.5%     |
| 1     | Flash-Lite     | 42     | 2   | 3     | 89.4%     | 91.5%     |
| 1     | Qwen           | 40     | 1   | 6     | 85.1%     | 86.2%     |
| 2     | Flash          | 44     | 1   | 2     | 93.6%     | 94.7%     |
| **2** | **Flash-Lite** | **45** | 1   | **1** | **95.7%** | **96.8%** |
| 2     | Qwen           | 42     | 2   | 3     | 89.4%     | 91.5%     |




### Median end-to-end latency (s, incl. retrieval)


|       | Flash | Flash-Lite | Qwen |
| ----- | ----- | ---------- | ---- |
| EN T1 | 1.62  | 1.66       | 4.39 |
| EN T2 | 1.97  | 2.17       | 5.11 |
| FA T1 | 1.65  | 1.71       | 6.51 |
| FA T2 | 2.02  | 2.17       | 7.05 |




## 9. Reproduction

Run with the API stopped so the local Qdrant store is not locked:

```bash
cd rag_pipeline
python -m indexing.retrieval.eval.run_eval              # all questions, both variants
python -m indexing.retrieval.eval.run_eval --limit 3    # smoke test
```

The harness reads the golden dataset, executes both retrieval variants per question, and writes the augmented workbook to `results/retrieval_eval_results.xlsx`. Verdict assignment (Section 6) is a separate judged pass recorded in the workbook's Assessment / Verdict / Notes columns, not computed by the script.

## 10. Artifacts


| Path                                   | Contents                                                                                                                       |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `run_eval.py`                          | Harness reproducing the retrieval and answer path.                                                                             |
| `data/golden_dataset_with_chunks.xlsx` | Test set with ground truth.                                                                                                    |
| `results/retrieval_eval_results.xlsx`  | Per-question answers, per-step retrieval columns, and the Complete/Partial/Incomplete grades, head-to-head verdict, and notes. |


