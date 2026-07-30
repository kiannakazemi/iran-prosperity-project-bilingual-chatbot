# Agentic Chunk Validation — Workflow and Prompts

This document records the agentic validation step that sits between the deterministic chunker (`chunking.py`) and the embedding step (`embed_and_store.py`). The validator audits every chunk against the source Markdown, fixes structural metadata, repairs OCR artifacts (Persian only), and writes a per-chunk changelog. It is the final quality gate before chunks are indexed into Qdrant.

---

## Why this step exists

The chunker is deterministic — it splits Markdown by headings and merges/splits by character thresholds. That is fast and reproducible, but it produces a number of structural defects that hurt retrieval quality:

1. **`header_path` noise.** Heading text often arrives with markdown formatting (`**bold**`, `● bullet`, trailing whitespace) baked in. Those characters end up in chunk metadata and pollute the embedding.
2. **`header_path` errors.** When a section ends and a new one begins, `MarkdownNodeParser` sometimes keeps the previous heading. Some chunks are labelled with the wrong sibling section, others lose an intermediate level entirely.
3. **Repetitive, low-value summaries.** The auto-generated summaries drift toward generic phrasing ("Policy directive", "Policy framework", "Strategic planning…") that does not differentiate chunks. We want a consistent, categorical form instead.
4. **Persian OCR artifacts in the body.** The Marker PDF-to-MD step leaves two recurring Persian-script defects: lam-alef letter swaps (e.g., `اسالمی` instead of `اسلامی`) and missing ZWNJ in compound words (e.g., `بینالمللی` instead of `بین‌المللی`).
5. **Split tables lose their column headers.** When the chunker slices a long table into multiple `split_part` chunks, only part 1/M carries the `| HEADER | HEADER |` row. The remaining parts retrieve as columnless data dumps.

A deterministic script cannot fix #2, #3, or the judgement-required half of #4 reliably. An LLM agent that reads the source MD and the chunks side-by-side can. That is the agent's job.

---

## Inputs and outputs

The agent runs **once per (language, white_paper)** combination.

### Inputs

| Path | Role |
|---|---|
| `rag_pipeline/indexing/chunking/by_white_paper/md/<lang>/<NN_slug>.md` | Source MD slice for that white paper. Ground truth. |
| `rag_pipeline/indexing/chunking/by_white_paper/chunks/<lang>/<NN_slug>/` | Folder of `chunk_N.txt` files to audit, with metadata header + body. |

### Outputs

| Path | Role |
|---|---|
| `rag_pipeline/indexing/chunking/by_white_paper_validated/chunks/<lang>/<NN_slug>/` | Validated chunks. EVERY input chunk is written here, modified or unchanged. |
| `rag_pipeline/indexing/chunking/by_white_paper_validated/validation_changelog.xlsx` | One Excel workbook with two sheets ("English", "Persian"). Each row = one chunk. Columns = `chunk_id, change_1, change_2, …, change_N` where N = max changes per chunk in that sheet. |

The validated folder is the source of truth for the embedding step — `embed_and_store.py` should be pointed here (or these files should be copied into `indexing/chunks/`) after validation completes.

---

## How to run

1. Start a fresh Claude session for each (language, white_paper) pair. Claude must have read access to the project folder and write access to `by_white_paper_validated/`.
2. Open the English or Persian prompt (below) and replace the `<NN_slug>` placeholder with the white-paper slug to process (e.g., `06_macroeconomic_governance`).
3. Paste the prompt. The agent reads the MD, audits each chunk, writes the validated copies, and appends the changelog rows.
4. After all white papers for a language complete, review the Excel sheet for that language to verify the change distribution makes sense, then spot-check a few validated chunks against the originals.

---

## The English prompt (used verbatim per white paper)

````markdown
You are auditing the chunks of one white paper from a policy document against the source Markdown. The MD is the ground truth. Your job is to verify each chunk, fix structural metadata where needed, and produce a changelog. You must NEVER paraphrase or alter the chunk's policy content.

# Inputs you have read access to

1. Source MD (ground truth):
   rag_pipeline/indexing/chunking/by_white_paper/md/english/<NN_slug>.md

2. Chunks folder (one chunk per .txt file, with a metadata header and body):
   rag_pipeline/indexing/chunking/by_white_paper/chunks/english/<NN_slug>/

# Outputs you must write

1. Validated chunks folder (mirrors the input layout — write EVERY chunk here, modified or not):
   rag_pipeline/indexing/chunking/by_white_paper_validated/chunks/english/<NN_slug>/

2. Excel changelog at:
   rag_pipeline/indexing/chunking/by_white_paper_validated/validation_changelog.xlsx
   Sheet name: "English". One row per chunk. Columns: chunk_id, change_1, change_2, …, change_N
   N = the max number of changes seen on any chunk in this run. Unmodified chunks have only chunk_id filled.

# For each chunk, do this

Read the chunk file. Locate where this chunk sits in the source MD. Then evaluate three things — apply each only if it needs fixing.

1. header_path
   - Replace it with the correct H1 / H2 / H3 path above the chunk in the MD when it is wrong, off-by-one, or too generic (missing intermediate levels).
   - Strip these characters from each path segment: **  __  *  ●  ○  ▪  ■
   - Trim whitespace from each segment.
   - If the chunk body opens with a heading not already in header_path, append that heading as the deepest segment.
   - The goal of the header_path control is to make sure that each chunk has a correct and clear header based on the markdown file structure.

2. summary
   - Rewrite as: "This chunk contains <category labels of content>."
   - Category labels = high-level types only (foreword, contributors list, table of contents, recommendation list, phase description, structural table, definitions, monitoring metrics, action plan, glossary, etc.). No specific names, dates, institutions, or facts.
   - When a chunk has multiple content types, list them all, joined with commas and "and".
   - Exactly one line. No line breaks.

3. table column headers
   - If the chunk has split_part = N/M with N > 1 AND its body contains markdown table rows (lines starting with |) but NO column-header row, prepend the original column-header row(s) verbatim from the MD.

# Hard constraints

- Never paraphrase the chunk's policy text. The only body edit allowed is prepending verbatim table column headers from the MD.
- Never invent content. If a heading or column header is not in the MD, leave the field unchanged and add an entry to the chunk's `notes` array (do not put it in `changes`).
- Preserve the metadata field order in every output file:
  chunk_id, chunk_index, source_file, source_pdf, language, white_paper, header_path, page_start, page_end, split_part, summary
- Keep encoding as UTF-8.
- Write EVERY chunk to the validated folder, modified or not. If unmodified, copy the file verbatim.

# Procedure

1. Read the source MD once and build a mental map of its heading hierarchy.
2. List the chunk files in the input chunks folder, sorted by chunk_index.
3. For each chunk in order:
   a. Read the chunk.
   b. Locate its position in the MD.
   c. Decide which of the three tasks apply.
   d. Build the corrected metadata and body.
   e. Write the chunk file to the validated folder.
   f. Append one row to an in-memory changelog table: chunk_id + one string per change applied. If no change, the row has only chunk_id.
4. After all chunks are processed, write the changelog to the Excel file. If the Excel file already exists (from a previous white paper's run), open it, add or overwrite the "English" sheet's rows for this white paper's chunks, and save. Otherwise create it fresh.

# Per-change wording for the Excel

Each `change_N` cell should be one concise, specific sentence in plain English. Examples of the right level of detail:
- "Stripped ** markers from header_path: /**Phase A/B5**/ → /Phase A/B5/"
- "Appended missing subsection 'Establishing Fair Vetting Committees' to header_path"
- "Replaced summary with category-only form: 'This chunk contains a structural table classifying military institutions.'"
- "Prepended 2-row column header from MD page 51 (CURRENT INSTITUTION | MAIN FUNCTION | CATEGORY)"

# Begin

Read the MD file first. Then process the chunks folder. Report progress every 10 chunks. When done, confirm: number of chunks processed, number modified, path to the validated folder, path to the Excel changelog.
````

---

## The Persian prompt (used verbatim per white paper)

````markdown
You are auditing the chunks of one white paper from a policy document against the source Markdown. The MD is the ground truth. Your job is to verify each chunk, fix structural metadata where needed, repair OCR artifacts in the body, and produce a changelog. You must NEVER paraphrase or alter the meaning of the chunk's policy content.

# Inputs you have read access to

1. Source MD (ground truth):
   rag_pipeline/indexing/chunking/by_white_paper/md/persian/<NN_slug>.md

2. Chunks folder (one chunk per .txt file, with a metadata header and body):
   rag_pipeline/indexing/chunking/by_white_paper/chunks/persian/<NN_slug>/

# Outputs you must write

1. Validated chunks folder (mirrors the input layout — write EVERY chunk here, modified or not):
   rag_pipeline/indexing/chunking/by_white_paper_validated/chunks/persian/<NN_slug>/

2. Excel changelog at:
   rag_pipeline/indexing/chunking/by_white_paper_validated/validation_changelog.xlsx
   Sheet name: "Persian". One row per chunk. Columns: chunk_id, change_1, change_2, …, change_N
   N = the max number of changes seen on any chunk in this run. Unmodified chunks have only chunk_id filled.

# For each chunk, do this

Read the chunk file. Locate where this chunk sits in the source MD. Then evaluate four things — apply each only if it needs fixing.

1. header_path
   - Replace it with the correct H1 / H2 / H3 path above the chunk in the MD when it is wrong, off-by-one, or too generic (missing intermediate levels).
   - Strip these characters from each path segment: **  __  *  ●  ○  ▪  ■
   - Trim whitespace from each segment.
   - Keep all heading text in Persian script — do not translate.
   - If the chunk body opens with a heading not already in header_path, append that heading as the deepest segment.
   - The goal of the header_path control is to make sure that each chunk has a correct and clear header based on the markdown file structure.

2. summary
   - Rewrite as: «این تکه شامل <برچسب‌های دسته محتوا> است.»
   - Category labels = high-level types only (پیش‌گفتار، فهرست نویسندگان، فهرست مطالب، فهرست توصیه‌ها، توصیف فاز، جدول ساختاری، تعاریف، شاخص‌های پایش، طرح اقدام، واژه‌نامه, etc.). No specific names, dates, institutions, or facts.
   - When a chunk has multiple content types, list them all, joined with «،» and «و» for the last.
   - Exactly one line. No line breaks.

3. OCR artifact repair in body
   - Fix lam-alef swaps where the corrected form is a real Persian word. Common patterns: اسالمی → اسلامی, اعالم → اعلام, اطالع → اطلاع, الزم → لازم, تالش → تلاش, انحالل → انحلال, اصالحات → اصلاحات, بالفاصله → بلافاصله, سامالنه → سامانه. Apply only when you are certain the corrected form is a real Persian word.
   - Insert ZWNJ (‌, U+200C) in concatenated compounds where standard Persian writing uses one. Common patterns: بینالمللی → بین‌المللی, بهعنوان → به‌عنوان, داراییها → دارایی‌ها, دانشآموزان → دانش‌آموزان, هزینهها → هزینه‌ها, نهادها → نهاد‌ها.
   - Never invent new words. If you are unsure whether a token is OCR-damaged or intentional, leave it unchanged.

4. table column headers
   - If the chunk has split_part = N/M with N > 1 AND its body contains markdown table rows (lines starting with |) but NO column-header row, prepend the original column-header row(s) verbatim from the MD.

# Hard constraints

- Never paraphrase the chunk's policy text. The only body edits allowed are the OCR repairs listed above and prepending verbatim table column headers from the MD.
- Never invent content. If a heading or column header is not in the MD, leave the field unchanged and add an entry to the chunk's `notes` array (do not put it in `changes`).
- Preserve the metadata field order in every output file:
  chunk_id, chunk_index, source_file, source_pdf, language, white_paper, header_path, page_start, page_end, split_part, summary
- Keep encoding as UTF-8. Preserve all Persian characters and the ZWNJ character (U+200C) exactly.
- Write EVERY chunk to the validated folder, modified or not. If unmodified, copy the file verbatim.

# Procedure

1. Read the source MD once and build a mental map of its heading hierarchy.
2. List the chunk files in the input chunks folder, sorted by chunk_index.
3. For each chunk in order:
   a. Read the chunk.
   b. Locate its position in the MD.
   c. Decide which of the four tasks apply.
   d. Build the corrected metadata and body.
   e. Write the chunk file to the validated folder.
   f. Append one row to an in-memory changelog table: chunk_id + one string per change applied. If no change, the row has only chunk_id.
4. After all chunks are processed, write the changelog to the Excel file. If the Excel file already exists (from a previous run), open it, add or overwrite the "Persian" sheet's rows for this white paper's chunks, and save. Preserve all other sheets (e.g., "English") intact.

# Per-change wording for the Excel

Each `change_N` cell should be one concise, specific sentence in plain English (English even when describing Persian content — for consistency with the changelog). Examples of the right level of detail:
- "Stripped ** markers from header_path: /**بخش دوم**/ → /بخش دوم/"
- "Appended missing subsection 'دسته‌بندی: نیروهای نظامی' to header_path"
- "Replaced summary with category-only form: «این تکه شامل جدول ساختاری است.»"
- "Fixed 4 lam-alef OCR artifacts in body (اسالمی → اسلامی ×3, اعالم → اعلام ×1)"
- "Inserted ZWNJ in 6 concatenated compounds (بینالمللی → بین‌المللی ×4, داراییها → دارایی‌ها ×2)"
- "Prepended 1-row column header from MD page 50 (نهاد | وظیفه / مأموریت | اقدام پیشنهادی)"

# Begin

Read the MD file first. Then process the chunks folder. Report progress every 10 chunks. When done, confirm: number of chunks processed, number modified, path to the validated folder, path to the Excel changelog.
````

---

## Notes on the prompt design

- **Two prompts, not one.** Persian carries an extra task (OCR artifact repair) that does not apply to English. Splitting keeps each prompt focused.
- **Inputs and outputs are absolute-relative paths inside the prompt.** The agent must run from the project root for these to resolve. Replace `<NN_slug>` with the actual folder name before pasting.
- **Body edits are tightly constrained.** English permits only one body edit (prepending verbatim table column headers from the MD). Persian permits the same plus two pattern-bounded OCR repairs (lam-alef swaps and ZWNJ insertion), explicitly forbidden from inventing words.
- **Change log strings are written in English even for Persian content.** This keeps the changelog scannable in one language and avoids RTL/LTR mixing inside Excel cells.
- **The `notes` field captures things the agent cannot fix.** When a heading or column header is missing from the MD itself, the agent reports it in `notes` rather than guessing or polluting `changes` with non-edits.
