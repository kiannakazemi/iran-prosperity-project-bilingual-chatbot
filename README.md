<div align="center">
  <img src="web_app/public/images/ipp-azadi-tower-logo.png" alt="Iran Prosperity Project" height="76">

  <h1>Iran Prosperity Project — Bilingual RAG Chatbot </h1>

  <p><strong>Ask questions about the Emergency Phase Booklet in English or Persian.<br>
  Every answer cites the page it came from.</strong></p>

  <p>
    <img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-blue">
    <img alt="React" src="https://img.shields.io/badge/frontend-React%20%2B%20Vite-61dafb">
    <img alt="Accuracy EN" src="https://img.shields.io/badge/accuracy_EN-95.7%25-brightgreen">
    <img alt="Accuracy FA" src="https://img.shields.io/badge/accuracy_FA-88.3%25-brightgreen">
    <img alt="Licence" src="https://img.shields.io/badge/licence-Noncommercial-blue">
  </p>
</div>

---

## Contents

- [Contents](#contents)
- [About the Iran Prosperity Project](#about-the-iran-prosperity-project)
- [The problem](#the-problem)
- [What the chatbot does](#what-the-chatbot-does)
- [Demo](#demo)
- [Accuracy](#accuracy)
- [Getting started](#getting-started)
  - [1. Clone and install](#1-clone-and-install)
  - [2. Add your API keys](#2-add-your-api-keys)
  - [3. Run it](#3-run-it)
- [API keys](#api-keys)
- [Known issues and roadmap](#known-issues-and-roadmap)
- [Contributing](#contributing)
- [Documentation](#documentation)
- [Credits and licence](#credits-and-licence)

---

## About the Iran Prosperity Project

The **Iran Prosperity Project (IPP)** is a policy-planning initiative backed by the [National Union for Democracy in Iran (NUFDI)](https://nufdi.org/), a non-profit, non-partisan organisation based in Washington, DC.

It was created to answer a question that decades of debate about Iran's future had largely left open: not *whether* the Islamic Republic ends, but **what happens in the days and weeks immediately afterwards**. 

The Project convened subject-matter experts from inside Iran and across the diaspora to produce a practical, peer-reviewed plan rather than a manifesto. Its first output is the **Emergency Phase Booklet**: a 178-page blueprint for the first 180 days of a transitional period, published in parallel English and Persian editions.

The booklet spans 15 white papers: Front Matter, Legal, Political, Military and Security, Foreign Policy, Government Essential Functions, Macroeconomic Governance, National Assets, Energy, Industry, Cybersecurity, Environment, Water, Healthcare, and the Educational System.

## The problem

The booklet is thorough, and that is exactly what makes it hard to use.

It is 178 pages of policy text. Most people with a genuine stake in it will never read it end to end — an Iranian trying to picture the morning after the Islamic Republic falls, a journalist working to deadline, a researcher comparing transition frameworks, an engineer who maintains part of the national grid. They do not need the whole document. They have one specific question.

This repository is a **bilingual chatbot** built to close that gap: it reads the booklet, answers questions in whichever language they were asked, and shows exactly which page each answer came from.

It takes no position on the plan. It reports what the document says — and says so plainly when the document doesn't cover something.

## What the chatbot does

**- Every answer is traceable.** Answers arrive with the white paper and page number beneath them. Clicking a citation opens that page of the source PDF, so any claim can be checked against the original in one step.

**- Runs on low-cost models.** It uses low-cost models with free tiers that are plenty for personal use. 
This wasn't a compromise: in testing, the cheaper Gemini model outperformed the more expensive one on this corpus.

**- Fast.** Median response time is about 2.2 seconds end to end.

**- It declines rather than guesses.** If the booklet doesn't contain an answer, the chatbot says so instead of improvising. 

**- Genuinely bilingual.** It works fully in both English and Persian, and keeps them separate — a Persian question is always answered from the Persian text, and an English one from the English text.

## Demo


https://github.com/user-attachments/assets/64c49199-3078-42b3-8c5b-e8efa4092eaa



## Accuracy

Every answer is graded against the booklet itself. English is the stronger of the two today; Persian is close behind, and improving it is where most of the ongoing work goes.

<table style="width:100%; border-collapse:collapse; font-size:16px;">
  <thead>
    <tr style="border-bottom:2px solid #57606a; text-align:left;">
      <th style="padding:12px 6px; font-weight:700;">Language</th>
      <th style="padding:12px 6px; font-weight:700; text-align:center;">Answer accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr style="border-bottom:1px solid #e5e8eb;">
      <td style="padding:14px 6px;">English</td>
      <td style="padding:14px 6px; text-align:center;"><strong>95.7%</strong></td>
    </tr>
    <tr style="border-bottom:1px solid #e5e8eb;">
      <td style="padding:14px 6px;">Persian</td>
      <td style="padding:14px 6px; text-align:center;"><strong>88.3%</strong></td>
    </tr>
  </tbody>
</table>

📊 Full methodology, per-question verdicts and negative results: **[EVALUATION.md](rag_pipeline/indexing/retrieval/eval/README.md)**

## Getting started

**Prerequisites:** Python 3.11+, Node.js 18+, and free API keys for Cohere and Google Gemini.

### 1. Clone and install

```bash
git clone https://github.com/kiannakazemi/iran-prosperity-project-bilingual-rag.git
cd iran-prosperity-project-bilingual-rag

python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1

pip install -U pip wheel
pip install -r requirements.txt
```

### 2. Add your API keys

```bash
cp .env.example .env               # Windows: Copy-Item .env.example .env
```

Open `.env` and fill in `COHERE_API_KEY` and `GEMINI_API_KEY`.

### 3. Run it

The prebuilt vector index ships with the repository, so there's nothing to embed and no indexing cost — you can query immediately.

```bash
# Terminal 1 — backend API
cd rag_pipeline
python -m api.main

# Terminal 2 — frontend
cd web_app
npm install
npm run dev
```

Open the URL Vite prints (usually `http://localhost:5173`).

**Prefer the terminal?**

```bash
cd rag_pipeline
python -m chatbot.cli
```

## API keys

| Service | Used for | Required | Get a key |
|---|---|:---:|---|
| **Cohere** | Embeddings (`embed-multilingual-v3.0`) and reranking (`rerank-v3.5`) | Yes | [dashboard.cohere.com](https://dashboard.cohere.com/api-keys) |
| **Google Gemini** | Answer generation (`gemini-3.5-flash-lite`) | Yes | [aistudio.google.com](https://aistudio.google.com/apikey) |
| **Alibaba DashScope** | Qwen model — only for regenerating chunk summaries or running the Qwen arm of the evaluation | No | [dashscope.console.aliyun.com](https://dashscope.console.aliyun.com/) |

Both required services offer free tiers that comfortably cover local use. `.env` is gitignored and should never be committed.


## Known issues and roadmap

- **Real-world user testing.** Every figure here comes from a fixed 47-question set, not live traffic. Actual phrasing will differ, and this is the biggest open question about real quality.
- **Improve Persian answer quality further**, using feedback from real Persian-speaking users rather than a curated test set.
- **Better whole-document summarisation** via a bulk white-paper load path rather than top-5 retrieval.
  
## Contributing

The most useful contribution is **a question the chatbot answers badly.**

Open an issue with:

1. the question, exactly as you asked it
2. the language
3. what the booklet actually says, and the page

Each becomes a permanent evaluation case, so the next version is measurably better. Bug reports, Persian language corrections and documentation improvements are equally welcome.

## Documentation

| Document | Contents |
|---|---|
| [rag_pipeline/README.md](rag_pipeline/README.md) | Technical reference — every pipeline stage, metadata schema, configuration constants |
| [EVALUATION.md](rag_pipeline/indexing/retrieval/eval/README.md) | Evaluation methodology, full results, limitations, negative findings |
| [AGENTIC_VALIDATION_README.md](rag_pipeline/indexing/chunking/AGENTIC_VALIDATION_README.md) | How the chunk corpus was audited against the source |

## Credits and licence

Two different things live in this repository, and they have different owners.

**The plan** — the *Emergency Phase Booklet* — is the work of the **Iran Prosperity Project**, a project of **[NUFDI](https://nufdi.org/)**. The booklet and its converted text remain their intellectual property. This repository adds nothing to the plan and takes no position on it.

**The chatbot** — the retrieval pipeline, evaluation harness, API and web interface — was designed and built by our team, and is released under the PolyForm Noncommercial License 1.0.0. It is free of charge for personal use; for any other use, please contact kianna.kazemi99@gmail.com.

---

<div align="center">
  <img src="web_app/public/images/NUFDI-logo.png" alt="NUFDI" height="42">
</div>
