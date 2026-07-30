"""
Provide the chat engine for the bilingual RAG assistant.

This module is the single source of truth for the answer path: language
detection, retrieval, prompt construction, answer generation, and source
resolution. Both the CLI and the web API consume ``ChatEngine.plan``,
``stream_answer``, and ``finalize``.

Retrieval is dense search plus Cohere reranking ("Test 2"), identical for both
languages. Answers are generated with Gemini Flash-Lite (:data:`ANSWER_MODEL`);
short helper completions such as chat titles use :data:`GEMINI_MODEL`. Both
settings were chosen by measurement — see
``indexing/retrieval/eval/EVALUATION.md``.

Usage:
    from chatbot.engine import ChatEngine
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Generator

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace"
    )

import requests
from dotenv import load_dotenv

from indexing.retrieval import RetrievalResult, ReliableRetriever

load_dotenv()

# ── USED_PASSAGES extraction ─────────────────────────────────────────

_USED_MARKER = "USED_PASSAGES:"
_DIGIT_TRANS = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
    "01234567890123456789",
)


# ── Source grounding verification ────────────────────────────────────
# The model's own USED_PASSAGES self-report is unreliable: it tends to list
# every passage that sat in its context rather than the ones it actually drew
# on. ("Who is the director?" → a one-name answer, but 3 passages claimed.)
# Showing those unused passages as sources makes the citations look random.
#
# So the self-report is treated as a *candidate list* only, and each candidate
# is then verified against the answer text: a passage is kept only if enough of
# its distinctive vocabulary actually surfaces in the answer.

_STOPWORDS_EN = {
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "at",
    "by", "for", "with", "from", "as", "is", "are", "was", "were", "be", "been",
    "being", "this", "that", "these", "those", "it", "its", "will", "shall",
    "should", "must", "may", "can", "not", "no", "all", "any", "each", "which",
    "who", "whom", "what", "when", "where", "how", "than", "then", "there",
    "their", "them", "they", "he", "she", "his", "her", "you", "your", "we",
    "our", "also", "such", "into", "under", "over", "during", "while", "both",
    "other", "more", "most", "some", "only", "own", "same", "so", "up", "out",
    "about", "after", "before", "between", "through", "within", "without",
    "including", "include", "includes", "based", "new", "used", "using", "one",
    "two", "three", "first", "second", "phase", "emergency", "booklet",
    "document", "iran", "iranian", "transitional", "transition", "national",
    "passage", "summary", "topic", "cont", "has", "have", "had", "does",
    "serves", "serve", "features", "addressing", "covers", "cover",
}
_STOPWORDS_FA = {
    "و", "در", "به", "از", "که", "این", "را", "با", "است", "برای", "آن", "یک",
    "می", "بر", "های", "تا", "هم", "یا", "اما", "اگر", "شود", "شده", "کرد",
    "کند", "باید", "دارد", "دارند", "بود", "بودن", "هر", "همه", "نیز", "چه",
    "چیست", "کسی", "کدام", "چگونه", "طبق", "بین", "روی", "سپس", "دیگر", "بیش",
    "خود", "ما", "شما", "آنها", "او", "وی", "ای", "هایی", "شامل", "عبارتند",
    "مورد", "طور", "بخش", "اساس", "اند", "کرده", "گیرد", "گردد", "نیست",
    "سند", "کتابچه", "مرحله", "اضطراری", "ایران", "گذار", "ملی", "متن",
}
# Content-bearing tokens: words of 3+ chars, or any run of digits (so "22",
# "180", "95" count as evidence — they are often the whole point of an answer).
_TOKEN_RE = re.compile(r"[0-9]+|[^\W\d_]{3,}", re.UNICODE)
# Scaffolding the chunker prepends to every chunk body; never evidence.
_CHUNK_PREFIX_RE = re.compile(r"^\s*\[(?:Summary|Topic)[^\]]*\]\s*", re.M)

# Shared *phrases* (adjacent content-word pairs) are the primary signal, not
# shared single words. Single words badly over-match: an answer that lists the
# booklet's topics naturally contains the word "cybersecurity", which would
# wrongly credit the Cybersecurity white paper. A passage genuinely used in an
# answer shares whole phrases with it ("saeed ghasseminejad", "planning tool").
_MIN_SHARED_PHRASES = 2
# Unigram fallback, used only when no passage shares any phrase — e.g. a derived
# answer like "22 authors" whose wording appears nowhere in the source.
_MIN_FALLBACK_TERMS = 2


def _content_tokens(text: str, lang: str) -> list[str]:
    """Ordered content tokens of ``text``, minus stopwords and chunk scaffolding."""
    if not text:
        return []
    text = _CHUNK_PREFIX_RE.sub("", text)
    stop = _STOPWORDS_FA if lang == "fa" else _STOPWORDS_EN
    out = []
    for tok in _TOKEN_RE.findall(text.lower()):
        tok = tok.strip("‌")      # zero-width non-joiner (Persian)
        if tok and tok not in stop:
            out.append(tok)
    return out


def _content_terms(text: str, lang: str) -> set[str]:
    """Set form of :func:`_content_tokens`."""
    return set(_content_tokens(text, lang))


def _phrases(tokens: list[str]) -> set[tuple[str, str]]:
    """Adjacent content-token pairs — a cheap stand-in for shared phrasing."""
    return {(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)}


def _grounded_passage_idxs(
    answer: str,
    sources: list[dict],
    lang: str,
    *,
    min_phrases: int = _MIN_SHARED_PHRASES,
) -> set[int]:
    """Return the ``_passage_idx`` values whose content is evident in ``answer``.

    Primary test: how many content-word *phrases* the passage shares with the
    answer. This distinguishes a passage the answer was built from (many shared
    phrases) from one that merely shares a topic keyword (none).

    If no passage shares any phrase — which happens when the answer is derived
    rather than quoted, e.g. "the booklet has 22 authors" — falls back to
    counting shared distinctive single terms, and finally to the single
    best-scoring passage, so a real answer is never left with no source.
    """
    if not sources:
        return set()

    a_tokens = _content_tokens(answer, lang)
    if not a_tokens:
        return set()
    a_phrases = _phrases(a_tokens)
    a_terms = set(a_tokens)

    per_passage = {
        s["_passage_idx"]: _content_tokens(s.get("_text", ""), lang)
        for s in sources
    }

    phrase_hits = {
        idx: len(_phrases(toks) & a_phrases) for idx, toks in per_passage.items()
    }
    kept = {i for i, n in phrase_hits.items() if n >= min_phrases}
    if kept:
        return kept

    # ── Fallback: distinctive single terms ───────────────────────────
    # A term shared by more than half the passages says nothing about which one
    # an answer came from, so discount it. Guarded with max(1, ...) so small
    # passage counts don't make the rule vacuously strict.
    doc_freq: dict[str, int] = {}
    for toks in per_passage.values():
        for t in set(toks):
            doc_freq[t] = doc_freq.get(t, 0) + 1
    limit = max(1, (len(per_passage) + 1) // 2)

    term_hits: dict[int, int] = {}
    for idx, toks in per_passage.items():
        distinctive = {t for t in set(toks) if doc_freq.get(t, 0) <= limit}
        term_hits[idx] = len(distinctive & a_terms)

    kept = {i for i, n in term_hits.items() if n >= _MIN_FALLBACK_TERMS}
    if kept:
        return kept

    best = max(term_hits, key=lambda i: term_hits[i], default=None)
    if best is not None and term_hits[best] > 0:
        return {best}
    return set()


def _extract_used_passages(text: str) -> tuple[str, set[int]]:
    """Strip the USED_PASSAGES: line and return (clean_text, passage_indices).

    Uses rfind so the last occurrence wins (guards against the model accidentally
    repeating the marker).  Extracts ALL digit sequences after the marker via
    re.findall, which tolerates any separator and any digit form (Western 0-9,
    Arabic-Indic ٠-٩, Extended Persian ۰-۹).
    """
    idx = text.rfind(_USED_MARKER)
    if idx == -1:
        return text, set()
    clean = text[:idx].rstrip()
    passage_str = text[idx + len(_USED_MARKER):]
    passage_str = passage_str.translate(_DIGIT_TRANS)
    indices = {int(x) for x in re.findall(r"\d+", passage_str)}
    return clean, indices


# ── Constants ────────────────────────────────────────────────────────

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
QWEN_API_KEY = os.environ.get("QWEN_API_KEY", "") or os.environ.get(
    "DASHSCOPE_API_KEY", ""
)

GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)
GEMINI_STREAM_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:streamGenerateContent"
)

# ── Answer model (set by evaluation) ─────────────────────────────────
# Model used to generate user-facing answers. Flash-Lite took the top cell in
# both languages across 47 graded questions per language, and was the only
# configuration with essentially no hard failures — see
# indexing/retrieval/eval/EVALUATION.md:
#   EN Test 2 + Flash-Lite  89.4% strict / 94.7% weighted, 0 incomplete
#   FA Test 2 + Flash-Lite  95.7% strict / 96.8% weighted, 1 incomplete
# Gemini Flash scored lower in every English cell, and its failure mode is the
# costlier one for a policy document: it confabulates plausible content from a
# neighbouring section (e.g. attributing another phase's actions to Phase A/B4)
# instead of declining. Flash-Lite's failures are over-caution.
# GEMINI_MODEL above is kept for short helper completions.
ANSWER_MODEL = "gemini-3.5-flash-lite"

# Qwen (DashScope OpenAI-compatible endpoint) — used by _llm_complete for
# short helper completions like chunk summaries and pronoun rewrites.
# Switch to "https://dashscope.aliyuncs.com/..." if running from mainland China.
QWEN_MODEL = "qwen-plus"
QWEN_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"

PERSIAN_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")

TOP_K = 5

# ── Retrieval configuration (set by evaluation) ──────────────────────
# Cohere rerank-v3.5 over a wider dense pool, on section-aware documents.
# Recovers rows that dense-only retrieval falsely refuses (e.g. the Healthcare
# 30-day priorities) and lets the reranker tell apart same-named sections
# ("Key Priorities") repeated under different phases.
#
# The LLM header-sibling router that was evaluated alongside this has been
# removed: it altered the retrieved set on only 5/47 EN and 9/47 FA questions,
# and every chunk it usefully added was already in the dense pool, just ranked
# below top-5 — which section-aware reranking addresses without an extra LLM
# call. See retrieval/eval/EVALUATION.md.
USE_RERANK = True

# Max output tokens per language. 1536/2048 balances list completeness vs latency;
# raise to 2048/3072 if long enumerations still truncate.
MAX_OUTPUT_TOKENS: dict[str, int] = {
    "en": 1536,
    "fa": 2048,
}


# ── Greeting detection ───────────────────────────────────────────────

_GREETINGS_EN = {
    "hi", "hello", "hey", "hi there", "hello there", "greetings",
    "good morning", "good afternoon", "good evening", "howdy",
    "what's up", "whats up", "sup",
}
_GREETINGS_FA = {
    "سلام", "درود", "سلام علیکم", "خسته نباشید", "صبح بخیر",
    "عصر بخیر", "شب بخیر",
}

GREETING_RESPONSE_EN = (
    "Hello! I'm the **IPP Assistant** for the Iran Prosperity Project's "
    "Emergency Phase Booklet. How can I help you?"
)
GREETING_RESPONSE_FA = (
    "سلام! من **دستیار IPP** برای کتابچه مرحله اضطراری پروژه شکوفایی "
    "ایران هستم. چطور می‌توانم کمکتان کنم؟"
)

GREETING_PREFIX_EN = "Hello! "
GREETING_PREFIX_FA = "سلام! "

_GREETING_ONLY_SYSTEM = """\
You judge whether the user message is ONLY a greeting or social nicety, with NO \
substantive question or request attached.

PURE greeting (reply YES): "Hi", "Hello", "Hey", "Good morning", "سلام", \
"درود", "صبح بخیر" — and nothing else.

NOT pure greeting (reply NO): any message that also asks something, requests \
help, or mentions a topic — e.g. "Hi, who is the director?", "سلام مدیر \
پروژه کیست؟", "Hello, tell me about the water white paper."

The message may be in English or Persian. Reply with exactly one word: YES or NO."""


def _is_greeting_only(question: str) -> bool:
    """True when the LLM judges the input to be a greeting with no real question."""
    try:
        raw = _gemini_generate(
            _GREETING_ONLY_SYSTEM, question, max_tokens=4, temperature=0.0
        )
        return raw.strip().upper().startswith("YES")
    except Exception:
        return False


def _starts_with_greeting(text: str) -> bool:
    """Return True if the text begins with a greeting followed by a question."""
    normalised = re.sub(r"[؟?!.,،؛\s]+", " ", text).strip().lower()
    for g in sorted(_GREETINGS_EN | _GREETINGS_FA, key=len, reverse=True):
        if normalised.startswith(g) and len(normalised) > len(g) + 2:
            return True
    return False


# ── TOC page references (PDF page markers from chunking) ─────────────

TOC_PAGE = {"en": 16, "fa": 15}
TOC_PDF = {
    "en": "Emergency_Phase_ENGLISH_20260301_1440.pdf",
    "fa": "Emergency_Phase_PERSIAN_20260301_1440.pdf",
}

# ── Shared rule fragments ─────────────────────────────────────────────

_NO_REF_RULE_EN = (
    'NEVER write "Passage N", "as stated in Passage N", "Passage 1 says", '
    '"see Passage X", "(Chunk Y)", or any similar inline reference inside your '
    "answer text — not even when defending your answer to a follow-up. The UI "
    "displays sources separately; in-text references look broken to users."
)
_NO_REASONING_RULE_EN = (
    "Present only the final answer. Never show reasoning, working, intermediate "
    "steps, or self-correction. Do any counting or combining internally and "
    "output only the finished result. Never recount or verify inside the answer."
)
_SCOPE_RULE_EN = (
    "SECTION SCOPE. Each passage carries a [Section: ...] heading trail. When "
    "the question names a specific phase, day range, section, or time window "
    '(e.g. "the first 30 days", "Phase 1", "Week 1"), use ONLY passages whose '
    "Section trail matches it. The document repeats headings like Objectives, "
    "Key Priorities and Actions under every phase, so an identical-looking list "
    "from a different phase is the WRONG answer. If a passage's Section shows a "
    "different phase than the one asked about, ignore it — do not merge it in. "
    "State the phase you are answering for."
)
_SCOPE_RULE_FA = (
    "محدوده بخش. هر متن یک مسیر عنوان [Section: ...] دارد. وقتی پرسش به مرحله، "
    "بازه روزی، بخش یا دوره زمانی مشخصی اشاره می‌کند (مثلاً «۳۰ روز اول»، «مرحله "
    "۱»، «هفته اول»)، فقط از متن‌هایی استفاده کنید که مسیر عنوان آنها با همان "
    "مطابقت دارد. سند عنوان‌هایی مانند اهداف، اولویت‌های کلیدی و اقدامات را در هر "
    "مرحله تکرار می‌کند، بنابراین فهرستی مشابه از مرحله‌ای دیگر پاسخ نادرست است. "
    "اگر مسیر عنوان یک متن مرحله‌ای متفاوت از پرسش را نشان می‌دهد، آن را نادیده "
    "بگیرید و ادغام نکنید. مرحله‌ای که پاسخ می‌دهید را ذکر کنید."
)
_COUNT_RULE_EN = (
    "COUNTS. When the user asks how many of something there are, give the "
    "number AND then enumerate the items themselves if the CONTEXT contains "
    'them (e.g. "The booklet has 22 authors:" followed by the list of names). '
    "A bare number is not a sufficient answer. If the CONTEXT has the count but "
    "not the individual items, give the number and say the list is not included "
    "in the retrieved sections."
)
_COUNT_RULE_FA = (
    "پرسش‌های شمارشی. وقتی کاربر می‌پرسد تعداد چیزی چقدر است، هم عدد را بدهید و "
    "هم اگر CONTEXT خود موارد را دارد آنها را فهرست کنید (مثلاً «این کتابچه ۲۲ "
    "نویسنده دارد:» و سپس فهرست نام‌ها). پاسخ دادن با عدد تنها کافی نیست. اگر "
    "CONTEXT عدد را دارد اما موارد را ندارد، عدد را بدهید و بگویید فهرست در "
    "بخش‌های بازیابی‌شده نیامده است."
)
_USED_LINE_EN = (
    "LAST LINE (hidden from user):\n"
    "Write exactly: USED_PASSAGES: followed by the numbers of the passages you "
    "actually quoted or paraphrased (e.g. USED_PASSAGES: 1, 3). Include a number "
    "ONLY if a specific fact from that passage appears in your answer text. Do "
    "NOT list a passage merely because it was provided, looked relevant, or "
    "shares a topic with the question. Fewer is better: if two passages say the "
    "same thing, cite only the one you drew from. If none: USED_PASSAGES:"
)

_NO_REF_RULE_FA = (
    "هرگز در داخل متن پاسخ ننویسید: «متن N»، «طبق متن N»، «متن ۱ می‌گوید»، "
    "«به متن X مراجعه کنید»، «(چانک Y)» یا هر ارجاع مشابه — حتی هنگام دفاع از "
    "پاسخ در سؤال بعدی. سیستم منابع را جداگانه نمایش می‌دهد."
)
_NO_REASONING_RULE_FA = (
    "فقط پاسخ نهایی را بنویسید. هرگز استدلال، محاسبات میانی، مراحل کار یا تصحیح "
    "خود را نشان ندهید. هر شمارش یا ترکیب را به‌صورت داخلی انجام دهید و فقط نتیجه "
    "نهایی را بنویسید. هرگز درون پاسخ بازشماری یا تأیید نکنید."
)
_USED_LINE_FA = (
    "آخرین خط (از کاربر پنهان است):\n"
    "دقیقاً بنویسید: USED_PASSAGES: و سپس شماره متن‌هایی که واقعاً از آنها نقل یا "
    "برداشت کردید با ارقام انگلیسی (مثلاً USED_PASSAGES: 1, 3). شماره‌ای را تنها "
    "زمانی بنویسید که واقعیتی مشخص از آن متن در پاسخ شما آمده باشد. متنی را فقط "
    "به این دلیل که ارائه شده، مرتبط به نظر می‌رسید یا موضوع مشترکی با پرسش دارد "
    "ذکر نکنید. کمتر بهتر است: اگر دو متن یک چیز می‌گویند، فقط همان را که از آن "
    "استفاده کردید ذکر کنید. اگر هیچ‌کدام: USED_PASSAGES:"
)

_REFUSAL_LINE_EN = (
    'If the answer is not in the CONTEXT, reply EXACTLY: "This information is not '
    'covered in the Emergency Phase Booklet." and then write USED_PASSAGES: with '
    "no numbers. Do not pad a refusal with related-but-different content."
)
_REFUSAL_LINE_FA = (
    "اگر پاسخ در متن‌ها نیست، دقیقاً بنویسید: «این اطلاعات در کتابچه مرحله "
    "اضطراری وجود ندارد.» و سپس USED_PASSAGES: را بدون هیچ شماره‌ای بنویسید. "
    "پاسخ «وجود ندارد» را با محتوای مرتبط اما متفاوت پر نکنید."
)

_PARTIAL_COVERAGE_RULE_EN = (
    "PARTIAL COVERAGE. If the question has multiple parts and some parts are "
    "covered in the CONTEXT while others are not, address each part: state the "
    "covered parts from the CONTEXT, and for each uncovered part explicitly "
    "say it is not covered in the Emergency Phase Booklet. Do NOT silently "
    "omit any part of the question. Do NOT refuse the whole answer just "
    "because one part is missing. Do NOT invent information for the missing "
    "parts."
)
_PARTIAL_COVERAGE_RULE_FA = (
    "پوشش نسبی. اگر پرسش چند بخشی دارد و برخی بخش‌ها در متن‌ها آمده اما "
    "برخی نیامده‌اند، به هر بخش جداگانه پاسخ دهید: بخش‌های پوشش‌داده‌شده را "
    "از متن‌ها بیان کنید، و برای هر بخش پوشش‌داده‌نشده صریحاً بگویید که در "
    "کتابچه مرحله اضطراری نیامده است. هیچ بخشی از پرسش را بی‌سر‌و‌صدا نادیده "
    "نگیرید. تنها به این دلیل که یک بخش پاسخ ندارد، کل پاسخ را رد نکنید. "
    "برای بخش‌های مفقود اطلاعات نسازید."
)

# ── Document identity (a fixed, always-available context tag) ──────────
# These are established facts about THIS specific booklet — its producer,
# backer, key people, and coverage. They are a valid source for questions
# about the document/project/booklet as a whole, alongside the retrieved
# CONTEXT. Update this block by hand if the booklet's leadership changes.
_DOC_IDENTITY_EN = (
    "[DOCUMENT IDENTITY] Established facts about this specific booklet, usable "
    "as context for questions about the document/project/booklet as a whole:\n"
    "- Producer: the Iran Prosperity Project (IPP).\n"
    "- Backing organization: the National Union for Democracy in Iran (NUFDI).\n"
    "- Project Director: Saeed Ghasseminejad.\n"
    "- NUFDI President/CEO: Dr. Saeed Ganji.\n"
    "- Foreword contributor / supporter: Crown Prince Reza Pahlavi.\n"
    "- Contributors: 22 authors, 39 advisors, and an 11-member executive team.\n"
    "- Coverage: 15 white papers — Front Matter, Legal, Political, Military and "
    "Security, Foreign Policy, Government Essential Functions, Macroeconomic "
    "Governance, National Assets, Energy, Industry, Cybersecurity, Environment, "
    "Water, Healthcare, and Educational System."
)
_DOC_IDENTITY_FA = (
    "[هویت سند] واقعیت‌های تثبیت‌شده درباره این کتابچه مشخص، قابل استفاده "
    "به‌عنوان زمینه برای پرسش‌هایی درباره کل سند/پروژه/کتابچه:\n"
    "- تهیه‌کننده: پروژه شکوفایی ایران (IPP).\n"
    "- سازمان پشتیبان: اتحادیه ملی برای دموکراسی در ایران (نوفدی).\n"
    "- مدیر پروژه: سعید قاسمی‌نژاد.\n"
    "- رئیس/مدیرعامل نوفدی: دکتر سعید گنجی.\n"
    "- نویسنده پیش‌گفتار / پشتیبان: شاهزاده رضا پهلوی.\n"
    "- مشارکت‌کنندگان: ۲۲ نویسنده، ۳۹ مشاور، و تیم اجرایی ۱۱ نفره.\n"
    "- پوشش: ۱۵ سپیدنامه — پیش‌گفتار، حقوقی، سیاسی، نظامی و امنیتی، سیاست "
    "خارجی، کارکردهای ضروری دولت، حکمرانی اقتصاد کلان، دارایی‌های ملی، انرژی، "
    "صنعت، امنیت سایبری، محیط‌زیست، آب، بهداشت و درمان، و نظام آموزشی."
)

_PROJECT_RULE_EN = (
    "PROJECT QUESTIONS. When the user asks about the document, the project, or "
    "the booklet as a whole (what it is, who is behind it, who produced it, who "
    "supports it, what it covers), treat DOCUMENT IDENTITY above as valid "
    "context and answer from it together with anything relevant you retrieved: "
    "name the key parties explicitly — the Iran Prosperity Project, NUFDI, the "
    "Project Director (Saeed Ghasseminejad), the NUFDI President/CEO (Dr. Saeed "
    "Ganji), and the foreword contributor (Crown Prince Reza Pahlavi). Do NOT "
    "apply this to questions about a specific white paper, section, person, or "
    "fact — for those, answer only from the CONTEXT passages as usual."
)
_PROJECT_RULE_FA = (
    "پرسش‌های مربوط به پروژه. وقتی کاربر درباره سند، پروژه یا کتابچه به‌عنوان "
    "یک کل می‌پرسد (چیست، چه کسی پشت آن است، چه کسی آن را تهیه کرده، چه کسی "
    "از آن حمایت می‌کند، چه چیزی را پوشش می‌دهد)، «هویت سند» بالا را زمینه "
    "معتبر بدانید و با استفاده از آن به‌همراه هر مطلب مرتبطی که بازیابی کرده‌اید "
    "پاسخ دهید: طرف‌های کلیدی را صریحاً نام ببرید — پروژه شکوفایی ایران، نوفدی، "
    "مدیر پروژه (سعید قاسمی‌نژاد)، رئیس/مدیرعامل نوفدی (دکتر سعید گنجی)، و "
    "نویسنده پیش‌گفتار (شاهزاده رضا پهلوی). این قانون را برای پرسش‌های مربوط "
    "به یک سپیدنامه، بخش، شخص یا واقعیت خاص به‌کار نبرید — برای آن‌ها طبق "
    "معمول فقط از متن‌های CONTEXT پاسخ دهید."
)

# ── System prompts (unified EN + FA) ─────────────────────────────────

SYSTEM_PROMPT_EN = f"""\
You are the IPP Assistant for the Iran Prosperity Project's Emergency Phase Booklet.

This is a governance blueprint document. Answer using ONLY the CONTEXT passages \
below (plus DOCUMENT IDENTITY for whole-document questions). Neutral, factual \
tone. No emojis. No outside knowledge.

{_DOC_IDENTITY_EN}

ABSOLUTE RULES:
1. Answer ONLY from the CONTEXT. {_REFUSAL_LINE_EN}
2. {_PROJECT_RULE_EN}
3. {_PARTIAL_COVERAGE_RULE_EN}
4. {_NO_REF_RULE_EN}
5. {_NO_REASONING_RULE_EN}
6. {_SCOPE_RULE_EN}
7. {_COUNT_RULE_EN}
8. Complete every list and sentence you start. Do not truncate mid-item.
9. Answer in English.



{_USED_LINE_EN}"""

SYSTEM_PROMPT_FA = f"""\
شما دستیار IPP برای کتابچه مرحله اضطراری پروژه شکوفایی ایران هستید.

این یک سند نقشه راه حکمرانی است. فقط با استفاده از متن‌های \
زمینه (CONTEXT) زیر (به‌علاوه «هویت سند» برای پرسش‌های مربوط به کل سند) پاسخ \
دهید. لحن خنثی و واقع‌محور. بدون ایموجی. بدون اطلاعات خارج از سند.

{_DOC_IDENTITY_FA}

قوانین مطلق:
۱. فقط از CONTEXT پاسخ دهید. {_REFUSAL_LINE_FA}
۲. {_PROJECT_RULE_FA}
۳. {_PARTIAL_COVERAGE_RULE_FA}
۴. {_NO_REF_RULE_FA}
۵. {_NO_REASONING_RULE_FA}
۶. {_SCOPE_RULE_FA}
۷. {_COUNT_RULE_FA}
۸. هر فهرست و جمله‌ای را که شروع می‌کنید تمام کنید. نیمه‌کاره متوقف نشوید.
۹. به فارسی پاسخ دهید.



{_USED_LINE_FA}"""


def _system_prompt(lang: str) -> str:
    return SYSTEM_PROMPT_FA if lang == "fa" else SYSTEM_PROMPT_EN


LOW_CONFIDENCE_EN = (
    "This information is not covered in the Emergency Phase Booklet. "
    "You can check the Table of Contents for available topics."
)
LOW_CONFIDENCE_FA = (
    "این اطلاعات در کتابچه مرحله اضطراری وجود ندارد. "
    "می‌توانید فهرست مطالب را برای موضوعات موجود بررسی کنید."
)

# Sentinels used to detect a refusal answer so we can suppress sources.
_REFUSAL_SENTINELS = (
    "not covered in the emergency phase booklet",
    "در کتابچه مرحله اضطراری وجود ندارد",
)
# Below this many characters of non-refusal text, an answer counts as a pure
# refusal (no sources). Above it, the answer covered something and keeps them.
_REFUSAL_REMAINDER_CHARS = 60

# White paper name translations (English → Persian)
_WP_NAME_FA = {
    "Front Matter": "پیش‌گفتار",
}


def _is_refusal(text: str) -> bool:
    """True when the answer is *nothing but* a 'not covered' refusal.

    Must not fire on partial-coverage answers. Those legitimately end with the
    refusal sentence for the part that isn't in the document — "here are the six
    priorities … the specific budget is not covered" — and a plain substring
    test would treat that as a refusal and suppress the sources for the part
    that *was* answered, leaving a fully-sourced answer with no citations.

    So the sentinel sentence is removed first, and the answer only counts as a
    refusal if little of substance remains.
    """
    low = text.strip().lower()
    if not any(s in low for s in _REFUSAL_SENTINELS):
        return False

    # Drop the sentences carrying a sentinel, keeping line structure intact so
    # list markers stay detectable.
    kept_lines: list[str] = []
    for line in low.splitlines():
        sentences = re.split(r"(?<=[.!?۔])\s+", line)
        keep = [
            s for s in sentences
            if s.strip() and not any(x in s for x in _REFUSAL_SENTINELS)
        ]
        if keep:
            kept_lines.append(" ".join(keep))
    remainder = "\n".join(kept_lines).strip()

    # A list of any length means the answer covered something.
    if re.search(r"^\s*(?:[-*•‣]|\d+[.)])\s", remainder, re.M):
        return False
    return len(remainder) < _REFUSAL_REMAINDER_CHARS


# ── Gemini REST helpers ───────────────────────────────────────────────

def _gemini_generate(
    system: str,
    user_text: str,
    *,
    max_tokens: int,
    temperature: float = 0.1,
    model: str = GEMINI_MODEL,
) -> str:
    """Single non-streaming Gemini call. Returns the answer text (may be empty).

    The ``model`` kwarg lets callers target a specific Gemini variant
    (e.g. ``gemini-3.5-flash-lite``) using the same API key. Defaults to
    ``GEMINI_MODEL`` so existing callers are unaffected.

    On 400 errors caused by the ``thinkingConfig`` block (which newer 3.x
    models reject), the call automatically retries without that block so
    the same code path works across model generations.
    """
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set")
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent"
    )

    def _build_payload(include_thinking: bool) -> dict:
        gen_cfg: dict = {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        }
        if include_thinking:
            gen_cfg["thinkingConfig"] = {"thinkingBudget": 0}
        p: dict = {
            "contents": [{"role": "user", "parts": [{"text": user_text}]}],
            "generationConfig": gen_cfg,
        }
        if system:
            p["system_instruction"] = {"parts": [{"text": system}]}
        return p

    def _post(payload: dict):
        return requests.post(
            url, params={"key": GEMINI_API_KEY}, json=payload, timeout=60
        )

    resp = _post(_build_payload(include_thinking=True))

    # Any 400 → try the simpler payload without thinkingConfig once.
    # Google's error text is often generic ("Request contains an invalid
    # argument.") and doesn't say which field is the problem, so we always
    # give the simpler shape a chance.
    if resp.status_code == 400:
        resp2 = _post(_build_payload(include_thinking=False))
        if resp2.ok:
            resp = resp2
        else:
            # Still failing — keep whichever error is more informative.
            def _detail_len(r) -> int:
                try:
                    j = r.json()
                    return len(((j.get("error") or {}).get("message") or "")) + len(str((j.get("error") or {}).get("details") or ""))
                except Exception:
                    return 0
            if _detail_len(resp2) > _detail_len(resp):
                resp = resp2

    if not resp.ok:
        # Surface Google's actual error including any details array.
        try:
            j = resp.json()
            err = j.get("error") or {}
            msg = err.get("message") or resp.text
            details = err.get("details")
            if details:
                msg = f"{msg} | details={details}"
        except Exception:
            msg = resp.text
        raise RuntimeError(f"Gemini HTTP {resp.status_code} ({model}): {msg[:600]}")

    data = resp.json()
    candidates = data.get("candidates", [])
    if not candidates:
        return ""
    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts).strip()


def _gemini_generate_stream(
    system: str,
    user_text: str,
    *,
    max_tokens: int,
    temperature: float = 0.1,
    model: str = ANSWER_MODEL,
) -> Generator[str, None, None]:
    """Streaming Gemini call via SSE; yields text chunks.

    ``model`` defaults to :data:`ANSWER_MODEL`. Gemini 3.x models reject the
    ``thinkingConfig`` block, so on a 400 the call retries without it — same
    behaviour as :func:`_gemini_generate`, kept in sync deliberately.
    """
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set")
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:streamGenerateContent"
    )

    def _build_payload(include_thinking: bool) -> dict:
        gen_cfg: dict = {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        }
        if include_thinking:
            gen_cfg["thinkingConfig"] = {"thinkingBudget": 0}
        p: dict = {
            "contents": [{"role": "user", "parts": [{"text": user_text}]}],
            "generationConfig": gen_cfg,
        }
        if system:
            p["system_instruction"] = {"parts": [{"text": system}]}
        return p

    def _post(payload: dict):
        return requests.post(
            url,
            params={"key": GEMINI_API_KEY, "alt": "sse"},
            json=payload,
            timeout=60,
            stream=True,
        )

    resp = _post(_build_payload(True))
    if resp.status_code == 400:
        resp.close()
        resp = _post(_build_payload(False))
    resp.raise_for_status()

    # Gemini sends `Content-Type: text/event-stream` with no charset. Per the
    # HTTP spec, requests then defaults `text/*` to ISO-8859-1, so
    # iter_lines(decode_unicode=True) would decode UTF-8 bytes as latin-1 and
    # mangle every non-ASCII character — fatal for Persian, and visible in
    # English as mojibake in transliterated names ("Nasırkhani" → "NasÄ±rkhani").
    # Pin the encoding before iterating.
    resp.encoding = "utf-8"

    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        raw = line[6:]
        if raw.strip() == "[DONE]":
            break
        try:
            chunk = json.loads(raw)
            parts = (
                chunk.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [])
            )
            for p in parts:
                text = p.get("text", "")
                if text:
                    yield text
        except (json.JSONDecodeError, IndexError):
            continue


def _qwen_complete(
    system: str, user_text: str, *, max_tokens: int, temperature: float = 0.1
) -> str:
    """Single non-streaming Qwen call via DashScope's OpenAI-compatible API.

    Used by ``_llm_complete`` for short helper completions: chunk summaries,
    pronoun rewriting on follow-up questions, etc. Returns the response text
    (may be empty on rare API hiccups). Retries once on 5xx / network errors;
    surfaces the actual API error body on 4xx so debugging is possible.
    """
    if not QWEN_API_KEY:
        raise RuntimeError("QWEN_API_KEY (or DASHSCOPE_API_KEY) not set")
    # Reject obviously-empty input early — DashScope returns 400 on empty user
    # content, and the caller will treat the missing return as a no-op anyway.
    if not (user_text or "").strip():
        return ""
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_text})
    payload = {
        "model": QWEN_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {QWEN_API_KEY}",
        "Content-Type": "application/json; charset=utf-8",
        "Accept": "application/json",
        "User-Agent": "IPP-Assistant/1.0 (python-requests)",
    }
    last_err: Exception | None = None
    for attempt in range(2):
        try:
            # allow_redirects=False — DashScope sometimes 30x's to a different
            # path which 'requests' would re-issue as a GET, surfacing as
            # 'Request method GET is not supported'. We surface the redirect
            # explicitly instead so the URL can be fixed at the source.
            resp = requests.post(
                QWEN_URL,
                json=payload,
                headers=headers,
                timeout=60,
                allow_redirects=False,
            )
            if 300 <= resp.status_code < 400:
                loc = resp.headers.get("Location", "<no Location header>")
                raise RuntimeError(
                    f"Qwen returned redirect HTTP {resp.status_code} → {loc}. "
                    f"Update QWEN_URL to this target."
                )
            if not resp.ok:
                # Surface the actual API error body — raise_for_status() hides it.
                try:
                    body = resp.json()
                    msg = (
                        (body.get("error") or {}).get("message")
                        or body.get("message")
                        or json.dumps(body, ensure_ascii=False)[:300]
                    )
                except Exception:
                    msg = (resp.text or "")[:300]
                err = RuntimeError(f"Qwen HTTP {resp.status_code}: {msg}")
                # Don't retry 4xx — those are payload/content problems, not transient.
                if 400 <= resp.status_code < 500:
                    raise err
                last_err = err
                if attempt < 1:
                    time.sleep(1.5)
                    continue
                raise err
            data = resp.json()
            choices = data.get("choices", [])
            if not choices:
                return ""
            return (choices[0].get("message") or {}).get("content", "").strip()
        except requests.RequestException as e:
            last_err = e
            if attempt < 1:
                time.sleep(1.5)
                continue
            raise
    raise RuntimeError(f"Qwen failed after retry: {last_err}")


def _llm_complete(system: str, prompt: str, *, max_tokens: int = 60) -> str:
    """Minimal completion used for chunk summaries, titles, rewrites.

    Tries Qwen first. Falls back to Gemini only when DashScope refuses the
    request for content-moderation reasons (a small fraction of politically
    sensitive passages in the IPP booklet trip Alibaba's safety filter).
    Other failures (auth, network, malformed payload) are re-raised so
    the caller can decide how to handle them.
    """
    try:
        return _qwen_complete(system, prompt, max_tokens=max_tokens)
    except RuntimeError as e:
        msg = str(e).lower()
        moderated = (
            "inappropriate content" in msg
            or "data_inspection_failed" in msg
            or "dataınspectionfailed" in msg  # observed variant
        )
        if not moderated:
            raise
        # Silent fallback. If Gemini isn't configured this will raise too,
        # which the caller (chunking.py) already handles by storing "" for
        # that chunk's summary.
        return _gemini_generate(system, prompt, max_tokens=max_tokens)


# ── Data classes ─────────────────────────────────────────────────────

@dataclass
class Source:
    """One retrieved passage used to form the answer."""
    white_paper: str
    page_start: int
    page_end: int
    text_preview: str
    rerank_score: float
    retrieval_source: str
    language: str = "en"


@dataclass
class ChatResponse:
    """Full response from the chat engine."""
    text: str
    language: str
    confident: bool
    provider: str = ""
    sources: list[Source] = field(default_factory=list)


@dataclass
class StreamPlan:
    """Everything the transport layer needs to stream one answer.

    ``kind`` is one of:
      - "greeting": emit ``early_text`` and finish (no retrieval).
      - "lowconf":  emit ``early_text`` + ``early_sources`` (TOC) and finish.
      - "answer":   stream the model using ``system_prompt`` / ``context`` /
                    ``question`` / ``max_tokens``, then call ``finalize``.
    """
    lang: str
    kind: str
    early_text: str = ""
    early_sources: list[dict] = field(default_factory=list)
    system_prompt: str = ""
    context: str = ""
    question: str = ""
    max_tokens: int = 1024
    has_greeting_prefix: bool = False
    all_sources: list[dict] = field(default_factory=list)


# ── Chat engine ──────────────────────────────────────────────────────

class ChatEngine:
    """End-to-end RAG chatbot: detect language → retrieve → generate.

    Answer model: :data:`ANSWER_MODEL` (Gemini Flash-Lite). Short helper
    completions — pronoun rewrites, chat titles — still use
    :data:`GEMINI_MODEL`.

    Retrieval is dense + Cohere rerank (Test 2) for both languages, as measured
    in ``retrieval/eval/EVALUATION.md``. ``TOP_K`` passages plus split-chunk
    siblings and any stitched section continuations are sent to the LLM. There is
    no confidence floor; refusal is handled by the system prompt.
    """

    def __init__(
        self,
        *,
        retriever: ReliableRetriever | None = None,
        gemini_key: str = GEMINI_API_KEY,
    ):
        self._gemini_key = gemini_key
        self._qwen_key = QWEN_API_KEY  # unused; kept for env compatibility

        if not self._gemini_key:
            raise ValueError("No GEMINI_API_KEY set. Add it to .env.")

        if retriever is not None:
            self._retriever = retriever
        else:
            print("  Initialising retriever …")
            self._retriever = ReliableRetriever()

        print(f"  Answer model: {ANSWER_MODEL}")
        print(f"  Retrieval:    Test 2 (dense + section-aware rerank)  top_k={TOP_K}")
        print("  ChatEngine ready.\n")

    # ── Planning (blocking: retrieve → build context) ─────────────────

    def plan(self, question: str, history_text: str = "") -> StreamPlan:
        """Run everything up to (but not including) answer generation.

        Blocking — run from a thread in async contexts.
        """
        lang = self._detect_language(question)

        if _is_greeting_only(question):
            greeting = GREETING_RESPONSE_FA if lang == "fa" else GREETING_RESPONSE_EN
            return StreamPlan(lang=lang, kind="greeting", early_text=greeting)

        has_prefix = _starts_with_greeting(question)

        # Search with the raw question, or rewrite follow-ups using chat history.
        if history_text:
            search_query = self._rewrite_with_history(question, history_text, lang)
        else:
            search_query = question

        results = self._retriever.query(
            search_query,
            language=lang,
            final_top_k=TOP_K,
            rerank=USE_RERANK,
        )

        # No confidence floor. Refusal is handled by the system prompt
        # ("information not covered") when the passages don't contain the
        # answer. We only fall back if retrieval returns nothing.
        if not results:
            return StreamPlan(
                lang=lang,
                kind="lowconf",
                early_text=LOW_CONFIDENCE_FA if lang == "fa" else LOW_CONFIDENCE_EN,
                early_sources=self._toc_source_dicts(lang),
            )

        context = self._build_context(results)
        if history_text:
            context = (
                f"[CONVERSATION HISTORY]\n{history_text}\n"
                f"[END CONVERSATION HISTORY]\n\n{context}"
            )

        all_sources = [
            self._result_to_source_dict(r, i + 1)
            for i, r in enumerate(results)
        ]

        return StreamPlan(
            lang=lang,
            kind="answer",
            system_prompt=_system_prompt(lang),
            context=context,
            question=question,
            max_tokens=MAX_OUTPUT_TOKENS.get(lang, 1536),
            has_greeting_prefix=has_prefix,
            all_sources=all_sources,
        )

    def stream_answer(
        self, plan: StreamPlan, provider_out: list[str] | None = None
    ) -> Generator[str, None, None]:
        """Stream answer tokens for an ``kind == 'answer'`` plan.

        Yields the greeting prefix first (if any), then the model output, so the
        caller can simply concatenate everything it receives.
        """
        if plan.has_greeting_prefix:
            yield GREETING_PREFIX_FA if plan.lang == "fa" else GREETING_PREFIX_EN
        yield from self._call_llm_stream(
            plan.system_prompt,
            plan.context,
            plan.question,
            provider_out=provider_out,
            max_tokens=plan.max_tokens,
        )

    def finalize(
        self, full_text: str, plan: StreamPlan
    ) -> tuple[str, list[dict]]:
        """Strip the USED_PASSAGES marker and resolve the sources to display.

        Returns ``(clean_text, sources_dicts)``. Sources are suppressed entirely
        for refusal answers. Otherwise the model's USED_PASSAGES self-report is
        taken as a *candidate* list and then verified against the answer text —
        see :func:`_grounded_passage_idxs`. Only passages whose content is
        actually evident in the answer are shown, so citations match what the
        user just read instead of everything that happened to be retrieved.
        """
        clean_text, used_ids = _extract_used_passages(full_text)

        if _is_refusal(clean_text):
            return clean_text, []

        candidates = plan.all_sources
        if used_ids:
            claimed = [s for s in candidates if s["_passage_idx"] in used_ids]
            if claimed:
                candidates = claimed

        grounded = _grounded_passage_idxs(clean_text, candidates, plan.lang)
        sources = [
            self._strip_idx(s) for s in candidates
            if s["_passage_idx"] in grounded
        ]

        if not sources:
            sources = self._fallback_sources(plan.all_sources)

        return clean_text, sources

    # ── Public API (used by the CLI) ─────────────────────────────────

    def ask(self, question: str) -> ChatResponse:
        """Answer a question using the full RAG pipeline (non-streaming)."""
        plan = self.plan(question)
        if plan.kind == "greeting":
            return ChatResponse(
                text=plan.early_text, language=plan.lang, confident=True,
                provider="greeting",
            )
        if plan.kind == "lowconf":
            return ChatResponse(
                text=plan.early_text, language=plan.lang, confident=False,
                sources=self._dicts_to_sources(plan.early_sources),
            )

        prefix = ""
        if plan.has_greeting_prefix:
            prefix = GREETING_PREFIX_FA if plan.lang == "fa" else GREETING_PREFIX_EN
        answer_text, provider = self._call_llm(
            plan.system_prompt, plan.context, plan.question,
            max_tokens=plan.max_tokens,
        )
        full_text = prefix + answer_text
        clean_text, sources = self.finalize(full_text, plan)
        return ChatResponse(
            text=clean_text, language=plan.lang, confident=True,
            provider=provider, sources=self._dicts_to_sources(sources),
        )

    def ask_stream(self, question: str) -> Generator[str, None, ChatResponse]:
        """Streaming version: yields text chunks, returns full ChatResponse."""
        plan = self.plan(question)

        if plan.kind == "greeting":
            yield plan.early_text
            return ChatResponse(
                text=plan.early_text, language=plan.lang, confident=True,
                provider="greeting",
            )
        if plan.kind == "lowconf":
            yield plan.early_text
            return ChatResponse(
                text=plan.early_text, language=plan.lang, confident=False,
                sources=self._dicts_to_sources(plan.early_sources),
            )

        provider_holder: list[str] = []
        full_text = ""
        for chunk in self.stream_answer(plan, provider_holder):
            full_text += chunk
            yield chunk
        provider = provider_holder[0] if provider_holder else ANSWER_MODEL

        clean_text, sources = self.finalize(full_text, plan)
        return ChatResponse(
            text=clean_text, language=plan.lang, confident=True,
            provider=provider, sources=self._dicts_to_sources(sources),
        )

    # ── Language detection ───────────────────────────────────────────

    def _detect_language(self, text: str) -> str:
        persian_chars = len(PERSIAN_RE.findall(text))
        latin_chars = len(re.findall(r"[a-zA-Z]", text))
        return "fa" if persian_chars > latin_chars else "en"

    # ── Follow-up query rewrite ──────────────────────────────────────

    def _rewrite_with_history(
        self, question: str, history: str, lang: str = "en"
    ) -> str:
        """Rewrite a follow-up question into a standalone search query."""
        try:
            if lang == "fa":
                system = "شما یک بازنویس پرسش هستید. فقط پرسش بازنویسی‌شده را به فارسی خروجی دهید."
                prompt = (
                    "سؤال FOLLOW-UP زیر را به یک پرسش جستجوی مستقل تبدیل کنید که تمام "
                    "ضمایر و ارجاعات را با استفاده از تاریخچه مکالمه حل کند. فقط پرسش "
                    "بازنویسی‌شده را به فارسی برگردانید. اگر سؤال قبلاً مستقل است، "
                    "بدون تغییر برگردانید.\n\n"
                    f"تاریخچه مکالمه:\n{history}\n\n"
                    f"سؤال FOLLOW-UP: {question}\n\nپرسش مستقل:"
                )
            else:
                system = "You are a query rewriter. Output only the rewritten query in English."
                prompt = (
                    "Rewrite the following FOLLOW-UP QUESTION into a STANDALONE search "
                    "query that resolves all pronouns (he, she, it, they, this, that) "
                    "using the conversation history below. Return ONLY the rewritten "
                    "query in English. If it is already standalone, return it unchanged.\n\n"
                    f"CONVERSATION HISTORY:\n{history}\n\n"
                    f"FOLLOW-UP QUESTION: {question}\n\nSTANDALONE QUERY:"
                )
            rewritten = _llm_complete(system, prompt, max_tokens=120).strip()
            if rewritten and len(rewritten) < 500:
                return rewritten
        except Exception:
            pass
        return question

    # ── Context building ─────────────────────────────────────────────

    def _build_context(self, results: list[RetrievalResult]) -> str:
        """Format retrieved passages for the LLM.

        Each passage carries its ``header_path`` — the heading trail it was cut
        from. Without it the model cannot tell which section a bullet list
        belongs to: the Healthcare paper has a "Key Priorities" list under every
        phase, so "priorities for the first 30 days" would be answered by mixing
        Phase 1, Phase 2 and Preparedness Phase items indiscriminately. The
        heading trail is what makes phase- and section-scoped questions
        answerable, and it costs one short line per passage.
        """
        parts: list[str] = []
        for i, r in enumerate(results, 1):
            wp = r.metadata.get("white_paper", "Unknown")
            ps = int(r.metadata.get("page_start", 0))
            pe = int(r.metadata.get("page_end", 0))
            page_info = f"p.{ps}" if ps == pe else f"pp.{ps}-{pe}"

            header = (r.metadata.get("header_path") or "").strip("/")
            section = f"\n[Section: {header.replace('/', ' > ')}]" if header else ""

            parts.append(
                f"[Passage {i} — Source: {wp}, {page_info}]{section}\n{r.text}"
            )
        return "\n\n---\n\n".join(parts)

    # ── Source helpers ───────────────────────────────────────────────

    @staticmethod
    def _result_to_source_dict(r: RetrievalResult, idx: int) -> dict:
        wp = r.metadata.get("white_paper", "Unknown")
        lang = r.metadata.get("language", "en")
        if lang == "fa" and wp in _WP_NAME_FA:
            wp = _WP_NAME_FA[wp]
        return {
            "white_paper": wp,
            "page_start": int(r.metadata.get("page_start", 0)),
            "page_end": int(r.metadata.get("page_end", 0)),
            "rerank_score": round(r.rerank_score, 3),
            "retrieval_source": r.source,
            "language": lang,
            "_passage_idx": idx,
            # Kept only for grounding verification in finalize(); stripped
            # before the dict is ever serialised to the client.
            "_text": r.text,
        }

    @staticmethod
    def _strip_idx(s: dict) -> dict:
        return {k: v for k, v in s.items() if not k.startswith("_")}

    @classmethod
    def _fallback_sources(cls, all_sources: list[dict]) -> list[dict]:
        """Top-3 by similarity score, preferring directly-retrieved (non-sibling)."""
        non_siblings = [
            s for s in all_sources if "+sibling" not in s["retrieval_source"]
        ]
        pool = non_siblings if non_siblings else all_sources
        top = sorted(pool, key=lambda s: s["rerank_score"], reverse=True)[:3]
        return [cls._strip_idx(s) for s in top]

    @staticmethod
    def _toc_source_dicts(lang: str) -> list[dict]:
        wp = "فهرست مطالب" if lang == "fa" else "Table of Contents"
        return [{
            "white_paper": wp,
            "page_start": TOC_PAGE.get(lang, 16),
            "page_end": TOC_PAGE.get(lang, 16),
            "rerank_score": 0.0,
            "retrieval_source": "toc_fallback",
            "language": lang,
        }]

    @staticmethod
    def _dicts_to_sources(dicts: list[dict]) -> list[Source]:
        return [
            Source(
                white_paper=d["white_paper"],
                page_start=d["page_start"],
                page_end=d["page_end"],
                text_preview=d.get("text_preview", ""),
                rerank_score=d["rerank_score"],
                retrieval_source=d["retrieval_source"],
                language=d.get("language", "en"),
            )
            for d in dicts
        ]

    # ── LLM calls (Gemini) ───────────────────────────────────────────

    def _call_llm(
        self, system_prompt: str, context: str, question: str, *, max_tokens: int = 512
    ) -> tuple[str, str]:
        user = (
            f"CONTEXT:\n{context}\n\nQUESTION:\n{question}" if context else question
        )
        text = _gemini_generate(
            system_prompt, user, max_tokens=max_tokens, model=ANSWER_MODEL
        )
        return (text or "(No response generated)"), ANSWER_MODEL

    def _call_llm_stream(
        self,
        system_prompt: str,
        context: str,
        question: str,
        *,
        provider_out: list[str] | None = None,
        max_tokens: int = 512,
    ) -> Generator[str, None, None]:
        if provider_out is not None:
            provider_out[:] = [ANSWER_MODEL]
        user = (
            f"CONTEXT:\n{context}\n\nQUESTION:\n{question}" if context else question
        )
        yield from _gemini_generate_stream(
            system_prompt, user, max_tokens=max_tokens, model=ANSWER_MODEL
        )
