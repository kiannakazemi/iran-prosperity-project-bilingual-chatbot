"""
Split the booklet MD by white paper AND group chunks by white paper.

Produces a single folder ``by_white_paper/`` with two siblings:

    by_white_paper/
    ├── md/
    │   ├── english/
    │   │   ├── 00_front_matter.md
    │   │   ├── 01_legal.md
    │   │   ├── 02_political.md
    │   │   └── …  (14 white papers + Front Matter)
    │   └── persian/
    │       └── …  (mirrors English filenames; Persian content inside)
    └── chunks/
        ├── english/
        │   ├── 00_front_matter/
        │   │   ├── chunk_1.txt
        │   │   └── …
        │   └── …
        └── persian/
            └── …

White-paper filenames are English transliterations (snake_case) prefixed
with a two-digit booklet-order index (00 = Front Matter, 01–14 = the white
papers), so both language folders use identical names and pair at a glance.

Usage (standalone):
    python -m indexing.chunking.organize_by_white_paper

The validator (validate_chunks_agentic.py) calls ``organize()`` at startup
so the folder is always fresh before validation begins.
"""

from __future__ import annotations

import io
import re
import shutil
import sys
from pathlib import Path

# Make rag_pipeline/ importable AND this folder importable, so 'chunking'
# resolves whether this file is run as a script or imported from a sibling.
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[1]
for _p in (str(_REPO_ROOT), str(_THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from chunking import _ALL_TOPICS, _EN_TOPICS, _FA_TOPICS  # noqa: E402

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


# ── Paths ─────────────────────────────────────────────────────────────
CHUNKING_PKG_DIR = Path(__file__).resolve().parent
INDEXING_ROOT = CHUNKING_PKG_DIR.parent
REPO_ROOT = INDEXING_ROOT.parent
MD_DIR = REPO_ROOT / "document_preprocessing" / "iran_prosperity_project_md"
CHUNKS_RAW = CHUNKING_PKG_DIR / "chunks_raw"
BY_WP_DIR = CHUNKING_PKG_DIR / "by_white_paper"

LANG_DIRS = {
    "en": ("emergency_phase_english", "Emergency_Phase_ENGLISH_20260301_1440"),
    "fa": ("emergency_phase_persian", "Emergency_Phase_PERSIAN_20260301_1440"),
}


# ── Canonical filename order (00 = Front Matter, 01..14 = white papers) ──
# Maps the canonical EN white paper string → (order_index, snake_case_slug).
# Persian uses the same numeric order via _FA_TOPICS sharing the same index.
_ORDER: list[tuple[str, str]] = [
    ("Front Matter",                  "front_matter"),
    ("LEGAL",                          "legal"),
    ("POLITICAL",                      "political"),
    ("MILITARY AND SECURITY",          "military_and_security"),
    ("FOREIGN POLICY",                 "foreign_policy"),
    ("GOVERNMENT ESSENTIAL FUNCTIONS", "government_essential_functions"),
    ("MACROECONOMIC GOVERNANCE",       "macroeconomic_governance"),
    ("NATIONAL ASSETS",                "national_assets"),
    ("ENERGY",                         "energy"),
    ("INDUSTRY",                       "industry"),
    ("CYBERSECURITY",                  "cybersecurity"),
    ("ENVIRONMENT",                    "environment"),
    ("WATER",                          "water"),
    ("HEALTHCARE",                     "healthcare"),
    ("EDUCATIONAL SYSTEM",             "educational_system"),
]

# Map each EN white-paper name to (index, slug).
_EN_NAME_TO_SLUG: dict[str, tuple[int, str]] = {
    name: (i, slug) for i, (name, slug) in enumerate(_ORDER)
}
# Map each FA white-paper name to the same (index, slug) so Persian files
# share the English slug for cross-language pairing.
# _FA_TOPICS is parallel to _EN_TOPICS, so we can zip them with offset 1
# (Front Matter is index 0 and has no FA counterpart in the list itself).
_FA_NAME_TO_SLUG: dict[str, tuple[int, str]] = {"Front Matter": (0, "front_matter")}
for i, fa_name in enumerate(_FA_TOPICS, start=1):
    if i < len(_ORDER):
        _FA_NAME_TO_SLUG[fa_name] = (i, _ORDER[i][1])


# ── MD splitting ─────────────────────────────────────────────────────

_HEADING_RE = re.compile(r"^#{1,2}\s+(?:\*\*)?(.+?)(?:\*\*)?$", re.MULTILINE)


def _build_wp_offsets(raw_md: str) -> list[tuple[int, int, str]]:
    """Find white-paper headings in RAW md. Returns [(start, end, name)]."""
    entries: list[tuple[int, str]] = []
    for m in _HEADING_RE.finditer(raw_md):
        full = m.group(1).strip().strip("*").strip()
        if ":" not in full:
            continue
        topic = full.split(":")[0].strip().strip("*").strip()
        if topic in _ALL_TOPICS:
            entries.append((m.start(), topic))
    out: list[tuple[int, int, str]] = []
    for i, (pos, name) in enumerate(entries):
        end = entries[i + 1][0] if i + 1 < len(entries) else len(raw_md)
        out.append((pos, end, name))
    return out


def _name_to_filename(language: str, white_paper: str) -> str | None:
    """Return '01_legal.md' style filename for a given (language, white_paper).
    Returns None if the name isn't in the canonical order."""
    table = _EN_NAME_TO_SLUG if language == "en" else _FA_NAME_TO_SLUG
    if white_paper not in table:
        return None
    idx, slug = table[white_paper]
    return f"{idx:02d}_{slug}.md"


def _slug_for(language: str, white_paper: str) -> str | None:
    """Return '01_legal' style folder name (no extension)."""
    fname = _name_to_filename(language, white_paper)
    return fname[:-3] if fname else None


# Trailing page marker (e.g. "{18}----") sitting right before a heading,
# separated only by whitespace. The marker denotes the start of the PDF page
# on which the NEXT white paper begins, so it must head the next file.
_TRAILING_PAGE_MARKER_RE = re.compile(r"\{\d+\}-{10,}\s*\Z")


def _shift_marker_back(raw: str, pos: int) -> int:
    """Move a white-paper boundary back so a page marker immediately preceding
    the heading belongs to the NEXT slice (the new white paper) rather than the
    previous one. Returns the adjusted boundary, or ``pos`` unchanged when no
    such trailing marker exists."""
    m = _TRAILING_PAGE_MARKER_RE.search(raw[:pos])
    return m.start() if m else pos


# ── MD splitter ──────────────────────────────────────────────────────

def split_md_by_white_paper() -> dict[str, int]:
    """Write per-white-paper MD slices to by_white_paper/md/<lang>/."""
    md_out = BY_WP_DIR / "md"
    if md_out.exists():
        shutil.rmtree(md_out)

    counts: dict[str, int] = {}
    for lang, (_folder, stem) in LANG_DIRS.items():
        md_path = MD_DIR / f"{stem}.md"
        if not md_path.exists():
            print(f"  [warn] missing MD: {md_path}")
            continue
        raw = md_path.read_text(encoding="utf-8")
        lang_out = md_out / ("english" if lang == "en" else "persian")
        lang_out.mkdir(parents=True, exist_ok=True)

        offsets = _build_wp_offsets(raw)

        # Adjust each white-paper start backwards so a page marker that sits
        # right before the heading moves into that white paper's own file.
        # Each slice ends where the next one begins, so the marker naturally
        # leaves the previous file and heads the next.
        starts = [_shift_marker_back(raw, s) for (s, _e, _n) in offsets]

        # Front Matter = everything before the first (adjusted) white paper start.
        front_end = starts[0] if starts else len(raw)
        fm_text = raw[:front_end]
        fm_name = _name_to_filename(lang, "Front Matter")
        if fm_name:
            (lang_out / fm_name).write_text(fm_text, encoding="utf-8")
            counts[f"{lang}/Front Matter"] = len(fm_text)

        # Each white paper: from its adjusted start to the next adjusted start.
        for i, (_s, _e, name) in enumerate(offsets):
            fname = _name_to_filename(lang, name)
            if not fname:
                print(f"  [warn] {lang}: unmapped white paper '{name}' — skipped")
                continue
            seg_start = starts[i]
            seg_end = starts[i + 1] if i + 1 < len(starts) else len(raw)
            (lang_out / fname).write_text(raw[seg_start:seg_end], encoding="utf-8")
            counts[f"{lang}/{name}"] = seg_end - seg_start

    return counts


# ── Chunk grouping ───────────────────────────────────────────────────

_META_TEXT_RE = re.compile(r"^--- text \(\d+ chars\) ---$", re.MULTILINE)


def _extract_white_paper_from_chunk(chunk_text: str) -> str:
    m = re.search(r"^white_paper:\s*(.+)$", chunk_text, re.MULTILINE)
    return m.group(1).strip() if m else ""


def group_chunks_by_white_paper() -> dict[str, int]:
    """Copy chunks from chunks_raw/ into by_white_paper/chunks/<lang>/<wp>/."""
    chunks_out = BY_WP_DIR / "chunks"
    if chunks_out.exists():
        shutil.rmtree(chunks_out)

    counts: dict[str, int] = {}
    for lang, (folder, stem) in LANG_DIRS.items():
        src_dir = CHUNKS_RAW / folder / stem
        if not src_dir.is_dir():
            print(f"  [warn] no chunks dir: {src_dir}")
            continue
        for chunk_path in sorted(
            src_dir.glob("chunk_*.txt"),
            key=lambda p: int(re.search(r"\d+", p.name).group()),
        ):
            content = chunk_path.read_text(encoding="utf-8")
            wp = _extract_white_paper_from_chunk(content)
            slug = _slug_for(lang, wp)
            if slug is None:
                print(
                    f"  [warn] {lang}: chunk {chunk_path.name} has unmapped "
                    f"white_paper '{wp}' — skipped"
                )
                continue
            dest_dir = chunks_out / ("english" if lang == "en" else "persian") / slug
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(chunk_path, dest_dir / chunk_path.name)
            counts[f"{lang}/{wp}"] = counts.get(f"{lang}/{wp}", 0) + 1
    return counts


# ── Top-level entry ─────────────────────────────────────────────────

def organize() -> None:
    """Run both steps. Used standalone and from validate_chunks_agentic.py."""
    print(f"Organizing by white paper → {BY_WP_DIR}")
    BY_WP_DIR.mkdir(parents=True, exist_ok=True)

    print("\nSplitting MD files by white paper …")
    md_counts = split_md_by_white_paper()
    for key, size in sorted(md_counts.items()):
        print(f"  {key:<55}  {size:>7,} chars")

    print("\nGrouping chunks by white paper …")
    chunk_counts = group_chunks_by_white_paper()
    for key, n in sorted(chunk_counts.items()):
        print(f"  {key:<55}  {n:>4} chunks")

    print(f"\nDone. See {BY_WP_DIR}/md/ and {BY_WP_DIR}/chunks/.")


if __name__ == "__main__":
    organize()
