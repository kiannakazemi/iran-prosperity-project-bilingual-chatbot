"""
Convert PDFs with Marker into structured JSON and paginated Markdown.

This script processes one or more PDF files, builds a Marker document for each,
then saves both a structured JSON representation and a paginated Markdown version
suitable for downstream chunking or retrieval workflows.

Defaults:
  Input:  document_preprocessing/iran_prosperity_project_pdf/
  Output: document_preprocessing/iran_prosperity_project_md/
  Logs:   document_preprocessing/logs/
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import IO, TextIO

os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from marker.config.parser import ConfigParser
from marker.converters.pdf import PdfConverter
from marker.logger import configure_logging, get_logger
from marker.models import create_model_dict
from marker.output import save_output
from marker.renderers.json import JSONRenderer
from marker.renderers.markdown import MarkdownRenderer

SCRIPT_DIR = Path(__file__).resolve().parent
PREPROCESSING_ROOT = SCRIPT_DIR

DEFAULT_DATA_DIR = PREPROCESSING_ROOT / "iran_prosperity_project_pdf"
DEFAULT_OUT_DIR = PREPROCESSING_ROOT / "iran_prosperity_project_md"
LOG_DIR = SCRIPT_DIR / "logs"
DEFAULT_LOG_FILE = "marker_iran_prosperity_run.log"


class _Tee(TextIO):
    """Mirror writes to multiple text streams (e.g. console + log file)."""

    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        n = len(data)
        for s in self._streams:
            s.write(data)
            s.flush()
        return n

    def flush(self) -> None:
        for s in self._streams:
            s.flush()

    def isatty(self) -> bool:
        return self._streams[0].isatty() if self._streams else False

    def writable(self) -> bool:
        return True

    def readable(self) -> bool:
        return False

    def seekable(self) -> bool:
        return False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Convert each PDF to Marker JSON (citations, bboxes) and paginated Markdown."
        ),
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Input folder (default: ./iran_prosperity_project_pdf)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output folder (default: ./iran_prosperity_project_md)",
    )
    p.add_argument(
        "--extract-images",
        action="store_true",
        help="Extract images into the output folder (Markdown pass only).",
    )
    p.add_argument(
        "--use-llm",
        action="store_true",
        help="Enable LLM enhancement (requires API keys per Marker docs).",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Marker debug mode (layout/debug artifacts).",
    )
    p.add_argument(
        "--page-range",
        type=str,
        default=None,
        help='Optional page range, e.g. "0,5-10" (marker_single --page_range).',
    )
    p.add_argument(
        "--pdf",
        type=Path,
        action="append",
        dest="only_pdfs",
        default=None,
        help="Process only this PDF (repeatable). Resolves against cwd or --data-dir.",
    )
    p.add_argument(
        "--log-file",
        type=str,
        default=DEFAULT_LOG_FILE,
        help=f"Log filename inside logs/ (default: {DEFAULT_LOG_FILE}).",
    )
    return p.parse_args(argv)


def _resolve_pdf_path(raw: Path, data_dir: Path) -> Path:
    raw = Path(raw)
    if raw.is_absolute():
        return raw.resolve()
    if raw.exists():
        return raw.resolve()
    return (data_dir / raw.name).resolve()


def _run(args: argparse.Namespace, log) -> int:
    data_dir = (
        args.data_dir.resolve()
        if args.data_dir.is_absolute()
        else (PREPROCESSING_ROOT / args.data_dir).resolve()
    )
    out_root = (
        args.out_dir.resolve()
        if args.out_dir.is_absolute()
        else (PREPROCESSING_ROOT / args.out_dir).resolve()
    )

    if args.only_pdfs:
        pdfs = []
        for raw in args.only_pdfs:
            pdf_path = _resolve_pdf_path(raw, data_dir)
            if not pdf_path.is_file():
                print(f"PDF not found: {raw} (resolved {pdf_path})", file=sys.stderr)
                return 1
            if pdf_path.suffix.lower() != ".pdf":
                print(f"Not a PDF file: {pdf_path}", file=sys.stderr)
                return 1
            pdfs.append(pdf_path)
        pdfs = sorted({p.resolve() for p in pdfs})
    else:
        pdfs = sorted(data_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {data_dir}", file=sys.stderr)
        return 1

    out_root.mkdir(parents=True, exist_ok=True)

    disable_images = not args.extract_images
    models = create_model_dict()
    total_start = time.time()

    print(f"Found {len(pdfs)} PDF(s)\nInput:  {data_dir}\nOutput: {out_root}\n")

    for i, pdf_path in enumerate(pdfs, 1):
        fpath = str(pdf_path.resolve())
        cli_kwargs: dict = {
            "output_dir": str(out_root),
            "output_format": "json",
            "paginate_output": True,
            "debug": args.debug,
            "use_llm": args.use_llm,
            "disable_image_extraction": disable_images,
        }
        if args.page_range:
            cli_kwargs["page_range"] = args.page_range

        config_parser = ConfigParser(cli_kwargs)
        config_dict = config_parser.generate_config_dict()
        converter_cls = config_parser.get_converter_cls()
        converter = converter_cls(
            config=config_dict,
            artifact_dict=models,
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
            llm_service=config_parser.get_llm_service(),
        )

        base = pdf_path.stem
        print(f"[{i}/{len(pdfs)}] {pdf_path.name} (JSON + paginated Markdown) ...")
        t0 = time.time()
        try:
            document = converter.build_document(fpath)
        except Exception as exc:
            print(f"        FAILED (build_document): {exc}\n", file=sys.stderr)
            log.exception("build_document failed for %s", fpath)
            continue

        json_renderer = converter.resolve_dependencies(JSONRenderer)
        md_renderer = converter.resolve_dependencies(MarkdownRenderer)

        try:
            json_rendered = json_renderer(document)
            md_rendered = md_renderer(document)
        except Exception as exc:
            print(f"        FAILED (render): {exc}\n", file=sys.stderr)
            log.exception("Rendering failed for %s", fpath)
            continue

        try:
            save_output(json_rendered, str(out_root), base)
            save_output(md_rendered, str(out_root), base)
        except Exception as exc:
            print(f"        FAILED (save): {exc}\n", file=sys.stderr)
            log.exception("save_output failed for %s", fpath)
            continue

        json_path = out_root / f"{base}.json"
        md_path = out_root / f"{base}.md"
        meta_path = out_root / f"{base}_meta.json"
        print(
            f"        -> {json_path.name}, {md_path.name}, {meta_path.name} "
            f"({time.time() - t0:.1f}s)\n"
        )

    print(f"Done in {time.time() - total_start:.1f}s total.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_name = Path(args.log_file).name.strip()
    if not log_name:
        log_name = DEFAULT_LOG_FILE
    log_path = (LOG_DIR / log_name).resolve()

    log_fp: IO[str]
    log_fp = open(log_path, "w", encoding="utf-8", newline="")
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(old_stdout, log_fp)
    sys.stderr = _Tee(old_stderr, log_fp)
    try:
        configure_logging()
        marker_log = get_logger()
        print(f"Log file: {log_path}\n")
        return _run(args, marker_log)
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        log_fp.close()


if __name__ == "__main__":
    raise SystemExit(main())
