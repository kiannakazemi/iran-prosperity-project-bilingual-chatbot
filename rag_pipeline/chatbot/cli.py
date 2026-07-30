"""
Run an interactive command-line interface for the bilingual IPP chatbot.

This module starts a local chat session, accepts user questions in English or
Persian, streams answers from the chat engine, and displays the retrieved
source passages associated with each response.

Usage:
    python -m chatbot.cli
"""

from __future__ import annotations

import io
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace"
    )

from chatbot.engine import ChatEngine


def main() -> None:
    print("=" * 60)
    print("  Iran Prosperity Project — Emergency Phase Chatbot")
    print("  Type your question in English or Persian.")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 60)
    print()
    print("Loading…")

    engine = ChatEngine()

    while True:
        print("-" * 60)
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        print()
        lang = engine._detect_language(question)
        print(f"Bot ({lang.upper()}): ", end="", flush=True)

        # One pipeline pass: ask_stream returns ChatResponse via StopIteration.value
        stream = engine.ask_stream(question)
        try:
            while True:
                print(next(stream), end="", flush=True)
        except StopIteration as ex:
            response = ex.value
        print("\n")
        if response.sources:
            print("Sources:")
            for s in response.sources:
                pages = f"p.{s.page_start}" if s.page_start == s.page_end else f"pp.{s.page_start}-{s.page_end}"
                score = f"{s.rerank_score:+.3f}"
                print(f"  - {s.white_paper} ({pages}, score: {score}, via: {s.retrieval_source})")
            print()


if __name__ == "__main__":
    main()
