"""
Embed validated chunks and store them in a local Qdrant vector database.

Loads chunk files from ``chunking/by_white_paper_validated/chunks/``, builds
LlamaIndex nodes, embeds with Cohere ``embed-multilingual-v3.0``, and writes
the index to ``embedding/qdrant_db/``. Runs an EN/FA sanity check at the end.

Usage:
    python -m indexing.embedding.embed_and_store
"""

from __future__ import annotations

import io
import re
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from dotenv import find_dotenv, load_dotenv
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.vector_stores.qdrant import QdrantVectorStore

from indexing.embedding.cohere_embed import COHERE_MODEL, make_cohere_embed_model
from qdrant_client import QdrantClient

# Load project-root .env so COHERE_API_KEY is available.
load_dotenv(find_dotenv(usecwd=True), override=False)

EMBEDDING_DIR = Path(__file__).resolve().parent
INDEXING_ROOT = EMBEDDING_DIR.parent
CHUNKS_DIR = INDEXING_ROOT / "chunking" / "by_white_paper_validated" / "chunks"
QDRANT_PATH = EMBEDDING_DIR / "qdrant_db"
COLLECTION_NAME = "emergency_phase_cohere_v3_validated"

EMBED_MODEL = COHERE_MODEL

TEXT_PATTERN = re.compile(r"^--- text \(\d+ chars\) ---$", re.MULTILINE)


def load_chunks(chunks_dir: Path = CHUNKS_DIR) -> list[TextNode]:
    """Read every chunk_*.txt and return a list of TextNode objects."""
    nodes: list[TextNode] = []

    chunk_files = sorted(
        chunks_dir.rglob("chunk_*.txt"),
        key=lambda p: (str(p.parent), _natural_sort_key(p.name)),
    )
    if not chunk_files:
        raise FileNotFoundError(f"No chunk_*.txt files found under {chunks_dir}")

    for path in chunk_files:
        raw = path.read_text(encoding="utf-8")
        metadata, text = _parse_chunk_file(raw)
        if not text.strip():
            continue
        nodes.append(TextNode(text=text, metadata=metadata))

    return nodes


def _natural_sort_key(name: str):
    """Sort chunk_2.txt before chunk_10.txt."""
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", name)]


def _parse_chunk_file(raw: str) -> tuple[dict, str]:
    """Split a chunk file into (metadata_dict, text_body)."""
    text_match = TEXT_PATTERN.search(raw)
    if not text_match:
        return {}, raw

    meta_block = raw[: text_match.start()].strip()
    text_body = raw[text_match.end() :].lstrip("\n")

    metadata: dict[str, str] = {}
    for line in meta_block.splitlines():
        if line.startswith("---"):
            continue
        key, _, value = line.partition(":")
        if key.strip():
            metadata[key.strip()] = value.strip()

    return metadata, text_body


def build_index(nodes: list[TextNode]) -> VectorStoreIndex:
    """Configure embeddings + Qdrant, then index all nodes."""

    print(f"  Loading embedding model ({EMBED_MODEL}) …")
    Settings.embed_model = make_cohere_embed_model(EMBED_MODEL)
    print("  Embedding model ready.")

    QDRANT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lock_file = QDRANT_PATH / ".lock"
    if lock_file.exists():
        try:
            lock_file.unlink()
        except OSError:
            pass
    client = QdrantClient(path=str(QDRANT_PATH))
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print(f"Embedding {len(nodes)} nodes (this may take a few minutes) …")
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        show_progress=True,
    )
    return index


def sanity_check(index: VectorStoreIndex) -> None:
    """Run one English and one Persian test query, print top-3 results."""
    from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters

    queries = [
        ("English", "en", "What is the transitional government structure?"),
        ("Persian", "fa", "ساختار دولت انتقالی چیست؟"),
    ]

    for label, lang_code, query in queries:
        lang_filter = MetadataFilters(
            filters=[MetadataFilter(key="language", value=lang_code)]
        )
        retriever = index.as_retriever(similarity_top_k=3, filters=lang_filter)

        print(f"\n{'='*60}")
        print(f"  Query ({label}): {query}")
        print(f"{'='*60}")
        results = retriever.retrieve(query)
        for i, r in enumerate(results, 1):
            lang = r.node.metadata.get("language", "?")
            score = f"{r.score:.4f}" if r.score is not None else "N/A"
            preview = r.node.text[:150].replace("\n", " ")
            print(f"  [{i}] lang={lang}  score={score}")
            print(f"      {preview}…")
        print()


def main() -> None:
    print(
        "\n  NOTE: If you changed embedding model or backend, delete the folder:\n"
        f"        {QDRANT_PATH}\n"
        "        then run this script again so vectors match the current embedder.\n"
    )
    print("Step 1: Loading chunks …")
    nodes = load_chunks()
    print(f"  Loaded {len(nodes)} nodes\n")

    en = sum(1 for n in nodes if n.metadata.get("language") == "en")
    fa = sum(1 for n in nodes if n.metadata.get("language") == "fa")
    print(f"  English: {en}  |  Persian: {fa}\n")

    print("Step 2-3: Embedding + storing in Qdrant …")
    print(f"  Model:      {EMBED_MODEL}")
    print(f"  Qdrant DB:  {QDRANT_PATH}")
    print(f"  Collection: {COLLECTION_NAME}\n")
    index = build_index(nodes)

    print("\nStep 4: Sanity-check retrieval …")
    sanity_check(index)

    print("Done.")


if __name__ == "__main__":
    main()
