"""Search a persistent local vector index and print passages, not LLM answers."""

import argparse
import math
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

import chromadb
from chromadb.api.types import Metadata, PyEmbeddings

from spinoza_rag.config import EMBEDDING, OLLAMA, PATHS, SEARCH
from spinoza_rag.embeddings import Embedder, OllamaClient
from spinoza_rag.index import open_index


class SearchResult(TypedDict):
    """A retrieved passage with its PDF provenance and cosine distance."""

    chunk_id: str
    text: str
    metadata: Metadata
    distance: float


def search(
    question: str,
    index_dir: Path,
    embedder: Embedder,
    *,
    top_k: int = SEARCH.top_k,
) -> list[SearchResult]:
    """Embed only the question and return nearest passages with source metadata."""
    if not question.strip() or top_k <= 0:
        raise ValueError("Question must not be empty and top_k must be positive.")
    with open_index(index_dir, embedder) as (collection, manifest):
        # Qwen3 embedding models expect this instruction format for queries;
        # documents are embedded without it.
        vectors = embedder.embed(
            [f"Instruct: {EMBEDDING.query_instruction}\nQuery: {question}"]
        )
        if (
            len(vectors) != 1
            or len(vectors[0]) != manifest["dimensions"]
            or not all(math.isfinite(value) for value in vectors[0])
            or not any(vectors[0])
        ):
            raise ValueError(
                "Query embedding dimensions/values do not match the index."
            )
        # Lists are invariant, so copy into the element type Chroma declares.
        query: PyEmbeddings = list(vectors)
        result = collection.query(
            query_embeddings=query,
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        documents = result["documents"]
        metadatas = result["metadatas"]
        distances = result["distances"]
        if documents is None or metadatas is None or distances is None:
            raise ValueError("Chroma returned no documents, metadata, or distances.")
        return [
            {
                "chunk_id": identifier,
                "text": text,
                "metadata": metadata,
                "distance": distance,
            }
            for identifier, text, metadata, distance in zip(
                result["ids"][0],
                documents[0],
                metadatas[0],
                distances[0],
                strict=True,
            )
        ]


def main(argv: Sequence[str] | None = None) -> int:
    """Print retrieved passages and physical PDF page references."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("question", help="question to embed and search for")
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=PATHS.index_dir,
        help="complete persisted index",
    )
    parser.add_argument(
        "--top-k", type=int, default=SEARCH.top_k, help="number of passages to return"
    )
    parser.add_argument(
        "--model", default=EMBEDDING.model, help="installed Ollama embedding model"
    )
    parser.add_argument("--ollama-url", default=OLLAMA.url, help="loopback Ollama URL")
    parser.add_argument(
        "--timeout",
        type=float,
        default=OLLAMA.timeout_seconds,
        help="seconds per Ollama request",
    )
    args = parser.parse_args(argv)
    try:
        results = search(
            args.question,
            args.index_dir,
            OllamaClient(args.model, args.ollama_url, args.timeout),
            top_k=args.top_k,
        )
        for rank, item in enumerate(results, start=1):
            metadata = item["metadata"]
            print(
                f"\n[{rank}] {metadata['source_file']} | "
                f"PDF pp. {metadata['page_start']}–{metadata['page_end']} | "
                f"lecture: {metadata.get('lecture_date', 'unknown')} | "
                f"cosine distance: {item['distance']:.4f}"
            )
            print(f"Chunk: {item['chunk_id']}\n{item['text']}")
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        chromadb.errors.ChromaError,
    ) as error:
        parser.exit(1, f"Search failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
