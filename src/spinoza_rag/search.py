"""Search a persistent local vector index and print passages, not LLM answers."""

import argparse
import math
from pathlib import Path

import chromadb

from spinoza_rag.embeddings import (
    DEFAULT_MODEL,
    QUERY_INSTRUCTION,
    Embedder,
    OllamaClient,
)
from spinoza_rag.index import open_index


def search(
    question: str, index_dir: Path, embedder: Embedder, *, top_k: int = 5
) -> list[dict]:
    """Embed only the question and return nearest passages with source metadata."""
    if not question.strip() or top_k <= 0:
        raise ValueError("Question must not be empty and top_k must be positive.")
    with open_index(index_dir, embedder) as (collection, manifest):
        vectors = embedder.embed([f"Instruct: {QUERY_INSTRUCTION}\nQuery: {question}"])
        if (
            len(vectors) != 1
            or len(vectors[0]) != manifest["dimensions"]
            or not all(math.isfinite(value) for value in vectors[0])
            or not any(vectors[0])
        ):
            raise ValueError(
                "Query embedding dimensions/values do not match the index."
            )
        result = collection.query(
            query_embeddings=vectors,
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        return [
            {
                "chunk_id": identifier,
                "text": text,
                "metadata": metadata,
                "distance": distance,
            }
            for identifier, text, metadata, distance in zip(
                result["ids"][0],
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
                strict=True,
            )
        ]


def main() -> int:
    """Print retrieved passages and physical PDF page references."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--index-dir", type=Path, default=Path("storage/index"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
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
                f"\n[{rank}] {metadata['source_file']} | PDF pp. {metadata['page_start']}–{metadata['page_end']} | lecture: {metadata.get('lecture_date', 'unknown')} | cosine distance: {item['distance']:.4f}"
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
