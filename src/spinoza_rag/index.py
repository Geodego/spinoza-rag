"""Resume or reuse a persistent local Chroma index of saved chunks."""

import argparse
import hashlib
import json
import math
import os
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings

from spinoza_rag.artifacts import json_bytes, save_json
from spinoza_rag.chunk import load_chunks
from spinoza_rag.config import EMBEDDING, INDEX, OLLAMA, PATHS
from spinoza_rag.embeddings import Embedder, OllamaClient


def _load_manifest(path: Path) -> dict:
    try:
        if path.is_symlink():
            raise ValueError("manifest must not be a symlink")
        value = json.loads(path.read_bytes())
        if (
            not isinstance(value, dict)
            or not isinstance(value["configuration"], dict)
            or value["state"] not in ("building", "ready")
            or not isinstance(value["chunks_dir"], str)
            or type(value["chunk_count"]) is not int
            or value["chunk_count"] <= 0
            or not isinstance(value["indexing_seconds"], (float, int))
            or not math.isfinite(value["indexing_seconds"])
            or value["indexing_seconds"] < 0
        ):
            raise ValueError("invalid index checkpoint fields")
        dimensions = value["dimensions"]
        if dimensions is not None and (type(dimensions) is not int or dimensions <= 0):
            raise ValueError("invalid embedding dimensions")
        if value["state"] == "ready" and dimensions is None:
            raise ValueError("ready index has no embedding dimensions")
        return value
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid index manifest in {path.parent}: {error}") from error


def _stored_dimensions(collection: Collection) -> int:
    return len(collection.get(limit=1, include=["embeddings"])["embeddings"][0])


def _configuration(manifest: dict, identity: dict, collection_name: str) -> dict:
    return {
        "schema_version": 1,
        "chunks_manifest_sha256": hashlib.sha256(json_bytes(manifest)).hexdigest(),
        "embedding": identity,
        "collection": collection_name,
        "distance": "cosine",
        "query_instruction": EMBEDDING.query_instruction,
    }


def _metadata(row: dict, source: dict) -> dict:
    metadata = {
        key: row[key]
        for key in (
            "document_id",
            "source_file",
            "page_start",
            "page_end",
            "page_chunk_index",
            "token_count",
            "lecture_date",
            "lecture_id",
        )
        if row[key] is not None
    }
    for key in ("Title", "Author"):
        if source["metadata"].get(key):
            metadata[key.lower()] = source["metadata"][key]
    return metadata


def _client(path: Path):
    return chromadb.PersistentClient(
        path=str(path), settings=Settings(anonymized_telemetry=False)
    )


def _verify_collection(
    collection, rows: list[dict], source: dict, fingerprint: str, *, complete: bool
) -> set[str]:
    if (
        collection.metadata != {"pipeline_fingerprint": fingerprint}
        or (collection.configuration_json.get("hnsw") or {}).get("space") != "cosine"
    ):
        raise ValueError(
            "Incompatible Chroma collection metadata or distance metric; choose a new --index-dir."
        )
    expected = {row["chunk_id"]: row for row in rows}
    saved = collection.get(include=["documents", "metadatas"])
    for identifier, text, metadata in zip(
        saved["ids"], saved["documents"], saved["metadatas"], strict=True
    ):
        if (
            identifier not in expected
            or text != expected[identifier]["text"]
            or metadata != _metadata(expected[identifier], source)
        ):
            raise ValueError(
                "Chroma contains unexpected or modified records; choose a new --index-dir."
            )
    present = set(saved["ids"])
    if complete and present != set(expected):
        raise ValueError(
            "Index is incomplete; rerun indexing to resume missing records."
        )
    return present


def index_chunks(
    chunks_dir: Path,
    index_dir: Path,
    embedder: Embedder,
    *,
    collection_name: str = INDEX.collection,
    batch_size: int = INDEX.batch_size,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Index missing chunk IDs, checkpoint each batch, and never re-embed saved IDs.

    A pipeline fingerprint prevents mixing sources, tokenizers, settings, or
    model digests. Interrupted runs resume only compatible missing records.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    rows, chunks_manifest = load_chunks(chunks_dir)
    identity = embedder.identity()
    configuration = _configuration(chunks_manifest, identity, collection_name)
    fingerprint = hashlib.sha256(json_bytes(configuration)).hexdigest()
    manifest_path = index_dir / "manifest.json"
    if index_dir.is_symlink():
        raise ValueError("Index directory must not be a symlink.")
    if manifest_path.exists():
        manifest = _load_manifest(manifest_path)
        if manifest.get("configuration") != configuration:
            raise ValueError(
                "Stale or incompatible index configuration; choose a new --index-dir."
            )
        if (index_dir / manifest["chunks_dir"]).resolve() != chunks_dir.resolve():
            raise ValueError("Index references a different chunks directory.")
    else:
        if index_dir.exists() and any(index_dir.iterdir()):
            raise ValueError(
                "Index directory contains unrecognized data; choose a new --index-dir."
            )
        index_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "configuration": configuration,
            "chunks_dir": os.path.relpath(chunks_dir.resolve(), index_dir.resolve()),
            "state": "building",
            "dimensions": None,
            "chunk_count": len(rows),
            "indexing_seconds": 0.0,
        }
        save_json(manifest_path, manifest)
    if manifest["chunk_count"] != len(rows):
        raise ValueError("Index checkpoint chunk count is inconsistent.")
    client = _client(index_dir / "chroma")
    try:
        collection = client.get_or_create_collection(
            collection_name,
            embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}},
            metadata={"pipeline_fingerprint": fingerprint},
        )
        present = _verify_collection(
            collection, rows, chunks_manifest["source"], fingerprint, complete=False
        )
        if present:
            saved_dimensions = _stored_dimensions(collection)
            if manifest["dimensions"] not in (None, saved_dimensions):
                raise ValueError(
                    "Index dimension checkpoint does not match stored vectors."
                )
            manifest["dimensions"] = saved_dimensions
        missing = [row for row in rows if row["chunk_id"] not in present]
        started = time.perf_counter()
        for offset in range(0, len(missing), batch_size):
            batch = missing[offset : offset + batch_size]
            vectors = embedder.embed([row["text"] for row in batch])
            if len(vectors) != len(batch) or not vectors:
                raise ValueError(
                    "Embedding count does not match the batch; index remains resumable."
                )
            dimensions = manifest["dimensions"] or len(vectors[0])
            if dimensions <= 0 or any(
                len(vector) != dimensions
                or not all(math.isfinite(value) for value in vector)
                or not any(vector)
                for vector in vectors
            ):
                raise ValueError(
                    "Invalid or inconsistent embedding dimensions; index remains resumable."
                )
            manifest["dimensions"] = dimensions
            collection.upsert(
                ids=[row["chunk_id"] for row in batch],
                documents=[row["text"] for row in batch],
                metadatas=[_metadata(row, chunks_manifest["source"]) for row in batch],
                embeddings=vectors,
            )
            manifest["indexing_seconds"] += time.perf_counter() - started
            save_json(manifest_path, manifest)
            started = time.perf_counter()
            if progress:
                progress(
                    f"Indexed {len(present) + min(offset + batch_size, len(missing))}/{len(rows)} chunks"
                )
        _verify_collection(
            collection, rows, chunks_manifest["source"], fingerprint, complete=True
        )
        manifest["state"] = "ready"
        save_json(manifest_path, manifest)
        return {
            "chunks": len(rows),
            "embedded_this_run": len(missing),
            "reused": len(present),
            "dimensions": manifest["dimensions"],
            "indexing_seconds": manifest["indexing_seconds"],
        }
    finally:
        client.close()


@contextmanager
def open_index(
    index_dir: Path, embedder: Embedder
) -> Iterator[tuple[Collection, dict]]:
    """Open a complete, compatible existing index without embedding documents."""
    manifest_path = index_dir / "manifest.json"
    if (
        index_dir.is_symlink()
        or not manifest_path.is_file()
        or not (index_dir / "chroma/chroma.sqlite3").is_file()
    ):
        raise ValueError("Persistent index is missing; run indexing first.")
    manifest = _load_manifest(manifest_path)
    if manifest.get("state") != "ready":
        raise ValueError("Index is incomplete; rerun indexing before search.")
    rows, chunks_manifest = load_chunks(index_dir / manifest["chunks_dir"])
    configuration = _configuration(
        chunks_manifest, embedder.identity(), manifest["configuration"]["collection"]
    )
    if manifest["configuration"] != configuration:
        raise ValueError(
            "Stale or incompatible index/model configuration; rebuild in a new --index-dir."
        )
    client = _client(index_dir / "chroma")
    try:
        collection = client.get_collection(
            configuration["collection"], embedding_function=None
        )
        _verify_collection(
            collection,
            rows,
            chunks_manifest["source"],
            hashlib.sha256(json_bytes(configuration)).hexdigest(),
            complete=True,
        )
        if (
            manifest["chunk_count"] != len(rows)
            or _stored_dimensions(collection) != manifest["dimensions"]
        ):
            raise ValueError(
                "Index checkpoint count or vector dimensions are inconsistent."
            )
        yield collection, manifest
    finally:
        client.close()


def main(argv: Sequence[str] | None = None) -> int:
    """Build or resume an index from saved chunks using local Ollama embeddings."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=PATHS.chunks_dir,
        help="persisted chunk artifacts",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=PATHS.index_dir,
        help="new or resumable index directory",
    )
    parser.add_argument(
        "--collection", default=INDEX.collection, help="Chroma collection name"
    )
    parser.add_argument(
        "--model", default=EMBEDDING.model, help="installed Ollama embedding model"
    )
    parser.add_argument("--ollama-url", default=OLLAMA.url, help="loopback Ollama URL")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=INDEX.batch_size,
        help="chunks per embedding request",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=OLLAMA.timeout_seconds,
        help="seconds per Ollama request",
    )
    args = parser.parse_args(argv)
    started = time.perf_counter()
    try:
        result = index_chunks(
            args.chunks_dir,
            args.index_dir,
            OllamaClient(args.model, args.ollama_url, args.timeout),
            collection_name=args.collection,
            batch_size=args.batch_size,
            progress=lambda message: print(message, flush=True),
        )
        result["wall_seconds"] = time.perf_counter() - started
        print(json.dumps(result, sort_keys=True))
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        chromadb.errors.ChromaError,
    ) as error:
        parser.exit(1, f"Indexing failed: {error}\n")
    except KeyboardInterrupt:
        parser.exit(
            130, "Indexing interrupted; completed batches are saved. Rerun to resume.\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
