"""Persist page-bounded, token-aware LangChain chunks from extracted JSONL."""

import argparse
import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import TypedDict

from langchain_text_splitters import RecursiveCharacterTextSplitter
from tokenizers import Tokenizer

from spinoza_rag.artifacts import (
    JsonObject,
    json_bytes,
    jsonl_bytes,
    publish_artifacts,
)
from spinoza_rag.config import CHUNKING, PATHS, ChunkConfig
from spinoza_rag.ingest import load_extraction


class ChunkFields(TypedDict):
    """The content and provenance of a chunk, which determine its ID."""

    schema_version: int
    document_id: str
    source_file: str
    page_start: int
    page_end: int
    page_chunk_index: int
    lecture_date: str | None
    lecture_id: str | None
    text: str
    token_count: int


class ChunkRow(ChunkFields):
    """One persisted chunk record in ``chunks.jsonl``."""

    chunk_id: str


class ChunkManifest(TypedDict):
    """The ``manifest.json`` record that ties chunks to their source pages."""

    schema_version: int
    source: JsonObject
    pages_sha256: str
    extraction: JsonObject
    extraction_dir: str
    chunking: JsonObject
    chunk_count: int
    chunks_sha256: str


def normalize_page(text: str, page_number: int) -> str:
    """Collapse layout whitespace and remove only a matching edge page number."""
    lines = text.strip().splitlines()
    if lines and lines[0].strip() == str(page_number):
        lines.pop(0)
    if lines and lines[-1].strip() == str(page_number):
        lines.pop()
    paragraphs = re.split(r"\n\s*\n", "\n".join(lines).strip())
    return "\n\n".join(" ".join(part.split()) for part in paragraphs if part.strip())


def _lecture_sections(
    text: str, page_number: int, previous_date: str | None
) -> tuple[list[tuple[str, str | None]], str | None]:
    top = " ".join(text.splitlines()[:12]).casefold()
    if (
        "table des matières" in " ".join(top.split())
        or len(re.findall(r"\.{4,}\s*\d+", text)) >= 3
    ):
        return [(normalize_page(text, page_number), None)], None
    sections = []
    cursor = 0
    current = previous_date
    for match in re.finditer(r"(?m)^[ \t]*(\d{2}\.\d{2}\.\d{4})[ \t]*$", text):
        try:
            date = datetime.strptime(match.group(1), "%d.%m.%Y").date().isoformat()
        except ValueError:
            continue
        before = normalize_page(text[cursor : match.start()], page_number)
        if before:
            sections.append((before, current))
        current, cursor = date, match.start()
    after = normalize_page(text[cursor:], page_number)
    if after:
        sections.append((after, current))
    return sections, current


def _chunk_id(row: Mapping[str, object], configuration: Mapping[str, object]) -> str:
    identity = {key: value for key, value in row.items() if key != "chunk_id"}
    return (
        "sha256:"
        + hashlib.sha256(
            json_bytes({"chunk": identity, "configuration": configuration})
        ).hexdigest()
    )


def load_chunks(output_dir: Path) -> tuple[list[ChunkRow], ChunkManifest]:
    """Verify persisted chunks and their upstream page artifacts without a PDF."""
    try:
        if output_dir.is_symlink() or {p.name for p in output_dir.iterdir()} != {
            "chunks.jsonl",
            "manifest.json",
        }:
            raise ValueError("expected exactly chunks.jsonl and manifest.json")
        if any(
            (output_dir / name).is_symlink()
            for name in ("chunks.jsonl", "manifest.json")
        ):
            raise ValueError("chunk artifacts must not be symlinks")
        manifest = json.loads((output_dir / "manifest.json").read_bytes())
        raw = (output_dir / "chunks.jsonl").read_bytes()
        if (
            manifest["schema_version"] != 1
            or hashlib.sha256(raw).hexdigest() != manifest["chunks_sha256"]
        ):
            raise ValueError("unsupported chunk schema or checksum mismatch")
        document, extraction = load_extraction(output_dir / manifest["extraction_dir"])
        if (
            manifest["source"] != extraction["document"]
            or manifest["pages_sha256"] != extraction["pages_jsonl_sha256"]
        ):
            raise ValueError(
                "source pages changed; regenerate chunks in a new directory"
            )
        rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
        if not rows or len(rows) != manifest["chunk_count"]:
            raise ValueError("missing or inconsistent chunk records")
        ids: set[str] = set()
        for row in rows:
            if (
                row["schema_version"] != 1
                or row["document_id"] != document.document_id
                or row["source_file"] != document.source_file
                or not 1 <= row["page_start"] == row["page_end"] <= len(document.pages)
                or not isinstance(row["text"], str)
                or not row["text"].strip()
                or row["chunk_id"] != _chunk_id(row, manifest["chunking"])
                or row["chunk_id"] in ids
            ):
                raise ValueError("invalid chunk identity, text, or page provenance")
            ids.add(row["chunk_id"])
        return rows, manifest
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid chunk artifacts in {output_dir}: {error}") from error


def chunk_pages(
    input_dir: Path,
    output_dir: Path,
    tokenizer_path: Path,
    config: ChunkConfig | None = None,
) -> tuple[list[ChunkRow], ChunkManifest]:
    """Create immutable chunks, or reuse matching verified artifacts.

    Only stored page records are read. No PDF parser, embedding model, or vector
    store is used. Failed extraction pages block chunking; empty pages are skipped.
    Lecture dates are inferred from standalone headings and carried across pages.
    """
    config = config or CHUNKING
    document, extraction = load_extraction(input_dir)
    if any(page.error for page in document.pages):
        raise ValueError(
            "Extraction has failed pages; review report.json before chunking."
        )
    tokenizer_data = tokenizer_path.read_bytes()
    configuration: JsonObject = {
        **asdict(config),
        "splitter": "RecursiveCharacterTextSplitter",
        "splitter_version": version("langchain-text-splitters"),
        "tokenizer_version": version("tokenizers"),
        "tokenizer_sha256": hashlib.sha256(tokenizer_data).hexdigest(),
        "page_boundaries": "hard",
        "normalization": "paragraph_whitespace_and_edge_page_number_v1",
        "lecture_detection": "standalone_date_sections_v1",
        "revision": 1,
    }
    if output_dir.exists() or output_dir.is_symlink():
        rows, manifest = load_chunks(output_dir)
        if (
            manifest["chunking"] != configuration
            or manifest["source"] != extraction["document"]
            or manifest["pages_sha256"] != extraction["pages_jsonl_sha256"]
        ):
            raise ValueError(
                "Existing chunks have different source/settings; "
                "choose a new --output-dir."
            )
        return rows, manifest
    tokenizer = Tokenizer.from_str(tokenizer_data.decode("utf-8"))

    def token_count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False).ids)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.max_tokens,
        chunk_overlap=config.overlap_tokens,
        length_function=token_count,
        separators=["\n\n", "\n", ". ", "; ", " ", ""],
        keep_separator="end",
    )
    rows = []
    lecture_date = None
    for page in document.pages:
        sections, lecture_date = _lecture_sections(
            page.text, page.page_number, lecture_date
        )
        ordinal = 0
        for text, section_date in sections:
            for content in splitter.split_text(text):
                fields: ChunkFields = {
                    "schema_version": 1,
                    "document_id": document.document_id,
                    "source_file": document.source_file,
                    "page_start": page.page_number,
                    "page_end": page.page_number,
                    "page_chunk_index": ordinal,
                    "lecture_date": section_date,
                    "lecture_id": f"{document.document_id}:lecture:{section_date}"
                    if section_date
                    else None,
                    "text": content,
                    "token_count": token_count(content),
                }
                rows.append({**fields, "chunk_id": _chunk_id(fields, configuration)})
                ordinal += 1
    if not rows:
        raise ValueError("No nonempty text is available for chunking.")
    raw = jsonl_bytes(rows)
    manifest = {
        "schema_version": 1,
        "source": extraction["document"],
        "pages_sha256": extraction["pages_jsonl_sha256"],
        "extraction": extraction["extraction"],
        "extraction_dir": os.path.relpath(input_dir.resolve(), output_dir.resolve()),
        "chunking": configuration,
        "chunk_count": len(rows),
        "chunks_sha256": hashlib.sha256(raw).hexdigest(),
    }
    publish_artifacts(
        output_dir, {"chunks.jsonl": raw, "manifest.json": json_bytes(manifest)}
    )
    return rows, manifest


def main(argv: Sequence[str] | None = None) -> int:
    """Persist chunks from existing page JSONL with a local tokenizer file."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PATHS.pages_dir,
        help="extracted page artifacts",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PATHS.chunks_dir,
        help="new or matching chunk directory",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=PATHS.tokenizer,
        help="local Qwen tokenizer.json",
    )
    parser.add_argument(
        "--chunk-tokens",
        type=int,
        default=CHUNKING.max_tokens,
        help="maximum tokens per chunk",
    )
    parser.add_argument(
        "--overlap-tokens",
        type=int,
        default=CHUNKING.overlap_tokens,
        help="overlap target in tokens",
    )
    args = parser.parse_args(argv)
    try:
        rows, _ = chunk_pages(
            args.input_dir,
            args.output_dir,
            args.tokenizer,
            ChunkConfig(args.chunk_tokens, args.overlap_tokens),
        )
        counts = sorted(row["token_count"] for row in rows)
        print(
            json.dumps(
                {
                    "chunks": len(rows),
                    "min_tokens": counts[0],
                    "median_tokens": counts[len(counts) // 2],
                    "max_tokens": counts[-1],
                    "output": str(args.output_dir),
                }
            )
        )
    except (OSError, ValueError, TypeError) as error:
        parser.exit(1, f"Chunking failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
