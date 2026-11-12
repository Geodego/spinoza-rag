"""Shared synthetic fixtures; no test here needs Ollama, models, or the corpus."""

import hashlib
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from spinoza_rag.artifacts import json_bytes
from spinoza_rag.chunk import chunk_pages
from spinoza_rag.config import ChunkConfig
from spinoza_rag.ingest import ExtractedDocument, ExtractedPage, write_extraction


class FakeEmbedder:
    """Deterministic two-dimensional embedder that records every request."""

    def __init__(self, digest="fake-weights-v1", fail_after=None, offline=False):
        self.digest = digest
        self.fail_after = fail_after
        self.offline = offline
        self.calls = []

    def identity(self):
        if self.offline:
            raise ValueError("Ollama is unavailable")
        return {"model": "fake", "digest": self.digest, "ollama_version": "test"}

    def embed(self, texts):
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise ValueError("Synthetic unavailable embedding server")
        self.calls.append(list(texts))
        return [
            [1.0, 0.0] if "joie" in text.casefold() else [0.0, 1.0] for text in texts
        ]


@pytest.fixture
def make_embedder():
    """Create fake embedders; options set the digest, failures, or offline mode."""
    return FakeEmbedder


@pytest.fixture
def snapshot():
    """Capture a directory's file bytes, optionally with modification times.

    Comparing times as well detects files rewritten with identical bytes, which
    a reuse path must not do.
    """

    def capture(directory, *, with_mtime=False):
        return {
            path.name: (path.read_bytes(), path.stat().st_mtime_ns)
            if with_mtime
            else path.read_bytes()
            for path in directory.iterdir()
        }

    return capture


@pytest.fixture
def forbid_pdf_parsing(monkeypatch):
    """Return a switch that makes any later PDF parsing fail the test."""

    def forbidden(*args, **kwargs):
        pytest.fail("PDF parsing must not happen at this point")

    def activate():
        monkeypatch.setattr("spinoza_rag.ingest.PdfReader", forbidden)

    return activate


@pytest.fixture
def pipeline(tmp_path):
    """Build page and chunk artifacts from synthetic page texts.

    Pages are written directly as extraction artifacts, so no PDF is involved.
    A whitespace word-level tokenizer stands in for the Qwen tokenizer.
    """
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    def build(texts, *, name="default", config=None, failed=False):
        root = tmp_path / name
        input_dir, output_dir = root / "pages", root / "chunks"
        source = json_bytes(texts)
        document = ExtractedDocument(
            document_id="sha256:" + hashlib.sha256(source).hexdigest(),
            source_file="synthetic.pdf",
            source_size_bytes=len(source),
            metadata={"Title": "Synthetic lectures"},
            pages=tuple(
                ExtractedPage(
                    i,
                    text,
                    ("empty_text",) if not text else (),
                    (),
                    "synthetic failure" if failed else None,
                )
                for i, text in enumerate(texts, start=1)
            ),
            parser_warnings=(),
            min_page_characters=0,
            extraction_mode="plain",
        )
        write_extraction(document, input_dir)
        rows, manifest = chunk_pages(
            input_dir, output_dir, tokenizer_path, config or ChunkConfig()
        )
        return SimpleNamespace(
            root=root,
            input=input_dir,
            chunks=output_dir,
            index=root / "index",
            rows=rows,
            manifest=manifest,
            tokenizer=tokenizer_path,
        )

    return build
