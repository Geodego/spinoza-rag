import json
import subprocess
import sys

import pytest

from spinoza_rag import chunk, index, ingest, search
from spinoza_rag.config import (
    CHUNKING,
    EMBEDDING,
    EXTRACTION,
    INDEX,
    OLLAMA,
    PATHS,
    SEARCH,
    ChunkConfig,
)


def test_ingest_cli_defaults_come_from_config(monkeypatch):
    seen = {}

    def fake_load(source, output_dir, min_page_characters, extraction_mode):
        seen.update(
            source=source,
            output_dir=output_dir,
            min_page_characters=min_page_characters,
            extraction_mode=extraction_mode,
        )
        return {"summary": {"failed_pages": 0, "pages_with_text": 1}}

    monkeypatch.setattr(ingest, "_load_existing_extraction", fake_load)
    assert ingest.main([]) == 0
    assert seen == {
        "source": PATHS.source_pdf,
        "output_dir": PATHS.pages_dir,
        "min_page_characters": EXTRACTION.min_page_characters,
        "extraction_mode": EXTRACTION.mode,
    }


def test_chunk_cli_defaults_come_from_config(monkeypatch):
    seen = {}

    def fake_chunk_pages(input_dir, output_dir, tokenizer_path, config):
        seen.update(
            input_dir=input_dir,
            output_dir=output_dir,
            tokenizer=tokenizer_path,
            config=config,
        )
        return [{"token_count": 1}], {}

    monkeypatch.setattr(chunk, "chunk_pages", fake_chunk_pages)
    assert chunk.main([]) == 0
    assert seen == {
        "input_dir": PATHS.pages_dir,
        "output_dir": PATHS.chunks_dir,
        "tokenizer": PATHS.tokenizer,
        "config": CHUNKING,
    }


def test_index_cli_defaults_come_from_config(monkeypatch):
    seen = {}

    def fake_index_chunks(chunks_dir, index_dir, embedder, **options):
        seen.update(
            chunks_dir=chunks_dir,
            index_dir=index_dir,
            model=embedder.model,
            url=embedder.url,
            timeout=embedder.timeout,
            **options,
        )
        return {}

    monkeypatch.setattr(index, "index_chunks", fake_index_chunks)
    assert index.main([]) == 0
    assert seen == {
        "chunks_dir": PATHS.chunks_dir,
        "index_dir": PATHS.index_dir,
        "model": EMBEDDING.model,
        "url": OLLAMA.url,
        "timeout": OLLAMA.timeout_seconds,
        "collection_name": INDEX.collection,
        "batch_size": INDEX.batch_size,
        "progress": seen["progress"],
    }


def test_search_cli_defaults_come_from_config(monkeypatch):
    seen = {}

    def fake_search(question, index_dir, embedder, *, top_k):
        seen.update(
            question=question,
            index_dir=index_dir,
            model=embedder.model,
            url=embedder.url,
            timeout=embedder.timeout,
            top_k=top_k,
        )
        return []

    monkeypatch.setattr(search, "search", fake_search)
    assert search.main(["joie"]) == 0
    assert seen == {
        "question": "joie",
        "index_dir": PATHS.index_dir,
        "model": EMBEDDING.model,
        "url": OLLAMA.url,
        "timeout": OLLAMA.timeout_seconds,
        "top_k": SEARCH.top_k,
    }


def test_chunk_config_rejects_invalid_limits():
    with pytest.raises(ValueError, match="max_tokens"):
        ChunkConfig(10, 10)


def test_config_module_prints_all_defaults():
    result = subprocess.run(
        [sys.executable, "-I", "-m", "spinoza_rag.config"],
        text=True,
        capture_output=True,
        check=True,
    )
    printed = json.loads(result.stdout)
    assert set(printed) == {
        "paths",
        "extraction",
        "chunking",
        "ollama",
        "embedding",
        "index",
        "search",
    }
    assert printed["paths"]["pages_dir"] == str(PATHS.pages_dir)
    assert printed["chunking"]["max_tokens"] == CHUNKING.max_tokens
    assert printed["embedding"]["model"] == EMBEDDING.model
