import hashlib
import io
import json
import subprocess
import sys
from types import SimpleNamespace
from urllib.error import URLError

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from spinoza_rag.artifacts import json_bytes
from spinoza_rag.chunk import chunk_pages, load_chunks, normalize_page
from spinoza_rag.config import EMBEDDING, ChunkConfig
from spinoza_rag.embeddings import OllamaClient
from spinoza_rag.index import index_chunks, open_index
from spinoza_rag.ingest import ExtractedDocument, ExtractedPage, write_extraction
from spinoza_rag.search import search


class FakeEmbedder:
    def __init__(self, digest="fake-weights-v1", fail_after=None):
        self.digest = digest
        self.calls = []
        self.fail_after = fail_after

    def identity(self):
        return {"model": "fake", "digest": self.digest, "ollama_version": "test"}

    def embed(self, texts):
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise ValueError("Synthetic unavailable embedding server")
        self.calls.append(list(texts))
        return [
            [1.0, 0.0] if "joie" in text.casefold() else [0.0, 1.0] for text in texts
        ]


@pytest.fixture
def pipeline(tmp_path):
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


def test_page_chunks_are_stable_bounded_and_do_not_read_pdf(pipeline, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("PDF parsing is forbidden during chunking")

    monkeypatch.setattr("spinoza_rag.ingest.PdfReader", forbidden)
    data = pipeline(
        [
            "02.12.1980\n\n" + "Une joie partagée augmente la puissance. " * 20,
            "La tristesse réduit la puissance. " * 20,
        ],
        config=ChunkConfig(35, 7),
    )
    assert len(data.rows) > 2
    assert all(1 <= row["token_count"] <= 35 for row in data.rows)
    assert all(row["page_start"] == row["page_end"] in (1, 2) for row in data.rows)
    assert all(row["lecture_date"] == "1980-12-02" for row in data.rows)
    assert len({r["chunk_id"] for r in data.rows}) == len(data.rows)
    before = {
        p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in data.chunks.iterdir()
    }
    rows, manifest = chunk_pages(
        data.input, data.chunks, data.tokenizer, ChunkConfig(35, 7)
    )
    assert rows == data.rows and manifest == data.manifest
    assert before == {
        p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in data.chunks.iterdir()
    }
    # Recursive overlap must retain some preceding words at an ordinary boundary.
    assert any(
        set(a["text"].split()[-7:]) & set(b["text"].split()[:7])
        for a, b in zip(rows, rows[1:], strict=False)
        if a["page_start"] == b["page_start"]
    )


def test_lecture_boundaries_inside_pages_are_not_misattributed(pipeline):
    data = pipeline(
        [
            "02.12.1980\n\nUne première leçon.",
            "Fin de la première leçon.\n\n09.12.1980\n\nDébut de la deuxième leçon.",
            "Suite de la deuxième leçon.",
            "Table des matières\n\n02.12.1980...... 1\n09.12.1980...... 2\n16.12.1980...... 9",
        ]
    )
    second = [r for r in data.rows if r["page_start"] == 2]
    assert second[0]["lecture_date"] == "1980-12-02"
    assert second[-1]["lecture_date"] == "1980-12-09"
    assert all(
        "première" not in r["text"] for r in second if r["lecture_date"] == "1980-12-09"
    )
    assert {r["lecture_date"] for r in data.rows if r["page_start"] == 3} == {
        "1980-12-09"
    }
    assert {r["lecture_date"] for r in data.rows if r["page_start"] == 4} == {None}


def test_normalization_only_removes_matching_edge_page_number():
    assert (
        normalize_page(
            " 2\n\nLe corps    peut agir.\nUne seconde ligne.\n\nUn autre paragraphe.\n2",
            2,
        )
        == "Le corps peut agir. Une seconde ligne.\n\nUn autre paragraphe."
    )
    assert "123" in normalize_page("Le nombre 123 compte.", 2)


@pytest.mark.parametrize("size,overlap", [(0, 0), (20, -1), (20, 20), (20, 21)])
def test_invalid_chunk_limits(size, overlap):
    with pytest.raises(ValueError):
        ChunkConfig(size, overlap)


def test_chunks_reject_new_settings_and_corruption(pipeline):
    data = pipeline(["Une joie synthétique."])
    with pytest.raises(ValueError, match="different source/settings"):
        chunk_pages(data.input, data.chunks, data.tokenizer, ChunkConfig(100, 20))
    (data.chunks / "chunks.jsonl").write_text("damaged")
    with pytest.raises(ValueError, match="checksum"):
        load_chunks(data.chunks)


def test_stale_upstream_pages_block_indexing_and_search(pipeline):
    data = pipeline(["Une joie synthétique."])
    embedder = FakeEmbedder()
    index_chunks(data.chunks, data.index, embedder)
    (data.input / "pages.jsonl").write_text("damaged upstream")
    with pytest.raises(ValueError, match="checksum"):
        search("joie", data.index, embedder)
    assert len(embedder.calls) == 1


def test_index_reuse_and_query_embed_only_the_question(pipeline):
    data = pipeline(
        ["La joie augmente la puissance.", "La tristesse diminue la puissance."]
    )
    embedder = FakeEmbedder()
    first = index_chunks(data.chunks, data.index, embedder, batch_size=1)
    assert first["embedded_this_run"] == 2 and first["dimensions"] == 2
    assert all(
        not text.startswith("Instruct:") for batch in embedder.calls for text in batch
    )
    calls = len(embedder.calls)
    again = index_chunks(data.chunks, data.index, embedder, batch_size=1)
    assert again["embedded_this_run"] == 0 and again["chunks"] == 2
    assert len(embedder.calls) == calls
    results = search("Quelle est la joie ?", data.index, embedder, top_k=1)
    assert len(embedder.calls) == calls + 1
    assert embedder.calls[-1] == [
        f"Instruct: {EMBEDDING.query_instruction}\nQuery: Quelle est la joie ?"
    ]
    assert results[0]["metadata"]["page_start"] == 1
    assert results[0]["metadata"]["source_file"] == "synthetic.pdf"
    assert results[0]["chunk_id"] == data.rows[0]["chunk_id"]


def test_restart_search_opens_persisted_vectors_without_document_embeddings(pipeline):
    data = pipeline(
        ["La joie augmente la puissance.", "La tristesse diminue la puissance."]
    )
    index_chunks(data.chunks, data.index, FakeEmbedder())
    code = """import json,sys
from pathlib import Path
from spinoza_rag.search import search
class Fake:
    def identity(self):
        return {"model":"fake","digest":"fake-weights-v1","ollama_version":"test"}
    def embed(self,texts):
        assert len(texts)==1 and texts[0].startswith("Instruct:")
        return [[1.0,0.0]]
result=search("joie",Path(sys.argv[1]),Fake(),top_k=1)
print(json.dumps(result))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, str(data.index)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)[0]["metadata"]["page_start"] == 1


def test_interrupted_index_resumes_only_missing_ids(pipeline):
    data = pipeline(["La joie.", "La tristesse.", "Le désir."])
    with pytest.raises(ValueError, match="unavailable"):
        index_chunks(data.chunks, data.index, FakeEmbedder(fail_after=1), batch_size=1)
    with pytest.raises(ValueError, match="incomplete"):
        search("joie", data.index, FakeEmbedder())
    resumed = FakeEmbedder()
    result = index_chunks(data.chunks, data.index, resumed, batch_size=1)
    assert result["reused"] == 1 and result["embedded_this_run"] == 2
    assert all(data.rows[0]["text"] not in batch for batch in resumed.calls)


def test_upsert_before_checkpoint_failure_does_not_reembed(pipeline, monkeypatch):
    import spinoza_rag.index as module

    data = pipeline(["Une joie."])
    original = module.save_json
    calls = 0

    def interrupted(path, value):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic checkpoint failure")
        return original(path, value)

    monkeypatch.setattr(module, "save_json", interrupted)
    with pytest.raises(OSError, match="checkpoint failure"):
        index_chunks(data.chunks, data.index, FakeEmbedder())
    monkeypatch.setattr(module, "save_json", original)
    resumed = FakeEmbedder()
    result = index_chunks(data.chunks, data.index, resumed)
    assert result["embedded_this_run"] == 0 and result["dimensions"] == 2
    assert resumed.calls == []
    assert search("joie", data.index, resumed, top_k=1)


def test_changed_model_and_tampered_collection_are_rejected(pipeline):
    data = pipeline(["Une joie."])
    index_chunks(data.chunks, data.index, FakeEmbedder())
    changed = FakeEmbedder(digest="new-weights")
    with pytest.raises(ValueError, match="incompatible"):
        index_chunks(data.chunks, data.index, changed)
    with pytest.raises(ValueError, match="incompatible"):
        search("joie", data.index, changed)
    assert changed.calls == []
    with open_index(data.index, FakeEmbedder()) as (collection, _):
        collection.upsert(
            ids=[data.rows[0]["chunk_id"]],
            documents=["modified text"],
            embeddings=[[1.0, 0.0]],
        )
    with pytest.raises(ValueError, match="modified"):
        search("joie", data.index, FakeEmbedder())


def test_missing_input_and_unavailable_embedding_service_are_clear(pipeline, tmp_path):
    with pytest.raises((OSError, ValueError)):
        load_chunks(tmp_path / "missing")
    with pytest.raises(ValueError, match="missing"):
        search("joie", tmp_path / "missing", FakeEmbedder())
    data = pipeline(["Une joie."])

    class Offline(FakeEmbedder):
        def identity(self):
            raise ValueError("Ollama is unavailable")

    with pytest.raises(ValueError, match="unavailable"):
        index_chunks(data.chunks, data.index, Offline())
    assert not data.index.exists()


@pytest.mark.parametrize(
    "url",
    [
        "https://api.example.com",
        "http://192.168.1.2:11434",
        "http://user:pass@localhost:11434",
        "http://localhost:11434/api",
    ],
)
def test_ollama_rejects_nonlocal_or_ambiguous_endpoints(url):
    with pytest.raises(ValueError, match="loopback"):
        OllamaClient(url=url)


def test_ollama_requests_are_batched_untruncated_and_normalized():
    client = OllamaClient()
    requests = []

    def response(request, timeout):
        requests.append(request)
        return io.BytesIO(json.dumps({"embeddings": [[3.0, 4.0], [0.0, 2.0]]}).encode())

    client._opener = SimpleNamespace(open=response)
    assert client.embed(["first", "second"]) == [[0.6, 0.8], [0.0, 1.0]]
    body = json.loads(requests[0].data)
    assert body["input"] == ["first", "second"] and body["truncate"] is False
    assert requests[0].full_url == "http://127.0.0.1:11434/api/embed"


@pytest.mark.parametrize(
    "vectors", [[[0, 0]], [[float("nan"), 1]], [], [[1, 0], [0, 1]]]
)
def test_bad_ollama_vectors_fail_clearly(vectors):
    client = OllamaClient()
    client._opener = SimpleNamespace(
        open=lambda request, timeout: io.BytesIO(
            json.dumps({"embeddings": vectors}).encode()
        )
    )
    with pytest.raises(ValueError, match="Invalid Ollama embedding"):
        client.embed(["one input"])


def test_ollama_unavailable_and_missing_model_do_not_download():
    client = OllamaClient()

    def offline(*args, **kwargs):
        raise URLError("synthetic connection refused")

    client._opener = SimpleNamespace(open=offline)
    with pytest.raises(ValueError, match="unavailable"):
        client.identity()
    client._opener = SimpleNamespace(
        open=lambda request, timeout: io.BytesIO(b'{"models":[]}')
    )
    with pytest.raises(ValueError, match="ollama pull qwen3-embedding:0.6b"):
        client.identity()


def test_overlap_repeats_unique_boundary_tokens(pipeline):
    data = pipeline([" ".join(f"mot{i}" for i in range(75))], config=ChunkConfig(20, 5))
    words = [row["text"].split() for row in data.rows]
    assert len(words) > 2
    assert all(a[-5:] == b[:5] for a, b in zip(words, words[1:], strict=False))


def test_new_chunk_configuration_cannot_reuse_old_vectors(pipeline):
    data = pipeline(["La joie augmente la puissance. " * 20], config=ChunkConfig(30, 5))
    embedder = FakeEmbedder()
    index_chunks(data.chunks, data.index, embedder)
    changed = data.root / "different-chunks"
    chunk_pages(data.input, changed, data.tokenizer, ChunkConfig(20, 5))
    calls = len(embedder.calls)
    with pytest.raises(ValueError, match="incompatible"):
        index_chunks(changed, data.index, embedder)
    assert len(embedder.calls) == calls


@pytest.mark.parametrize(
    "damage", ["invalid_json", "wrong_type", "dimensions", "count"]
)
def test_bad_index_manifest_fails_before_query_embedding(pipeline, damage):
    data = pipeline(["Une joie."])
    index_chunks(data.chunks, data.index, FakeEmbedder())
    path = data.index / "manifest.json"
    manifest = json.loads(path.read_text())
    if damage == "invalid_json":
        path.write_text("invalid")
    elif damage == "wrong_type":
        path.write_text("[]")
    else:
        manifest["dimensions" if damage == "dimensions" else "chunk_count"] += 1
        path.write_text(json.dumps(manifest))
    embedder = FakeEmbedder()
    with pytest.raises(ValueError):
        search("joie", data.index, embedder)
    assert embedder.calls == []
