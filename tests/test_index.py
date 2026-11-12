import pytest

from spinoza_rag.artifacts import save_json
from spinoza_rag.chunk import chunk_pages
from spinoza_rag.config import ChunkConfig
from spinoza_rag.index import index_chunks
from spinoza_rag.search import search


def test_documents_are_embedded_once_without_query_prefix(pipeline, make_embedder):
    data = pipeline(
        ["La joie augmente la puissance.", "La tristesse diminue la puissance."]
    )
    embedder = make_embedder()
    first = index_chunks(data.chunks, data.index, embedder, batch_size=1)
    assert first["embedded_this_run"] == 2
    assert first["dimensions"] == 2
    assert all(
        not text.startswith("Instruct:") for batch in embedder.calls for text in batch
    )
    calls = len(embedder.calls)
    again = index_chunks(data.chunks, data.index, embedder, batch_size=1)
    assert again["embedded_this_run"] == 0
    assert again["chunks"] == 2
    assert len(embedder.calls) == calls


def test_interrupted_index_resumes_only_missing_ids(pipeline, make_embedder):
    data = pipeline(["La joie.", "La tristesse.", "Le désir."])
    with pytest.raises(ValueError, match="unavailable"):
        index_chunks(data.chunks, data.index, make_embedder(fail_after=1), batch_size=1)
    resumed = make_embedder()
    result = index_chunks(data.chunks, data.index, resumed, batch_size=1)
    assert result["reused"] == 1
    assert result["embedded_this_run"] == 2
    assert all(data.rows[0]["text"] not in batch for batch in resumed.calls)


def test_upsert_before_checkpoint_failure_does_not_reembed(
    pipeline, make_embedder, monkeypatch
):
    data = pipeline(["Une joie."])
    calls = 0

    def interrupted(path, value):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic checkpoint failure")
        return save_json(path, value)

    monkeypatch.setattr("spinoza_rag.index.save_json", interrupted)
    with pytest.raises(OSError, match="checkpoint failure"):
        index_chunks(data.chunks, data.index, make_embedder())
    monkeypatch.undo()
    resumed = make_embedder()
    result = index_chunks(data.chunks, data.index, resumed)
    assert result["embedded_this_run"] == 0
    assert result["dimensions"] == 2
    assert resumed.calls == []
    assert search("joie", data.index, resumed, top_k=1)


def test_changed_model_digest_is_rejected_before_embedding(pipeline, make_embedder):
    data = pipeline(["Une joie."])
    index_chunks(data.chunks, data.index, make_embedder())
    changed = make_embedder(digest="new-weights")
    with pytest.raises(ValueError, match="incompatible"):
        index_chunks(data.chunks, data.index, changed)
    assert changed.calls == []


def test_new_chunk_configuration_cannot_reuse_old_vectors(pipeline, make_embedder):
    data = pipeline(["La joie augmente la puissance. " * 20], config=ChunkConfig(30, 5))
    embedder = make_embedder()
    index_chunks(data.chunks, data.index, embedder)
    changed = data.root / "different-chunks"
    chunk_pages(data.input, changed, data.tokenizer, ChunkConfig(20, 5))
    calls = len(embedder.calls)
    with pytest.raises(ValueError, match="incompatible"):
        index_chunks(changed, data.index, embedder)
    assert len(embedder.calls) == calls


def test_unavailable_embedding_service_creates_no_index(pipeline, make_embedder):
    data = pipeline(["Une joie."])
    with pytest.raises(ValueError, match="unavailable"):
        index_chunks(data.chunks, data.index, make_embedder(offline=True))
    assert not data.index.exists()
