import json
import subprocess
import sys

import pytest

from spinoza_rag.config import EMBEDDING
from spinoza_rag.index import index_chunks, open_index
from spinoza_rag.search import search

TWO_PAGES = ["La joie augmente la puissance.", "La tristesse diminue la puissance."]


def test_search_embeds_only_the_question_and_returns_provenance(
    pipeline, make_embedder
):
    data = pipeline(TWO_PAGES)
    embedder = make_embedder()
    index_chunks(data.chunks, data.index, embedder)
    calls = len(embedder.calls)
    results = search("Quelle est la joie ?", data.index, embedder, top_k=1)
    assert embedder.calls[calls:] == [
        [f"Instruct: {EMBEDDING.query_instruction}\nQuery: Quelle est la joie ?"]
    ]
    assert results[0]["metadata"]["page_start"] == 1
    assert results[0]["metadata"]["source_file"] == "synthetic.pdf"
    assert results[0]["chunk_id"] == data.rows[0]["chunk_id"]


def test_restart_search_opens_persisted_vectors_without_document_embeddings(
    pipeline, make_embedder
):
    data = pipeline(TWO_PAGES)
    index_chunks(data.chunks, data.index, make_embedder())
    # A fresh interpreter proves vectors come from disk, not process memory.
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


def test_incomplete_index_is_rejected(pipeline, make_embedder):
    data = pipeline(["La joie.", "La tristesse."])
    with pytest.raises(ValueError, match="unavailable"):
        index_chunks(data.chunks, data.index, make_embedder(fail_after=1), batch_size=1)
    with pytest.raises(ValueError, match="incomplete"):
        search("joie", data.index, make_embedder())


def test_changed_model_digest_is_rejected_before_query_embedding(
    pipeline, make_embedder
):
    data = pipeline(["Une joie."])
    index_chunks(data.chunks, data.index, make_embedder())
    changed = make_embedder(digest="new-weights")
    with pytest.raises(ValueError, match="incompatible"):
        search("joie", data.index, changed)
    assert changed.calls == []


def test_tampered_collection_is_rejected(pipeline, make_embedder):
    data = pipeline(["Une joie."])
    index_chunks(data.chunks, data.index, make_embedder())
    with open_index(data.index, make_embedder()) as (collection, _):
        collection.upsert(
            ids=[data.rows[0]["chunk_id"]],
            documents=["modified text"],
            embeddings=[[1.0, 0.0]],
        )
    with pytest.raises(ValueError, match="modified"):
        search("joie", data.index, make_embedder())


def test_stale_upstream_pages_block_search(pipeline, make_embedder):
    data = pipeline(["Une joie synthétique."])
    embedder = make_embedder()
    index_chunks(data.chunks, data.index, embedder)
    (data.input / "pages.jsonl").write_text("damaged upstream")
    with pytest.raises(ValueError, match="checksum"):
        search("joie", data.index, embedder)
    # Only the indexing call: the question was rejected before being embedded.
    assert len(embedder.calls) == 1


def test_missing_index_is_reported(tmp_path, make_embedder):
    with pytest.raises(ValueError, match="missing"):
        search("joie", tmp_path / "missing", make_embedder())


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("invalid_json", "Invalid index manifest"),
        ("wrong_type", "Invalid index manifest"),
        ("dimensions", "inconsistent"),
        ("count", "inconsistent"),
    ],
)
def test_bad_index_manifest_fails_before_query_embedding(
    pipeline, make_embedder, damage, message
):
    data = pipeline(["Une joie."])
    index_chunks(data.chunks, data.index, make_embedder())
    path = data.index / "manifest.json"
    manifest = json.loads(path.read_text())
    if damage == "invalid_json":
        path.write_text("invalid")
    elif damage == "wrong_type":
        path.write_text("[]")
    else:
        manifest["dimensions" if damage == "dimensions" else "chunk_count"] += 1
        path.write_text(json.dumps(manifest))
    embedder = make_embedder()
    with pytest.raises(ValueError, match=message):
        search("joie", data.index, embedder)
    assert embedder.calls == []
