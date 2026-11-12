import io
import json
from types import SimpleNamespace
from urllib.error import URLError

import pytest

from spinoza_rag.config import EMBEDDING, OLLAMA
from spinoza_rag.embeddings import OllamaClient


def fake_opener(client, respond):
    """Replace the client's HTTP opener; ``respond(request, timeout)`` answers."""
    client._opener = SimpleNamespace(open=respond)


def json_response(value):
    return lambda request, timeout: io.BytesIO(json.dumps(value).encode())


@pytest.mark.parametrize(
    "url",
    [
        "https://api.example.com",
        "http://192.168.1.2:11434",
        "http://user:pass@localhost:11434",
        "http://localhost:11434/api",
    ],
)
def test_nonlocal_or_ambiguous_endpoints_are_rejected(url):
    with pytest.raises(ValueError, match="loopback"):
        OllamaClient(url=url)


def test_requests_are_batched_untruncated_and_normalized():
    client = OllamaClient()
    requests = []

    def respond(request, timeout):
        requests.append(request)
        return io.BytesIO(json.dumps({"embeddings": [[3.0, 4.0], [0.0, 2.0]]}).encode())

    fake_opener(client, respond)
    assert client.embed(["first", "second"]) == [[0.6, 0.8], [0.0, 1.0]]
    body = json.loads(requests[0].data)
    assert body["input"] == ["first", "second"]
    assert body["truncate"] is False
    assert requests[0].full_url == f"{OLLAMA.url}/api/embed"


@pytest.mark.parametrize(
    "vectors",
    [[[0, 0]], [[float("nan"), 1]], [], [[1, 0], [0, 1]]],
    ids=["zero", "nan", "missing", "extra"],
)
def test_bad_vectors_fail_clearly(vectors):
    client = OllamaClient()
    fake_opener(client, json_response({"embeddings": vectors}))
    with pytest.raises(ValueError, match="Invalid Ollama embedding"):
        client.embed(["one input"])


def test_unavailable_service_gives_actionable_error():
    client = OllamaClient()

    def offline(request, timeout):
        raise URLError("synthetic connection refused")

    fake_opener(client, offline)
    with pytest.raises(ValueError, match="unavailable"):
        client.identity()


def test_missing_model_suggests_pull_instead_of_downloading():
    client = OllamaClient()
    fake_opener(client, json_response({"models": []}))
    with pytest.raises(ValueError, match=f"ollama pull {EMBEDDING.model}"):
        client.identity()
