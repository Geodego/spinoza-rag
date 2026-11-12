"""Embed text through the loopback-only Ollama API without model downloads."""

import json
import math
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

from spinoza_rag.config import EMBEDDING, OLLAMA


class Embedder(Protocol):
    """Minimal embedding contract used by the index and synthetic tests."""

    def identity(self) -> dict[str, str]:
        """Return the installed model name, digest, and runtime version."""
        ...

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector for each input, without silently truncating text."""
        ...


class OllamaClient:
    """Use an already-running local Ollama server and an already-installed model."""

    def __init__(
        self,
        model: str = EMBEDDING.model,
        url: str = OLLAMA.url,
        timeout: float = OLLAMA.timeout_seconds,
    ) -> None:
        parsed = urlsplit(url)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in ("127.0.0.1", "localhost", "::1")
            or parsed.username
            or parsed.password
            or parsed.path not in ("", "/")
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "Ollama URL must be an HTTP loopback address "
                "without a path or credentials."
            )
        if timeout <= 0 or not math.isfinite(timeout):
            raise ValueError("Ollama timeout must be positive and finite.")
        self.model, self.url, self.timeout = model, url.rstrip("/"), timeout
        self._opener = build_opener(ProxyHandler({}))

    def _request(self, path: str, body: dict[str, object] | None = None) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        request = Request(
            self.url + path, data=data, headers={"Content-Type": "application/json"}
        )
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                result = json.load(response)
            if not isinstance(result, dict):
                raise ValueError("response is not an object")
            return result
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:400]
            raise ValueError(f"Ollama HTTP {error.code}: {detail}") from error
        except (URLError, TimeoutError, OSError) as error:
            raise ValueError(
                f"Ollama is unavailable at {self.url}: {error}. "
                "Start Ollama and confirm the model is installed."
            ) from error
        except ValueError as error:
            raise ValueError(f"Invalid Ollama response: {error}") from error

    def identity(self) -> dict[str, str]:
        """Check model availability and identify its exact installed weights."""
        try:
            models = self._request("/api/tags")["models"]
            model = next((item for item in models if item["name"] == self.model), None)
            if model is None:
                raise ValueError(
                    f"Ollama model {self.model!r} is not installed; "
                    f"run: ollama pull {self.model}"
                )
            return {
                "model": self.model,
                "digest": model["digest"],
                "ollama_version": self._request("/api/version")["version"],
            }
        except (KeyError, TypeError) as error:
            raise ValueError("Invalid Ollama model/version response.") from error

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Request vectors with context truncation disabled."""
        result = self._request(
            "/api/embed",
            {
                "model": self.model,
                "input": texts,
                "truncate": False,
                "keep_alive": "10m",
            },
        )
        try:
            vectors = result["embeddings"]
            if len(vectors) != len(texts):
                raise ValueError("embedding count does not match input count")
            normalized = []
            for vector in vectors:
                values = [float(value) for value in vector]
                if not values or not all(math.isfinite(value) for value in values):
                    raise ValueError("empty or non-finite embedding")
                norm = math.sqrt(sum(value * value for value in values))
                if not norm:
                    raise ValueError("zero embedding")
                normalized.append([value / norm for value in values])
            return normalized
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid Ollama embedding response: {error}") from error
