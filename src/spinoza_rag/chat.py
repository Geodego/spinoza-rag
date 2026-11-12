"""Configure loopback-only Ollama chat without changing the embedding client."""

from langchain_ollama import ChatOllama

from spinoza_rag.config import AGENT, OLLAMA
from spinoza_rag.embeddings import OllamaClient


def local_chat(
    model: str = AGENT.model,
    url: str = OLLAMA.url,
    timeout: float = OLLAMA.timeout_seconds,
) -> ChatOllama:
    """Return a non-thinking chat model using installed local weights only.

    Validate the same loopback policy as embeddings; bypass environment proxies.
    Callers must disable tracing around execution, including injected callbacks.
    """
    endpoint = OllamaClient(model, url, timeout)
    return ChatOllama(
        model=endpoint.model,
        base_url=endpoint.url,
        reasoning=False,
        temperature=AGENT.temperature,
        num_ctx=AGENT.num_ctx,
        num_predict=AGENT.num_predict,
        client_kwargs={"timeout": endpoint.timeout, "trust_env": False},
        validate_model_on_init=True,
    )
