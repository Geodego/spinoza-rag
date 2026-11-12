import pytest

from spinoza_rag import chat
from spinoza_rag.config import AGENT


def test_local_chat_disables_thinking_and_proxies(monkeypatch):
    seen = {}

    def factory(**kwargs):
        seen.update(kwargs)
        return object()

    monkeypatch.setattr(chat, "ChatOllama", factory)
    chat.local_chat()
    assert seen["reasoning"] is False
    assert seen["client_kwargs"]["trust_env"] is False
    assert seen["num_predict"] == AGENT.num_predict
    assert seen["validate_model_on_init"] is True


@pytest.mark.parametrize(
    "url",
    ["https://example.org", "http://user@localhost:11434"],
    ids=["remote", "credentials"],
)
def test_local_chat_rejects_nonlocal_endpoints(url):
    with pytest.raises(ValueError, match="loopback"):
        chat.local_chat(url=url)
