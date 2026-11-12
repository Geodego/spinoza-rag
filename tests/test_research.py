import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from spinoza_rag import research


def test_context_cli_never_constructs_chat_model(monkeypatch, capsys):
    def forbidden(*args):
        pytest.fail("Context mode must not construct a generation model")

    monkeypatch.setattr(research, "local_chat", forbidden)
    monkeypatch.setattr(research, "search", lambda *args, **kwargs: [])
    monkeypatch.setattr("spinoza_rag.agent.monotonic", lambda: 0.0)
    assert research.main(["context", "joie"]) == 0
    assert '"answer": null' in capsys.readouterr().out


def test_cli_reports_retrieval_failure(monkeypatch, capsys):
    def broken(*args, **kwargs):
        raise ValueError("incompatible index")

    monkeypatch.setattr(research, "search", broken)
    monkeypatch.setattr("spinoza_rag.agent.monotonic", lambda: 0.0)
    with pytest.raises(SystemExit, match="1"):
        research.main(["context", "joie"])
    captured = capsys.readouterr()
    assert "incompatible index" in captured.out
    assert "Research failed" in captured.err


@pytest.mark.parametrize(
    "flag", ["--top-k", "--context-bytes"], ids=["retrieval-limit", "evidence-budget"]
)
def test_cli_rejects_nonpositive_limits(flag):
    with pytest.raises(SystemExit, match="1"):
        research.main(["context", "joie", flag, "0"])


def test_answer_cli_reports_retrieval_failure(monkeypatch, capsys):
    class ScriptedModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    def unavailable(*args, **kwargs):
        raise ValueError("Index is stale.")

    model = ScriptedModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_spinoza",
                        "args": {"query": "joie"},
                        "id": "search-1",
                    }
                ],
            ),
            AIMessage(content="Réponse."),
        ]
    )
    monkeypatch.setattr(research, "search", unavailable)
    monkeypatch.setattr(research, "local_chat", lambda *args: model)
    monkeypatch.setattr("spinoza_rag.agent.monotonic", lambda: 0.0)
    with pytest.raises(SystemExit) as error:
        research.main(["answer", "joie"])
    assert error.value.code == 1
    captured = capsys.readouterr()
    assert "Index is stale" in captured.out
    assert "Retrieval failed" in captured.err
