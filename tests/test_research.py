import pytest

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
