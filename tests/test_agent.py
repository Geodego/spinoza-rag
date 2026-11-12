import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langsmith import get_tracing_context

from spinoza_rag import agent
from spinoza_rag.agent import ResearchError, ResearchSession, research
from spinoza_rag.config import AGENT


class ToolModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


@pytest.fixture(autouse=True)
def fake_timer(monkeypatch):
    monkeypatch.setattr(agent, "monotonic", lambda: 1.0)


@pytest.fixture
def retrieve():
    def synthetic(query):
        return [
            {
                "chunk_id": query,
                "text": "Synthetic evidence: joy increases power.",
                "distance": 0.2,
                "metadata": {
                    "source_file": "synthetic.pdf",
                    "page_start": 1,
                    "page_end": 1,
                },
            }
        ]

    return synthetic


def tool_message(*queries):
    return AIMessage(
        content="",
        tool_calls=[
            {"name": "search_spinoza", "args": {"query": query}, "id": str(i)}
            for i, query in enumerate(queries)
        ],
    )


def test_context_returns_passages_without_model(retrieve):
    result = research("joie", retrieve)
    assert result["answer"] is None
    assert result["sources"][0]["chunk_id"] == "joie"
    assert result["sources"][0]["label"] == "S1"
    assert [e["event"] for e in result["events"]] == ["search_start", "search_end"]


def test_agent_chooses_followup_queries_and_maps_citations(retrieve):
    model = ToolModel(
        responses=[
            tool_message("joie"),
            tool_message("tristesse"),
            AIMessage(content="La joie [S1]; la tristesse [S2]."),
        ]
    )
    result = research("Comparer", retrieve, model=model)
    assert result["answer"] == "La joie [S1]; la tristesse [S2]."
    assert [s["chunk_id"] for s in result["sources"]] == ["joie", "tristesse"]
    assert len([e for e in result["events"] if e["event"] == "model_start"]) == 3


def test_parallel_tool_requests_cannot_exceed_two_retrievals(retrieve):
    session = ResearchSession(retrieve)
    with ThreadPoolExecutor(max_workers=3) as executor:
        outputs = list(executor.map(session.search, ["a", "b", "c"]))
    assert session.search_calls == 2
    assert len(session.evidence.sources) == 2
    assert sum("error" in json.loads(output) for output in outputs) == 1


def test_graph_enforces_limit_with_multiple_tool_calls(retrieve):
    model = ToolModel(
        responses=[tool_message("a", "b", "c"), AIMessage(content="Preuve [S1].")]
    )
    result = research("Question", retrieve, model=model)
    assert len(result["sources"]) == 2
    assert len([e for e in result["events"] if e["event"] == "search_blocked"]) == 1


def test_repeated_tool_loop_hits_model_limit(retrieve):
    model = ToolModel(responses=[tool_message("a")])
    with pytest.raises(ResearchError, match="Model call limit"):
        research("Question", retrieve, model=model)


def test_agent_withholds_answer_if_model_skips_retrieval(retrieve):
    result = research(
        "Question",
        retrieve,
        model=ToolModel(
            responses=[AIMessage(content="An unsupported assertion [S1].")]
        ),
    )
    assert (
        result["answer"] == "Les preuves recueillies sont insuffisantes pour répondre."
    )
    assert result["sources"] == []


def test_agent_detects_unknown_citations(retrieve):
    model = ToolModel(responses=[tool_message("a"), AIMessage(content="False [S99].")])
    with pytest.raises(ResearchError, match=r"Unknown citation labels.*S99"):
        research("Question", retrieve, model=model)


def test_agent_rejects_uncited_claims(retrieve):
    model = ToolModel(responses=[tool_message("a"), AIMessage(content="A claim.")])
    with pytest.raises(ResearchError, match="no source citations"):
        research("Question", retrieve, model=model)


@pytest.mark.parametrize(
    "refusal",
    [
        "Les preuves sont insuffisantes pour répondre.",
        "Les preuves sont insuffisantes pour répondre. "
        "Une similarité vectorielle ne prouve rien.",
    ],
    ids=["bare-refusal", "refusal-with-safe-caution"],
)
def test_agent_allows_explicit_claim_free_refusal(retrieve, refusal):
    model = ToolModel(
        responses=[
            tool_message("a"),
            AIMessage(content=refusal),
        ]
    )
    assert research("Question", retrieve, model=model)["answer"].startswith(
        "Les preuves"
    )


def test_agent_rejects_truncated_answer(retrieve):
    model = ToolModel(
        responses=[
            tool_message("a"),
            AIMessage(
                content="Claim [S1]", response_metadata={"done_reason": "length"}
            ),
        ]
    )
    with pytest.raises(ResearchError, match="generation limit"):
        research("Question", retrieve, model=model)


def test_tool_failure_is_visible_and_does_not_supply_evidence():
    def unavailable(query):
        raise ValueError("Index is stale.")

    session = ResearchSession(unavailable)
    assert "error" in json.loads(session.search("joie"))
    assert session.evidence.sources == []
    assert session.events[-1]["error"] == "Index is stale."
    assert session.search_calls == 1


def test_context_failure_is_not_reported_as_empty_success():
    def unavailable(query):
        raise OSError("Unavailable")

    with pytest.raises(ResearchError, match="Context retrieval failed"):
        research("Question", unavailable)


def test_model_failure_reports_local_error(retrieve):
    class FailingModel(ToolModel):
        def _generate(self, *args, **kwargs):
            raise ConnectionError("Ollama unavailable")

    with pytest.raises(ResearchError, match="Ollama unavailable"):
        research("Question", retrieve, model=FailingModel(responses=[]))


def test_ollama_prompt_and_generation_timings_are_separate(retrieve):
    model = ToolModel(
        responses=[
            tool_message("a"),
            AIMessage(
                content="Claim [S1].",
                response_metadata={
                    "prompt_eval_duration": 2_000_000_000,
                    "eval_duration": 3_000_000_000,
                },
            ),
        ]
    )
    result = research("Question", retrieve, model=model)
    metrics = [e for e in result["events"] if e["event"] == "model_end"]
    assert metrics[-1]["prompt_seconds"] == 2
    assert metrics[-1]["generation_seconds"] == 3
    assert metrics[0]["prompt_seconds"] is None


def test_hosted_tracing_is_disabled_even_when_environment_enables_it(
    monkeypatch, retrieve
):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")

    class InspectingModel(ToolModel):
        def _generate(self, *args, **kwargs):
            assert get_tracing_context()["enabled"] is False
            return super()._generate(*args, **kwargs)

    research(
        "Question",
        retrieve,
        model=InspectingModel(
            responses=[tool_message("a"), AIMessage(content="Claim [S1].")]
        ),
    )


def test_budget_exhaustion_produces_safe_insufficient_evidence(retrieve):
    model = ToolModel(responses=[tool_message("a"), AIMessage(content="Claim [S1].")])
    result = research("Question", retrieve, model=model, context_bytes=1)
    assert result["sources"] == []
    assert result["evidence_bytes"] == 0
    assert "insuffisantes" in result["answer"]


@pytest.mark.parametrize("question", [" ", "é" * 1001], ids=["empty", "over-budget"])
def test_invalid_question_is_rejected_before_retrieval(question, retrieve):
    with pytest.raises(ValueError, match="Question must"):
        research(question, retrieve)


def test_context_reuses_validated_search_and_preserves_index(pipeline, make_embedder):
    from spinoza_rag.index import index_chunks
    from spinoza_rag.search import search

    data = pipeline(["Synthetic joy increases power."])
    embedder = make_embedder()
    index_chunks(data.chunks, data.index, embedder)
    manifest = (data.index / "manifest.json").read_bytes()
    result = research("joy", lambda query: search(query, data.index, embedder))
    assert result["sources"][0]["metadata"]["page_start"] == 1
    assert result["sources"][0]["chunk_id"] == data.rows[0]["chunk_id"]
    assert (data.index / "manifest.json").read_bytes() == manifest
    assert len(embedder.calls) == 2


def test_invalid_query_consumes_slot_without_retrieval(retrieve):
    session = ResearchSession(retrieve)
    session.search("é" * 501)
    assert session.search_calls == 1
    assert session.evidence.sources == []
    assert session.events[-1]["event"] == "search_error"


def test_no_evidence_still_detects_unknown_labels(retrieve):
    result = research(
        "Question",
        retrieve,
        model=ToolModel(responses=[AIMessage(content="Unsupported [S99].")]),
    )
    event = next(e for e in result["events"] if e["event"] == "unknown_citations")
    assert event["labels"] == ["S99"]


def test_refusal_prefix_cannot_hide_uncited_claims(retrieve):
    model = ToolModel(
        responses=[
            tool_message("a"),
            AIMessage(
                content=(
                    "Les preuves sont insuffisantes pour répondre. An uncited claim."
                )
            ),
        ]
    )
    with pytest.raises(ResearchError, match="no source citations"):
        research("Question", retrieve, model=model)


def test_answer_retrieval_failure_without_evidence_raises():
    def unavailable(query):
        raise ValueError("Index is stale.")

    model = ToolModel(responses=[tool_message("a", "b"), AIMessage(content="Réponse.")])
    with pytest.raises(ResearchError, match="Retrieval failed"):
        research("Question", unavailable, model=model)


def test_partial_retrieval_failure_keeps_admitted_evidence(retrieve):
    def intermittent(query):
        if query == "unavailable":
            raise ValueError("Embedding temporarily unavailable.")
        return retrieve(query)

    model = ToolModel(
        responses=[
            tool_message("unavailable"),
            tool_message("available"),
            AIMessage(content="Synthetic evidence [S1]."),
        ]
    )
    result = research("Question", intermittent, model=model)
    assert result["answer"] == "Synthetic evidence [S1]."
    assert result["sources"][0]["chunk_id"] == "available"
    assert [e["event"] for e in result["events"]].count("search_error") == 1


@pytest.mark.parametrize(
    "citation",
    ["[S1, S2]", "[S1,S2]", "[S1; S2]", "(S1)"],
    ids=["spaced-group", "compact-group", "semicolon", "parenthesis"],
)
def test_agent_accepts_known_grouped_citations(retrieve, citation):
    model = ToolModel(
        responses=[
            tool_message("a", "b"),
            AIMessage(content=f"Synthetic evidence {citation}."),
        ]
    )
    result = research("Question", retrieve, model=model)
    assert result["answer"] == f"Synthetic evidence {citation}."


@pytest.mark.parametrize(
    "citation", ["[S1, S9]", "(S9)"], ids=["mixed-group", "parenthesis"]
)
def test_agent_rejects_unknown_label_alongside_known_citation(retrieve, citation):
    model = ToolModel(
        responses=[
            tool_message("a", "b"),
            AIMessage(content=f"Claim [S1]; claim {citation}."),
        ]
    )
    with pytest.raises(ResearchError, match=r"Unknown citation labels.*S9"):
        research("Question", retrieve, model=model)


@pytest.mark.parametrize(
    "question",
    ["a" * (AGENT.query_bytes + 1), "é" * (AGENT.query_bytes // 2 + 1)],
    ids=["ascii", "utf8"],
)
def test_context_question_over_query_limit_is_rejected_before_retrieval(question):
    calls = []

    def recording(query):
        calls.append(query)
        return []

    with pytest.raises(ValueError, match=str(AGENT.query_bytes)):
        research(question, recording)
    assert calls == []
