import hashlib
import json
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from spinoza_rag import evaluate
from spinoza_rag.artifacts import json_bytes
from spinoza_rag.config import EMBEDDING, EVALUATION, OLLAMA, PATHS
from spinoza_rag.index import index_chunks

EXECUTED_AT = datetime(2000, 1, 2, tzinfo=UTC)


def chunk(start, end=None, identifier=None):
    return {
        "chunk_id": identifier or f"chunk-{start}-{end}",
        "text": "Synthetic corpus text, excluded from reports.",
        "metadata": {"page_start": start, "page_end": start if end is None else end},
        "distance": 0.1,
    }


@pytest.fixture
def frozen_clock(monkeypatch):
    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return EXECUTED_AT

    monkeypatch.setattr(evaluate, "datetime", FrozenDatetime)


@pytest.fixture
def gold_factory(tmp_path):
    def build(data=None, *, validated=True):
        identity = (
            data.manifest["source"]["document_id"] if data else "sha256:" + "a" * 64
        )
        checksum = data.manifest["pages_sha256"] if data else "b" * 64
        value = {
            "schema_version": 1,
            "version": "test-v1",
            "document_id": identity,
            "pages_sha256": checksum,
            "discovery": {
                "method": "keyword_search_and_direct_page_inspection",
                "date": "2000-01-01",
                "annotation": "Synthetic annotations; no human corpus approval.",
            },
            "questions": [
                {
                    "id": "q1",
                    "question_fr": "Quelle joie ?",
                    "references": [{"pdf_page": 1}],
                    "status": "validated" if validated else "proposed",
                }
            ],
        }
        if validated:
            value["questions"][0]["validation"] = {
                "reviewer": "Synthetic reviewer",
                "validated_on": "2000-01-01",
            }
        path = tmp_path / "gold.json"
        path.write_bytes(json_bytes(value))
        return path, value

    return build


def test_rank_pages_deduplicates_repeated_and_overlapping_pages():
    ranked = evaluate.rank_pages(
        [chunk(2, identifier="a"), chunk(2), chunk(4, 6, "b"), chunk(5, 7, "c")]
    )
    assert ranked == [
        {"pdf_page": 2, "rank": 1, "chunk_id": "a"},
        {"pdf_page": 4, "rank": 4, "chunk_id": "b"},
        {"pdf_page": 5, "rank": 4, "chunk_id": "b"},
        {"pdf_page": 6, "rank": 4, "chunk_id": "b"},
        {"pdf_page": 7, "rank": 5, "chunk_id": "c"},
    ]


@pytest.mark.parametrize(
    ("k", "expected"),
    [
        (1, {"hit": 0, "recall": 0, "mrr": 0}),
        (3, {"hit": 1, "recall": 1, "mrr": 1 / 3}),
    ],
    ids=["partial-block-no-credit", "whole-block-full-cost"],
)
def test_spanning_chunk_costs_all_new_page_slots(k, expected):
    ranked = evaluate.rank_pages([chunk(4, 6)])
    assert evaluate.score_pages(ranked, {4, 6}, k) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("k", "expected"),
    [
        (1, {"hit": 0, "recall": 0, "mrr": 0}),
        (3, {"hit": 1, "recall": 1 / 3, "mrr": 1 / 2}),
        (5, {"hit": 1, "recall": 2 / 3, "mrr": 1 / 2}),
        (10, {"hit": 1, "recall": 2 / 3, "mrr": 1 / 2}),
    ],
    ids=["miss-at-one", "first-support-at-two", "partial-support", "short-list"],
)
def test_score_pages_matches_hand_calculated_metrics(k, expected):
    ranked = evaluate.rank_pages([chunk(9), chunk(2), chunk(3), chunk(4)])
    assert evaluate.score_pages(ranked, {2, 4, 8}, k) == pytest.approx(expected)


def test_empty_retrieval_scores_zero():
    assert evaluate.score_pages([], {1}, 10) == {"hit": 0, "recall": 0, "mrr": 0}


@pytest.mark.parametrize(
    ("start", "end"),
    [(True, 2), (0, 1), (3, 2), (1, "2")],
    ids=["boolean", "zero", "reversed", "string"],
)
def test_bad_retrieved_page_ranges_are_rejected(start, end):
    with pytest.raises(ValueError, match="page range"):
        evaluate.rank_pages([chunk(start, end)])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("status", "approved", "status"),
        ("references", [], "empty"),
        ("references", [{"pdf_page": True}], "PDF page"),
        ("references", [{"pdf_page": 1}, {"pdf_page": 1}], "duplicate"),
        ("references", [{"pdf_page": 1, "lecture_date": "2000-02-30"}], "date"),
        ("question_fr", " ", "question"),
        ("passage", "Forbidden corpus text", "unknown"),
        ("validation", {"reviewer": "", "validated_on": "2000-01-01"}, "reviewer"),
    ],
    ids=[
        "status",
        "no-pages",
        "bool-page",
        "duplicate-page",
        "bad-date",
        "empty-question",
        "unknown-passage-field",
        "empty-reviewer",
    ],
)
def test_invalid_question_annotations_are_rejected(gold_factory, field, value, message):
    path, gold = gold_factory()
    gold["questions"][0][field] = value
    path.write_bytes(json_bytes(gold))
    with pytest.raises(ValueError, match=message):
        evaluate.load_gold_set(path)


def test_validation_requires_explicit_human_attribution(gold_factory):
    path, gold = gold_factory(validated=False)
    gold["questions"][0]["status"] = "validated"
    path.write_bytes(json_bytes(gold))
    with pytest.raises(ValueError, match="missing"):
        evaluate.load_gold_set(path)


def test_proposed_annotations_cannot_carry_validation(gold_factory):
    path, gold = gold_factory()
    gold["questions"][0]["status"] = "proposed"
    path.write_bytes(json_bytes(gold))
    with pytest.raises(ValueError, match="must not carry"):
        evaluate.load_gold_set(path)


def test_duplicate_question_ids_are_rejected(gold_factory):
    path, gold = gold_factory()
    gold["questions"].append(gold["questions"][0])
    path.write_bytes(json_bytes(gold))
    with pytest.raises(ValueError, match="Duplicate question"):
        evaluate.load_gold_set(path)


def test_vector_reference_discovery_is_rejected(gold_factory):
    path, gold = gold_factory()
    gold["discovery"]["method"] = "vector_search"
    path.write_bytes(json_bytes(gold))
    with pytest.raises(ValueError, match="independent"):
        evaluate.load_gold_set(path)


def test_no_validated_questions_refuses_official_run_before_service_access(
    gold_factory, tmp_path, make_embedder
):
    path, _ = gold_factory(validated=False)
    with pytest.raises(ValueError, match="No validated questions"):
        evaluate.evaluate(
            path,
            tmp_path / "absent",
            make_embedder(offline=True),
            executed_at=EXECUTED_AT,
        )


@pytest.fixture
def evaluated_index(pipeline, make_embedder, gold_factory):
    data = pipeline(["CORPUS_ONLY joie.", "CORPUS_ONLY tristesse."])
    index_chunks(data.chunks, data.index, make_embedder())
    path, gold = gold_factory(data)
    gold["questions"].append(
        {
            "id": "q2",
            "question_fr": "Quelle tristesse ?",
            "references": [{"pdf_page": 1}],
            "status": "proposed",
        }
    )
    path.write_bytes(json_bytes(gold))
    return data, path, gold


@pytest.fixture
def evaluation_run(evaluated_index, make_embedder, forbid_pdf_parsing):
    data, path, _ = evaluated_index
    before = path.read_bytes()
    manifest_before = (data.index / "manifest.json").read_bytes()
    embedder = make_embedder()
    forbid_pdf_parsing()
    report = evaluate.evaluate(path, data.index, embedder, executed_at=EXECUTED_AT)
    return SimpleNamespace(
        data=data,
        path=path,
        embedder=embedder,
        report=report,
        gold_before=before,
        manifest_before=manifest_before,
    )


def test_evaluation_embeds_only_validated_questions(evaluation_run):
    assert evaluation_run.embedder.calls == [
        [f"Instruct: {EMBEDDING.query_instruction}\nQuery: Quelle joie ?"]
    ]


def test_evaluation_reports_official_metrics(evaluation_run):
    assert evaluation_run.report["official"]["metrics"]["1"] == {
        "hit": 1,
        "recall": 1,
        "mrr": 1,
    }


def test_evaluation_has_no_exploratory_metrics_by_default(evaluation_run):
    assert evaluation_run.report["exploratory_proposed"]["metrics"] is None


def test_evaluation_reports_excluded_proposed_count(evaluation_run):
    assert evaluation_run.report["excluded_proposed_count"] == 1


def test_evaluation_records_execution_date_in_utc(evaluation_run):
    assert evaluation_run.report["executed_at"] == "2000-01-02T00:00:00+00:00"


def test_evaluation_records_model_identity(evaluation_run):
    assert evaluation_run.report["index_manifest"]["configuration"]["embedding"] == {
        "model": "fake",
        "digest": "fake-weights-v1",
        "ollama_version": "test",
    }


def test_evaluation_records_chunking_configuration(evaluation_run):
    assert evaluation_run.report["chunking"] == evaluation_run.data.manifest["chunking"]


def test_evaluation_records_index_fingerprint(evaluation_run):
    configuration = json.loads(evaluation_run.manifest_before)["configuration"]
    assert (
        evaluation_run.report["index_fingerprint"]
        == hashlib.sha256(json_bytes(configuration)).hexdigest()
    )


def test_evaluation_report_excludes_corpus_text(evaluation_run):
    assert "CORPUS_ONLY" not in json.dumps(evaluation_run.report)


def test_evaluation_preserves_annotations(evaluation_run):
    assert evaluation_run.path.read_bytes() == evaluation_run.gold_before


def test_evaluation_preserves_index_manifest(evaluation_run):
    assert (evaluation_run.data.index / "manifest.json").read_bytes() == (
        evaluation_run.manifest_before
    )


def test_evaluation_fingerprints_embedding_implementation(evaluation_run):
    from spinoza_rag import embeddings

    assert evaluation_run.report["implementation_sha256"]["embeddings.py"] == (
        hashlib.sha256(Path(embeddings.__file__).read_bytes()).hexdigest()
    )


def test_proposed_questions_never_change_official_metrics(
    evaluated_index, make_embedder
):
    data, path, _ = evaluated_index
    report = evaluate.evaluate(
        path,
        data.index,
        make_embedder(),
        executed_at=EXECUTED_AT,
        include_proposed=True,
    )
    assert report["official"]["question_count"] == 1
    assert report["official"]["metrics"]["1"]["hit"] == 1
    assert report["exploratory_proposed"]["question_count"] == 1
    assert report["exploratory_proposed"]["metrics"]["1"]["hit"] == 0
    assert len(report["per_question"]) == 2


def test_exploratory_only_report_has_no_official_metrics(
    evaluated_index, make_embedder
):
    data, path, gold = evaluated_index
    gold["questions"].pop(0)
    path.write_bytes(json_bytes(gold))
    report = evaluate.evaluate(
        path,
        data.index,
        make_embedder(),
        executed_at=EXECUTED_AT,
        include_proposed=True,
    )
    assert report["official"] == {
        "available": False,
        "question_count": 0,
        "metrics": None,
    }
    assert report["exploratory_proposed"]["question_count"] == 1


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("document_id", "sha256:" + "c" * 64, "fingerprint"),
        ("pages_sha256", "d" * 64, "fingerprint"),
    ],
    ids=["wrong-document", "changed-pages"],
)
def test_mismatched_gold_corpus_is_rejected_before_embedding(
    evaluated_index, make_embedder, field, value, message
):
    data, path, gold = evaluated_index
    gold[field] = value
    path.write_bytes(json_bytes(gold))
    embedder = make_embedder()
    with pytest.raises(ValueError, match=message):
        evaluate.evaluate(path, data.index, embedder, executed_at=EXECUTED_AT)
    assert embedder.calls == []


@pytest.mark.parametrize(
    "question_index", [0, 1], ids=["validated", "excluded-proposed"]
)
def test_out_of_document_support_is_rejected(
    evaluated_index, make_embedder, question_index
):
    data, path, gold = evaluated_index
    gold["questions"][question_index]["references"] = [{"pdf_page": 3}]
    path.write_bytes(json_bytes(gold))
    with pytest.raises(ValueError, match="outside"):
        evaluate.evaluate(path, data.index, make_embedder(), executed_at=EXECUTED_AT)


def test_changed_model_is_rejected_before_embedding(evaluated_index, make_embedder):
    data, path, _ = evaluated_index
    embedder = make_embedder(digest="changed")
    with pytest.raises(ValueError, match="incompatible"):
        evaluate.evaluate(path, data.index, embedder, executed_at=EXECUTED_AT)
    assert embedder.calls == []


@pytest.fixture
def cli_run(
    evaluated_index, make_embedder, monkeypatch, tmp_path, capsys, frozen_clock
):
    data, path, _ = evaluated_index
    seen = {}
    real_evaluate = evaluate.evaluate

    def run(gold_set, index_dir, embedder, **options):
        seen.update(
            gold_set=gold_set,
            index_dir=index_dir,
            model=embedder.model,
            url=embedder.url,
            timeout=embedder.timeout,
            **options,
        )
        return real_evaluate(path, data.index, make_embedder(), **options)

    monkeypatch.setattr(evaluate, "evaluate", run)
    monkeypatch.chdir(tmp_path)
    status = evaluate.main([])
    report = json.loads((PATHS.evaluation_dir / "report.json").read_bytes())
    return SimpleNamespace(
        status=status,
        seen=seen,
        report=report,
        summary=json.loads(capsys.readouterr().out),
    )


def test_cli_defaults_come_from_config(cli_run):
    assert cli_run.seen == {
        "gold_set": PATHS.gold_set,
        "index_dir": PATHS.index_dir,
        "model": EMBEDDING.model,
        "url": OLLAMA.url,
        "timeout": OLLAMA.timeout_seconds,
        "executed_at": EXECUTED_AT,
        "include_proposed": False,
        "chunk_limit": EVALUATION.chunk_limit,
    }


def test_cli_success_returns_zero(cli_run):
    assert cli_run.status == 0


def test_cli_report_keeps_only_selected_status(cli_run):
    assert [row["annotation_status"] for row in cli_run.report["per_question"]] == [
        "validated"
    ]


def test_cli_summary_matches_persisted_report(cli_run):
    assert cli_run.summary == {
        "official": cli_run.report["official"],
        "exploratory_proposed": cli_run.report["exploratory_proposed"],
        "report": str(PATHS.evaluation_dir / "report.json"),
    }


def test_cli_failure_returns_nonzero_without_output(
    gold_factory, tmp_path, monkeypatch, capsys, frozen_clock
):
    path, _ = gold_factory(validated=False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="1"):
        evaluate.main(["--gold-set", str(path)])
    assert "No validated questions" in capsys.readouterr().err
    assert not Path(PATHS.evaluation_dir).exists()


def test_official_aggregation_uses_macro_recall(evaluated_index, make_embedder):
    data, path, gold = evaluated_index
    gold["questions"][1].update(
        question_fr="Encore une joie ?",
        references=[{"pdf_page": 1}, {"pdf_page": 2}],
        status="validated",
        validation={"reviewer": "Synthetic reviewer", "validated_on": "2000-01-01"},
    )
    path.write_bytes(json_bytes(gold))
    report = evaluate.evaluate(
        path, data.index, make_embedder(), executed_at=EXECUTED_AT
    )
    assert report["official"]["question_count"] == 2
    assert report["official"]["metrics"]["1"] == pytest.approx(
        {"hit": 1, "recall": 0.75, "mrr": 1}
    )


def test_official_aggregation_includes_misses(evaluated_index, make_embedder):
    data, path, gold = evaluated_index
    gold["questions"][1].update(
        status="validated",
        validation={"reviewer": "Synthetic reviewer", "validated_on": "2000-01-01"},
    )
    path.write_bytes(json_bytes(gold))
    report = evaluate.evaluate(
        path, data.index, make_embedder(), executed_at=EXECUTED_AT
    )
    assert report["official"]["metrics"]["1"] == pytest.approx(
        {"hit": 0.5, "recall": 0.5, "mrr": 0.5}
    )


def test_small_chunk_budget_is_rejected_before_embedding(
    gold_factory, tmp_path, make_embedder
):
    path, _ = gold_factory()
    with pytest.raises(ValueError, match="largest page cutoff"):
        evaluate.evaluate(
            path,
            tmp_path,
            make_embedder(offline=True),
            executed_at=EXECUTED_AT,
            chunk_limit=9,
        )


def test_naive_execution_date_is_rejected_before_embedding(
    gold_factory, tmp_path, make_embedder
):
    path, _ = gold_factory()
    with pytest.raises(ValueError, match="timezone-aware"):
        evaluate.evaluate(
            path,
            tmp_path,
            make_embedder(offline=True),
            executed_at=datetime(2000, 1, 1),
        )


def test_cli_rejects_existing_output_before_embedding(
    evaluated_index, make_embedder, monkeypatch, tmp_path, capsys, frozen_clock
):
    data, path, _ = evaluated_index
    embedder = make_embedder()
    monkeypatch.setattr(evaluate, "OllamaClient", lambda *args: embedder)
    output = tmp_path / "occupied"
    output.mkdir()
    with pytest.raises(SystemExit, match="1"):
        evaluate.main(
            [
                "--gold-set",
                str(path),
                "--index-dir",
                str(data.index),
                "--output-dir",
                str(output),
            ]
        )
    assert embedder.calls == []
    assert "already exists" in capsys.readouterr().err


def test_evaluation_verifies_and_opens_index_once(
    evaluated_index, make_embedder, monkeypatch
):
    from spinoza_rag import index

    data, path, _ = evaluated_index
    verified_opens = []
    real_open = index.open_index

    @contextmanager
    def counting_open(index_dir, embedder):
        verified_opens.append(index_dir)
        with real_open(index_dir, embedder) as opened:
            yield opened

    monkeypatch.setattr(evaluate, "open_index", counting_open)
    monkeypatch.setattr("spinoza_rag.search.open_index", counting_open)
    evaluate.evaluate(
        path,
        data.index,
        make_embedder(),
        executed_at=EXECUTED_AT,
        include_proposed=True,
    )
    assert verified_opens == [data.index]


@pytest.mark.parametrize(
    "method",
    [
        "keyword_search_and_direct_page_inspection",
        "two_pass_ai_keyword_and_direct_page_inspection",
    ],
    ids=["direct-inspection", "two-pass-ai"],
)
def test_independent_discovery_methods_are_preserved(gold_factory, method):
    path, gold = gold_factory()
    gold["discovery"]["method"] = method
    path.write_bytes(json_bytes(gold))
    assert evaluate.load_gold_set(path)["discovery"]["method"] == method


def test_annotation_errors_name_missing_and_unknown_fields(gold_factory):
    path, gold = gold_factory()
    del gold["questions"][0]["question_fr"]
    gold["questions"][0]["passage"] = "Synthetic unknown field"
    path.write_bytes(json_bytes(gold))
    with pytest.raises(ValueError, match="Invalid gold set") as error:
        evaluate.load_gold_set(path)
    assert str(error.value) == (
        "Invalid gold set: Object fields: missing=['question_fr'], unknown=['passage']."
    )


def test_include_proposed_excludes_no_questions(evaluated_index, make_embedder):
    data, path, _ = evaluated_index
    report = evaluate.evaluate(
        path,
        data.index,
        make_embedder(),
        executed_at=EXECUTED_AT,
        include_proposed=True,
    )
    assert report["excluded_proposed_count"] == 0


@pytest.mark.parametrize(
    "artifact", ["gold", "index"], ids=["changed-gold", "changed-index"]
)
def test_mutation_during_evaluation_refuses_report(
    evaluated_index, make_embedder, artifact
):
    data, path, _ = evaluated_index
    target = {"gold": path, "index": data.index / "manifest.json"}[artifact]

    class ChangingEmbedder(make_embedder):
        def embed(self, texts):
            vectors = super().embed(texts)
            value = json.loads(target.read_bytes())
            value["concurrent_mutation"] = True
            target.write_bytes(json_bytes(value))
            return vectors

    with pytest.raises(ValueError, match="changed during evaluation"):
        evaluate.evaluate(path, data.index, ChangingEmbedder(), executed_at=EXECUTED_AT)


@pytest.mark.parametrize("kind", ["file", "directory", "dangling-symlink"])
def test_occupied_output_is_rejected_before_service_access(tmp_path, monkeypatch, kind):
    def service_forbidden(*args, **kwargs):
        pytest.fail("An occupied output must be rejected before service construction")

    monkeypatch.setattr(evaluate, "OllamaClient", service_forbidden)
    output = tmp_path / "occupied"
    creators = {
        "file": lambda: output.write_bytes(b"keep"),
        "directory": lambda: output.mkdir(),
        "dangling-symlink": lambda: output.symlink_to(tmp_path / "absent"),
    }
    creators[kind]()
    with pytest.raises(SystemExit, match="1"):
        evaluate.main(["--output-dir", str(output)])
