import json

from spinoza_rag.provenance import Evidence


def passage(identifier="chunk-a", text="Synthetic evidence."):
    return {
        "chunk_id": identifier,
        "text": text,
        "distance": 0.2,
        "metadata": {
            "source_file": "synthetic.pdf",
            "page_start": 2,
            "page_end": 2,
            "lecture_date": "1981-01-01",
        },
    }


def test_evidence_preserves_metadata_and_stable_labels():
    evidence = Evidence()
    first = json.loads(evidence.add([passage()]))[0]
    assert first == {**passage(), "label": "S1"}
    assert evidence.add([passage()]) == "[]"
    assert json.loads(evidence.add([passage("chunk-b")]))[0]["label"] == "S2"


def test_evidence_budget_includes_utf8_metadata_and_json():
    result = passage(text="é" * 20)
    size = len(json.dumps([{**result, "label": "S1"}], ensure_ascii=False).encode())
    evidence = Evidence(size)
    evidence.add([result])
    assert evidence.used_bytes == size
    assert evidence.add([passage("chunk-b")]) == "[]"
    assert len(evidence.sources) == 1


def test_evidence_does_not_admit_partial_chunks():
    evidence = Evidence(1)
    assert evidence.add([passage()]) == "[]"
    assert evidence.sources == []


def test_evidence_detects_unknown_labels_without_claiming_grounding():
    evidence = Evidence()
    evidence.add([passage()])
    assert evidence.unknown_citations(
        "Claim [S1], invented [S99], malformed [Sx]."
    ) == ["S99", "Sx"]
