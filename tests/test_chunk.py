from itertools import pairwise

import pytest

from spinoza_rag.chunk import chunk_pages, load_chunks, normalize_page
from spinoza_rag.config import ChunkConfig


def test_page_chunks_are_stable_bounded_and_do_not_read_pdf(
    pipeline, forbid_pdf_parsing, snapshot
):
    forbid_pdf_parsing()
    data = pipeline(
        [
            "02.12.1980\n\n" + "Une joie partagée augmente la puissance. " * 20,
            "La tristesse réduit la puissance. " * 20,
        ],
        config=ChunkConfig(35, 7),
    )
    assert len(data.rows) > 2
    assert all(1 <= row["token_count"] <= 35 for row in data.rows)
    assert all(row["page_start"] == row["page_end"] in (1, 2) for row in data.rows)
    assert all(row["lecture_date"] == "1980-12-02" for row in data.rows)
    assert len({row["chunk_id"] for row in data.rows}) == len(data.rows)
    before = snapshot(data.chunks, with_mtime=True)
    rows, manifest = chunk_pages(
        data.input, data.chunks, data.tokenizer, ChunkConfig(35, 7)
    )
    assert rows == data.rows
    assert manifest == data.manifest
    assert snapshot(data.chunks, with_mtime=True) == before
    # Recursive overlap must retain some preceding words at an ordinary boundary.
    assert any(
        set(a["text"].split()[-7:]) & set(b["text"].split()[:7])
        for a, b in pairwise(rows)
        if a["page_start"] == b["page_start"]
    )


def test_overlap_repeats_unique_boundary_tokens(pipeline):
    data = pipeline([" ".join(f"mot{i}" for i in range(75))], config=ChunkConfig(20, 5))
    words = [row["text"].split() for row in data.rows]
    assert len(words) > 2
    assert all(a[-5:] == b[:5] for a, b in pairwise(words))


def test_lecture_boundaries_inside_pages_are_not_misattributed(pipeline):
    """Page 2 ends one lecture and starts the next; page 4 is a contents page.

    The contents page lists dates that must not start lectures or carry over.
    """
    data = pipeline(
        [
            "02.12.1980\n\nUne première leçon.",
            "Fin de la première leçon.\n\n09.12.1980\n\nDébut de la deuxième leçon.",
            "Suite de la deuxième leçon.",
            "Table des matières\n\n"
            "02.12.1980...... 1\n09.12.1980...... 2\n16.12.1980...... 9",
        ]
    )
    second = [row for row in data.rows if row["page_start"] == 2]
    assert second[0]["lecture_date"] == "1980-12-02"
    assert second[-1]["lecture_date"] == "1980-12-09"
    assert all(
        "première" not in row["text"]
        for row in second
        if row["lecture_date"] == "1980-12-09"
    )
    assert {row["lecture_date"] for row in data.rows if row["page_start"] == 3} == {
        "1980-12-09"
    }
    assert {row["lecture_date"] for row in data.rows if row["page_start"] == 4} == {
        None
    }


def test_normalization_only_removes_matching_edge_page_number():
    assert (
        normalize_page(
            " 2\n\nLe corps    peut agir.\nUne seconde ligne.\n\n"
            "Un autre paragraphe.\n2",
            2,
        )
        == "Le corps peut agir. Une seconde ligne.\n\nUn autre paragraphe."
    )
    assert "123" in normalize_page("Le nombre 123 compte.", 2)


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (20, -1), (20, 20), (20, 21)])
def test_invalid_chunk_limits_are_rejected(size, overlap):
    with pytest.raises(ValueError, match="max_tokens"):
        ChunkConfig(size, overlap)


def test_existing_chunks_reject_different_settings(pipeline):
    data = pipeline(["Une joie synthétique."])
    with pytest.raises(ValueError, match="different source/settings"):
        chunk_pages(data.input, data.chunks, data.tokenizer, ChunkConfig(100, 20))


def test_corrupted_chunks_fail_checksum_verification(pipeline):
    data = pipeline(["Une joie synthétique."])
    (data.chunks / "chunks.jsonl").write_text("damaged")
    with pytest.raises(ValueError, match="checksum"):
        load_chunks(data.chunks)


def test_missing_chunk_directory_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_chunks(tmp_path / "missing")


def test_failed_extraction_pages_block_chunking(pipeline):
    with pytest.raises(ValueError, match="failed pages"):
        pipeline(["Une joie synthétique."], failed=True)
