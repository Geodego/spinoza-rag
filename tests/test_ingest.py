import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path

import pytest
from pypdf import PageObject, PdfReader, PdfWriter
from pypdf.errors import PdfReadError
from pypdf.generic import (
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
    NumberObject,
)

from spinoza_rag.ingest import extract_pdf, main, write_extraction


@pytest.fixture
def make_pdf(tmp_path):
    def create(pages, *, filename="source.pdf", metadata=None, password=None):
        writer = PdfWriter()
        for text in pages:
            page = writer.add_blank_page(width=612, height=792)
            if text is None:
                continue
            font = DictionaryObject(
                {
                    NameObject("/Type"): NameObject("/Font"),
                    NameObject("/Subtype"): NameObject("/Type1"),
                    NameObject("/BaseFont"): NameObject("/Helvetica"),
                    NameObject("/Encoding"): NameObject("/WinAnsiEncoding"),
                }
            )
            page[NameObject("/Resources")] = DictionaryObject(
                {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
            )
            stream = DecodedStreamObject()
            commands = [b"BT /F1 12 Tf 72 720 Td 14 TL"]
            for line in text.split("\n"):
                commands.append(
                    b"<" + line.encode("cp1252").hex().encode() + b"> Tj T*"
                )
            stream.set_data(b"\n".join([*commands, b"ET"]))
            page[NameObject("/Contents")] = stream
        if metadata:
            writer.add_metadata(metadata)
        if password:
            writer.encrypt(password, algorithm="RC4-128")
        path = tmp_path / filename
        writer.write(path)
        return path

    return create


def read_artifacts(output):
    pages = [
        json.loads(line) for line in (output / "pages.jsonl").read_text().splitlines()
    ]
    report = json.loads((output / "report.json").read_text())
    return pages, report


def test_extracts_french_text_metadata_and_physical_page_numbers(make_pdf):
    source = make_pdf(
        ["La joie augmente notre puissance d'agir.\nDeuxième ligne.", None, "Spinoza"],
        metadata={"/Title": "Séminaire sur Spinoza", "/Author": "Synthetic author"},
    )
    result = extract_pdf(source)
    assert (
        result.document_id
        == "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest()
    )
    assert result.source_file == "source.pdf"
    assert result.source_size_bytes == source.stat().st_size
    assert result.metadata["Title"] == "Séminaire sur Spinoza"
    assert result.metadata["Author"] == "Synthetic author"
    assert [page.page_number for page in result.pages] == [1, 2, 3]
    assert "Deuxième ligne." in result.pages[0].text
    assert "puissance d'agir" in result.pages[0].text
    assert result.pages[1].text == ""
    assert result.pages[1].issues == ("empty_text",)
    assert result.pages[2].issues == ("short_text",)
    assert source.exists()


def test_jsonl_report_match_and_identical_rerun_is_noop(make_pdf, tmp_path):
    source = make_pdf(["Spinoza et Deleuze", None])
    output = tmp_path / "results"
    document = extract_pdf(source)
    write_extraction(document, output)
    initial = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in output.iterdir()
    }
    write_extraction(extract_pdf(source), output)
    assert initial == {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in output.iterdir()
    }
    pages, report = read_artifacts(output)
    assert len(pages) == 2
    assert [page["page_number"] for page in pages] == [1, 2]
    assert all(page["document_id"] == document.document_id for page in pages)
    assert pages[0]["source_file"] == "source.pdf"
    assert pages[0]["text"] == document.pages[0].text
    assert report["document"]["metadata"] == document.metadata
    assert report["summary"]["total_pages"] == 2
    assert report["summary"]["empty_pages"] == 1
    assert report["summary"]["pages_with_text"] == 1
    assert (
        report["pages_jsonl_sha256"]
        == hashlib.sha256((output / "pages.jsonl").read_bytes()).hexdigest()
    )
    assert report["pages"][1]["issues"] == ["empty_text"]
    assert report["extraction"]["library"] == "pypdf"
    assert report["extraction"]["normalization"] == "line_endings_only"
    assert str(tmp_path) not in (output / "report.json").read_text()


def test_changed_source_preserves_existing_outputs(make_pdf, tmp_path):
    source = make_pdf(["First document"])
    output = tmp_path / "results"
    write_extraction(extract_pdf(source), output)
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    make_pdf(["Different document"], filename="source.pdf")
    with pytest.raises(FileExistsError, match="choose a new --output-dir"):
        write_extraction(extract_pdf(source), output)
    assert before == {path.name: path.read_bytes() for path in output.iterdir()}


@pytest.mark.parametrize(
    "contents", [{}, {"pages.jsonl": b"incomplete"}, {"notes.txt": b"keep me"}]
)
def test_existing_directory_is_not_overwritten(make_pdf, tmp_path, contents):
    output = tmp_path / "results"
    output.mkdir()
    for name, data in contents.items():
        (output / name).write_bytes(data)
    with pytest.raises(FileExistsError):
        write_extraction(extract_pdf(make_pdf(["Text"])), output)
    assert contents == {path.name: path.read_bytes() for path in output.iterdir()}


def test_failed_write_publishes_neither_file(make_pdf, tmp_path, monkeypatch):
    document = extract_pdf(make_pdf(["Synthetic text"]))
    output = tmp_path / "results"
    original = Path.write_bytes

    def fail_report(path, content):
        if path.name == "report.json":
            raise OSError("simulated disk failure")
        return original(path, content)

    monkeypatch.setattr(Path, "write_bytes", fail_report)
    with pytest.raises(OSError, match="disk failure"):
        write_extraction(document, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".results-*"))


def test_failed_publish_cleans_staging(make_pdf, tmp_path, monkeypatch):
    document = extract_pdf(make_pdf(["Synthetic text"]))
    output = tmp_path / "results"

    def fail_rename(*args):
        raise OSError("simulated publication failure")

    monkeypatch.setattr(Path, "rename", fail_rename)
    with pytest.raises(OSError, match="publication failure"):
        write_extraction(document, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".results-*"))


def test_page_failure_keeps_numbering_and_extracts_later_pages(
    make_pdf, tmp_path, monkeypatch
):
    source = make_pdf(["First", "Broken", "Third"])
    original = PageObject.extract_text
    calls = 0

    def extract(page, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise PdfReadError("synthetic damaged page")
        return original(page, *args, **kwargs)

    monkeypatch.setattr(PageObject, "extract_text", extract)
    output = tmp_path / "results"
    result = extract_pdf(source)
    write_extraction(result, output)
    pages, report = read_artifacts(output)
    assert [page["page_number"] for page in pages] == [1, 2, 3]
    assert "Third" in pages[2]["text"]
    assert pages[1]["text"] == ""
    assert pages[1]["diagnostics"]["issues"] == ["extraction_error"]
    assert "synthetic damaged page" in pages[1]["diagnostics"]["error"]
    assert report["summary"]["failed_pages"] == 1
    assert report["summary"]["empty_pages"] == 0


def test_parser_warnings_and_text_anomalies_are_reported(
    make_pdf, tmp_path, monkeypatch
):
    source = make_pdf(["Synthetic"])

    def extract(*args, **kwargs):
        logging.getLogger("pypdf").warning("synthetic recovery warning")
        return "\r\nÉthique\r\ufffd\x00\t"

    monkeypatch.setattr(PageObject, "extract_text", extract)
    result = extract_pdf(source, min_page_characters=0)
    page = result.pages[0]
    assert page.text == "\nÉthique\n\ufffd\x00\t"
    assert set(page.issues) == {
        "replacement_characters",
        "control_characters",
        "parser_warning",
    }
    assert page.parser_warnings == ("synthetic recovery warning",)
    assert result.parser_warnings == page.parser_warnings
    report = write_extraction(result, tmp_path / "results")
    assert report["summary"]["pages_needing_review"] == 1
    assert report["summary"]["parser_warning_count"] == 1


def test_image_only_pdf_is_flagged_and_cli_returns_failure(tmp_path):
    source = tmp_path / "scan.pdf"
    writer = PdfWriter()
    page = writer.add_blank_page(width=100, height=100)
    image = DecodedStreamObject()
    image.set_data(b"\xff\x00\x00")
    image.update(
        {
            NameObject("/Type"): NameObject("/XObject"),
            NameObject("/Subtype"): NameObject("/Image"),
            NameObject("/Width"): NumberObject(1),
            NameObject("/Height"): NumberObject(1),
            NameObject("/ColorSpace"): NameObject("/DeviceRGB"),
            NameObject("/BitsPerComponent"): NumberObject(8),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/XObject"): DictionaryObject({NameObject("/Im0"): image})}
    )
    stream = DecodedStreamObject()
    stream.set_data(b"q 100 0 0 100 0 0 cm /Im0 Do Q")
    page[NameObject("/Contents")] = stream
    writer.write(source)
    output = tmp_path / "results"
    assert main([str(source), "--output-dir", str(output)]) == 1
    pages, report = read_artifacts(output)
    assert pages[0]["diagnostics"]["issues"] == ["empty_text"]
    assert report["summary"]["pages_with_text"] == 0


@pytest.mark.parametrize("kind", ["missing", "malformed", "zero_pages", "encrypted"])
def test_document_errors_create_no_outputs(make_pdf, tmp_path, kind):
    source = tmp_path / "source.pdf"
    if kind == "malformed":
        source.write_bytes(b"not a PDF")
    elif kind == "zero_pages":
        source = make_pdf([])
    elif kind == "encrypted":
        source = make_pdf(["Secret synthetic text"], password="secret")
    output = tmp_path / "results"
    with pytest.raises(SystemExit) as error:
        main([str(source), "--output-dir", str(output)])
    assert error.value.code == 1
    assert not output.exists()


def test_short_page_threshold_can_be_disabled(make_pdf):
    source = make_pdf(["Short", None])
    assert extract_pdf(source, min_page_characters=0).pages[0].issues == ()
    assert extract_pdf(source, min_page_characters=0).pages[1].issues == ("empty_text",)
    with pytest.raises(ValueError, match="non-negative"):
        extract_pdf(source, min_page_characters=-1)


def test_source_fingerprint_changes_with_content_not_location(make_pdf, tmp_path):
    source = make_pdf(["Original"])
    first = extract_pdf(source)
    moved = tmp_path / "renamed.pdf"
    moved.write_bytes(source.read_bytes())
    assert extract_pdf(moved).document_id == first.document_id
    source = make_pdf(["Changed"])
    assert extract_pdf(source).document_id != first.document_id


def test_cli_defaults_and_rerun(make_pdf, tmp_path):
    (tmp_path / "data").mkdir()
    source = make_pdf(
        ["Synthetic lecture text with enough characters to pass the heuristic."]
    )
    source.rename(tmp_path / "data/spinoza.pdf")
    command = [sys.executable, "-I", "-m", "spinoza_rag.ingest"]
    for _ in range(2):
        result = subprocess.run(
            command, cwd=tmp_path, text=True, capture_output=True, check=False
        )
        assert result.returncode == 0, result.stderr
    pages, report = read_artifacts(tmp_path / "storage/ingestion")
    assert len(pages) == 1
    assert report["summary"]["total_pages"] == 1
    assert PdfReader(tmp_path / "data/spinoza.pdf").pages


def test_missing_metadata_is_allowed(make_pdf):
    source = make_pdf(["Synthetic text"])
    writer = PdfWriter(clone_from=source)
    writer.metadata = None
    writer.write(source)
    assert extract_pdf(source).metadata == {}


def test_metadata_failure_is_reported_without_losing_pages(make_pdf, monkeypatch):
    source = make_pdf(["Synthetic text"])

    def broken_metadata(reader):
        raise PdfReadError("synthetic bad metadata")

    monkeypatch.setattr(PdfReader, "metadata", property(broken_metadata))
    result = extract_pdf(source)
    assert "Synthetic text" in result.pages[0].text
    assert result.metadata == {}
    assert any("Metadata unavailable" in warning for warning in result.parser_warnings)


def test_cli_page_failure_writes_report_and_returns_nonzero(
    make_pdf, tmp_path, monkeypatch
):
    source = make_pdf(["Broken", "Usable synthetic text"])
    original = PageObject.extract_text
    calls = 0

    def extract(page, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise PdfReadError("synthetic page failure")
        return original(page, *args, **kwargs)

    monkeypatch.setattr(PageObject, "extract_text", extract)
    output = tmp_path / "results"
    assert main([str(source), "--output-dir", str(output)]) == 1
    pages, report = read_artifacts(output)
    assert "Usable synthetic text" in pages[1]["text"]
    assert report["summary"]["failed_pages"] == 1


def test_corrupt_output_is_preserved_and_not_reused(make_pdf, tmp_path):
    document = extract_pdf(make_pdf(["Synthetic text"]))
    output = tmp_path / "results"
    write_extraction(document, output)
    (output / "pages.jsonl").write_text("damaged output\n")
    with pytest.raises(FileExistsError):
        write_extraction(document, output)
    assert (output / "pages.jsonl").read_text() == "damaged output\n"


def test_symlink_output_is_rejected_without_touching_target(make_pdf, tmp_path):
    target = tmp_path / "keep"
    target.mkdir()
    (target / "notes.txt").write_text("keep")
    link = tmp_path / "results"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        write_extraction(extract_pdf(make_pdf(["Synthetic"])), link)
    assert list(target.iterdir()) == [target / "notes.txt"]
    assert (target / "notes.txt").read_text() == "keep"


def test_layout_mode_uses_spatial_order_and_records_setting(make_pdf, tmp_path):
    source = make_pdf(["Synthetic placeholder"])
    writer = PdfWriter(clone_from=source)
    stream = DecodedStreamObject()
    stream.set_data(
        b"BT /F1 12 Tf 72 24 Td (Synthetic footer) Tj ET\n"
        b"BT /F1 12 Tf 72 720 Td (Synthetic body at the top of the page.) Tj ET"
    )
    writer.pages[0][NameObject("/Contents")] = stream
    writer.write(source)
    plain = extract_pdf(source, extraction_mode="plain")
    layout = extract_pdf(source, extraction_mode="layout")
    assert plain.pages[0].text.index("footer") < plain.pages[0].text.index("body")
    assert layout.pages[0].text.index("body") < layout.pages[0].text.index("footer")
    assert layout.document_id == plain.document_id
    assert layout.pages[0].page_number == 1
    report = write_extraction(layout, tmp_path / "layout")
    assert report["extraction"]["mode"] == "layout"
    with pytest.raises(ValueError, match="extraction_mode"):
        extract_pdf(source, extraction_mode="unknown")


def test_cli_accepts_layout_mode(make_pdf, tmp_path):
    source = make_pdf(["A synthetic text page with enough characters for the check."])
    output = tmp_path / "layout"
    assert (
        main([str(source), "--extraction-mode", "layout", "--output-dir", str(output)])
        == 0
    )
    pages, report = read_artifacts(output)
    assert "synthetic text page" in pages[0]["text"]
    assert report["extraction"]["mode"] == "layout"
