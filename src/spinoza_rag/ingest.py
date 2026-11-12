"""Extract a local PDF into page-aware JSONL and quality diagnostics.

Run with ``python -m spinoza_rag.ingest``. Extraction preserves page boundaries
and text content; chunking, OCR, and indexing belong to later stages.
"""

import argparse
import hashlib
import json
import logging
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import pypdf
from pypdf import PdfReader
from pypdf.errors import PyPdfError

from spinoza_rag.artifacts import publish_artifacts
from spinoza_rag.config import EXTRACTION, PATHS, ExtractionMode

# Bump when the output format or extraction behavior changes.
_SCHEMA_VERSION = 1

_PAGE_ERRORS = (
    PyPdfError,
    ValueError,
    TypeError,
    KeyError,
    IndexError,
    NotImplementedError,
)
_METADATA_FIELDS = (
    "Title",
    "Author",
    "Subject",
    "Keywords",
    "Creator",
    "Producer",
    "CreationDate",
    "ModDate",
)


@dataclass(frozen=True)
class ExtractedPage:
    """Text and review signals for a physical, one-based PDF page."""

    page_number: int
    text: str
    issues: tuple[str, ...]
    parser_warnings: tuple[str, ...]
    error: str | None = None


@dataclass(frozen=True)
class ExtractedDocument:
    """A source fingerprint, PDF metadata, and ordered extraction results."""

    document_id: str
    source_file: str
    source_size_bytes: int
    metadata: dict[str, str]
    pages: tuple[ExtractedPage, ...]
    parser_warnings: tuple[str, ...]
    min_page_characters: int
    extraction_mode: ExtractionMode


class _PdfWarnings(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def _text_issues(text: str, min_page_characters: int) -> list[str]:
    issues = []
    non_whitespace = sum(not character.isspace() for character in text)
    if not non_whitespace:
        issues.append("empty_text")
    elif non_whitespace < min_page_characters:
        issues.append("short_text")
    if "\ufffd" in text:
        issues.append("replacement_characters")
    if any(
        unicodedata.category(character) == "Cc" and character not in "\n\t"
        for character in text
    ):
        issues.append("control_characters")
    return issues


def extract_pdf(
    source: Path,
    *,
    min_page_characters: int = EXTRACTION.min_page_characters,
    extraction_mode: ExtractionMode = EXTRACTION.mode,
) -> ExtractedDocument:
    """Extract a PDF snapshot without writing files or discarding problem pages.

    Only CRLF/CR line endings are normalized. Metadata dates remain raw PDF
    strings. Parser recovery warnings are retained, including per-page warnings.

    Args:
        source: A locally provided PDF; only its basename is stored in artifacts.
        min_page_characters: Non-whitespace character threshold for short pages.
            Zero disables this heuristic, but empty pages are always flagged.
        extraction_mode: Plain content-stream order, or spatial layout order.
            Layout mode may improve word spacing but preserves extra whitespace.

    Raises:
        OSError: The source cannot be read.
        ValueError: Settings are invalid, or the PDF is encrypted or empty.
        PyPdfError: The PDF cannot be parsed at the document level.
    """
    if min_page_characters < 0:
        raise ValueError("min_page_characters must be non-negative")
    if extraction_mode not in ("plain", "layout"):
        raise ValueError("extraction_mode must be plain or layout")
    source = Path(source)
    data = source.read_bytes()
    collector = _PdfWarnings()
    logger = logging.getLogger("pypdf")
    logger.addHandler(collector)
    try:
        with BytesIO(data) as stream:
            reader = PdfReader(stream, strict=False)
            if reader.is_encrypted:
                raise ValueError(
                    "Encrypted PDFs are not supported; provide a decrypted copy."
                )
            page_count = len(reader.pages)
            if not page_count:
                raise ValueError("The PDF contains no pages.")
            metadata: dict[str, str] = {}
            try:
                info = reader.metadata or {}
                metadata = {
                    name: str(info[f"/{name}"])
                    for name in _METADATA_FIELDS
                    if info.get(f"/{name}") is not None
                }
            except _PAGE_ERRORS as error:
                collector.messages.append(
                    f"Metadata unavailable: {type(error).__name__}: {error}"
                )

            pages = []
            for index in range(page_count):
                warning_start = len(collector.messages)
                text = ""
                failure = None
                try:
                    text = (
                        reader.pages[index].extract_text(
                            extraction_mode=extraction_mode
                        )
                        or ""
                    )
                    text = text.replace("\r\n", "\n").replace("\r", "\n")
                    issues = _text_issues(text, min_page_characters)
                except _PAGE_ERRORS as error:
                    failure = f"{type(error).__name__}: {error}"
                    issues = ["extraction_error"]
                page_warnings = tuple(collector.messages[warning_start:])
                if page_warnings:
                    issues.append("parser_warning")
                pages.append(
                    ExtractedPage(
                        index + 1, text, tuple(issues), page_warnings, failure
                    )
                )
    finally:
        logger.removeHandler(collector)
    return ExtractedDocument(
        document_id="sha256:" + hashlib.sha256(data).hexdigest(),
        source_file=source.name,
        source_size_bytes=len(data),
        metadata=metadata,
        pages=tuple(pages),
        parser_warnings=tuple(collector.messages),
        min_page_characters=min_page_characters,
        extraction_mode=extraction_mode,
    )


def _page_diagnostics(page: ExtractedPage) -> dict[str, object]:
    return {
        "page_number": page.page_number,
        "character_count": len(page.text),
        "non_whitespace_character_count": sum(not char.isspace() for char in page.text),
        "word_count": len(page.text.split()),
        "issues": list(page.issues),
        "parser_warnings": list(page.parser_warnings),
        "error": page.error,
    }


def _extraction_settings(
    min_page_characters: int, extraction_mode: ExtractionMode
) -> dict[str, object]:
    return {
        "library": "pypdf",
        "version": pypdf.__version__,
        "mode": extraction_mode,
        "normalization": "line_endings_only",
        "min_page_characters": min_page_characters,
    }


def _serialize(document: ExtractedDocument) -> tuple[bytes, dict[str, object]]:
    rows = [
        {
            "schema_version": _SCHEMA_VERSION,
            "document_id": document.document_id,
            "source_file": document.source_file,
            "page_number": page.page_number,
            "text": page.text,
            "diagnostics": _page_diagnostics(page),
        }
        for page in document.pages
    ]
    pages_jsonl = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")
    report: dict[str, object] = {
        "schema_version": _SCHEMA_VERSION,
        "document": {
            "document_id": document.document_id,
            "source_file": document.source_file,
            "source_size_bytes": document.source_size_bytes,
            "metadata": document.metadata,
        },
        "extraction": _extraction_settings(
            document.min_page_characters, document.extraction_mode
        ),
        "summary": {
            "total_pages": len(document.pages),
            "pages_with_text": sum(bool(page.text.strip()) for page in document.pages),
            "empty_pages": sum("empty_text" in page.issues for page in document.pages),
            "failed_pages": sum(page.error is not None for page in document.pages),
            "pages_needing_review": sum(bool(page.issues) for page in document.pages),
            "total_characters": sum(len(page.text) for page in document.pages),
            "parser_warning_count": len(document.parser_warnings),
        },
        "parser_warnings": list(document.parser_warnings),
        "pages": [_page_diagnostics(page) for page in document.pages],
        "pages_jsonl_sha256": hashlib.sha256(pages_jsonl).hexdigest(),
    }
    return pages_jsonl, report


def load_extraction(output_dir: Path) -> tuple[ExtractedDocument, dict[str, object]]:
    """Read verified page artifacts without opening or re-extracting the PDF.

    The recorded extractor version may differ from the installed one. Integrity,
    provenance, diagnostics, and the supported schema are still validated.
    """
    try:
        if output_dir.is_symlink() or not output_dir.is_dir():
            raise ValueError("expected a real extraction directory")
        if {p.name for p in output_dir.iterdir()} != {"pages.jsonl", "report.json"}:
            raise ValueError("expected exactly pages.jsonl and report.json")
        if any(
            (output_dir / name).is_symlink() for name in ("pages.jsonl", "report.json")
        ):
            raise ValueError("artifact files must not be symlinks")
        report = json.loads((output_dir / "report.json").read_bytes())
        if (
            type(report["schema_version"]) is not int
            or report["schema_version"] != _SCHEMA_VERSION
        ):
            raise ValueError("unsupported output schema")
        settings = report["extraction"]
        if (
            settings["library"] != "pypdf"
            or not isinstance(settings["version"], str)
            or settings["mode"] not in ("plain", "layout")
            or settings["normalization"] != "line_endings_only"
            or type(settings["min_page_characters"]) is not int
            or settings["min_page_characters"] < 0
        ):
            raise ValueError("invalid extraction settings")
        identity = report["document"]
        if not isinstance(identity["metadata"], dict) or any(
            not isinstance(k, str) or not isinstance(v, str)
            for k, v in identity["metadata"].items()
        ):
            raise ValueError("invalid document metadata")
        raw = (output_dir / "pages.jsonl").read_bytes()
        if report["pages_jsonl_sha256"] != hashlib.sha256(raw).hexdigest():
            raise ValueError("JSONL checksum mismatch")
        rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
        if not rows or any(
            not isinstance(row["text"], str)
            or row["schema_version"] != _SCHEMA_VERSION
            or type(row["page_number"]) is not int
            or row["page_number"] != number
            or row["document_id"] != identity["document_id"]
            or row["source_file"] != identity["source_file"]
            for number, row in enumerate(rows, start=1)
        ):
            raise ValueError("invalid page sequence or provenance")
        document = ExtractedDocument(
            document_id=identity["document_id"],
            source_file=identity["source_file"],
            source_size_bytes=identity["source_size_bytes"],
            metadata=identity["metadata"],
            pages=tuple(
                ExtractedPage(
                    row["page_number"],
                    row["text"],
                    tuple(row["diagnostics"]["issues"]),
                    tuple(row["diagnostics"]["parser_warnings"]),
                    row["diagnostics"]["error"],
                )
                for row in rows
            ),
            parser_warnings=tuple(report["parser_warnings"]),
            min_page_characters=settings["min_page_characters"],
            extraction_mode=settings["mode"],
        )
        expected_raw, expected_report = _serialize(document)
        expected_report["extraction"] = settings
        if expected_raw != raw or expected_report != report:
            raise ValueError("page diagnostics or report are inconsistent")
        return document, report
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid extraction artifacts in {output_dir}: {error}"
        ) from error


def _load_existing_extraction(
    source: Path,
    output_dir: Path,
    min_page_characters: int,
    extraction_mode: ExtractionMode,
) -> dict[str, object] | None:
    if output_dir.is_symlink():
        raise ValueError("The output directory must not be a symlink.")
    if not output_dir.exists():
        return None
    try:
        document, report = load_extraction(output_dir)
        if report["extraction"] != _extraction_settings(
            min_page_characters, extraction_mode
        ):
            raise ValueError("extractor version or settings changed")
        data = source.read_bytes()
        if (
            document.document_id != "sha256:" + hashlib.sha256(data).hexdigest()
            or document.source_file != source.name
            or document.source_size_bytes != len(data)
        ):
            raise ValueError("source PDF changed")
        return report
    except ValueError as error:
        raise FileExistsError(
            f"{output_dir} cannot be reused ({error}); choose a new --output-dir. "
            "Existing files were preserved."
        ) from error


def write_extraction(
    document: ExtractedDocument, output_dir: Path
) -> dict[str, object]:
    """Publish pages.jsonl and report.json together, or reuse identical output.

    A sibling temporary directory is renamed into place only after both files
    are written. Existing, different artifacts are never overwritten or merged.
    This component is intended for sequential local runs, not concurrent writers.

    Raises:
        FileExistsError: The destination exists with different or incomplete data.
        ValueError: The destination is a symlink.
        OSError: Artifacts cannot be written or published.
    """
    output_dir = Path(output_dir)
    if output_dir.is_symlink():
        raise ValueError("The output directory must not be a symlink.")
    pages_jsonl, report = _serialize(document)
    artifacts = {
        "pages.jsonl": pages_jsonl,
        "report.json": (
            json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        ).encode("utf-8"),
    }
    publish_artifacts(output_dir, artifacts)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Run local ingestion; return nonzero for failures or unusable extraction."""
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "source",
        nargs="?",
        type=Path,
        default=PATHS.source_pdf,
        help="local PDF to extract",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PATHS.pages_dir,
        help="destination for pages.jsonl and report.json",
    )
    parser.add_argument(
        "--min-page-characters",
        type=int,
        default=EXTRACTION.min_page_characters,
        help="non-whitespace threshold for short_text; 0 disables this heuristic",
    )
    parser.add_argument(
        "--extraction-mode",
        choices=("plain", "layout"),
        default=EXTRACTION.mode,
        help="plain content-stream order or spatial layout order",
    )
    args = parser.parse_args(argv)
    if args.min_page_characters < 0:
        parser.error("--min-page-characters must be non-negative")
    try:
        report = _load_existing_extraction(
            args.source, args.output_dir, args.min_page_characters, args.extraction_mode
        )
        if report is None:
            document = extract_pdf(
                args.source,
                min_page_characters=args.min_page_characters,
                extraction_mode=args.extraction_mode,
            )
            report = write_extraction(document, args.output_dir)
        else:
            print("Reusing verified artifacts; PDF extraction skipped.")
    except (OSError, *_PAGE_ERRORS) as error:
        parser.exit(1, f"Ingestion failed: {error}\n")
    summary = report["summary"]
    print(json.dumps(summary, sort_keys=True))
    print(
        f"Artifacts: {args.output_dir / 'pages.jsonl'} and "
        f"{args.output_dir / 'report.json'}"
    )
    if summary["failed_pages"] or not summary["pages_with_text"]:
        print(
            "Review required: page extraction failed or no extractable text was found."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
