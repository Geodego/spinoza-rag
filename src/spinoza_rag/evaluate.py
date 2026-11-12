"""Evaluate saved retrieval against human annotations without generating answers."""

import argparse
import hashlib
import json
import re
from collections.abc import Sequence
from datetime import UTC, date, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Literal, NotRequired, TypedDict, cast

import chromadb

from spinoza_rag.artifacts import JsonObject, json_bytes, publish_artifacts
from spinoza_rag.chunk import load_chunks
from spinoza_rag.config import EMBEDDING, EVALUATION, OLLAMA, PATHS
from spinoza_rag.embeddings import Embedder, OllamaClient
from spinoza_rag.index import IndexManifest, open_index
from spinoza_rag.ingest import load_extraction
from spinoza_rag.search import SearchResult, search_collection

_PROTOCOL = "page-blocks-v1"


class Reference(TypedDict):
    """One annotated supporting PDF page, not an exhaustive relevance judgment."""

    pdf_page: int
    lecture_date: NotRequired[str]
    annotation: NotRequired[str]


class Validation(TypedDict):
    """The human review attribution required for an approved question."""

    reviewer: str
    validated_on: str


class Question(TypedDict):
    """A French question and its proposed or human-validated supporting pages."""

    id: str
    question_fr: str
    references: list[Reference]
    status: Literal["proposed", "validated"]
    annotation: NotRequired[str]
    validation: NotRequired[Validation]


class Discovery(TypedDict):
    """The independent reference-discovery method and annotation date."""

    method: Literal[
        "keyword_search_and_direct_page_inspection",
        "two_pass_ai_keyword_and_direct_page_inspection",
    ]
    date: str
    annotation: str


class GoldSet(TypedDict):
    """A versioned, corpus-bound set containing no source passages."""

    schema_version: int
    version: str
    document_id: str
    pages_sha256: str
    discovery: Discovery
    questions: list[Question]


class RankedPage(TypedDict):
    """A unique page with a conservative rank and its first retrieving chunk."""

    pdf_page: int
    rank: int
    chunk_id: str


class Metrics(TypedDict):
    """Hit, annotated-support recall, and truncated reciprocal rank for one k."""

    hit: float
    recall: float
    mrr: float


class RankedChunk(TypedDict):
    """A retrieved chunk's identity, distance, and page range without its text."""

    chunk_id: str
    distance: float
    page_start: int
    page_end: int


class QuestionResult(TypedDict):
    """One question's annotations, retrieved provenance, and page metrics."""

    id: str
    question_fr: str
    annotation_status: Literal["proposed", "validated"]
    annotation: str | None
    validation: Validation | None
    references: list[Reference]
    ranked_pages: list[RankedPage]
    ranked_chunks: list[RankedChunk]
    retrieved_chunk_count: int
    unique_pages_in_chunk_pool: int
    metrics: dict[str, Metrics]


class Aggregate(TypedDict):
    """Macro metrics and membership count for one annotation stratum."""

    question_count: int
    metrics: dict[str, Metrics] | None


class OfficialAggregate(Aggregate):
    """Validated-only aggregate with an explicit availability flag."""

    available: bool


class EvaluationSetIdentity(TypedDict):
    """Exact annotation bytes, version, and bound corpus identity."""

    version: str
    sha256: str
    document_id: str
    pages_sha256: str


class RetrievalProtocol(TypedDict):
    """The candidate budget, cutoffs, ranking, and tie policy used for a run."""

    chunk_limit: int
    ks: list[int]
    ranking: str
    ties: str


class EvaluationReport(TypedDict):
    """A persisted evaluation run with provenance and separate metric strata."""

    protocol: str
    executed_at: str
    evaluation_set: EvaluationSetIdentity
    index_fingerprint: str
    index_manifest: IndexManifest
    chunking: JsonObject
    extraction: JsonObject
    runtime: dict[str, str]
    implementation_sha256: dict[str, str]
    retrieval: RetrievalProtocol
    official: OfficialAggregate
    exploratory_proposed: Aggregate
    excluded_proposed_count: int
    per_question: list[QuestionResult]
    limitations: list[str]


class EvaluationSummary(TypedDict):
    """The CLI's aggregate summary and location of the detailed report."""

    official: OfficialAggregate
    exploratory_proposed: Aggregate
    report: str


def _object(value: object, required: set[str], optional: set[str]) -> JsonObject:
    """Reject unknown fields as well as missing fields in annotation objects."""
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object; required fields are missing.")
    missing = required - value.keys()
    unknown = value.keys() - required - optional
    if missing or unknown:
        raise ValueError(
            f"Object fields: missing={sorted(missing)}, unknown={sorted(unknown)}."
        )
    return value


def _text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _date(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return date.fromisoformat(value).isoformat() == value
    except ValueError:
        return False


def _validate_question(value: object) -> Question:
    """Validate a complete reference set and require explicit human attribution."""
    row = _object(
        value,
        {"id", "question_fr", "references", "status"},
        {"annotation", "validation"},
    )
    if (
        not _text(row["id"])
        or not _text(row["question_fr"])
        or row["status"] not in ("proposed", "validated")
        or ("annotation" in row and not _text(row["annotation"]))
        or not isinstance(row["references"], list)
        or not row["references"]
    ):
        raise ValueError("Invalid question, status, or empty supporting references.")
    pages: set[int] = set()
    for reference in row["references"]:
        ref = _object(reference, {"pdf_page"}, {"lecture_date", "annotation"})
        page = ref["pdf_page"]
        if (
            type(page) is not int
            or page <= 0
            or page in pages
            or ("lecture_date" in ref and not _date(ref["lecture_date"]))
            or ("annotation" in ref and not _text(ref["annotation"]))
        ):
            raise ValueError("Invalid or duplicate supporting PDF page/date.")
        pages.add(page)
    if row["status"] == "validated":
        review = _object(row.get("validation"), {"reviewer", "validated_on"}, set())
        if not _text(review["reviewer"]) or not _date(review["validated_on"]):
            raise ValueError("Validated questions require a reviewer and review date.")
    elif "validation" in row:
        raise ValueError("Proposed questions must not carry a validation record.")
    # Every persisted Question field has been checked above.
    return cast(Question, row)


def load_gold_set(path: Path) -> GoldSet:
    """Load strict, versioned annotations without changing their approval status."""
    try:
        value = _object(
            json.loads(path.read_bytes()),
            {
                "schema_version",
                "version",
                "document_id",
                "pages_sha256",
                "discovery",
                "questions",
            },
            set(),
        )
        if (
            type(value["schema_version"]) is not int
            or value["schema_version"] != 1
            or not _text(value["version"])
            or not isinstance(value["document_id"], str)
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", value["document_id"])
            or not isinstance(value["pages_sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", value["pages_sha256"])
            or not isinstance(value["questions"], list)
            or not value["questions"]
        ):
            raise ValueError("Invalid schema, version, corpus identity, or questions.")
        discovery = _object(value["discovery"], {"method", "date", "annotation"}, set())
        if (
            discovery["method"]
            not in (
                "keyword_search_and_direct_page_inspection",
                "two_pass_ai_keyword_and_direct_page_inspection",
            )
            or not _date(discovery["date"])
            or not _text(discovery["annotation"])
        ):
            raise ValueError("Require independent keyword/page-inspection discovery.")
        questions = [_validate_question(row) for row in value["questions"]]
        if len({row["id"] for row in questions}) != len(questions):
            raise ValueError("Duplicate question IDs.")
        # All GoldSet fields and nested records have been validated.
        return cast(GoldSet, value)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid gold set: {error}") from error


def _page_range(result: SearchResult) -> tuple[int, int]:
    """Return a checked inclusive PDF range without coercing provenance values."""
    start, end = result["metadata"]["page_start"], result["metadata"]["page_end"]
    if type(start) is not int or type(end) is not int or not 1 <= start <= end:
        raise ValueError("Invalid retrieved PDF page range.")
    return start, end


def _chunk_summary(result: SearchResult) -> RankedChunk:
    """Retain retrieved identity and provenance while excluding corpus text."""
    start, end = _page_range(result)
    return {
        "chunk_id": result["chunk_id"],
        "distance": result["distance"],
        "page_start": start,
        "page_end": end,
    }


def rank_pages(results: Sequence[SearchResult]) -> list[RankedPage]:
    """Deduplicate pages, charging a multi-page chunk its full new-page budget.

    Preserve chunk order. All previously unseen pages of a chunk form an
    indivisible block, assigned the block's ending rank. Thus a chunk spanning
    pages 8–10 cannot score any of them at k=1; all three cost three page slots.
    Overlapping or repeated pages never receive a second rank or credit.
    """
    seen: set[int] = set()
    ranked: list[RankedPage] = []
    for result in results:
        start, end = _page_range(result)
        new_pages = [page for page in range(start, end + 1) if page not in seen]
        rank = len(seen) + len(new_pages)
        ranked.extend(
            {"pdf_page": page, "rank": rank, "chunk_id": result["chunk_id"]}
            for page in new_pages
        )
        seen.update(new_pages)
    return ranked


def score_pages(
    ranked: Sequence[RankedPage], supporting_pages: set[int], k: int
) -> Metrics:
    """Score a page prefix against non-exhaustive annotated supporting pages."""
    if not supporting_pages or k <= 0:
        raise ValueError("Require supporting pages and a positive cutoff.")
    matched = {
        row["pdf_page"]: row["rank"]
        for row in ranked
        if row["rank"] <= k and row["pdf_page"] in supporting_pages
    }
    return {
        "hit": float(bool(matched)),
        "recall": len(matched) / len(supporting_pages),
        "mrr": 1 / min(matched.values()) if matched else 0.0,
    }


def _aggregate(rows: list[QuestionResult]) -> Aggregate:
    """Return macro averages or null metrics when the stratum has no questions."""
    if not rows:
        return {"question_count": 0, "metrics": None}
    return {
        "question_count": len(rows),
        "metrics": {
            str(k): {
                "hit": sum(row["metrics"][str(k)]["hit"] for row in rows) / len(rows),
                "recall": sum(row["metrics"][str(k)]["recall"] for row in rows)
                / len(rows),
                "mrr": sum(row["metrics"][str(k)]["mrr"] for row in rows) / len(rows),
            }
            for k in EVALUATION.ks
        },
    }


def evaluate(
    gold_set_path: Path,
    index_dir: Path,
    embedder: Embedder,
    *,
    executed_at: datetime,
    include_proposed: bool = False,
    chunk_limit: int = EVALUATION.chunk_limit,
) -> EvaluationReport:
    """Reuse search and its saved index; embed only selected evaluation questions.

    Official results always exclude proposed annotations. Refuse an official
    run with no validated questions before contacting the embedding service.
    The supplied timestamp makes tests independent of the wall clock.
    """
    if chunk_limit < max(EVALUATION.ks):
        raise ValueError("chunk_limit must be at least the largest page cutoff.")
    if executed_at.tzinfo is None or executed_at.utcoffset() is None:
        raise ValueError("executed_at must be timezone-aware.")
    raw = gold_set_path.read_bytes()
    gold = load_gold_set(gold_set_path)
    selected = [
        row
        for row in gold["questions"]
        if row["status"] == "validated" or include_proposed
    ]
    if not selected:
        raise ValueError(
            "No validated questions: official baseline unavailable. "
            "Validate references first, or use --include-proposed for exploration."
        )
    with open_index(index_dir, embedder) as (collection, manifest):
        _, chunks = load_chunks(index_dir / manifest["chunks_dir"])
        document, _ = load_extraction(
            index_dir / manifest["chunks_dir"] / chunks["extraction_dir"]
        )
        if (
            gold["document_id"] != document.document_id
            or gold["pages_sha256"] != chunks["pages_sha256"]
        ):
            raise ValueError("Gold set corpus fingerprint differs from the index.")
        if any(
            ref["pdf_page"] > len(document.pages)
            for question in gold["questions"]
            for ref in question["references"]
        ):
            raise ValueError("Supporting PDF page is outside the source document.")
        fingerprint = hashlib.sha256(json_bytes(manifest["configuration"])).hexdigest()
        per_question: list[QuestionResult] = []
        for question in selected:
            results = search_collection(
                question["question_fr"],
                collection,
                manifest,
                embedder,
                top_k=chunk_limit,
            )
            ranked = rank_pages(results)
            support = {ref["pdf_page"] for ref in question["references"]}
            per_question.append(
                {
                    "id": question["id"],
                    "question_fr": question["question_fr"],
                    "annotation_status": question["status"],
                    "annotation": question.get("annotation"),
                    "validation": question.get("validation"),
                    "references": question["references"],
                    "ranked_pages": ranked,
                    "ranked_chunks": [_chunk_summary(row) for row in results],
                    "retrieved_chunk_count": len(results),
                    "unique_pages_in_chunk_pool": len(ranked),
                    "metrics": {
                        str(k): score_pages(ranked, support, k) for k in EVALUATION.ks
                    },
                }
            )
    if (
        gold_set_path.read_bytes() != raw
        or json.loads((index_dir / "manifest.json").read_bytes()) != manifest
    ):
        raise ValueError("Gold set or index changed during evaluation; rerun.")
    official = [r for r in per_question if r["annotation_status"] == "validated"]
    proposed = [r for r in per_question if r["annotation_status"] == "proposed"]
    return {
        "protocol": _PROTOCOL,
        "executed_at": executed_at.astimezone(UTC).isoformat(),
        "evaluation_set": {
            "version": gold["version"],
            "sha256": hashlib.sha256(raw).hexdigest(),
            "document_id": gold["document_id"],
            "pages_sha256": gold["pages_sha256"],
        },
        "index_fingerprint": fingerprint,
        "index_manifest": manifest,
        "chunking": chunks["chunking"],
        "extraction": chunks["extraction"],
        "runtime": {
            "chromadb": version("chromadb"),
            "spinoza-rag": version("spinoza-rag"),
        },
        "implementation_sha256": {
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in (
                "evaluate.py",
                "search.py",
                "config.py",
                "index.py",
                "embeddings.py",
            )
        },
        "retrieval": {
            "chunk_limit": chunk_limit,
            "ks": list(EVALUATION.ks),
            "ranking": "chunk order; unique page blocks at block-ending rank",
            "ties": "preserve Chroma order; no cross-run tie guarantee",
        },
        "official": {"available": bool(official), **_aggregate(official)},
        "exploratory_proposed": _aggregate(proposed),
        "excluded_proposed_count": sum(
            r["status"] == "proposed" for r in gold["questions"]
        )
        - len(proposed),
        "per_question": per_question,
        "limitations": [
            "Recall is annotated-support coverage, not exhaustive relevance recall.",
            "Unannotated pages are unjudged; a miss is not proof of irrelevance.",
            "Page credit does not establish that the chunk contains evidence.",
            "Hit/MRR do not prove all synthesis evidence was retrieved.",
            "A fixed chunk pool can yield fewer unique pages than k.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Publish a corpus-text-free JSON report, preserving existing run artifacts."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--gold-set",
        type=Path,
        default=PATHS.gold_set,
        help="versioned questions and proposed/validated supporting pages",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=PATHS.index_dir,
        help="complete persistent index with its upstream artifacts",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PATHS.evaluation_dir,
        help="new report directory; existing paths are rejected before embedding",
    )
    parser.add_argument(
        "--chunk-limit",
        type=int,
        default=EVALUATION.chunk_limit,
        help="fixed candidate-chunk pool before page deduplication",
    )
    parser.add_argument(
        "--include-proposed",
        action="store_true",
        help="also report proposed questions in a separate exploratory stratum",
    )
    parser.add_argument(
        "--model",
        default=EMBEDDING.model,
        help="installed Ollama embedding model matching the index",
    )
    parser.add_argument("--ollama-url", default=OLLAMA.url, help="loopback Ollama URL")
    parser.add_argument(
        "--timeout",
        type=float,
        default=OLLAMA.timeout_seconds,
        help="seconds per Ollama request",
    )
    args = parser.parse_args(argv)
    try:
        if args.output_dir.is_symlink() or args.output_dir.exists():
            raise FileExistsError(
                f"{args.output_dir} already exists; choose a new --output-dir."
            )
        report = evaluate(
            args.gold_set,
            args.index_dir,
            OllamaClient(args.model, args.ollama_url, args.timeout),
            executed_at=datetime.now(UTC),
            include_proposed=args.include_proposed,
            chunk_limit=args.chunk_limit,
        )
        publish_artifacts(args.output_dir, {"report.json": json_bytes(report)})
        summary: EvaluationSummary = {
            "official": report["official"],
            "exploratory_proposed": report["exploratory_proposed"],
            "report": str(args.output_dir / "report.json"),
        }
        print(json.dumps(summary, sort_keys=True))
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        chromadb.errors.ChromaError,
    ) as error:
        parser.exit(1, f"Evaluation failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
