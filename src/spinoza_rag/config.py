"""Central defaults for pipeline settings, storage paths, and local services.

These are defaults, not records of past runs. CLI flags override them, library
functions accept explicit values, and each artifact manifest records the
settings actually used. Several values enter configuration fingerprints:
changing them makes existing artifacts incompatible, and the next run then
requires a new output directory instead of silently mixing configurations.

Implementation revisions, such as schema versions or the normalization
revision, stay with the code that implements them. Changing them is a
behavior change, not a setting.

Run ``python -m spinoza_rag.config`` to print every default.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

ExtractionMode = Literal["plain", "layout"]


@dataclass(frozen=True)
class Paths:
    """Default artifact locations; each stage's output is the next stage's input."""

    source_pdf: Path = Path("data/spinoza.pdf")
    pages_dir: Path = Path("storage/ingestion-layout")
    chunks_dir: Path = Path("storage/chunks")
    index_dir: Path = Path("storage/index")
    gold_set: Path = Path("evaluation/gold.v1.json")
    evaluation_dir: Path = Path("storage/evaluations/baseline")
    research_dir: Path = Path("storage/research")
    tokenizer: Path = Path("models/qwen3-embedding-tokenizer.json")


@dataclass(frozen=True)
class ExtractionSettings:
    """PDF extraction defaults, recorded in ``report.json``.

    Layout mode fixes the word splits that plain mode produced on the lecture
    PDF. Plain mode remains available with ``--extraction-mode plain``.
    """

    mode: ExtractionMode = "layout"
    min_page_characters: int = 40


@dataclass(frozen=True)
class ChunkConfig:
    """Token limits; overlap is a target constrained by recursive separators.

    Field names and values enter chunk IDs and the chunk manifest fingerprint.
    """

    max_tokens: int = 450
    overlap_tokens: int = 70

    def __post_init__(self) -> None:
        """Reject invalid token limits before processing input."""
        if self.max_tokens <= 0 or not 0 <= self.overlap_tokens < self.max_tokens:
            raise ValueError(
                "Require max_tokens > 0 and 0 <= overlap_tokens < max_tokens."
            )


@dataclass(frozen=True)
class OllamaSettings:
    """Loopback Ollama service defaults; not part of any fingerprint."""

    url: str = "http://127.0.0.1:11434"
    timeout_seconds: float = 180


@dataclass(frozen=True)
class EmbeddingSettings:
    """Embedding model and query prompt; both enter the index fingerprint.

    The model tag is resolved to its installed digest at run time, so changed
    weights behind the same tag are also detected.
    """

    model: str = "qwen3-embedding:0.6b"
    query_instruction: str = (
        "Given a question about Deleuze's lectures on Spinoza, "
        "retrieve relevant lecture passages that answer the question."
    )


@dataclass(frozen=True)
class IndexSettings:
    """Index defaults; the collection name enters the index fingerprint."""

    collection: str = "spinoza"
    batch_size: int = 8


@dataclass(frozen=True)
class SearchSettings:
    """Retrieval defaults; not persisted."""

    top_k: int = 5


@dataclass(frozen=True)
class EvaluationSettings:
    """Page cutoffs and the fixed candidate-chunk budget for evaluation."""

    ks: tuple[int, ...] = (1, 3, 5, 10)
    chunk_limit: int = 50


@dataclass(frozen=True)
class AgentSettings:
    """Bound local research runs independently of model instructions."""

    model: str = "qwen3:8b"
    max_search_calls: int = 2
    max_model_calls: int = 3
    recursion_limit: int = 12
    context_bytes: int = 8000
    question_bytes: int = 2000
    query_bytes: int = 1000
    num_ctx: int = 16384
    num_predict: int = 700
    temperature: float = 0


AGENT = AgentSettings()
PATHS = Paths()
EXTRACTION = ExtractionSettings()
CHUNKING = ChunkConfig()
OLLAMA = OllamaSettings()
EMBEDDING = EmbeddingSettings()
INDEX = IndexSettings()
SEARCH = SearchSettings()
EVALUATION = EvaluationSettings()


def defaults() -> dict[str, dict[str, object]]:
    """Return all defaults as JSON-compatible values, grouped by area."""
    groups: dict[str, DataclassInstance] = {
        "paths": PATHS,
        "extraction": EXTRACTION,
        "chunking": CHUNKING,
        "ollama": OLLAMA,
        "embedding": EMBEDDING,
        "index": INDEX,
        "search": SEARCH,
        "evaluation": EVALUATION,
        "agent": AGENT,
    }
    return {
        name: {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(group).items()
        }
        for name, group in groups.items()
    }


def main() -> int:
    """Print every default so the effective configuration is easy to inspect."""
    print(json.dumps(defaults(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
