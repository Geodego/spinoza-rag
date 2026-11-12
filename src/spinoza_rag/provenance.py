"""Assign per-run source labels and bound serialized evidence sent to a model."""

import json
import re

from spinoza_rag.config import AGENT
from spinoza_rag.search import SearchResult


class Source(SearchResult):
    """A retrieved chunk with a stable short label within one run."""

    label: str


class Evidence:
    """Keep a deduplicated source registry within a UTF-8 JSON byte budget.

    Admit only complete passages. The budget includes serialized labels and
    metadata; repeated chunks are not resent. Labels resolve to exactly the
    text made available to the model, never a hidden full passage.
    """

    def __init__(self, budget: int = AGENT.context_bytes) -> None:
        if budget <= 0:
            raise ValueError("Context budget must be positive.")
        self.budget = budget
        self.used_bytes = 0
        self.sources: list[Source] = []

    def add(self, results: list[SearchResult]) -> str:
        """Return newly admitted evidence as JSON, charging its full encoding."""
        admitted: list[Source] = []
        for result in results:
            if any(s["chunk_id"] == result["chunk_id"] for s in self.sources):
                continue
            source: Source = {**result, "label": f"S{len(self.sources) + 1}"}
            candidate = json.dumps([*admitted, source], ensure_ascii=False)
            if self.used_bytes + len(candidate.encode("utf-8")) > self.budget:
                continue
            admitted.append(source)
            self.sources.append(source)
        payload = json.dumps(admitted, ensure_ascii=False)
        # An empty result has no evidence and is not charged to the evidence budget.
        if admitted:
            self.used_bytes += len(payload.encode("utf-8"))
        return payload

    def unknown_citations(self, answer: str) -> list[str]:
        """Return unresolved bracketed source labels, without judging entailment."""
        known = {source["label"] for source in self.sources}
        return sorted(set(re.findall(r"\[(S[^\]\s]*)\]", answer)) - known)
