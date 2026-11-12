"""Expose context retrieval and bounded French answers as local CLI modes."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from spinoza_rag.agent import Event, research
from spinoza_rag.chat import local_chat
from spinoza_rag.config import AGENT, EMBEDDING, OLLAMA, PATHS, SEARCH
from spinoza_rag.embeddings import OllamaClient
from spinoza_rag.search import SearchResult, search


def main(argv: Sequence[str] | None = None) -> int:
    """Print evidence, tool activity, and timings without persisting a conversation."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("mode", choices=["context", "answer"])
    parser.add_argument("question")
    parser.add_argument("--index-dir", type=Path, default=PATHS.index_dir)
    parser.add_argument("--top-k", type=int, default=SEARCH.top_k)
    parser.add_argument("--context-bytes", type=int, default=AGENT.context_bytes)
    parser.add_argument("--model", default=AGENT.model, help="installed chat model")
    parser.add_argument("--embedding-model", default=EMBEDDING.model)
    parser.add_argument("--ollama-url", default=OLLAMA.url)
    parser.add_argument("--timeout", type=float, default=OLLAMA.timeout_seconds)
    args = parser.parse_args(argv)

    def display(event: Event) -> None:
        print(json.dumps(event, ensure_ascii=False), flush=True)

    try:
        if args.top_k <= 0 or not 0 < args.context_bytes <= AGENT.context_bytes:
            raise ValueError(
                "Require top_k > 0 and context bytes within the default cap."
            )
        embedder = OllamaClient(args.embedding_model, args.ollama_url, args.timeout)

        def retrieve(query: str) -> list[SearchResult]:
            return search(query, args.index_dir, embedder, top_k=args.top_k)

        model = (
            local_chat(args.model, args.ollama_url, args.timeout)
            if args.mode == "answer"
            else None
        )
        result = research(
            args.question,
            retrieve,
            model=model,
            context_bytes=args.context_bytes,
            observer=display,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as error:
        parser.exit(1, f"Research failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
