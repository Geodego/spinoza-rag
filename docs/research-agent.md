# Local research agent

## Commands

Run from the repository root after building the existing index and starting
Ollama with `qwen3:8b` and `qwen3-embedding:0.6b` already installed; see
[one-time setup](../README.md#one-time-setup):

```sh
uv run --locked python -m spinoza_rag.research context \
  'Selon les passages, quels éléments composent le premier genre de connaissance ? Limite la réponse à cette définition.' --top-k 1
uv run --locked python -m spinoza_rag.research answer \
  'Selon les passages, quels éléments composent le premier genre de connaissance ? Limite la réponse à cette définition.' --top-k 1 --timeout 600
```

`context` retrieves passages and provenance without constructing a chat model.
It still uses Ollama **embeddings**. `answer` asks the local model to choose
queries, optionally gather follow-up evidence, and answer in French with `[S1]`
citations. The `spinoza_rag.search` CLI stays retrieval-only. `--help` lists
index, model, retrieval, timeout and evidence limits. The timeout applies to
each request, not to the whole run.

The [answer review log](answer-review.md#observed-behavior) includes the
source-supported definition case used above, alongside broader questions whose
answers exceeded their evidence. Repeated runs can produce different output.

## Reading output and errors

On a successful research run, **stdout is a stream of JSON objects**, not one
JSON document: each progress event occupies one line, followed by the final
result object formatted across multiple lines. The final object includes
`answer`, `sources`, `events`, `evidence_bytes`, and `elapsed_seconds`; its
`events` list repeats the progress events. Context mode sets `answer` to `null`
and emits search events without model events.

The shortened example below shows selected fields from an answer run; other
fields and events are omitted. All text, identifiers, metadata, and numeric
values are illustrative placeholders, not corpus output or measurements.

Progress events on stdout:

```json
{"event": "model_start", "call": 1}
{"event": "search_start", "tool": "search_spinoza", "query": "<search query>"}
{"event": "search_end", "admitted_labels": ["S1"], "evidence_bytes": 500}
{"event": "model_end", "call": 2, "prompt_seconds": 0.1, "generation_seconds": 0.2}
```

The final result follows on stdout. This excerpt omits `events`; in actual
output it contains the full event list:

```json
{
  "answer": "<réponse française> [S1]",
  "sources": [
    {
      "label": "S1",
      "chunk_id": "sha256:<chunk hash>",
      "text": "<passage text omitted>",
      "metadata": {
        "source_file": "synthetic.pdf",
        "page_start": 1,
        "page_end": 1,
        "lecture_date": "1981-01-01"
      },
      "distance": 0.2
    }
  ],
  "evidence_bytes": 500,
  "elapsed_seconds": 0.5
}
```

`[S1]` resolves through `sources` to its chunk and provenance. The result lists
all admitted sources, including any the answer did not cite. A resolved label
alone does not establish that its passage supports the claim.

Failures write a summary to **stderr** and do not print a final result object.
Progress events already emitted remain on stdout, including available
`search_error`, `unknown_citations`, or `agent_error` details. For example, a
search against a missing index can emit these selected stdout events:

```json
{"event": "search_error", "error": "Persistent index is missing; run indexing first."}
{"event": "agent_error", "error": "Retrieval failed; no evidence admitted."}
```

The stderr summary is:

```text
Research failed: Local agent failed: Retrieval failed; no evidence admitted.
```

| Exit status | Meaning |
| --- | --- |
| `0` | A result was produced, including an explicit insufficient-evidence response when no retrieval error prevents it. `--help` also exits successfully without a result. |
| `1` | Research failed: invalid input limits, model or retrieval failure, rejected citations, or another execution error. |
| `2` | Argument parsing failed, such as an invalid mode or missing question. |

## Keeping runs

Nothing is saved by default. To keep both channels, redirect them to ignored
`storage/research/`:

```sh
mkdir -p storage/research
uv run --locked python -m spinoza_rag.research answer \
  'Selon les passages, quels éléments composent le premier genre de connaissance ? Limite la réponse à cette définition.' \
  --top-k 1 --timeout 600 > storage/research/answer.log 2>&1
```

Keep corpus passages, generated answers, and execution traces there. Terminal
scrollback and shell redirection remain the user's local responsibility.
Tracked documentation contains questions and metadata, never corpus quotations.

## How it works

The [agent workflow](agent-workflow.md) describes each step of a run, the
components and versions involved, the search and evidence limits, the answer
checks, every event, privacy controls, and design tradeoffs. Centralized
model, question, output, and evidence budgets are listed in
[configuration](configuration.md#local-chat-and-evidence-defaults).
