# Spinoza RAG

A fully local Retrieval-Augmented Generation (RAG) project for exploring
Gilles Deleuze's lectures on Spinoza.

The project aims to provide grounded answers to philosophical questions, supported by references to the original lecture transcripts.

## Architecture

PDF extraction, persistent chunking, local embeddings, vector indexing, and
retrieval-only search are implemented. Answer generation remains a future stage.

**Offline ingestion**

PDF → Page-aware JSONL → Persisted chunks → Ollama embeddings → Persistent Chroma index

**Question answering**

Implemented: User question → Query embedding → Retrieved passages + page references

Planned later: Retrieved passages → Local LLM → Source-grounded answer

Extraction, chunking, and indexing are separate commands. Queries reopen the
persisted index and embed only the question; they never reprocess the PDF or
recompute document embeddings.

## Technology Stack

- **Python 3.12** — Application development
- **uv** — Dependency and environment management
- **pypdf** — Implemented page-aware PDF extraction
- **Ollama** — Local embeddings with `qwen3-embedding:0.6b`; inference is not implemented
- **LangChain text splitters** — Token-aware recursive chunking
- **Chroma** — Embedded persistent storage with cosine-distance retrieval
- **Ruff** — Linting and formatting
- **pytest** — Automated testing

Runtime dependencies are pypdf, LangChain text splitters, tokenizers, and Chroma.
No LangGraph, answer generation, reranking, hybrid search, web interface, or cloud
integration is implemented. All corpus processing and inference requests remain
local. Setup downloads the required packages, model, and tokenizer once; runtime
commands do not download models or call hosted inference APIs.

## Development Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and use a
current Git (at least 2.28) first.
The verified toolchain is uv 0.12.1 and Python 3.12.13. The `.python-version`
file pins the Python patch release; `requires-python` limits support to 3.12.
On this macOS Intel machine, `/usr/local/bin/git` is an old Git 2.23 that breaks
prek's staged-file checks (`invalid option: --no-relative`). Apple's existing
`/usr/bin/git` 2.50.1 works. For this shell session, select it before setup:

```sh
export PATH="/usr/bin:$PATH"
git --version
```

In PyCharm, set **Version Control → Git → Path to Git executable** to
`/usr/bin/git`. Global shell and IDE settings are not modified by this repository.
On other machines, ensure the Git on PATH is current.

From a fresh checkout:

```sh
git clone https://github.com/Geodego/spinoza-rag.git
cd spinoza-rag
uv python install
uv sync --locked
uv run --locked prek install
```

`uv sync --locked` creates `.venv`, installs this package in editable mode and
installs the development tools from `uv.lock`. Activation and a separate pip
installation are unnecessary. In PyCharm, select `.venv/bin/python` as the
project interpreter (`.venv\Scripts\python.exe` on Windows).

`prek install` installs both pre-commit and commit-msg hooks. The hook
configuration uses prek's built-in hygiene checks; use prek rather than the
separate pre-commit runner. All configured hooks are check-only and do not
rewrite source files.
Format intentionally with `uv run --locked ruff format .`.

Run the same checks before submitting changes:

```sh
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest
uv run --locked prek run --all-files
uv build --no-sources
```

CI checks Linux using locked dependencies and a non-editable package
installation; local verification covers macOS Intel. The smoke test verifies
imports and installed distribution metadata from outside the checkout. Synthetic
PDF tests exercise extraction, diagnostics, reruns, and failure handling. Retrieval
tests use synthetic documents, a generated tokenizer, fake embeddings, and real
local Chroma persistence. Tests run without the corpus, Ollama, or model downloads.

## Repository Layout

```text
src/spinoza_rag/          Installable application package
src/spinoza_rag/ingest.py PDF extraction and verified artifact loading
src/spinoza_rag/chunk.py Token-aware persistent chunking
src/spinoza_rag/embeddings.py Local Ollama HTTP client
src/spinoza_rag/index.py Resumable persistent Chroma indexing
src/spinoza_rag/search.py Retrieval-only CLI
tests/                   Synthetic ingestion, indexing, and retrieval tests
AGENTS.md                Shared coding-agent guidance
CLAUDE.md                Imports AGENTS.md for Claude Code
pyproject.toml           Metadata, dependencies, and tool configuration
uv.lock                  Committed dependency resolution
```

`AGENTS.md` is the shared source of repository conventions for Codex and
Claude Code. `CLAUDE.md` explicitly imports it so the same guidance is available
across Claude Code versions. No global agent configuration is required.

## PDF Preprocessing and Ingestion

Place your locally obtained PDF at `data/spinoza.pdf`, then run from the repository
root:

```sh
uv sync --locked
uv run --locked python -m spinoza_rag.ingest
```

No corpus is bundled. If `data/spinoza.pdf` is absent, the command reports the
missing file without publishing output. Tests generate their own synthetic PDFs.

The default destination is `storage/ingestion/`, containing:

- `pages.jsonl`: one UTF-8 JSON object per physical PDF page, including empty or
  failed pages. Each has `schema_version`, `document_id`, `source_file`,
  `page_number` (one-based), `text`, and `diagnostics`.
- `report.json`: source SHA-256 identity and size, PDF metadata (title, author,
  subject, keywords, creator, producer, and raw date strings when available),
  pypdf version, extraction settings (including mode), page counts, per-page diagnostics, parser
  warnings, and a SHA-256 digest of the exact JSONL bytes.

Only the source basename is persisted, not a machine-specific absolute path.
`document_id` is `sha256:<PDF byte digest>`; moving or renaming a source does not
change its identity. A changed PDF has a new identity. Page numbers refer to PDF
positions, not printed page labels. Future chunks must retain these identifiers.

Diagnostics flag:

- `empty_text`: blank, whitespace-only, or image-only pages with no extracted text.
- `short_text`: fewer than 40 non-whitespace characters by default.
- `replacement_characters` or `control_characters`: suspicious decoded text.
- `parser_warning`: pypdf recovered from an issue while processing a page.
- `extraction_error`: a page could not be extracted; its record retains its page
  number and error, and later pages are still processed.

The report's summary includes empty, failed, text-bearing, and review-page
counts. Document-level parser/metadata warnings are retained separately. Review
signals are heuristics, not proof of quality: a legitimate title page may be
short, and plausible-looking text can still have incorrect reading order.

The CLI prints the summary and artifact paths, never the extracted text. Exit 0
means extraction completed without failed pages and at least one page has text;
empty/short-page and other review flags may still exist. Exit 1 means an input or
output error, a page extraction failure, or no extractable text anywhere. In the
last two cases, artifacts are still published for diagnosis. Argument syntax
errors use argparse's standard exit 2. Encrypted and zero-page PDFs are rejected.

### Reruns and Output Safety

Results contain no run timestamps. With the same PDF basename, settings, and
locked extractor version, the CLI verifies the existing PDF fingerprint,
settings, JSONL checksum, page sequence/provenance, and report consistency before
invoking pypdf. Valid artifacts are reused without PDF extraction or file
rewrites; the CLI prints `Reusing verified artifacts; PDF extraction skipped.`
Only a missing destination triggers extraction. The component never appends
page records. Validation still reads the PDF bytes and existing JSON artifacts.

Incomplete, corrupt, or outdated artifacts fail early without re-extraction or
overwriting. Select a new destination for changed PDFs/settings or repairs.
Cached runs retain the original diagnostic summary and failure exit status.
The output schema version is also the extraction revision: bump it when changing
extraction behavior so older results cannot be silently reused.

Both artifacts are written into a temporary sibling directory before a single
rename publishes the directory. Write or publication errors leave no partial
new destination. Existing output must match both expected files exactly;
changed inputs/settings, incomplete output, or additional files cause a refusal
without changing existing files. Choose a new destination when re-extracting a
changed input or testing different settings:

```sh
uv run --locked python -m spinoza_rag.ingest data/spinoza.pdf \
  --output-dir storage/ingestion-v2 --min-page-characters 40
```

Use `--min-page-characters 0` to disable only the short-text heuristic. This is a
sequential local component; concurrent writers are outside its scope.

For the locally provided lecture PDF, use layout mode in a separate destination:

```sh
uv run --locked python -m spinoza_rag.ingest data/spinoza.pdf \
  --extraction-mode layout --output-dir storage/ingestion-layout
```

The first plain-mode run extracted 531 pages without empty pages or parser
errors, but spot checks found spaces inside words. Layout mode fixes those
sampled word splits and orders page-number footers after the body text. It also
adds layout whitespace. The default remains `plain` for reproducibility of the
original run; mode selection is explicit and recorded in the report. The
original plain output is preserved. No heuristic report replaces visual review
of representative PDF passages and reading order.

### Extraction Decisions and Limits

`extract_pdf()` performs extraction without writing artifacts;
`write_extraction()` handles serialization and publication. No chunking, vector
index is involved in PDF extraction. Later stages consume the saved artifacts.

[pypdf](https://pypdf.readthedocs.io/en/stable/user/extract-text.html) was selected
as a pure-Python, [BSD-licensed](https://github.com/py-pdf/pypdf/blob/main/LICENSE)
starting point. Tests verify French accents, page provenance, metadata, and
failure behavior. The actual lecture PDF has been processed locally; sampled plain and layout
results were compared as described above. Full transcription accuracy and complex
reading order remain unverified. Inspect representative passages against the PDF
before choosing normalization or chunking rules.

Plain extraction normalizes only CRLF/CR line endings to LF. It preserves accents,
line breaks, headers, footers, and hyphenation; it does not infer paragraphs or
remove repeated text. pypdf performs no OCR, so scanned pages need a separate
future decision. The source and extracted text are held in memory, which is
appropriate for this initial single-document component; large-document memory
and extraction performance have not been systematically benchmarked.

## Persistent Chunking, Embeddings, and Search

Use the previously extracted layout artifacts in `storage/ingestion-layout/`.
The following stages do not open the original PDF. Keep the page artifacts and
chunk artifacts alongside the index: their fingerprints are verified during
indexing and search.

### One-Time Local Setup

```sh
uv sync --locked
# Start the Ollama desktop application first, if it is not already running.
ollama pull qwen3-embedding:0.6b
mkdir -p models
curl --fail --location \
  https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/resolve/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3/tokenizer.json \
  --output models/qwen3-embedding-tokenizer.json
shasum -a 256 models/qwen3-embedding-tokenizer.json
```

The pinned tokenizer SHA-256 is
`def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a`.
Only this tokenizer JSON is needed; PyTorch and Transformers are not required.
The model download is about 639 MB and the tokenizer about 11.4 MB. Neither is
stored in Git. Application commands never pull a missing model automatically.

The verified local stack uses Chroma 1.5.9, langchain-text-splitters 1.1.3,
tokenizers 0.22.2, and Ollama 0.40.16 on macOS Intel. uv constrains Chroma's
transitive ONNX Runtime dependency below 1.21 only on Intel macOS because later
releases lack compatible wheels. ONNX is not used for embeddings: Chroma's
default embedding function is explicitly disabled. Chroma telemetry is disabled.
The Ollama URL is restricted to HTTP loopback addresses, defaulting to
`http://127.0.0.1:11434`; environment HTTP proxies are bypassed.

### Run the Three Independent Stages

```sh
uv run --locked python -m spinoza_rag.chunk
uv run --locked python -m spinoza_rag.index
uv run --locked python -m spinoza_rag.search \
  "Quelle est la différence entre la joie et la tristesse ?" --top-k 3
```

Restart the process and run the search command again: document vectors are read
from disk, and only the question is embedded. Ollama must be running for query
embeddings. No answer is generated; the CLI prints the nearest passages, source
filename, one-based PDF page range, inferred lecture date, stable chunk ID, and
cosine distance (smaller is closer, not an answer-confidence score).

Default inputs and outputs:

| Stage | Input | Persisted output |
| --- | --- | --- |
| Chunk | `storage/ingestion-layout/` and the local tokenizer | `storage/chunks/chunks.jsonl` and `manifest.json` |
| Index | `storage/chunks/` | `storage/index/chroma/` and `manifest.json` |
| Search | `storage/index/`, existing upstream artifacts, and a question | Printed passages; no document embedding generation |

Paths between artifacts are recorded relatively, so the storage tree can move
together. Staleness is checked against saved page/chunk artifacts. If the original
PDF changes, explicitly rerun extraction first; indexing and search never read
that PDF to guess whether extraction is out of date.

### Chunking Decisions and Provenance

The defaults are 450 Qwen tokens and a 70-token **overlap target**. Tokens are
counted with the local Qwen tokenizer, excluding special tokens. LangChain's
recursive splitter prefers paragraphs, lines, sentences, then words; overlap can
be smaller at structural boundaries. Custom settings are explicit:

```sh
uv run --locked python -m spinoza_rag.chunk \
  --input-dir storage/ingestion-layout --output-dir storage/chunks-experiment \
  --chunk-tokens 500 --overlap-tokens 80
uv run --locked python -m spinoza_rag.index \
  --chunks-dir storage/chunks-experiment --index-dir storage/index-experiment
uv run --locked python -m spinoza_rag.search "Que sont les notions communes ?" \
  --index-dir storage/index-experiment --top-k 3
```

Chunks have hard page boundaries, so `page_start == page_end` in this baseline.
This makes citations exact but can split an argument across pages. Within-page
lecture boundaries are also respected: standalone `DD.MM.YYYY` headings become
ISO lecture dates, carried forward until the next heading. A page can have chunks
from two lectures. Dates are inferred, not authoritative annotations; cover and
contents pages may have no lecture date.

Whitespace normalization joins wrapped lines, preserves paragraph breaks, and
removes a matching page number only at page edges. It does not repair hyphenation,
remove arbitrary headers, summarize text, or invent section titles. Empty pages
are skipped; extraction-error pages block chunking rather than silently omitting
content. Very short fragments remain visible for later quality evaluation.

Every chunk stores the PDF document fingerprint, source filename, one-based page
range, page-local ordinal, text, Qwen token count, and available lecture identity.
Its stable ID hashes the record and exact chunking configuration. The manifest
records the original PDF metadata, page JSONL checksum, tokenizer checksum,
normalization revision, LangChain/tokenizer versions, and chunk JSONL checksum.
Identical chunking runs reuse verified output without rewriting it; changed inputs
or settings require a new output directory.

### Index Compatibility and Recovery

Indexing uses `/api/embed` with `truncate=false`, validates vectors, and stores
normalized embeddings in a cosine Chroma collection. Documents are embedded
without a query prefix. Questions use Qwen's `Instruct: ...\nQuery: ...` format;
the instruction is recorded in the index configuration.

The index fingerprint binds the chunk manifest, exact installed Ollama model
digest and runtime version, collection, distance metric, and query instruction.
Changing any of these is rejected with a request for a new index directory;
weights behind a reused model tag cannot be silently mixed. Chroma record IDs,
passages, and metadata are checked against persisted chunks when reopening.

Stable IDs and upserts prevent duplicates. Repeated indexing reports
`embedded_this_run: 0` when complete. Each completed batch is checkpointed; an
interrupted or unavailable-server run resumes only missing IDs. A vector batch
committed immediately before a checkpoint failure is recovered without embedding
it again. Search rejects incomplete indexes. Concurrent writers are outside this
sequential baseline's scope.

`--batch-size` defaults to 8. Both index and search accept `--timeout` (180 seconds
by default) and `--ollama-url`; a missing model or unreachable service produces
an actionable error. Schema/configuration changes are never automatically applied
to old indexes. Keep the existing output and choose a new directory to rebuild.

### Current Local Corpus Observations

The page artifacts produce **1,194 chunks** across 531 pages and 13 detected
lecture dates. Chunk lengths range from 1 to 450 tokens, with a median of 299;
seven fragments contain fewer than 20 tokens. These are measurement results for
the current local PDF, not guarantees about other documents. The default can be
evaluated and changed later without mixing its index with another configuration.

The index is being built and validated locally; final duration, storage size, and
retrieval examples will be recorded after that run completes. Semantic accuracy
is not established by fake-embedding tests, and there is no formal gold-set
retrieval evaluation yet. Page boundaries, short fragments, retained headers,
and imperfect extraction can affect relevance. No reranking or answer generation
is included in this milestone.

## Dependency Maintenance

Use `uv add <package>` for runtime dependencies and `uv add --dev <package>` for
development tools. Commit `pyproject.toml` and `uv.lock` together. Avoid
`uv pip install` for project dependencies because it does not update the lock.

For a deliberate update, run `uv lock --upgrade-package <package>`, then
`uv sync --locked` and the checks above. Use `--locked` for routine checks so
stale metadata fails instead of silently changing the lockfile. Review tool
upgrades before accepting their new rules or behavior.

Update `.python-version` deliberately when adopting a new Python 3.12 patch.
The build backend is pinned in `[build-system]`; CI pins uv to the same release.
Update both together when moving uv versions, check `[tool.uv].required-version`,
and verify sync, tests, and build. Review the pinned GitHub Action revisions
periodically; dependency and toolchain updates remain explicit.

## Local Data

Keep source PDFs in `data/`, generated indexes and extraction output in
`storage/`, and downloaded model files in `models/`. These directories are
ignored, as are legacy `chroma/`, PDF files, and GGUF weights. Tests should use
small synthetic fixtures with explicit page numbers and source identities.
There are no application environment variables yet; add `.env.example` when
configuration is implemented, documenting only settings the code consumes.

## Project Status

**Work in progress.**

Implemented: PDF extraction with page provenance and diagnostics; persistent
LangChain/Qwen-tokenized chunking; local Ollama embeddings; resumable Chroma
indexing; and retrieval-only search. Synthetic tests cover persistence, restart,
compatibility checks, errors, and no-document-reembedding behavior.

Answer generation, LangGraph orchestration, agents, reranking, hybrid search,
and a web interface remain out of scope. Stop here for retrieval-quality review
before choosing the next milestone.

## Corpus and Licensing

The project is designed around transcripts of Gilles Deleuze's lectures on Spinoza.

The source PDF and extracted text are not included in this repository. Users must provide their own lawfully obtained copy of the material.

The repository contains application code and synthetic test fixtures, not the
underlying copyrighted lecture transcripts. Source PDFs, extracted text, chunks,
embeddings, indexes, and models belong in ignored local directories. A code
license has not yet been chosen; corpus rights remain separate.
