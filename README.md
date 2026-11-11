# Spinoza RAG

A fully local Retrieval-Augmented Generation (RAG) project for exploring
Gilles Deleuze's lectures on Spinoza.

The project aims to provide grounded answers to philosophical questions, supported by references to the original lecture transcripts.

## Architecture

PDF extraction is implemented. Chunking, indexing, and question answering are
planned and remain separate stages.

**Offline ingestion**

Implemented: PDF → Page-aware JSONL + extraction diagnostics

Planned: JSONL → Page-aware chunking → Ollama embeddings → Persistent vector index

**Question answering**

User question → Semantic retrieval → Local LLM → Source-grounded answer

The planned application will process and index documents offline, then reuse the
persisted index rather than reprocessing the PDF for each query.

## Technology Stack

- **Python 3.12** — Application development
- **uv** — Dependency and environment management
- **pypdf** — Implemented page-aware PDF extraction
- **Ollama** — Local LLM inference and embeddings
- **LangChain** — Document processing and retrieval
- **Chroma** — Persistent vector storage
- **Ruff** — Linting and formatting
- **pytest** — Automated testing

pypdf is the only runtime dependency. Ollama, LangChain, and Chroma
are planned and are not installed or integrated yet. No embeddings, retrieval,
or LLM inference is implemented. The aim is a simple, free local workflow, with
architecture, source traceability, tests, evaluation, and measured tradeoffs
providing the technical substance.

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
PDF tests exercise extraction, diagnostics, reruns, and failure handling. Tests
run without the real corpus, Ollama, model downloads, or network access.

## Repository Layout

```text
src/spinoza_rag/          Installable application package
src/spinoza_rag/ingest.py PDF extraction, persistence, and standalone CLI
tests/                   Synthetic ingestion and packaging tests
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
index, orchestration framework, or generic backend abstraction is introduced.

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

Implemented: standalone PDF preprocessing with page-level provenance, JSONL,
extraction diagnostics, safe reruns, and synthetic behavior tests.

Next: inspect extraction quality on the real PDF, then develop and evaluate
page-aware chunking. Embeddings, indexing, source-grounded question answering,
and systematic RAG evaluation are later milestones.

## Corpus and Licensing

The project is designed around transcripts of Gilles Deleuze's lectures on Spinoza.

The source PDF and extracted text are not included in this repository. Users must provide their own lawfully obtained copy of the material.

The repository contains application code and synthetic test fixtures, not the
underlying copyrighted lecture transcripts. Source PDFs, extracted text, chunks,
embeddings, indexes, and models belong in ignored local directories. A code
license has not yet been chosen; corpus rights remain separate.
