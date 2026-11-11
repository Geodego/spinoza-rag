# Spinoza RAG

A fully local Retrieval-Augmented Generation (RAG) project for exploring Gilles Deleuze's lectures on Spinoza.

The project aims to provide grounded answers to philosophical questions, supported by references to the original lecture transcripts.

## Objectives

- Build a reproducible document ingestion pipeline.
- Preserve PDF page references throughout extraction, chunking, and retrieval.
- Implement semantic search over the lecture transcripts.
- Generate answers grounded in retrieved passages, with verifiable citations.
- Evaluate retrieval quality and answer faithfulness.
- Explore LangGraph-based orchestration where it provides measurable benefits.

## Architecture

The application separates document ingestion from question answering.

**Offline ingestion**

PDF → Text extraction → Page-aware chunking → Local embeddings → Persistent vector index

**Question answering**

User question → Semantic retrieval → Local LLM → Source-grounded answer

The planned application will process and index documents offline, then reuse the
persisted index rather than reprocessing the PDF for each query.

## Technology Stack

- **Python 3.12** — Application development
- **uv** — Dependency and environment management
- **PyMuPDF** — Planned page-aware PDF extraction
- **Ollama** — Local LLM inference and embeddings
- **LangChain** — Document processing and retrieval
- **LangGraph** — Future conditional orchestration, when justified
- **Chroma** — Persistent vector storage
- **Ruff** — Linting and formatting
- **pytest** — Automated testing

Ollama, PyMuPDF, LangChain, LangGraph, and Chroma are planned; none are
installed or integrated yet. Python, uv, Ruff, and pytest are configured.
The current scaffold has no runtime dependencies and implements no RAG behavior.

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
imports and installed distribution metadata from outside the checkout. Add unit tests as application behavior is
implemented. Unit tests must not depend on the real corpus, running Ollama, or
model downloads. CI currently tests the scaffold only.

## Repository Layout

```text
src/spinoza_rag/          Installable application package
tests/                   Automated tests
AGENTS.md                Shared coding-agent guidance
CLAUDE.md                Imports AGENTS.md for Claude Code
pyproject.toml           Metadata, dependencies, and tool configuration
uv.lock                  Committed dependency resolution
```

`AGENTS.md` is the shared source of repository conventions for Codex and
Claude Code. `CLAUDE.md` explicitly imports it so the same guidance is available
across Claude Code versions. No global agent configuration is required.

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

The initial development focuses on document extraction, page-level provenance, persistent indexing, and source-grounded question answering.

Agentic retrieval strategies and systematic evaluation are planned for subsequent iterations.

## Corpus and Licensing

The project is designed around transcripts of Gilles Deleuze's lectures on Spinoza.

The source PDF and extracted text are not included in this repository. Users must provide their own lawfully obtained copy of the material.

The repository contains the Python package scaffold and documentation, not the
underlying copyrighted lecture transcripts. A code license has not yet been
chosen; select one before public distribution. Before adding PyMuPDF, review
its [AGPL/commercial licensing](https://pymupdf.readthedocs.io/en/latest/about.html#license-and-copyright)
against the intended code license and distribution model.
