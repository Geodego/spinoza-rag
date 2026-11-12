# Development

Shared coding conventions (testing, docstrings, commits) live in
[`AGENTS.md`](../AGENTS.md), which `CLAUDE.md` imports for Claude Code. This page
covers the toolchain, checks, and repository layout.

## Toolchain

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and use a
current Git (at least 2.28). The verified toolchain is uv 0.12.1 and Python
3.12.13. The `.python-version` file pins the Python patch release;
`requires-python` limits support to 3.12.

An old Git on PATH breaks prek's staged-file checks
(`invalid option: --no-relative`). On the audited macOS Intel machine,
`/usr/local/bin/git` is Git 2.23 while Apple's `/usr/bin/git` 2.50.1 works.
Select it for the shell session before setup, and set PyCharm's **Version
Control → Git → Path to Git executable** to `/usr/bin/git`:

```sh
export PATH="/usr/bin:$PATH"
git --version
```

Global shell and IDE settings are not modified by this repository. See
[the knowledge entry](knowledge/git-prek-no-relative.md) for the diagnosis.

## Setup

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
rewrite source files. Format intentionally with `uv run --locked ruff format .`.

## Checks

Run the same checks before submitting changes:

```sh
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked mypy
uv run --locked pytest
uv run --locked prek run --all-files
uv build --no-sources
```

CI checks Linux using locked dependencies and a non-editable package
installation; local verification covers macOS Intel. The smoke test verifies
imports and installed distribution metadata from outside the checkout. Synthetic
PDF tests exercise extraction, diagnostics, reruns, and failure handling.
Retrieval tests use synthetic documents, a generated tokenizer, fake embeddings,
and real local Chroma persistence. Tests run without the corpus, Ollama, or
model downloads, and use small synthetic fixtures with explicit page numbers
and source identities.

The `check-docs` prek hook runs `scripts/check_docs.py` whenever a Markdown file
changes, and in CI through `prek run --all-files`. It scans every Markdown file,
not only changed ones, because renaming a page breaks links in files that did
not change. It fails on relative links whose target file or heading anchor is
missing, and on a README longer than 250 lines. External URLs are not checked.
Run it directly with `uv run --locked python scripts/check_docs.py`.

## Evaluating from a worktree

A fresh worktree has no ignored corpus/index/model artifacts. To reuse an existing
index, pass an absolute `--index-dir` pointing at the original checkout's
`storage/index`; keep that index's relative upstream page/chunk paths intact.
Run from the feature worktree so its code and annotation file are used, and
choose a new output directory there. Example, replacing the index path:

```sh
uv run --locked python -m spinoza_rag.evaluate \
  --index-dir /absolute/path/to/original-checkout/storage/index \
  --output-dir storage/evaluations/baseline-v1
```

Do not copy individual Chroma files or rebuild the corpus just to evaluate it.
If subprocesses fail to import an editable package on macOS, see
[hidden .pth files](knowledge/python-hidden-pth.md).

## Repository layout

```text
src/spinoza_rag/               Installable application package
src/spinoza_rag/config.py      Central defaults: paths, models, and stage settings
src/spinoza_rag/ingest.py      PDF extraction and verified artifact loading
src/spinoza_rag/chunk.py       Token-aware persistent chunking
src/spinoza_rag/embeddings.py  Local Ollama HTTP client
src/spinoza_rag/index.py       Resumable persistent Chroma indexing
src/spinoza_rag/search.py      Retrieval-only CLI
src/spinoza_rag/evaluate.py    Validated/proposed page-level retrieval evaluation
evaluation/gold.v1.json        Corpus-free candidate questions and supporting pages
tests/                         Synthetic ingestion, indexing, and retrieval tests
scripts/check_docs.py          Markdown link and README length check (prek hook)
docs/                          Design, reference, and evaluation notes
docs/knowledge/                Engineering lessons and compatibility notes, indexed
AGENTS.md                      Shared coding-agent guidance
CLAUDE.md                      Imports AGENTS.md for Claude Code
pyproject.toml                 Metadata, dependencies, and tool configuration
uv.lock                        Committed dependency resolution
```

`AGENTS.md` is the shared source of repository conventions for Codex and
Claude Code. `CLAUDE.md` explicitly imports it so the same guidance is available
across Claude Code versions. No global agent configuration is required.

## Dependency maintenance

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
