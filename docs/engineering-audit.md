# Initial engineering audit

Date: 2026-11-11. Baseline: commit `3f515c9` (clean working tree).
Scope: repository foundation, not application implementation.

## Assessment

The original tooling choices were sound, but the repository was not yet a
reproducible, testable starting point. It contained seven tracked configuration
and documentation files, no application package, no tests, and no CI.

The changes establish an installable Python scaffold with one meaningful
packaging smoke test. No ingestion, retrieval, inference, or other RAG behavior
has been implemented. No models or runtime application dependencies were added.

## What was already correct

- uv project metadata and the committed lockfile separated development tools
  from the empty runtime dependency list. Keep the lockfile in version control.
- Ruff was already the single linter and formatter, with import sorting, Python
  3.12 syntax, bug prevention, and Google-style docstrings.
- Commitizen and local prek hooks already checked Conventional Commits and Ruff.
  Both Git hook shims were installed in this checkout.
- AGENTS.md already held shared conventions and prohibited unsolicited commits,
  bypassing hooks, and rewriting history.
- The README described offline ingestion separately from question answering,
  preserved page-aware provenance as an objective, and excluded the corpus.
- No source PDFs, credentials, extracted corpus, or local databases were tracked.

## Findings and changes

| Priority | Finding | Resolution |
| --- | --- | --- |
| High | Default Git 2.23 breaks actual installed hooks on `--no-relative`, even though all-files checks pass. | Verified the existing Apple Git 2.50.1; documented a session PATH selection and PyCharm executable setting. No global configuration changed. |
| High | No repeatable automated gate; the documented pytest command collected zero tests and exited 5. | Added a real installed-package smoke test and minimal Linux CI for sync, Ruff, pytest, hygiene, and build. Empty test suites still fail. |
| Medium | `>=3.12` allowed untested later minor Python versions, and no interpreter pin existed. | Pinned the verified 3.12.13 patch; declared `>=3.12,<3.13` for this application. Upgrade deliberately after validation. |
| Medium | No importable package or build backend; future imports could depend on the working directory. | Added `src/spinoza_rag/__init__.py` and the native uv build backend. Verified metadata and imports outside the checkout, without path hacks. |
| Medium | Commitizen accepted a 75-character subject despite the configured 72-character limit. | Added the explicit `--message-length-limit 72` flag to the hook; verified rejection and acceptance of long explanatory bodies. |
| Medium | Hook commands used `uv run --no-sync`, which could use stale installed tools. | Changed hooks to `uv run --locked`; stale lock metadata fails, and installed tools synchronize with the lock. |
| Medium | Hook installation was undocumented and hook stages were not installed by default. | Declared both default hook types and documented `uv run --locked prek install`. |
| Medium | CLAUDE.md only told Claude to read AGENTS.md in prose. | Replaced that prompt with the supported `@AGENTS.md` import. AGENTS.md stays canonical. |
| Medium | README had no working development instructions and could imply planned integrations existed. | Added clone/setup/check/update instructions and explicitly separated implemented tooling from the planned RAG stack, including PyMuPDF. |
| Medium | Ignore rules omitted Ruff/build outputs and model files. | Added cache, build, coverage, model, and local agent preference exclusions; retained corpus and secret exclusions. |
| Low | Agent instructions repeated docstring rules but lacked project boundaries and isolated-test expectations. | Consolidated guidance, added local processing and provenance boundaries, external-document instruction handling, dependency discipline, and offline unit-test expectations. |
| Low | Text conventions depended on individual editor settings. | Added EditorConfig and Git text attributes for UTF-8/LF and final newlines. |
| Decision | No code license is present, and PDF parser licensing affects future distribution. | Flagged both in README; no license was selected on the owner's behalf. |

## Engineering decisions

### Minimal packaging

One package, one backend, no CLI, no artificial layers, and no planned pipeline
modules. `uv_build==0.12.1` matches the tested uv release and uses its default
src layout. The exact backend pin also fixes the build tool outside uv's runtime
lockfile. Packaging enables local installation; no package-registry publishing or
release workflow was added.

### Dependency reproducibility

Retain existing development dependency versions. The regenerated lockfile changes
the project from virtual to editable and narrows Python support; it is generated
by uv, not edited by hand. Minimum constraints in pyproject.toml express supported
tool versions; uv.lock selects exact direct and transitive versions for normal
work. CI pins uv 0.12.1; `[tool.uv]` guards the supported uv release series.
Python's patch pin is a reproducibility choice, not a promise of bit-for-bit
identical operating-system or future model artifacts.

### Ruff and docstrings

Keep the focused existing rule selection: E4/E7/E9, F, I, UP, B, D. Do not enable
ALL or introduce a second formatter. Keep Google conventions, constructor and
argument-documentation exemptions, and test docstring exemptions. Package
initializers may omit docstrings; application modules must document their
purpose. Type hints remain a coding convention; Ruff is not a type checker.

### pytest

Use explicit tests/ discovery and importlib imports with no sys.path/PYTHONPATH
patches. Strict config/markers catch misspellings; warnings become errors so
new warnings are reviewed. Add narrow warning filters only for understood
third-party warnings. The smoke test starts an isolated Python subprocess in a
temporary directory and checks package import and installed metadata. It tests
the new packaging boundary, not an invented application feature.

### Commit hooks and CI

Retain prek and Commitizen instead of adding parallel runners. Pre-commit runs
Ruff lint and format checks plus builtin TOML/YAML, merge-conflict, and file-size
checks. Every configured hook is check-only. Format explicitly before committing.
Ruff also checks pyproject.toml with the installed version's TOML support.

The builtin repository intentionally uses prek's implementation and needs no
remote hook clone or extra environment. pytest runs explicitly and in CI; it is
not attached to every commit before the test suite has application behavior.
Commit messages and the explicit 72-character subject limit are validated at
commit-msg; scope, imperative wording, and
English remain review conventions rather than custom brittle regex policies.

A single Linux CI job is justified: local hooks are optional and this macOS
Intel development environment alone cannot exercise Linux. CI uses a
non-editable installation for the smoke test, builds the sdist and wheel, has
read-only repository permissions, disables persisted checkout credentials, and
pins action revisions. Remote execution is only verified after the changes are
pushed; this audit does not push or alter GitHub settings.

## Deferred deliberately

- Add Ollama, LangChain integration packages, Chroma, PyMuPDF, and LangGraph only
  with features that consume them. Resolve macOS Intel wheels and Python 3.12
  compatibility at that time; none is established by this tooling-only audit.
- Add a static type checker when application interfaces exist; do not add a tool
  that currently has no application signatures to check.
- Add behavior tests with the first ingestion feature using synthetic documents
  and explicit page provenance. Add model/service integration tests separately.
- Defer Docker, task runners, release automation, coverage gates, complex test
  matrices, settings classes, and example environment variables until needed.
- Choose a code license before making the repository public and review PyMuPDF's
  AGPL/commercial terms before adopting it. Corpus rights remain separate.
- Review Python patch releases, pinned CI actions, and dependency updates
  regularly. No hosted-service access or global agent configuration was changed.

The default shell still resolves the old Git until the documented PATH selection
is applied. Validation uses the installed Apple Git through a process-local PATH.
This is an environment prerequisite, not a hidden repository workaround.

## Official references

- [uv locking and syncing](https://docs.astral.sh/uv/concepts/projects/sync/)
- [uv build backend](https://docs.astral.sh/uv/concepts/build-backend/)
- [pytest integration practices](https://docs.pytest.org/en/stable/explanation/goodpractices.html)
- [Git 2.28 diff options](https://git-scm.com/docs/git-diff/2.28.0)
- [prek builtin hooks](https://prek.j178.dev/builtin/)
- [Codex AGENTS.md](https://learn.chatgpt.com/docs/agent-configuration/agents-md)
- [Claude Code shared instructions](https://code.claude.com/docs/en/memory#share-one-file-with-other-coding-tools)
- [PyMuPDF license](https://pymupdf.readthedocs.io/en/latest/about.html#license-and-copyright)

## Validation

Final validation on macOS Intel, Python 3.12.13, uv 0.12.1, and Apple Git 2.50.1:

| Check | Result |
| --- | --- |
| `uv sync --locked` | Passed; 23 installed packages including the scaffold. |
| `uv run --locked ruff check .` | Passed. |
| `uv run --locked ruff format --check .` | Passed. |
| `uv run --locked pytest` | Passed: one packaging test, no warnings. |
| `uv run --locked prek validate-config .pre-commit-config.yaml` | Passed. |
| `uv run --locked prek install` | Both hook shims installed. |
| `uv run --locked prek run --all-files` | Passed. New files were also checked explicitly with `--files`. |
| Installed commit-msg shim, valid scoped Conventional Commit | Passed. |
| Installed commit-msg shim, invalid message | Rejected as expected; Commitizen exit 14, hook exit 1. |
| Overlong subject and valid long body | Subject rejected; explanatory body accepted. |
| Installed pre-commit shim | Passed using a temporary alternate Git index. |
| Deliberately invalid lint and formatting fixtures | Rejected as expected; source bytes unchanged. |
| `uv build --no-sources` | Passed: sdist and wheel, including wheel rebuilt from sdist. |
| Fresh isolated environment, built wheel import and metadata | Passed; imported from the fresh environment's site-packages. |
| Non-editable sync and pytest, matching CI | Passed: one packaging test. |
| Restore editable sync and pytest | Passed; development environment restored. |
| `git diff --check` | Passed. |

Integrity checks confirmed existing dependency package versions were retained
and no runtime dependencies were added. Ignore rules cover corpus, chunks,
embeddings, vector indexes, models, secrets, caches, and build output. The
.env.example exception remains available. Hook testing preserved the real Git
index and working source bytes. Only newly created temporary fixtures were
removed; no existing user data was deleted.

Failures investigated:

- Baseline pytest collected zero tests and exited 5; fixed with an installed-
  package test, without suppressing the empty-suite failure.
- A scratch-workspace editable import failed because macOS marked its generated
  .pth file hidden. Editable and non-editable imports in the actual repository
  passed; no application workaround was added.
- Commitizen CLI accepted an overlong subject despite the pyproject limit. An
  explicit hook flag fixed the check; both subject rejection and long-body
  acceptance were verified.
- Default Git 2.23 failed actual hook execution on --no-relative. Hooks passed
  with existing Apple Git 2.50.1. The documented shell/IDE selection is still
  needed because global settings were intentionally left unchanged.
- Direct hook invocation refused an unstaged hook configuration. Testing with
  a temporary alternate index simulated proper staging without changing the
  user's real index.
- An audit helper invoked through /usr/bin/python3 lacked tomllib; rerunning
  it with the uv-managed Python 3.12 interpreter resolved this. Project checks
  already used uv and were unaffected.

GitHub-hosted CI has not run because no commit or push was requested. Commands
and configuration passed locally; Linux runner behavior remains unverified
until the first remote run. No RAG dependencies, models, Ollama configuration,
or hosted services were changed.

## Readiness

The scaffold is ready for PDF-ingestion development once the shell and IDE
select compatible Git. Introduce the parser deliberately, define source/page
provenance, and add synthetic extraction tests with the first feature. Public
distribution still needs a code-license decision and PyMuPDF license review.
