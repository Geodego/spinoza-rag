# Non-editable sync installs a stale wheel of this project

- **Status:** Verified and prevented by configuration; the CI impact is a
  hypothesis
- **Last verified:** 2026-11-12, macOS Intel, uv 0.12.1
- **Keywords:** uv, `--no-editable`, `cache-keys`, stale wheel,
  `ModuleNotFoundError`, `--reinstall-package`, CI parity

## Symptom

After a new module was added under `src/`, the CI-style test run failed:

```text
$ uv sync --locked --no-editable && uv run --locked --no-editable pytest
E   ModuleNotFoundError: No module named 'spinoza_rag.config'
```

The editable run passed. The installed `site-packages/spinoza_rag/` lacked
`config.py`, although the file was committed. Editing an existing module is
worse: the tests silently run the old code.

## Cause

For a local project, uv rebuilds the wheel only when a cache key changes. By
default the keys are the project's build files, such as `pyproject.toml`, not
`src/**/*.py`. A wheel built on 2026-11-11, before `config.py` existed, was
therefore reused. Editable installs link to the source tree and are
unaffected.

## Resolution

`pyproject.toml` adds the source files to uv's cache keys:

```toml
[tool.uv]
cache-keys = [{ file = "pyproject.toml" }, { file = "src/**/*.py" }]
```

Listing `cache-keys` replaces uv's defaults, so `pyproject.toml` is kept
explicitly. The project has no `setup.py` or `setup.cfg`. If another build
input appears, such as non-Python package data under `src/`, add it to this
list.

To force a rebuild manually, for example when diagnosing:

```sh
uv sync --locked --no-editable --reinstall-package spinoza-rag
```

## Verification (2026-11-12)

Each step was a non-editable sync followed by an inspection of
`site-packages`:

| Change | With `cache-keys` | Without (control) |
| --- | --- | --- |
| None | No rebuild (`Checked 120 packages`) | — |
| Comment appended to an existing module | Rebuilt; edit installed | No rebuild; edit missing |
| New module added | Rebuilt; module installed | Not tested; matches the original failure |
| Probe files removed | Rebuilt; probes gone | — |

All 95 tests passed in both non-editable and editable installs, and
`uv lock --locked` still passes.

## Uncertain or unverified

CI checks out fresh files and probably gets new modification times, so it is
unlikely to reuse a stale wheel. This has not been confirmed in a CI run.

## Revalidate when

- uv's caching of local projects or the `cache-keys` syntax changes.
- Build inputs other than `pyproject.toml` and `src/**/*.py` are added.
- A non-editable failure mentions a module that exists in `src/`.
