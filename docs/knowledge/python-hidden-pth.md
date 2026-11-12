# Python skips hidden .pth files on macOS

- **Status:** Verified for import failure and flag removal; flag origin unknown
- **Last verified:** 2026-11-12, macOS Intel, Python 3.12.13, uv 0.12.1
- **Keywords:** editable install, worktree, `.pth`, `UF_HIDDEN`, `chflags`,
  `ModuleNotFoundError: spinoza_rag`, subprocess

## Symptom

An installed editable package exists under `src/` and its `.pth` points there,
but isolated Python subprocesses cannot import it. In this feature worktree,
subprocess ingestion, packaging, search and configuration checks encountered
`ModuleNotFoundError: spinoza_rag`. Inspect flags, not only file contents:

```sh
ls -lO .venv/lib/python3.12/site-packages/*.pth
```

The affected files carried macOS's `hidden` flag.

## Cause

The installed Python's `site.py:addpackage` explicitly skips `.pth` files whose
`st_flags` includes `stat.UF_HIDDEN`. The editable package relies on this file
to add its installed source directory at interpreter startup. A flag toggle
reproduced failure/success without changing the file contents or source code.
This is distinct from the [stale wheel problem](uv-stale-local-wheel.md).

## Resolution

From the checkout with the affected environment, after sync, clear only the
local virtual environment's `.pth` flags and verify installed imports:

```sh
uv sync --locked
chflags nohidden .venv/lib/python3.12/site-packages/*.pth
.venv/bin/python -I -c 'import spinoza_rag; print(spinoza_rag.__file__)'
uv run --locked --no-sync pytest
```

`--no-sync` uses the installation just checked; the preceding locked sync
already verified dependencies. If flags recur after environment/file restoration,
inspect and clear them again. This changes local file attributes only. No
application import-path hacks, warning filters, skipped tests, or repository
configuration changes are needed. For CI-parity validation without relying on `.pth` processing, use a
non-editable installation:

```sh
uv sync --locked --no-editable
uv run --locked --no-editable pytest
```

This is a verified fallback, not evidence that editable imports stay repaired.

## Verification

On 2026-11-12, two isolated subprocesses used this worktree's Python:

1. `chflags hidden` on `spinoza_rag.pth`: import exited 1 with
   `ModuleNotFoundError`.
2. `chflags nohidden` on the same file: import exited 0 and resolved the
   worktree's installed `src/spinoza_rag/__init__.py`.

The Python source check and subprocess flag toggle establish this cause. In a
later editable full-suite run, flags recurred and 172 tests passed while three
isolated subprocess imports failed. A subsequent non-editable installed run
passed all 175 tests, including those three, without skips or import-path
changes. Ruff, mypy, hooks and package build also passed. An isolated successful
import alone must not be reported as an editable full-suite pass.

## Uncertain or unverified

The process setting the hidden flags has not been identified. Flags recurred
between local runs; a permanent prevention mechanism is not established. No
claim is made about other Python versions or platforms.

## Revalidate when

Python or uv changes; environments are recreated or restored; `.pth` flags
reappear; or isolated imports fail despite an unhidden, correct installed path.
