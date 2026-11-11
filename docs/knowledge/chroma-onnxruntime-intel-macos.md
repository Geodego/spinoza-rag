# Chroma pulls an ONNX Runtime without Intel macOS wheels

- **Status:** Verified
- **Last verified:** 2026-11-11, macOS 26 Intel (`x86_64`), Python 3.12.13,
  uv 0.12.1, chromadb 1.5.9, onnxruntime 1.23.2
- **Keywords:** chromadb, onnxruntime, uv, `required-environments`,
  `constraint-dependencies`, `--upgrade-package`, wheel, `macosx_x86_64`,
  Intel Mac

## Symptom

Adding Chroma failed at installation, after resolution had succeeded:

```text
$ uv add 'langchain-text-splitters>=1,<2' 'chromadb>=1,<2' 'tokenizers>=0.22,<0.23'
Resolved 117 packages in 359ms
error: Distribution `onnxruntime==1.31.1 @ registry+https://pypi.org/simple`
can't be installed because it doesn't have a source distribution or wheel for
the current platform
hint: You're on macOS (`macosx_26_0_x86_64`), but `onnxruntime` (v1.31.1) only
has wheels for [...] `macosx_14_0_arm64` [...]; consider adding
"sys_platform == 'darwin' and platform_machine == 'x86_64'" to
`tool.uv.required-environments` [...]
```

Source: the Codex session that added Chroma (exit code 2).

## Cause

The dependency resolver and PyPI establish the cause:

- `chromadb` depends on `onnxruntime` on every platform (see `uv.lock`).
- uv's universal lock does not require wheels for every platform by default.
  It picked the newest `onnxruntime`, 1.31.1, which ships only an Apple
  Silicon macOS wheel. The failure appears only when installing on Intel macOS.
- On PyPI (checked 2026-11-11), cp312 wheels that install on Intel macOS exist
  up to **1.23.2**: `universal2` wheels for 1.19.0–1.22.1, then dedicated
  `x86_64` wheels for 1.23.0–1.23.2. Releases from 1.24.1 on ship macOS wheels
  for `arm64` only.

**Correction:** the earlier explanation, that releases from 1.21 onward lack
Intel wheels, was wrong. It appears in commit `18d5590` and was previously in
the README. The evidence supports the 1.24 boundary instead.

## Resolution

`pyproject.toml` declares the platform that must be installable:

```toml
[tool.uv]
required-environments = ["sys_platform == 'darwin' and platform_machine == 'x86_64'"]
```

uv must now find an installable wheel for Intel macOS, so lock resolution, not
a later install, catches this class of failure. This is the automated
safeguard.

### History

- The first workaround was a direct dependency, `onnxruntime>=1.20,<1.21`, on
  Intel macOS.
- It then became `constraint-dependencies = ["onnxruntime<1.21; <Intel macOS
  marker>"]`, alongside `required-environments`. The lock held 1.20.1 for all
  platforms.
- On 2026-11-11 the constraint was removed and the lock updated with
  `uv lock --upgrade-package onnxruntime`. The lock now holds a single 1.23.2
  entry for all platforms, and no other package changed.

**Lesson:** removing a constraint and running a plain `uv lock` changed only
the lock's `[manifest]` section. uv keeps already-locked versions as
preferences, so relaxing a bound needs a focused `--upgrade-package` to take
effect.

### Lock results (2026-11-11)

From a scratch copy of `pyproject.toml` with no existing lock:

| Configuration | Lock result |
| --- | --- |
| Neither setting | 1.31.1 everywhere, uninstallable on Intel macOS |
| `required-environments` only | Forked: 1.23.2 on Intel macOS, 1.31.1 elsewhere |
| Both settings (former) | 1.20.1 everywhere |

The repository lock, after the focused upgrade, holds 1.23.2 everywhere. A
broader `uv lock --upgrade` would probably fork the lock as in the second row:
Intel macOS would stay at 1.23.2 while other platforms move ahead.

## Independent decision: Ollama embeddings, not Chroma's default

The ONNX problem did not cause the choice of Ollama embeddings:

- `AGENTS.md` already required "Ollama for future inference and embeddings" in
  commit `6bfdb30`, before Chroma was added.
- `index.py` opens collections with `embedding_function=None` and always
  passes explicit vectors (`embeddings=`, `query_embeddings=`).
- Chroma still requires `onnxruntime` as a dependency, so it remains installed.
- Probes with both 1.20.1 and 1.23.2 created a collection, upserted vectors and
  queried it the way the project does. `onnxruntime` was never imported.
- No unit test asserts this.

## Verification (onnxruntime 1.23.2, 2026-11-11)

- `uv sync --locked` replaced 1.20.1 with 1.23.2. The native library loads,
  and `get_available_providers()` reports CoreML, Azure, and CPU.
- Ruff, pytest (78 passed, editable and `--no-editable`), all prek hooks,
  `uv lock --locked`, and `uv build --no-sources` passed.
- Reopening the full 1,194-chunk index reported `embedded_this_run: 0` with a
  byte-identical manifest.
- A real search returned the same pages and cosine distances as with 1.20.1.

## Uncertain or unverified

- Linux: CI has not yet run with Chroma and either `onnxruntime` version.
- Chroma's built-in ONNX embedding function was not tested, because it would
  download a model and the project does not use it.

## Revalidate when

- `chromadb` or uv is upgraded, or a broad `uv lock --upgrade` forks
  `onnxruntime` per platform.
- Intel macOS is no longer a target; in that case remove `required-environments`.
- Chroma's built-in embedding functions are adopted.
- A change makes `onnxruntime` load at runtime. Consider a test that fails if
  `onnxruntime` appears in `sys.modules` after indexing and search.
