# Engineering knowledge

Reusable lessons from non-obvious problems in this repository: compatibility
constraints, root causes, and verified workarounds. Code, configuration, tests,
and `uv.lock` remain authoritative; these entries explain why they look the
way they do.

## Index

| Entry | Technology | Symptom or trigger |
| --- | --- | --- |
| [Chroma pulls an ONNX Runtime without Intel macOS wheels](chroma-onnxruntime-intel-macos.md) | uv, chromadb, onnxruntime | `onnxruntime` "doesn't have a source distribution or wheel for the current platform"; changing `[tool.uv]` settings, relaxing a bound that `uv lock` ignores, or upgrading Chroma |
| [Non-editable sync installs a stale wheel](uv-stale-local-wheel.md) | uv, packaging | `ModuleNotFoundError` for a module that exists in `src/`, only with `--no-editable`; changing `[tool.uv] cache-keys` |
| [prek hooks fail with old Git](git-prek-no-relative.md) | Git, prek | `git commit` hook fails with `invalid option: --no-relative`, while `prek run --all-files` passes |

## Entry format

One problem per file, named `<technology>-<symptom>.md`. Keep each entry short
and searchable:

- A header with **Status**, **Last verified** (date and environment), and
  **Keywords**, including exact error strings.
- Sections in this order: Symptom, Cause, Resolution, Verification, Uncertain
  or unverified (when anything is), and Revalidate when.
  - **Symptom:** the observed error, quoted exactly.
  - **Cause:** separate what was established (reproduced, or reported by a
    tool) from what was inferred.
  - **Resolution:** what was actually implemented, and where it lives.
  - **Verification:** the commands, tests, or runs that confirmed it, with
    dates.
  - **Revalidate when:** the changes that could make the entry obsolete.

**Status** values are `Verified`, `Partially verified`, `Hypothesis`, or
`Obsolete`. Mark obsolete entries and say what replaced them rather than
deleting them silently. Leave out chronological transcripts, trivial
debugging steps, and anything the README, the other docs/ pages, or the code
already states.
