# prek hooks fail with old Git: `invalid option: --no-relative`

- **Status:** Verified for the symptom and workaround; the mechanism is partly
  inferred
- **Last verified:** 2026-11-11, macOS Intel, `/usr/local/bin/git` 2.23.0,
  `/usr/bin/git` 2.50.1 (Apple Git-155)
- **Keywords:** prek, pre-commit, git hook, `--no-relative`, Git 2.23, PATH,
  PyCharm, commit fails

## Symptom

`git commit` fails in the installed pre-commit hook:

```text
error: Command `/usr/local/git/libexec/git-core/git diff --name-status -z
--no-renames --no-relative --no-ext-diff --ignore-submodules` exited with an error:
[stderr] error: invalid option: --no-relative
```

The trap is that `prek run --all-files` **passes** in the same environment, so
it does not prove that installed hooks work.

## Cause

- prek collects staged files with `git diff ... --no-relative`. This was
  observed in the error above.
- Git 2.23 rejects that option and Apple Git 2.50.1 accepts it. Reproduced on
  2026-11-11: exit code 255 versus 0. The project requires Git 2.28 or later,
  based on the Git 2.28 documentation for `git diff`.
- `--all-files` did not hit this command in practice. The internal code path
  was not inspected.
- Which Git runs depends on the PATH of whatever launches the commit: the
  login shell, PyCharm, or an agent's shell. On 2026-11-11 the Claude Code
  shell resolved `/usr/bin/git` first, while the Codex session that failed
  used the old Git.

## Resolution

No repository workaround and no global configuration change; see the Toolchain
section of [docs/development.md](../development.md#toolchain):

- Shell: `export PATH="/usr/bin:$PATH"` for the session, or call `/usr/bin/git`.
- PyCharm: set **Version Control → Git → Path to Git executable** to `/usr/bin/git`.
- Never bypass hooks with `--no-verify`.

## Verification

- The initial engineering audit, commit `542a91d` (the file was removed in
  `8751141`), recorded the hook passing with Apple Git 2.50.1. It also showed
  the commit-msg hook rejecting invalid messages.
- The audit tested the hook against a temporary alternate index
  (`GIT_INDEX_FILE`) so the real index stayed untouched. Reuse that technique.
- Every later commit, through `f612f7c`, ran hooks with `/usr/bin/git`.

## Revalidate when

- `/usr/local/bin/git` is upgraded or removed, or the machine changes.
- prek is upgraded: the Git commands it runs, and the minimum version they
  need, may change.
- Hooks fail with another `invalid option` error; check `git --version` first.
