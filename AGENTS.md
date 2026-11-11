## Git and commit conventions

- Follow Conventional Commits 1.0.0.
- Write commit messages in English.
- Format: <type>(<optional-scope>): <imperative summary>.
- Use lowercase subjects, without a trailing period.
- Keep the subject within 72 characters.
- Prefer focused, atomic commits representing one coherent change.
- Use the body to explain motivation, design decisions,
  trade-offs, and non-obvious consequences.
- Do not describe changes that are not included in the commit.
- Do not claim tests passed unless they were actually executed.
- Inspect the staged diff before committing.
- Never include secrets, source PDFs, or generated corpus data.
- Never bypass Git hooks with --no-verify.
- Do not amend, rebase, or force-push without explicit authorization.
- Only create commits when explicitly requested.

## Python code quality

- Use Ruff as the single Python linter and formatter.
- Follow the configuration in pyproject.toml.
- Before committing, run:
  - `uv run ruff check .`
  - `uv run ruff format --check .`
  - `uv run pytest`
- Fix lint and formatting errors before committing.
- Never bypass quality checks without explicit authorization.

## Docstring conventions

- Follow PEP 257 and Google-style docstrings.
- Write all docstrings in English.
- Use Python type hints for public function signatures.
- Document public modules, classes, and functions.
- Prefer concise, meaningful docstrings over boilerplate.
- Use a one-line docstring for simple functions.
- For complex functions, use Args, Returns, Raises, and
  Yields sections when applicable.
- Do not repeat type information already expressed by type hints.
- Explain behavior, contracts, invariants, side effects,
  and non-obvious design decisions.
- Keep docstrings synchronized with implementation changes.
- Do not generate trivial docstrings merely to satisfy linting.
- Follow the Ruff configuration in pyproject.toml.

### Module docstrings

- Every application module must begin with a concise module docstring.
- Follow PEP 257 and write docstrings in English.
- Describe the module's purpose and primary responsibility.
- Document architectural invariants and important side effects
  when they are not obvious.
- For CLI modules, describe their intended usage.
- Avoid repeating the filename or enumerating obvious functions.
- Do not add boilerplate merely to increase documentation coverage.
- Test modules and internal package __init__.py files are exempt.