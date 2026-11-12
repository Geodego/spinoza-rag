# Delimit exact refusal text separately from nearby instructions

- **Status:** Verified
- **Last verified:** 2026-11-12, macOS Intel, Python 3.12.13,
  qwen3:8b, Ollama 0.40.16, LangChain 1.5.0
- **Keywords:** refusal, citation guard, prompt boundary, insufficient evidence,
  `Answer has no source citations; draft withheld.`

## Symptom

A missing-detail corpus question triggered retrieval, then failed the citation
guard rather than returning a safe insufficient-evidence response. The trace
contained a completed 25-token response with no tool calls or citations.

## Cause

A local replay of the final model step returned the required refusal followed
by the prompt's adjacent methodological caution. Both sentences had appeared
on the same line after the instruction to respond exactly. The guard accepted
only the first sentence. This was a prompt/validator contract mismatch, not a
retrieval error. Corpus-bearing replay inputs and outputs stay in ignored
`storage/research/`; no passage or generated corpus answer is quoted here.

## Resolution

Quote the required refusal on its own line and explicitly prohibit additions.
Also accept the observed fixed, claim-free caution as an optional suffix.
Do not use a substring refusal check: arbitrary uncited claims following the
refusal must still fail. Keep these exact strings alongside their validator.

## Verification

On 2026-11-12, the local final-step replay reproduced the extra caution. A
synthetic graph regression failed before the fix for that exact variant. Tests
now cover both allowed refusals and rejection of an uncited substantive suffix.
Run `uv run --locked pytest tests/test_agent.py`.

## Revalidate when

- The model, refusal prompt, accepted phrases or citation guard changes.
- Structured model output replaces the free-text refusal contract.
