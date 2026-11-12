# Retrieval evaluation

This milestone measures whether **Qwen embeddings, Chroma search, and the current
PDF extraction/chunking** retrieve the passages needed for French philosophical
questions. Its purpose is to establish retrieval evidence before choosing
retrieval improvements or introducing answer generation.

## Current status and next steps

| Deliverable | Status |
| --- | --- |
| Evaluation CLI and synthetic metric tests | Implemented |
| [Candidate gold set](../evaluation/gold.v1.json) | AI-annotated draft; every question remains `proposed`, with no human validation |
| Equivalent-page groups and graded metrics | Deferred design; not implemented |
| Official retrieval baseline and quality decision | Pending; no official metrics measured |

Complete the milestone by resolving the scoring design, validating the final
references, freezing their file hash/version, running the baseline, and reviewing
per-question failures. If an explicitly approved subset is used, identify it as
a partial baseline. Choose retrieval changes or answer generation only after
that review. No retrieval optimization or answer generation is included here.

## Annotation format and provenance

Schema 1 is strict JSON: unknown/missing fields, duplicate question IDs/pages,
invalid dates and unsupported discovery methods are rejected. It contains no
corpus passages. The annotation file records:

- `version`, `document_id` (PDF SHA-256) and `pages_sha256` (persisted page-file
  checksum). Both corpus fingerprints must match the index's upstream artifacts.
- `discovery`: method, date and a durable description of the annotation process.
  Allowed methods are `keyword_search_and_direct_page_inspection` and
  `two_pass_ai_keyword_and_direct_page_inspection`.
- `questions`: `id`, `question_fr`, `status` (`proposed` or `validated`), nonempty
  `references` and optional `annotation`. Each reference has a unique positive
  `pdf_page`, optional ISO `lecture_date` and optional `annotation`.
- `validation`: required only for validated questions, with `reviewer` and ISO
  `validated_on`. Proposed items cannot carry this record.

The current draft uses the two-pass AI method: initial keyword search and direct
page reading, then two independent blind annotators with questions and page files
but no prior labels, followed by adjudication of added core pages. Neither pass
uses vector results to choose references. This avoids circular evaluation in
which the retriever effectively defines its own correct answers.

Numeric references contain proposed **core** pages, with only the first page of
each equivalent transcription group. Lower-confidence supporting candidates and
alternate page groups are recorded in annotations, which the current scorer
ignores. Supporting candidates were not systematically reviewed. Per-question
relevance notes, exceptions and overlapping groups live in the gold file.

Exact annotator prompts, model identities, independent raw outputs and a detailed
adjudication log are unavailable. The merged labels and method are inspectable;
the AI annotation experiment cannot be fully replayed or independently audited.
AI annotation and adjudication do not constitute human validation.

### Human validation

1. Read each candidate and neighbouring page in persisted extraction; confirm
   physical PDF numbering, wording and relevance. Inspect the PDF manually when
   extraction is ambiguous. Use lexical search, never vector output, for labels.
2. Correct insufficient references. These are non-exhaustive supporting evidence,
   not all relevant pages or necessarily alternative complete answers. Review
   secondary candidates before allowing them to contribute to any future metric.
3. Explicitly approve a complete question/reference list, then set `validated`
   and add `validation`, for example
   `{"reviewer": "reviewer's name", "validated_on": "2026-11-12"}`.
   This example confers no approval; incomplete review leaves the item proposed.
4. Bump the set version, inspect the diff and freeze its exact bytes before runs.

Inspect pages locally without Ollama, vectors or re-extracting the PDF:

```sh
uv run --locked python - <<'PY'
from spinoza_rag.config import PATHS
from spinoza_rag.ingest import load_extraction

document, _ = load_extraction(PATHS.pages_dir)
keyword = "conatus"
print([p.page_number for p in document.pages
       if keyword.casefold() in p.text.casefold()])
print(document.pages[70 - 1].text)
PY
```

## Ranking and exact metrics

Search and evaluation share `search_collection`. Evaluation keeps one verified
`open_index` context for the run; it checks saved-index/upstream integrity and
model compatibility at open, then embeds only selected questions. The same
query instruction, vector checks and Chroma query are used by ordinary search.
A final manifest/annotation check refuses reports if these inputs change during
the run. Concurrent index writers are unsupported.

`--chunk-limit` defaults to 50, fixed independently of labels. Retrieve that many
chunks (or all in a smaller index), then apply `page-blocks-v1`:

1. Preserve Chroma's chunk order, including equal-distance order.
2. Expand each inclusive page range to previously unseen physical PDF pages.
   Repeated and overlapping pages receive no second slot or credit.
3. All new pages in one chunk form an indivisible block. Assign each its block's
   ending rank: the cumulative unique-page count through that chunk.
4. At k include only complete blocks ending at or before k; a block crossing the
   cutoff is excluded, and later blocks cannot fill its slots.

Example: chunks `[2], [2], [4..6], [5..7]` produce page 2 at rank 1, pages 4/5/6
at rank 4, and page 7 at rank 5. Only page 2 counts at k=3. This conservative rule
prevents a broad multi-page chunk from receiving cheap top-1 credit. Current
chunk artifacts enforce one page per chunk; the general converter remains tested
for multi-page inputs without relaxing the upstream invariant.

Let S be a question's numeric reference-page set and P(k) its admitted pages:

| Metric at k = 1, 3, 5, 10 | Definition |
| --- | --- |
| Hit@k | 1 if P(k) intersects S, otherwise 0 |
| Recall@k | Number of matched distinct reference pages divided by size of S |
| MRR@k per question | 1/r for the first reference page at assigned rank r <= k; otherwise 0 |

Aggregates are equal-weight macro means over questions. MRR is truncated at k.
`official` includes only validated questions. `--include-proposed` adds a separate
`exploratory_proposed` stratum; it never changes official membership. Empty strata
report `metrics: null`, not zero. Short result lists use available complete blocks
at every cutoff without shrinking the recall denominator. The report includes
per-question metrics, ranked pages and corpus-free chunk IDs/ranges/distances.

### Limitations and deferred scoring

Recall is **annotated-support coverage**, not exhaustive relevance recall.
Unannotated pages are unjudged; an apparent miss may be another valid passage.
A credited page may contain a returned chunk that omits the evidence. Hit/MRR
need only one reference, so they do not establish complete synthesis coverage.
More alternative references can lower recall without making an answer worse.

The current scorer treats all numeric references as the same flat set. It does
not interpret equivalent-page or relevance-grade notes. Duplicate transcription
copies can therefore be unfairly missed. The deferred proposal requires explicit
approval: structured page groups with core/supporting grades, recall over core,
and hit/MRR over core plus supporting, with synthetic tests for both. Migration
must use reviewed structured fields rather than parse annotation prose. Resolve
shared-page groups and grade conflicts before implementing that design.

## Reproducing a run

With local Ollama running and the exact indexed model/digest available:

```sh
uv run --locked python -m spinoza_rag.evaluate \
  --gold-set evaluation/gold.v1.json --index-dir storage/index \
  --output-dir storage/evaluations/baseline-v1
```

The all-proposed draft causes a nonzero exit before service access and creates
no report. For explicitly exploratory diagnostics, use `--include-proposed` and
another new directory. Defaults are in [configuration](configuration.md); every
flag has `--help`. The default output is `storage/evaluations/baseline`.
Existing output paths, including symlinks, are rejected before embedding. Use a
fresh name for each run; publication also preserves files if a path appears late.

`report.json` records UTC execution time, set version/exact hash, corpus/index
fingerprints, index manifest (model/digest, Ollama version, query instruction),
chunking/tokenizer/extraction settings, installed package versions, retrieval
budget/protocol, and source hashes including the embedding client. Store local
reports under ignored `storage/`, and approved measurement summaries here.
For a separate worktree, see [development](development.md#evaluating-from-a-worktree).
Synthetic checks need neither the corpus nor Ollama; see
[development checks](development.md#checks).

## Fair comparison and the next decision

Freeze corpus extraction, questions, annotations, validated membership, page
protocol, cutoffs and candidate budget before comparing configurations. Build
intended changes in separate indexes and record the changed fingerprints. Do not
compare aggregates from different label sets or use retrieval results to revise
labels in favour of one configuration. Inspect paired per-question changes and
apparent misses locally before attributing them to retrieval quality.

Approximate HNSW search and equal-distance ordering limit exact repeatability.
A larger chunk pool can change results; different chunk sizes expose different
amounts of text at a fixed budget. The small, correlated, deliberately selected
question set is a pilot, not a statistical guarantee; reserve independently
annotated held-out questions if tuning follows. There is no agreed numeric
acceptance threshold. Review coverage, alternate valid pages, duplicate evidence
and fragmented arguments before choosing retrieval optimization or generation.

## Earlier informal observations

These observations predate the gold-set framework and are not benchmark labels.
On the local 531-page PDF, chunking produced 1,194 chunks across 13 detected
lecture dates, ranging from 1 to 450 tokens (median 299); seven had under 20 tokens.
The full 1,024-dimensional index build took 29 min 37 s on macOS Intel with
Ollama 0.40.16 and batch size 8; a compatible rerun embedded zero chunks. The
Chroma directory occupied about 21 MB. Its internal SQLite checksum can change
on reopen, so use manifest/collection verification rather than byte identity.

Six French spot checks embedded one question each and took about 0.9 s after
startup. All 30 top-5 passages were found on their cited persisted pages:

| Question | Leading PDF pages | Cosine distances | Informal observation |
| --- | --- | --- | --- |
| Joie et tristesse | 66, 291, 292 | 0.21–0.25 | On topic |
| Notions communes | 308, 306, 304 | 0.17–0.25 | On topic, one lecture |
| Troisième genre de connaissance | 451, 314, 446 | 0.13–0.25 | On topic |
| Mal, morale et éthique | 35, 86, 137, 151 | 0.19–0.23 | On topic |
| Conatus | 69, then unrelated | 0.30, ~0.40 | One relevant hit, then weak |
| Puissance et pouvoir | 47, 64 | 0.30–0.32 | Related, no direct contrast |

The last two are investigation leads, not sufficient evidence for optimization.
Extraction quality, hard page boundaries and short/header fragments remain open
questions for a validated baseline.
