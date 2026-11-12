# Retrieval evaluation

The evaluation measures whether **Qwen embeddings, Chroma search, and the current
PDF extraction/chunking** retrieve the passages needed for French philosophical
questions. It scores retrieval on its own, so that retrieval changes are judged
on page evidence rather than on the fluency of generated answers.

## Current status and next steps

| Component | Status |
| --- | --- |
| Evaluation CLI and synthetic metric tests | Implemented |
| [Partial gold set](../evaluation/gold.v1.json) | Version `1.4.0-partial`: 10 human-validated questions, 15 unchanged proposed questions |
| Equivalent-page groups and graded metrics | Deferred design; not implemented |
| Official retrieval baseline and quality decision | First partial baseline measured; retrieval experiment deferred to [later improvements](future-improvements.md) |

The first partial baseline uses the existing exact-page scorer and a frozen,
human-approved subset; see [results and decision](#first-official-partial-baseline).
Equivalent-page groups and graded scoring remain deferred. The measurement
covers retrieval only; answer generation is reviewed separately in the
[answer review log](answer-review.md).

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

The initial draft used the two-pass AI method: initial keyword search and direct
page reading, then two independent blind annotators with questions and page files
but no prior labels, followed by adjudication of added core pages. Neither pass
uses vector results to choose references. This avoids circular evaluation in
which the retriever effectively defines its own correct answers.

Numeric references contain **core** pages from that draft, with only the first page of
each equivalent transcription group. Lower-confidence supporting candidates and
alternate page groups are recorded in annotations, which the current scorer
ignores. Supporting candidates were not systematically reviewed. Per-question
relevance notes, exceptions and overlapping groups live in the gold file.

Exact annotator prompts, model identities, independent raw outputs and a detailed
adjudication log are unavailable. The merged labels and method are inspectable;
the AI annotation experiment cannot be fully replayed or independently audited.
AI annotation and adjudication do not constitute human validation. The ten
validated entries now carry GG's explicit human review; the other fifteen retain
their proposed status. Approval covers the numeric reference lists, not the
unstructured secondary candidates in the historical AI notes.

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

A gold set with no validated questions causes a nonzero exit before service
access and creates no report. The current partial set evaluates its ten validated
questions by default. For explicitly exploratory diagnostics, use `--include-proposed` and
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
and fragmented arguments before choosing a retrieval change.

## First official partial baseline

Measured on 2026-11-12 at 11:19:11 Europe/Paris (10:19:11 UTC), on
macOS 26.6.2 Intel (`x86_64`), Python 3.12.13.
The evaluation CLI ran against the saved index with the local Ollama HTTP
embedding client.

GG explicitly approved q01, q02, q03, q05, q08, q09, q11, q13, q17 and q26
on 2026-11-12, without changing their reference lists. Their 45 reference
entries are validated in gold version `1.4.0-partial`; all other 15 questions
remain proposed and unchanged. The review form left the version field blank;
the version was assigned when the approvals were imported into the gold file.
Original AI notes were retained as historical context; secondary candidates
were not promoted to references.

Labels and membership were frozen before retrieval. The human decisions,
import record and pre-review gold are retained in ignored
`storage/reviews/retrieval-pilot-2026-11-12/`. The run is in ignored
`storage/evaluations/baseline-partial-v1/report.json`.

### Protocol and aggregate results

Official validated-only stratum: 10 questions; 15 proposed questions excluded;
no exploratory stratum. Protocol `page-blocks-v1`, 50 candidate chunks per
question, exact physical PDF pages, unchanged flat references. Cutoffs count
unique pages rather than chunks. Aggregates are macro means across questions.

| k | Hit@k | Support Recall@k | MRR@k |
| --- | --- | --- | --- |
| 1 | 80.0% | 21.35% | 0.8000 |
| 3 | 100.0% | 40.94% | 0.8833 |
| 5 | 100.0% | 48.27% | 0.8833 |
| 10 | 100.0% | 64.20% | 0.8833 |

### Per-question results

First-hit rank determines Hit and truncated MRR: at k below that rank
both are zero; otherwise Hit is 1 and MRR is its reciprocal. Thus all
questions have Hit@3/5/10 = 1; q13 has MRR 0.5 and q26 has MRR 1/3
at those cutoffs, and the other eight have MRR 1. Full precision,
ranked pages, chunks and distances are in the local JSON report.

| ID / subject | First hit (PDF page; rank) | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Missing approved pages at 10 |
| --- | --- | --- | --- | --- | --- | --- |
| q01 Affectio / affectus | 198; 1 | 20.00% | 20.00% | 40.00% | 60.00% | 199, 201 |
| q02 Instant / lived change | 202; 1 | 14.29% | 28.57% | 28.57% | 42.86% | 198, 201, 287, 288 |
| q03 Conatus / effort | 69; 1 | 25.00% | 25.00% | 25.00% | 25.00% | 58, 70, 93 |
| q05 Power in act | 64; 1 | 50.00% | 50.00% | 50.00% | 100.00% | None |
| q08 Domination / impotence | 61; 1 | 33.33% | 66.67% | 66.67% | 66.67% | 211 |
| q09 Joy / sadness | 66; 1 | 12.50% | 37.50% | 37.50% | 62.50% | 205, 208, 209 |
| q11 Common notions | 306; 1 | 25.00% | 50.00% | 50.00% | 50.00% | 303, 305 |
| q13 Learning to swim | 298; 2 | 0.00% | 40.00% | 60.00% | 60.00% | 300, 450 |
| q17 Becoming free | 229; 1 | 33.33% | 66.67% | 100.00% | 100.00% | None |
| q26 Relations / singular essences | 477; 3 | 0.00% | 25.00% | 25.00% | 75.00% | 313 |

### Failure analysis and evidence limits

Returned passages were inspected from persisted chunk records. This is
a qualitative diagnostic, not a new human validation or an extra metric.
No reference lists or statuses were changed after seeing vector results.
All corpus text used for this inspection stays in ignored storage.

- **q01:** First chunk on page 198 names the distinction; the passage definition is in the returned page-287 chunk at chunk/page rank 5. Page 201 is only page rank 22.
- **q02:** Pages 202 and 200 lead; 199 arrives at page rank 8. Pages 287 and 288 are absent from the 50-chunk pool.
- **q03:** Only page 69 is credited through page rank 10. The direct effort discussion on page 70 is chunk rank 14 / page rank 13; page 58 is page rank 22 and page 93 is absent from the pool. The first hit supplies framing, not the full two-part answer.
- **q05:** Page 64 leads and contains the central claim. Two initial chunks share that page; page 65 is page rank 8. Raw chunk rank and unique-page rank differ.
- **q08:** The leading page-61 chunk directly explains power founded on sadness; page 60 is second. The explicit contrast on page 211 arrives at chunk rank 14 / page rank 13. This is a coverage gap, not a complete retrieval miss.
- **q09:** Five of eight approved pages appear by page rank 10. Page 208 is absent from the pool, 209 is page rank 12, and 205 is page rank 14. The leading pages already include definitions and explanatory evidence.
- **q11:** Page 306 leads with the cause / part-whole justification; page 304 is rank 3. Page 303 is absent from the pool and page 305 is page rank 21.
- **q13:** The unannotated top page 448 is a relevant introduction on inspection, but receives no official credit. Page 298 is the first approved hit at rank 2; the page-449 chunk at rank 4 explains rhythmic composition. Approved continuation pages 300 and 450 are page ranks 16 and 26.
- **q17:** All three approved pages appear by page rank 4. The first page-229 chunk explains becoming free but omits the experimental-learning passage later on that page; the page-230 chunk at rank 4 supplies selection and composition.
- **q26:** Top pages 346 and 325 discuss bodily composition rather than the requested contrast. The returned page-477 chunk at rank 3 explicitly states the limit of relational knowledge. Pages 451 and 314 arrive at page ranks 6 and 10; page 313 is absent from the pool.

There are no complete misses by k=3, but high Hit/MRR does not establish
complete answer support. Recall covers the approved non-exhaustive pages;
several are alternative explanations or consecutive parts of an argument.
Missing one is not automatically an answer failure. Unannotated results
such as page 448 remain unjudged for scoring. The ten selected questions
are small, correlated (especially q01/q02), and not a representative
statistical sample. No numeric acceptance threshold was agreed.

### Decision and deferred follow-up

The baseline review recommended a focused retrieval experiment for incomplete
conatus evidence before grounded generation. The q08 evidence is already useful
at rank 1; its lower page coverage alone does not justify a broad retrieval
overhaul or reranking.

On 2026-11-12 the adjacent-page experiment was postponed and ceased to be a
prerequisite for answer generation. Its hypothesis, scope and comparison criteria are recorded in
[later improvements](future-improvements.md#adjacent-page-context). The measured
coverage limitations remain, and the deferral does not establish complete
answer support. The gold set, baseline report and retrieval implementation were
not changed by this decision.

Answer generation was then built as the [research agent](research-agent.md) on
the same retrieval path, under constraints set here: keep the Ollama HTTP
integration, explicitly disable Qwen3 thinking, preserve citation provenance
through short source labels, measure Ollama prompt-processing and generation
times separately, and store corpus-derived examples outside Git. Reranking and
retrieval enhancements remain unimplemented.

### Frozen configuration fingerprints

| Item | Value |
| --- | --- |
| Gold version | `1.4.0-partial` |
| Gold exact-byte SHA-256 | `dda3c165aeb8c870fb58f49823697a5a4cd44c6656f2b52f9256338c6582d5fd` |
| PDF document ID | `sha256:66ddeacf8dc52cc7dde8156cd1fdd7766dc0c2587854f94a3b78442117966862` |
| Persisted pages SHA-256 | `77fbfefc622eef5b3e7ce2790bc40c53a3d6ad0460e7f3ef841c53338b0ef023` |
| Index configuration SHA-256 | `a49f779a2fd2ebe11ca4cb986ba4596576fe49547b46041dd35afa7c34b3f7dd` |
| Chunks manifest SHA-256 | `c3a268a85f054ec36591b4d92eaa9782e0e4fffd5132d4cb1af131a1f706e1d8` |
| Embedding model | `qwen3-embedding:0.6b` |
| Embedding digest | `ac6da0dfba84a81fdbfbaf330198c33cd77c4cdfc53e8bc50eb581914a15621d` |
| Ollama version | `0.40.16` |
| Collection / metric / dimensions / chunks | `spinoza / cosine / 1024 / 1194` |
| Chunking | `450 tokens; overlap target 70; hard page boundaries; revision 1` |
| Tokenizer SHA-256 | `def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a` |
| Extraction | `pypdf 7.0.1; layout; line_endings_only` |
| Runtime packages | `spinoza-rag 0.1.0; Chroma 1.5.9; tokenizers 0.22.2; splitter 1.1.3` |

Query instruction:

`Given a question about Deleuze's lectures on Spinoza, retrieve relevant lecture passages that answer the question.`

The JSON report also records SHA-256 hashes of evaluate.py, search.py,
config.py, index.py and embeddings.py, plus the complete index manifest.
Approximate search and tie ordering can vary on reruns. This baseline did
not measure inference latency or answer quality.

To reproduce with the same local artifacts, choose a fresh output path:

```sh
uv run --locked python -m spinoza_rag.evaluate \
  --gold-set evaluation/gold.v1.json --index-dir storage/index \
  --chunk-limit 50 --output-dir storage/evaluations/baseline-partial-v1-rerun
```

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

These earlier questions differ from the frozen partial baseline and are not a
fair before/after comparison. The measured failure analysis above supersedes
these informal investigation leads for the next retrieval decision.
