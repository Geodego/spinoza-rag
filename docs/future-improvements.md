# Later improvements

This page records deferred improvement ideas. These are hypotheses to test,
not implemented features or scheduled commitments. Measurements and known
limitations remain in [retrieval evaluation](evaluation.md).

## Adjacent-page context

**Status: deferred on 2026-11-12.** Revisit when retrieval coverage becomes
the focus again. Deferral does not resolve the known evidence gaps: the
[answer review log](answer-review.md) records answers whose cited passage ends
before the support the answer claims.

### Motivation and hypothesis

The [baseline failure analysis](evaluation.md#failure-analysis-and-evidence-limits)
shows that the conatus question retrieves an introduction before the passage
needed to explain the disputed translation. Hard page boundaries can separate
these parts of an argument.

Hypothesis: including the immediately following page for a leading result may
recover the missing continuation within a modest context budget. Its benefit
and the amount of irrelevant text it adds remain unmeasured.

### Bounded experiment

- Assemble context from persisted page/chunk artifacts and preserve source,
  physical PDF page and lecture provenance for every added passage.
- Define page and token budgets before evaluation. Charge all added context
  against those budgets; do not give neighboring pages free retrieval credit.
- Deduplicate overlapping context and handle document ends and session changes
  explicitly. Keep the original retrieved ranks available for inspection.
- Reuse the existing embeddings and index. No re-extraction, re-embedding,
  dependency addition or framework migration is needed for this experiment.

### How to resume and decide

1. Keep gold version `1.4.0-partial`, validated membership, reference lists and
   the original baseline frozen. Use the same ten questions and 50-chunk
   candidate budget. Define the experimental expansion/ranking protocol and
   context budgets explicitly, and fingerprint its configuration separately.
2. Save the experiment in a new ignored evaluation directory. Compare paired
   question results under equivalent context budgets and inspect the actual
   supporting passages, especially conatus and the relations/essences paraphrase.
   Keep corpus-derived diagnostic text outside Git.
3. Retain the option only if it brings missing evidence into the chosen budget
   without material coverage regressions or excessive irrelevant context.
   A higher Hit score alone is insufficient. If it does not help, record that
   result and reconsider the hypothesis before expanding the scope.

## Evaluation design

Equivalent-page groups and graded references are also deferred. Their existing
design and annotation requirements are described in
[deferred scoring](evaluation.md#limitations-and-deferred-scoring). They have
not been implemented and are not prerequisites for using the current baseline.

If tuning resumes, add independently reviewed held-out questions before claiming
general quality improvements; see
[fair comparison](evaluation.md#fair-comparison-and-the-next-decision).
