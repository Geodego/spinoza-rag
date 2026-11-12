# Answer review log

Research-agent runs on the local corpus, with the source review of each answer.
These are individual runs, not an answer-quality benchmark; the retrieval
benchmark is in [retrieval evaluation](evaluation.md).

## Commands

Run these commands from the repository root. They require a built index and
both installed Ollama models. They do not ingest, rebuild or change the
retrieval baseline. Keep the logs local; they contain corpus passages and model
output. The runs below were recorded on 2026-11-12.

```sh
mkdir -p storage/research
uv run --locked python -m spinoza_rag.research context \
  'Quelle est la différence entre la joie et la tristesse ?' --top-k 2 \
  > storage/research/context.log 2>&1
uv run --locked python -m spinoza_rag.research answer \
  'Quelle est la différence entre la joie et la tristesse ?' --top-k 1 --timeout 600 \
  > storage/research/straightforward.log 2>&1
uv run --locked python -m spinoza_rag.research answer \
  'Comment relier le conatus et les trois genres de connaissance ? Cherche séparément les preuves pour ces deux thèmes avant de les rapprocher.' \
  --top-k 1 --timeout 600 > storage/research/multiple-searches.log 2>&1
uv run --locked python -m spinoza_rag.research answer \
  'Quel protocole expérimental Deleuze propose-t-il dans ces cours pour mesurer l’efficacité d’un vaccin à ARNm ?' \
  --top-k 1 --timeout 600 > storage/research/insufficient.log 2>&1
uv run --locked python -m spinoza_rag.research answer \
  'Quelle est la différence entre la joie et la tristesse ?' --top-k 2 --timeout 600 \
  > storage/research/straightforward-two-passages.log 2>&1
uv run --locked python -m spinoza_rag.research answer \
  'Quels numéros de version du noyau Linux sont cités dans le cours du 3 février 1981 ? Recherche les passages de ce cours avant de répondre.' \
  --top-k 1 --timeout 600 > storage/research/insufficient-after-search.log 2>&1
uv run --locked python -m spinoza_rag.research answer \
  'Selon les passages, quels éléments composent le premier genre de connaissance ? Limite la réponse à cette définition.' \
  --top-k 1 --timeout 600 > storage/research/definition.log 2>&1
uv run --locked python -m spinoza_rag.research answer \
  'Dans le cours sur les trois genres de connaissance, quel est le numéro de page de l’édition originale de l’Éthique que Deleuze utilise ? Cherche ce renseignement dans le corpus.' \
  --top-k 1 --timeout 600 > storage/research/insufficient-edition-page-fixed.log 2>&1
```

The last question was first run before the refusal fix described below, with
output in `storage/research/insufficient-edition-page.log`; it exited 1.

## Environment and interpretation

Informal checks on 2026-11-12, macOS Intel Core i9-9880H, 32 GiB RAM, CPU
inference (Ollama reported zero VRAM), Python 3.12.13, Ollama 0.40.16,
LangChain 1.5.0, langchain-ollama 1.1.0 and LangGraph 1.2.20.
Chat weights: installed `qwen3:8b`, Q4_K_M, digest prefix `500a1f067a9f`;
embedding weights: `qwen3-embedding:0.6b`, Q8_0, prefix `ac6da0dfba84`.
The existing 1,194-chunk index was reused without changing its fingerprint.

Each measurement is one run, not a benchmark average. Model loading, prompt caching,
CPU contention, and query choice affect timing. Temperature zero does not
promise byte-identical output. Parallel tool requests can change admission
order and therefore short labels between runs; labels stay stable within a run.
The trace always maps each label to its full chunk ID and metadata.

## Observed behavior

The context run admitted two passages (PDF pages 66 and 291) with labels,
lecture dates and full chunk IDs, used 3,693 evidence bytes and completed in
2.29 seconds without generation.

The one-passage answer made one search, used 2,174 evidence bytes and completed
in 99.44 seconds. Across its two model calls, Ollama reported 35.99 seconds of
prompt processing and 49.72 seconds of generation. Its `[S1]` label resolved
correctly to page 66. Source review found that the passage supports the sadness
part of the explanation but ends before explicitly supporting the joy-increase
claim. The fluent answer was therefore **not fully supported by its evidence**.

The two-topic run chose queries for conatus and the three kinds of knowledge,
issued both tool requests in its first model response, and admitted distinct
sources on PDF pages 69 and 314. It used 3,443 evidence bytes and completed in
102.99 seconds, with 40.17 seconds of prompt processing and 61.32 seconds of
generation. Both labels resolved. The passages supported topic definitions and
some related concepts, but the answer's proposed conatus-as-driving-force
connection exceeded those sources. Two searches did not establish full support.
This run issued both searches in one round; a sequential follow-up round is
covered by the scripted graph tests, not by this run.

The vaccine question completed in 6.61 seconds. The model skipped retrieval;
the application withheld its draft and returned the fixed no-evidence response.
This exercises the no-evidence guard, not semantic assessment of retrieved
passages, and does not prove that the corpus lacks an answer.

The two-passage answer used 3,692 evidence bytes and completed in 82.62 seconds,
with 38.80 seconds of prompt processing and 42.98 seconds of generation. Its
sources were again pages 66 and 291. The second source supported a discussion
of mixed affective lines, but did not close the missing support for the
joy-increase claim attributed to the first source. The final generalization
also went beyond the explicit passages. Increasing top-k alone did not fix
semantic support in this run.

The Linux question also skipped retrieval despite the explicit request to
search. The no-evidence guard returned its fixed response in 6.76 seconds. The
filename `insufficient-after-search.log` records the intended scenario; its
actual trace contains zero searches. This is a model instruction-compliance
limitation, not evidence of irrelevance detection.

A corpus-specific missing-detail question did trigger a search: it asked for
the page number in the original Ethics edition used for the discussion of the
three kinds of knowledge. The retrieved passage was on PDF page 2, without that
edition-specific number. The model produced the requested refusal plus a fixed
methodological caution, which the original validator rejected. A local replay
and a synthetic regression test reproduced the mismatch; commit `dda9968` fixed
the prompt boundary and narrowly accepted that harmless variant (see
[the knowledge entry](knowledge/agent-refusal-boundary.md)). This run and those
described above precede that fix; their timings and wording are historical
observations, not promised output from the current prompt.

The focused first-kind definition made one search, admitted `[S1]` from PDF
page 314 (lecture 1981-02-03), used 1,792 evidence bytes and completed in
45.27 seconds. Ollama reported 27.67 seconds of prompt processing and
16.34 seconds of generation across two model calls. Source review found the
short French answer supported by that passage, without the unsupported
cross-topic synthesis seen in the broader questions.

The corrected missing-detail run made one search, admitted one passage on PDF
page 489 (lecture 1981-03-17), then returned the exact insufficient-evidence
refusal. Source review confirmed that the admitted passage did not supply the
requested edition-page number. It used 1,604 evidence bytes and completed in
37.32 seconds, with 22.80 seconds of prompt processing and 13.32 seconds of
generation. The rephrased model query retrieved a different passage than the
pre-fix run; this is a behavioral check, not a controlled retrieval-quality
comparison.

## Review procedure

For every successful answer, inspect the actual admitted passage text and the
label-to-chunk mapping in the local log. Check each material claim against its
cited passage, including whether the excerpt ends before the needed support.
Compare source text and page/lecture metadata byte for byte with the persisted
chunk artifacts; every admitted passage in the runs above matched its chunk.
Do not substitute the presence of a label or a small cosine distance for that
review. Do not copy passages or generated answers into tracked documentation.
Local review notes are kept in `storage/research/source-review.json`.

The review above establishes observations for these runs only. There is no
answer-quality gold set or calibrated irrelevance threshold. A strict uncited
refusal format can reject otherwise reasonable paraphrases; errors are surfaced
instead of silently labelling those drafts grounded. Adjacent-page expansion,
reranking and retrieval/evaluation changes remain deferred. The gold set and
retrieval baseline were not modified or rerun for these runs.

See [the design tradeoffs](research-agent.md#design-tradeoffs) for the
architectural constraints behind these results. Unsupported synthesis remains
the main observed limitation of answer generation.
