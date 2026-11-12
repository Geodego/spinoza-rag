# Current corpus observations

These are measurements on the current local PDF, not guarantees about other
documents, and informal spot checks, not a benchmark. There is no formal
gold-set retrieval evaluation yet.

## Chunking

The page artifacts produce **1,194 chunks** across 531 pages and 13 detected
lecture dates. Chunk lengths range from 1 to 450 tokens, with a median of 299;
seven fragments contain fewer than 20 tokens. The default can be evaluated and
changed later without mixing its index with another configuration.

## Index build and reruns

The first full index build embedded all 1,194 chunks as 1,024-dimensional vectors
in 29 minutes 37 seconds (`indexing_seconds` 1,776.9, batch size 8) on the macOS
Intel machine with Ollama 0.40.16. The Chroma directory occupies
about 21 MB. Rerunning indexing reported `embedded_this_run: 0` and
`reused: 1194` in under one second and left the index manifest byte-identical.
Chroma's SQLite file still changes on reopen, apparently from internal
bookkeeping, so its checksum is not a reliable "unchanged" signal. After the
reruns, Chroma still held 1,194 embeddings with a maximum write sequence of
1,194, one write per chunk.

## Retrieval spot checks

Six French questions were run in a fresh process with an embedder wrapper that
counted inputs. Each search embedded exactly one text, the question, and took
about 0.9 seconds after startup. All 30 top-5 passages carried the source
filename, PDF page, lecture date, and chunk ID. Each passage's whitespace-
normalized text was also found on its cited page in the saved page artifacts.
Top passages, with PDF pages and cosine distances:

| Question | Top PDF pages | Distances | Observation |
| --- | --- | --- | --- |
| Joie et tristesse | 66, 291, 292 | 0.21–0.25 | On topic |
| Notions communes | 308, 306, 304 | 0.17–0.25 | On topic, one lecture |
| Troisième genre de connaissance | 451, 314, 446 | 0.13–0.25 | On topic |
| Mal, morale et éthique | 35, 86, 137, 151 | 0.19–0.23 | On topic |
| Conatus | 69, then unrelated | 0.30, ~0.40 | One relevant hit, then weak |
| Puissance et pouvoir | 47, 64 | 0.30–0.32 | Related, no direct contrast |

Semantic accuracy is not established by fake-embedding tests. Page boundaries,
short fragments, retained headers, and imperfect extraction can affect
relevance. No reranking or answer generation is included in this milestone.
