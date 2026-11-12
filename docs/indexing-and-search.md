# Chunking, indexing, and search

These stages read the layout artifacts in `storage/ingestion-layout/` by
default. They never open the original PDF. Keep the page artifacts and chunk
artifacts alongside the index: their fingerprints are verified during indexing
and search. For the commands, see the [README quickstart](../README.md#quickstart).

## Stage inputs and outputs

| Stage | Input | Persisted output |
| --- | --- | --- |
| Ingest | `data/spinoza.pdf` | `storage/ingestion-layout/pages.jsonl` and `report.json` |
| Chunk | `storage/ingestion-layout/` and the local tokenizer | `storage/chunks/chunks.jsonl` and `manifest.json` |
| Index | `storage/chunks/` | `storage/index/chroma/` and `manifest.json` |
| Search | `storage/index/`, existing upstream artifacts, and a question | Printed passages; no document embedding generation |

Paths between artifacts are recorded relatively, so the storage tree can move
together. Staleness is checked against saved page/chunk artifacts. If the original
PDF changes, explicitly rerun extraction first; indexing and search never read
that PDF to guess whether extraction is out of date. Extraction itself is
described in [PDF extraction](ingestion.md).

## Local model stack

The one-time setup (Ollama model and pinned tokenizer) is in the
[README](../README.md#one-time-setup). The pinned tokenizer SHA-256 is
`def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a`.
Only this tokenizer JSON is needed; PyTorch and Transformers are not required.
The model download is about 639 MB and the tokenizer about 11.4 MB. Neither is
stored in Git. Application commands never pull a missing model automatically.

| Component | Used for | Verified version |
| --- | --- | --- |
| [tokenizers](https://github.com/huggingface/tokenizers) with the pinned Qwen3-Embedding tokenizer JSON | Counting chunk tokens | 0.22.2 |
| [langchain-text-splitters](https://reference.langchain.com/python/langchain-text-splitters/) `RecursiveCharacterTextSplitter` | Splitting page text into chunks | 1.1.3 |
| [Ollama](https://ollama.com/) server with `qwen3-embedding:0.6b` | Embedding chunks and questions (1,024 dimensions) | 0.40.16 |
| Python standard library (`urllib`) | Calling the Ollama HTTP API; no Ollama client package | Python 3.12 |
| [Chroma](https://docs.trychroma.com/) `PersistentClient` | Storing vectors, passages, and metadata; cosine nearest-neighbor search | 1.5.9 |

Library versions are those in `uv.lock` on 2026-11-12, which stays authoritative.
The Ollama version and model digest are recorded in each index manifest; the
row above is the one in the current local index, on macOS Intel.

Recent ONNX Runtime releases, a transitive Chroma dependency, have no Intel
macOS wheels, so uv must resolve for that platform (`required-environments`);
the lock currently selects ONNX Runtime 1.23.2. See
[the knowledge entry](knowledge/chroma-onnxruntime-intel-macos.md).
ONNX is not used for embeddings: Chroma's default embedding function is
explicitly disabled. Chroma telemetry is disabled.
The Ollama URL is restricted to HTTP loopback addresses, defaulting to
`http://127.0.0.1:11434`; environment HTTP proxies are bypassed.

## Chunking decisions and provenance

The defaults are 450 Qwen tokens and a 70-token **overlap target**. Tokens are
counted with the local Qwen tokenizer, excluding special tokens. LangChain's
recursive splitter prefers paragraphs, lines, sentences, then words; overlap can
be smaller at structural boundaries. Custom settings are explicit:

```sh
uv run --locked python -m spinoza_rag.chunk \
  --input-dir storage/ingestion-layout --output-dir storage/chunks-experiment \
  --chunk-tokens 500 --overlap-tokens 80
uv run --locked python -m spinoza_rag.index \
  --chunks-dir storage/chunks-experiment --index-dir storage/index-experiment
uv run --locked python -m spinoza_rag.search "Que sont les notions communes ?" \
  --index-dir storage/index-experiment --top-k 3
```

Chunks have hard page boundaries, so `page_start == page_end` in this baseline.
This makes citations exact but can split an argument across pages. Within-page
lecture boundaries are also respected: standalone `DD.MM.YYYY` headings become
ISO lecture dates, carried forward until the next heading. A page can have chunks
from two lectures. Dates are inferred, not authoritative annotations; cover and
contents pages may have no lecture date.

Whitespace normalization joins wrapped lines, preserves paragraph breaks, and
removes a matching page number only at page edges. It does not repair hyphenation,
remove arbitrary headers, summarize text, or invent section titles. Empty pages
are skipped; extraction-error pages block chunking rather than silently omitting
content. Very short fragments remain visible for later quality evaluation.

Every chunk stores the PDF document fingerprint, source filename, one-based page
range, page-local ordinal, text, Qwen token count, and available lecture identity.
Its stable ID hashes the record and exact chunking configuration. The manifest
records the original PDF metadata, page JSONL checksum, tokenizer checksum,
normalization revision, LangChain/tokenizer versions, and chunk JSONL checksum.
Identical chunking runs reuse verified output without rewriting it; changed inputs
or settings require a new output directory.

## Index build, step by step

`python -m spinoza_rag.index` runs `index_chunks` in `index.py`:

1. **Load and verify chunks.** `load_chunks` requires exactly `chunks.jsonl`
   and `manifest.json`, checks the chunk checksum, reopens the page artifacts
   to check that they are unchanged, and recomputes every chunk ID.
2. **Identify the embedding model.** `OllamaClient.identity` asks Ollama for
   the installed models (`GET /api/tags`) and its version (`GET /api/version`).
   A missing model stops the run with the `ollama pull` command to run.
3. **Fingerprint the configuration.** The chunk manifest hash, model name and
   digest, Ollama version, collection name, cosine metric, and query
   instruction form the index configuration; its SHA-256 is the pipeline
   fingerprint. An existing `manifest.json` must match exactly (see
   [compatibility](#index-compatibility-and-recovery)); otherwise a new one is
   written with state `building`.
4. **Open the collection.** Chroma's `PersistentClient` opens
   `storage/index/chroma/` and creates or reopens the collection with
   `embedding_function=None`, HNSW cosine space, and the fingerprint as
   collection metadata.
5. **Find what is already stored.** Every stored record's ID, text, and
   metadata is compared with the chunks; IDs already present are skipped.
6. **Embed missing chunks in batches** (8 by default). Each batch is one
   `POST /api/embed` request with `truncate=false`, so an overlong chunk fails
   instead of being cut, and `keep_alive` keeps the model loaded between
   batches. Documents are embedded without an instruction prefix. The client
   rejects a wrong vector count and empty, non-finite, or zero vectors, then
   L2-normalizes each vector itself. All vectors must share one dimension.
7. **Store, then checkpoint.** `collection.upsert` writes IDs, passages,
   metadata, and vectors; only then is `manifest.json` rewritten with the
   dimension and elapsed time. A batch stored just before a failed checkpoint
   is found at step 5 next time and not embedded again.
8. **Verify and mark ready.** After the last batch, the collection must hold
   exactly one unmodified record per chunk; the manifest state becomes `ready`.

The command prints a JSON summary with `embedded_this_run`, `reused`,
`dimensions`, and timings.

## Query path, step by step

`python -m spinoza_rag.search` and the [research agent](agent-workflow.md)
both call `search` in `search.py`. The
[evaluation](evaluation.md) opens the index once and then calls the same
`search_collection` for every question.

1. **Validate the request.** The question must be nonempty and `top_k`
   positive (5 by default).
2. **Reopen and verify the index** (`open_index`). The index manifest and
   `chroma.sqlite3` must exist and the state must be `ready`. The chunk and
   page artifacts are verified again as in index step 1, Ollama is asked again
   for the model digest and version, and the recomputed configuration must
   equal the saved one, so a model re-pulled under the same tag is rejected.
   The collection is opened with `embedding_function=None`, and every stored
   record, the record count, and the vector dimension are checked. These
   checks read the whole collection on each search, which is cheap at the
   current 1,194 chunks.
3. **Embed the question only.** The question is wrapped in Qwen's query
   format, `Instruct: <query instruction>\nQuery: <question>`, using the
   instruction in [configuration](configuration.md), and sent through the same
   `POST /api/embed` call and normalization as documents. Its length must
   equal the index dimension.
4. **Find nearest passages.** `collection.query` asks Chroma for
   `min(top_k, count)` neighbors of the question vector in the HNSW cosine
   index, including passages, metadata, and distances. Document vectors are
   read from disk, so a restarted process gives the same results without
   re-embedding.
5. **Return results.** Each result keeps Chroma's order and carries the chunk
   ID, passage text, provenance metadata, and cosine distance. The Chroma
   client is closed.

The search CLI prints, for each result, the source filename, one-based PDF
page range, inferred lecture date, stable chunk ID, cosine distance, and
passage. Smaller distances are closer; a distance is not an
answer-confidence score, and the nearest neighbors are returned even for
unrelated questions. No answer is generated: the CLI stays retrieval-only.
Ollama must be running, since the question is embedded at query time.

## Embedding storage

Chunk embeddings are stored only in `storage/index/chroma/`. Neither
`chunks.jsonl` nor the index `manifest.json` contains vectors; the manifest
records only their dimensions. Each vector is a unit-normalized embedding of
1,024 float32 values, stored with the chunk ID, text, and provenance metadata.
Read them back through Chroma, for example with
`collection.get(include=["embeddings"])`, not from the files.

Treat the `chroma/` directory as one opaque unit and copy, back up, or delete it
whole. The layout is a Chroma implementation detail, observed with Chroma 1.5.9
in the current index:

- `chroma.sqlite3` holds IDs, passages, metadata, and a write-ahead log that
  also stores raw vectors.
- A UUID-named folder holds the HNSW vector index, written to disk in batches.
  It held 1,000 of the 1,194 vectors; Chroma replays the remaining 194 from the
  SQLite log when the index opens.

Copying only the HNSW files would therefore silently lose recent vectors.

## Index compatibility and recovery

The index fingerprint binds the chunk manifest, exact installed Ollama model
digest and runtime version, collection, distance metric, and query instruction.
Changing any of these is rejected with a request for a new index directory;
weights behind a reused model tag cannot be silently mixed. Chroma record IDs,
passages, and metadata are checked against persisted chunks when reopening.

Stable IDs and upserts prevent duplicates. Repeated indexing reports
`embedded_this_run: 0` when complete. Each completed batch is checkpointed; an
interrupted or unavailable-server run resumes only missing IDs. Search rejects
incomplete indexes. Concurrent writers are outside this sequential baseline's
scope.

Index and search accept `--batch-size` (index only), `--timeout`, and
`--ollama-url`; a missing model or unreachable service produces an actionable
error. Schema/configuration changes are never automatically applied
to old indexes. Keep the existing output and choose a new directory to rebuild.

## Evaluating retrieval

The [evaluation CLI](evaluation.md) calls the same saved-index search without
changing the pipeline or re-embedding documents. It scores unique PDF pages
against explicitly validated annotations; proposed references are kept separate.
