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
that PDF to guess whether extraction is out of date.

## Local model stack

The one-time setup (Ollama model and pinned tokenizer) is in the
[README](../README.md#one-time-setup). The pinned tokenizer SHA-256 is
`def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a`.
Only this tokenizer JSON is needed; PyTorch and Transformers are not required.
The model download is about 639 MB and the tokenizer about 11.4 MB. Neither is
stored in Git. Application commands never pull a missing model automatically.

The verified local stack uses Chroma 1.5.9, langchain-text-splitters 1.1.3,
tokenizers 0.22.2, and Ollama 0.40.16 on macOS Intel. Recent ONNX Runtime
releases, a transitive Chroma dependency, have no Intel macOS wheels, so uv must
resolve for that platform (`required-environments`); the lock currently selects
ONNX Runtime 1.23.2. See
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

## Embeddings and the Chroma index

Indexing uses `/api/embed` with `truncate=false`, validates vectors, and stores
normalized embeddings in a cosine Chroma collection. Documents are embedded
without a query prefix. Questions use Qwen's `Instruct: ...\nQuery: ...` format;
the instruction is recorded in the index configuration.

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
interrupted or unavailable-server run resumes only missing IDs. A vector batch
committed immediately before a checkpoint failure is recovered without embedding
it again. Search rejects incomplete indexes. Concurrent writers are outside this
sequential baseline's scope.

Index and search accept `--batch-size` (index only), `--timeout`, and
`--ollama-url`; a missing model or unreachable service produces an actionable
error. Schema/configuration changes are never automatically applied
to old indexes. Keep the existing output and choose a new directory to rebuild.

## Search output

Search reopens the persisted index and embeds only the question; document
vectors are read from disk, so a restarted process gives the same results
without re-embedding. Ollama must be running for query embeddings. No answer is
generated; the CLI prints the nearest passages, source filename, one-based PDF
page range, inferred lecture date, stable chunk ID, and cosine distance (smaller
is closer, not an answer-confidence score).
