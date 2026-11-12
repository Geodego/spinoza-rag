# Configuration defaults

All defaults live in `src/spinoza_rag/config.py`, grouped by area: paths,
extraction, chunking, Ollama, embedding, index, search, and evaluation. Each stage's
default output is the next stage's default input, so the stages chain without
flags. Every CLI shows its effective defaults in `--help`; print them all with:

```sh
uv run --locked python -m spinoza_rag.config
```

CLI flags override defaults for one run. Library functions take explicit values
and fall back to the same defaults. The defaults say what the next run will
use; each artifact's manifest or report records what an existing run actually
used.

Several defaults enter configuration fingerprints: the extraction mode and
short-page threshold, the chunk sizes, the embedding model and query
instruction, and the collection name. Changing one makes existing artifacts
incompatible, so the next run requires a new output directory. Schema and
normalization revisions are code behavior, not settings, and stay in their
modules.

There are no environment variables or settings files; add one only when a
concrete need appears, documenting only settings the code consumes.

Evaluation defaults include `evaluation/gold.v1.json`, `storage/evaluations/baseline/`,
a 50-chunk candidate pool, and page cutoffs 1, 3, 5, and 10. The annotation
file is version controlled; generated run reports stay under ignored storage.
See [evaluation](evaluation.md) for the ranking protocol and validation gate.

## Local data directories

Keep source PDFs in `data/`, generated indexes and extraction output in
`storage/`, and downloaded model files in `models/`. These directories are
ignored by Git, as are legacy `chroma/`, PDF files, and GGUF weights.
