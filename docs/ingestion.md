# PDF extraction

The first stage turns a local PDF into page-level artifacts. It is the only
stage that opens the PDF; chunking, indexing, and search consume the saved
artifacts. For the commands, see the [README quickstart](../README.md#quickstart).

## Output artifacts

The default uses layout mode and writes to `storage/ingestion-layout/`, the
chunking stage's default input. The destination contains:

- `pages.jsonl`: one UTF-8 JSON object per physical PDF page, including empty or
  failed pages. Each has `schema_version`, `document_id`, `source_file`,
  `page_number` (one-based), `text`, and `diagnostics`.
- `report.json`: source SHA-256 identity and size, PDF metadata (title, author,
  subject, keywords, creator, producer, and raw date strings when available),
  pypdf version, extraction settings (including mode), page counts, per-page
  diagnostics, parser warnings, and a SHA-256 digest of the exact JSONL bytes.

No corpus is bundled. If `data/spinoza.pdf` is absent, the command reports the
missing file without publishing output. Tests generate their own synthetic PDFs.

## Source identity and page provenance

Only the source basename is persisted, not a machine-specific absolute path.
`document_id` is `sha256:<PDF byte digest>`; moving or renaming a source does not
change its identity. A changed PDF has a new identity. Page numbers refer to PDF
positions, not printed page labels. Chunks and index records retain these
identifiers.

## Diagnostics

Diagnostics flag:

- `empty_text`: blank, whitespace-only, or image-only pages with no extracted text.
- `short_text`: fewer than 40 non-whitespace characters by default.
- `replacement_characters` or `control_characters`: suspicious decoded text.
- `parser_warning`: pypdf recovered from an issue while processing a page.
- `extraction_error`: a page could not be extracted; its record retains its page
  number and error, and later pages are still processed.

The report's summary includes empty, failed, text-bearing, and review-page
counts. Document-level parser/metadata warnings are retained separately. Review
signals are heuristics, not proof of quality: a legitimate title page may be
short, and plausible-looking text can still have incorrect reading order.

## Exit status

The CLI prints the summary and artifact paths, never the extracted text. Exit 0
means extraction completed without failed pages and at least one page has text;
empty/short-page and other review flags may still exist. Exit 1 means an input or
output error, a page extraction failure, or no extractable text anywhere. In the
last two cases, artifacts are still published for diagnosis. Argument syntax
errors use argparse's standard exit 2. Encrypted and zero-page PDFs are rejected.

## Reruns and output safety

Results contain no run timestamps. With the same PDF basename, settings, and
locked extractor version, the CLI verifies the existing PDF fingerprint,
settings, JSONL checksum, page sequence/provenance, and report consistency before
invoking pypdf. Valid artifacts are reused without PDF extraction or file
rewrites; the CLI prints `Reusing verified artifacts; PDF extraction skipped.`
Only a missing destination triggers extraction. The component never appends
page records. Validation still reads the PDF bytes and existing JSON artifacts.

Incomplete, corrupt, or outdated artifacts fail early without re-extraction or
overwriting. Select a new destination for changed PDFs/settings or repairs.
Cached runs retain the original diagnostic summary and failure exit status.
The output schema version is also the extraction revision: bump it when changing
extraction behavior so older results cannot be silently reused.

Both artifacts are written into a temporary sibling directory before a single
rename publishes the directory. Write or publication errors leave no partial
new destination. Existing output must match both expected files exactly;
changed inputs/settings, incomplete output, or additional files cause a refusal
without changing existing files. Choose a new destination when re-extracting a
changed input or testing different settings:

```sh
uv run --locked python -m spinoza_rag.ingest data/spinoza.pdf \
  --output-dir storage/ingestion-v2 --min-page-characters 40
```

Use `--min-page-characters 0` to disable only the short-text heuristic. This is a
sequential local component; concurrent writers are outside its scope.

## Plain versus layout mode

The first plain-mode run extracted 531 pages without empty pages or parser
errors, but spot checks found spaces inside words. Layout mode fixes those
sampled word splits and orders page-number footers after the body text. It also
adds layout whitespace. Layout is therefore the default; the mode is recorded in
the report. The original plain output is preserved and remains reproducible:

```sh
uv run --locked python -m spinoza_rag.ingest \
  --extraction-mode plain --output-dir storage/ingestion
```

No heuristic report replaces visual review of representative PDF passages and
reading order.

## Decisions and limits

`extract_pdf()` performs extraction without writing artifacts;
`write_extraction()` handles serialization and publication. Extraction involves
no chunking or vector index; later stages consume the saved artifacts.

[pypdf](https://pypdf.readthedocs.io/en/stable/user/extract-text.html) was selected
as a pure-Python, [BSD-licensed](https://github.com/py-pdf/pypdf/blob/main/LICENSE)
starting point. Tests verify French accents, page provenance, metadata, and
failure behavior. The actual lecture PDF has been processed locally; sampled
plain and layout results were compared as described above. Full transcription
accuracy and complex reading order remain unverified. Inspect representative
passages against the PDF before choosing normalization or chunking rules.

Plain extraction normalizes only CRLF/CR line endings to LF. It preserves accents,
line breaks, headers, footers, and hyphenation; it does not infer paragraphs or
remove repeated text. pypdf performs no OCR, so scanned pages need a separate
future decision. The source and extracted text are held in memory, which is
appropriate for this initial single-document component; large-document memory
and extraction performance have not been systematically benchmarked.
