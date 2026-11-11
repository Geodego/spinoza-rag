# Spinoza RAG

A fully local Retrieval-Augmented Generation (RAG) project for exploring Gilles Deleuze's lectures on Spinoza.

The project aims to provide grounded answers to philosophical questions, supported by references to the original lecture transcripts.

## Objectives

- Build a reproducible document ingestion pipeline.
- Preserve PDF page references throughout extraction, chunking, and retrieval.
- Implement semantic search over the lecture transcripts.
- Generate answers grounded in retrieved passages, with verifiable citations.
- Evaluate retrieval quality and answer faithfulness.
- Explore LangGraph-based orchestration where it provides measurable benefits.

## Architecture

The application separates document ingestion from question answering.

**Offline ingestion**

PDF → Text extraction → Page-aware chunking → Local embeddings → Persistent vector index

**Question answering**

User question → Semantic retrieval → Local LLM → Source-grounded answer

Documents are processed and indexed offline. The application reuses the persisted index rather than reprocessing the PDF for each query.

## Technology Stack

- **Python 3.12** — Application development
- **uv** — Dependency and environment management
- **Ollama** — Local LLM inference and embeddings
- **LangChain** — Document processing and retrieval
- **LangGraph** — Workflow orchestration
- **Chroma** — Persistent vector storage
- **Ruff** — Linting and formatting
- **pytest** — Automated testing

## Getting Started

The application is under development. Installation and usage instructions will be added as the ingestion and retrieval pipelines become operational.

## Project Status

**Work in progress.**

The initial development focuses on document extraction, page-level provenance, persistent indexing, and source-grounded question answering.

Agentic retrieval strategies and systematic evaluation are planned for subsequent iterations.

## Corpus and Licensing

The project is designed around transcripts of Gilles Deleuze's lectures on Spinoza.

The source PDF and extracted text are not included in this repository. Users must provide their own lawfully obtained copy of the material.

The repository contains application code and documentation, not the underlying copyrighted lecture transcripts.