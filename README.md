# Enterprise Knowledge Intelligence Platform

Enterprise knowledge intelligence platform for grounded Q&A over handbooks, PDFs, webpages, and internal documents.

## What it includes

- Document ingestion for DOCX, PDF, TXT, and webpages
- Semantic chunking with metadata on each chunk
- Hybrid retrieval over Qdrant vectors and BM25
- Lightweight reranking using overlap and sentence-level signals
- Query rewriting backed by conversation memory
- Inline citations and a source appendix in final answers
- Hallucination detection and faithfulness checks
- Evaluation harness with retrieval and answer-quality metrics
- REST API and admin dashboard with bearer-token auth

## Installation

Requires Python 3.11 or newer. From a fresh clone:

```powershell
python -m venv .venv
\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

On macOS or Linux, activate with `source .venv/bin/activate`.

The default local mode does not require an OpenAI API key. It uses a grounded extractive fallback. Set `OPENAI_API_KEY` to enable the OpenAI provider. See `.env.example` for supported configuration.

## Quick Start

Run these commands from the repository root:

```powershell
python -m production_rag doctor
python -m production_rag index-handbook
python -m production_rag ask-handbook "What is the annual leave entitlement?"
python -m production_rag evaluate-rag --json
python -m production_rag serve
```

For a normal clone with an activated virtual environment, replace the long bundled-runtime path with `python`.

## Useful commands

```powershell
# Ask with verbose tracing
python -m production_rag ask-handbook --verbose "What is the annual leave entitlement?"

# Ask with machine-readable JSON
python -m production_rag ask-handbook --json "What is the annual leave entitlement?"

# Continue a conversation
python -m production_rag ask-handbook --conversation-id <conversation-id> "What about unused leave?"

# Filter by metadata
python -m production_rag ask-handbook --metadata-filter page_number=1 "What is the annual leave entitlement?"

# Index a webpage
python -m production_rag index-webpage "https://example.com/policy"

# Index a PDF through the Qdrant pipeline
python -m production_rag index-handbook "path/to/policy.pdf"
```

## REST API and dashboard

- `GET /api/health`
- `GET /api/conversations`
- `GET /api/conversations/{conversation_id}`
- `POST /api/ask`
- `POST /api/index/webpage`
- `GET /` or `GET /dashboard` for the admin UI

If `PRODUCTION_RAG_API_TOKEN` is set, the API routes require `Authorization: Bearer <token>`.

## Evaluation

The default evaluation set measures:

- `recall@1`
- `recall@5`
- `MRR`
- `nDCG@5`
- context relevance
- answer correctness against expected answers
- faithfulness pass rate
- hallucination rate
- latency in milliseconds
- estimated input and output token usage

Evaluation cases can be supplied as JSONL with `--cases-path`. Each record contains `name`, `question`, `relevant_chunk_ids`, `expected_answer_contains`, and optionally `expects_no_answer`.

## Repository Layout

```text
production_rag/       Core ingestion, retrieval, generation, API, and evaluation code
scripts/              Sample document-generation utilities
output/               Small sample source document
data/                 Local Qdrant and SQLite state, ignored by Git
tmp/                  Temporary conversion files, ignored by Git
```

## Notes

The CLI, server, and evaluation harness all share the same retrieval, grounding, citation, and guardrail layers so the system behaves consistently across terminal, API, and dashboard usage.

## Operational Notes

- Embedded Qdrant permits one active process per storage directory. Stop `serve` before indexing or evaluation, or use standalone Qdrant for concurrent workloads.
- The sample handbook is a demonstration document, not legal advice. Replace it with authorized company content before deployment.
- Do not commit API keys, private documents, `data/`, model caches, or generated temporary files.
