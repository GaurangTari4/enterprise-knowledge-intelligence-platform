from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

from docx import Document

from .chunking import chunk_text
from .embeddings import BgeM3Embedder
from .eval_runner import DEFAULT_EVALUATION_CASES, format_evaluation_results, load_cases_from_jsonl, run_evaluation
from .pipeline import RagPipeline
from .server import RagAdminServer
from .qdrant_index import QdrantVectorIndex, chunks_to_preview
from .pdf_ingest import extract_pdf_text, iter_page_summaries
from .web_ingest import fetch_webpage_text


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STORAGE_PATH = PROJECT_ROOT / "data" / "qdrant"
DEFAULT_CONVERSATION_STORE_PATH = PROJECT_ROOT / "data" / "production_rag.sqlite3"
DEFAULT_HANDBOOK_PATH = PROJECT_ROOT / "output" / "docx" / "employee_handbook_sample.docx"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="production-rag")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser("extract-pdf", help="Extract page text from a PDF")
    extract_parser.add_argument("pdf_path", type=Path, help="Path to the PDF file")
    extract_parser.add_argument("--max-length", type=int, default=800, help="Preview length per page")

    chunk_parser = subparsers.add_parser("chunk-text", help="Chunk a text or DOCX source")
    chunk_parser.add_argument("input_path", type=Path, help="Path to a .txt or .docx source")
    chunk_parser.add_argument("--min-tokens", type=int, default=500)
    chunk_parser.add_argument("--max-tokens", type=int, default=800)
    chunk_parser.add_argument("--overlap-tokens", type=int, default=75)
    chunk_parser.add_argument("--show", type=int, default=5, help="Number of chunks to print")

    embed_parser = subparsers.add_parser("embed-text", help="Embed a text or DOCX source")
    embed_parser.add_argument("input_path", type=Path, help="Path to a .txt or .docx source")
    embed_parser.add_argument("--min-tokens", type=int, default=500)
    embed_parser.add_argument("--max-tokens", type=int, default=800)
    embed_parser.add_argument("--overlap-tokens", type=int, default=75)
    embed_parser.add_argument("--show", type=int, default=1, help="Number of chunk embeddings to print")

    index_parser = subparsers.add_parser("index-handbook", help="Chunk, embed, and store a DOCX, TXT, or PDF document in Qdrant")
    index_parser.add_argument(
        "input_path",
        type=Path,
        nargs="?",
        default=DEFAULT_HANDBOOK_PATH,
        help="Path to a DOCX, TXT, or PDF source",
    )
    index_parser.add_argument("--storage-path", type=Path, default=DEFAULT_STORAGE_PATH)
    index_parser.add_argument("--collection-name", type=str, default="production_rag_chunks")
    index_parser.add_argument("--min-tokens", type=int, default=500)
    index_parser.add_argument("--max-tokens", type=int, default=800)
    index_parser.add_argument("--overlap-tokens", type=int, default=75)

    query_parser = subparsers.add_parser("query-qdrant", help="Query the Qdrant collection with a question")
    query_parser.add_argument("query", type=str, help="Query text")
    query_parser.add_argument("--storage-path", type=Path, default=DEFAULT_STORAGE_PATH)
    query_parser.add_argument("--collection-name", type=str, default="production_rag_chunks")
    query_parser.add_argument("--limit", type=int, default=5)
    query_parser.add_argument("--metadata-filter", action="append", default=[], help="Optional metadata filter as key=value or JSON")

    hybrid_parser = subparsers.add_parser("query-hybrid", help="Query the handbook with hybrid retrieval")
    hybrid_parser.add_argument("query", type=str, help="Query text")
    hybrid_parser.add_argument("--storage-path", type=Path, default=DEFAULT_STORAGE_PATH)
    hybrid_parser.add_argument("--collection-name", type=str, default="production_rag_chunks")
    hybrid_parser.add_argument("--limit", type=int, default=5)
    hybrid_parser.add_argument("--candidate-k", type=int, default=20)
    hybrid_parser.add_argument("--metadata-filter", action="append", default=[], help="Optional metadata filter as key=value or JSON")

    answer_parser = subparsers.add_parser("ask-handbook", help="Retrieve top chunks and answer using the context")
    answer_parser.add_argument("question", type=str, help="Question to answer")
    answer_parser.add_argument("--storage-path", type=Path, default=DEFAULT_STORAGE_PATH)
    answer_parser.add_argument("--collection-name", type=str, default="production_rag_chunks")
    answer_parser.add_argument("--limit", type=int, default=5)
    answer_parser.add_argument("--conversation-id", type=str, default=None, help="Continue an existing conversation")
    answer_parser.add_argument("--metadata-filter", action="append", default=[], help="Optional metadata filter as key=value or JSON")
    answer_parser.add_argument("--verbose", action="store_true", help="Show prompt and retrieved chunks")
    answer_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    eval_parser = subparsers.add_parser("evaluate-rag", help="Run the RAG evaluation harness")
    eval_parser.add_argument("--storage-path", type=Path, default=DEFAULT_STORAGE_PATH)
    eval_parser.add_argument("--collection-name", type=str, default="production_rag_chunks")
    eval_parser.add_argument("--cases-path", type=Path, default=None, help="Optional JSONL file with evaluation cases")
    eval_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    doctor_parser = subparsers.add_parser("doctor", help="Check the local runtime and handbook setup")
    doctor_parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_HANDBOOK_PATH,
        help="Path to the sample handbook source",
    )
    doctor_parser.add_argument("--storage-path", type=Path, default=DEFAULT_STORAGE_PATH)
    doctor_parser.add_argument("--collection-name", type=str, default="production_rag_chunks")
    doctor_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    index_web_parser = subparsers.add_parser("index-webpage", help="Fetch a webpage, chunk it, and store it in Qdrant")
    index_web_parser.add_argument("url", type=str, help="Webpage URL to ingest")
    index_web_parser.add_argument("--document-title", type=str, default=None, help="Optional title override")
    index_web_parser.add_argument("--document-id", type=str, default=None, help="Optional document id override")
    index_web_parser.add_argument("--storage-path", type=Path, default=DEFAULT_STORAGE_PATH)
    index_web_parser.add_argument("--collection-name", type=str, default="production_rag_chunks")
    index_web_parser.add_argument("--page-number", type=int, default=1)
    index_web_parser.add_argument("--min-tokens", type=int, default=500)
    index_web_parser.add_argument("--max-tokens", type=int, default=800)
    index_web_parser.add_argument("--overlap-tokens", type=int, default=75)

    serve_parser = subparsers.add_parser("serve", help="Run the admin dashboard and REST API")
    serve_parser.add_argument("--host", type=str, default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--storage-path", type=Path, default=DEFAULT_STORAGE_PATH)
    serve_parser.add_argument("--collection-name", type=str, default="production_rag_chunks")
    serve_parser.add_argument("--conversation-store-path", type=Path, default=DEFAULT_CONVERSATION_STORE_PATH)
    serve_parser.add_argument("--api-token", type=str, default=None)

    return parser


def cmd_extract_pdf(pdf_path: Path, max_length: int) -> int:
    result = extract_pdf_text(pdf_path)
    print(f"PDF: {result.path}")
    print(f"Pages: {result.page_count}")
    print()
    for summary in iter_page_summaries(result, max_length=max_length):
        print(summary)
        print()
    return 0


def _read_text_source(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return path.read_text(encoding="utf-8")
    if suffix == ".docx":
        doc = Document(path)
        return "\n".join(paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip())
    if suffix == ".pdf":
        return extract_pdf_text(path).full_text
    raise ValueError(f"Unsupported text source: {path.suffix}")


def _supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("PRODUCTION_RAG_COLOR") not in {"1", "true", "TRUE", "yes", "YES"}:
        return False
    return bool(getattr(sys.stdout, "isatty", lambda: False)())


def _ansi(text: str, code: str, enabled: bool) -> str:
    if not enabled:
        return text
    return f"\x1b[{code}m{text}\x1b[0m"


def _print_section(title: str, *, color: bool = False) -> None:
    print(_ansi(title, "1;36", color))
    print(_ansi("-" * len(title), "36", color))


def _print_label(label: str, value: str, *, color: bool = False, label_code: str = "1;37", value_code: str = "0") -> None:
    print(f"{_ansi(label, label_code, color)} {_ansi(value, value_code, color)}")


def _response_payload(question: str, collection_name: str, storage_path: Path, result) -> dict:
    return {
        "question": question,
        "collection_name": collection_name,
        "storage_path": str(storage_path),
        "response": asdict(result),
    }


def _parse_metadata_filters(values: list[str] | None) -> dict[str, object]:
    filters: dict[str, object] = {}
    for raw in values or []:
        raw = raw.strip()
        if not raw:
            continue
        if raw.startswith("{"):
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError("metadata filter JSON must be an object")
            filters.update(payload)
            continue
        if "=" not in raw:
            raise ValueError("metadata filter must use key=value or JSON syntax")
        key, value = raw.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError("metadata filter key cannot be empty")
        if value.lower() in {"true", "false"}:
            parsed: object = value.lower() == "true"
        else:
            try:
                parsed = int(value)
            except ValueError:
                try:
                    parsed = float(value)
                except ValueError:
                    parsed = value
        filters[key] = parsed
    return filters


def _summary_payload(collection_name: str, storage_path: Path, results, summary) -> dict:
    return {
        "collection_name": collection_name,
        "storage_path": str(storage_path),
        "summary": asdict(summary),
        "results": [asdict(result) for result in results],
    }


def cmd_chunk_text(input_path: Path, min_tokens: int, max_tokens: int, overlap_tokens: int, show: int) -> int:
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    text = _read_text_source(input_path)
    chunks = chunk_text(text, min_tokens=min_tokens, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
    print(f"Source: {input_path}")
    print(f"Chunks: {len(chunks)}")
    print(f"Token range: {min_tokens}-{max_tokens}")
    print(f"Overlap: {overlap_tokens}")
    print()
    for chunk in chunks[:show]:
        print(f"Chunk {chunk.index}")
        print(f"Tokens: {chunk.token_count} | Sentences: {chunk.sentence_count} | Overlap from previous: {chunk.overlap_from_previous_tokens}")
        print(chunk.text)
        print()
    return 0


def cmd_embed_text(input_path: Path, min_tokens: int, max_tokens: int, overlap_tokens: int, show: int) -> int:
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    text = _read_text_source(input_path)
    chunks = chunk_text(text, min_tokens=min_tokens, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
    embedder = BgeM3Embedder()

    print(f"Source: {input_path}")
    print(f"Chunks: {len(chunks)}")
    print(f"Model: {embedder.model_name}")
    if embedder.load_error is not None:
        print(f"Model load fallback: {type(embedder.load_error).__name__}: {embedder.load_error}")
    print()

    for chunk in chunks[:show]:
        vector = embedder.embed_one(chunk.text)
        print(f"Chunk {chunk.index}")
        print(f"len(embedding): {len(vector)}")
        print(f"Tokens: {chunk.token_count} | Sentences: {chunk.sentence_count}")
        print(chunk.text[:400])
        print()
    return 0


def cmd_index_handbook(
    input_path: Path,
    storage_path: Path,
    collection_name: str,
    min_tokens: int,
    max_tokens: int,
    overlap_tokens: int,
) -> int:
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    text = _read_text_source(input_path)
    index = QdrantVectorIndex(storage_path=storage_path, collection_name=collection_name)
    document_id = input_path.stem
    chunks = index.index_document(
        text,
        document_id=document_id,
        document_title=input_path.stem.replace("_", " ").title(),
        page_number=1,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )
    print(f"Indexed document: {input_path}")
    print(f"Collection: {collection_name}")
    print(f"Storage: {storage_path}")
    print(f"Chunks stored: {len(chunks)}")
    print()
    for line in chunks_to_preview(chunks, limit=5):
        print(line)
    return 0


def cmd_query_qdrant(
    query: str,
    storage_path: Path,
    collection_name: str,
    limit: int,
    metadata_filters: dict[str, object] | None = None,
) -> int:
    index = QdrantVectorIndex(storage_path=storage_path, collection_name=collection_name)
    results = index.query(query, limit=limit, metadata_filters=metadata_filters)
    print(f"Query: {query}")
    print(f"Collection: {collection_name}")
    print(f"Storage: {storage_path}")
    print()
    for position, result in enumerate(results, start=1):
        print(f"Result {position}")
        print(f"Score: {result.score:.4f}")
        print(f"Document: {result.document_title}")
        print(f"Page: {result.page_number}")
        print(f"Chunk: {result.chunk_id}")
        print(result.text)
        print()
    return 0


def cmd_query_hybrid(
    query: str,
    storage_path: Path,
    collection_name: str,
    limit: int,
    candidate_k: int,
    metadata_filters: dict[str, object] | None = None,
) -> int:
    pipeline = RagPipeline.from_storage_path(
        storage_path,
        collection_name=collection_name,
        top_k=limit,
        candidate_k=candidate_k,
    )
    results = pipeline.retrieve(query, limit=limit, metadata_filters=metadata_filters)
    print(f"Query: {query}")
    print(f"Collection: {collection_name}")
    print(f"Storage: {storage_path}")
    print()
    for position, result in enumerate(results, start=1):
        print(f"Result {position}")
        print(f"Score: {result.score:.4f} | vector={result.vector_score:.4f} | bm25={result.bm25_score:.4f}")
        print(f"Document: {result.chunk.document_title}")
        print(f"Page: {result.chunk.metadata.get('page_number', '?')}")
        print(f"Chunk: {result.chunk.id}")
        if result.reasons:
            print(f"Reasons: {', '.join(result.reasons)}")
        print(result.chunk.text)
        print()
    return 0
def cmd_ask_handbook(
    question: str,
    storage_path: Path,
    collection_name: str,
    limit: int,
    verbose: bool,
    json_mode: bool,
    conversation_id: str | None = None,
    metadata_filters: dict[str, object] | None = None,
) -> int:
    pipeline = RagPipeline.from_storage_path(storage_path, collection_name=collection_name, top_k=limit)
    try:
        result = pipeline.answer(question, limit=limit, conversation_id=conversation_id, metadata_filters=metadata_filters)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print("Run a new question first and copy the UUID shown after 'Conversation:'.", file=sys.stderr)
        return 2
    if json_mode:
        print(json.dumps(_response_payload(question, collection_name, storage_path, result), indent=2, ensure_ascii=False))
        return 0

    color = _supports_color()

    _print_label("Question:", question, color=color, label_code="1;36")
    _print_label("Provider:", result.answer_provider or "unknown", color=color)
    if result.answer_model_name:
        _print_label("Model:", result.answer_model_name, color=color)
    _print_label("Collection:", collection_name, color=color)
    _print_label("Storage:", str(storage_path), color=color)
    _print_label(
        "Faithfulness:",
        f"{'passed' if result.faithfulness.passed else 'failed'} | coverage={result.faithfulness.coverage:.2f}",
        color=color,
        label_code="1;32" if result.faithfulness.passed else "1;31",
    )
    _print_label("Faithfulness reason:", result.faithfulness.reason, color=color)
    _print_label(
        "Hallucination:",
        f"{'detected' if result.hallucination.detected else 'not detected'} | "
        f"severity={result.hallucination.severity:.2f} | coverage={result.hallucination.coverage:.2f}",
        color=color,
        label_code="1;31" if result.hallucination.detected else "1;32",
    )
    _print_label("Hallucination reason:", result.hallucination.reason, color=color)

    if verbose:
        print()
    _print_section("Answer", color=color)
    print(f"Conversation: {result.conversation_id}")
    if result.rewritten_query != result.question:
        _print_label("Rewritten query:", result.rewritten_query, color=color)
    print(result.answer)
    if verbose:
        print()
        _print_section("Prompt", color=color)
        print(result.prompt)
        print()
        _print_section("Sources", color=color)
        for position, source in enumerate(result.sources, start=1):
            print(f"[{position}] {source.document_title} | page {source.metadata.get('page_number', '?')} | {source.chunk_id}")
            print(source.excerpt)
            print()
    return 0


def cmd_index_webpage(
    url: str,
    storage_path: Path,
    collection_name: str,
    document_title: str | None,
    document_id: str | None,
    page_number: int,
    min_tokens: int,
    max_tokens: int,
    overlap_tokens: int,
) -> int:
    webpage = fetch_webpage_text(url)
    index = QdrantVectorIndex(storage_path=storage_path, collection_name=collection_name)
    title = document_title or webpage.title
    chunk_records = index.index_document(
        webpage.text,
        document_id=document_id or url,
        document_title=title,
        page_number=page_number,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )
    print(f"Indexed webpage: {url}")
    print(f"Title: {title}")
    print(f"Collection: {collection_name}")
    print(f"Storage: {storage_path}")
    print(f"Chunks stored: {len(chunk_records)}")
    return 0


def cmd_serve(
    host: str,
    port: int,
    storage_path: Path,
    collection_name: str,
    conversation_store_path: Path,
    api_token: str | None,
) -> int:
    server = RagAdminServer(
        storage_path=storage_path,
        collection_name=collection_name,
        conversation_store_path=conversation_store_path,
        api_token=api_token,
    )
    server.serve(host=host, port=port)
    return 0


def cmd_evaluate_rag(storage_path: Path, collection_name: str, cases_path: Path | None) -> int:
    pipeline = RagPipeline.from_storage_path(storage_path, collection_name=collection_name)
    cases = load_cases_from_jsonl(cases_path) if cases_path is not None else DEFAULT_EVALUATION_CASES
    results, summary = run_evaluation(pipeline, cases)
    print(format_evaluation_results(results, summary))
    return 0


def cmd_doctor(input_path: Path, storage_path: Path, collection_name: str, json_mode: bool) -> int:
    try:
        import production_rag as package  # noqa: F401

        package_ok = True
        package_detail = "production_rag imports successfully"
    except Exception as exc:  # pragma: no cover - environment dependent
        package_ok = False
        package_detail = f"import failed: {type(exc).__name__}: {exc}"

    checks = [
        {
            "name": "project root",
            "ok": (PROJECT_ROOT / "production_rag").exists(),
            "detail": str(PROJECT_ROOT),
        },
        {
            "name": "sample handbook",
            "ok": input_path.exists(),
            "detail": str(input_path),
        },
        {
            "name": "qdrant storage",
            "ok": storage_path.exists(),
            "detail": str(storage_path),
        },
        {
            "name": "package import",
            "ok": package_ok,
            "detail": package_detail,
        },
        {
            "name": "HF token",
            "ok": bool(os.getenv("HF_TOKEN")),
            "detail": "set" if os.getenv("HF_TOKEN") else "not set",
        },
        {
            "name": "OpenAI key",
            "ok": bool(os.getenv("OPENAI_API_KEY")),
            "detail": "set" if os.getenv("OPENAI_API_KEY") else "not set",
        },
    ]

    payload = {
        "python": sys.executable,
        "version": sys.version.split()[0],
        "collection_name": collection_name,
        "storage_path": str(storage_path),
        "input_path": str(input_path),
        "checks": checks,
    }

    if json_mode:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    color = _supports_color()
    _print_section("Doctor", color=color)
    _print_label("Python:", sys.executable, color=color)
    _print_label("Version:", sys.version.split()[0], color=color)
    _print_label("Collection:", collection_name, color=color)
    _print_label("Storage:", str(storage_path), color=color)
    _print_label("Source:", str(input_path), color=color)
    print()
    for check in checks:
        ok = bool(check["ok"])
        status = "ok" if ok else "warn"
        status_code = "1;32" if ok else "1;33"
        print(f"{_ansi(status.upper(), status_code, color):<6} {check['name']}: {check['detail']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "extract-pdf":
        return cmd_extract_pdf(args.pdf_path, args.max_length)
    if args.command == "chunk-text":
        return cmd_chunk_text(args.input_path, args.min_tokens, args.max_tokens, args.overlap_tokens, args.show)
    if args.command == "embed-text":
        return cmd_embed_text(args.input_path, args.min_tokens, args.max_tokens, args.overlap_tokens, args.show)
    if args.command == "index-handbook":
        return cmd_index_handbook(
            args.input_path,
            args.storage_path,
            args.collection_name,
            args.min_tokens,
            args.max_tokens,
            args.overlap_tokens,
        )
    if args.command == "query-qdrant":
        metadata_filters = _parse_metadata_filters(getattr(args, "metadata_filter", None))
        return cmd_query_qdrant(args.query, args.storage_path, args.collection_name, args.limit, metadata_filters)
    if args.command == "query-hybrid":
        metadata_filters = _parse_metadata_filters(getattr(args, "metadata_filter", None))
        return cmd_query_hybrid(
            args.query,
            args.storage_path,
            args.collection_name,
            args.limit,
            args.candidate_k,
            metadata_filters,
        )
    if args.command == "ask-handbook":
        metadata_filters = _parse_metadata_filters(getattr(args, "metadata_filter", None))
        return cmd_ask_handbook(
            args.question,
            args.storage_path,
            args.collection_name,
            args.limit,
            args.verbose,
            args.json,
            conversation_id=args.conversation_id,
            metadata_filters=metadata_filters,
        )
    if args.command == "evaluate-rag":
        if args.json:
            pipeline = RagPipeline.from_storage_path(args.storage_path, collection_name=args.collection_name)
            cases = load_cases_from_jsonl(args.cases_path) if args.cases_path is not None else DEFAULT_EVALUATION_CASES
            results, summary = run_evaluation(pipeline, cases)
            print(json.dumps(_summary_payload(args.collection_name, args.storage_path, results, summary), indent=2, ensure_ascii=False))
            return 0
        return cmd_evaluate_rag(args.storage_path, args.collection_name, args.cases_path)
    if args.command == "doctor":
        return cmd_doctor(args.input_path, args.storage_path, args.collection_name, args.json)
    if args.command == "index-webpage":
        return cmd_index_webpage(
            args.url,
            args.storage_path,
            args.collection_name,
            args.document_title,
            args.document_id,
            args.page_number,
            args.min_tokens,
            args.max_tokens,
            args.overlap_tokens,
        )
    if args.command == "serve":
        return cmd_serve(args.host, args.port, args.storage_path, args.collection_name, args.conversation_store_path, args.api_token)

    parser.error(f"Unsupported command: {args.command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
