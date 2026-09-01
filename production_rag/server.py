from __future__ import annotations

import argparse
import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .pipeline import RagPipeline
from .serialization import to_jsonable
from .web_ingest import fetch_webpage_text


def _parse_metadata_filters(raw_values: list[str] | None) -> dict[str, object]:
    filters: dict[str, object] = {}
    for raw in raw_values or []:
        raw = raw.strip()
        if not raw:
            continue
        if raw.startswith("{"):
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError("metadata filters must be a JSON object")
            filters.update(payload)
            continue
        if "=" not in raw:
            raise ValueError("metadata filters must use key=value or JSON object syntax")
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


def build_dashboard_html() -> str:
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Enterprise Knowledge Intelligence</title>
  <style>
    :root {
      --bg: #0b1020;
      --panel: #121a33;
      --panel-2: #17213f;
      --text: #eef3ff;
      --muted: #9fb0d0;
      --accent: #7dd3fc;
      --border: rgba(125, 211, 252, 0.16);
      --good: #86efac;
      --bad: #fca5a5;
    }
    body {
      margin: 0;
      font-family: Segoe UI, Inter, Arial, sans-serif;
      background:
        radial-gradient(circle at top left, rgba(125, 211, 252, 0.18), transparent 28%),
        radial-gradient(circle at top right, rgba(134, 239, 172, 0.12), transparent 22%),
        linear-gradient(180deg, #091022 0%, #0b1020 100%);
      color: var(--text);
    }
    .shell {
      max-width: 1180px;
      margin: 0 auto;
      padding: 32px 20px 56px;
    }
    .hero {
      display: grid;
      gap: 12px;
      grid-template-columns: 1.3fr 0.7fr;
      align-items: end;
      margin-bottom: 22px;
    }
    .card {
      background: rgba(18, 26, 51, 0.92);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 18px;
      box-shadow: 0 20px 60px rgba(0, 0, 0, 0.35);
      backdrop-filter: blur(14px);
    }
    h1 { margin: 0; font-size: clamp(2rem, 4vw, 3.5rem); line-height: 1.05; }
    .sub { color: var(--muted); margin: 8px 0 0; max-width: 66ch; }
    .grid {
      display: grid;
      grid-template-columns: 1fr 0.9fr;
      gap: 18px;
    }
    label { display: block; font-size: 0.9rem; color: var(--muted); margin-bottom: 6px; }
    input, textarea, button {
      width: 100%;
      box-sizing: border-box;
      border-radius: 12px;
      border: 1px solid rgba(159, 176, 208, 0.24);
      background: rgba(7, 12, 25, 0.78);
      color: var(--text);
      padding: 12px 14px;
      font: inherit;
    }
    textarea { min-height: 110px; resize: vertical; }
    button {
      cursor: pointer;
      border: 1px solid rgba(125, 211, 252, 0.35);
      background: linear-gradient(135deg, #1f4e79, #0f766e);
      font-weight: 700;
    }
    button:hover { filter: brightness(1.08); }
    .row { display: grid; gap: 12px; grid-template-columns: repeat(2, 1fr); }
    .meta { display: grid; gap: 12px; margin-top: 12px; }
    .small { font-size: 0.88rem; color: var(--muted); }
    pre {
      white-space: pre-wrap;
      word-break: break-word;
      background: rgba(7, 12, 25, 0.8);
      border: 1px solid rgba(159, 176, 208, 0.18);
      border-radius: 14px;
      padding: 14px;
      margin: 0;
      min-height: 180px;
    }
    .pill {
      display: inline-block;
      padding: 4px 10px;
      border-radius: 999px;
      background: rgba(125, 211, 252, 0.12);
      border: 1px solid rgba(125, 211, 252, 0.2);
      color: var(--accent);
      font-size: 0.82rem;
      margin-right: 8px;
    }
    @media (max-width: 920px) {
      .hero, .grid, .row { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <div class="hero">
      <div class="card">
        <div class="pill">Enterprise Knowledge Intelligence</div>
        <h1>Enterprise Knowledge Intelligence</h1>
        <p class="sub">Ask questions, inspect citations, review conversation history, and index web pages or handbooks from one admin surface.</p>
      </div>
      <div class="card">
        <div class="small">Auth</div>
        <label for="token">Bearer token</label>
        <input id="token" placeholder="Paste API token if configured" />
        <div class="small" style="margin-top: 8px;">Saved locally in your browser for subsequent requests.</div>
      </div>
    </div>

    <div class="grid">
      <div class="card">
        <label for="question">Question</label>
        <textarea id="question" placeholder="What is the annual leave entitlement?"></textarea>
        <div class="row" style="margin-top: 12px;">
          <div>
            <label for="conversation_id">Conversation ID</label>
            <input id="conversation_id" placeholder="Optional: continue an existing thread" />
          </div>
          <div>
            <label for="metadata_filters">Metadata filters</label>
            <input id="metadata_filters" placeholder='Optional JSON like {"page_number":1}' />
          </div>
        </div>
        <div class="row" style="margin-top: 12px;">
          <div>
            <label for="limit">Top K</label>
            <input id="limit" type="number" value="5" min="1" max="20" />
          </div>
          <div style="display:flex; align-items:end;">
            <button id="ask">Ask</button>
          </div>
        </div>
        <div class="meta">
          <div>
            <label>Answer</label>
            <pre id="answer"></pre>
          </div>
          <div>
            <label>Raw response</label>
            <pre id="response"></pre>
          </div>
        </div>
      </div>
      <div class="card">
        <label for="web_url">Index webpage</label>
        <input id="web_url" placeholder="https://example.com/policy" />
        <div class="row" style="margin-top: 12px;">
          <div>
            <label for="web_document_title">Document title</label>
            <input id="web_document_title" placeholder="Optional title override" />
          </div>
          <div style="display:flex; align-items:end;">
            <button id="index_web">Index webpage</button>
          </div>
        </div>
        <div class="meta">
          <div>
            <label>Web ingestion result</label>
            <pre id="index_result"></pre>
          </div>
          <div>
            <label>Conversations</label>
            <pre id="conversations"></pre>
          </div>
        </div>
      </div>
    </div>
  </div>
  <script>
    const tokenInput = document.getElementById("token");
    const savedToken = localStorage.getItem("production_rag_token") || "";
    tokenInput.value = savedToken;
    tokenInput.addEventListener("change", () => localStorage.setItem("production_rag_token", tokenInput.value.trim()));

    function headers() {
      const token = tokenInput.value.trim();
      const value = token || localStorage.getItem("production_rag_token") || "";
      const result = {"Content-Type": "application/json"};
      if (value) result["Authorization"] = "Bearer " + value;
      return result;
    }

    async function refreshConversations() {
      const response = await fetch("/api/conversations", {headers: headers()});
      document.getElementById("conversations").textContent = await response.text();
    }

    document.getElementById("ask").addEventListener("click", async () => {
      const payload = {
        question: document.getElementById("question").value,
        conversation_id: document.getElementById("conversation_id").value || null,
        limit: Number(document.getElementById("limit").value || 5),
      };
      const filters = document.getElementById("metadata_filters").value.trim();
      if (filters) {
        try {
          payload.metadata_filters = JSON.parse(filters);
        } catch (error) {
          document.getElementById("response").textContent = "Invalid metadata filter JSON: " + error;
          return;
        }
      }
      const response = await fetch("/api/ask", {method: "POST", headers: headers(), body: JSON.stringify(payload)});
      const text = await response.text();
      document.getElementById("response").textContent = text;
      try {
        const data = JSON.parse(text);
        document.getElementById("answer").textContent = data.response ? data.response.answer : data.answer;
        if (data.response && data.response.conversation_id) {
          document.getElementById("conversation_id").value = data.response.conversation_id;
        }
      } catch (error) {
        document.getElementById("answer").textContent = "";
      }
      refreshConversations();
    });

    document.getElementById("index_web").addEventListener("click", async () => {
      const payload = {
        url: document.getElementById("web_url").value,
        document_title: document.getElementById("web_document_title").value || null,
      };
      const response = await fetch("/api/index/webpage", {method: "POST", headers: headers(), body: JSON.stringify(payload)});
      document.getElementById("index_result").textContent = await response.text();
      refreshConversations();
    });

    refreshConversations();
  </script>
</body>
</html>
""".strip()


def _json_response(handler: BaseHTTPRequestHandler, status: HTTPStatus, payload: Any) -> None:
    body = json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _text_response(handler: BaseHTTPRequestHandler, status: HTTPStatus, text: str, content_type: str = "text/plain; charset=utf-8") -> None:
    body = text.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class RagAdminServer:
    def __init__(
        self,
        *,
        storage_path: Path,
        collection_name: str,
        conversation_store_path: Path | str = Path("data/production_rag.sqlite3"),
        api_token: str | None = None,
    ) -> None:
        self.pipeline = RagPipeline.from_storage_path(
            storage_path,
            collection_name=collection_name,
            conversation_store_path=str(conversation_store_path),
        )
        self.api_token = api_token or os.getenv("PRODUCTION_RAG_API_TOKEN") or ""

    def _authorized(self, handler: BaseHTTPRequestHandler) -> bool:
        if not self.api_token:
            return True
        header = handler.headers.get("Authorization", "")
        return header == f"Bearer {self.api_token}"

    def _require_auth(self, handler: BaseHTTPRequestHandler) -> bool:
        if self._authorized(handler):
            return True
        _json_response(handler, HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
        return False

    def handler_class(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args) -> None:  # noqa: A003
                return

            def _read_json(self) -> dict[str, Any]:
                content_length = int(self.headers.get("Content-Length", "0") or 0)
                raw = self.rfile.read(content_length) if content_length > 0 else b"{}"
                if not raw:
                    return {}
                payload = json.loads(raw.decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("JSON body must be an object")
                return payload

            def do_GET(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                if parsed.path in {"/", "/dashboard"}:
                    _text_response(self, HTTPStatus.OK, build_dashboard_html(), content_type="text/html; charset=utf-8")
                    return
                if parsed.path == "/api/health":
                    _json_response(
                        self,
                        HTTPStatus.OK,
                        {
                            "status": "ok",
                            "collection_name": server.pipeline.index.collection_name,
                            "storage_path": str(server.pipeline.index.storage_path),
                        },
                    )
                    return
                if not server._require_auth(self):
                    return
                if parsed.path == "/api/conversations":
                    query = parse_qs(parsed.query)
                    limit = int(query.get("limit", ["20"])[0])
                    conversations = []
                    store = server.pipeline.conversation_store
                    if store is not None:
                        conversations = store.list_conversations(limit=limit)
                    _json_response(self, HTTPStatus.OK, {"conversations": conversations})
                    return
                if parsed.path.startswith("/api/conversations/"):
                    conversation_id = parsed.path.rsplit("/", 1)[-1]
                    store = server.pipeline.conversation_store
                    if store is None:
                        _json_response(self, HTTPStatus.NOT_FOUND, {"error": "conversation store unavailable"})
                        return
                    turns = store.list_recent_turns(conversation_id, limit=100)
                    _json_response(
                        self,
                        HTTPStatus.OK,
                        {
                            "conversation_id": conversation_id,
                            "summary": store.summarize_history(conversation_id, limit=100),
                            "turns": [to_jsonable(turn) for turn in turns],
                        },
                    )
                    return
                _json_response(self, HTTPStatus.NOT_FOUND, {"error": "not found"})

            def do_POST(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                if not server._require_auth(self):
                    return

                try:
                    payload = self._read_json()
                except Exception as exc:
                    _json_response(self, HTTPStatus.BAD_REQUEST, {"error": f"invalid JSON: {exc}"})
                    return

                if parsed.path == "/api/ask":
                    question = str(payload.get("question", "")).strip()
                    if not question:
                        _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "question is required"})
                        return
                    limit = int(payload.get("limit", 5))
                    conversation_id = payload.get("conversation_id")
                    metadata_filters = payload.get("metadata_filters")
                    if metadata_filters is not None and not isinstance(metadata_filters, dict):
                        _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "metadata_filters must be an object"})
                        return
                    try:
                        response = server.pipeline.answer(
                            question,
                            limit=limit,
                            conversation_id=str(conversation_id) if conversation_id else None,
                            metadata_filters=metadata_filters,
                        )
                    except ValueError as exc:
                        _json_response(self, HTTPStatus.NOT_FOUND, {"error": str(exc)})
                        return
                    _json_response(
                        self,
                        HTTPStatus.OK,
                        {
                            "question": question,
                            "response": response,
                        },
                    )
                    return

                if parsed.path == "/api/index/webpage":
                    url = str(payload.get("url", "")).strip()
                    if not url:
                        _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "url is required"})
                        return
                    document_title = payload.get("document_title")
                    min_tokens = int(payload.get("min_tokens", 500))
                    max_tokens = int(payload.get("max_tokens", 800))
                    overlap_tokens = int(payload.get("overlap_tokens", 75))
                    page_number = int(payload.get("page_number", 1))
                    webpage = fetch_webpage_text(url)
                    title = str(document_title).strip() if document_title else webpage.title
                    chunks = server.pipeline.index.index_document(
                        webpage.text,
                        document_id=payload.get("document_id") or url,
                        document_title=title,
                        page_number=page_number,
                        min_tokens=min_tokens,
                        max_tokens=max_tokens,
                        overlap_tokens=overlap_tokens,
                    )
                    _json_response(
                        self,
                        HTTPStatus.OK,
                        {
                            "url": url,
                            "title": title,
                            "chunk_count": len(chunks),
                            "chunks": [to_jsonable(chunk) for chunk in chunks],
                        },
                    )
                    return

                _json_response(self, HTTPStatus.NOT_FOUND, {"error": "not found"})

        return Handler

    def serve(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        httpd = ThreadingHTTPServer((host, port), self.handler_class())
        print(f"Serving Enterprise Knowledge Intelligence on http://{host}:{port}")
        if self.api_token:
            print("Authentication: enabled via PRODUCTION_RAG_API_TOKEN")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            httpd.server_close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="production-rag-serve")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--storage-path", type=Path, default=Path("data/qdrant"))
    parser.add_argument("--collection-name", default="production_rag_chunks")
    parser.add_argument("--conversation-store-path", type=Path, default=Path("data/production_rag.sqlite3"))
    parser.add_argument("--api-token", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    server = RagAdminServer(
        storage_path=args.storage_path,
        collection_name=args.collection_name,
        conversation_store_path=args.conversation_store_path,
        api_token=args.api_token,
    )
    server.serve(host=args.host, port=args.port)
    return 0
