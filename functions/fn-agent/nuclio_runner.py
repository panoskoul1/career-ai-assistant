#!/usr/bin/env python3
"""Minimal Nuclio-compatible HTTP runner for Docker Compose.

Simulates the Nuclio serverless runtime contract:
  - init_context(context): called ONCE at startup with a Context object
  - handler(context, event): called per HTTP request

No FastAPI. No uvicorn. Pure Python stdlib http.server.

To port to real Nuclio: keep function.py unchanged, replace this runner
with a Nuclio function YAML that references function.py as the handler.
"""

from __future__ import annotations

import json
import logging
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("nuclio_runner")


# ---------------------------------------------------------------------------
# Nuclio runtime primitives
# ---------------------------------------------------------------------------

class _Log:
    def info(self, msg: str) -> None:    logger.info(msg)
    def warning(self, msg: str) -> None: logger.warning(msg)
    def error(self, msg: str) -> None:   logger.error(msg)
    def debug(self, msg: str) -> None:   logger.debug(msg)


class Context:
    """Mimics nuclio.Context for Docker Compose deployment."""

    def __init__(self) -> None:
        self.logger = _Log()
        self.user_data = type("UserData", (), {})()


class Event:
    """Mimics nuclio.Event."""

    def __init__(
        self,
        body: bytes,
        headers: dict,
        path: str = "/",
        method: str = "POST",
    ) -> None:
        self.body = body
        self.headers = headers
        self.path = path
        self.method = method

    def get_json(self) -> Any:
        return json.loads(self.body or b"{}")


# ---------------------------------------------------------------------------
# HTTP server — delegates every request to function.handler()
# ---------------------------------------------------------------------------

_ctx: Context  # set in __main__ after init_context() runs


class _Handler(BaseHTTPRequestHandler):

    def do_GET(self) -> None:
        if self.path == "/health":
            self._respond_raw(200, '{"status":"ok"}')
        else:
            self._respond_raw(404, '{"error":"not found"}')

    def do_POST(self) -> None:
        self._delegate("POST")

    def do_DELETE(self) -> None:
        self._delegate("DELETE")

    def _delegate(self, method: str) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        event = Event(body=body, headers=dict(self.headers), path=self.path, method=method)
        try:
            from function import handler
            response = handler(_ctx, event)
        except Exception as exc:
            logger.exception("Handler error: %s", exc)
            response = _ErrorResponse(str(exc))
        self._send(response)

    def _send(self, response: Any) -> None:
        """Send any duck-typed response object with .body/.status_code/.content_type."""
        self._respond_raw(
            getattr(response, "status_code", 200),
            getattr(response, "body", ""),
            getattr(response, "content_type", "application/json"),
        )

    def _respond_raw(self, code: int, body: str, ct: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ct)
        self.end_headers()
        self.wfile.write(body.encode() if isinstance(body, str) else body)

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.info(fmt, *args)


class _ErrorResponse:
    def __init__(self, msg: str) -> None:
        self.body = json.dumps({"error": msg})
        self.status_code = 500
        self.content_type = "application/json"


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))

    # Initialise context — mimics Nuclio calling init_context() once
    _ctx = Context()
    from function import init_context
    logger.info("Calling init_context() ...")
    init_context(_ctx)
    logger.info("Function ready on port %d", port)

    server = HTTPServer(("0.0.0.0", port), _Handler)
    server.serve_forever()
