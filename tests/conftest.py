"""Shared fixtures for the scheduler tests."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest


class FakeLlamaServer(BaseHTTPRequestHandler):
    """Minimal llama-server stand-in: /v1/models and /v1/chat/completions (completion_tokens = 7 + seed)."""

    def log_message(self, *a):
        pass

    def do_GET(self):
        self._send({"data": [{"id": "fake"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        n = 7 + body["seed"]
        self._send({"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": n}})

    def _send(self, body):
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def fake_llama_url():
    server = HTTPServer(("127.0.0.1", 0), FakeLlamaServer)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
