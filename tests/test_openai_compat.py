"""The free-provider path: an OpenAI-compatible endpoint (Gemini, Groq, Ollama...) that
rejects JSON mode and wraps its answer in a code fence. Uses a tiny local fake server."""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.config import Settings
from agent.llm import LLM, extract_json
from agent.schemas import GtmRationale

CALLS = []


class Fake(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        CALLS.append(body)
        if "response_format" in body:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": {"message": "response_format is not supported"}}).encode())
            return
        content = 'Sure!\n```json\n{"rationale": "r \\"q\\" {x}", "ae_role": "a", "partner_role": "p", "risks": []}\n```'
        out = {"id": "1", "object": "chat.completion", "created": 0, "model": body["model"],
               "choices": [{"index": 0, "finish_reason": "stop",
                            "message": {"role": "assistant", "content": content}}],
               "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(out).encode())


class Log:
    def llm_call(self, r): pass
    def warn(self, m): pass


def test_openai_compatible_fallback():
    srv = HTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    s = Settings()
    s.llm_provider, s.openai_model = "openai", "free-model"
    s.openai_base_url = f"http://127.0.0.1:{srv.server_port}/v1"
    llm = LLM(s, Log())
    out = llm.structured(system="sys", user="u", schema=GtmRationale)
    assert out.rationale == 'r "q" {x}' and out.ae_role == "a"
    assert "response_format" in CALLS[0] and "response_format" not in CALLS[-1]
    srv.shutdown()


def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('Here you go:\n{"a": {"b": "}"}} trailing') == {"a": {"b": "}"}}
    assert extract_json("no json") == {}


def test_truncated_reply_keeps_outer_object():
    cut = '{"framework_rationale": "x", "elements": [{"element": "Metrics", "status": "unknown"}, {"element": "Econ'
    v = extract_json(cut)
    assert "framework_rationale" in v and isinstance(v["elements"], list)   # never the inner fragment


def test_two_objects_takes_first_outer():
    assert extract_json('{"a": 1}\n{"b": 2}') in ({"a": 1}, {"a": 1, "b": 2})
