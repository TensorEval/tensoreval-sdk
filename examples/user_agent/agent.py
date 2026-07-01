"""Minimal OpenAI-compatible agent on port 8000.

No external deps. Returns canned answers so the SDK can call
POST /v1/chat/completions and get deterministic responses for the
5 test queries in test_cases.jsonl.
"""

import json
from http.server import HTTPServer, BaseHTTPRequestHandler

ANSWERS = {
    "refund":  "I'll help you with your refund. Let me check your order details.",
    "cancel":  "I understand you want to cancel. Let me process that for you.",
    "upgrade": "Great choice! Let me help you upgrade your plan.",
    "crash":   "Sorry to hear that. Could you share your device model and iOS version?",
    "help":    "I'm here to help. Could you tell me more about your issue?",
}


def answer_for(query: str) -> str:
    q = query.lower()
    if "refund" in q:  return ANSWERS["refund"]
    if "cancel" in q:  return ANSWERS["cancel"]
    if "upgrade" in q: return ANSWERS["upgrade"]
    if "crash" in q or "iphone" in q: return ANSWERS["crash"]
    if "help" in q or "account" in q: return ANSWERS["help"]
    return "I'm here to help. Could you tell me more about your issue?"


class Agent(BaseHTTPRequestHandler):
    def do_GET(self):
        body = b'{"status":"ok"}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(body or b"{}")
        except Exception:
            data = {}
        messages = data.get("messages", [])
        query = messages[-1].get("content", "") if messages else ""
        text = answer_for(query)
        resp = {"choices": [{"message": {"role": "assistant", "content": text}}]}
        out = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a, **k): pass


if __name__ == "__main__":
    print("Support agent running on port 8000")
    HTTPServer(("0.0.0.0", 8000), Agent).serve_forever()
