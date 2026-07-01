"""Minimal MCP-style HTTP server on port 9000.

Implements the bare minimum JSON-RPC surface so the SDK's MCPServer
client can hit a live target on http://localhost:9000/mcp. No deps.
"""

import json
from http.server import HTTPServer, BaseHTTPRequestHandler

TOOLS = [
    {
        "name": "echo",
        "description": "Returns the input string unchanged.",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "lookup_order",
        "description": "Returns a fake order for a given order id.",
        "inputSchema": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    },
]


def handle(body):
    method = body.get("method")
    req_id = body.get("id")
    params = body.get("params", {})

    if method in ("initialize",):
        return {
            "jsonrpc": "2.0", "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "tensoreval-stub-mcp", "version": "0.1.0"},
                "capabilities": {"tools": {}},
            },
        }
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS}}
    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments", {})
        if name == "echo":
            content = args.get("text", "")
        elif name == "lookup_order":
            oid = args.get("order_id", "?")
            content = json.dumps({
                "order_id": oid, "total_cents": 4999,
                "delivered_at": "2024-01-15", "policy": "30d-full",
            })
        else:
            return {"jsonrpc": "2.0", "id": req_id,
                    "error": {"code": -32601, "message": f"unknown tool: {name}"}}
        return {"jsonrpc": "2.0", "id": req_id,
                "result": {"content": [{"type": "text", "text": content}]}}

    return {"jsonrpc": "2.0", "id": req_id,
            "error": {"code": -32601, "message": f"unknown method: {method}"}}


class MCPHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/healthz"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(body or b"{}")
            resp = handle(data)
        except Exception as e:
            resp = {"jsonrpc": "2.0", "id": None,
                    "error": {"code": -32700, "message": str(e)}}
        out = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a, **k): pass


if __name__ == "__main__":
    print("Stub MCP server running on port 9000")
    HTTPServer(("0.0.0.0", 9000), MCPHandler).serve_forever()
