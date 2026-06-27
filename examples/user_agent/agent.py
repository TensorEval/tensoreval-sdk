import json
from http.server import HTTPServer, BaseHTTPRequestHandler

class SupportAgent(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        data = json.loads(body)
        messages = data.get("messages", [])
        query = messages[-1].get("content", "") if messages else ""
        
        # Simple support agent logic
        q = query.lower()
        if "refund" in q:
            response = "I'll help you with your refund. Let me check your order details."
        elif "cancel" in q:
            response = "I understand you want to cancel. Let me process that for you."
        elif "upgrade" in q:
            response = "Great choice! Let me help you upgrade your plan."
        else:
            response = "I'm here to help. Could you tell me more about your issue?"
        
        result = {"choices": [{"message": {"content": response}}]}
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(result).encode())
    
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')
    
    def log_message(self, *a): pass

if __name__ == "__main__":
    print("Support agent running on port 8000")
    HTTPServer(("0.0.0.0", 8000), SupportAgent).serve_forever()
