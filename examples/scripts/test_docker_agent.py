"""Test: Docker compose with simple HTTP agent."""

import sys
import asyncio
import tempfile
import os
sys.path.insert(0, ".")
import tensoreval as te
import httpx


async def wait_for_agent(url: str, timeout: int = 30):
    """Wait for agent to be ready."""
    start = asyncio.get_event_loop().time()
    while asyncio.get_event_loop().time() - start < timeout:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, timeout=2.0)
                if resp.status_code == 200:
                    return True
        except:
            pass
        await asyncio.sleep(1)
    return False


async def test():
    print("=" * 60)
    print("Docker Agent Test")
    print("=" * 60)

    # Create agent code
    agent_code = '''
import json
from http.server import HTTPServer, BaseHTTPRequestHandler

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        data = json.loads(body)
        messages = data.get("messages", [])
        query = messages[-1].get("content", "") if messages else ""
        q = query.lower()

        if "2+2" in q or "2 + 2" in q:
            answer = "4"
        elif "12" in q and "15" in q:
            answer = "180"
        elif "10" in q and "5" in q:
            answer = "50"
        elif "100" in q and "4" in q:
            answer = "25"
        else:
            answer = "I can help with math. What calculation?"

        result = {"choices": [{"message": {"content": answer}}]}
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(result).encode())

    def log_message(self, format, *args):
        pass  # Suppress logs

if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", 8000), Handler)
    server.serve_forever()
'''

    # Write to temp dir
    tmpdir = tempfile.mkdtemp(prefix="tensoreval-")
    with open(os.path.join(tmpdir, "agent.py"), "w") as f:
        f.write(agent_code)

    # Create DockerCompose
    compose = te.DockerCompose(services={
        "agent": {
            "image": "python:3.12-slim",
            "command": "python /app/agent.py",
            "port": 8001,
            "volumes": [f"{tmpdir}:/app"],
        },
    })

    # Start containers
    print("[1] Starting Docker container...")
    try:
        ports = await compose.up()
        print(f"    Started on port {ports['agent']}")
    except Exception as e:
        print(f"    Failed: {e}")
        return

    # Wait for agent to be ready
    print("[2] Waiting for agent to be ready...")
    ready = await wait_for_agent("http://localhost:8001/", timeout=30)
    if not ready:
        print("    Agent not ready after 30s")
        await compose.down()
        return
    print("    Agent ready!")

    # Test agent directly
    print("[3] Testing agent directly...")
    async with httpx.AsyncClient() as client:
        for q, expected in [("What is 2+2?", "4"), ("What is 12*15?", "180")]:
            resp = await client.post(
                "http://localhost:8001/v1/chat/completions",
                json={"messages": [{"role": "user", "content": q}]},
                timeout=10.0,
            )
            data = resp.json()
            answer = data["choices"][0]["message"]["content"]
            status = "OK" if expected in answer else "FAIL"
            print(f"    {q} -> {answer} [{status}]")

    # Run TensorEval evaluation
    print("[4] Running TensorEval evaluation...")
    ds = te.Datasets.load_from_dict([
        {"query": "What is 2+2?", "reference_answer": "4"},
        {"query": "What is 12*15?", "reference_answer": "180"},
        {"query": "What is 10*5?", "reference_answer": "50"},
        {"query": "What is 100/4?", "reference_answer": "25"},
    ])
    grader = te.RubricGrader()

    results = await te.Evaluation.run_async(
        datasets=ds,
        grader=grader,
        agent_port=8001,
        workers=4,
    )

    summary = results.summary()
    print()
    print("Results:")
    print(f"  Samples: {summary['num_runs']}")
    print(f"  Avg Reward: {summary['avg_reward']}")
    print(f"  Pass Rate: {summary['pass_rate']}")
    print()
    for i, run in enumerate(results.runs):
        sample = ds[i]
        resp = run.get("response", "").encode("ascii", "replace").decode()[:30]
        reward = run.get("reward", 0)
        status = "PASS" if reward >= 0.8 else "FAIL"
        print(f"  Q{i+1}: {sample.input} -> {resp} [{status}]")

    # Cleanup
    print()
    print("[5] Cleaning up...")
    await compose.down()
    print("    Done!")

    print()
    print("=" * 60)
    print("DOCKER AGENT TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test())
