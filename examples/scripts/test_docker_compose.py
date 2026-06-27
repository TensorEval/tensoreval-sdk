"""Test: Docker compose with proper Windows patterns.

Tests:
1. DockerCompose generates correct compose.yaml
2. Containers start and expose ports
3. Commands execute inside containers
4. Files can be read/written
5. Cleanup works
"""

import sys
import asyncio
import tempfile
import os
sys.path.insert(0, ".")
import tensoreval as te


async def test():
    print("=" * 60)
    print("Docker Compose Test (Windows-compatible)")
    print("=" * 60)

    # 1. Create agent script
    agent_code = '''
import json
from http.server import HTTPServer, BaseHTTPRequestHandler

class H(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        data = json.loads(body)
        msgs = data.get("messages", [])
        query = msgs[-1].get("content", "") if msgs else ""
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
            answer = "I can help with math."
        
        result = {"choices": [{"message": {"content": answer}}]}
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(result).encode())

    def log_message(self, *a): pass

HTTPServer(("0.0.0.0", 8000), H).serve_forever()
'''

    # Write to temp dir
    tmpdir = tempfile.mkdtemp(prefix="tensoreval-")
    with open(os.path.join(tmpdir, "agent.py"), "w") as f:
        f.write(agent_code)

    # 2. Create DockerCompose
    compose = te.DockerCompose(
        services={
            "agent": {
                "image": "python:3.12-slim",
                "command": "python /app/agent.py",
                "port": 8002,
                "container_port": 8000,  # port inside container
                "volumes": [f"{tmpdir}:/app"],
            },
        }
    )

    # 3. Show generated compose.yaml
    yaml_content = compose._generate_compose_yaml()
    print()
    print("[1] Generated compose.yaml:")
    for line in yaml_content.strip().split("\n"):
        print(f"    {line}")

    # 4. Start containers
    print()
    print("[2] Starting Docker containers...")
    try:
        ports = await compose.up()
        print(f"    Started! Ports: {ports}")
    except Exception as e:
        print(f"    Failed: {e}")
        return

    # 5. Wait for agent to be ready
    print()
    print("[3] Waiting for agent...")
    import httpx
    for i in range(30):
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get("http://localhost:8002/", timeout=2.0)
                if resp.status_code == 200:
                    print("    Agent ready!")
                    break
        except:
            pass
        await asyncio.sleep(1)
    else:
        print("    Agent not ready after 30s")
        await compose.down()
        return

    # 6. Test agent directly
    print()
    print("[4] Testing agent directly:")
    async with httpx.AsyncClient() as client:
        tests = [
            ("What is 2+2?", "4"),
            ("What is 12*15?", "180"),
            ("What is 10*5?", "50"),
        ]
        for q, expected in tests:
            resp = await client.post(
                "http://localhost:8002/v1/chat/completions",
                json={"messages": [{"role": "user", "content": q}]},
                timeout=10.0,
            )
            data = resp.json()
            answer = data["choices"][0]["message"]["content"]
            status = "OK" if expected in answer else "FAIL"
            print(f"    {q} -> {answer} [{status}]")

    # 7. Run TensorEval evaluation
    print()
    print("[5] Running TensorEval evaluation:")
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
        agent_port=8002,
        workers=4,
    )

    summary = results.summary()
    print(f"    Samples: {summary['num_runs']}")
    print(f"    Avg Reward: {summary['avg_reward']}")
    print(f"    Pass Rate: {summary['pass_rate']}")
    print()
    for i, run in enumerate(results.runs):
        sample = ds[i]
        resp = run.get("response", "").encode("ascii", "replace").decode()[:30]
        reward = run.get("reward", 0)
        status = "PASS" if reward >= 0.8 else "FAIL"
        print(f"    Q{i+1}: {sample.input} -> {resp} [{status}]")

    # 8. Cleanup
    print()
    print("[6] Cleaning up...")
    await compose.down()
    print("    Done!")

    print()
    print("=" * 60)
    print("DOCKER COMPOSE TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test())
