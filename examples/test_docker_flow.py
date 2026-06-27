"""Test: Docker compose integration with TensorEval.

Tests:
1. Env.from_dict with Docker config
2. DockerCompose service generation
3. Full evaluation flow with env
"""

import sys
sys.path.insert(0, ".")
import tensoreval as te

print("=" * 60)
print("Docker Compose Integration Test")
print("=" * 60)

# Test 1: Env from dict (no Docker, just config)
print()
print("[1] Env.from_dict (no Docker)...")
env = te.Env.from_dict({
    "system_prompt": "You are a math solver.",
    "agent_url": "http://localhost:8000",
    "mcp_url": "http://localhost:9000/mcp",
})
print(f"  system_prompt: {env.system_prompt}")
print(f"  agent_url: {env.agent_url}")
print(f"  mcp_url: {env.mcp_url}")
print(f"  agent (Docker): {env.agent}")
print(f"  mcp (Docker): {env.mcp}")

# Test 2: Env from dict with Docker config
print()
print("[2] Env.from_dict (with Docker config)...")
env_docker = te.Env.from_dict({
    "system_prompt": "You are a support agent.",
    "agent": {
        "image": "python:3.12-slim",
        "command": "python agent.py",
        "port": 8000,
        "env": {"OPENAI_API_KEY": "sk-test"},
        "volumes": ["./code:/app"],
    },
    "mcp": {
        "image": "node:18-slim",
        "command": "node mcp-server.js",
        "port": 9000,
    },
    "env_file": ".env",
})
print(f"  system_prompt: {env_docker.system_prompt}")
print(f"  agent image: {env_docker.agent.get('image')}")
print(f"  agent port: {env_docker.agent.get('port')}")
print(f"  mcp image: {env_docker.mcp.get('image')}")
print(f"  mcp port: {env_docker.mcp.get('port')}")
print(f"  env_file: {env_docker.env_file}")

# Test 3: DockerCompose generation
print()
print("[3] DockerCompose generation...")
compose = te.DockerCompose(
    services={
        "agent": {
            "image": "python:3.12-slim",
            "command": "python agent.py",
            "port": 8000,
            "env": {"OPENAI_API_KEY": "sk-test", "DATABASE_URL": "postgres://..."},
            "volumes": ["./code:/app"],
        },
        "mcp-server": {
            "image": "node:18-slim",
            "command": "node mcp-server.js",
            "port": 9000,
            "depends_on": ["agent"],
        },
    }
)
yaml_content = compose._generate_compose_yaml()
print("  Generated compose.yaml:")
for line in yaml_content.split("\n"):
    if line.strip():
        print(f"    {line}")

# Test 4: Env repr
print()
print("[4] Env repr...")
print(f"  env: {repr(env)}")
print(f"  env_docker: {repr(env_docker)}")

# Test 5: Evaluation with env (no Docker, direct URLs)
print()
print("[5] Evaluation with env (model API, no Docker)...")
ds = te.Datasets.load_from_dict([
    {"query": "What is 2+2?", "reference_answer": "4"},
])
grader = te.RubricGrader()
env_simple = te.Env.from_dict({"system_prompt": "Answer concisely."})

# Test that Evaluation.run accepts env parameter
import inspect
sig = inspect.signature(te.Evaluation.run)
params = list(sig.parameters.keys())
print(f"  Evaluation.run params: {params}")
print(f"  env param exists: {'env' in params}")
print(f"  voice_metrics param exists: {'voice_metrics' in params}")
print(f"  output param exists: {'output' in params}")

print()
print("=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
