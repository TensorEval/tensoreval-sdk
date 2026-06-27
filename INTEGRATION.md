# TensorEval SDK — Integration Guide

## How To Use In Your Repo

### Option 1: Install from GitHub

```bash
pip install git+https://github.com/TensorEval/tensoreval-sdk.git
```

### Option 2: Install as local package

```bash
# Clone the SDK
git clone https://github.com/TensorEval/tensoreval-sdk.git
cd tensoreval-sdk

# Install in your project
pip install -e .
```

### Option 3: Copy into your project

```bash
# Copy the tensoreval/ directory into your project
cp -r tensoreval-sdk/tensoreval your-project/tensoreval/
```

---

## Quick Start: Evaluate Your Agent

### Step 1: Create a dataset

```python
# tasks.jsonl — one JSON object per line
{"query": "What is 2+2?", "reference_answer": "4", "rubrics": [{"name": "correctness", "criteria": "Must answer 4", "weight": 1.0}]}
{"query": "Customer wants refund for order delivered 10 days ago", "reference_answer": "Issue refund", "rubrics": [{"name": "policy", "criteria": "Must verify 30-day window", "weight": 0.5}, {"name": "action", "criteria": "Must take action", "weight": 0.5}]}
```

### Step 2: Run evaluation

```python
import tensoreval as te

# Load dataset
ds = te.Datasets.load_from_file("tasks.jsonl")

# Create grader
grader = te.RubricGrader()  # Simple answer matching
# OR
grader = te.AgentGrader(    # LLM reads rubrics and judges
    model="mimo-v2.5-pro",
    api_key="your-key",
    base_url="https://api.example.com/v1",
)

# Create environment
env = te.Env.from_dict({"system_prompt": "You are a support agent."})

# Run evaluation
results = te.Evaluation.run(ds, env, grader, workers=4)

# View results
print(results.summary())
# {'model': 'mimo-v2.5-pro', 'num_runs': 2, 'avg_reward': 0.85, 'pass_rate': 1.0}

# Save results
results.save("results.json")
```

---

## Using Docker

### Step 1: Create config.yaml

```yaml
system_prompt: "You are a customer support agent..."
agent:
  image: python:3.12-slim
  command: python agent.py
  port: 8000
  env:
    OPENAI_API_KEY: sk-...
  volumes:
    - ./my_agent_code:/app
mcp:
  image: node:18-slim
  command: node mcp-server.js
  port: 9000
env_file: .env
```

### Step 2: Load and use

```python
import tensoreval as te

env = te.Env.load_from_file("config.yaml")
# Docker containers start automatically when Evaluation.run() is called

ds = te.Datasets.load_from_file("tasks.jsonl")
grader = te.AgentGrader(model="mimo-v2.5-pro", api_key="...", base_url="...")
results = te.Evaluation.run(ds, env, grader)
```

### Step 3: Or use DockerCompose directly

```python
compose = te.DockerCompose(services={
    "agent": {
        "image": "my-agent:latest",
        "port": 8000,
        "env": {"API_KEY": "sk-..."},
        "volumes": ["./code:/app"],
    },
    "mcp-server": {
        "image": "my-mcp:latest",
        "port": 9000,
    },
})

ports = await compose.up()
# agent is at localhost:8000
# mcp-server is at localhost:9000

# ... run evaluation ...

await compose.down()
```

---

## Testing With Real Agent Endpoint

If you already have an agent running:

```python
import tensoreval as te

ds = te.Datasets.load_from_file("tasks.jsonl")
grader = te.RubricGrader()

# Point to your running agent
results = te.Evaluation.run(
    ds, grader,
    agent_port=8000,  # calls http://localhost:8000/v1/chat/completions
    system_prompt="You are a support agent.",
)
```

---

## MCP Integration

```python
import tensoreval as te

# Connect to MCP server
server = te.MCPServer(url="http://localhost:9000/mcp", name="my-tools")
registry = te.MCPToolRegistry()
registry.add_server("cx_app", server)

# Tools are auto-discovered
# tools = await registry.list_all_tools()
```

---

## Grader Types

```python
# 1. RubricGrader — simple answer matching
grader = te.RubricGrader()

# 2. AgentGrader — LLM reads rubrics and judges each one
grader = te.AgentGrader(model="mimo-v2.5-pro", api_key="...", base_url="...")

# 3. RulerGrader — zero-config relative ranking (for GRPO training)
grader = te.RulerGrader(model="mimo-v2.5-pro", api_key="...", base_url="...")
```

---

## File Structure In Your Repo

```
your-project/
├── .env                    # API keys
├── config.yaml             # Docker/MCP config
├── tasks.jsonl             # Test dataset
├── my_agent_code/          # Your agent code
│   └── agent.py
├── results/                # Evaluation results
│   └── results.json
├── tests/
│   └── test_agent.py       # Your tests using TensorEval
└── requirements.txt
```

---

## Example: Full Test File

```python
# tests/test_agent.py
import tensoreval as te
import asyncio

async def test_customer_support():
    ds = te.Datasets.load_from_file("tasks.jsonl")
    grader = te.AgentGrader(
        model="mimo-v2.5-pro",
        api_key="tp-...",
        base_url="https://token-plan-sgp.xiaomimimo.com/anthropic",
    )
    env = te.Env.from_dict({
        "system_prompt": "You are a professional support agent..."
    })

    results = await te.Evaluation.run_async(
        ds, env, grader,
        model="mimo-v2.5-pro",
        api_key="tp-...",
        base_url="https://token-plan-sgp.xiaomimimo.com/anthropic",
        workers=5,
    )

    summary = results.summary()
    assert summary["pass_rate"] >= 0.7, f"Pass rate {summary['pass_rate']} below threshold"
    assert summary["avg_reward"] >= 0.6, f"Avg reward {summary['avg_reward']} below threshold"

    results.save("results/test_results.json")

if __name__ == "__main__":
    asyncio.run(test_customer_support())
```
