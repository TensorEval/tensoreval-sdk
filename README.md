# TensorEval SDK

**Evaluation SDK for AI Agents — Docker, MCP, Voice, and more.**

## Quick Start

```bash
pip install git+https://github.com/TensorEval/tensoreval-sdk.git
```

```python
import tensoreval as te

# Load dataset
ds = te.Datasets.load_from_file("tasks.jsonl")

# Create grader
grader = te.RubricGrader()

# Run evaluation against your agent
results = te.Evaluation.run(ds, grader, agent_port=8000)
print(results.summary())
```

## Features

| Feature | Status | Description |
|---------|--------|-------------|
| **Docker Compose** | Working | Run agent in container, evaluate automatically |
| **Agent Endpoint** | Working | Test any HTTP endpoint (OpenAI-compatible) |
| **MCP Tools** | Working | Connect to MCP servers for tool access |
| **RubricGrader** | Working | Rule-based scoring with weighted rubrics |
| **AgentGrader** | Working | LLM reads rubrics and judges each one |
| **RulerGrader** | Working | Zero-config relative ranking |
| **Voice Metrics** | Working | TTFT, WPM, latency tracking |
| **Auto-generation** | Working | Generate tests from agent description |
| **Persistence** | Working | Save/load results to JSON |
| **Real API Tested** | Working | Tested with Mimo v2.5 Pro |

## Usage Examples

### 1. Evaluate with RubricGrader (simple answer matching)

```python
import tensoreval as te

ds = te.Datasets.load_from_dict([
    {"query": "What is 2+2?", "reference_answer": "4"},
    {"query": "What is 10*5?", "reference_answer": "50"},
])

grader = te.RubricGrader()
env = te.Env.from_dict({"system_prompt": "Answer concisely."})

results = te.Evaluation.run(ds, grader, env=env, model="mimo-v2.5-pro", api_key="...", base_url="...")
print(results.summary())
```

### 2. Evaluate with AgentGrader (LLM judges rubrics)

```python
ds = te.Datasets.load_from_dict([{
    "query": "Customer wants refund for order delivered 10 days ago",
    "reference_answer": "Issue refund of $49.99",
    "rubrics": [
        {"name": "policy", "criteria": "Must verify within 30-day window", "weight": 0.4},
        {"name": "empathy", "criteria": "Must show empathy", "weight": 0.3},
        {"name": "action", "criteria": "Must state refund amount", "weight": 0.3},
    ],
}])

grader = te.AgentGrader(model="mimo-v2.5-pro", api_key="...", base_url="...")
results = te.Evaluation.run(ds, grader, env=env, model="mimo-v2.5-pro", api_key="...", base_url="...")
```

### 3. Evaluate with Docker (agent runs in container)

```python
ds = te.Datasets.load_from_file("tasks.jsonl")
grader = te.RubricGrader()

env = te.Env.from_dict({
    "system_prompt": "You are a support agent.",
    "agent": {
        "image": "python:3.12-slim",
        "command": "python /app/agent.py",
        "port": 8000,
        "volumes": ["./my_agent:/app"],
    },
})

results = te.Evaluation.run(ds, env, grader, agent_port=8000)
```

### 4. Evaluate with agent endpoint (no Docker)

```python
ds = te.Datasets.load_from_file("tasks.jsonl")
grader = te.RubricGrader()

results = te.Evaluation.run(ds, grader, agent_port=8000)
```

### 5. RULER zero-config (no rubrics needed)

```python
ds = te.Datasets.load_from_dict([
    {"query": "Explain quantum computing"},
    {"query": "Write a haiku about programming"},
])

grader = te.RulerGrader(model="mimo-v2.5-pro", api_key="...", base_url="...")
results = te.Evaluation.run(ds, grader, env=env, model="mimo-v2.5-pro", api_key="...", base_url="...")
```

### 6. Save and load results

```python
results.save("results.json")
loaded = te.EvaluationResult.load("results.json")
print(loaded.summary())
```

## Docker Compose

```python
compose = te.DockerCompose(services={
    "agent": {
        "image": "my-agent:latest",
        "port": 8000,
        "env": {"OPENAI_API_KEY": "sk-..."},
        "volumes": ["./code:/app"],
    },
    "mcp-server": {
        "image": "my-mcp:latest",
        "port": 9000,
    },
})

ports = await compose.up()
# agent at localhost:8000, mcp at localhost:9000
await compose.down()
```

## MCP Integration

```python
server = te.MCPServer(url="http://localhost:9000/mcp", name="my-tools")
registry = te.MCPToolRegistry()
registry.add_server("cx_app", server)
```

## Built-in Environments

```python
# Load from HuggingFace
ds = te.Datasets.from_huggingface("gsm8k", split="test", n=10)

# Load from dict
ds = te.Datasets.load_from_dict([{"query": "...", "reference_answer": "..."}])

# Load from file
ds = te.Datasets.load_from_file("tasks.jsonl")
```

## Test Results

### Math Evaluation (Mimo v2.5 Pro)
```
Q01: 12 * 15?                    -> 180      [PASS]
Q02: 24 + 36?                    -> 60       [PASS]
Q03: 100 / 4?                    -> 25       [PASS]
Q04: 7 * 8?                      -> 56       [PASS]
Q05: 15% of 200?                 -> 30       [PASS]
Avg Reward: 1.0 | Pass Rate: 100%
```

### Customer Support (21 scenarios, AgentGrader)
```
Billing:       75% pass
Security:      100% pass
Policy:        100% pass
Account:       67% pass
Technical:     25% pass
Avg Reward: 0.74 | Pass Rate: 57%
```

### Voice Pipeline (TTS -> ASR -> LLM)
```
Q1: "What is 12 * 15?" -> Audio -> Transcribe -> "180" [PASS]
Q2: "3 items at $25"    -> Audio -> Transcribe -> "75"  [PASS]
Q3: "15% of 200"       -> Audio -> Transcribe -> "30"  [PASS]
Avg Reward: 1.0 | Pass Rate: 100%
```

### Docker Compose
```
Container started on port 8002
Agent ready
Q1: What is 2+2? -> 4 [PASS]
Q2: What is 12*15? -> 180 [PASS]
Q3: What is 10*5? -> 50 [PASS]
Q4: What is 100/4? -> 25 [PASS]
Avg Reward: 1.0 | Pass Rate: 100%
```

## API Reference

### Core Classes

| Class | Purpose |
|-------|---------|
| `te.Env` | Environment config + Docker lifecycle |
| `te.Datasets` | Load datasets from file/dict/HuggingFace |
| `te.RubricGrader` | Rule-based scoring |
| `te.AgentGrader` | LLM-as-judge scoring |
| `te.RulerGrader` | Zero-config relative ranking |
| `te.Evaluation` | Run evaluations |
| `te.EvaluationResult` | Results with summary/save/load |
| `te.DockerCompose` | Docker compose manager |
| `te.MCPServer` | MCP server client |

### Evaluation.run() Signature

```python
te.Evaluation.run(
    datasets: Datasets,           # Test cases
    env: Env = None,              # Environment config
    grader: Grader = None,        # Scorer (default: RubricGrader)
    model: str = "mimo-v2.5-pro", # Model name
    api_key: str = None,          # API key
    base_url: str = None,         # API base URL
    workers: int = 4,             # Concurrent workers
    agent_port: int = None,       # Agent endpoint port
    mcp_port: int = None,         # MCP server port
    system_prompt: str = None,    # System prompt (overrides env)
    output: str = None,           # Save results to file
) -> EvaluationResult
```

## Repository Structure

```
tensoreval-sdk/
├── tensoreval/              # SDK source
│   ├── __init__.py
│   ├── env.py               # Env config + Docker lifecycle
│   ├── datasets.py          # Dataset loading
│   ├── evaluation.py        # Evaluation runner
│   ├── docker_compose.py    # Docker compose manager
│   ├── enums.py             # EnvType, Modality, GraderType
│   ├── mcp_tools.py         # MCP server/client
│   ├── graders/             # RubricGrader, AgentGrader, RulerGrader
│   ├── voice/               # Voice metrics
│   └── utils/               # Helpers
├── tests/                   # Unit tests (11 passing)
├── examples/
│   ├── scripts/             # Test scripts
│   ├── scenarios/           # Test scenarios
│   ├── results/             # Test results
│   └── user_agent/          # Example user agent with Docker
├── INTEGRATION.md           # How to use in your repo
├── REPORT.md                # Customer support evaluation report
└── README.md                # This file
```

## Links

- **GitHub:** https://github.com/TensorEval/tensoreval-sdk
- **Integration Guide:** [INTEGRATION.md](INTEGRATION.md)
- **Evaluation Report:** [examples/REPORT.md](examples/REPORT.md)
