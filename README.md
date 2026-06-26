# TensorEval SDK

**Evaluation, Training, and Deployment for AI Agents.**

TensorEval takes the best patterns from 8 open-source projects (Verifiers, ART, HUD, Inspect AI, Tinker, verl) and combines them into one clean API with 9 CLI commands.

## Quick Start

```bash
pip install tensoreval
```

```python
import tensoreval as te

# Load a built-in environment
env = te.load_env("gsm8k", n=10)

# Evaluate with real API
results = te.Evaluation.run(
    datasets=te.Datasets.from_dicts(env.dataset),
    grader=env.rubric,
    env=env,
    model="mimo-v2.5-pro",
    api_key="your-key",
    base_url="https://api.anthropic.com",
)
print(results.summary())
```

## CLI

```bash
# List built-in environments
tensoreval envs

# Evaluate a model
tensoreval eval gsm8k --model mimo-v2.5-pro --n 10

# Evaluate with custom dataset
tensoreval eval tasks.jsonl --model mimo-v2.5-pro --judge

# Scaffold a new environment
tensoreval init my-agent --preset cx

# Train with GRPO
tensoreval train tasks.jsonl --base-model Qwen/Qwen3-8B --algorithm grpo

# Deploy trained model
tensoreval deploy <run-id> --name my-agent-v2 --provider together

# Show version and Docker status
tensoreval version
```

## Built-in Environments

| Environment | Category | Description |
|-------------|----------|-------------|
| `gsm8k` | math | Grade-school math problems |
| `math` | math | Competition math problems |
| `code-generation` | coding | Python code generation tasks |
| `customer-support` | business | Customer support agent scenarios |
| `data-analysis` | analytics | Business data analysis tasks |
| `reasoning` | reasoning | Logic and reasoning puzzles |

```python
# Load any built-in environment
env = te.load_env("customer-support", n=20)
env = te.load_env("code-generation", n=10)
env = te.load_env("reasoning", n=15)
```

## Features

### Auto-Generated Tests (Unique)

```python
# Generate test suite from agent description
datasets = te.AutoGenerator.generate(
    agent_name="SupportBot",
    agent_description="Handles billing inquiries for SaaS platform",
    capabilities=["lookup_order", "issue_refund"],
    count=20,
    model="mimo-v2.5-pro",
)
```

### Verified Evaluation (Unique)

```python
# Evaluator runs code, reads files, searches web to VERIFY answers
grader = te.Grader(model="mimo-v2.5-pro", verified=True)
```

### RULER Zero-Config Reward

```python
# No reward function needed — LLM judge ranks trajectories
grader = te.Grader.ruler(model="mimo-v2.5-pro")
```

### Weighted Multi-Grader

```python
# Combine multiple reward functions with weights
def correctness(state, **kwargs) -> float:
    ...

def completeness(state, **kwargs) -> float:
    ...

grader = te.Grader(funcs=[correctness, completeness], weights=[0.7, 0.3])
```

### Docker Sandbox

```python
# Run evaluation in isolated Docker containers
env = te.DockerSandboxEnv(
    image="python:3.12-slim",
    rubric=my_rubric,
    system_prompt="You are a coding agent.",
)
```

### Multi-Turn Tool Calling

```python
# Agent can call Python functions as tools
def search_database(query: str) -> str:
    '''Search the database for matching records.'''
    return db.search(query)

env = te.MultiTurnToolEnv(
    tools=[search_database],
    rubric=my_rubric,
    max_turns=10,
)
```

### MCP Tool Integration

```python
# Connect to existing MCP servers
server = te.MCPServer(url="http://localhost:9000/mcp", name="my-tools")
registry = te.MCPToolRegistry()
registry.add_server("cx_app", server)
```

### GRPO Training

```python
# Train with Group Relative Policy Optimization
trainer = te.Training.run(
    datasets=datasets,
    base_model="Qwen/Qwen3-8B",
    algorithm="grpo",
    rollouts_per_example=8,
)
```

### SFT/DPO Export

```python
# Export evaluation results as training data
exporter = te.TrainingDataExporter()
sft_data = exporter.export_sft(results.runs, teacher_model="mimo-v2.5-pro")
exporter.save_jsonl(sft_data, "training_data.jsonl")
```

### Deploy

```python
# Deploy trained model as OpenAI-compatible endpoint
endpoint = trainer.deploy(name="my-agent-v3", provider="together")
print(endpoint.model_id)  # "tensoreval/my-agent-v3"
print(endpoint.base_url)  # "https://api.together.xyz/v1"
```

## Architecture

```
tensoreval/
├── core/           # Types, errors, decorators (from Verifiers)
├── parsers/        # Answer extraction (from Verifiers)
├── rubrics/        # Scoring system (Verifiers + ART RULER)
├── envs/           # Environment classes (Verifiers + HUD)
├── training/       # GRPO, SFT/DPO export (verl + Tinker patterns)
├── deploy/         # Model deployment (ART pattern)
├── cli/            # CLI commands (HUD pattern)
├── datasets.py     # Dataset loading
├── evaluation.py   # Evaluation runner
├── grader.py       # Grader with RULER fallback
├── environments.py # Built-in environments
├── auto_generate.py # Auto-generate tests from description
├── verified_evaluator.py # Tool-verified scoring
├── mcp_tools.py    # MCP server integration
└── verifiers_integration.py # Load Verifiers environments
```

## Comparison with Other SDKs

| Feature | TensorEval | Verifiers | Inspect AI | ART | HUD |
|---------|-----------|-----------|------------|-----|-----|
| Auto-generated tests | **YES** | No | No | No | No |
| Verified evaluation | **YES** | No | No | No | No |
| RULER zero-config | **YES** | No | No | YES | No |
| Weighted multi-grader | **YES** | YES | No | No | No |
| GRPO training | **YES** | YES | No | YES | YES |
| Deploy endpoint | **YES** | No | No | YES | No |
| CLI | **YES** | YES | YES | YES | YES |
| MCP tools | **YES** | YES | YES | No | YES |
| Docker sandbox | **YES** | YES | YES | No | YES |
| Built-in envs | **6** | 30+ | 100+ | 0 | 14 presets |
| Lines of code | **6,800** | 35,000 | 80,000 | 103,000 | 50,000 |
| Install size | **~20 MB** | ~80 MB | ~150 MB | ~500 MB | ~50 MB |

## Dependencies

```toml
[project]
dependencies = [
    "pydantic>=2.0",
    "openai>=1.0",
    "rich>=13.0",
    "typer>=0.12",
    "httpx>=0.27",
    "tqdm>=4.0",
    "tenacity>=8.0",
]

[project.optional-dependencies]
anthropic = ["anthropic>=0.30"]
litellm = ["litellm>=1.0"]
datasets = ["datasets>=2.0"]
tinker = ["tinker>=0.22.0"]
verifiers = ["verifiers>=0.1.14"]
```

## License

MIT

## Links

- [GitHub](https://github.com/krissint/tensoreval-sdk)
- [Documentation](https://github.com/krissint/tensoreval-sdk#readme)
