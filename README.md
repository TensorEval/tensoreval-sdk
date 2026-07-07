# TensorEval SDK

Evaluation SDK for AI agents with Docker, MCP, backend ingestion, and Vercel AI Gateway grading.

## Quick Start

```bash
pip install git+https://github.com/TensorEval/tensoreval-sdk.git
```

```python
import tensoreval as te

ds = te.Datasets.load_from_dict([
    {"query": "What is 2+2?", "reference_answer": "4"},
])

grader = te.AgentGrader(fallback_on_error=True)
results = te.Evaluation.run(ds, grader, agent=my_agent)
print(results.summary())
```

## Grading

`te.AgentGrader` is backed by Vercel AI Gateway. It evaluates each response against dataset rubrics and can use hosted web search to verify current facts.

```python
grader = te.VercelAgentGrader(model="openai/gpt-5.5")
```

For offline/local tests, use deterministic fallback scoring:

```python
grader = te.AgentGrader(fallback_on_error=True)
```

## Rubric Dataset

```python
ds = te.Datasets.load_from_dict([{
    "query": "Customer wants refund for order delivered 10 days ago",
    "reference_answer": "Issue refund of $49.99",
    "rubrics": [
        {"name": "policy", "criteria": "Must verify this is inside the refund window", "weight": 0.4},
        {"name": "empathy", "criteria": "Must acknowledge customer frustration", "weight": 0.3},
        {"name": "action", "criteria": "Must clearly state the refund action", "weight": 0.3},
    ],
}])
```

## Docker And MCP

```python
env = te.Environment(
    system_prompt="You are a support agent.",
    agent={"image": "my-agent:latest", "port": 8000},
    mcp={"image": "my-mcp:latest", "port": 9000},
)

results = te.Evaluation.run(ds, grader, env=env, agent_port=8000, mcp_port=9000)
```

## Public API

| Class | Purpose |
|-------|---------|
| `te.AgentGrader` / `te.VercelAgentGrader` | Vercel AI Gateway rubric grader |
| `te.Evaluation` | Run evaluations |
| `te.EvaluationResult` | Results with summary/save/load |
| `te.Datasets` | Load datasets from dict/file/HuggingFace |
| `te.Environment` | Environment config, Docker lifecycle, local tools, and MCP |
| `te.MCPServer` / `te.tool` | MCP/local tool integration |

## Environment

Live grading uses Vercel AI Gateway credentials. Set `AI_GATEWAY_API_KEY` locally or rely on `VERCEL_OIDC_TOKEN` in Vercel deployments. `TENSOREVAL_MODEL_API_KEY` is also accepted as a fallback.
