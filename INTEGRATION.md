# TensorEval SDK Integration Guide

## Install

```bash
pip install git+https://github.com/TensorEval/tensoreval-sdk.git
```

For local development:

```bash
pip install -e .
```

## Evaluate An Agent

```python
import tensoreval as te

ds = te.Datasets.load_from_file("tasks.jsonl")
grader = te.AgentGrader(fallback_on_error=True)

results = te.Evaluation.run(
    ds,
    grader,
    agent_port=8000,
    workers=4,
)

print(results.summary())
results.save("results.json")
```

## Task Format

```jsonl
{"query":"What is 2+2?","reference_answer":"4","rubrics":[{"name":"correctness","criteria":"Must answer 4","weight":1.0}]}
{"query":"Customer wants refund after 10 days","reference_answer":"Issue refund","rubrics":[{"name":"policy","criteria":"Must verify refund window","weight":0.5},{"name":"action","criteria":"Must take the correct refund action","weight":0.5}]}
```

## Vercel AI Gateway Grading

```python
grader = te.VercelAgentGrader(model="openai/gpt-5.5")
```

The grader uses Vercel AI Gateway and can verify current facts with hosted web search. Set `AI_GATEWAY_API_KEY` locally or use `VERCEL_OIDC_TOKEN` in Vercel deployments.

Use fallback mode for local tests or CI without Gateway credentials:

```python
grader = te.AgentGrader(fallback_on_error=True)
```

## Docker / MCP

```python
env = te.Environment(
    system_prompt="You are a customer support agent.",
    agent={"image": "my-agent:latest", "port": 8000},
    mcp={"image": "my-mcp:latest", "port": 9000},
)

results = te.Evaluation.run(ds, grader, env=env, agent_port=8000, mcp_port=9000)
```
