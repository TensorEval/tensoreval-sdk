# TensorEval SDK

Minimal evaluation SDK for black-box agents.

TensorEval sends one OpenAI-shaped request to an agent endpoint, waits for a
response, extracts the final answer, grades it, and writes a report. Tool traces
are optional: if the agent returns `trace.steps`, the report includes them.

## Quick Start

```python
import tensoreval as te

env = te.Env.from_endpoint(
    agent_url="http://localhost:8000/v1/chat/completions",
    mcp_urls=["http://localhost:8001/mcp"],  # optional
)

dataset = te.Dataset.from_jsonl("tasks.jsonl")
grader = te.AgenticGrader(
    provider="openai",
    model="gpt-4.1",
    api_key="sk-...",
    pass_threshold=0.8,
)

result = te.Evaluation.run(
    dataset=dataset,
    env=env,
    grader=grader,
    workers=1,
    timeout=300,
    agent_config={"max_steps": 20, "mode": "eval"},
)

print(result.summary())
result.report("report.html")
```

If no provider config is passed, TensorEval uses local heuristic grading unless
`TENSOREVAL_GRADER_API_KEY` is set. With a grader API key, `AgenticGrader`
defaults to OpenAI `gpt-4.1`.

## Agent Endpoint Contract

The endpoint receives an OpenAI-shaped request plus TensorEval metadata.

```json
{
  "model": "default",
  "messages": [
    {"role": "user", "content": "Refund order O123"}
  ],
  "agent_config": {
    "max_steps": 20,
    "mode": "eval"
  },
  "metadata": {
    "sample_id": "case_001"
  },
  "mcp_servers": [
    {"name": "mcp_0", "url": "http://localhost:8001/mcp"}
  ]
}
```

`agent_config` is passed through unchanged. TensorEval only uses `timeout` as
the HTTP request timeout.

## Accepted Responses

Preferred response:

```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "Order O123 is eligible for refund."
      }
    }
  ]
}
```

TensorEval also accepts common final-output fields:

```json
{"final_response": "..."}
{"output_text": "..."}
{"answer": "..."}
{"response": "..."}
{"output": "..."}
{"result": "..."}
```

## Optional Trace Schema

Agents can return traces for richer reports:

```json
{
  "choices": [
    {"message": {"role": "assistant", "content": "Refund approved."}}
  ],
  "trace": {
    "schema_version": "tensoreval.trace.v1",
    "steps": [
      {
        "id": "s1",
        "type": "message",
        "role": "assistant",
        "content": "Checking the order."
      },
      {
        "id": "s2",
        "type": "tool_call",
        "name": "lookup_order",
        "arguments": {"order_id": "O123"},
        "mcp_server": "mcp_0"
      },
      {
        "id": "s3",
        "type": "tool_result",
        "tool_call_id": "s2",
        "name": "lookup_order",
        "result": {"status": "delivered"},
        "is_error": false
      }
    ]
  }
}
```

Supported step types:

```text
message
tool_call
tool_result
reasoning_summary
artifact
error
```

Do not return hidden chain-of-thought. Use `reasoning_summary` only for safe,
agent-provided summaries.

## Agentic Grader

`AgenticGrader` is a grader-side tool loop. It receives the query, final
response, rubrics, reference answer, optional agent trace, and the same
`mcp_urls` configured on the environment. It can call those MCP tools while
grading.

```python
grader = te.AgenticGrader(
    provider="openai",        # openai | anthropic | ollama | openrouter
    model="gpt-4.1",
    api_key="sk-...",
    base_url=None,
    instructions="Grade using rubrics, trace, reference answer, and MCP state.",
    max_steps=10,
    timeout=120,
    temperature=0,
)
```

Supported providers:

```text
openai      -> https://api.openai.com/v1/chat/completions
openrouter  -> https://openrouter.ai/api/v1/chat/completions
ollama      -> http://localhost:11434/v1/chat/completions
anthropic   -> https://api.anthropic.com/v1/messages
```

OpenAI, OpenRouter, and Ollama use an OpenAI-compatible tool-call loop.
Anthropic uses the Messages API tool-use format.

The grader returns:

```json
{
  "reward": 0.9,
  "passed": true,
  "rubric_scores": {
    "correctness": {"score": 0.9, "reason": "Verified with order state."}
  },
  "reasoning_summary": "The answer matches policy and tool state.",
  "grader_trace": {
    "steps": [
      {"type": "tool_call", "name": "lookup_order", "arguments": {"order_id": "O123"}},
      {"type": "tool_result", "name": "lookup_order", "result": {"status": "delivered"}}
    ]
  }
}
```

## Dataset Format

`query` is required. Rubrics and reference answers are optional.

```json
{
  "id": "case_001",
  "query": "Refund order O123",
  "reference_answer": "Refund approved",
  "rubrics": [
    {
      "id": "correctness",
      "description": "Must approve eligible refund",
      "weight": 1.0
    }
  ],
  "setup_script": "",
  "verify_script": "",
  "metadata": {}
}
```

Load it:

```python
dataset = te.Dataset.from_jsonl("tasks.jsonl")
```

## Public API

```python
te.Env.from_endpoint(agent_url, mcp_urls=None)
te.Env.from_dockerfile(path="Dockerfile", agent_port=8000, mcp_ports=None)
te.Env.from_yaml("tensoreval.yaml")
te.Env.from_registry("browseros")  # reserved

te.Dataset.from_jsonl("tasks.jsonl")
te.Dataset.from_dicts([...])

te.AgenticGrader(...)
te.ProviderConfig.from_name("openai")
te.Evaluation.run(...)
te.EvaluationResult.report("report.html")
```

`workers > 1` is reserved for Docker-backed isolated environments. Endpoint
environments run with `workers=1`.
