"""TensorEval SDK — Complete Usage Examples.

Shows every major feature with real, runnable code.
All examples use Mimo v2.5 Pro API.
"""

import tensoreval as te

# ============================================================
# Config
# ============================================================
API_KEY = "tp-sjgd0v2ckg08hbi46e8uvkfmonq4ywqvxv4dsa27qz6xk5zg"
BASE_URL = "https://token-plan-sgp.xiaomimimo.com/anthropic"


# ============================================================
# Example 1: Simple evaluation with RubricGrader
# ============================================================
def example_rubric_grader():
    """Simple answer matching — checks if reference answer is in response."""
    ds = te.Datasets.load_from_dict([
        {"query": "What is 2+2?", "reference_answer": "4"},
        {"query": "What is 10*5?", "reference_answer": "50"},
    ])

    grader = te.RubricGrader()
    env = te.Env.from_dict({"system_prompt": "Answer concisely."})

    results = te.Evaluation.run(
        ds, grader, env=env,
        model="mimo-v2.5-pro", api_key=API_KEY, base_url=BASE_URL,
    )
    print(results.summary())


# ============================================================
# Example 2: LLM-graded rubrics (AgentGrader)
# ============================================================
def example_agent_grader():
    """LLM reads rubrics and judges each one."""
    ds = te.Datasets.load_from_dict([{
        "query": "Customer wants refund for order delivered 10 days ago",
        "reference_answer": "Issue refund of $49.99",
        "rubrics": [
            {"name": "policy", "criteria": "Must verify within 30-day window", "weight": 0.4},
            {"name": "empathy", "criteria": "Must show empathy", "weight": 0.3},
            {"name": "action", "criteria": "Must state refund amount", "weight": 0.3},
        ],
    }])

    grader = te.AgentGrader(model="mimo-v2.5-pro", api_key=API_KEY, base_url=BASE_URL)
    env = te.Env.from_dict({"system_prompt": "You are a support agent."})

    results = te.Evaluation.run(ds, grader, env=env, model="mimo-v2.5-pro", api_key=API_KEY, base_url=BASE_URL)
    print(results.summary())


# ============================================================
# Example 3: RULER zero-config (no rubrics needed)
# ============================================================
def example_ruler():
    """LLM ranks responses relative to each other. No rubrics needed."""
    ds = te.Datasets.load_from_dict([
        {"query": "Explain quantum computing in simple terms"},
        {"query": "Write a haiku about programming"},
    ])

    grader = te.RulerGrader(model="mimo-v2.5-pro", api_key=API_KEY, base_url=BASE_URL)
    env = te.Env.from_dict({"system_prompt": "You are a helpful assistant."})

    results = te.Evaluation.run(ds, grader, env=env, model="mimo-v2.5-pro", api_key=API_KEY, base_url=BASE_URL)
    print(results.summary())


# ============================================================
# Example 4: Environment with Docker
# ============================================================
def example_docker_env():
    """Environment with Docker containers for agent and MCP server."""
    env = te.Env.from_dict({
        "system_prompt": "You are a support agent.",
        "agent": {
            "image": "python:3.12-slim",
            "command": "python agent.py",
            "port": 8000,
            "env": {"OPENAI_API_KEY": "sk-..."},
            "volumes": ["./agent_code:/app"],
        },
        "mcp": {
            "image": "node:18-slim",
            "command": "node mcp-server.js",
            "port": 9000,
        },
        "env_file": ".env",
    })

    # Docker starts automatically when env.start() is called
    # Or: Evaluation.run() handles it
    print(repr(env))


# ============================================================
# Example 5: MCP tools
# ============================================================
def example_mcp():
    """Connect to MCP server for tool access."""
    server = te.MCPServer(url="http://localhost:9000/mcp", name="my-tools")
    registry = te.MCPToolRegistry()
    registry.add_server("cx_app", server)

    # Tools are auto-discovered
    # tools = await registry.list_all_tools()


# ============================================================
# Example 6: Voice metrics
# ============================================================
def example_voice():
    """Evaluation with voice metrics (latency, WPM)."""
    ds = te.Datasets.load_from_dict([
        {"query": "What is the refund policy?"},
    ])

    grader = te.RubricGrader()
    env = te.Env.from_dict({"system_prompt": "You are a support agent."})

    results = te.Evaluation.run(
        ds, grader, env=env,
        model="mimo-v2.5-pro", api_key=API_KEY, base_url=BASE_URL,
        voice_metrics=True,  # Enables TTFT, WPM, latency tracking
    )
    print(results.summary())


# ============================================================
# Example 7: Save and load results
# ============================================================
def example_persistence():
    """Save results to JSON and load them back."""
    ds = te.Datasets.load_from_dict([{"query": "What is 2+2?", "reference_answer": "4"}])
    grader = te.RubricGrader()
    env = te.Env.from_dict({"system_prompt": "Answer concisely."})

    # Save
    results = te.Evaluation.run(ds, grader, env=env, model="mimo-v2.5-pro", api_key=API_KEY, base_url=BASE_URL, output="results.json")

    # Load
    loaded = te.EvaluationResult.load("results.json")
    print(loaded.summary())


# ============================================================
# Example 8: CLI usage
# ============================================================
"""
# Initialize environment
tensoreval init my-agent --preset cx

# List built-in environments
tensoreval envs

# Evaluate
tensoreval eval tasks.jsonl --model mimo-v2.5-pro

# Train
tensoreval train tasks.jsonl --base-model Qwen/Qwen3-8B --algorithm grpo

# Deploy
tensoreval deploy <run-id> --name my-agent-v2

# Version
tensoreval version
"""


# ============================================================
# Run examples
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("TensorEval SDK Examples")
    print("=" * 50)

    print("\n[1] RubricGrader (simple answer matching)")
    example_rubric_grader()

    print("\n[2] AgentGrader (LLM judges rubrics)")
    example_agent_grader()

    print("\n[3] Voice metrics")
    example_voice()

    print("\n[4] Persistence")
    example_persistence()

    print("\nAll examples complete!")
