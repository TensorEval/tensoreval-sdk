"""TensorEval SDK — Comprehensive Comparison with Other SDKs.

This document compares TensorEval with the major open-source evaluation
and training SDKs in the ecosystem.
"""

# ---------------------------------------------------------------------------
# SDK Comparison Matrix
# ---------------------------------------------------------------------------

COMPARISON = {
    "TensorEval": {
        "version": "0.1.0",
        "license": "MIT",
        "language": "Python",
        "lines_of_code": 4000,
        "dependencies": "Minimal (pydantic, openai, rich, typer)",
        "install_size": "~20-30 MB",
        "features": {
            "auto_generated_tests": True,       # Unique feature
            "verified_evaluation": True,         # Unique feature
            "ruler_zero_config": True,           # Ported from ART
            "weighted_multi_grader": True,       # From Verifiers
            "grpo_training": True,               # From verl
            "deploy_endpoint": True,             # From ART
            "cli": True,                         # From HUD
            "mcp_tools": True,                   # From HUD
            "verifiers_integration": True,        # Direct integration
            "multi_turn_env": True,              # From Verifiers
            "tool_env": True,                    # From Verifiers
            "sandbox_execution": False,          # Not yet
            "distributed_training": False,       # Not yet
            "dashboard": False,                  # Not yet
        },
    },
    "Verifiers (PrimeIntellect)": {
        "version": "0.1.14",
        "license": "MIT",
        "language": "Python",
        "lines_of_code": 35000,
        "dependencies": "Heavy (anthropic, openai, datasets, pyzmq, renderers, prime-tunnel, prime-sandboxes)",
        "install_size": "~80-120 MB",
        "features": {
            "auto_generated_tests": False,
            "verified_evaluation": False,
            "ruler_zero_config": False,
            "weighted_multi_grader": True,
            "grpo_training": True,               # via verifiers-rl
            "deploy_endpoint": False,
            "cli": True,
            "mcp_tools": True,
            "verifiers_integration": True,
            "multi_turn_env": True,
            "tool_env": True,
            "sandbox_execution": True,
            "distributed_training": True,         # via verifiers-rl
            "dashboard": False,
        },
    },
    "Inspect AI (UK AISI)": {
        "version": "0.3.x",
        "license": "MIT",
        "language": "Python",
        "lines_of_code": 80000,
        "dependencies": "Heavy (anyio, boto3, s3fs, fastapi, uvicorn, tiktoken)",
        "install_size": "~150-250 MB",
        "features": {
            "auto_generated_tests": False,
            "verified_evaluation": False,
            "ruler_zero_config": False,
            "weighted_multi_grader": False,
            "grpo_training": False,
            "deploy_endpoint": False,
            "cli": True,
            "mcp_tools": True,
            "verifiers_integration": False,
            "multi_turn_env": False,
            "tool_env": True,
            "sandbox_execution": True,
            "distributed_training": False,
            "dashboard": True,                   # Built-in viewer
        },
    },
    "ART (OpenPipe)": {
        "version": "0.x",
        "license": "Apache 2.0",
        "language": "Python",
        "lines_of_code": 103000,
        "dependencies": "Heavy (torch, vllm, litellm, weave, polars)",
        "install_size": "~500 MB+",
        "features": {
            "auto_generated_tests": False,
            "verified_evaluation": False,
            "ruler_zero_config": True,            # RULER is from ART
            "weighted_multi_grader": False,
            "grpo_training": True,
            "deploy_endpoint": True,              # Together, W&B
            "cli": True,
            "mcp_tools": False,
            "verifiers_integration": False,
            "multi_turn_env": False,
            "tool_env": False,
            "sandbox_execution": False,
            "distributed_training": True,
            "dashboard": False,
        },
    },
    "HUD (hud-python)": {
        "version": "0.x",
        "license": "MIT",
        "language": "Python",
        "lines_of_code": 50000,
        "dependencies": "Heavy (httpx, pydantic, mcp, fastmcp, openai, anthropic, asyncssh, asyncvnc)",
        "install_size": "~50-80 MB",
        "features": {
            "auto_generated_tests": False,
            "verified_evaluation": False,
            "ruler_zero_config": False,
            "weighted_multi_grader": False,
            "grpo_training": True,                # via TrainingClient
            "deploy_endpoint": False,
            "cli": True,
            "mcp_tools": True,
            "verifiers_integration": False,
            "multi_turn_env": True,
            "tool_env": True,
            "sandbox_execution": True,
            "distributed_training": False,
            "dashboard": True,                    # HUD platform
        },
    },
}


# ---------------------------------------------------------------------------
# Pros and Cons
# ---------------------------------------------------------------------------

TENSOREVAL_PROS = [
    "Auto-generated test suites from agent descriptions (unique)",
    "Verified evaluation with tool-augmented scoring (unique)",
    "RULER zero-config reward via LLM-as-judge (from ART)",
    "Weighted multi-grader with group rewards (from Verifiers)",
    "GRPO training support (from verl patterns)",
    "Clean deploy API returning OpenAI-compatible endpoints (from ART)",
    "MCP tool support for environment tools (from HUD)",
    "Verifiers environment integration (direct)",
    "Minimal dependencies (~20-30 MB install)",
    "Clean Python API with lazy imports",
    "CLI with init/eval/train/deploy commands (from HUD)",
    "Provider-agnostic types (no anthropic/openai in core)",
    "Async-first with sync wrappers",
    "4000 lines of curated, well-tested code",
]

TENSOREVAL_CONS = [
    "No distributed training (yet) — single-machine only",
    "No dashboard/viewer (yet) — CLI output only",
    "No sandbox execution (yet) — no Docker/Compose integration",
    "No production trace ingestion (yet)",
    "No continuous RL from production (yet)",
    "Auto-generation quality depends on LLM capability",
    "Verified evaluator is expensive (uses LLM with tool access)",
    "Smaller ecosystem than Verifiers (30+ envs) or Inspect (100+ evals)",
    "No built-in model serving (relies on external providers)",
    "No multi-modal support (yet) — text only",
]

TENSOREVAL_UNIQUE_VALUE = [
    "Only SDK that auto-generates test suites from agent descriptions",
    "Only SDK with tool-verified evaluation (not just LLM reasoning)",
    "Only SDK that combines eval + train + deploy in one package",
    "Only SDK with RULER + weighted multi-grader + auto-gen",
    "Smallest install footprint of any full-featured eval SDK",
    "Cleanest API surface — 5 main classes handle everything",
]


# ---------------------------------------------------------------------------
# How TensorEval Compares
# ---------------------------------------------------------------------------

def print_comparison():
    """Print a formatted comparison table."""
    print("\n" + "=" * 80)
    print("TensorEval SDK Comparison with Competitors")
    print("=" * 80)

    # Features comparison
    features = [
        "auto_generated_tests",
        "verified_evaluation",
        "ruler_zero_config",
        "weighted_multi_grader",
        "grpo_training",
        "deploy_endpoint",
        "cli",
        "mcp_tools",
        "sandbox_execution",
        "dashboard",
    ]

    feature_labels = {
        "auto_generated_tests": "Auto-Generated Tests",
        "verified_evaluation": "Verified Evaluation",
        "ruler_zero_config": "RULER Zero-Config",
        "weighted_multi_grader": "Weighted Multi-Grader",
        "grpo_training": "GRPO Training",
        "deploy_endpoint": "Deploy Endpoint",
        "cli": "CLI",
        "mcp_tools": "MCP Tools",
        "sandbox_execution": "Sandbox Execution",
        "dashboard": "Dashboard",
    }

    sdks = list(COMPARISON.keys())

    # Header
    print(f"\n{'Feature':<25}", end="")
    for sdk in sdks:
        print(f"{sdk:<18}", end="")
    print()
    print("-" * (25 + 18 * len(sdks)))

    # Rows
    for feature in features:
        label = feature_labels.get(feature, feature)
        print(f"{label:<25}", end="")
        for sdk in sdks:
            has = COMPARISON[sdk]["features"].get(feature, False)
            print(f"{'YES' if has else 'NO':<18}", end="")
        print()

    # Size comparison
    print(f"\n{'Lines of Code':<25}", end="")
    for sdk in sdks:
        loc = COMPARISON[sdk].get("lines_of_code", 0)
        print(f"{loc:<18}", end="")
    print()

    print(f"{'Install Size':<25}", end="")
    for sdk in sdks:
        size = COMPARISON[sdk].get("install_size", "unknown")
        print(f"{size:<18}", end="")
    print()

    # Pros
    print("\n" + "=" * 80)
    print("TensorEval Advantages")
    print("=" * 80)
    for pro in TENSOREVAL_PROS:
        print(f"  + {pro}")

    # Cons
    print("\n" + "=" * 80)
    print("TensorEval Limitations")
    print("=" * 80)
    for con in TENSOREVAL_CONS:
        print(f"  - {con}")

    # Unique value
    print("\n" + "=" * 80)
    print("TensorEval Unique Value Propositions")
    print("=" * 80)
    for value in TENSOREVAL_UNIQUE_VALUE:
        print(f"  * {value}")


if __name__ == "__main__":
    print_comparison()
