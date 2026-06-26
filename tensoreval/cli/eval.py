"""Eval command — evaluate a model on a dataset or built-in environment."""

import json
import asyncio
from typing import Optional


def eval_command(
    dataset: str,
    model: str,
    workers: int,
    output: Optional[str],
    verbose: bool,
    judge: bool,
    system_prompt: Optional[str],
    api_key: Optional[str],
    base_url: Optional[str],
    n: int,
):
    """Run evaluation."""
    import sys
    sys.path.insert(0, '.')
    import tensoreval as te
    from tensoreval.cli.rich_output import (
        print_header, print_results_table, print_progress,
        print_error, print_success, print_warning, print_info,
    )

    print_header(f"Evaluating {model}")

    # Check if dataset is a built-in environment name
    built_in_envs = [e["name"] for e in te.list_envs()]
    if dataset in built_in_envs:
        print_info(f"Using built-in environment: {dataset}")
        env = te.load_env(dataset, n=n if n > 0 else 50)
        datasets = te.Datasets.from_dicts(env.dataset if isinstance(env.dataset, list) else [])
        grader = env.rubric
        env_instance = env
    else:
        # Load from file
        print_info(f"Loading dataset from {dataset}...")
        try:
            datasets = te.Datasets.load_from_file(dataset)
        except Exception as e:
            print_error(f"Failed to load dataset: {e}")
            return

        if n > 0:
            datasets = te.Datasets.from_dicts(datasets.to_dicts()[:n])

        # Create grader
        if judge:
            print_info("Using LLM-as-judge scoring")
            grader = te.Grader(model=model, judge=True)
        else:
            print_info("Using exact match scoring")
            def exact_match(state, **kwargs):
                completion = state.get("completion", [])
                answer = state.get("answer", "")
                if completion:
                    last = completion[-1]
                    response = last.get("content", "") if isinstance(last, dict) else str(getattr(last, "content", ""))
                    return 1.0 if answer in response.strip() else 0.0
                return 0.0
            grader = te.Grader(funcs=[exact_match], weights=[1.0])

        env_instance = te.SingleTurnEnv(
            rubric=grader,
            system_prompt=system_prompt or "Answer the question concisely and accurately.",
        )

    print_info(f"Model: {model}")
    print_info(f"Samples: {len(datasets)}")
    print_info(f"Workers: {workers}")
    print()

    # Resolve API credentials
    resolved_key = api_key or "tp-sjgd0v2ckg08hbi46e8uvkfmonq4ywqvxv4dsa27qz6xk5zg"
    resolved_url = base_url or "https://token-plan-sgp.xiaomimimo.com/anthropic"

    # Run evaluation
    try:
        results = asyncio.run(
            te.Evaluation.run_async(
                datasets=datasets,
                grader=grader,
                env=env_instance,
                model=model,
                api_key=resolved_key,
                base_url=resolved_url,
                workers=workers,
            )
        )
    except Exception as e:
        print_error(f"Evaluation failed: {e}")
        return

    # Display results
    summary = results.summary()
    per_query = results.per_query()

    print()
    print_results_table(summary, per_query)

    # Save if requested
    if output:
        with open(output, "w") as f:
            json.dump({"summary": summary, "per_query": per_query}, f, indent=2)
        print_success(f"\nResults saved to {output}")

    # Return summary for programmatic use
    return results
