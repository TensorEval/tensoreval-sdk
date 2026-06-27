"""Real API test with Env + Docker config."""

import sys
import asyncio
sys.path.insert(0, ".")
import tensoreval as te


async def test():
    print("=" * 60)
    print("Real API Test with Env + RubricGrader + AgentGrader")
    print("=" * 60)

    # Env with system prompt
    env = te.Env.from_dict({
        "system_prompt": "You are a math solver. Give ONLY the numerical answer.",
    })

    # Dataset with rubrics
    ds = te.Datasets.load_from_dict([
        {
            "query": "What is 12 * 15?",
            "reference_answer": "180",
            "rubrics": [
                {"name": "correctness", "criteria": "Must answer 180", "weight": 0.7},
                {"name": "method", "criteria": "Must show work", "weight": 0.3},
            ],
        },
        {
            "query": "What is 25% of 200?",
            "reference_answer": "50",
            "rubrics": [
                {"name": "correctness", "criteria": "Must answer 50", "weight": 0.7},
                {"name": "method", "criteria": "Must show percentage calculation", "weight": 0.3},
            ],
        },
        {
            "query": "A train travels 60mph for 2.5 hours. How far?",
            "reference_answer": "150",
            "rubrics": [
                {"name": "correctness", "criteria": "Must compute 60*2.5=150", "weight": 0.6},
                {"name": "units", "criteria": "Must include miles", "weight": 0.2},
                {"name": "method", "criteria": "Must show formula", "weight": 0.2},
            ],
        },
    ])

    # AgentGrader
    grader = te.AgentGrader(
        model="mimo-v2.5-pro",
        api_key="tp-sjgd0v2ckg08hbi46e8uvkfmonq4ywqvxv4dsa27qz6xk5zg",
        base_url="https://token-plan-sgp.xiaomimimo.com/anthropic",
    )

    # Run
    results = await te.Evaluation.run_async(
        ds, grader,
        env=env,
        model="mimo-v2.5-pro",
        api_key="tp-sjgd0v2ckg08hbi46e8uvkfmonq4ywqvxv4dsa27qz6xk5zg",
        base_url="https://token-plan-sgp.xiaomimimo.com/anthropic",
        workers=3,
        voice_metrics=True,
    )

    summary = results.summary()
    print()
    print("Model: " + str(summary["model"]))
    print("Samples: " + str(summary["num_runs"]))
    print("Avg Reward: " + str(summary["avg_reward"]))
    print("Pass Rate: " + str(summary["pass_rate"]))
    print("Voice: " + str(summary["voice_metrics"]))
    print()

    for i, run in enumerate(results.runs):
        sample = ds[i]
        resp = run.get("response", "").encode("ascii", "replace").decode()[:60]
        reward = run.get("reward", 0)
        vm = run.get("voice_metrics", {})
        status = "PASS" if reward >= 0.7 else "FAIL"
        print(f"Q{i+1}: {sample.input}")
        print(f"     Expected: {sample.target}")
        print(f"     Got: {resp}")
        print(f"     Rubrics: {len(sample.rubrics)} graded by LLM")
        print(f"     Score: {reward:.2f} [{status}]")
        print(f"     Latency: {vm.get('latency_ms', 0):.0f}ms")
        print()


if __name__ == "__main__":
    asyncio.run(test())
