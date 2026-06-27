"""Test: Complete rubric + env + evaluation flow."""

import sys
import asyncio
sys.path.insert(0, ".")
import tensoreval as te


async def test():
    print("=" * 60)
    print("Complete Flow: Rubrics + Env + Evaluation")
    print("=" * 60)
    print()

    # Step 1: Define rubrics per sample (user-defined)
    # Each sample has its OWN rubrics with weights summing to 1.0
    ds = te.Datasets.load_from_dict([
        {
            "query": "Customer wants refund for order #12345, delivered 10 days ago.",
            "reference_answer": "Issue full refund of $49.99",
            "rubrics": [
                {"name": "policy_check", "criteria": "Must verify the order is within 30-day refund window", "weight": 0.4},
                {"name": "action_taken", "criteria": "Must state the refund amount clearly", "weight": 0.3},
                {"name": "empathy", "criteria": "Must acknowledge the customer concern professionally", "weight": 0.2},
                {"name": "professionalism", "criteria": "Must be polite and helpful throughout", "weight": 0.1},
            ],
        },
        {
            "query": "Customer wants refund for order #67890, delivered 45 days ago.",
            "reference_answer": "Politely refuse citing 30-day policy",
            "rubrics": [
                {"name": "policy_compliance", "criteria": "Must refuse citing 30-day refund policy window", "weight": 0.5},
                {"name": "empathy", "criteria": "Must be understanding not dismissive of the situation", "weight": 0.3},
                {"name": "alternative", "criteria": "May offer alternative solution like store credit exchange", "weight": 0.2},
            ],
        },
        {
            "query": "What is 12 multiplied by 15?",
            "reference_answer": "180",
            "rubrics": [
                {"name": "correctness", "criteria": "Must compute the correct answer 180", "weight": 0.7},
                {"name": "methodology", "criteria": "Must show multiplication steps clearly", "weight": 0.3},
            ],
        },
    ], name="test_scenarios")

    print(f"[1] Dataset: {len(ds)} samples")
    for i, s in enumerate(ds):
        print(f"    Sample {i+1}: {len(s.rubrics)} rubrics")
        for r in s.rubrics:
            print(f"      - {r['name']} (weight: {r['weight']}): {r['criteria'][:50]}...")
    print()

    # Step 2: Create environment config
    env = te.Env.from_dict({
        "system_prompt": "You are a professional customer support agent. Be helpful and follow company policies.",
    })

    print(f"[2] Env: system_prompt = '{env.system_prompt[:40]}...'")
    print()

    # Step 3: Create grader (uses per-sample rubrics automatically)
    grader = te.RubricGrader()

    print("[3] Grader: RubricGrader (uses per-sample rubrics)")
    print()

    # Step 4: Run evaluation
    print("[4] Running evaluation...")
    results = await te.Evaluation.run_async(
        datasets=ds,
        grader=grader,
        env=env,
        model="mimo-v2.5-pro",
        api_key="tp-sjgd0v2ckg08hbi46e8uvkfmonq4ywqvxv4dsa27qz6xk5zg",
        base_url="https://token-plan-sgp.xiaomimimo.com/anthropic",
        workers=3,
    )

    # Step 5: Results
    summary = results.summary()
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Model:      {summary['model']}")
    print(f"  Samples:    {summary['num_runs']}")
    print(f"  Avg Reward: {summary['avg_reward']:.4f}")
    print(f"  Pass Rate:  {summary['pass_rate']:.1%}")
    print()

    for i, run in enumerate(results.runs):
        sample = ds[i]
        completion = run.get("completion", [])
        response = completion[-1].get("content", "") if completion else ""
        reward = run.get("reward", 0)
        status = "PASS" if reward >= 0.6 else "FAIL"

        print(f"  Q{i+1}: {sample.input[:50]}...")
        print(f"       Expected: {sample.target}")
        print(f"       Got: {response.strip()[:60]}")
        print(f"       Rubrics: {len(sample.rubrics)} checked")
        print(f"       Score: {reward:.2f} [{status}]")
        print()


if __name__ == "__main__":
    asyncio.run(test())
