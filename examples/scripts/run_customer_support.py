"""Run customer support evaluation with 20 scenarios.

Tests a customer support agent across billing, technical, security,
account management, and policy scenarios. Uses AgentGrader (LLM judges rubrics).
"""

import sys
import asyncio
sys.path.insert(0, ".")
import tensoreval as te
from examples.customer_support_eval import CUSTOMER_SUPPORT_SCENARIOS

MIMO_KEY = "tp-sjgd0v2ckg08hbi46e8uvkfmonq4ywqvxv4dsa27qz6xk5zg"
MIMO_URL = "https://token-plan-sgp.xiaomimimo.com/anthropic"


async def run_eval():
    print("=" * 70)
    print("TensorEval — Customer Support Agent Evaluation")
    print("Model: mimo-v2.5-pro | Grader: AgentGrader (LLM judges rubrics)")
    print("=" * 70)

    # 1. Create dataset from scenarios
    ds = te.Datasets.load_from_dict(CUSTOMER_SUPPORT_SCENARIOS)
    print(f"\nLoaded {len(ds)} scenarios")
    categories = {}
    for s in ds:
        cat = s.metadata.get("category", "unknown") if s.metadata else "unknown"
        categories[cat] = categories.get(cat, 0) + 1
    print(f"Categories: {categories}")

    # 2. Create environment
    env = te.Env.from_dict({
        "system_prompt": """You are a professional customer support agent for a SaaS company.

Products:
- Free: $0/mo (basic features, 100 API calls/hr)
- Starter: $29/mo (1000 API calls/hr)
- Pro: $79/mo (advanced features, 1000 API calls/hr)
- Enterprise: $199/mo (unlimited, SSO, dedicated support)

Policies:
- 30-day refund window from delivery date
- Annual plans: 20% discount, non-refundable after 30 days
- API rate limits enforced per plan tier
- Security incidents: immediate key rotation
- Cancellation: access continues until end of billing period

Be empathetic, professional, and take concrete action. Always acknowledge the customer's concern before explaining policy.""",
    })

    # 3. Create grader
    grader = te.AgentGrader(
        model="mimo-v2.5-pro",
        api_key=MIMO_KEY,
        base_url=MIMO_URL,
    )

    # 4. Run evaluation
    print(f"\nRunning evaluation with {len(ds)} scenarios...")
    print(f"Workers: 3 | Grader: AgentGrader (LLM judges each rubric)")
    print()

    results = await te.Evaluation.run_async(
        ds, grader,
        env=env,
        model="mimo-v2.5-pro",
        api_key=MIMO_KEY,
        base_url=MIMO_URL,
        workers=3,
        voice_metrics=True,
    )

    # 5. Show results
    summary = results.summary()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Model:      {summary['model']}")
    print(f"  Scenarios:  {summary['num_runs']}")
    print(f"  Avg Reward: {summary['avg_reward']:.4f}")
    print(f"  Pass Rate:  {summary['pass_rate']:.1%}")
    print(f"  Passed:     {summary['pass_count']}")
    print(f"  Failed:     {summary['fail_count']}")
    print(f"  Voice:      {summary['voice_metrics']}")
    print()

    # Per-category breakdown
    cat_scores = {}
    for i, run in enumerate(results.runs):
        sample = ds[i]
        cat = sample.metadata.get("category", "unknown") if sample.metadata else "unknown"
        reward = run.get("reward", 0)
        if cat not in cat_scores:
            cat_scores[cat] = []
        cat_scores[cat].append(reward)

    print("Per-category breakdown:")
    for cat, scores in sorted(cat_scores.items()):
        avg = sum(scores) / len(scores)
        passed = sum(1 for s in scores if s >= 0.7)
        print(f"  {cat:15s}: avg={avg:.2f}, passed={passed}/{len(scores)}")
    print()

    # Per-query details
    print("Per-query details:")
    print("-" * 70)
    for i, run in enumerate(results.runs):
        sample = ds[i]
        resp = run.get("response", "").encode("ascii", "replace").decode()[:80]
        reward = run.get("reward", 0)
        status = "PASS" if reward >= 0.7 else "FAIL"
        cat = sample.metadata.get("category", "") if sample.metadata else ""
        diff = sample.metadata.get("difficulty", "") if sample.metadata else ""

        print(f"  [{status}] {sample.id}: {sample.input[:55]}...")
        print(f"       Category: {cat} | Difficulty: {diff}")
        print(f"       Response: {resp}")
        print(f"       Rubrics: {len(sample.rubrics)} graded | Score: {reward:.2f}")
        print()

    # Save results
    results.save("customer_support_results.json")
    print(f"Results saved to customer_support_results.json")

    return results


if __name__ == "__main__":
    asyncio.run(run_eval())
