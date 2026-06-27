"""Test: LLM-graded rubric evaluation with Mimo."""

import sys, asyncio
sys.path.insert(0, ".")
import tensoreval as te

MIMO_KEY = "tp-sjgd0v2ckg08hbi46e8uvkfmonq4ywqvxv4dsa27qz6xk5zg"
MIMO_URL = "https://token-plan-sgp.xiaomimimo.com/anthropic"

async def test():
    # 1. Dataset with rubrics
    ds = te.Datasets.load_from_dict([
        {
            "query": "Customer wants refund for order #12345, delivered 10 days ago. Total was $49.99.",
            "reference_answer": "Issue full refund of $49.99",
            "rubrics": [
                {"name": "policy", "criteria": "Must verify within 30-day window and approve refund", "weight": 0.4},
                {"name": "amount", "criteria": "Must mention the exact refund amount $49.99", "weight": 0.3},
                {"name": "empathy", "criteria": "Must show empathy toward customer", "weight": 0.2},
                {"name": "clarity", "criteria": "Response must be clear and actionable", "weight": 0.1},
            ],
        },
        {
            "query": "Customer wants refund for order #67890, delivered 45 days ago. Total was $120.",
            "reference_answer": "Politely refuse citing 30-day policy",
            "rubrics": [
                {"name": "policy", "criteria": "Must refuse citing 30-day refund window policy", "weight": 0.5},
                {"name": "empathy", "criteria": "Must be understanding, not dismissive", "weight": 0.3},
                {"name": "alternative", "criteria": "Should offer alternative like store credit", "weight": 0.2},
            ],
        },
        {
            "query": "Customer's API key leaked on GitHub. Need immediate action.",
            "reference_answer": "Revoke key immediately, generate new one",
            "rubrics": [
                {"name": "urgency", "criteria": "Must treat as critical security incident", "weight": 0.4},
                {"name": "action", "criteria": "Must revoke old key and issue new one", "weight": 0.4},
                {"name": "communication", "criteria": "Must explain what happened clearly", "weight": 0.2},
            ],
        },
    ])

    # 2. Env config
    env = te.Env.from_dict({
        "system_prompt": "You are a customer support agent for a SaaS company. Policy: 30-day refund window. Be helpful and professional."
    })

    # 3. AgentGrader — LLM reads rubrics and judges each one
    grader = te.AgentGrader(
        model="mimo-v2.5-pro",
        api_key=MIMO_KEY,
        base_url=MIMO_URL,
    )

    # 4. Run — model generates response, then LLM judge scores it
    results = await te.Evaluation.run_async(
        ds, grader,
        env=env,
        model="mimo-v2.5-pro",
        api_key=MIMO_KEY,
        base_url=MIMO_URL,
        workers=3,
    )

    # 5. Results
    s = results.summary()
    print("Model:", s["model"])
    print("Samples:", s["num_runs"])
    print("Avg Reward:", s["avg_reward"])
    print("Pass Rate:", s["pass_rate"])
    print()

    for i, run in enumerate(results.runs):
        sample = ds[i]
        resp = run.get("completion", [{}])[-1].get("content", "")
        reward = run.get("reward", 0)
        status = "PASS" if reward >= 0.7 else "FAIL"

        print(f"Q{i+1}: {sample.input[:55]}...")
        print(f"   Response: {resp.strip()[:70]}...")
        print(f"   Rubrics: {len(sample.rubrics)} graded by LLM")
        print(f"   Score: {reward:.2f} [{status}]")
        print()

asyncio.run(test())
