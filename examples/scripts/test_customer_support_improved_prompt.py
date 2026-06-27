"""Test customer support with better system prompt."""

import sys
import asyncio
sys.path.insert(0, ".")
import tensoreval as te


async def test():
    # Use the customer support eval scenarios
    from examples.scenarios.customer_support_eval import CUSTOMER_SUPPORT_SCENARIOS

    datasets = te.Datasets.load_from_dict(CUSTOMER_SUPPORT_SCENARIOS)

    def cs_reward(state, **kwargs):
        completion = state.get("completion", [])
        answer = state.get("answer", "")
        if not completion:
            return 0.0
        last = completion[-1]
        response = last.get("content", "") if isinstance(last, dict) else str(getattr(last, "content", ""))
        r = response.lower()

        score = 0.0
        # Check if response addresses the expected answer
        if answer:
            answer_lower = answer.lower()
            key_phrases = [w.strip() for w in answer_lower.split() if len(w.strip()) > 3]
            matches = sum(1 for w in key_phrases if w in r)
            if key_phrases:
                score += min(matches / len(key_phrases), 1.0) * 0.5

        # Action words
        action_words = ["will", "have", "processed", "issued", "applied", "credited", "removed", "escalated", "refunded"]
        if any(w in r for w in action_words):
            score += 0.2

        # Empathy
        empathy = ["understand", "apologize", "sorry", "appreciate"]
        if any(w in r for w in empathy):
            score += 0.1

        # Length check
        if 50 < len(response) < 2000:
            score += 0.1

        return min(score, 1.0)

    grader = te.AgentGrader(
        model="mimo-v2.5-pro",
        api_key="tp-sjgd0v2ckg08hbi46e8uvkfmonq4ywqvxv4dsa27qz6xk5zg",
        base_url="https://token-plan-sgp.xiaomimimo.com/anthropic",
    )

    # Better system prompt with product knowledge
    env = te.Env.from_dict({
        "system_prompt": """You are a professional customer support agent for a SaaS platform.

PRODUCTS & PRICING:
- Free: $0/mo, 100 API calls/hr
- Starter: $29/mo, 1000 API calls/hr
- Pro: $79/mo, advanced features, 10000 API calls/hr
- Enterprise: $199/mo, unlimited API, SSO, dedicated support
- Team members: $10/seat/month
- Annual billing: 17% discount

POLICIES:
- 30-day refund window from delivery date
- Duplicate charges: refund immediately + $10 credit
- Rate limit increase: grant 24hr temporary increase for demos/urgent needs
- Cancellation: access continues until end of billing period
- Data loss: escalate to engineering IMMEDIATELY as critical
- Security incidents: revoke keys immediately, escalate to security team

YOUR ROLE:
- Take action immediately. Don't ask unnecessary questions.
- For technical issues: escalate to engineering with device/OS details.
- For data loss or security: treat as CRITICAL, escalate immediately.
- For billing issues: verify then act (refund or explain policy).
- Be empathetic but efficient. State what you did.""",
    })

    results = await te.Evaluation.run_async(
        datasets=datasets, grader=grader, env=env,
        model="mimo-v2.5-pro",
        api_key="tp-sjgd0v2ckg08hbi46e8uvkfmonq4ywqvxv4dsa27qz6xk5zg",
        base_url="https://token-plan-sgp.xiaomimimo.com/anthropic",
        workers=5,
    )

    s = results.summary()
    print()
    print("Results with improved system prompt:")
    print("  Avg Reward: " + str(s["avg_reward"]))
    print("  Pass Rate:  " + str(s["pass_rate"]))
    print("  Passed:     " + str(s["pass_count"]))
    print("  Failed:     " + str(s["fail_count"]))
    print()
    for i, run in enumerate(results.runs):
        sample = datasets[i]
        completion = run.get("completion", [])
        response = completion[-1].get("content", "") if completion else ""
        reward = run.get("reward", 0)
        status = "PASS" if reward >= 0.6 else "FAIL"
        resp_short = response.encode("ascii", "replace").decode()[:70]
        print("  " + status + " S" + str(i+1) + ": " + str(round(reward, 2)) + " | " + sample.input[:45] + "...")
        print("       -> " + resp_short + "...")


if __name__ == "__main__":
    asyncio.run(test())
