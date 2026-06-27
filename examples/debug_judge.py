"""Debug AgentGrader judge call."""
import sys, asyncio
sys.path.insert(0, ".")
from tensoreval.graders.agent_grader import AgentGrader

async def test():
    grader = AgentGrader(
        model="mimo-v2.5-pro",
        api_key="tp-sjgd0v2ckg08hbi46e8uvkfmonq4ywqvxv4dsa27qz6xk5zg",
        base_url="https://token-plan-sgp.xiaomimimo.com/anthropic",
    )

    prompt = """Score this response against rubrics.

Query: Customer wants refund, delivered 10 days ago, total $49.99
Response: I will process your refund of $49.99 for order #12345.

Rubrics:
1. policy: Must verify within 30-day window
2. amount: Must mention exact amount $49.99
3. empathy: Must show empathy

Output JSON: {"scores": [{"name": "policy", "score": 0.9, "reason": "ok"}]}"""

    try:
        content = await grader._call_anthropic(prompt)
        print("Raw judge output:")
        print(content[:500])
        print()
        scores = grader._parse_scores(content)
        print("Parsed scores:", scores)
    except Exception as e:
        print("Error:", type(e).__name__, str(e)[:300])

asyncio.run(test())
