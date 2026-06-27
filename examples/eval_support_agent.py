"""
Customer Support Triage Evaluation
===================================
Evaluates a GPT-4o-mini endpoint on ticket triage scenarios.

Usage:
    python eval_support_agent.py --api-key sk-...

What it does:
    1. Creates 10 support ticket scenarios
    2. Calls your GPT-4o-mini endpoint for each
    3. Scores responses on triage quality
    4. Shows results table
"""

import sys
import asyncio
import argparse
sys.path.insert(0, ".")

import tensoreval as te


# ============================================================
# 1. Triage test cases
# ============================================================

TRIAGE_CASES = [
    {
        "query": "Ticket #1001: Customer reports being charged twice for their Pro subscription ($29/mo). They want the duplicate removed immediately.",
        "reference_answer": "Escalate to billing, remove duplicate charge of $29",
        "rubrics": [
            {"name": "triage_category", "rubric": "Must categorize as billing/payment issue", "weight": 0.3},
            {"name": "urgency", "rubric": "Must recognize this as urgent (duplicate charge)", "weight": 0.3},
            {"name": "action", "rubric": "Must specify concrete next steps (remove charge, confirm)", "weight": 0.4},
        ],
    },
    {
        "query": "Ticket #1002: Customer says the app crashes every time they open the Settings page on iPhone 15, iOS 17.2.",
        "reference_answer": "Categorize as bug report, escalate to engineering with device details",
        "rubrics": [
            {"name": "triage_category", "rubric": "Must categorize as bug/technical issue", "weight": 0.3},
            {"name": "reproduction_info", "rubric": "Must note device model and OS version", "weight": 0.3},
            {"name": "action", "rubric": "Must suggest escalation to engineering or bug tracker", "weight": 0.4},
        ],
    },
    {
        "query": "Ticket #1003: Enterprise customer wants to know if SSO (SAML) is supported and how to set it up.",
        "reference_answer": "Confirm SSO is Enterprise feature, provide setup docs link",
        "rubrics": [
            {"name": "triage_category", "rubric": "Must categorize as feature inquiry or integration request", "weight": 0.3},
            {"name": "knowledge", "rubric": "Must know SSO is Enterprise-tier feature", "weight": 0.4},
            {"name": "action", "rubric": "Must provide next steps (docs link, setup guide, or sales contact)", "weight": 0.3},
        ],
    },
    {
        "query": "Ticket #1004: Customer wants to cancel their subscription. They are on the Pro plan ($79/mo) with 6 months remaining on annual billing.",
        "reference_answer": "Process cancellation request, offer retention incentive",
        "rubrics": [
            {"name": "triage_category", "rubric": "Must categorize as cancellation/churn risk", "weight": 0.3},
            {"name": "retention", "rubric": "Should offer retention incentive (discount, downgrade option)", "weight": 0.4},
            {"name": "professionalism", "rubric": "Must be professional, not pushy or aggressive", "weight": 0.3},
        ],
    },
    {
        "query": "Ticket #1005: Customer reports their API key was exposed in a public GitHub repo. They need it rotated immediately.",
        "reference_answer": "Immediately revoke exposed key, generate new key, flag security incident",
        "rubrics": [
            {"name": "triage_category", "rubric": "Must categorize as security incident", "weight": 0.3},
            {"name": "urgency", "rubric": "Must treat as critical/immediate", "weight": 0.4},
            {"name": "action", "rubric": "Must revoke old key and generate new one", "weight": 0.3},
        ],
    },
    {
        "query": "Ticket #1006: Customer on Free plan is hitting API rate limits (100 requests/hour). They have a demo with a potential client tomorrow and need a temporary increase.",
        "reference_answer": "Grant temporary rate limit increase, suggest upgrading to Pro",
        "rubrics": [
            {"name": "triage_category", "rubric": "Must categorize as rate limit / capacity request", "weight": 0.3},
            {"name": "empathy", "rubric": "Must understand the urgency of the demo", "weight": 0.3},
            {"name": "action", "rubric": "Must provide temporary increase AND suggest Pro upgrade", "weight": 0.4},
        ],
    },
    {
        "query": "Ticket #1007: Customer is asking for a refund on an order delivered 45 days ago. Company policy is 30-day refund window.",
        "reference_answer": "Politely refuse refund citing 30-day policy, offer alternative",
        "rubrics": [
            {"name": "triage_category", "rubric": "Must categorize as refund request", "weight": 0.2},
            {"name": "policy_compliance", "rubric": "Must refuse citing 30-day window", "weight": 0.5},
            {"name": "empathy", "rubric": "Must be understanding, offer alternative (store credit, exchange)", "weight": 0.3},
        ],
    },
    {
        "query": "Ticket #1008: Customer reports data loss after the latest system update. Critical business data is missing from their dashboard.",
        "reference_answer": "Escalate immediately to engineering, initiate data recovery",
        "rubrics": [
            {"name": "triage_category", "rubric": "Must categorize as critical data loss incident", "weight": 0.3},
            {"name": "urgency", "rubric": "Must treat as highest priority", "weight": 0.4},
            {"name": "action", "rubric": "Must escalate to engineering and initiate recovery process", "weight": 0.3},
        ],
    },
    {
        "query": "Ticket #1009: Customer wants to upgrade from Starter ($29/mo) to Enterprise ($199/mo) mid-cycle. They want to know about prorated billing.",
        "reference_answer": "Explain prorated billing for mid-cycle upgrade, confirm data preserved",
        "rubrics": [
            {"name": "triage_category", "rubric": "Must categorize as upgrade/billing inquiry", "weight": 0.2},
            {"name": "billing_knowledge", "rubric": "Must explain prorated billing correctly", "weight": 0.4},
            {"name": "data_preservation", "rubric": "Must confirm their data will be preserved during upgrade", "weight": 0.4},
        ],
    },
    {
        "query": "Ticket #1010: Customer says the search feature returns wrong results when searching for contacts with special characters in their names (e.g., O'Brien, Müller).",
        "reference_answer": "Categorize as bug, note it's a character encoding issue, escalate to engineering",
        "rubrics": [
            {"name": "triage_category", "rubric": "Must categorize as bug/technical issue", "weight": 0.3},
            {"name": "diagnosis", "rubric": "Must identify this as a character encoding/search issue", "weight": 0.4},
            {"name": "action", "rubric": "Must escalate to engineering with reproduction steps", "weight": 0.3},
        ],
    },
]


# ============================================================
# 2. Reward function
# ============================================================

def triage_reward(state, **kwargs) -> float:
    """Score based on triage quality."""
    completion = state.get("completion", [])
    answer = state.get("answer", "")
    rubrics = state.get("info", {}).get("rubrics", [])

    if not completion:
        return 0.0

    last = completion[-1]
    response = last.get("content", "") if isinstance(last, dict) else str(getattr(last, "content", ""))
    response_lower = response.lower()

    score = 0.0

    # Check for category identification
    category_words = ["billing", "bug", "technical", "security", "cancellation",
                      "refund", "upgrade", "rate limit", "data loss", "incident"]
    if any(w in response_lower for w in category_words):
        score += 0.25

    # Check for urgency recognition
    urgency_words = ["urgent", "immediate", "critical", "asap", "priority", "escalat"]
    if any(w in response_lower for w in urgency_words):
        score += 0.2

    # Check for action steps
    action_words = ["will", "should", "recommend", "suggest", "escalat", "contact", "provide", "issue"]
    if any(w in response_lower for w in action_words):
        score += 0.25

    # Check for empathy/professionalism
    empathy_words = ["understand", "apologize", "sorry", "appreciate", "help", "assist"]
    if any(w in response_lower for w in empathy_words):
        score += 0.15

    # Check for reasonable length (not too short, not too long)
    if 50 < len(response) < 2000:
        score += 0.15

    return min(score, 1.0)


# ============================================================
# 3. Main evaluation
# ============================================================

async def run_evaluation(api_key: str, base_url: str, model: str):
    """Run the evaluation."""
    print()
    print("=" * 60)
    print("Customer Support Triage Evaluation")
    print("=" * 60)
    print()

    # Load dataset
    datasets = te.Datasets.from_dicts(TRIAGE_CASES, name="support_triage")
    print(f"Loaded {len(datasets)} triage scenarios")

    # Create grader
    grader = te.Grader(funcs=[triage_reward], weights=[1.0])

    # Create environment
    env = te.SingleTurnEnv(
        rubric=grader,
        system_prompt="""You are a customer support triage agent for a SaaS company.

Your job is to:
1. Categorize the ticket (billing, bug, security, cancellation, etc.)
2. Assess urgency (critical, high, medium, low)
3. Suggest concrete next steps
4. Be professional and empathetic

Products:
- Starter: $29/mo (basic features, 100 API calls/hr)
- Pro: $79/mo (advanced features, 1000 API calls/hr)
- Enterprise: $199/mo (unlimited, SSO, dedicated support)

Policies:
- 30-day refund window
- Annual plans: 20% discount, non-refundable after 30 days
- API rate limits enforced per plan tier
- Security incidents: immediate key rotation""",
    )

    # Run evaluation
    print(f"Model: {model}")
    print(f"Endpoint: {base_url}")
    print(f"Scenarios: {len(datasets)}")
    print()

    results = await te.Evaluation.run_async(
        datasets=datasets,
        grader=grader,
        env=env,
        model=model,
        api_key=api_key,
        base_url=base_url,
        workers=5,
    )

    # Show results
    summary = results.summary()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Model:      {summary['model']}")
    print(f"  Scenarios:  {summary['num_runs']}")
    print(f"  Avg Reward: {summary['avg_reward']:.4f}")
    print(f"  Pass Rate:  {summary['pass_rate']:.1%}")
    print(f"  Passed:     {summary['pass_count']}")
    print(f"  Failed:     {summary['fail_count']}")
    print()

    print("Per-ticket breakdown:")
    print("-" * 60)
    for i, run in enumerate(results.runs):
        sample = datasets[i]
        completion = run.get("completion", [])
        response = ""
        if completion:
            last = completion[-1]
            response = last.get("content", "") if isinstance(last, dict) else str(getattr(last, "content", ""))
        reward = run.get("reward", 0)
        status = "PASS" if reward >= 0.6 else "FAIL"
        ticket_id = sample.metadata.get("id", f"T{i+1}") if sample.metadata else f"T{i+1}"

        print(f"  {ticket_id}: {sample.input[:50]}...")
        print(f"       Response: {response.strip()[:80]}...")
        print(f"       Score: {reward:.2f} [{status}]")
        print()

    return results


# ============================================================
# 4. CLI entry point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a customer support triage agent")
    parser.add_argument("--api-key", required=True, help="OpenAI API key")
    parser.add_argument("--base-url", default="https://api.openai.com/v1", help="API base URL")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model name")

    args = parser.parse_args()

    results = asyncio.run(run_evaluation(args.api_key, args.base_url, args.model))
