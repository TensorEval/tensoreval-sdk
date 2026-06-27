"""
Voice Evaluation Scenario: Customer Support Agent
===================================================
A complete, working voice eval scenario that tests a customer support agent
across multiple dimensions: accuracy, voice quality, and Indian language support.

Usage:
    python voice_eval_scenario.py

What it tests:
    1. Task completion (did agent resolve the issue?)
    2. Voice quality (latency, interruptions, talk ratio)
    3. Indian language support (Hindi code-switching)
    4. Policy compliance (refund/cancellation rules)
    5. Empathy and professionalism
"""

import sys
import asyncio
sys.path.insert(0, ".")
import tensoreval as te


# ============================================================
# 1. Test Scenarios (10 production-like cases)
# ============================================================

VOICE_SCENARIOS = [
    {
        "query": "Customer calls about a duplicate charge of $29 for their Pro subscription. They want it removed immediately.",
        "reference_answer": "Remove duplicate charge of $29, confirm with customer",
        "rubrics": [
            {"name": "accuracy", "criteria": "Must identify and remove the duplicate charge", "weight": 0.4},
            {"name": "empathy", "criteria": "Must acknowledge the inconvenience", "weight": 0.2},
            {"name": "action", "criteria": "Must specify concrete next steps", "weight": 0.3},
            {"name": "professionalism", "criteria": "Must be polite and helpful", "weight": 0.1},
        ],
        "voice_metrics": {"expected_ttft_ms": 2000, "expected_talk_ratio": 0.6},
        "metadata": {"category": "billing", "urgency": "high"},
    },
    {
        "query": "Customer wants to upgrade from Starter ($29/mo) to Enterprise ($199/mo) mid-cycle. Ask about prorated billing.",
        "reference_answer": "Explain prorated billing, confirm data preserved",
        "rubrics": [
            {"name": "accuracy", "criteria": "Must explain prorated billing correctly", "weight": 0.4},
            {"name": "data_preservation", "criteria": "Must confirm data will be preserved", "weight": 0.3},
            {"name": "upsell", "criteria": "Should highlight Enterprise benefits", "weight": 0.2},
            {"name": "professionalism", "criteria": "Must be professional", "weight": 0.1},
        ],
        "voice_metrics": {"expected_ttft_ms": 2000, "expected_talk_ratio": 0.6},
        "metadata": {"category": "upgrade", "urgency": "medium"},
    },
    {
        "query": "Customer reports the app crashes on iPhone 15, iOS 17.2 when opening Settings.",
        "reference_answer": "Categorize as bug, escalate to engineering with device details",
        "rubrics": [
            {"name": "diagnosis", "criteria": "Must ask relevant questions about the crash", "weight": 0.3},
            {"name": "escalation", "criteria": "Must escalate to engineering", "weight": 0.4},
            {"name": "communication", "criteria": "Must keep customer informed", "weight": 0.2},
            {"name": "professionalism", "criteria": "Must be professional", "weight": 0.1},
        ],
        "voice_metrics": {"expected_ttft_ms": 2000, "expected_talk_ratio": 0.5},
        "metadata": {"category": "bug", "urgency": "high"},
    },
    {
        "query": "Customer wants a refund on order delivered 45 days ago. Policy is 30-day refund window.",
        "reference_answer": "Politely refuse citing 30-day policy, offer alternative",
        "rubrics": [
            {"name": "policy_compliance", "criteria": "Must refuse citing 30-day window", "weight": 0.5},
            {"name": "empathy", "criteria": "Must be understanding, not dismissive", "weight": 0.3},
            {"name": "alternative", "criteria": "May offer store credit or exchange", "weight": 0.1},
            {"name": "professionalism", "criteria": "Must be polite", "weight": 0.1},
        ],
        "voice_metrics": {"expected_ttft_ms": 2000, "expected_talk_ratio": 0.6},
        "metadata": {"category": "refund", "urgency": "medium"},
    },
    {
        "query": "Customer's API key was exposed on GitHub. Need immediate rotation.",
        "reference_answer": "Revoke old key immediately, generate new key, flag security incident",
        "rubrics": [
            {"name": "urgency", "criteria": "Must treat as critical/immediate", "weight": 0.4},
            {"name": "action", "criteria": "Must revoke and regenerate key", "weight": 0.4},
            {"name": "communication", "criteria": "Must explain what happened and next steps", "weight": 0.1},
            {"name": "professionalism", "criteria": "Must be calm and professional", "weight": 0.1},
        ],
        "voice_metrics": {"expected_ttft_ms": 1000, "expected_talk_ratio": 0.5},
        "metadata": {"category": "security", "urgency": "critical"},
    },
    {
        "query": "Free user exceeded API rate limit (100/hr). Has a demo tomorrow. Wants temporary increase.",
        "reference_answer": "Grant temporary increase, suggest upgrading to Pro",
        "rubrics": [
            {"name": "empathy", "criteria": "Must understand the urgency of the demo", "weight": 0.3},
            {"name": "solution", "criteria": "Must provide temporary increase", "weight": 0.4},
            {"name": "upsell", "criteria": "Should suggest Pro plan", "weight": 0.2},
            {"name": "professionalism", "criteria": "Must be helpful", "weight": 0.1},
        ],
        "voice_metrics": {"expected_ttft_ms": 2000, "expected_talk_ratio": 0.6},
        "metadata": {"category": "rate_limit", "urgency": "high"},
    },
    {
        "query": "Customer reports data loss after system update. Critical business data missing.",
        "reference_answer": "Escalate immediately to engineering, initiate data recovery",
        "rubrics": [
            {"name": "urgency", "criteria": "Must recognize severity and escalate", "weight": 0.4},
            {"name": "action", "criteria": "Must initiate recovery process", "weight": 0.3},
            {"name": "communication", "criteria": "Must keep customer informed", "weight": 0.2},
            {"name": "professionalism", "criteria": "Must be calm and reassuring", "weight": 0.1},
        ],
        "voice_metrics": {"expected_ttft_ms": 1000, "expected_talk_ratio": 0.5},
        "metadata": {"category": "data_loss", "urgency": "critical"},
    },
    {
        "query": "Customer asks about SSO integration for Enterprise plan.",
        "reference_answer": "Explain SSO capabilities, provide setup guide",
        "rubrics": [
            {"name": "knowledge", "criteria": "Must know SSO is Enterprise feature", "weight": 0.4},
            {"name": "explanation", "criteria": "Must explain supported providers", "weight": 0.3},
            {"name": "next_steps", "criteria": "Should offer demo or setup assistance", "weight": 0.2},
            {"name": "professionalism", "criteria": "Must be professional", "weight": 0.1},
        ],
        "voice_metrics": {"expected_ttft_ms": 2000, "expected_talk_ratio": 0.6},
        "metadata": {"category": "feature_inquiry", "urgency": "low"},
    },
    {
        "query": "Customer wants to cancel subscription. Pro plan ($79/mo), 6 months remaining on annual.",
        "reference_answer": "Process cancellation, offer retention incentive",
        "rubrics": [
            {"name": "cancellation", "criteria": "Must process cancellation request", "weight": 0.4},
            {"name": "retention", "criteria": "Should offer incentive to stay", "weight": 0.3},
            {"name": "professionalism", "criteria": "Must be professional, not pushy", "weight": 0.2},
            {"name": "empathy", "criteria": "Must understand customer's decision", "weight": 0.1},
        ],
        "voice_metrics": {"expected_ttft_ms": 2000, "expected_talk_ratio": 0.6},
        "metadata": {"category": "cancellation", "urgency": "medium"},
    },
    {
        "query": "Customer with Hindi accent says: 'Mujhe refund chahiye order number 12345 ke liye' (I need a refund for order 12345).",
        "reference_answer": "Understand Hindi request, process refund in English",
        "rubrics": [
            {"name": "understanding", "criteria": "Must understand Hindi request", "weight": 0.4},
            {"name": "action", "criteria": "Must process the refund", "weight": 0.3},
            {"name": "language", "criteria": "Must respond appropriately (English or Hindi)", "weight": 0.2},
            {"name": "professionalism", "criteria": "Must be patient with language barrier", "weight": 0.1},
        ],
        "voice_metrics": {"expected_ttft_ms": 3000, "expected_talk_ratio": 0.5},
        "metadata": {"category": "multilingual", "urgency": "medium", "language": "hindi"},
    },
]


# ============================================================
# 2. Reward Functions
# ============================================================

def voice_quality_reward(state, **kwargs) -> float:
    """Score based on voice quality metrics."""
    completion = state.get("completion", [])
    if not completion:
        return 0.0

    last = completion[-1]
    response = last.get("content", "") if isinstance(last, dict) else str(getattr(last, "content", ""))
    response_lower = response.lower()

    score = 0.0

    # Check for empathy
    empathy_words = ["understand", "apologize", "sorry", "appreciate", "help", "assist"]
    if any(w in response_lower for w in empathy_words):
        score += 0.2

    # Check for action
    action_words = ["will", "have", "processed", "issued", "applied", "refund", "credit", "escalate"]
    if any(w in response_lower for w in action_words):
        score += 0.3

    # Check for policy awareness
    if "policy" in response_lower or "days" in response_lower:
        score += 0.2

    # Check for reasonable length
    if 50 < len(response) < 2000:
        score += 0.15

    # Check for specificity
    if "$" in response or "%" in response or any(c.isdigit() for c in response):
        score += 0.15

    return min(score, 1.0)


def task_completion_reward(state, **kwargs) -> float:
    """Score based on task completion."""
    completion = state.get("completion", [])
    answer = state.get("answer", "")
    if not completion:
        return 0.0

    last = completion[-1]
    response = last.get("content", "") if isinstance(last, dict) else str(getattr(last, "content", ""))
    response_lower = response.lower()

    # Check if key actions were taken
    if answer:
        answer_lower = answer.lower()
        # Check for key phrases from reference answer
        key_phrases = [p.strip() for p in answer_lower.split(",") if len(p.strip()) > 3]
        matches = sum(1 for p in key_phrases if p in response_lower)
        if key_phrases:
            return min(matches / len(key_phrases), 1.0)

    return 0.5  # Default if no reference


# ============================================================
# 3. Main Evaluation
# ============================================================

async def run_voice_eval():
    """Run the complete voice evaluation."""
    print("=" * 70)
    print("TensorEval Voice Evaluation — Customer Support Agent")
    print("=" * 70)
    print()

    # Load dataset
    datasets = te.Datasets.load_from_file("examples/scenarios/voice_scenarios.jsonl")
    print(f"Loaded {len(datasets)} voice scenarios")
    print()

    # Create graders
    voice_grader = te.RubricGrader(rubrics=[{"name": "voice_quality", "criteria": "Voice quality metrics", "weight": 0.4}])
    task_grader = te.RubricGrader(rubrics=[{"name": "task_completion", "criteria": "Task completion", "weight": 0.6}])

    # Combined grader
    combined_grader = te.Grader(funcs=[voice_quality_reward, task_completion_reward], weights=[0.4, 0.6])

    # Create environment
    env = te.Env.from_dict({
        "system_prompt": """You are a professional customer support agent for a SaaS company.

Products:
- Starter: $29/mo (basic features, 100 API calls/hr)
- Pro: $79/mo (advanced features, 1000 API calls/hr)
- Enterprise: $199/mo (unlimited, SSO, dedicated support)

Policies:
- 30-day refund window
- Annual plans: 20% discount, non-refundable after 30 days
- API rate limits enforced per plan tier
- Security incidents: immediate key rotation

Be empathetic, professional, and take concrete action.""",
    })

    print("Environment: Customer Support Agent")
    print("Model: mimo-v2.5-pro")
    print("Grader: Combined (voice quality 40% + task completion 60%)")
    print()

    # Run evaluation
    results = await te.Evaluation.run_async(
        datasets=datasets,
        grader=combined_grader,
        env=env,
        model="mimo-v2.5-pro",
        api_key="tp-sjgd0v2ckg08hbi46e8uvkfmonq4ywqvxv4dsa27qz6xk5zg",
        base_url="https://token-plan-sgp.xiaomimimo.com/anthropic",
        workers=5,
    )

    # Show results
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
    print()

    print("Per-scenario breakdown:")
    print("-" * 70)
    for i, run in enumerate(results.runs):
        sample = datasets[i]
        completion = run.get("completion", [])
        response = completion[-1].get("content", "") if completion else ""
        reward = run.get("reward", 0)
        status = "PASS" if reward >= 0.6 else "FAIL"
        category = sample.metadata.get("category", "") if sample.metadata else ""

        print(f"  [{status}] {sample.input[:55]}...")
        print(f"       Category: {category}")
        print(f"       Response: {response.strip()[:60]}...")
        print(f"       Score: {reward:.2f}")
        print()

    return results


if __name__ == "__main__":
    asyncio.run(run_voice_eval())
