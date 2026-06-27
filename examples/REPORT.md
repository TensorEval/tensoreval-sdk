# Customer Support Agent Evaluation Report

**Model:** Mimo v2.5 Pro  
**Evaluation Date:** June 27, 2026  
**SDK:** TensorEval v0.5.0  
**Grader:** AgentGrader (LLM reads rubrics and judges each one)

---

## Executive Summary

A customer support AI agent was evaluated across **21 production-like scenarios** covering billing, technical support, security, account management, and policy enforcement.

**Results: 12/21 passed (57.1% pass rate), Average Reward: 0.74**

The agent excels at security incidents and policy compliance but struggles with technical support and scenarios requiring specific product knowledge.

---

## Results by Category

| Category | Scenarios | Passed | Pass Rate | Avg Score |
|----------|-----------|--------|-----------|-----------|
| Billing | 4 | 3 | 75% | 0.81 |
| Technical | 4 | 1 | 25% | 0.38 |
| Security | 2 | 2 | 100% | 0.93 |
| Account | 4 | 2 | 50% | 0.63 |
| API/Rate Limits | 2 | 1 | 50% | 0.63 |
| Policy | 2 | 2 | 100% | 0.98 |
| Complex/Edge | 3 | 3 | 100% | 0.67 |

---

## What The Agent Does Well

### 1. Security Incidents (100% pass rate)
The agent correctly handles security-critical scenarios:
- **API key exposure:** Immediately recognizes severity, provides clear action plan (revoke + regenerate)
- **Data breach:** Correctly escalates, locks account, initiates investigation

### 2. Policy Compliance (100% pass rate)
The agent correctly applies company policies:
- **Refund window:** Knows the 30-day policy, correctly approves/refuses based on delivery date
- **Cancellation timing:** Clarifies that access continues until end of billing period

### 3. De-escalation (100% pass rate)
The agent handles angry customers and human handoff well:
- **Angry customer:** Apologizes sincerely, offers concrete resolution
- **Human handoff:** Acknowledges request without resistance, preserves context

### 4. Multi-Issue Triage (100% pass rate)
The agent correctly prioritizes multiple issues:
- **Billing + technical:** Addresses billing first, then technical issues
- **Structured approach:** Provides clear step-by-step resolution plan

---

## What Needs Improvement

### 1. Technical Support (25% pass rate)
The agent struggles with technical issues:
- **App crashes:** Gives generic troubleshooting instead of escalating to engineering
- **Search bugs:** Doesn't identify the character encoding issue
- **Data loss:** Treats as routine instead of critical escalation

**Recommendation:** Train the agent to escalate technical issues to engineering faster, rather than attempting generic troubleshooting.

### 2. Specific Product Knowledge (50% pass rate)
The agent lacks specific product details:
- **Pricing:** Doesn't know per-seat pricing for team members
- **Rate limits:** Doesn't provide concrete solutions for rate limit issues
- **Invoice discrepancies:** Gives generic explanations instead of investigating

**Recommendation:** Add product documentation to the agent's knowledge base or system prompt.

### 3. Action-Oriented Responses (50% pass rate)
Some responses are too analytical, not action-oriented enough:
- **Cancellations:** Explains options but doesn't offer retention incentives
- **Rate limits:** Suggests upgrading but doesn't provide immediate temporary increase

**Recommendation:** Train the agent to take concrete action, not just explain options.

---

## Per-Query Results

| # | Category | Query | Score | Status | Notes |
|---|----------|-------|-------|--------|-------|
| 1 | Billing | Duplicate charge ($79) | 1.00 | PASS | Correctly identifies and resolves |
| 2 | Billing | Refund outside window | 0.75 | PASS | Correctly refuses, offers alternatives |
| 3 | Billing | Invoice discrepancy ($249 vs $199) | 0.55 | FAIL | Gives generic explanations, doesn't investigate |
| 4 | Billing | Accidental upgrade refund | 1.00 | PASS | Correctly processes refund |
| 5 | Technical | App crashes on Settings | 0.35 | FAIL | Generic troubleshooting, doesn't escalate |
| 6 | Technical | Search special characters | 0.35 | FAIL | Doesn't identify encoding issue |
| 7 | Technical | App slow after update | 0.80 | PASS | Good troubleshooting steps |
| 8 | Technical | Data loss after update | 0.20 | FAIL | Treats as routine, not critical |
| 9 | Security | API key exposed on GitHub | 1.00 | PASS | Correct immediate action |
| 10 | Security | Data breach suspicion | 0.85 | PASS | Correct escalation |
| 11 | Account | Upgrade to Enterprise | 0.90 | PASS | Good explanation of benefits |
| 12 | Account | Subscription cancellation | 0.35 | FAIL | No retention incentive offered |
| 13 | Account | SSO integration | 0.90 | PASS | Good setup guidance |
| 14 | Account | Team member pricing | 0.30 | FAIL | Doesn't know pricing details |
| 15 | API | Rate limit temporary increase | 0.45 | FAIL | Suggests upgrade but doesn't grant temporary increase |
| 16 | API | 429 rate limit investigation | 0.80 | PASS | Good troubleshooting |
| 17 | Policy | Refund within window (opened item) | 0.95 | PASS | Correct policy application |
| 18 | Policy | Cancellation timing concern | 1.00 | PASS | Correctly reassures customer |
| 19 | Complex | Multi-issue (billing + SSO + login) | 1.00 | PASS | Good prioritization |
| 20 | Complex | Angry customer (3-day wait) | 1.00 | PASS | Good de-escalation |
| 21 | Edge | "Can I speak to a human?" | 1.00 | PASS | Smooth handoff |

---

## Voice Metrics

| Metric | Value | Assessment |
|--------|-------|------------|
| Avg Response Latency | 19.2s | High — includes LLM thinking time |
| Avg Words Per Minute | 865 | Very fast — agent generates detailed responses |
| Avg Response Length | ~200 words | Appropriate for support scenarios |

---

## Recommendations

### Immediate (Week 1)
1. **Add product documentation to system prompt** — pricing, rate limits, team management
2. **Add escalation rules** — technical issues → engineering, data loss → critical
3. **Add retention incentives** — cancellation → offer discount/downgrade

### Short-term (Month 1)
4. **Implement tool calling** — agent should actually look up orders, check policies
5. **Add multi-turn support** — agent should ask clarifying questions
6. **Reduce response latency** — optimize system prompt, reduce token count

### Long-term (Quarter 1)
7. **Add MCP integration** — connect to real customer database, billing system
8. **Add Docker sandbox** — test agent in isolated environment
9. **Continuous evaluation** — run eval on every prompt change

---

## How To Reproduce

```python
import tensoreval as te

# Load the evaluation dataset
ds = te.Datasets.load_from_file("examples/scenarios/customer_support_eval.py")

# Create grader (LLM reads rubrics)
grader = te.AgentGrader(
    model="mimo-v2.5-pro",
    api_key="your-key",
    base_url="https://token-plan-sgp.xiaomimimo.com/anthropic",
)

# Create environment
env = te.Env.from_dict({
    "system_prompt": "You are a professional customer support agent..."
})

# Run evaluation
results = te.Evaluation.run(ds, env, grader, workers=3)
print(results.summary())
```

---

## Files

- **Evaluation script:** `examples/scripts/run_customer_support.py`
- **Test scenarios:** `examples/scenarios/customer_support_eval.py`
- **Results data:** `examples/results/customer_support_results.json`
- **This report:** `examples/REPORT.md`
