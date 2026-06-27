"""Customer Support Agent Evaluation — 20 Production Scenarios.

Tests a customer support agent across:
- Billing issues (duplicates, refunds, charges)
- Technical support (bugs, crashes, performance)
- Security incidents (leaked keys, unauthorized access)
- Account management (upgrades, cancellations, SSO)
- Policy enforcement (refund windows, rate limits)

Each query has specific rubrics graded by LLM judge.
"""

# ============================================================
# 1. MCP Tools — Simulated customer database
# ============================================================

# Mock customer database
CUSTOMER_DB = {
    "C001": {"name": "Alice Johnson", "plan": "Pro", "monthly": 79, "joined": "2024-03-15"},
    "C002": {"name": "Bob Smith", "plan": "Starter", "monthly": 29, "joined": "2023-11-01"},
    "C003": {"name": "Carol Davis", "plan": "Enterprise", "monthly": 199, "joined": "2024-01-10"},
    "C004": {"name": "Dan Wilson", "plan": "Pro", "monthly": 79, "joined": "2024-06-20"},
    "C005": {"name": "Eve Martinez", "plan": "Free", "monthly": 0, "joined": "2024-08-01"},
}

ORDER_DB = {
    "O7841": {"customer": "C001", "amount": 4999, "delivered": "2024-12-20", "status": "delivered"},
    "O7842": {"customer": "C002", "amount": 12000, "delivered": "2023-06-01", "status": "delivered"},
    "O7843": {"customer": "C003", "amount": 2999, "delivered": "2024-12-28", "status": "delivered"},
    "O7844": {"customer": "C004", "amount": 8500, "delivered": "2024-11-15", "status": "returned"},
    "O7845": {"customer": "C005", "amount": 0, "delivered": None, "status": "pending"},
}

POLICY_DB = {
    "refund_window_days": 30,
    "max_refund_without_approval_cents": 5000,
    "rate_limit_free": 100,
    "rate_limit_pro": 1000,
    "rate_limit_enterprise": -1,
    "cancellation_notice_days": 0,
}


# ============================================================
# 2. Test Scenarios — 20 production-like cases
# ============================================================

CUSTOMER_SUPPORT_SCENARIOS = [
    # --- Billing Issues ---
    {
        "query": "Customer C001 reports being charged twice for their Pro subscription ($79/mo) this month. They want the duplicate removed immediately.",
        "reference_answer": "Remove duplicate $79 charge, confirm with customer",
        "rubrics": [
            {"name": "billing_accuracy", "criteria": "Must identify and remove the duplicate $79 charge", "weight": 0.4},
            {"name": "empathy", "criteria": "Must acknowledge the inconvenience of being double-charged", "weight": 0.2},
            {"name": "action", "criteria": "Must specify concrete next steps (refund timeline, confirmation)", "weight": 0.3},
            {"name": "professionalism", "criteria": "Must be polite and helpful throughout", "weight": 0.1},
        ],
        "category": "billing",
        "difficulty": "easy",
    },
    {
        "query": "Customer C002 wants a refund on order O7842 delivered June 1, 2023. Total was $120. Policy is 30-day refund window.",
        "reference_answer": "Politely refuse — outside 30-day refund window",
        "rubrics": [
            {"name": "policy_compliance", "criteria": "Must refuse the refund citing the 30-day window policy", "weight": 0.5},
            {"name": "empathy", "criteria": "Must be understanding, not dismissive of the customer", "weight": 0.2},
            {"name": "alternative", "criteria": "Should offer alternative (store credit, exchange, escalation)", "weight": 0.2},
            {"name": "professionalism", "criteria": "Must be polite and not condescending", "weight": 0.1},
        ],
        "category": "billing",
        "difficulty": "medium",
    },
    {
        "query": "Customer C003 on Enterprise plan ($199/mo) says their invoice for this month shows $249. They want to know why.",
        "reference_answer": "Investigate the charge discrepancy, check for add-ons or overages",
        "rubrics": [
            {"name": "investigation", "criteria": "Must investigate the billing discrepancy by checking account details", "weight": 0.4},
            {"name": "explanation", "criteria": "Must provide a clear explanation of why the charge differs", "weight": 0.3},
            {"name": "resolution", "criteria": "Must offer a resolution (credit, adjustment, or explanation)", "weight": 0.2},
            {"name": "professionalism", "criteria": "Must be professional and thorough", "weight": 0.1},
        ],
        "category": "billing",
        "difficulty": "medium",
    },
    {
        "query": "Customer C005 on Free plan accidentally upgraded to Pro ($29/mo) yesterday. They want to cancel and get a refund for the accidental upgrade.",
        "reference_answer": "Process cancellation and refund for accidental upgrade",
        "rubrics": [
            {"name": "empathy", "criteria": "Must understand this was an accidental upgrade", "weight": 0.2},
            {"name": "action", "criteria": "Must process the cancellation and refund", "weight": 0.5},
            {"name": "timeline", "criteria": "Must specify when the refund will be processed", "weight": 0.2},
            {"name": "professionalism", "criteria": "Must be helpful and not make the customer feel bad", "weight": 0.1},
        ],
        "category": "billing",
        "difficulty": "easy",
    },

    # --- Technical Support ---
    {
        "query": "Customer C004 reports the app crashes every time they open Settings on iPhone 15, iOS 17.2. They've tried restarting.",
        "reference_answer": "Categorize as bug, escalate to engineering with device details",
        "rubrics": [
            {"name": "diagnosis", "criteria": "Must ask relevant questions about the crash (frequency, error messages)", "weight": 0.3},
            {"name": "escalation", "criteria": "Must escalate to engineering with device model and OS version", "weight": 0.4},
            {"name": "communication", "criteria": "Must keep customer informed about next steps and timeline", "weight": 0.2},
            {"name": "professionalism", "criteria": "Must be patient and not blame the customer", "weight": 0.1},
        ],
        "category": "technical",
        "difficulty": "medium",
    },
    {
        "query": "Customer reports the search feature returns wrong results when searching for contacts with special characters (O'Brien, Müller).",
        "reference_answer": "Categorize as encoding bug, escalate to engineering",
        "rubrics": [
            {"name": "diagnosis", "criteria": "Must identify this as a character encoding/search issue", "weight": 0.3},
            {"name": "reproduction", "criteria": "Must gather enough details to reproduce the issue", "weight": 0.3},
            {"name": "escalation", "criteria": "Must escalate with specific reproduction steps", "weight": 0.3},
            {"name": "workaround", "criteria": "Should suggest a temporary workaround if possible", "weight": 0.1},
        ],
        "category": "technical",
        "difficulty": "medium",
    },
    {
        "query": "Customer says the app is extremely slow after the latest update. Loading a page takes 10+ seconds. Other apps work fine.",
        "reference_answer": "Collect device info, check for known issues, escalate if needed",
        "rubrics": [
            {"name": "diagnosis", "criteria": "Must ask about device, OS, network conditions", "weight": 0.3},
            {"name": "troubleshooting", "criteria": "Must suggest clearing cache, reinstalling, checking network", "weight": 0.3},
            {"name": "escalation", "criteria": "Must escalate if troubleshooting doesn't resolve", "weight": 0.3},
            {"name": "empathy", "criteria": "Must acknowledge the frustration of slow performance", "weight": 0.1},
        ],
        "category": "technical",
        "difficulty": "medium",
    },
    {
        "query": "Customer's data loss after system update. Critical business data missing from dashboard. They need immediate help.",
        "reference_answer": "Escalate immediately to engineering, initiate data recovery",
        "rubrics": [
            {"name": "urgency", "criteria": "Must recognize this as critical and escalate immediately", "weight": 0.4},
            {"name": "action", "criteria": "Must initiate data recovery process or escalate to engineering", "weight": 0.3},
            {"name": "communication", "criteria": "Must keep customer informed about recovery status", "weight": 0.2},
            {"name": "professionalism", "criteria": "Must be calm and reassuring despite the urgency", "weight": 0.1},
        ],
        "category": "technical",
        "difficulty": "hard",
    },

    # --- Security Incidents ---
    {
        "query": "Customer's API key was exposed on GitHub. They need it rotated immediately before any damage.",
        "reference_answer": "Revoke old key immediately, generate new key, flag security incident",
        "rubrics": [
            {"name": "urgency", "criteria": "Must treat as critical security incident and act immediately", "weight": 0.4},
            {"name": "action", "criteria": "Must revoke old key and generate new one", "weight": 0.3},
            {"name": "communication", "criteria": "Must explain what happened and what the customer should do next", "weight": 0.2},
            {"name": "followup", "criteria": "Should offer to monitor for unauthorized usage", "weight": 0.1},
        ],
        "category": "security",
        "difficulty": "hard",
    },
    {
        "query": "Customer reports seeing someone else's data in their dashboard. Possible data breach or account compromise.",
        "reference_answer": "Immediate account lockdown, escalate to security team",
        "rubrics": [
            {"name": "urgency", "criteria": "Must treat as potential security breach", "weight": 0.4},
            {"name": "action", "criteria": "Must lock account and escalate to security team", "weight": 0.3},
            {"name": "investigation", "criteria": "Must gather details about what data was visible", "weight": 0.2},
            {"name": "communication", "criteria": "Must keep customer informed about investigation", "weight": 0.1},
        ],
        "category": "security",
        "difficulty": "hard",
    },

    # --- Account Management ---
    {
        "query": "Customer C005 wants to upgrade from Free to Enterprise ($199/mo). They want to know about prorated billing and data preservation.",
        "reference_answer": "Explain prorated billing, confirm data preserved, highlight Enterprise benefits",
        "rubrics": [
            {"name": "billing_clarity", "criteria": "Must explain prorated billing correctly", "weight": 0.3},
            {"name": "data_preservation", "criteria": "Must confirm data will be preserved during upgrade", "weight": 0.3},
            {"name": "upsell", "criteria": "Should highlight Enterprise benefits (SSO, unlimited API, dedicated support)", "weight": 0.2},
            {"name": "professionalism", "criteria": "Must be helpful and not pushy", "weight": 0.2},
        ],
        "category": "account",
        "difficulty": "easy",
    },
    {
        "query": "Customer C004 wants to cancel their Pro subscription. They've been a customer for 2 years.",
        "reference_answer": "Process cancellation, offer retention incentive, be professional",
        "rubrics": [
            {"name": "cancellation", "criteria": "Must process the cancellation request", "weight": 0.3},
            {"name": "retention", "criteria": "Should offer a retention incentive (discount, downgrade option)", "weight": 0.3},
            {"name": "professionalism", "criteria": "Must be professional and not guilt-trip the customer", "weight": 0.2},
            {"name": "followup", "criteria": "Should mention they can return anytime", "weight": 0.2},
        ],
        "category": "account",
        "difficulty": "medium",
    },
    {
        "query": "Customer C003 wants to add SSO integration for their Enterprise plan. They use Okta.",
        "reference_answer": "Explain SSO setup process for Enterprise with Okta",
        "rubrics": [
            {"name": "knowledge", "criteria": "Must know SSO is an Enterprise feature", "weight": 0.3},
            {"name": "setup_guidance", "criteria": "Must provide clear setup steps or documentation link", "weight": 0.3},
            {"name": "compatibility", "criteria": "Must confirm Okta is supported", "weight": 0.2},
            {"name": "followup", "criteria": "Should offer setup assistance or demo", "weight": 0.2},
        ],
        "category": "account",
        "difficulty": "easy",
    },
    {
        "query": "Customer C001 wants to add 5 team members to their Pro plan. How does pricing work for additional seats?",
        "reference_answer": "Explain per-seat pricing for Pro plan team members",
        "rubrics": [
            {"name": "pricing", "criteria": "Must explain per-seat pricing correctly", "weight": 0.4},
            {"name": "process", "criteria": "Must explain how to add team members", "weight": 0.3},
            {"name": "limits", "criteria": "Must mention any limits on Pro plan team size", "weight": 0.2},
            {"name": "professionalism", "criteria": "Must be helpful and clear", "weight": 0.1},
        ],
        "category": "account",
        "difficulty": "easy",
    },

    # --- Rate Limits / API ---
    {
        "query": "Customer C005 on Free plan exceeded API rate limit (100/hr). They have a demo with a potential client tomorrow and need a temporary increase.",
        "reference_answer": "Grant temporary increase, suggest upgrading to Pro",
        "rubrics": [
            {"name": "empathy", "criteria": "Must understand the urgency of the demo", "weight": 0.3},
            {"name": "solution", "criteria": "Must provide a temporary rate limit increase", "weight": 0.3},
            {"name": "upsell", "criteria": "Should suggest upgrading to Pro for permanent higher limits", "weight": 0.2},
            {"name": "followup", "criteria": "Should offer to help with the demo if needed", "weight": 0.2},
        ],
        "category": "api",
        "difficulty": "medium",
    },
    {
        "query": "Customer C002 on Starter plan is getting 429 rate limit errors even though they're within their 1000/hr limit. They've checked their usage.",
        "reference_answer": "Investigate the rate limiting issue, check for bugs or misconfiguration",
        "rubrics": [
            {"name": "investigation", "criteria": "Must investigate the discrepancy between expected and actual limits", "weight": 0.4},
            {"name": "troubleshooting", "criteria": "Must check for known issues, rate limit headers, burst limits", "weight": 0.3},
            {"name": "resolution", "criteria": "Must provide a concrete resolution or escalation path", "weight": 0.2},
            {"name": "communication", "criteria": "Must keep customer informed about investigation progress", "weight": 0.1},
        ],
        "category": "api",
        "difficulty": "hard",
    },

    # --- Policy Edge Cases ---
    {
        "query": "Customer wants a refund on order O7843 delivered Dec 28. Today is Jan 15. That's 18 days — within the 30-day window. But the item was opened and used.",
        "reference_answer": "Process refund within policy, note opened item if policy requires",
        "rubrics": [
            {"name": "policy_compliance", "criteria": "Must recognize this is within the 30-day refund window", "weight": 0.4},
            {"name": "investigation", "criteria": "Must check if opened/used items have different policy", "weight": 0.3},
            {"name": "resolution", "criteria": "Must process refund or explain policy for opened items", "weight": 0.2},
            {"name": "professionalism", "criteria": "Must be fair and transparent about policies", "weight": 0.1},
        ],
        "category": "policy",
        "difficulty": "hard",
    },
    {
        "query": "Customer C001's subscription renews in 3 days. They want to cancel but are worried about losing access immediately.",
        "reference_answer": "Clarify cancellation policy — access continues until end of billing period",
        "rubrics": [
            {"name": "policy_clarity", "criteria": "Must clarify that access continues until end of billing period", "weight": 0.4},
            {"name": "reassurance", "criteria": "Must reassure customer they won't lose access immediately", "weight": 0.3},
            {"name": "process", "criteria": "Must explain the cancellation process clearly", "weight": 0.2},
            {"name": "professionalism", "criteria": "Must be helpful and not pressure to stay", "weight": 0.1},
        ],
        "category": "policy",
        "difficulty": "easy",
    },

    # --- Complex Multi-Issue ---
    {
        "query": "Customer C003 reports: (1) charged $249 instead of $199, (2) SSO setup isn't working, (3) their team member can't log in. Multiple issues at once.",
        "reference_answer": "Prioritize issues, address billing first, then SSO, then login",
        "rubrics": [
            {"name": "prioritization", "criteria": "Must prioritize issues logically (billing first, then technical)", "weight": 0.3},
            {"name": "billing_resolution", "criteria": "Must address the $50 overcharge", "weight": 0.3},
            {"name": "technical_support", "criteria": "Must address SSO and login issues", "weight": 0.3},
            {"name": "communication", "criteria": "Must keep customer informed about each issue's status", "weight": 0.1},
        ],
        "category": "complex",
        "difficulty": "hard",
    },
    {
        "query": "Customer is angry. They've been waiting 3 days for a response to their support ticket about a billing error. They're threatening to leave a negative review.",
        "reference_answer": "Apologize for delay, escalate billing issue, offer compensation",
        "rubrics": [
            {"name": "empathy", "criteria": "Must sincerely apologize for the delay", "weight": 0.3},
            {"name": "escalation", "criteria": "Must escalate the billing issue immediately", "weight": 0.3},
            {"name": "compensation", "criteria": "Should offer compensation (credit, discount) for the delay", "weight": 0.2},
            {"name": "de_escalation", "criteria": "Must de-escalate the situation and retain the customer", "weight": 0.2},
        ],
        "category": "complex",
        "difficulty": "hard",
    },

    # --- Edge Cases ---
    {
        "query": "Customer asks: 'Can I speak to a human?' after interacting with the AI agent for 5 minutes about a complex billing issue.",
        "reference_answer": "Acknowledge request, transfer to human agent smoothly",
        "rubrics": [
            {"name": "acknowledgment", "criteria": "Must acknowledge the request without resistance", "weight": 0.3},
            {"name": "transfer", "criteria": "Must initiate transfer to human agent", "weight": 0.4},
            {"name": "context_preservation", "criteria": "Must preserve conversation context for the human agent", "weight": 0.2},
            {"name": "professionalism", "criteria": "Must not make the customer feel bad for asking", "weight": 0.1},
        ],
        "category": "edge_case",
        "difficulty": "medium",
    },
]
