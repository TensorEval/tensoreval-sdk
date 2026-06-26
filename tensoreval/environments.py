"""Built-in example environments for TensorEval.

Pre-built environments based on Verifiers patterns (MIT License).
Users can load these directly without writing any environment code.

Usage:
    env = te.load_env("gsm8k")
    env = te.load_env("math")
    env = te.load_env("code-generation")
    env = te.load_env("customer-support")
"""

from typing import Any, Callable


# ---------------------------------------------------------------------------
# Environment registry
# ---------------------------------------------------------------------------

_ENV_REGISTRY: dict[str, dict[str, Any]] = {}


def register_env(
    name: str,
    description: str,
    category: str,
    difficulty: str,
    system_prompt: str,
    dataset_loader: Callable,
    reward_fn: Callable,
    parser: Any | None = None,
):
    """Register a built-in environment."""
    _ENV_REGISTRY[name] = {
        "name": name,
        "description": description,
        "category": category,
        "difficulty": difficulty,
        "system_prompt": system_prompt,
        "dataset_loader": dataset_loader,
        "reward_fn": reward_fn,
        "parser": parser,
    }


def list_envs() -> list[dict[str, str]]:
    """List all registered environments."""
    return [
        {"name": e["name"], "description": e["description"], "category": e["category"], "difficulty": e["difficulty"]}
        for e in _ENV_REGISTRY.values()
    ]


def load_env(name: str, **kwargs) -> Any:
    """Load a built-in environment by name.

    Args:
        name: Environment name (e.g., "gsm8k", "math", "code-generation")
        **kwargs: Additional arguments passed to the environment constructor

    Returns:
        Environment instance ready for evaluation
    """
    from tensoreval.envs.singleturn_env import SingleTurnEnv
    from tensoreval.rubrics.rubric import Rubric
    from tensoreval.datasets import Datasets

    if name not in _ENV_REGISTRY:
        available = ", ".join(_ENV_REGISTRY.keys())
        raise ValueError(f"Unknown environment: {name}. Available: {available}")

    entry = _ENV_REGISTRY[name]
    dataset = entry["dataset_loader"](**kwargs)
    reward_fn = entry["reward_fn"]
    rubric = Rubric(funcs=[reward_fn], weights=[1.0])

    env = SingleTurnEnv(
        rubric=rubric,
        system_prompt=entry["system_prompt"],
        **{k: v for k, v in kwargs.items() if k not in ["n", "split", "seed"]},
    )

    # Set dataset on environment
    if isinstance(dataset, Datasets):
        env.dataset = dataset.to_dicts()
        env.eval_dataset = dataset.to_dicts()
    else:
        env.dataset = dataset
        env.eval_dataset = dataset

    return env


# ---------------------------------------------------------------------------
# Reward functions (from Verifiers patterns)
# ---------------------------------------------------------------------------

def _exact_match_reward(state: dict, **kwargs) -> float:
    """Exact match reward — checks if answer appears in response."""
    completion = state.get("completion", [])
    answer = state.get("answer", "")
    if not completion:
        return 0.0
    last = completion[-1]
    response = last.get("content", "") if isinstance(last, dict) else str(getattr(last, "content", ""))
    return 1.0 if answer.strip() in response.strip() else 0.0


def _numeric_match_reward(state: dict, **kwargs) -> float:
    """Numeric match reward — extracts number from response and compares."""
    import re
    completion = state.get("completion", [])
    answer = state.get("answer", "")
    if not completion:
        return 0.0
    last = completion[-1]
    response = last.get("content", "") if isinstance(last, dict) else str(getattr(last, "content", ""))
    # Extract numbers from response
    numbers = re.findall(r'-?\d+\.?\d*', response)
    if not numbers:
        return 0.0
    try:
        answer_num = float(answer)
        for num_str in numbers:
            if abs(float(num_str) - answer_num) < 0.01:
                return 1.0
    except ValueError:
        pass
    return 0.0


def _boxed_answer_reward(state: dict, **kwargs) -> float:
    """Boxed answer reward — extracts \\boxed{} and compares."""
    from tensoreval.utils.data_utils import extract_boxed_answer
    completion = state.get("completion", [])
    answer = state.get("answer", "")
    if not completion:
        return 0.0
    last = completion[-1]
    response = last.get("content", "") if isinstance(last, dict) else str(getattr(last, "content", ""))
    extracted = extract_boxed_answer(response, strict=True)
    if not extracted:
        extracted = response.strip()
    return 1.0 if extracted == answer else 0.0


def _code_quality_reward(state: dict, **kwargs) -> float:
    """Code quality reward — checks for function definition, docstring, return."""
    completion = state.get("completion", [])
    if not completion:
        return 0.0
    last = completion[-1]
    response = last.get("content", "") if isinstance(last, dict) else str(getattr(last, "content", ""))
    score = 0.0
    if "def " in response or "class " in response:
        score += 0.3
    if '"""' in response or "'''" in response:
        score += 0.2
    if "return " in response:
        score += 0.2
    if ":" in response and "(" in response and ")" in response:
        score += 0.1
    if 50 < len(response) < 5000:
        score += 0.1
    if "try:" in response or "except" in response or "if " in response:
        score += 0.1
    return min(score, 1.0)


def _customer_support_reward(state: dict, **kwargs) -> float:
    """Customer support reward — checks for empathy, policy compliance, action."""
    completion = state.get("completion", [])
    answer = state.get("answer", "")
    if not completion:
        return 0.0
    last = completion[-1]
    response = last.get("content", "") if isinstance(last, dict) else str(getattr(last, "content", ""))
    response_lower = response.lower()
    score = 0.0
    # Empathy
    empathy_words = ["understand", "apologize", "sorry", "appreciate", "thank", "help"]
    if any(w in response_lower for w in empathy_words):
        score += 0.2
    # Action
    action_words = ["will", "have", "processed", "issued", "applied", "refund", "credit"]
    if any(w in response_lower for w in action_words):
        score += 0.3
    # Policy awareness
    if "policy" in response_lower or "days" in response_lower:
        score += 0.2
    # Professional tone
    if len(response) > 50 and len(response) < 2000:
        score += 0.15
    # Specificity
    if "$" in response or "%" in response or any(c.isdigit() for c in response):
        score += 0.15
    return min(score, 1.0)


def _data_analysis_reward(state: dict, **kwargs) -> float:
    """Data analysis reward — checks for calculations, analysis, presentation."""
    import re
    completion = state.get("completion", [])
    if not completion:
        return 0.0
    last = completion[-1]
    response = last.get("content", "") if isinstance(last, dict) else str(getattr(last, "content", ""))
    score = 0.0
    # Numbers present
    numbers = re.findall(r'[\d,]+\.?\d*', response)
    if numbers:
        score += 0.25
    # Percentages
    if "%" in response:
        score += 0.15
    # Analysis words
    analysis_words = ["total", "average", "growth", "increase", "decrease", "trend", "highest", "lowest", "rate"]
    if any(w in response.lower() for w in analysis_words):
        score += 0.25
    # Reasonable length
    if 100 < len(response) < 3000:
        score += 0.15
    # Step-by-step
    if "step" in response.lower() or "1." in response or "first" in response.lower():
        score += 0.1
    # Formatting
    if "\n" in response:
        score += 0.1
    return min(score, 1.0)


def _reasoning_reward(state: dict, **kwargs) -> float:
    """Reasoning reward — checks for step-by-step reasoning."""
    completion = state.get("completion", [])
    answer = state.get("answer", "")
    if not completion:
        return 0.0
    last = completion[-1]
    response = last.get("content", "") if isinstance(last, dict) else str(getattr(last, "content", ""))
    score = 0.0
    # Check for reasoning indicators
    reasoning_words = ["because", "therefore", "since", "thus", "hence", "so", "step"]
    if any(w in response.lower() for w in reasoning_words):
        score += 0.3
    # Check for answer
    if answer and answer.lower() in response.lower():
        score += 0.5
    # Check for structured response
    if "\n" in response or "." in response:
        score += 0.1
    # Length check
    if 20 < len(response) < 5000:
        score += 0.1
    return min(score, 1.0)


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def _load_gsm8k(n: int = 50, split: str = "test", **kwargs):
    """Load GSM8K math dataset."""
    from tensoreval.datasets import Datasets
    try:
        return Datasets.from_huggingface("gsm8k", split=split, n=n, name="gsm8k")
    except Exception:
        # Fallback: built-in sample data
        return _builtin_math_dataset(n)


def _load_math(n: int = 50, split: str = "train", **kwargs):
    """Load MATH dataset."""
    from tensoreval.datasets import Datasets
    try:
        return Datasets.from_huggingface("math", split=split, n=n, name="math")
    except Exception:
        return _builtin_math_dataset(n)


def _load_code_generation(n: int = 10, **kwargs):
    """Load code generation tasks."""
    from tensoreval.datasets import Datasets
    tasks = [
        {"query": "Write a Python function that validates an email address.", "reference_answer": "email validation function", "rubrics": [{"name": "correctness", "rubric": "Must validate email format", "weight": 0.5}, {"name": "code_quality", "rubric": "Must have docstring and proper structure", "weight": 0.3}, {"name": "edge_cases", "rubric": "Must handle edge cases", "weight": 0.2}]},
        {"query": "Write a Python function that sorts a list of dictionaries by a specified key.", "reference_answer": "dict sort function", "rubrics": [{"name": "correctness", "rubric": "Must sort correctly", "weight": 0.5}, {"name": "missing_keys", "rubric": "Must handle missing keys", "weight": 0.3}, {"name": "code_quality", "rubric": "Clean code with docstring", "weight": 0.2}]},
        {"query": "Write a Python class implementing an LRU cache.", "reference_answer": "LRU cache class", "rubrics": [{"name": "correctness", "rubric": "Must implement get/put with O(1) time", "weight": 0.5}, {"name": "design", "rubric": "Must use OrderedDict or doubly-linked list", "weight": 0.3}, {"name": "code_quality", "rubric": "Clean API with docstrings", "weight": 0.2}]},
        {"query": "Write a Python decorator that retries a function on exception.", "reference_answer": "retry decorator", "rubrics": [{"name": "correctness", "rubric": "Must retry on exception with configurable attempts", "weight": 0.5}, {"name": "flexibility", "rubric": "Must support max_retries and delay parameters", "weight": 0.3}, {"name": "code_quality", "rubric": "Clean decorator with docstring", "weight": 0.2}]},
        {"query": "Write a Python function that flattens a nested dictionary.", "reference_answer": "flatten dict function", "rubrics": [{"name": "correctness", "rubric": "Must handle arbitrary nesting depth", "weight": 0.5}, {"name": "key_format", "rubric": "Must use dot notation for nested keys", "weight": 0.3}, {"name": "code_quality", "rubric": "Clean recursive implementation", "weight": 0.2}]},
        {"query": "Write a Python function that implements binary search.", "reference_answer": "binary search function", "rubrics": [{"name": "correctness", "rubric": "Must return correct index or -1", "weight": 0.5}, {"name": "edge_cases", "rubric": "Must handle empty list and single element", "weight": 0.3}, {"name": "code_quality", "rubric": "Clean iterative or recursive implementation", "weight": 0.2}]},
        {"query": "Write a Python context manager for database connections.", "reference_answer": "context manager class", "rubrics": [{"name": "correctness", "rubric": "Must implement __enter__ and __exit__", "weight": 0.5}, {"name": "error_handling", "rubric": "Must handle connection errors gracefully", "weight": 0.3}, {"name": "code_quality", "rubric": "Clean implementation with docstring", "weight": 0.2}]},
        {"query": "Write a Python function that parses CSV data into dictionaries.", "reference_answer": "CSV parser function", "rubrics": [{"name": "correctness", "rubric": "Must handle headers and data rows", "weight": 0.5}, {"name": "edge_cases", "rubric": "Must handle quoted fields and empty values", "weight": 0.3}, {"name": "code_quality", "rubric": "Clean implementation", "weight": 0.2}]},
        {"query": "Write a Python generator that yields prime numbers.", "reference_answer": "prime number generator", "rubrics": [{"name": "correctness", "rubric": "Must yield correct primes", "weight": 0.5}, {"name": "efficiency", "rubric": "Should use sieve or trial division", "weight": 0.3}, {"name": "code_quality", "rubric": "Clean generator implementation", "weight": 0.2}]},
        {"query": "Write a Python function that validates JSON schema.", "reference_answer": "JSON schema validator", "rubrics": [{"name": "correctness", "rubric": "Must validate types and required fields", "weight": 0.5}, {"name": "flexibility", "rubric": "Must support nested schemas", "weight": 0.3}, {"name": "code_quality", "rubric": "Clean implementation with error messages", "weight": 0.2}]},
    ]
    return Datasets.from_dicts(tasks[:n], name="code_generation")


def _load_customer_support(n: int = 10, **kwargs):
    """Load customer support tasks."""
    from tensoreval.datasets import Datasets
    tasks = [
        {"query": "Customer wants refund on order delivered 10 days ago ($49.99). Policy: 30-day refund.", "reference_answer": "Full refund issued", "rubrics": [{"name": "policy", "rubric": "Must verify within 30 days and issue refund", "weight": 0.5}, {"name": "empathy", "rubric": "Must acknowledge concern", "weight": 0.3}, {"name": "clarity", "rubric": "Must state refund amount clearly", "weight": 0.2}]},
        {"query": "Customer wants refund on order delivered 45 days ago ($120). Policy: 30-day refund.", "reference_answer": "Politely refused - outside policy window", "rubrics": [{"name": "policy", "rubric": "Must refuse citing 30-day policy", "weight": 0.5}, {"name": "empathy", "rubric": "Must be understanding", "weight": 0.3}, {"name": "alternative", "rubric": "May offer alternative solutions", "weight": 0.2}]},
        {"query": "Customer was charged twice for Pro plan ($29/mo). Wants duplicate removed.", "reference_answer": "Duplicate charge removed", "rubrics": [{"name": "identification", "rubric": "Must identify the duplicate", "weight": 0.4}, {"name": "resolution", "rubric": "Must remove duplicate charge", "weight": 0.4}, {"name": "communication", "rubric": "Must explain clearly", "weight": 0.2}]},
        {"query": "Customer wants to upgrade from Free to Enterprise ($99/mo) and keep data.", "reference_answer": "Upgrade path explained, data preserved", "rubrics": [{"name": "upgrade_clarity", "rubric": "Must explain upgrade process", "weight": 0.3}, {"name": "data_preservation", "rubric": "Must confirm data preserved", "weight": 0.4}, {"name": "pricing", "rubric": "Must provide accurate pricing", "weight": 0.3}]},
        {"query": "Free user exceeded API rate limit (100/hr). Has demo tomorrow. Wants temporary increase.", "reference_answer": "Temporary increase granted", "rubrics": [{"name": "empathy", "rubric": "Must understand urgency", "weight": 0.3}, {"name": "solution", "rubric": "Must provide temporary increase", "weight": 0.4}, {"name": "upsell", "rubric": "Should suggest Pro plan", "weight": 0.3}]},
        {"query": "Customer reports app crashes on mobile. Using iPhone 15, iOS 17.", "reference_answer": "Troubleshooting steps provided", "rubrics": [{"name": "diagnosis", "rubric": "Must ask relevant questions about the crash", "weight": 0.4}, {"name": "solution", "rubric": "Must provide actionable troubleshooting steps", "weight": 0.4}, {"name": "followup", "rubric": "Should offer to escalate if issue persists", "weight": 0.2}]},
        {"query": "Customer wants to cancel subscription. Currently on Pro ($29/mo), 6 months remaining.", "reference_answer": "Cancellation processed, retention offer made", "rubrics": [{"name": "cancellation", "rubric": "Must process cancellation request", "weight": 0.4}, {"name": "retention", "rubric": "Should offer incentive to stay", "weight": 0.3}, {"name": "professionalism", "rubric": "Must be professional and not pushy", "weight": 0.3}]},
        {"query": "Customer reports data loss after system update. Critical business data affected.", "reference_answer": "Escalated to engineering, data recovery initiated", "rubrics": [{"name": "urgency", "rubric": "Must recognize severity and escalate", "weight": 0.4}, {"name": "communication", "rubric": "Must keep customer informed of next steps", "weight": 0.3}, {"name": "documentation", "rubric": "Must document the issue properly", "weight": 0.3}]},
        {"query": "Customer asks about SSO integration for Enterprise plan.", "reference_answer": "SSO capabilities explained, setup process outlined", "rubrics": [{"name": "knowledge", "rubric": "Must know SSO is Enterprise feature", "weight": 0.4}, {"name": "explanation", "rubric": "Must explain supported providers and setup", "weight": 0.4}, {"name": "next_steps", "rubric": "Should offer demo or setup assistance", "weight": 0.2}]},
        {"query": "Customer reports slow performance on large datasets (>1GB).", "reference_answer": "Performance optimization suggestions provided", "rubrics": [{"name": "diagnosis", "rubric": "Must ask about dataset size and operations", "weight": 0.3}, {"name": "solutions", "rubric": "Must provide concrete optimization suggestions", "weight": 0.5}, {"name": "escalation", "rubric": "Should offer engineering support if needed", "weight": 0.2}]},
    ]
    return Datasets.from_dicts(tasks[:n], name="customer_support")


def _load_data_analysis(n: int = 10, **kwargs):
    """Load data analysis tasks."""
    from tensoreval.datasets import Datasets
    tasks = [
        {"query": "Sales: Q1=$45K, Q2=$52K, Q3=$48K, Q4=$61K. Calculate total, growth rates, best quarter.", "reference_answer": "Total $206K, best Q4", "rubrics": [{"name": "calculations", "rubric": "Must compute totals and growth correctly", "weight": 0.5}, {"name": "analysis", "rubric": "Must identify trends and best quarter", "weight": 0.3}, {"name": "presentation", "rubric": "Must present clearly", "weight": 0.2}]},
        {"query": "Departments: Eng(50 emp, $95K avg), Sales(30, $75K), Marketing(20, $70K). Company avg? Highest total cost?", "reference_answer": "Avg $83.5K, highest total: Eng $4.75M", "rubrics": [{"name": "weighted_average", "rubric": "Must compute weighted average correctly", "weight": 0.5}, {"name": "total_cost", "rubric": "Must compute total cost per department", "weight": 0.3}, {"name": "identification", "rubric": "Must identify highest cost department", "weight": 0.2}]},
        {"query": "Monthly revenue: Jan-Dec varying. Calculate 3-month moving average and identify trends.", "reference_answer": "Moving averages computed, trends identified", "rubrics": [{"name": "moving_average", "rubric": "Must compute correctly", "weight": 0.5}, {"name": "trend_analysis", "rubric": "Must identify meaningful trends", "weight": 0.3}, {"name": "interpretation", "rubric": "Must provide business interpretation", "weight": 0.2}]},
        {"query": "A/B test: Group A (1000 users, 50 conv), Group B (1000, 65 conv). Calculate rates and significance.", "reference_answer": "A: 5%, B: 6.5%, 30% improvement", "rubrics": [{"name": "rates", "rubric": "Must calculate conversion rates correctly", "weight": 0.4}, {"name": "improvement", "rubric": "Must calculate relative improvement", "weight": 0.3}, {"name": "significance", "rubric": "Must apply appropriate statistical test", "weight": 0.3}]},
        {"query": "Churn data: Month 1-5 customers declining. Calculate churn rates and predict Month 6.", "reference_answer": "Churn rates calculated, prediction made", "rubrics": [{"name": "churn_rates", "rubric": "Must calculate monthly churn rates", "weight": 0.4}, {"name": "prediction", "rubric": "Must predict Month 6 using average churn", "weight": 0.4}, {"name": "trend", "rubric": "Should note if churn is improving or worsening", "weight": 0.2}]},
        {"query": "Product A: 1000 units, $50 each. Product B: 500 units, $120 each. Product C: 2000 units, $25 each. Revenue mix analysis.", "reference_answer": "Revenue by product and percentages", "rubrics": [{"name": "revenue_calculation", "rubric": "Must calculate revenue per product", "weight": 0.4}, {"name": "mix_analysis", "rubric": "Must calculate percentage of total", "weight": 0.4}, {"name": "insights", "rubric": "Should provide actionable insights", "weight": 0.2}]},
        {"query": "Website traffic: 10K visits, 2% conversion, $50 avg order. Calculate revenue, CPA if ad spend is $5K.", "reference_answer": "Revenue $10K, CPA $25", "rubrics": [{"name": "revenue", "rubric": "Must calculate revenue correctly", "weight": 0.4}, {"name": "cpa", "rubric": "Must calculate cost per acquisition", "weight": 0.4}, {"name": "metrics", "rubric": "Should provide additional relevant metrics", "weight": 0.2}]},
        {"query": "Cohort analysis: Month 1 cohort (1000 users). Month 2: 600 active, Month 3: 450, Month 4: 380. Retention rates?", "reference_answer": "Retention: 60%, 45%, 38%", "rubrics": [{"name": "retention_rates", "rubric": "Must calculate correctly", "weight": 0.5}, {"name": "interpretation", "rubric": "Must interpret what rates mean", "weight": 0.3}, {"name": "recommendations", "rubric": "Should suggest improvements", "weight": 0.2}]},
        {"query": "Salary data: mean $75K, median $65K, std $20K. What does the gap between mean and median suggest?", "reference_answer": "Right-skewed distribution, high earners pulling up the mean", "rubrics": [{"name": "interpretation", "rubric": "Must correctly interpret mean vs median gap", "weight": 0.5}, {"name": "distribution", "rubric": "Must identify right-skewed distribution", "weight": 0.3}, {"name": "implications", "rubric": "Should discuss business implications", "weight": 0.2}]},
        {"query": "Customer LTV: avg purchase $50, frequency 4x/year, lifespan 3 years, margin 30%. Calculate LTV.", "reference_answer": "LTV = $50 * 4 * 3 * 0.30 = $180", "rubrics": [{"name": "calculation", "rubric": "Must compute LTV correctly", "weight": 0.5}, {"name": "formula", "rubric": "Must show the formula used", "weight": 0.3}, {"name": "context", "rubric": "Should discuss what LTV means for business", "weight": 0.2}]},
    ]
    return Datasets.from_dicts(tasks[:n], name="data_analysis")


def _load_reasoning(n: int = 10, **kwargs):
    """Load reasoning tasks."""
    from tensoreval.datasets import Datasets
    tasks = [
        {"query": "A bat and ball cost $1.10 total. The bat costs $1 more than the ball. How much does the ball cost?", "reference_answer": "$0.05", "rubrics": [{"name": "correctness", "rubric": "Must answer $0.05 (not $0.10)", "weight": 0.6}, {"name": "reasoning", "rubric": "Must show the algebraic reasoning", "weight": 0.4}]},
        {"query": "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?", "reference_answer": "5 minutes", "rubrics": [{"name": "correctness", "rubric": "Must answer 5 minutes", "weight": 0.6}, {"name": "reasoning", "rubric": "Must explain rate per machine", "weight": 0.4}]},
        {"query": "A lily pad doubles in size every day. If it takes 48 days to cover the lake, how many days to cover half?", "reference_answer": "47 days", "rubrics": [{"name": "correctness", "rubric": "Must answer 47 days", "weight": 0.6}, {"name": "reasoning", "rubric": "Must explain the doubling logic", "weight": 0.4}]},
        {"query": "You have 8 balls, one is heavier. Using a balance scale, what's the minimum weighings to find it?", "reference_answer": "2 weighings", "rubrics": [{"name": "correctness", "rubric": "Must answer 2", "weight": 0.5}, {"name": "strategy", "rubric": "Must describe the weighing strategy", "weight": 0.5}]},
        {"query": "A farmer has 17 sheep. All but 9 die. How many are left?", "reference_answer": "9", "rubrics": [{"name": "correctness", "rubric": "Must answer 9", "weight": 0.6}, {"name": "reasoning", "rubric": "Must explain 'all but 9' means 9 remain", "weight": 0.4}]},
        {"query": "If you're running a race and pass the person in 2nd place, what place are you in?", "reference_answer": "2nd place", "rubrics": [{"name": "correctness", "rubric": "Must answer 2nd (not 1st)", "weight": 0.6}, {"name": "reasoning", "rubric": "Must explain why not 1st", "weight": 0.4}]},
        {"query": "How many times can you subtract 5 from 25?", "reference_answer": "Once (then it's 20, not 25)", "rubrics": [{"name": "correctness", "rubric": "Must answer once", "weight": 0.5}, {"name": "reasoning", "rubric": "Must explain the trick", "weight": 0.5}]},
        {"query": "A clock shows 3:15. What is the angle between the hour and minute hands?", "reference_answer": "7.5 degrees", "rubrics": [{"name": "correctness", "rubric": "Must calculate 7.5 degrees", "weight": 0.5}, {"name": "calculation", "rubric": "Must show the calculation method", "weight": 0.5}]},
        {"query": "You have a 3-gallon jug and a 5-gallon jug. How do you measure exactly 4 gallons?", "reference_answer": "Fill 5, pour into 3, empty 3, pour remaining 2 into 3, fill 5, pour into 3 (1 gallon space), leaves 4 in 5-gallon jug", "rubrics": [{"name": "correctness", "rubric": "Must provide correct sequence of steps", "weight": 0.5}, {"name": "clarity", "rubric": "Steps must be clear and executable", "weight": 0.5}]},
        {"query": "Two trains are 100 miles apart, heading toward each other at 50mph each. A bird flies at 75mph between them. How far does the bird fly before trains meet?", "reference_answer": "75 miles (1 hour until meet, bird flies 75mph * 1hr)", "rubrics": [{"name": "correctness", "rubric": "Must calculate 75 miles", "weight": 0.5}, {"name": "reasoning", "rubric": "Must explain time to meet and bird distance", "weight": 0.5}]},
    ]
    return Datasets.from_dicts(tasks[:n], name="reasoning")


def _builtin_math_dataset(n: int = 10):
    """Built-in math dataset as fallback when HuggingFace is unavailable."""
    from tensoreval.datasets import Datasets
    tasks = [
        {"query": "What is 12 * 15?", "reference_answer": "180"},
        {"query": "What is 24 + 36?", "reference_answer": "60"},
        {"query": "What is 100 / 4?", "reference_answer": "25"},
        {"query": "What is 7 * 8?", "reference_answer": "56"},
        {"query": "What is 144 / 12?", "reference_answer": "12"},
        {"query": "What is 15% of 200?", "reference_answer": "30"},
        {"query": "What is the square root of 169?", "reference_answer": "13"},
        {"query": "What is 2 to the power of 10?", "reference_answer": "1024"},
        {"query": "What is 3/4 as a decimal?", "reference_answer": "0.75"},
        {"query": "What is 15% of 80?", "reference_answer": "12"},
        {"query": "What is 999 + 1?", "reference_answer": "1000"},
        {"query": "What is 25 * 4?", "reference_answer": "100"},
        {"query": "What is 81 / 9?", "reference_answer": "9"},
        {"query": "What is 11 * 11?", "reference_answer": "121"},
        {"query": "What is 50% of 300?", "reference_answer": "150"},
    ]
    return Datasets.from_dicts(tasks[:n], name="math_builtin")


# ---------------------------------------------------------------------------
# Register all built-in environments
# ---------------------------------------------------------------------------

register_env(
    name="gsm8k",
    description="Grade-school math problems (GSM8K benchmark)",
    category="math",
    difficulty="easy",
    system_prompt="Solve the grade-school math problem. Reason step by step, then put your final answer within \\boxed{}.",
    dataset_loader=_load_gsm8k,
    reward_fn=_boxed_answer_reward,
)

register_env(
    name="math",
    description="Competition math problems (MATH benchmark)",
    category="math",
    difficulty="hard",
    system_prompt="Solve the math problem. Show your reasoning step by step. Put your final answer within \\boxed{}.",
    dataset_loader=_load_math,
    reward_fn=_boxed_answer_reward,
)

register_env(
    name="code-generation",
    description="Python code generation tasks",
    category="coding",
    difficulty="medium",
    system_prompt="You are an expert Python developer. Write clean, well-documented, production-ready code. Include docstrings and handle edge cases.",
    dataset_loader=_load_code_generation,
    reward_fn=_code_quality_reward,
)

register_env(
    name="customer-support",
    description="Customer support agent evaluation scenarios",
    category="business",
    difficulty="medium",
    system_prompt="You are a professional customer support agent for a SaaS company. Resolve customer issues according to company policy. Be empathetic, clear, and take appropriate action.",
    dataset_loader=_load_customer_support,
    reward_fn=_customer_support_reward,
)

register_env(
    name="data-analysis",
    description="Data analysis and business intelligence tasks",
    category="analytics",
    difficulty="medium",
    system_prompt="You are a data analyst. Provide clear, accurate calculations with explanations. Show your work step by step.",
    dataset_loader=_load_data_analysis,
    reward_fn=_data_analysis_reward,
)

register_env(
    name="reasoning",
    description="Logic and reasoning puzzles",
    category="reasoning",
    difficulty="medium",
    system_prompt="Think step by step. Explain your reasoning clearly before giving your final answer.",
    dataset_loader=_load_reasoning,
    reward_fn=_reasoning_reward,
)
