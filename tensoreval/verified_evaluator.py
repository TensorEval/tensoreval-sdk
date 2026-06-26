"""Verified evaluator — tool-augmented evaluation.

TensorEval's unique differentiator: the evaluator doesn't just reason about
answers — it VERIFIES them by running code, reading files, and fact-checking.

Ported from TensorEval's evaluator.ts.
"""

import asyncio
import json
from typing import Any

from tensoreval.core.sample import Sample
from tensoreval.core.score import RubricScore


# ---------------------------------------------------------------------------
# Evaluation prompt
# ---------------------------------------------------------------------------

EVAL_SYSTEM_PROMPT = """You are a rigorous AI agent evaluator. Your job is to VERIFY and score an agent's response against rubrics.

## CRITICAL: USE TOOLS TO VERIFY
You are NOT limited to text reasoning. You have full access to tools:
- **Bash**: Run python3, execute shell commands, run tests, compute values
- **Read files**: Read data files, source code, documents
- **WebSearch**: Search the web to fact-check claims

DO NOT just reason about whether the answer looks correct. ACTUALLY VERIFY IT.

## Evaluation Process
1. Read the query, reference answer, and agent response carefully
2. USE TOOLS to independently verify the agent's claims and computations
3. For EACH rubric, compare your verified findings against the agent's response
4. Assign a score (0.0 to 1.0) per rubric
5. Compute weighted score: sum of (score x weight)

## Rubrics to Evaluate
{rubric_list}

## Score Bands
- **1.0**: Fully satisfies the rubric
- **0.8-0.9**: Mostly satisfies, minor issues only
- **0.5-0.7**: Partially satisfies, significant gaps
- **0.2-0.4**: Barely satisfies, major issues
- **0.0-0.1**: Does not satisfy at all

## Special Rules
- **Empty/error responses**: All rubrics score 0
- **Numerical tolerance**: Accept +/-1% unless rubric specifies otherwise
- **Focus on substance**: Is the answer correct? Complete? Useful?
- **Do NOT penalize** for missing citations or formatting preferences

## Output Format
Output exactly ONE JSON object:
```json
{{
  "rubric_scores": [
    {{
      "rubric_name": "correctness",
      "score": 0.85,
      "weight": 0.4,
      "reasoning": "Detailed reasoning for this rubric score..."
    }}
  ],
  "grader_reasoning": "One paragraph summary of the overall evaluation"
}}
```"""


class VerifiedEvaluator:
    """Tool-augmented evaluator that verifies answers by running code.

    Unlike simple LLM judges, this evaluator:
    - Runs Python code to re-compute values
    - Searches the web to fact-check claims
    - Reads files to verify data
    - Executes tests to check code correctness

    Usage:
        score = await VerifiedEvaluator.evaluate_single(
            query="What is 12 * 15?",
            agent_response="180",
            reference_answer="180",
            rubrics=[{"name": "correctness", "rubric": "Must compute 12*15=180", "weight": 1.0}],
            model="mimo-v2.5-pro",
        )
    """

    @staticmethod
    async def evaluate_single(
        query: str,
        agent_response: str,
        reference_answer: str | None,
        rubrics: list[dict[str, Any]],
        model: str = "mimo-v2.5-pro",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> tuple[list[RubricScore], str]:
        """Evaluate a single response against rubrics.

        Args:
            query: The original query.
            agent_response: The agent's response.
            reference_answer: The reference answer (if available).
            rubrics: List of rubric definitions.
            model: Model to use for evaluation.
            api_key: API key for the model.
            base_url: Base URL for the model API.

        Returns:
            Tuple of (list of RubricScores, grader_reasoning).
        """
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=api_key or "dummy",
            base_url=base_url or "https://api.openai.com/v1",
        )

        # Build rubric list
        rubric_list = "\n".join(
            f"{i+1}. **{r['name']}** (weight: {r['weight']}): {r['rubric']}"
            for i, r in enumerate(rubrics)
        )

        # Build reference section
        reference_section = ""
        if reference_answer:
            reference_section = f"\n**Reference Answer (verified correct):**\n{reference_answer}\n"
        else:
            reference_section = "\n*No reference answer provided — evaluate against rubrics using your judgment.*\n"

        system_prompt = EVAL_SYSTEM_PROMPT.format(rubric_list=rubric_list)

        user_prompt = f"""## Query Under Evaluation

**Query:**
{query}
{reference_section}

**Agent's Response:**
{agent_response or '[EMPTY — agent returned no response]'}

---

Evaluate this response. Verify by running code if needed. Then output the structured JSON result."""

        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=2000,
            temperature=0.1,
        )

        content = response.choices[0].message.content or ""

        # Parse result
        rubric_scores, grader_reasoning = _parse_eval_result(content, rubrics)
        return rubric_scores, grader_reasoning

    @staticmethod
    async def evaluate_batch(
        samples: list[dict[str, Any]],
        agent_responses: list[str],
        model: str = "mimo-v2.5-pro",
        api_key: str | None = None,
        base_url: str | None = None,
        workers: int = 4,
    ) -> list[tuple[list[RubricScore], str]]:
        """Evaluate a batch of responses.

        Args:
            samples: List of sample dicts with query, rubrics, reference_answer.
            agent_responses: List of agent responses (parallel to samples).
            model: Model to use for evaluation.
            api_key: API key for the model.
            base_url: Base URL for the model API.
            workers: Number of concurrent workers.

        Returns:
            List of (rubric_scores, grader_reasoning) tuples.
        """
        sem = asyncio.Semaphore(workers)

        async def eval_one(i: int) -> tuple[list[RubricScore], str]:
            async with sem:
                sample = samples[i]
                return await VerifiedEvaluator.evaluate_single(
                    query=sample.get("query", ""),
                    agent_response=agent_responses[i],
                    reference_answer=sample.get("reference_answer"),
                    rubrics=sample.get("rubrics", []),
                    model=model,
                    api_key=api_key,
                    base_url=base_url,
                )

        tasks = [eval_one(i) for i in range(len(samples))]
        return await asyncio.gather(*tasks)


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------

def _parse_eval_result(text: str, rubrics: list[dict]) -> tuple[list[RubricScore], str]:
    """Parse evaluation result from LLM output."""
    # Extract JSON from markdown fences
    json_match = None
    for match in __import__('re').finditer(r'```(?:json)?\s*\n(\{[\s\S]*?\})\s*\n```', text):
        json_match = match.group(1)

    if not json_match:
        # Try raw JSON
        obj_match = __import__('re').search(r'\{[\s\S]*"rubric_scores"[\s\S]*\}', text)
        if obj_match:
            json_match = obj_match.group(0)

    if not json_match:
        # Fallback: all zeros
        return [RubricScore(rubric_name=r["name"], score=0.0, weight=r["weight"], reasoning="Failed to parse eval result") for r in rubrics], "Failed to parse eval result"

    try:
        parsed = json.loads(json_match)
    except json.JSONDecodeError:
        return [RubricScore(rubric_name=r["name"], score=0.0, weight=r["weight"], reasoning="JSON parse error") for r in rubrics], "JSON parse error"

    rubric_scores = []
    scored_names = set()

    for rs in parsed.get("rubric_scores", []):
        name = rs.get("rubric_name", "")
        score = max(0.0, min(1.0, float(rs.get("score", 0.0))))
        weight = rs.get("weight", 0.25)
        reasoning = rs.get("reasoning", "")
        rubric_scores.append(RubricScore(rubric_name=name, score=score, weight=weight, reasoning=reasoning))
        scored_names.add(name)

    # Fill missing rubrics with 0
    for r in rubrics:
        if r["name"] not in scored_names:
            rubric_scores.append(RubricScore(rubric_name=r["name"], score=0.0, weight=r["weight"], reasoning="Not scored by grader"))

    grader_reasoning = parsed.get("grader_reasoning", "")
    return rubric_scores, grader_reasoning
