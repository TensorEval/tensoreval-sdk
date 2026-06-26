"""Auto-generation of test suites from agent descriptions.

This is TensorEval's unique differentiator — no other SDK does this.
Uses LLM to generate evaluation scenarios and queries from agent metadata.
"""

import asyncio
import json
import re
from typing import Any

from tensoreval.core.sample import Sample
from tensoreval.datasets import Datasets


# ---------------------------------------------------------------------------
# Scenario generation
# ---------------------------------------------------------------------------

SCENARIO_SYSTEM_PROMPT = """You are a senior QA architect designing evaluation scenarios for an AI agent.

## Agent Under Test
- Name: "{agent_name}"
- Description: {agent_description}
{capabilities_section}
## YOUR TASK

Think about ALL the distinct ways a real user would use this agent. Generate a comprehensive list of test scenarios — each representing a different use case, capability, or risk area.

## QUALITY RULES

1. Every scenario must require multi-step thinking.
2. Difficulty distribution: ~25% easy, ~50% medium, ~25% hard.
3. 1-2 scenarios should be adversarial (safety/boundary testing).
4. Categories should be meaningful for THIS agent type.

## OUTPUT FORMAT
Output a JSON array of scenarios:
```json
[
  {{
    "name": "Short scenario name",
    "description": "Rich guidance for query generation (3-5 sentences)",
    "category": "dynamic_category_label",
    "difficulty": "easy|medium|hard",
    "is_adversarial": false
  }}
]
```"""

QUERY_SYSTEM_PROMPT = """You are an expert test engineer generating evaluation queries for an AI agent.

## Agent Under Test
- Name: "{agent_name}"
- Description: {agent_description}

## SCENARIO
**Name:** {scenario_name}
**Description:** {scenario_description}
**Category:** {scenario_category}
**Difficulty:** {scenario_difficulty}

## YOUR TASK

Generate exactly {query_count} evaluation queries for this scenario.

## QUERY QUALITY
- Every query must require multi-step thinking.
- Ask yourself: "Would a real user pay to have an agent do this?"
- Each query needs 3-5 rubrics with weights summing to 1.0.
- Include VERIFICATION INSTRUCTIONS in each rubric.

## OUTPUT FORMAT
```json
{{
  "queries": [
    {{
      "query": "The specific question/task for the agent",
      "rubrics": [
        {{"name": "rubric_name", "rubric": "specific criteria with verification instructions", "weight": 0.4}}
      ],
      "reference_answer": "verified correct answer (omit for open-ended)",
      "category": "{scenario_category}",
      "difficulty": "{scenario_difficulty}"
    }}
  ]
}}
```"""


class AutoGenerator:
    """Auto-generate test suites from agent descriptions.

    Usage:
        datasets = await AutoGenerator.generate(
            agent_name="DataAnalyst",
            agent_description="Analyzes CSV/Excel files to answer questions about data",
            capabilities=["read_file", "execute_code"],
            count=10,
        )
    """

    @staticmethod
    async def generate_async(
        agent_name: str,
        agent_description: str,
        capabilities: list[str] | None = None,
        count: int = 10,
        model: str = "mimo-v2.5-pro",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> Datasets:
        """Generate test queries from agent description.

        Args:
            agent_name: Name of the agent under test.
            agent_description: Description of what the agent does.
            capabilities: List of agent capabilities (e.g., ["read_file", "web_browsing"]).
            count: Number of queries to generate.
            model: Model to use for generation.
            api_key: API key for the model.
            base_url: Base URL for the model API.

        Returns:
            Datasets object with generated samples.
        """
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=api_key or "dummy",
            base_url=base_url or "https://api.openai.com/v1",
        )

        capabilities_section = ""
        if capabilities:
            caps = ", ".join(capabilities)
            capabilities_section = f"\n## Agent Capabilities\nThe agent has: {caps}\n"

        # Phase 1: Generate scenarios
        scenario_prompt = SCENARIO_SYSTEM_PROMPT.format(
            agent_name=agent_name,
            agent_description=agent_description,
            capabilities_section=capabilities_section,
        )

        scenario_response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": scenario_prompt},
                {"role": "user", "content": f"Design evaluation scenarios for the '{agent_name}' agent. Generate enough scenarios for {count} queries."},
            ],
            max_tokens=2000,
            temperature=0.7,
        )

        scenario_text = scenario_response.choices[0].message.content or "[]"
        scenarios = _extract_json_array(scenario_text)

        if not scenarios:
            scenarios = [{"name": "Core capability test", "description": "Test the agent's primary capabilities.", "category": "core", "difficulty": "medium", "is_adversarial": False}]

        # Phase 2: Generate queries from scenarios
        queries = []
        queries_per_scenario = max(1, count // len(scenarios))

        for scenario in scenarios[:count]:
            query_prompt = QUERY_SYSTEM_PROMPT.format(
                agent_name=agent_name,
                agent_description=agent_description,
                scenario_name=scenario.get("name", "test"),
                scenario_description=scenario.get("description", ""),
                scenario_category=scenario.get("category", "general"),
                scenario_difficulty=scenario.get("difficulty", "medium"),
                query_count=queries_per_scenario,
            )

            query_response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": query_prompt},
                    {"role": "user", "content": f"Generate {queries_per_scenario} evaluation queries for the '{scenario.get('name', 'test')}' scenario."},
                ],
                max_tokens=3000,
                temperature=0.7,
            )

            query_text = query_response.choices[0].message.content or '{"queries": []}'
            parsed = _extract_json_object(query_text)
            scenario_queries = parsed.get("queries", [])

            for q in scenario_queries:
                # Normalize rubric weights
                rubrics = q.get("rubrics", [])
                total_weight = sum(r.get("weight", 0.25) for r in rubrics)
                if total_weight > 0:
                    for r in rubrics:
                        r["weight"] = round(r["weight"] / total_weight, 2)

                queries.append({
                    "query": q.get("query", ""),
                    "rubrics": rubrics,
                    "reference_answer": q.get("reference_answer"),
                    "category": q.get("category", scenario.get("category", "general")),
                    "difficulty": q.get("difficulty", scenario.get("difficulty", "medium")),
                })

        # Convert to Datasets
        return Datasets.from_dicts(queries[:count], name=f"auto_{agent_name}")

    @staticmethod
    def generate(
        agent_name: str,
        agent_description: str,
        capabilities: list[str] | None = None,
        count: int = 10,
        model: str = "mimo-v2.5-pro",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> Datasets:
        """Synchronous wrapper for generate_async."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        coro = AutoGenerator.generate_async(
            agent_name=agent_name,
            agent_description=agent_description,
            capabilities=capabilities,
            count=count,
            model=model,
            api_key=api_key,
            base_url=base_url,
        )

        if loop is not None:
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(coro)
        return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_json_array(text: str) -> list:
    """Extract JSON array from text (handles markdown fences)."""
    # Try to find JSON array in code fences
    fence_match = re.search(r'```(?:json)?\s*\n(\[[\s\S]*?\])\s*\n```', text)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find raw JSON array
    array_match = re.search(r'\[[\s\S]*\]', text)
    if array_match:
        try:
            return json.loads(array_match.group(0))
        except json.JSONDecodeError:
            pass

    return []


def _extract_json_object(text: str) -> dict:
    """Extract JSON object from text (handles markdown fences)."""
    # Try to find JSON object in code fences
    fence_match = re.search(r'```(?:json)?\s*\n(\{[\s\S]*?\})\s*\n```', text)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find raw JSON object
    obj_match = re.search(r'\{[\s\S]*\}', text)
    if obj_match:
        try:
            return json.loads(obj_match.group(0))
        except json.JSONDecodeError:
            pass

    return {}
