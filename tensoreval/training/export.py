"""Training data export — SFT, DPO, and hybrid distillation.

Ported from TensorEval TypeScript engine training-connector.ts.
Exports evaluation results as training data for RL fine-tuning.
"""

import json
from typing import Any


class TrainingDataExporter:
    """Export evaluation results as training data.

    Supports:
    - SFT format: messages pairs with rewards
    - DPO format: prompt/chosen/rejected pairs
    - Hybrid distillation: teacher generates correct responses for failures

    Usage:
        exporter = TrainingDataExporter()
        sft_data = exporter.export_sft(eval_results, teacher_model="mimo-v2.5-pro")
        dpo_data = exporter.export_dpo(eval_results, teacher_model="mimo-v2.5-pro")
        exporter.save_jsonl(sft_data, "training_data.jsonl")
    """

    def export_sft(
        self,
        results: list[dict[str, Any]],
        teacher_model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> dict[str, Any]:
        """Export as SFT training format.

        Passing traces: keep agent's own responses.
        Failing traces: teacher generates correct responses (if teacher_model provided).

        Args:
            results: Evaluation results (list of run dicts).
            teacher_model: Model to generate corrections for failures.
            api_key: API key for teacher model.
            base_url: Base URL for teacher model.

        Returns:
            Dict with 'examples' list and 'stats' dict.
        """
        passing = [r for r in results if r.get("reward", 0) >= 0.8]
        failing = [r for r in results if r.get("reward", 0) < 0.8]

        examples = []

        # Add passing traces as-is
        for r in passing:
            messages = self._extract_messages(r)
            if messages:
                examples.append({"messages": messages})

        teacher_generated = 0

        # Generate teacher responses for failures
        if teacher_model and failing:
            for r in failing:
                query = self._extract_query(r)
                if query:
                    teacher_response = self._call_teacher(
                        query, teacher_model, api_key, base_url
                    )
                    if teacher_response:
                        examples.append({
                            "messages": [
                                {"role": "user", "content": query},
                                {"role": "assistant", "content": teacher_response},
                            ]
                        })
                        teacher_generated += 1

        return {
            "format": "sft",
            "examples": examples,
            "stats": {
                "agent_passing": len(passing),
                "teacher_generated": teacher_generated,
                "total": len(examples),
            },
        }

    def export_dpo(
        self,
        results: list[dict[str, Any]],
        teacher_model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> dict[str, Any]:
        """Export as DPO training format.

        Creates prompt/chosen/rejected pairs from passing and failing traces.

        Args:
            results: Evaluation results.
            teacher_model: Model to generate better responses for failures.
            api_key: API key for teacher model.
            base_url: Base URL for teacher model.

        Returns:
            Dict with 'examples' list (each has prompt, chosen, rejected).
        """
        passing = [r for r in results if r.get("reward", 0) >= 0.8]
        failing = [r for r in results if r.get("reward", 0) < 0.8]

        pairs = []

        # Match passing vs failing for same query
        pass_by_query = {self._extract_query(r): r for r in passing if self._extract_query(r)}
        fail_by_query = {self._extract_query(r): r for r in failing if self._extract_query(r)}

        for query, pass_r in pass_by_query.items():
            if query in fail_by_query:
                fail_r = fail_by_query[query]
                pass_response = self._extract_response(pass_r)
                fail_response = self._extract_response(fail_r)
                if pass_response and fail_response:
                    pairs.append({
                        "prompt": query,
                        "chosen": pass_response,
                        "rejected": fail_response,
                    })

        # Add teacher-generated pairs
        teacher_generated = 0
        if teacher_model:
            for r in failing:
                query = self._extract_query(r)
                if query:
                    teacher_response = self._call_teacher(
                        query, teacher_model, api_key, base_url
                    )
                    agent_response = self._extract_response(r)
                    if teacher_response and agent_response:
                        pairs.append({
                            "prompt": query,
                            "chosen": teacher_response,
                            "rejected": agent_response,
                        })
                        teacher_generated += 1

        return {
            "format": "dpo",
            "examples": pairs,
            "stats": {
                "matched_pairs": len(pairs) - teacher_generated,
                "teacher_generated": teacher_generated,
                "total": len(pairs),
            },
        }

    def save_jsonl(self, data: dict[str, Any], path: str):
        """Save training data as JSONL file."""
        with open(path, "w", encoding="utf-8") as f:
            for example in data["examples"]:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")

    def _extract_messages(self, result: dict) -> list[dict] | None:
        """Extract messages from an evaluation result."""
        completion = result.get("completion", [])
        if not completion:
            return None

        messages = []
        for msg in completion:
            if isinstance(msg, dict):
                messages.append({
                    "role": msg.get("role", "assistant"),
                    "content": msg.get("content", ""),
                })
            else:
                content = str(getattr(msg, "content", ""))
                role = getattr(msg, "role", "assistant")
                messages.append({"role": role, "content": content})

        return messages if messages else None

    def _extract_query(self, result: dict) -> str | None:
        """Extract the query/prompt from an evaluation result."""
        prompt = result.get("prompt", [])
        if isinstance(prompt, list) and prompt:
            last = prompt[-1]
            if isinstance(last, dict):
                return last.get("content", "")
            return str(getattr(last, "content", ""))
        return result.get("query", None)

    def _extract_response(self, result: dict) -> str | None:
        """Extract the response from an evaluation result."""
        completion = result.get("completion", [])
        if completion:
            last = completion[-1]
            if isinstance(last, dict):
                return last.get("content", "")
            return str(getattr(last, "content", ""))
        return None

    def _call_teacher(
        self,
        query: str,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> str | None:
        """Call teacher model to generate a correct response."""
        try:
            is_anthropic = base_url and "anthropic" in base_url.lower()

            if is_anthropic:
                import anthropic
                client = anthropic.AsyncAnthropic(api_key=api_key, base_url=base_url)

                async def _call():
                    response = await client.messages.create(
                        model=model,
                        max_tokens=1024,
                        messages=[{"role": "user", "content": query}],
                    )
                    return response.content[0].text if response.content else None

                import asyncio
                return asyncio.run(_call())
            else:
                from openai import OpenAI
                client = OpenAI(api_key=api_key, base_url=base_url)
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": query}],
                    max_tokens=1024,
                )
                return response.choices[0].message.content
        except Exception:
            return None
