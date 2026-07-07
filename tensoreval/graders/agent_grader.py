"""Vercel AI Gateway grader for TensorEval.

Uses Vercel AI Gateway through the OpenAI Python SDK. Live grading uses the
Responses API with the hosted ``web_search`` tool so the judge can verify
current facts from the web.

Sources:
- https://vercel.com/docs/ai-gateway/sdks-and-apis/python
- https://vercel.com/docs/ai-gateway/sdks-and-apis/openai-chat-completions
- https://vercel.com/docs/ai-gateway/models-and-providers/web-search
- https://developers.openai.com/api/docs/guides/tools-web-search
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any

from tensoreval.enums import GraderType
from tensoreval.graders.base import Grader


PASS_THRESHOLD = 0.8
DEFAULT_GATEWAY_BASE_URL = "https://ai-gateway.vercel.sh/v1"
DEFAULT_MODEL = "openai/gpt-5.5"


EVAL_RESULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "rubric_scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "rubric_name": {"type": "string"},
                    "score": {"type": "number"},
                    "weight": {"type": "number"},
                    "reasoning": {"type": "string"},
                },
                "required": ["rubric_name", "score", "weight", "reasoning"],
                "additionalProperties": False,
            },
        },
        "grader_reasoning": {"type": "string"},
    },
    "required": ["rubric_scores", "grader_reasoning"],
    "additionalProperties": False,
}


def _json_schema_unsupported(exc: Exception) -> bool:
    message = str(exc).lower()
    return "json_schema" in message and ("unsupported" in message or "not supported" in message)


def _chat_response_format(kind: str) -> dict[str, Any]:
    if kind == "json_schema":
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "tensoreval_grading_result",
                "schema": EVAL_RESULT_SCHEMA,
            },
        }
    return {"type": "json_object"}


def _responses_text_format(kind: str) -> dict[str, Any]:
    if kind == "json_schema":
        return {
            "format": {
                "type": "json_schema",
                "name": "tensoreval_grading_result",
                "schema": EVAL_RESULT_SCHEMA,
            }
        }
    return {"format": {"type": "json_object"}}


class VercelAgentGrader(Grader):
    """Web-aware rubric grader powered by Vercel AI Gateway.

    Args:
        model: AI Gateway model ID, for example ``openai/gpt-5.5``.
        api_key: AI Gateway API key. Defaults to ``AI_GATEWAY_API_KEY``, then
            ``VERCEL_OIDC_TOKEN``, then ``TENSOREVAL_MODEL_API_KEY``.
        base_url: AI Gateway OpenAI-compatible base URL.
        fallback_on_error: If true, use local reference-answer matching when
            the Gateway call fails.
        timeout: Gateway request timeout in seconds.
        use_web_search: If true, use the Responses API hosted ``web_search``
            tool for live verification.
        search_context_size: Context size for OpenAI hosted web search.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        fallback_on_error: bool = False,
        timeout: float = 60.0,
        use_web_search: bool = True,
        search_context_size: str = "medium",
    ):
        super().__init__(GraderType.AGENT)
        self.model = model or os.environ.get("TENSOREVAL_MODEL_NAME") or DEFAULT_MODEL
        self.api_key = (
            api_key
            or os.environ.get("AI_GATEWAY_API_KEY")
            or os.environ.get("VERCEL_OIDC_TOKEN")
            or os.environ.get("TENSOREVAL_MODEL_API_KEY")
        )
        self.base_url = base_url or os.environ.get("AI_GATEWAY_BASE_URL") or DEFAULT_GATEWAY_BASE_URL
        self.fallback_on_error = fallback_on_error
        self.timeout = timeout
        self.use_web_search = use_web_search
        self.search_context_size = search_context_size
        self.last_result: dict[str, Any] | None = None

    async def score(self, state: dict[str, Any], **kwargs: Any) -> float:
        """Score one response and return its weighted score."""
        completion = state.get("completion", [])
        answer = str(state.get("answer", ""))
        query_text = _extract_query(state)
        response = _extract_response(completion)

        if not completion:
            return 0.0

        raw_rubrics = state.get("info", {}).get("rubrics", [])
        rubrics = _normalize_rubrics(raw_rubrics, answer)
        tools = state.get("tools") or state.get("info", {}).get("mcp_tools", [])
        tool_registry = state.get("tool_registry")

        if self.fallback_on_error and not raw_rubrics:
            score = _simple_score(answer, response)
            self.last_result = _fallback_result(rubrics, score, "no rubrics provided")
            state["grader_result"] = self.last_result
            return score

        try:
            result = await self._evaluate_with_gateway(query_text, answer, response, rubrics, tools, tool_registry, state.get("info", {}).get("tool_trace", []))
            normalized = _normalize_eval_result(result, rubrics)
            self.last_result = normalized
            state["grader_result"] = normalized
            return float(normalized["weighted_score"])
        except Exception as exc:
            if self.fallback_on_error:
                score = _simple_score(answer, response)
                self.last_result = _fallback_result(rubrics, score, str(exc))
                state["grader_result"] = self.last_result
                return score
            raise RuntimeError(f"VercelAgentGrader failed: {exc}") from exc

    async def _evaluate_with_gateway(
        self,
        query_text: str,
        answer: str,
        response: str,
        rubrics: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_registry: Any = None,
        tool_trace: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Call Vercel AI Gateway and parse the structured judge result."""
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=self.api_key or "dummy",
            base_url=self.base_url,
            timeout=self.timeout,
        )
        system_prompt = _build_system_prompt(rubrics, bool(answer), self.use_web_search)
        user_prompt = _build_user_prompt(query_text, answer, response, rubrics, tool_trace or [])
        response_tools = _to_responses_tools(tools or [], self.use_web_search, self.search_context_size)

        if response_tools:
            # Vercel AI Gateway supports the OpenAI SDK and Responses API.
            # Hosted web_search and local/MCP function tools are both enabled here.
            return await self._evaluate_with_responses_tools(
                client=client,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                tools=response_tools,
                tool_registry=tool_registry,
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            result = await asyncio.wait_for(
                client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format=_chat_response_format("json_schema"),
                    max_tokens=2000,
                ),
                timeout=self.timeout,
            )
        except Exception as exc:
            if not _json_schema_unsupported(exc):
                raise
            result = await asyncio.wait_for(
                client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format=_chat_response_format("json_object"),
                    max_tokens=2000,
                ),
                timeout=self.timeout,
            )
        return _parse_eval_result(result.choices[0].message.content or "")

    async def _evaluate_with_responses_tools(
        self,
        client: Any,
        system_prompt: str,
        user_prompt: str,
        tools: list[dict[str, Any]],
        tool_registry: Any = None,
    ) -> dict[str, Any]:
        """Run a Responses API loop, executing local/MCP function calls."""
        text_format = _responses_text_format("json_schema")
        try:
            result = await asyncio.wait_for(
                client.responses.create(
                    model=self.model,
                    instructions=system_prompt,
                    input=user_prompt,
                    tools=tools,
                    tool_choice="auto",
                    max_output_tokens=2000,
                    text=text_format,
                ),
                timeout=self.timeout,
            )
        except Exception as exc:
            if not _json_schema_unsupported(exc):
                raise
            text_format = _responses_text_format("json_object")
            result = await asyncio.wait_for(
                client.responses.create(
                    model=self.model,
                    instructions=system_prompt,
                    input=user_prompt,
                    tools=tools,
                    tool_choice="auto",
                    max_output_tokens=2000,
                    text=text_format,
                ),
                timeout=self.timeout,
            )

        for _ in range(10):
            calls = _response_function_calls(result)
            if not calls:
                return _parse_eval_result(_response_output_text(result))

            outputs = []
            for call in calls:
                tool_result = await _execute_grader_tool_call(call, tool_registry)
                outputs.append({
                    "type": "function_call_output",
                    "call_id": call["call_id"],
                    "output": json.dumps(tool_result, default=str),
                })

            kwargs = {
                "model": self.model,
                "input": outputs,
                "tools": tools,
                "tool_choice": "auto",
                "max_output_tokens": 2000,
                "text": text_format,
            }
            previous_response_id = getattr(result, "id", None)
            if previous_response_id:
                kwargs["previous_response_id"] = previous_response_id
            result = await asyncio.wait_for(
                client.responses.create(**kwargs),
                timeout=self.timeout,
            )

        return _parse_eval_result(_response_output_text(result))


AgentGrader = VercelAgentGrader


def _to_responses_tools(
    tools: list[dict[str, Any]],
    use_web_search: bool,
    search_context_size: str,
) -> list[dict[str, Any]]:
    """Convert Chat Completions function tools to Responses API tools."""
    response_tools: list[dict[str, Any]] = []
    if use_web_search:
        response_tools.append({"type": "web_search", "search_context_size": search_context_size})

    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = fn.get("name")
        if not name:
            continue
        response_tools.append({
            "type": "function",
            "name": name,
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
        })
    return response_tools


def _response_function_calls(response: Any) -> list[dict[str, Any]]:
    """Extract function calls from an OpenAI Responses API response object."""
    calls: list[dict[str, Any]] = []
    for item in getattr(response, "output", []) or []:
        item_type = _obj_get(item, "type")
        if item_type != "function_call":
            continue
        name = _obj_get(item, "name")
        call_id = _obj_get(item, "call_id") or _obj_get(item, "id")
        arguments = _obj_get(item, "arguments") or "{}"
        if name and call_id:
            calls.append({"name": name, "call_id": call_id, "arguments": _parse_tool_arguments(arguments)})
    return calls


async def _execute_grader_tool_call(call: dict[str, Any], tool_registry: Any = None) -> Any:
    """Execute one grader-side local/MCP function call."""
    if not tool_registry:
        return {"error": f"No registry available to execute tool: {call.get('name', '')}"}
    try:
        return await tool_registry.call_tool_by_name(call["name"], call.get("arguments", {}))
    except Exception as exc:
        return {"error": str(exc)}


def _parse_tool_arguments(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    try:
        parsed = json.loads(arguments or "{}")
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _obj_get(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _extract_query(state: dict[str, Any]) -> str:
    query_text = state.get("query", "")
    if query_text:
        return str(query_text)
    prompt = state.get("prompt", [])
    if isinstance(prompt, list) and prompt:
        last = prompt[-1]
        return str(last.get("content", "")) if isinstance(last, dict) else str(last)
    return ""


def _extract_response(completion: Any) -> str:
    if not completion:
        return ""
    last = completion[-1]
    if isinstance(last, dict):
        return str(last.get("content", ""))
    return str(getattr(last, "content", ""))


def _normalize_rubrics(raw: Any, answer: str = "") -> list[dict[str, Any]]:
    if not raw:
        criteria = "Response should correctly answer the query."
        if answer:
            criteria = "Response should match the reference answer with equivalent meaning."
        return [{"name": "correctness", "criteria": criteria, "weight": 1.0}]

    rubrics: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "rubric")
        criteria = str(item.get("criteria") or item.get("rubric") or "")
        if criteria:
            rubrics.append({"name": name, "criteria": criteria, "weight": float(item.get("weight", 1.0))})

    if not rubrics:
        return _normalize_rubrics([], answer)

    total = sum(r["weight"] for r in rubrics)
    if total > 0:
        for rubric in rubrics:
            rubric["weight"] = rubric["weight"] / total
    return rubrics


def _build_system_prompt(rubrics: list[dict[str, Any]], has_reference: bool, use_web_search: bool) -> str:
    rubric_instructions = "\n\n".join(
        f"### {idx}. {rubric['name']} (weight: {rubric['weight']})\n"
        f"{rubric['criteria']}\n\n"
        "Score bands:\n"
        "- 1.0: Fully satisfies the rubric\n"
        "- 0.8-0.9: Mostly satisfies, minor issues only\n"
        "- 0.5-0.7: Partially satisfies, significant gaps\n"
        "- 0.2-0.4: Barely satisfies, major issues\n"
        "- 0.0-0.1: Does not satisfy at all"
        for idx, rubric in enumerate(rubrics, start=1)
    )
    web_instruction = (
        "- Use web search for current facts, external claims, links, prices, dates, or any fact that may have changed.\n"
        if use_web_search
        else ""
    )
    reference_text = " and a reference answer" if has_reference else ""

    return f"""You are a rigorous AI agent evaluator. Score an agent's response against rubrics{reference_text}.

Verification rules:
{web_instruction}- Verify substantive claims before scoring.
- Score substance over style. Do not penalize harmless formatting differences.
- Empty or error responses score 0 for all rubrics.
- If a reference answer is provided, use it as ground truth unless a rubric says otherwise.

Rubrics:
{rubric_instructions}

Return exactly one JSON object matching this shape:
{{
  "rubric_scores": [
    {{"rubric_name": "...", "score": 0.85, "weight": 1.0, "reasoning": "..."}}
  ],
  "grader_reasoning": "One paragraph summary of what you verified"
}}"""


def _build_user_prompt(query_text: str, answer: str, response: str, rubrics: list[dict[str, Any]], tool_trace: list[dict[str, Any]] | None = None) -> str:
    rubric_list = "\n".join(
        f"{idx}. {rubric['name']} (weight: {rubric['weight']}): {rubric['criteria']}"
        for idx, rubric in enumerate(rubrics, start=1)
    )
    reference = answer if answer else "No reference answer provided. Evaluate against rubrics."
    response_text = response if response else "[EMPTY - agent returned no response]"
    trace_text = json.dumps(tool_trace or [], indent=2, default=str)
    return f"""## Query
{query_text}

## Reference Answer
{reference}

## Rubrics
{rubric_list}

## Agent Response
{response_text}

## Tool Trace
{trace_text}

Evaluate this response. Use the tool trace as evidence of actions the agent actually took. Check the web when needed, then return the structured JSON result."""


def _response_output_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text)

    parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        content_items = item.get("content", []) if isinstance(item, dict) else getattr(item, "content", [])
        for content in content_items or []:
            text = content.get("text") if isinstance(content, dict) else getattr(content, "text", None)
            if text:
                parts.append(str(text))
    return "\n".join(parts)


def _parse_eval_result(text: str) -> dict[str, Any]:
    try:
        return json.loads(text.strip())
    except (json.JSONDecodeError, AttributeError):
        pass

    fence_regex = re.compile(r"```json\s*\n([\s\S]*?)```", re.IGNORECASE)
    matches = fence_regex.findall(text)
    for candidate in reversed(matches):
        try:
            return json.loads(candidate.strip())
        except json.JSONDecodeError:
            pass

    raw_match = re.search(r'\{[\s\S]*"rubric_scores"[\s\S]*\}', text)
    if raw_match:
        return json.loads(raw_match.group())
    raise ValueError("Failed to parse Vercel grader JSON output")


def _normalize_eval_result(obj: dict[str, Any], rubrics: list[dict[str, Any]]) -> dict[str, Any]:
    rubric_scores: list[dict[str, Any]] = []
    raw_scores = obj.get("rubric_scores", [])
    if isinstance(raw_scores, list):
        for score in raw_scores:
            if not isinstance(score, dict):
                continue
            name = str(score.get("rubric_name") or score.get("name") or "")
            if not name:
                continue
            rubric_scores.append({
                "rubric_name": name,
                "score": _to_score(score.get("score")),
                "weight": _find_weight(rubrics, name),
                "reasoning": str(score.get("reasoning") or score.get("reason") or ""),
            })

    scored_names = {score["rubric_name"] for score in rubric_scores}
    for rubric in rubrics:
        if rubric["name"] not in scored_names:
            rubric_scores.append({
                "rubric_name": rubric["name"],
                "score": 0.0,
                "weight": rubric["weight"],
                "reasoning": "Not scored by grader",
            })

    weighted_score = sum(score["score"] * score["weight"] for score in rubric_scores)
    return {
        "rubric_scores": rubric_scores,
        "weighted_score": round(max(0.0, min(1.0, weighted_score)), 3),
        "passed": weighted_score >= PASS_THRESHOLD,
        "grader_reasoning": str(obj.get("grader_reasoning", "")),
    }


def _find_weight(rubrics: list[dict[str, Any]], rubric_name: str) -> float:
    for rubric in rubrics:
        if rubric["name"] == rubric_name:
            return float(rubric["weight"])
    return 1.0 / len(rubrics) if rubrics else 1.0


def _to_score(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _simple_score(answer: str, response: str) -> float:
    if not answer or not response:
        return 0.0
    answer_nums = set(re.findall(r"-?\d+\.?\d*", answer))
    response_nums = set(re.findall(r"-?\d+\.?\d*", response))
    if answer_nums and answer_nums.issubset(response_nums):
        return 1.0
    return 1.0 if answer.lower().strip() in response.lower().strip() else 0.0


def _fallback_result(rubrics: list[dict[str, Any]], score: float, reason: str) -> dict[str, Any]:
    return {
        "rubric_scores": [
            {
                "rubric_name": rubric["name"],
                "score": score,
                "weight": rubric["weight"],
                "reasoning": f"Fallback reference-answer match. Gateway grader unavailable: {reason}",
            }
            for rubric in rubrics
        ],
        "weighted_score": score,
        "passed": score >= PASS_THRESHOLD,
        "grader_reasoning": "Fallback scoring used because Gateway grading failed.",
    }
