"""Parser that extracts structured XML tags from model output.

Ported from PrimeIntellect Verifiers (MIT License).
"""

import re
from types import SimpleNamespace
from typing import Any, Callable

from tensoreval.parsers.parser import Parser
from tensoreval.core.types import Messages


class XMLParser(Parser):
    """Parse structured XML tags from model output.

    Each field may be:
      - a string (e.g. "reasoning"): the XML tag is fixed.
      - a tuple of alternatives (e.g. ("code", "answer")): the first element is
        the canonical name used for formatting, and all elements are allowed tags
        when parsing.
    """

    def __init__(
        self,
        fields: list[str | tuple[str, ...]],
        answer_field: str = "answer",
        extract_fn: Callable[[str], str] = lambda x: x,
    ):
        super().__init__(extract_fn=extract_fn)
        self._fields: list[tuple[str, list[str]]] = []
        self.answer_field = answer_field
        seen = set()
        for field in fields:
            if isinstance(field, str):
                canonical = field
                alternatives = [field]
            elif isinstance(field, tuple):
                if not field:
                    raise ValueError("Field tuple cannot be empty.")
                canonical = field[0]
                alternatives = list(field)
            else:
                raise TypeError("Each field must be a string or a tuple of strings.")
            if canonical in seen:
                raise ValueError(f"Duplicate field name: {canonical}")
            seen.add(canonical)
            self._fields.append((canonical, alternatives))

    def parse(self, text: str, strip: bool = True, last: bool = False) -> Any:
        """Parse XML string and return an object with attributes for each tag."""
        results: dict[str, str | None] = {}
        for canonical, alternatives in self._fields:
            for alt in alternatives:
                pattern = rf"<{alt}>\s*(.*?)\s*</{alt}>"
                if last:
                    match = None
                    for match in re.finditer(pattern, text, re.DOTALL):
                        pass
                else:
                    match = re.search(pattern, text, re.DOTALL)
                if match:
                    results[alt] = match.group(1).strip() if strip else match.group(1)
                else:
                    results[alt] = None
        return SimpleNamespace(**results)

    def parse_answer(self, completion: Messages) -> str | None:
        """Extract the last answer from a completion."""
        if isinstance(completion, str):
            parsed = self.parse(completion, last=True)
            if parsed and hasattr(parsed, self.answer_field) and getattr(parsed, self.answer_field) is not None:
                return getattr(parsed, self.answer_field)
        else:
            for msg in reversed(self.get_assistant_messages(completion)):
                content = self._content_to_text(
                    msg.get("content", "") if isinstance(msg, dict) else (msg.content or "")
                )
                parsed = self.parse(content)
                if parsed and hasattr(parsed, self.answer_field) and getattr(parsed, self.answer_field) is not None:
                    return getattr(parsed, self.answer_field)
        return None

    def get_format_str(self) -> str:
        """Return a string describing the expected XML format."""
        format_str = ""
        for field in self._fields:
            if len(field[1]) > 1:
                options = " | ".join(field[1])
                format_str += f"<[ {options} ]>\n...\n</[ {options} ]>\n"
            else:
                format_str += f"<{field[0]}>\n...\n</{field[0]}>\n"
        return format_str.strip()

    def get_format_reward_func(self) -> Callable:
        """Return a reward function that checks XML format compliance."""

        def format_reward_func(completion: Messages):
            model_messages = self.get_assistant_messages(completion)
            if not model_messages:
                return 0.0
            format_scores = []
            for msg in model_messages:
                content = self._content_to_text(
                    msg.get("content", "") if isinstance(msg, dict) else (msg.content or "")
                )
                parsed = self.parse(content)
                parsed_no_strip = self.parse(content, strip=False)
                has_any_field = False
                expected_field_count = len(self._fields)
                present_field_sets = set()
                has_correct_spacing = True
                for i, (canonical, alternatives) in enumerate(self._fields):
                    field_set_present = False
                    for alt in alternatives:
                        if hasattr(parsed, alt) and getattr(parsed, alt) is not None:
                            has_any_field = True
                            field_set_present = True
                            if not (hasattr(parsed_no_strip, alt) and getattr(parsed_no_strip, alt) is not None):
                                has_correct_spacing = False
                        elif content.count(f"<{alt}>") > 0 or content.count(f"</{alt}>") > 0:
                            field_set_present = True
                    if field_set_present:
                        present_field_sets.add(i)
                format_score = 0.0
                first_field_set = self._fields[0][1]
                starts_with_any_field = any(content.strip().startswith(f"<{alt}>") for alt in first_field_set)
                last_field_set = self._fields[-1][1]
                ends_with_any_field = any(content.strip().endswith(f"</{alt}>") for alt in last_field_set)
                if has_any_field:
                    field_set_ratio = len(present_field_sets) / expected_field_count
                    format_score += 0.4 * field_set_ratio
                if has_correct_spacing:
                    format_score += 0.2
                if starts_with_any_field:
                    format_score += 0.2
                if ends_with_any_field:
                    format_score += 0.2
                format_scores.append(format_score)
            return sum(format_scores) / len(format_scores) if format_scores else 0.0

        return format_reward_func

    def get_fields(self) -> list[str]:
        """Return the canonical field names in order."""
        return [canonical for canonical, _ in self._fields]

    def format(self, **kwargs) -> str:
        """Format keyword arguments into an XML string."""
        parts = []
        for canonical, alternatives in self._fields:
            value = None
            if canonical in kwargs:
                value = kwargs[canonical]
            else:
                for alt in alternatives:
                    if alt in kwargs:
                        value = kwargs[alt]
                        break
            if value is None:
                raise ValueError(f"Missing value for field '{canonical}' (allowed: {alternatives})")
            parts.append(f"<{canonical}>\n{value}\n</{canonical}>")
        return "\n".join(parts)
