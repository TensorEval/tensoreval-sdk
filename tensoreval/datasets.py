"""Dataset loading for TensorEval.

Supports JSONL files, Python dicts, and HuggingFace datasets.

Usage:
    ds = Datasets.load_from_file("tasks.jsonl")
    ds = Datasets.load_from_dict([{"query": "...", "reference_answer": "..."}])
    ds = Datasets.from_huggingface("gsm8k", split="test", n=10)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from tensoreval.types import Rubric, Sample


class Datasets:
    """Collection of evaluation samples.

    Supports iteration, indexing, and loading from multiple sources.
    Field names are flexible — accepts `query`/`input`/`question` for input,
    `reference_answer`/`target`/`answer` for target.
    """

    def __init__(self, samples: list[Sample], name: str = ""):
        self.samples = samples
        self.name = name

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Sample:
        return self.samples[idx]

    def __iter__(self) -> Iterator[Sample]:
        return iter(self.samples)

    def __repr__(self) -> str:
        return f"Datasets(name={self.name!r}, samples={len(self.samples)})"

    @classmethod
    def load_from_file(cls, path: str | Path, name: str = "") -> Datasets:
        """Load samples from a JSONL file.

        Each line should be a JSON object with at minimum a `query` field.
        Optional fields: `reference_answer`, `rubrics`, `metadata`, `id`.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                samples.append(_row_to_sample(row, i))

        return cls(samples, name=name or path.stem)

    @classmethod
    def load_from_dict(cls, rows: list[dict[str, Any]], name: str = "") -> Datasets:
        """Create from a list of dicts.

        Each dict should have at minimum a `query` field.
        """
        if not rows:
            raise ValueError("rows must be a non-empty list of dicts")

        samples = [_row_to_sample(row, i) for i, row in enumerate(rows)]
        return cls(samples, name=name)

    @classmethod
    def from_huggingface(
        cls,
        dataset_name: str,
        split: str | None = None,
        n: int | None = None,
        seed: int = 0,
        name: str = "",
    ) -> Datasets:
        """Load from a HuggingFace dataset.

        Requires: pip install tensoreval[datasets]
        """
        from tensoreval.utils.data_utils import load_example_dataset

        dataset = load_example_dataset(dataset_name, split=split, n=n, seed=seed)
        samples = []
        for i, row in enumerate(dataset):
            samples.append(Sample(
                input=row.get("question", row.get("input", "")),
                target=row.get("answer", ""),
                id=f"q_{i+1}",
            ))
        return cls(samples, name=name or dataset_name)

    def to_dicts(self) -> list[dict[str, Any]]:
        """Convert back to list of dicts."""
        return [
            {
                "query": s.input,
                "reference_answer": s.target,
                "id": s.id,
                "rubrics": [{"name": r.name, "criteria": r.criteria, "weight": r.weight} for r in s.rubrics],
                "metadata": s.metadata,
            }
            for s in self.samples
        ]


def _row_to_sample(row: dict[str, Any], index: int) -> Sample:
    """Convert a dict row to a Sample, handling flexible field names."""
    # Input: query, input, question
    input_text = row.get("query", row.get("input", row.get("question", "")))
    if not input_text:
        raise ValueError(f"Row {index} missing 'query' field")

    # Target: reference_answer, target, answer
    target = row.get("reference_answer", row.get("target", row.get("answer", "")))

    # Rubrics: list of dicts with name/criteria/weight
    raw_rubrics = row.get("rubrics", [])
    rubrics = [Rubric.from_dict(r) for r in raw_rubrics] if raw_rubrics else []

    return Sample(
        input=input_text,
        target=target,
        id=row.get("id", f"q_{index + 1}"),
        rubrics=rubrics,
        metadata=row.get("metadata", row.get("info", {})),
    )
