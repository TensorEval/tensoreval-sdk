"""Dataset loading for TensorEval.

Supports JSONL files, Python dicts, and HuggingFace datasets.

Usage:
    ds = Dataset.load_from_file("tasks.jsonl")
    ds = Dataset.from_list([{"query": "...", "reference_answer": "..."}])
    ds = Dataset.from_huggingface("gsm8k", split="test", n=10)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from tensoreval.types import Rubric, Sample


class Dataset:
    """Collection of evaluation samples.

    Supports iteration, indexing, and loading from multiple sources.
    Field names are flexible — accepts ``query``/``input``/``question``
    for input, ``reference_answer``/``target``/``answer`` for target.

    Attributes:
        samples: List of :class:`Sample` objects.
        name: Optional dataset name.
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
        return f"Dataset(name={self.name!r}, samples={len(self.samples)})"

    @classmethod
    def load_from_file(cls, path: str | Path, name: str = "") -> "Dataset":
        """Load samples from a JSONL file.

        Each line should be a JSON object with at minimum a ``query`` field.
        Optional fields: ``reference_answer``, ``rubrics``, ``metadata``, ``id``.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        samples: list[Sample] = []
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                samples.append(_row_to_sample(row, i))

        return cls(samples, name=name or path.stem)

    @classmethod
    def from_list(cls, rows: list[dict[str, Any]], name: str = "") -> "Dataset":
        """Create from a list of dicts.

        Each dict should have at minimum a ``query`` field.
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
    ) -> "Dataset":
        """Load from a HuggingFace dataset.

        Requires: ``pip install tensoreval[datasets]``
        """
        from tensoreval.utils import load_example_dataset

        dataset = load_example_dataset(dataset_name, split=split, n=n, seed=seed)
        samples = [
            Sample(
                input=row.get("question", row.get("input", "")),
                target=row.get("answer", ""),
                id=f"q_{i + 1}",
            )
            for i, row in enumerate(dataset)
        ]
        return cls(samples, name=name or dataset_name)


def _row_to_sample(row: dict[str, Any], index: int) -> Sample:
    """Convert a dict row to a Sample, handling flexible field names."""
    input_text = row.get("query", row.get("input", row.get("question", "")))
    if not input_text:
        raise ValueError(f"Row {index} missing 'query' field")

    target = row.get("reference_answer", row.get("target", row.get("answer", "")))

    raw_rubrics = row.get("rubrics", [])
    rubrics = [Rubric.from_dict(r) for r in raw_rubrics] if raw_rubrics else []

    metadata = dict(row.get("metadata", row.get("info", {})) or {})
    for key in ("category", "difficulty"):
        if key in row and key not in metadata:
            metadata[key] = row[key]

    return Sample(
        input=input_text,
        target=target,
        id=row.get("id", f"q_{index + 1}"),
        rubrics=rubrics,
        metadata=metadata,
    )
