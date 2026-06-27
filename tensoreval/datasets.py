"""Datasets loader for TensorEval.

Supports loading from JSONL files, Python dicts, or HuggingFace datasets.
"""

import json
from pathlib import Path
from typing import Any


class Sample:
    """A single evaluation sample."""

    def __init__(
        self,
        input: str,
        target: str = "",
        id: str | int | None = None,
        rubrics: list[dict] | None = None,
        metadata: dict | None = None,
    ):
        self.input = input
        self.target = target
        self.id = id
        self.rubrics = rubrics or []
        self.metadata = metadata or {}


class Datasets:
    """Collection of evaluation samples.

    Usage:
        # From JSONL file
        ds = Datasets.load_from_file("tasks.jsonl")

        # From Python dicts
        ds = Datasets.load_from_dict([
            {"query": "What is 2+2?", "reference_answer": "4"},
        ])

        # From HuggingFace
        ds = Datasets.from_huggingface("gsm8k", split="test", n=10)
    """

    def __init__(self, samples: list[Sample], name: str = ""):
        self.samples = samples
        self.name = name

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Sample:
        return self.samples[idx]

    def __iter__(self):
        return iter(self.samples)

    @classmethod
    def load_from_file(cls, path: str | Path, name: str = "") -> "Datasets":
        """Load samples from a JSONL file."""
        path = Path(path)
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                samples.append(Sample(
                    input=row.get("query", row.get("input", row.get("question", ""))),
                    target=row.get("reference_answer", row.get("target", row.get("answer", ""))),
                    id=row.get("id", f"q_{i+1}"),
                    rubrics=row.get("rubrics", []),
                    metadata=row.get("metadata", row.get("info", {})),
                ))
        return cls(samples, name=name or path.stem)

    @classmethod
    def load_from_dict(cls, rows: list[dict], name: str = "") -> "Datasets":
        """Create from a list of dicts."""
        samples = []
        for i, row in enumerate(rows):
            samples.append(Sample(
                input=row.get("query", row.get("input", row.get("question", ""))),
                target=row.get("reference_answer", row.get("target", row.get("answer", ""))),
                id=row.get("id", f"q_{i+1}"),
                rubrics=row.get("rubrics", []),
                metadata=row.get("metadata", row.get("info", {})),
            ))
        return cls(samples, name=name)

    @classmethod
    def from_huggingface(
        cls,
        dataset_name: str,
        split: str | None = None,
        n: int | None = None,
        seed: int = 0,
        name: str = "",
    ) -> "Datasets":
        """Load from a HuggingFace dataset."""
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

    def to_dicts(self) -> list[dict]:
        """Convert to list of dicts."""
        return [
            {
                "query": s.input,
                "reference_answer": s.target,
                "id": s.id,
                "rubrics": s.rubrics,
                "metadata": s.metadata,
            }
            for s in self.samples
        ]
