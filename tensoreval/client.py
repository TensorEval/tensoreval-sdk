"""TensorEval dashboard client — push results to the backend in real time.

Activated when TENSOREVAL_API_KEY is set. All calls are best-effort:
errors are silently ignored so dashboard failures never block evaluation.
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any

DEFAULT_BASE_URL = "https://api.tensoreval.com"


class DashboardClient:
    """Thin HTTP client for the TensorEval SDK ingest endpoints."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        self.api_key = api_key or os.environ.get("TENSOREVAL_API_KEY", "")
        self.base_url = (base_url or os.environ.get("TENSOREVAL_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def start_evaluation(self, model: str, total_count: int, name: str | None = None, metadata: dict[str, Any] | None = None) -> str | None:
        """Create an in-progress evaluation run. Returns the run ID, or None on failure."""
        body = {"model": model, "total_count": total_count, "metadata": metadata or {}}
        if name:
            body["name"] = name
        data = self._post("/api/sdk/evaluations/start", body)
        if data and isinstance(data, dict):
            return data.get("evaluation_run_id")
        return None

    def append_result(self, run_id: str, run: dict[str, Any], total_count: int | None = None) -> None:
        """Append a single result to an in-progress evaluation run."""
        body = {"result": run}
        if total_count is not None:
            body["total_count"] = total_count
        self._post(f"/api/sdk/evaluations/{run_id}/results", body)

    def complete_evaluation(self, run_id: str, summary: dict[str, Any] | None = None, failed: bool = False, progress: str | None = None) -> None:
        """Mark an evaluation run as complete."""
        body: dict[str, Any] = {"status": "failed" if failed else "completed"}
        if summary:
            body["summary"] = summary
        if progress:
            body["progress"] = progress
        self._post(f"/api/sdk/evaluations/{run_id}/complete", body)

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any] | None:
        if not self.api_key:
            return None
        try:
            req = urllib.request.Request(
                f"{self.base_url}{path}",
                data=json.dumps(body).encode(),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode())
        except Exception:
            return None
