"""TensorEval platform client — connects the SDK to the hosted backend.

Authenticates with a long-lived API key (generated from the dashboard under
Settings → API Keys) and pushes evaluation results + observability traces so
they appear in the TensorEval dashboard.

Usage:

    from tensoreval.client import TensorEvalClient

    client = TensorEvalClient(api_key="te_...", base_url="http://localhost:4000")

    # Verify the key works
    print(client.whoami())

    # Post an evaluation result
    client.ingest_evaluation(results)

    # Post trace events
    client.ingest_trace("eval-001", events)
"""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from tensoreval.evaluation import EvaluationResult


class TensorEvalClient:
    """HTTP client for the TensorEval backend (API-key authenticated).

    Args:
        api_key:  Long-lived key starting with ``te_``. Falls back to the
                  ``TENSOREVAL_API_KEY`` env var.
        base_url: Engine URL. Falls back to the ``TENSOREVAL_BASE_URL`` env var,
                  then ``http://localhost:4000``.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 10.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("TENSOREVAL_API_KEY", "")
        self.base_url = (base_url or os.environ.get("TENSOREVAL_BASE_URL", "http://localhost:4000")).rstrip("/")
        self.timeout = timeout

        if not self.api_key:
            raise ValueError(
                "API key required. Generate one from the dashboard "
                "(Settings → API Keys) and pass api_key= or set TENSOREVAL_API_KEY."
            )

    # ------------------------------------------------------------------
    # Low-level
    # ------------------------------------------------------------------

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        url = self.base_url + path
        data = json.dumps(body, default=str).encode() if body else None
        req = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            err_body = ""
            try:
                err_body = e.read().decode()
            except Exception:
                pass
            raise TensorEvalError(
                f"HTTP {e.code} from {url}: {err_body}", status=e.code
            ) from None

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------

    def whoami(self) -> dict:
        """Verify the API key is valid. Returns {valid, user_id}."""
        return self._request("GET", "/api/sdk/whoami")

    def ingest_evaluation(
        self,
        result: "EvaluationResult",
        experiment_id: str | None = None,
        metadata: dict | None = None,
    ) -> dict:
        """Push a completed EvaluationResult to the backend.

        Returns {ingested, evaluation_run_id, result_count}.
        """
        summary = result.summary()
        body = {
            "experiment_id": experiment_id,
            "model": summary.model,
            "summary": summary.to_dict(),
            "runs": [
                {
                    "sample_id": r.sample_id,
                    "query": r.query,
                    "answer": r.answer,
                    "response": r.response,
                    "reward": r.reward,
                    "latency_ms": r.latency_ms,
                    "error": r.error,
                }
                for r in result.runs
            ],
            "metadata": metadata or {},
        }
        return self._request("POST", "/api/sdk/evaluations", body)

    def ingest_trace(self, run_name: str, events: list[dict]) -> dict:
        """Push observability trace events to the backend.

        Returns {ingested, trace_id, events}.
        """
        body = {"run_name": run_name, "events": events}
        return self._request("POST", "/api/sdk/traces", body)


class TensorEvalError(Exception):
    """Raised when the backend returns an error."""

    def __init__(self, message: str, status: int = 0) -> None:
        super().__init__(message)
        self.status = status
