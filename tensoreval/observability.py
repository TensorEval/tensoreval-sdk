"""Observability — opt-in tracing for TensorEval runs.

Pattern taken from LangSmith (RunTree, @traceable decorator, per-sink dispatch)
but with zero external dependencies. Records:

  * per-call spans (eval, agent, mcp, judge, generate)
  * per-span token estimate + wall latency
  * per-span cost estimate (when a model+price table has an entry)
  * per-run summary (avg reward, pass rate, gap list)

Sinks (all opt-in, all independent):

  TENSOREVAL_TRACE=1              stdout (human-readable)
  TENSOREVAL_TRACE_JSONL=path     append-only JSONL
  TENSOREVAL_TRACE_DASHBOARD=url  future: POST events to a TensorEval dashboard
                                  endpoint (requires TENSOREVAL_API_KEY)
  tracer = ObservabilityTracer(sinks=[...])  explicit sink list

Usage:

    from tensoreval.observability import observe, observe_run

    @observe("agent_call", kind="agent")
    async def call_agent(query, model="gpt-4o"): ...

    async with observe_run("eval-001") as run:
        result = await call_agent("refund order O1234")
        run.set_summary(avg_reward=0.85)
        run.log_gap("agent", "Used wrong policy for gift subscriptions")
"""

from __future__ import annotations

import contextvars
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Cost table (USD per 1M tokens). Override per-call via span metadata.
# ---------------------------------------------------------------------------
DEFAULT_PRICING: dict[str, tuple[float, float]] = {
    # model -> (input_per_1m, output_per_1m)
    "gpt-4o":            (2.50, 10.00),
    "gpt-4o-mini":       (0.15,  0.60),
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-haiku-4-5":  (0.80,  4.00),
    "mimo-v2.5-pro":     (0.30,  0.90),
}

_CHARS_PER_TOKEN = 4


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _estimate_cost(model: str, in_text: str, out_text: str) -> float:
    pricing = DEFAULT_PRICING.get(model)
    if not pricing:
        return 0.0
    in_per_m, out_per_m = pricing
    in_tok = _estimate_tokens(in_text)
    out_tok = _estimate_tokens(out_text)
    return (in_tok / 1_000_000) * in_per_m + (out_tok / 1_000_000) * out_per_m


# ---------------------------------------------------------------------------
# Span  —  one traced call. Parent/child via parent_id (LangSmith RunTree pattern).
# ---------------------------------------------------------------------------
@dataclass
class Span:
    name: str
    kind: str                       # eval | agent | mcp | judge | generate | llm
    run_id: str
    parent_id: str | None = None
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    start: float = field(default_factory=time.time)
    end: float | None = None
    latency_ms: float = 0.0
    status: str = "ok"              # ok | error
    error: str | None = None
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def finish(self, error: Exception | None = None, **meta) -> None:
        self.end = time.time()
        self.latency_ms = (self.end - self.start) * 1000
        if error is not None:
            self.status = "error"
            self.error = repr(error)
        self.metadata.update(meta)

    def to_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "run_id": self.run_id,
            "name": self.name,
            "kind": self.kind,
            "status": self.status,
            "error": self.error,
            "latency_ms": round(self.latency_ms, 2),
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# RunContext  —  root of one trace (one evaluation run).
# ---------------------------------------------------------------------------
@dataclass
class RunContext:
    run_id: str
    name: str
    start: float = field(default_factory=time.time)
    end: float | None = None
    spans: list[Span] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    gaps: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_span(self, span: Span) -> None:
        self.spans.append(span)

    def log_gap(self, area: str, finding: str, severity: str = "medium", **meta) -> None:
        self.gaps.append({
            "area": area, "finding": finding, "severity": severity, **meta,
        })

    def total_cost(self) -> float:
        return sum(s.cost_usd for s in self.spans)

    def total_tokens(self) -> tuple[int, int]:
        return (
            sum(s.input_tokens for s in self.spans),
            sum(s.output_tokens for s in self.spans),
        )

    def set_summary(self, **kw) -> None:
        self.summary.update(kw)

    def finish(self) -> None:
        self.end = time.time()
        self.metadata.setdefault("duration_ms", (self.end - self.start) * 1000)


# ---------------------------------------------------------------------------
# Active context  (contextvars — safe across asyncio tasks)
# ---------------------------------------------------------------------------
_active_run: contextvars.ContextVar[RunContext | None] = contextvars.ContextVar(
    "tensoreval_active_run", default=None
)
_active_span: contextvars.ContextVar[Span | None] = contextvars.ContextVar(
    "tensoreval_active_span", default=None
)


def current_run() -> RunContext | None:
    return _active_run.get()


def current_span() -> Span | None:
    return _active_span.get()


# ---------------------------------------------------------------------------
# Tracer
# ---------------------------------------------------------------------------
class ObservabilityTracer:
    """Owns a list of sinks. Sinks are callables taking a serialised event dict.
    They should never raise — the tracer catches exceptions from each sink.

    auto_env reads TENSOREVAL_TRACE, TENSOREVAL_TRACE_JSONL, TENSOREVAL_TRACE_DASHBOARD
    from the environment at construction time.
    """

    def __init__(self, *, sinks: list[Callable[[dict], None]] | None = None,
                 auto_env: bool = True) -> None:
        self.sinks: list[Callable[[dict], None]] = list(sinks or [])
        if auto_env:
            if os.environ.get("TENSOREVAL_TRACE") == "1":
                self.sinks.append(_stdout_sink)
            jsonl = os.environ.get("TENSOREVAL_TRACE_JSONL")
            if jsonl:
                self.sinks.append(_make_jsonl_sink(jsonl))
            dashboard = os.environ.get("TENSOREVAL_TRACE_DASHBOARD")
            if dashboard:
                self.sinks.append(_make_dashboard_sink(dashboard))

    def emit(self, event: dict) -> None:
        for sink in self.sinks:
            try:
                sink(event)
            except Exception as e:
                print(f"[tensoreval.trace] sink error: {e}", file=sys.stderr)

    def begin_run(self, name: str, **meta) -> RunContext:
        run = RunContext(run_id=uuid.uuid4().hex[:16], name=name, metadata=dict(meta))
        _active_run.set(run)
        self.emit({"event": "run_start", **run.__dict__["metadata"],
                    "run_id": run.run_id, "name": run.name})
        return run

    def end_run(self, run: RunContext) -> None:
        run.finish()
        in_tok, out_tok = run.total_tokens()
        self.emit({
            "event": "run_end",
            "run_id": run.run_id,
            "name": run.name,
            "duration_ms": round((run.end - run.start) * 1000, 2),
            "spans": len(run.spans),
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "cost_usd": round(run.total_cost(), 6),
            "summary": run.summary,
            "gaps": run.gaps,
        })
        if _active_run.get() is run:
            _active_run.set(None)



def get_tracer() -> ObservabilityTracer:
    return _DEFAULT_TRACER


def set_tracer(tracer: ObservabilityTracer) -> None:
    global _DEFAULT_TRACER
    _DEFAULT_TRACER = tracer


# ---------------------------------------------------------------------------
# @observe decorator  —  wraps any async/sync call in a Span.
# LangSmith's @traceable does the same thing, but we own the implementation.
# ---------------------------------------------------------------------------
def observe(name: str, *, kind: str = "custom",
            tracer: ObservabilityTracer | None = None) -> Callable:
    def deco(fn: Callable) -> Callable:
        import functools
        import inspect

        is_coro = inspect.iscoroutinefunction(fn)

        async def async_wrapper(*args, **kwargs):
            t = tracer or _DEFAULT_TRACER
            parent = _active_span.get()
            run = _active_run.get()
            if run is None:
                return await fn(*args, **kwargs)
            span = Span(
                name=name, kind=kind, run_id=run.run_id,
                parent_id=parent.span_id if parent else None,
                model=kwargs.get("model"),
            )
            tok = _active_span.set(span)
            try:
                result = await fn(*args, **kwargs)
                in_text = kwargs.get("input_text", "") or (
                    args[0] if args and isinstance(args[0], str) else ""
                )
                out_text = result if isinstance(result, str) else ""
                span.input_tokens = _estimate_tokens(in_text)
                span.output_tokens = _estimate_tokens(out_text)
                if span.model and out_text:
                    span.cost_usd = _estimate_cost(span.model, in_text, out_text)
                span.finish()
                run.add_span(span)
                t.emit({"event": "span_end", **span.to_dict()})
                return result
            except Exception as e:
                span.finish(error=e)
                run.add_span(span)
                t.emit({"event": "span_end", **span.to_dict()})
                raise
            finally:
                _active_span.reset(tok)

        def sync_wrapper(*args, **kwargs):
            t = tracer or _DEFAULT_TRACER
            parent = _active_span.get()
            run = _active_run.get()
            if run is None:
                return fn(*args, **kwargs)
            span = Span(
                name=name, kind=kind, run_id=run.run_id,
                parent_id=parent.span_id if parent else None,
                model=kwargs.get("model"),
            )
            tok = _active_span.set(span)
            try:
                result = fn(*args, **kwargs)
                in_text = kwargs.get("input_text", "") or (
                    args[0] if args and isinstance(args[0], str) else ""
                )
                out_text = result if isinstance(result, str) else ""
                span.input_tokens = _estimate_tokens(in_text)
                span.output_tokens = _estimate_tokens(out_text)
                if span.model and out_text:
                    span.cost_usd = _estimate_cost(span.model, in_text, out_text)
                span.finish()
                run.add_span(span)
                t.emit({"event": "span_end", **span.to_dict()})
                return result
            except Exception as e:
                span.finish(error=e)
                run.add_span(span)
                t.emit({"event": "span_end", **span.to_dict()})
                raise
            finally:
                _active_span.reset(tok)

        return async_wrapper if is_coro else sync_wrapper
    return deco


# ---------------------------------------------------------------------------
# Sinks
# ---------------------------------------------------------------------------
def _stdout_sink(event: dict) -> None:
    if event.get("event") == "run_start":
        print(f"\n[trace] run start  {event['name']}  id={event['run_id']}",
              file=sys.stderr)
    elif event.get("event") == "span_end":
        kind = event["kind"]
        lat = event["latency_ms"]
        cost = event["cost_usd"]
        model = event.get("model") or "-"
        status = event["status"]
        print(f"  [trace.{kind:7}] {event['name']:30} {lat:7.1f}ms  "
              f"model={model:18}  cost=${cost:.5f}  {status}",
              file=sys.stderr)
    elif event.get("event") == "run_end":
        print(f"[trace] run end    {event['name']}  "
              f"spans={event['spans']}  in_tok={event['input_tokens']}  "
              f"out_tok={event['output_tokens']}  cost=${event['cost_usd']:.5f}  "
              f"dur={event['duration_ms']:.0f}ms",
              file=sys.stderr)


def _make_jsonl_sink(path: str) -> Callable[[dict], None]:
    fh = open(path, "a", encoding="utf-8")

    def sink(event: dict) -> None:
        fh.write(json.dumps(event, default=str) + "\n")
        fh.flush()
    return sink


def _make_dashboard_sink(base_url: str) -> Callable[[dict], None]:
    """Sink that POSTs events to a TensorEval dashboard endpoint.

    Activated by TENSOREVAL_TRACE_DASHBOARD=<base_url>.
    Requires TENSOREVAL_API_KEY to be set in the environment, otherwise
    all events are silently dropped.

    This is a placeholder for the future TensorEval hosted dashboard.
    The POST body is the same JSON dict that every sink receives.
    """
    api_key = os.environ.get("TENSOREVAL_API_KEY", "")
    if not api_key:
        def drop(_event: dict) -> None:
            pass
        return drop

    import urllib.request

    def sink(event: dict) -> None:
        data = json.dumps(event, default=str).encode()
        req = urllib.request.Request(
            base_url.rstrip("/") + "/api/v1/traces",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            urllib.request.urlopen(req, timeout=5.0)
        except Exception:
            pass         # fire-and-forget; dashboard being down must not break eval

    return sink


_DEFAULT_TRACER = ObservabilityTracer()


# ---------------------------------------------------------------------------
# observe_run  —  async context manager for one trace
# ---------------------------------------------------------------------------
class observe_run:
    """`async with observe_run("name") as run: ...` for ad-hoc use."""

    def __init__(self, name: str, **meta) -> None:
        self.name = name
        self.meta = meta
        self._tracer = _DEFAULT_TRACER
        self.run: RunContext | None = None

    async def __aenter__(self) -> RunContext:
        self.run = self._tracer.begin_run(self.name, **self.meta)
        return self.run

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self.run is not None:
            self.run.set_summary(status="error" if exc else "ok")
            self._tracer.end_run(self.run)
