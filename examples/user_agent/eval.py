"""eval.py — single entrypoint for local eval against Docker stack.

Usage:
    TENSOREVAL_TRACE=1 python eval.py                       # run eval with span tracing
    python eval.py --generate 5                              # add 5 generated queries
    python eval.py --tasks my_tasks.jsonl                    # use a custom task file
    python eval.py --analyze results.json                    # gap analysis on saved results
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import httpx
import tensoreval as te
from tensoreval.observability import observe, observe_run


AGENT_URL = os.environ.get("AGENT_URL", "http://localhost:8000/v1/chat/completions")
MCP_URL   = os.environ.get("MCP_URL",   "http://localhost:9000/mcp")


# ---------------------------------------------------------------------------
# 1.  MCP client
# ---------------------------------------------------------------------------
@observe("mcp_init", kind="mcp")
async def _mcp_init(client: httpx.AsyncClient) -> None:
    await client.post(MCP_URL, json={
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2024-11-05", "capabilities": {}},
    })

@observe("mcp_list_tools", kind="mcp")
async def _mcp_tools(client: httpx.AsyncClient, nid: list[int]) -> list[str]:
    nid[0] += 1
    r = await client.post(MCP_URL, json={
        "jsonrpc": "2.0", "id": nid[0], "method": "tools/list", "params": {},
    })
    return [t["name"] for t in r.json().get("result", {}).get("tools", [])]


# ---------------------------------------------------------------------------
# 2.  Agent call
# ---------------------------------------------------------------------------
@observe("agent", kind="agent")
async def _call_agent(client: httpx.AsyncClient, query: str,
                      system: str | None = None) -> str:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": query})
    r = await client.post(AGENT_URL, json={"messages": msgs}, timeout=30)
    return r.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# 3.  Query generation  (deterministic, no LLM needed)
# ---------------------------------------------------------------------------
QUERY_TEMPLATES = [
    ("refund",  "I want my money back for order O{i}.",       "refund",   "policy"),
    ("cancel",  "Cancel my subscription right now.",          "cancel",   "policy"),
    ("upgrade", "Tell me about the Enterprise plan pricing.", "upgrade",  "knowledge"),
    ("technical", "My {dev} crashes when I open settings.",   "escalate", "diagnosis"),
    ("account", "I can't log in — says wrong password.",      "help",     "responsiveness"),
    ("edge",    "Can I refund a gift subscription?",          "refund",   "policy"),
]

def _generate(n: int) -> te.Datasets:
    devices = ["iPhone", "Android 15", "iPad", "MacBook Air"]
    out = []
    for i in range(n):
        kind, q, ref, _ = QUERY_TEMPLATES[i % len(QUERY_TEMPLATES)]
        q = q.format(i=1000+i, dev=devices[i % len(devices)])
        out.append({"query": q, "reference_answer": ref,
                    "rubrics": [{"name": kind, "criteria": f"Must handle {kind}",
                                 "weight": 1.0}]})
    return te.Datasets.load_from_dict(out, name="generated")


# ---------------------------------------------------------------------------
# 4.  Gap analysis  (deterministic, no LLM needed)
# ---------------------------------------------------------------------------
def _analyze_gaps(runs: list[te.Run]) -> str:
    failed = [r for r in runs if r.reward < 0.8]
    if not failed:
        return "All tasks passed — no gaps detected."

    buckets = Counter()
    for r in failed:
        q = r.query.lower()
        if "refund" in q or "money" in q:    buckets["refund"]    += 1
        if "cancel" in q:                     buckets["cancel"]    += 1
        if "upgrade" in q or "enterprise" in q: buckets["upgrade"] += 1
        if "crash" in q or "settings" in q:    buckets["technical"] += 1
        if "log in" in q or "password" in q:   buckets["account"]   += 1

    worst, cnt = buckets.most_common(1)[0] if buckets else ("unknown", 0)
    return (
        f"  Failures across {len(buckets)} categories.\n"
        f"  Worst: '{worst}' ({cnt}/{len(failed)} failures).\n"
        f"  MAIN GAP: Agent underperforms on {worst} — "
        f"likely missing instructions for that case.\n"
        f"  FOCUS NEXT: Add 2-3 examples for '{worst}' to the system prompt and re-run."
    )


# ---------------------------------------------------------------------------
# 5.  Single query eval
# ---------------------------------------------------------------------------
async def _score_one(client: httpx.AsyncClient, sample: te.Sample,
                     grader: te.Grader, nid: list[int],
                     system: str | None) -> te.Run:
    run = te.Run(sample_id=sample.id, query=sample.input,
                 answer=sample.target, response="", reward=0.0)
    t0 = time.monotonic()
    try:
        await _mcp_init(client)
        run.metadata["mcp_tools"] = await _mcp_tools(client, nid)
        resp = await _call_agent(client, sample.input, system)
        run.response = resp
        run.reward = float(await grader.score({
            "query": sample.input,
            "answer": sample.target,
            "completion": [{"role": "assistant", "content": resp}],
            "info": {"rubrics": [{"name":r.name,"criteria":r.criteria,"weight":r.weight}
                                 for r in sample.rubrics]},
        }))
    except Exception as e:
        run.error = repr(e)
    run.latency_ms = (time.monotonic() - t0) * 1000
    return run


# ---------------------------------------------------------------------------
# 6.  Main
# ---------------------------------------------------------------------------
async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", default="test_cases.jsonl")
    p.add_argument("--generate", type=int, default=0)
    p.add_argument("--analyze", default=None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--system-prompt",
                   default="You are a professional customer support agent.")
    p.add_argument("--output", default="results.json")
    args = p.parse_args()

    # Gap analysis mode
    if args.analyze:
        result = te.EvaluationResult.load(args.analyze)
        print("=" * 60)
        print("GAP ANALYSIS —", args.analyze)
        print("=" * 60)
        print(_analyze_gaps(result.runs))
        return

    # Load dataset
    ds = te.Datasets.load_from_file(args.tasks)
    if args.generate > 0:
        extra = _generate(args.generate)
        ds = te.Datasets(ds.samples + extra.samples, name="mixed")

    grader = te.RubricGrader(simple=True)

    async with observe_run("eval", tasks=len(ds)) as trace:
        async with httpx.AsyncClient() as client:
            sem = asyncio.Semaphore(args.workers)
            nid = [1]

            async def _go(i: int) -> te.Run:
                async with sem:
                    r = await _score_one(client, ds[i], grader, nid, args.system_prompt)
                    if r.reward < 0.8:
                        trace.log_gap("agent",
                            f"{r.sample_id}: {r.query[:50]} -> {r.reward:.2f}",
                            severity="high" if r.reward < 0.3 else "medium")
                    return r

            runs = await asyncio.gather(*[_go(i) for i in range(len(ds))])

    result = te.EvaluationResult(
        runs=list(runs), datasets=ds,
        config=te.EvalConfig(model="simple", workers=args.workers,
                             system_prompt=args.system_prompt),
    )
    summary = result.summary()

    print("=" * 60)
    print("EVAL SUMMARY")
    print("=" * 60)
    print(f"  Samples:  {summary.num_runs}")
    print(f"  Avg:      {summary.avg_reward:.3f}")
    print(f"  Pass:     {summary.pass_count}/{summary.num_runs}  "
          f"({summary.pass_rate*100:.0f}%)")
    print()
    print(f"  {'ID':<8} {'SCORE':>5}  {"QUERY":<48}  STATUS")
    print("  " + "-"*70)
    for r in runs:
        q = r.query[:46] + ("…" if len(r.query) > 46 else "")
        status = "PASS" if r.reward >= 0.8 else "FAIL"
        print(f"  {r.sample_id:<8} {r.reward:>5.2f}  {q:<48}  [{status}]")

    result.save(args.output)
    print(f"\n  Results -> {args.output}")

    trace.set_summary(avg_reward=summary.avg_reward,
                      pass_rate=summary.pass_rate,
                      gaps_found=len(trace.gaps))

    # Gap analysis
    if trace.gaps:
        print("\n" + _analyze_gaps(runs))


if __name__ == "__main__":
    asyncio.run(main())
