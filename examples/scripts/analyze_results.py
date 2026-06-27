"""Analyze customer support evaluation results."""

import json

with open("examples/results/customer_support_results.json") as f:
    data = json.load(f)

print("Model:", data["summary"]["model"])
print("Total:", data["summary"]["num_runs"])
print("Avg Reward:", data["summary"]["avg_reward"])
print("Pass Rate:", data["summary"]["pass_rate"])
print("Passed:", data["summary"]["pass_count"])
print("Failed:", data["summary"]["fail_count"])
print()

for run in data["runs"]:
    status = "PASS" if run["reward"] >= 0.7 else "FAIL"
    q = run["query"][:60]
    r = run.get("response", "").encode("ascii", "replace").decode()[:80]
    reward = run["reward"]
    qid = run["query_id"]
    print(f"[{status}] {qid}: {q}...")
    print(f"    Reward: {reward:.2f}")
    print(f"    Response: {r}...")
    print()
