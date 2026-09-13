#!/usr/bin/env python
"""アブレーション結果 JSON を短いテキストに要約する（Slack 通知の本文用）。

    python scripts/ablation_summary.py runs/ablation/<run>/results.json
"""
import json
import sys

d = json.load(open(sys.argv[1]))
rows = d["rows"]
b = rows[0]
lines = ["基準 {} {:.2%} (n={:,})".format(b["name"], b["wr"], b["n"])]
for r in sorted(rows[1:], key=lambda x: -x["delta"]):
    mark = "***" if abs(r["z"]) >= 2.58 else ("**" if abs(r["z"]) >= 1.96 else "")
    lines.append("  {:22s} {:.2%} d{:+.2%} z={:+.2f} {}".format(
        r["name"], r["wr"], r["delta"], r["z"], mark))
print("\n".join(lines[:10]))
