#!/usr/bin/env python
"""第2段の eval.json を集計して差分の差分を出す（run ディレクトリを直接指定する版）。

    python scripts/stage2_report.py runs/s2_grim_base_ runs/s2_grim__TS__Boss_ runs/s2_grim_field_v3_

先頭に指定した run を対照群として扱う。
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

rows = []
for d in sys.argv[1:]:
    p = Path(d)
    ev, mt = p / "eval.json", p / "stage2_meta.json"
    if not ev.exists():
        print("[skip] eval.json なし:", p)
        continue
    r = json.loads(ev.read_text())["rows"][0]
    m = json.loads(mt.read_text()) if mt.exists() else {}
    rows.append({"arm": m.get("arm", p.name), "note": m.get("note", ""),
                 "wr": r["wr"], "n": r["n"],
                 "iters": m.get("target_iter", 0) - m.get("start_iter", 0),
                 "per_opp": r.get("per_opp", {})})
if not rows:
    raise SystemExit("集計対象なし")

base = rows[0]
print("対照群 {} ({}iter 追加学習後): {:.2%} (n={:,})".format(base["arm"], base["iters"], base["wr"], base["n"]))
print("{:22s} {:>9s} {:>11s} {:>7s}  {}".format("arm", "勝率", "Δ(対照群比)", "z", "備考"))
for r in rows[1:]:
    se = math.sqrt(r["wr"] * (1 - r["wr"]) / r["n"] + base["wr"] * (1 - base["wr"]) / base["n"])
    d = r["wr"] - base["wr"]
    z = d / se if se else 0.0
    mark = "***" if abs(z) >= 2.58 else ("**" if abs(z) >= 1.96 else "")
    print("{:22s} {:8.2%} {:+10.2%} {:+7.2f} {}  {}".format(r["arm"], r["wr"], d, z, mark, r["note"]))

# 対面別（相手ごとにどこで差がついたか）
opps = sorted(base["per_opp"])
if opps:
    print("\n[対面別勝率]")
    print("{:22s} {}".format("arm", "  ".join(o[:14].rjust(14) for o in opps)))
    for r in rows:
        print("{:22s} {}".format(r["arm"], "  ".join(f"{r['per_opp'].get(o, 0):14.2%}" for o in opps)))
