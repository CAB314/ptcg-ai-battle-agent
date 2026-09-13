#!/usr/bin/env python
"""第2段の結果を差分の差分（diff-in-diff）で要約する。

    python scripts/stage2_summary.py <tag> "<arm1> <arm2> ..."

各 arm の「微調整後の方策 × 自分のデッキ」の勝率を集め、対照群（base）との差を出す。
第1段（微調整なし）の差と並べることで、
  ・微調整後も改善が残る → カード自体が強い
  ・微調整で差が消える／逆転する → 第1段の差は「方策の不慣れ」だった
を切り分ける。
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from _bootstrap import REPO_ROOT


def load(tag: str, arm: str):
    safe = re.sub(r"[^A-Za-z0-9]", "_", arm)
    p = REPO_ROOT / f"runs/s2_{tag}_{safe}/eval.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    r = d["rows"][0]
    m = json.loads((p.parent / "stage2_meta.json").read_text())
    return {"arm": arm, "wr": r["wr"], "n": r["n"], "note": m.get("note", ""),
            "iters": m["target_iter"] - m["start_iter"]}


def stage1_delta(cand_file: str, arm: str):
    """第1段（微調整なし）での差を探す。"""
    for p in sorted((REPO_ROOT / "runs/ablation").glob("*/results.json")):
        try:
            rows = json.loads(p.read_text())["rows"]
        except Exception:
            continue
        for r in rows:
            if r["name"] == arm:
                return r["delta"], r["z"]
    return None, None


def main() -> int:
    tag = sys.argv[1]
    arms = sys.argv[2].split()
    res = [x for x in (load(tag, a) for a in arms) if x]
    if not res:
        print("第2段の評価結果が見つからない")
        return 1
    base = next((x for x in res if x["arm"] == "base"), None)
    if base is None:
        print("対照群(base)が無いため差分の差分を計算できない")
        return 1
    print(f"対照群 base（{base['iters']}iter 追加学習後）: {base['wr']:.2%} (n={base['n']:,})")
    print(f"{'arm':22s} {'微調整後':>8s} {'Δ(対照群比)':>11s} {'z':>6s} {'第1段Δ':>8s}  判定")
    for x in sorted(res, key=lambda r: -r["wr"]):
        if x["arm"] == "base":
            continue
        se = math.sqrt(x["wr"] * (1 - x["wr"]) / x["n"] + base["wr"] * (1 - base["wr"]) / base["n"])
        d = x["wr"] - base["wr"]
        z = d / se if se else 0.0
        s1d, _ = stage1_delta("", x["arm"])
        verdict = ("本物（微調整後も改善）" if z >= 1.96 else
                   "不慣れが原因（微調整で改善に転じた）" if (s1d is not None and s1d < -0.005 and z > -1.96) else
                   "改善せず" if z > -1.96 else "有意に悪化")
        s1s = f"{s1d:+.2%}" if s1d is not None else "  n/a"
        print(f"{x['arm']:22s} {x['wr']:7.2%} {d:+10.2%} {z:+6.2f} {s1s:>8s}  {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
