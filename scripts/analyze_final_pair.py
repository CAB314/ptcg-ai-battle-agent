#!/usr/bin/env python
"""最終ペアのラダー実測 vs 内部予測（レポート F8 の一次データ）。

キャッシュ済みリプレイ（scripts/fetch_our_episodes.py で取得）から対面別成績を再集計し、
凍結前の内部評価値（docs/experiments-log.md に記録済み）と突き合わせる。

    poetry run python scripts/analyze_final_pair.py
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401
from _bootstrap import REPO_ROOT

CACHE = REPO_ROOT / "data/episodes_ours"
OUR_TEAM = "cabbage patch"
SUBS = {"kangaskhan_dh": 55541008, "hydrapple_k2": 55537575}

# 凍結前の内部評価（出典: docs/experiments-log.md 2026-08-11/08-16）。単位 %
INTERNAL = {
    "kangaskhan_dh": {"Dragapult ex": 71.8, "Hydrapple ex": 82.8, "Marnie's Grimmsnarl ex": 72.0,
                      "Mega Lopunny ex": 15.8, "Alakazam": 53.6, "Teal Mask Ogerpon ex": 99.3,
                      "Mega Kangaskhan ex": 50.0},
    "hydrapple_k2": {"Mega Kangaskhan ex": 29.3, "Marnie's Grimmsnarl ex": 84.6,
                     "Mega Lopunny ex": 80.2, "Alakazam": 64.9},
}


def wilson(w: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = w / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def tally(sub_id: int, ace_label) -> tuple[Counter, Counter, str]:
    """現在の episode 一覧（= 直近1000件の窓）に載っているものだけを集計する。

    kangaskhan_dh には 8/16-17 に取得した未収束期の 59 試合がキャッシュに残っているが、
    両エージェントで窓を揃えないと比較にならないため除外する。
    """
    n, w = Counter(), Counter()
    cdir = CACHE / str(sub_id)
    eps = json.loads((cdir / "_episodes.json").read_text(encoding="utf-8"))
    # CLI 出力の末尾案内文が1件混ざることがある（fetch_our_episodes.py の既知の癖）
    eps = [e for e in eps if isinstance(e, dict) and str(e.get("id", "")).strip().isdigit()]
    ids = {int(e["id"]) for e in eps}
    times = sorted(e.get("createTime", "")[:16] for e in eps if e.get("createTime"))
    window = f"{times[0]} 〜 {times[-1]}" if times else "?"
    cache_recs = []
    for rp in cdir.glob("episode-*-replay.json"):
        try:
            eid = int(rp.name.split("-")[1])
        except (IndexError, ValueError):
            continue
        if eid not in ids:
            continue
        try:
            rep = json.loads(rp.read_text(encoding="utf-8"))
            names = [a.get("Name") for a in rep["info"]["Agents"]]
            if OUR_TEAM not in names:
                continue
            mine = names.index(OUR_TEAM)
            opp = ace_label(rep["steps"][0][0]["visualize"][0]["action"][1 - mine])
            r = (rep.get("rewards") or [0, 0])[mine]
        except Exception:
            continue
        n[opp] += 1
        if (r or 0) > 0:
            w[opp] += 1
        cache_recs.append({"id": eid, "opp": opp, "reward": r})
    (cdir / "_matchups.json").write_text(json.dumps(cache_recs), encoding="utf-8")
    return n, w, window


def main() -> int:
    import sys
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from fetch_our_episodes import make_ace_label
    ace_label = make_ace_label()

    out: dict = {"window": "エージェントごとに agents[].window を見ること（APIの一覧は直近1000件で頭打ち）",
                 "note": "アーキタイプ判定は ACE SPEC ヒューリスティック（既知の限界あり）", "agents": {}}
    tot: dict[str, tuple[Counter, Counter]] = {}
    for name, sid in SUBS.items():
        n, w, window = tally(sid, ace_label)
        tot[name] = (n, w)
        N, W = sum(n.values()), sum(w.values())
        rows = []
        for opp, c in n.most_common():
            lo, hi = wilson(w[opp], c)
            rows.append({"opponent": opp, "n": c, "wins": w[opp], "wr": round(100 * w[opp] / c, 1),
                         "ci95": [round(100 * lo, 1), round(100 * hi, 1)],
                         "internal": INTERNAL.get(name, {}).get(opp)})
        out["agents"][name] = {"submission_id": SUBS[name], "window": window, "games": N, "wins": W,
                               "wr": round(100 * W / max(N, 1), 1), "matchups": rows}
        print(f"\n===== {name} (sub {sid}): {W}/{N} = {100*W/max(N,1):.1f}%")
        print(f"{'対面':30s} {'n':>5} {'実測':>7} {'95%CI':>14} {'内部':>7} {'差':>7}")
        for r in rows:
            if r["n"] < 20:
                continue
            d = "" if r["internal"] is None else f"{r['wr']-r['internal']:+.1f}"
            i = "" if r["internal"] is None else f"{r['internal']:.1f}"
            print(f"{r['opponent']:30s} {r['n']:>5} {r['wr']:>6.1f}% "
                  f"[{r['ci95'][0]:>4.1f},{r['ci95'][1]:>5.1f}] {i:>7} {d:>7}")

    # ペア被覆: 遭遇数の多い対面ごとに「良い方」を見る
    print("\n===== ペア被覆（遭遇100戦以上）")
    print(f"{'対面':30s} {'遭遇':>6} {'kang_dh':>9} {'hyd_k2':>9} {'ペア最良':>9}")
    cov = []
    keys = {o for name in SUBS for o, c in tot[name][0].items() if c >= 100}
    for opp in sorted(keys, key=lambda o: -(tot['kangaskhan_dh'][0][o] + tot['hydrapple_k2'][0][o])):
        enc = tot["kangaskhan_dh"][0][opp] + tot["hydrapple_k2"][0][opp]
        a = 100 * tot["kangaskhan_dh"][1][opp] / max(tot["kangaskhan_dh"][0][opp], 1)
        b = 100 * tot["hydrapple_k2"][1][opp] / max(tot["hydrapple_k2"][0][opp], 1)
        best = max(a, b)
        cov.append({"opponent": opp, "encounters": enc, "kangaskhan_dh": round(a, 1),
                    "hydrapple_k2": round(b, 1), "pair_best": round(best, 1)})
        print(f"{opp:30s} {enc:>6} {a:>8.1f}% {b:>8.1f}% {best:>8.1f}%")
    out["pair_coverage"] = cov

    p = REPO_ROOT / "docs/data/final_pair_matchups.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n書き出し: {p.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
