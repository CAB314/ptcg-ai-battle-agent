#!/usr/bin/env python
"""「1枚も違わない同一60枚」を使うチーム同士の勝率のばらつきを測る。

レポートの Deck 20% の中核主張「リストでは差がつかない。差がつくのは打ち回し」の
一次証拠を作る。デッキが完全一致していれば、残る差はパイロット（方策）だけ。

出力: docs/data/clone_winrates.json
    {"modal": [...60枚...], "dates": [...], "teams": [{"team","n","w","wr"}...],
     "totals": {...}}

使い方:
    poetry run python scripts/clone_winrates.py --days 3
    poetry run python scripts/clone_winrates.py --since 2026-08-01 --until 2026-08-03
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import _bootstrap  # noqa: F401
from _bootstrap import REPO_ROOT

EPI = REPO_ROOT / "data" / "episodes"
OUT = REPO_ROOT / "docs" / "data" / "clone_winrates.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=3, help="最新から何日分")
    ap.add_argument("--since"); ap.add_argument("--until")
    ap.add_argument("--variants-json", default="docs/data/grimmsnarl_field_variants.json")
    a = ap.parse_args()

    modal = tuple(sorted(json.loads((REPO_ROOT / a.variants_json).read_text())["modal"]))
    dates = sorted(p.name for p in EPI.iterdir() if p.is_dir() and p.name.startswith("2026"))
    if a.since:
        dates = [d for d in dates if d >= a.since]
    if a.until:
        dates = [d for d in dates if d <= a.until]
    else:
        dates = dates[-a.days:]
    files = [f for d in dates for f in sorted(glob.glob(str(EPI / d / "*.json")))]
    print(f"対象 {len(dates)}日 / {len(files):,} episodes（同一60枚は {len(modal)} 枚）", flush=True)

    per = defaultdict(lambda: [0, 0])   # team -> [n, wins]
    other = [0, 0]                      # 同アーキタイプだがリストが違う側（対照）
    parsed = skipped = 0
    for k, f in enumerate(files, 1):
        if k % 2000 == 0:
            print(f"  {k:,}/{len(files):,}  clone_sides={sum(v[0] for v in per.values()):,}", flush=True)
        try:
            d = json.load(open(f, encoding="utf-8"))
            decks = d["steps"][0][0]["visualize"][0]["action"]
            names = [x.get("Name") for x in d["info"]["Agents"]]
            rew = d.get("rewards") or [0, 0]
        except Exception:
            skipped += 1
            continue
        parsed += 1
        for i in (0, 1):
            try:
                deck = tuple(sorted(decks[i]))
            except Exception:
                continue
            win = 1 if (rew[i] or 0) > 0 else 0
            if deck == modal:
                per[names[i]][0] += 1
                per[names[i]][1] += win
            elif 648 in deck:            # Grimmsnarl だがリストが違う＝対照群
                other[0] += 1
                other[1] += win

    teams = [{"team": t, "n": n, "w": w, "wr": w / n}
             for t, (n, w) in sorted(per.items(), key=lambda kv: -kv[1][0]) if n > 0]
    tot_n = sum(t["n"] for t in teams)
    tot_w = sum(t["w"] for t in teams)
    res = {
        "modal": list(modal), "dates": dates,
        "episodes_parsed": parsed, "episodes_skipped": skipped,
        "teams": teams,
        "totals": {
            "clone_teams": len(teams), "clone_sides": tot_n,
            "clone_wr": tot_w / max(tot_n, 1),
            "nonclone_grimmsnarl_sides": other[0],
            "nonclone_grimmsnarl_wr": other[1] / max(other[0], 1),
        },
    }
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n同一60枚を使ったチーム: {len(teams)}  総サイド {tot_n:,}  全体勝率 {tot_w/max(tot_n,1):.1%}")
    print(f"同アーキタイプ・別リスト（対照）: {other[0]:,} サイド 勝率 {other[1]/max(other[0],1):.1%}")
    big = [t for t in teams if t["n"] >= 30]
    if big:
        w = [t["wr"] for t in big]
        print(f"n>=30 のチーム {len(big)}: 勝率 {min(w):.1%} 〜 {max(w):.1%}")
        for t in big[:8]:
            print(f"    {t['team'][:26]:28s} {t['w']:4d}/{t['n']:4d} = {t['wr']:5.1%}")
    print(f"[ok] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
