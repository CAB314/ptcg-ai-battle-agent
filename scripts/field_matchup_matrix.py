#!/usr/bin/env python
"""field benchmark 行列 — 他チームのエージェントがアーキ A でアーキ B と当たったときの勝率。

Strategy レポートの弱点対面の節で「方策の問題か、デッキの構造の問題か」を切り分ける測定器。
競技中はその場限りの計算で数値だけをログに残していたため、ここで再現可能にする。

入力は BC シャードの meta 配列 `[ep, agent, reward, team, my_ace, opp_ace, turn, sel_type]`
（`scripts/meta_timeline.py:compute_day` / `scripts/daily_data_update.sh` と同じ読み方）。
1試合の1側 = (date, ep, agent) で重複排除する（日次スクリプトはシャード単位で重複排除して
いたので、シャードを跨ぐ試合が二重に数わる余地があった。ここでは日単位で排除する）。

シャードの置き場所は2つ（日付で結合。両方にある日はローカル優先）:
  - data/bc_shards/v1/<date>/            ローカル（2026-08-05 以降）
  - $PTCG_SHARDS_ARCHIVE/<date>/          アーカイブ（既定 ~/nas/ptcg-ai-battle/bc_shards/v1、2026-06-16〜08-05）

アーキ名は data/EN_Card_Data.csv のカード名（同名の別印刷 = 同一アーキに統合）。

    poetry run python scripts/field_matchup_matrix.py --from 2026-06-16 --to 2026-08-30
    poetry run python scripts/field_matchup_matrix.py --from 2026-08-21 --to 2026-08-30 --tag final-window

既知の限界（JSON にも書き出す）:
  - エース判定は「枚数≥2 の最高HPポケモン」ヒューリスティック（docs/experiments-log.md 740行付近の欠陥記録）
  - Daily Top Episodes は平均レートの高い試合のみ収録 = 上位帯の値
  - レート帯マッチングにより、対面の相手分布はレート帯ごとに異なる（交絡）
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import _bootstrap  # noqa: F401
from _bootstrap import REPO_ROOT

LOCAL = REPO_ROOT / "data" / "bc_shards" / "v1"
ARCHIVE = Path(os.environ.get("PTCG_SHARDS_ARCHIVE",
                              os.path.expanduser("~/nas/ptcg-ai-battle/bc_shards/v1")))
CARD_CSV = REPO_ROOT / "data" / "EN_Card_Data.csv"
OUT_DIR = REPO_ROOT / "docs" / "data"

# ログに残っている「当時の値」（docs/experiments-log.md）。再計算値との突き合わせ用。単位 %
LOGGED = [
    # (my, opp, value, source)
    ("Teal Mask Ogerpon ex", "Mega Lopunny ex", 15.1, "L1018 53日窓 (8/11時点)"),
    ("Team Rocket's Mewtwo ex", "Mega Lopunny ex", 15.2, "L1018"),
    ("Hydrapple ex", "Mega Lopunny ex", 74.0, "L1018"),
    ("Mega Lucario ex", "Mega Lopunny ex", 86.4, "L1018"),
    ("Cynthia's Garchomp ex", "Mega Lopunny ex", 83.5, "L1018"),
    ("Mega Kangaskhan ex", "Mega Lopunny ex", 30.2, "L1018"),
    ("Teal Mask Ogerpon ex", "Mega Kangaskhan ex", 24.9, "commit 7b15cb4 (ogerpon P0診断)"),
    ("Hydrapple ex", "Mega Kangaskhan ex", 54.8, "L1140 (8/16)"),
    ("Mega Kangaskhan ex", "Dragapult ex", 59.0, "L1128 (8/16)"),
    ("Hydrapple ex", "Dragapult ex", 56.3, "L1128"),
    ("Teal Mask Ogerpon ex", "Dragapult ex", 28.3, "L1128"),
    ("Crustle", "Dragapult ex", 79.9, "L1128"),
]


def wilson(w: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = w / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def card_names() -> dict[int, str]:
    """ace ID → アーキ名。同名の別印刷は同名に落ちる。

    さらに vocab.ACE_MERGE を適用する: ace 判定「枚数>=2 の最高HP」は Alakazam デッキ
    （Alakazam 743 と Dudunsparce 66 が同じ HP140）で Dudunsparce を拾うことがあり、
    同じデッキが2アーキに割れる（docs/experiments-log.md 2026-08-04 の記録）。
    """
    from ptcg.ml.vocab import ACE_MERGE

    names: dict[int, str] = {}
    with open(CARD_CSV, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                names[int(r["Card ID"])] = r["Card Name"].strip()
            except (KeyError, ValueError):
                continue
    for src, group in ACE_MERGE.items():
        target = next((g for g in group if g in names), None)
        if target is not None:
            names[src] = names[target]
    return names


def date_dirs(d_from: str, d_to: str) -> list[tuple[str, Path]]:
    """[from, to] の各日付について、ローカル優先でシャードディレクトリを返す。"""
    out = []
    seen = set()
    for root in (LOCAL, ARCHIVE):
        if not root.is_dir():
            continue
        for d in sorted(os.listdir(root)):
            if d_from <= d <= d_to and d not in seen and (root / d).is_dir():
                seen.add(d)
                out.append((d, root / d))
    return sorted(out)


def tally_day(args: tuple[str, str]) -> tuple[str, list[tuple[int, int, int]]]:
    """1日分: (date, ep, agent) で重複排除した (my_ace, opp_ace, reward) のリスト。"""
    import numpy as np

    date, ddir = args
    sides: dict[tuple[int, int], tuple[int, int, int]] = {}
    for sp in sorted(glob.glob(os.path.join(ddir, "shard_*.npz"))):
        try:
            meta = np.load(sp)["meta"]
        except Exception as ex:  # 壊れたシャードは記録して飛ばす
            print(f"  [warn] {sp}: {ex}")
            continue
        for r in meta:
            sides[(int(r[0]), int(r[1]))] = (int(r[4]), int(r[5]), int(r[2]))
    return date, list(sides.values())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="d_from", required=True)
    ap.add_argument("--to", dest="d_to", required=True)
    ap.add_argument("--tag", default=None, help="出力ファイル名の接尾（既定は from_to）")
    ap.add_argument("--top", type=int, default=12, help="表示するアーキ数（シェア順）")
    ap.add_argument("--min-n", type=int, default=30, help="表示する最小対戦数")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    names = card_names()
    days = date_dirs(args.d_from, args.d_to)
    if not days:
        print("シャードが見つからない", LOCAL, ARCHIVE)
        return 1
    print(f"日数 {len(days)}: {days[0][0]} 〜 {days[-1][0]} "
          f"(ローカル {sum(1 for _, p in days if p.is_relative_to(LOCAL))} / アーカイブ "
          f"{sum(1 for _, p in days if not p.is_relative_to(LOCAL))})")

    n = Counter()
    w = Counter()
    sides_per_arch = Counter()
    total_sides = 0
    per_day_sides: dict[str, int] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for date, recs in ex.map(tally_day, [(d, str(p)) for d, p in days]):
            per_day_sides[date] = len(recs)
            for my, opp, rw in recs:
                a, b = names.get(my, str(my)), names.get(opp, str(opp))
                n[(a, b)] += 1
                if rw > 0:
                    w[(a, b)] += 1
                sides_per_arch[a] += 1
                total_sides += 1
            print(f"  {date}: {len(recs):,} sides")

    archs = [a for a, _ in sides_per_arch.most_common()]
    top = archs[: args.top]

    # --- 表示 ---
    print(f"\n合計 {total_sides:,} sides / {len(sides_per_arch)} アーキ。行=自分, 列=相手, 値=勝率%(n)。"
          f" n<{args.min_n} は '·'")
    colw = 12
    print(" " * 26 + "".join(f"{b[:11]:>{colw}}" for b in top))
    for a in top:
        cells = []
        for b in top:
            k = (a, b)
            if n[k] < args.min_n:
                cells.append(f"{'·':>{colw}}")
            else:
                cells.append(f"{100 * w[k] / n[k]:5.1f}({n[k]:>4})".rjust(colw))
        print(f"{a[:25]:25s} " + "".join(cells))

    # --- ログ値との突き合わせ ---
    print("\nログ記載値との突き合わせ（この窓での再計算値）:")
    checks = []
    for my, opp, val, src in LOGGED:
        k = (my, opp)
        if n[k] == 0:
            line = f"  {my:26s} vs {opp:22s} ログ {val:5.1f}  再計算   n=0"
            checks.append({"my": my, "opp": opp, "logged": val, "recomputed": None, "n": 0, "source": src})
        else:
            wr = 100 * w[k] / n[k]
            lo, hi = wilson(w[k], n[k])
            line = (f"  {my:26s} vs {opp:22s} ログ {val:5.1f}  再計算 {wr:5.1f} "
                    f"[{100*lo:4.1f},{100*hi:5.1f}] n={n[k]:<5d} 差 {wr - val:+5.1f}  ({src})")
            checks.append({"my": my, "opp": opp, "logged": val, "recomputed": round(wr, 1),
                           "ci95": [round(100 * lo, 1), round(100 * hi, 1)], "n": n[k], "source": src})
        print(line)

    # --- 書き出し ---
    tag = args.tag or f"{args.d_from}_{args.d_to}"
    cells = []
    for (a, b), c in n.items():
        lo, hi = wilson(w[(a, b)], c)
        cells.append({"my": a, "opp": b, "n": c, "wins": w[(a, b)], "wr": round(100 * w[(a, b)] / c, 1),
                      "ci95": [round(100 * lo, 1), round(100 * hi, 1)]})
    cells.sort(key=lambda r: -r["n"])
    out = {
        "window": {"from": days[0][0], "to": days[-1][0], "days": len(days)},
        "sources": {"local": str(LOCAL), "archive": str(ARCHIVE)},
        "total_sides": total_sides,
        "per_day_sides": per_day_sides,
        "archetype_share": {a: {"sides": c, "share": round(c / total_sides, 4)}
                            for a, c in sides_per_arch.most_common(40)},
        "cells": cells,
        "logged_value_checks": checks,
        "method": "Daily Top Episodes → BC shards meta [ep,agent,reward,team,my_ace,opp_ace]; "
                  "1 side = (date,ep,agent) 重複排除; ace = 枚数>=2 の最高HPポケモン; "
                  "アーキ名 = EN_Card_Data.csv のカード名（別印刷を統合）; CI = Wilson 95%",
        "limitations": [
            "エース判定ヒューリスティックの精度限界（ハイブリッド構築を片方のアーキに寄せる）",
            "Daily Top Episodes は平均レートの高い試合のみ = 上位帯の値",
            "レート帯マッチングにより相手分布がレート帯ごとに異なる（交絡）",
            "引き分けは reward ∈ {-1,+1} のため含まれない",
        ],
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jp = OUT_DIR / f"field_matchup_matrix_{tag}.json"
    jp.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    cp = OUT_DIR / f"field_matchup_matrix_{tag}.csv"
    with open(cp, "w", newline="", encoding="utf-8") as f:
        wr_ = csv.writer(f)
        wr_.writerow(["my", "opp", "n", "wins", "wr", "ci_lo", "ci_hi"])
        for r in cells:
            wr_.writerow([r["my"], r["opp"], r["n"], r["wins"], r["wr"], r["ci95"][0], r["ci95"][1]])
    print(f"\n書き出し: {jp.relative_to(REPO_ROOT)} / {cp.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
