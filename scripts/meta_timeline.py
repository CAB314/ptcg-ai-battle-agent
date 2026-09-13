#!/usr/bin/env python
"""ラダー環境メタの日次推移テーブルを作る（Strategy レポート用の一次資料）。

runs/daily_data/meta_shares_<date>.json があればそれを使い、無い日付は
data/bc_shards/v1/<date>/ から機械集計して同じ形式で書き出す（冪等）。

    poetry run python scripts/meta_timeline.py                 # 表示
    poetry run python scripts/meta_timeline.py --csv out.csv   # CSV も出す
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter

import _bootstrap  # noqa: F401
from _bootstrap import REPO_ROOT

SHARDS = REPO_ROOT / "data" / "bc_shards" / "v1"
# 2026-08-08 以前のシャードはアーカイブ側（NAS）。field_matchup_matrix.py と同じ既定値
ARCHIVE = REPO_ROOT.__class__(os.environ.get(
    "PTCG_SHARDS_ARCHIVE", os.path.expanduser("~/nas/ptcg-ai-battle/bc_shards/v1")))
DAILY = REPO_ROOT / "runs" / "daily_data"

# エース(カードID) → 表示名。data/EN_Card_Data.csv 由来（実データで確認済み）
ACE_NAMES = {
    648: "Grimmsnarl ex", 66: "Alakazam", 431: "TR Mewtwo ex", 756: "M Kangaskhan ex",
    381: "Garchomp ex", 1031: "M Starmie ex", 121: "Dragapult ex", 849: "M Lopunny ex",
    140: "Fezandipiti ex", 245: "Alakazam(245)", 90: "Thwackey", 607: "Terrakion",
    678: "M Lucario ex", 190: "Archaludon ex", 652: "M Venusaur ex",
    150: "Hydrapple ex", 117: "C Ogerpon ex",
}


def compute_day(date: str) -> dict | None:
    """シャードから 1 日分のメタシェアを集計（daily_data_update.sh と同一ロジック）。"""
    import numpy as np

    paths = sorted(glob.glob(str(SHARDS / date / "shard_*.npz")))
    if not paths and ARCHIVE.is_dir():
        paths = sorted(glob.glob(str(ARCHIVE / date / "shard_*.npz")))
    if not paths:
        return None
    arch: Counter = Counter()
    wins: Counter = Counter()
    for sp in paths:
        meta = np.load(sp)["meta"]  # [ep, agent, reward, team, my_ace, opp_ace, turn, sel_type]
        sides = {(int(r[0]), int(r[1])): (int(r[4]), int(r[2])) for r in meta}
        for ace, rw in sides.values():
            arch[ace] += 1
            if rw > 0:
                wins[ace] += 1
    total = sum(arch.values()) or 1
    return {
        "date": date,
        "total_sides": total,
        "shares": {str(a): {"share": n / total, "win": wins[a] / n, "n": n}
                   for a, n in arch.most_common(15)},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--top", type=int, default=8, help="表示するアーキタイプ数")
    ap.add_argument("--from", dest="d_from", default=None, help="開始日 (YYYY-MM-DD)")
    ap.add_argument("--to", dest="d_to", default=None, help="終了日 (YYYY-MM-DD)")
    args = ap.parse_args()

    # 日付の列挙は「集計済み JSON ∪ ローカルシャード ∪ アーカイブシャード」。
    # 2026-08-08 にシャード置き場をローカルへ切り替えた結果、ローカルには 8/5 以降しか無く、
    # シャードだけを列挙すると 6/16〜8/4 が落ちる（8/4 版 CSV との不整合の原因）。
    dates = {d for d in os.listdir(SHARDS) if d.startswith("2026")} if SHARDS.is_dir() else set()
    if ARCHIVE.is_dir():
        dates |= {d for d in os.listdir(ARCHIVE) if d.startswith("2026")}
    dates |= {f.name[len("meta_shares_"):-len(".json")] for f in DAILY.glob("meta_shares_2026-*.json")}
    if args.d_from:
        dates = {d for d in dates if d >= args.d_from}
    if args.d_to:
        dates = {d for d in dates if d <= args.d_to}
    days: list[dict] = []
    for date in sorted(dates):
        f = DAILY / f"meta_shares_{date}.json"
        if f.exists():
            days.append(json.loads(f.read_text()))
            continue
        day = compute_day(date)
        if day is None:
            continue
        f.write_text(json.dumps(day, indent=1))
        print(f"[gen] {f.name}")
        days.append(day)

    # 全期間シェア合計の上位アーキタイプを列に採る
    agg: Counter = Counter()
    for d in days:
        for ace, v in d["shares"].items():
            agg[int(ace)] += v["n"]
    cols = [a for a, _ in agg.most_common(args.top)]

    head = ["date", "sides"] + [ACE_NAMES.get(a, str(a)) for a in cols]
    rows = []
    for d in days:
        r = [d["date"], str(d["total_sides"])]
        for a in cols:
            v = d["shares"].get(str(a))
            r.append(f"{v['share']*100:.1f}/{v['win']*100:.0f}" if v else "-")
        rows.append(r)

    w = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(head)]
    print("シェア%/勝率% （" + f"{len(days)}日・計 {sum(d['total_sides'] for d in days):,} デッキ枠）")
    print("  ".join(h.rjust(w[i]) for i, h in enumerate(head)))
    for r in rows:
        print("  ".join(c.rjust(w[i]) for i, c in enumerate(r)))

    if args.csv:
        import csv as _csv
        with open(args.csv, "w", newline="") as fh:
            wr = _csv.writer(fh)
            wr.writerow(["date", "sides"] + [f"{ACE_NAMES.get(a, a)}_{k}" for a in cols for k in ("share", "win")])
            for d in days:
                row = [d["date"], d["total_sides"]]
                for a in cols:
                    v = d["shares"].get(str(a))
                    row += [round(v["share"], 4), round(v["win"], 4)] if v else ["", ""]
                wr.writerow(row)
        print(f"[ok] {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
