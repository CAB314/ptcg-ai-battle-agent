"""シャードの meta からチーム×アーキタイプ×決定数を集計（デッキゲート資料）。

使い方:
    poetry run python -m ptcg.ml.bc.stats data/bc_shards/v1
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


def date_stats(date_dir: Path):
    teams_rev = {}
    tj = date_dir / "teams.json"
    if tj.exists():
        teams_rev = {v: k for k, v in json.loads(tj.read_text(encoding="utf-8")).items()}
    per = Counter()  # (team, ace) -> decisions
    wins = Counter()
    for sp in sorted(date_dir.glob("shard_*.npz")):
        z = np.load(sp)
        meta = z["meta"]  # [ep, agent, reward, team_idx, my_ace, opp_ace, turn, sel_type]
        for row in meta:
            team = teams_rev.get(int(row[3]), f"#{int(row[3])}")
            key = (team, int(row[4]))
            per[key] += 1
            if int(row[2]) > 0:
                wins[key] += 1
    return per, wins


def main(root: str):
    rootp = Path(root)
    total = Counter()
    total_w = Counter()
    for date_dir in sorted(p for p in rootp.iterdir() if p.is_dir()):
        per, wins = date_stats(date_dir)
        total.update(per)
        total_w.update(wins)
        print(f"[{date_dir.name}] 決定数 {sum(per.values()):,}")
    print("\n=== チーム×エースID 決定数 上位40（勝ち側決定の比率付き）===")
    for (team, ace), n in total.most_common(40):
        w = total_w[(team, ace)]
        print(f"  {team:24s} ace={ace:<5d} {n:8,d}  win-side {w / n:5.1%}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/bc_shards/v1")
