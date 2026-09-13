#!/usr/bin/env python
"""デッキ総当たり比較（behavior を固定してデッキ強度を測る）。

同一の振る舞い（既定 greedy_lethal2）を両者に使い、decks/ のデッキ同士を
総当たりさせることで、**プレイングを一定にしたときのデッキ相性/強度**を切り出す。
「1提出=単一デッキ」なので、どれを main にするかの判断材料になる（プラン③）。

使い方:
    poetry run python scripts/compare_decks.py --games 200
    poetry run python scripts/compare_decks.py --behavior greedy_lethal2 --games 300

注意: ローカルはスクリーニング。相手プールが自分と同じ behavior なので絶対値ではなく
相対比較として読む。最終判断は実ラダー。大量対戦時は C++ のメモリリークに注意
（本スクリプトは単一プロセス逐次実行。数千戦回すなら分割実行を検討）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from _bootstrap import AGENTS_DIR, REPO_ROOT

from ptcg import engine
from ptcg.cards import read_deck_csv
from ptcg.eval import EvalResult, evaluate, wilson_interval

DECKS_DIR = REPO_ROOT / "decks"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--behavior", default="greedy_lethal2", help="両者に使う振る舞い(agent名)")
    ap.add_argument("--games", type=int, default=200, help="ペアあたり対戦数")
    ap.add_argument("--decks", default=None,
                    help="対象デッキをカンマ区切りの stem で限定（省略時は decks/ 全部）")
    args = ap.parse_args()

    agent = engine.load_agent(AGENTS_DIR / args.behavior / "main.py")
    deck_files = sorted(DECKS_DIR.glob("*.csv"))
    if args.decks:
        wanted = {s.strip() for s in args.decks.split(",") if s.strip()}
        deck_files = [f for f in deck_files if f.stem in wanted]
        missing = wanted - {f.stem for f in deck_files}
        if missing:
            print(f"指定デッキが見つかりません: {sorted(missing)}", file=sys.stderr)
            return 1
    if len(deck_files) < 2:
        print("decks/ に2つ以上のデッキが必要です", file=sys.stderr)
        return 1
    names = [f.stem for f in deck_files]
    decks = {f.stem: read_deck_csv(f) for f in deck_files}
    n = len(names)

    # matrix[i][j] = デッキ i の 対 デッキ j 勝率（i!=j）
    matrix = [[None] * n for _ in range(n)]
    agg = {name: EvalResult(0, 0, 0, 0) for name in names}

    for i in range(n):
        for j in range(i + 1, n):
            di, dj = names[i], names[j]
            res = evaluate(agent, agent, decks[di], decks[dj], games=args.games)
            # res は「seat=agent(=deck i)」視点
            pi = res.winrate
            matrix[i][j] = pi
            matrix[j][i] = 1 - pi if res.draws == 0 else (res.losses / res.games)
            agg[di] = EvalResult(agg[di].games + res.games, agg[di].wins + res.wins,
                                 agg[di].losses + res.losses, agg[di].draws + res.draws)
            agg[dj] = EvalResult(agg[dj].games + res.games, agg[dj].wins + res.losses,
                                 agg[dj].losses + res.wins, agg[dj].draws + res.draws)
            print(f"  {di} vs {dj}: {res}")

    # 相性表
    w = max(len(x) for x in names)
    print(f"\nbehavior = {args.behavior}  ({args.games} games/pair)")
    print("\nmatchup winrate (row vs column, %):")
    print("  " + " " * w + "  " + "  ".join(f"{x[:10]:>10}" for x in names))
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append(f"{'—':>10}")
            else:
                row.append(f"{matrix[i][j]*100:>10.1f}")
        print("  " + names[i].ljust(w) + "  " + "  ".join(row))

    # 総合ランキング（全対戦の集計勝率 + Wilson）
    print("\noverall (vs the other decks, equal games):")
    ranking = sorted(names, key=lambda name: agg[name].winrate, reverse=True)
    for name in ranking:
        r = agg[name]
        p, lo, hi = wilson_interval(r.wins, r.games)
        print(f"  {name.ljust(w)}  {r.wins}-{r.losses}-{r.draws}  "
              f"winrate={p*100:5.1f}%  [95% CI {lo*100:.1f}–{hi*100:.1f}]")
    print(f"\n=> best deck (this field, this behavior): {ranking[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
