#!/usr/bin/env python
"""相手アーキタイプ推定器の精度を測る（正解ラベル既知のローカル対戦で）。

decks/ の各デッキを相手に対戦し、我々の手番ごとに ArchetypeTracker で相手を推定。
- 最終正解率: 試合終了時の推定 == 真アーキタイプ の割合
- 確信到達率: 試合中に confident（エース確認, conf>=0.99）になれた割合
- 中央ロックターン: 初めて正しく確信したターン（confident な試合のみ）

真ラベルは decks/ のファイル名から引く（下の TRUTH）。

使い方:
    poetry run python scripts/eval_estimator.py --games 100
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from _bootstrap import AGENTS_DIR, REPO_ROOT

from ptcg import engine
from ptcg.cards import read_deck_csv
from ptcg.meta import ArchetypeTracker

DECKS_DIR = REPO_ROOT / "decks"

# decks/<stem>.csv -> 真アーキタイプ名（archetypes.SIGNATURES のキー）
TRUTH = {
    "sample_abomasnow": "abomasnow",
    "archaludon_cinderace": "archaludon",
    "greattusk_crustle_lo": "crustle_lo",
}


def eval_deck(agent, our_deck, opp_deck, truth: str, games: int):
    engine.ensure_cg_importable()
    from cg.api import to_observation_class
    from cg.game import battle_finish, battle_select, battle_start

    correct = confident_correct = 0
    lock_turns: list[int] = []
    for g in range(games):
        our_seat = g % 2
        d0, d1 = (our_deck, opp_deck) if our_seat == 0 else (opp_deck, our_deck)
        obs, start = battle_start(list(d0), list(d1))
        if start.errorPlayer >= 0:
            raise ValueError("illegal deck in estimator eval")
        tracker = ArchetypeTracker()
        lock_turn = None
        final = "unknown"
        try:
            while obs["current"]["result"] < 0:
                o = to_observation_class(obs)
                who = o.current.yourIndex
                if who == our_seat:
                    guess = tracker.update(o)
                    final = guess.name
                    if lock_turn is None and guess.confident and guess.name == truth:
                        lock_turn = o.current.turn
                move = agent(obs)
                obs = battle_select(move)
        finally:
            battle_finish()
        if final == truth:
            correct += 1
        if lock_turn is not None:
            confident_correct += 1
            lock_turns.append(lock_turn)
    med = statistics.median(lock_turns) if lock_turns else None
    return dict(
        truth=truth, games=games,
        final_acc=correct / games,
        lock_rate=confident_correct / games,
        median_lock_turn=med,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=100, help="デッキごとの対戦数")
    ap.add_argument("--behavior", default="greedy_first", help="対戦を進める振る舞い")
    args = ap.parse_args()

    agent = engine.load_agent(AGENTS_DIR / args.behavior / "main.py")
    our_deck = read_deck_csv(AGENTS_DIR / args.behavior / "deck.csv")

    rows = []
    for deck_file in sorted(DECKS_DIR.glob("*.csv")):
        stem = deck_file.stem
        truth = TRUTH.get(stem)
        if truth is None:
            print(f"[skip] {stem}: 真ラベル未定義")
            continue
        opp_deck = read_deck_csv(deck_file)
        r = eval_deck(agent, our_deck, opp_deck, truth, args.games)
        r["deck"] = stem
        rows.append(r)
        mlt = r["median_lock_turn"]
        print(
            f"[{stem:22s}] truth={truth:12s} final_acc={r['final_acc']*100:5.1f}%  "
            f"lock_rate={r['lock_rate']*100:5.1f}%  median_lock_turn={mlt}"
        )

    if rows:
        macro = sum(r["final_acc"] for r in rows) / len(rows)
        macro_lock = sum(r["lock_rate"] for r in rows) / len(rows)
        print(f"\nmacro final_acc={macro*100:.1f}%  macro lock_rate={macro_lock*100:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
