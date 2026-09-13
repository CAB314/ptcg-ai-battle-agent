"""勝敗理由の内訳計測（docs/ptcg-usable-excerpts.md #7 敗着分類の第一層）。

RESULT ログの reason（1=サイド取り切り, 2=山切れLO, 3=場切れ, 4=カード効果）を
勝ち/負け別に集計する。「どう勝ち・どう負けているか」はプレイブック分岐の
発火設計（何を直せば勝ち筋が伸び/負け筋が消えるか）に直結する。
例: greattusk_lo_playbook は勝ちがほぼ全て reason=2（ミル勝ち=設計通り）で、
負けは対TR Mewtwoの reason=3（場切れ）に集中 → 盤面生存分岐のA/Bに繋がった。

使い方:
    poetry run python scripts/reason_probe.py greattusk_lo_playbook --vs meta_alakazam --games 24
    poetry run python scripts/reason_probe.py greattusk_lo_playbook --vs all --games 16
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ptcg.cards import read_deck_csv  # noqa: E402
from ptcg.engine import ensure_cg_importable, load_agent  # noqa: E402

REASONS = {1: "サイド取り切り", 2: "山切れ(LO)", 3: "場切れ", 4: "カード効果", None: "不明"}


def run_match(my_agent, opp_agent, my_deck, opp_deck, my_idx):
    from cg.api import LogType
    from cg.game import battle_finish, battle_select, battle_start

    decks = (my_deck, opp_deck) if my_idx == 0 else (opp_deck, my_deck)
    obs, start = battle_start(list(decks[0]), list(decks[1]))
    if start.errorPlayer >= 0:
        raise ValueError(f"deck error: player={start.errorPlayer} type={start.errorType}")
    reason = None
    try:
        while obs["current"]["result"] < 0:
            who = obs["current"]["yourIndex"]
            move = my_agent(obs) if who == my_idx else opp_agent(obs)
            obs = battle_select(move)
            for lg in obs.get("logs") or []:
                if lg.get("type") == int(LogType.RESULT):
                    reason = lg.get("reason")
        return obs["current"]["result"], reason
    finally:
        battle_finish()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("agent", help="agents/ 配下のエージェント名")
    ap.add_argument("--vs", default="all", help="decks/ のデッキ名（拡張子不要）または all")
    ap.add_argument("--games", type=int, default=16, help="相手デッキあたりの対戦数（先後交互）")
    ap.add_argument("--behavior", default="greedy_first", help="相手の振る舞い（agents/ 名）")
    args = ap.parse_args()

    ensure_cg_importable()
    # ML エージェントは PTCG_AGENT_DIR で weights.npz 等を解決する（無いとモデル不発の
    # ルール縮退で戦ってしまい、勝率が silently 劣化する）。1変数のため ML は自側のみ対応。
    import os
    os.environ["PTCG_AGENT_DIR"] = str(REPO_ROOT / "agents" / args.agent)
    my_agent = load_agent(REPO_ROOT / "agents" / args.agent / "main.py")
    opp_agent = load_agent(REPO_ROOT / "agents" / args.behavior / "main.py")
    my_deck = read_deck_csv(REPO_ROOT / "agents" / args.agent / "deck.csv")

    if args.vs == "all":
        deck_paths = sorted((REPO_ROOT / "decks").glob("*.csv"))
    else:
        deck_paths = [REPO_ROOT / "decks" / f"{args.vs.removesuffix('.csv')}.csv"]

    grand = collections.Counter()
    for dp in deck_paths:
        opp_deck = read_deck_csv(dp)
        stats = collections.Counter()
        for g in range(args.games):
            my_idx = g % 2
            result, reason = run_match(my_agent, opp_agent, my_deck, opp_deck, my_idx)
            outcome = "win" if result == my_idx else ("draw" if result == 2 else "loss")
            stats[(outcome, reason)] += 1
            grand[(outcome, reason)] += 1
        wins = sum(v for (o, _), v in stats.items() if o == "win")
        print(f"=== vs {dp.stem}: {wins}/{args.games} ===")
        for (outcome, reason), v in sorted(stats.items()):
            print(f"  {outcome:4s} {REASONS.get(reason, reason)}: {v}")
    if len(deck_paths) > 1:
        total = sum(grand.values())
        wins = sum(v for (o, _), v in grand.items() if o == "win")
        print(f"\n=== TOTAL: {wins}/{total} ===")
        for (outcome, reason), v in sorted(grand.items()):
            print(f"  {outcome:4s} {REASONS.get(reason, reason)}: {v}")


if __name__ == "__main__":
    main()
