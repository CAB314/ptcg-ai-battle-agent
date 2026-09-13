#!/usr/bin/env python
"""ローカル自己対戦で勝率を評価する。

2 モード:
  1) 単一相手:  --vs <name|random>
  2) 多様プール: --pool（既定は decks/ 全デッキ × greedy_first 振る舞い）

先攻/後攻を交互に入れ替え、勝率と Wilson 信頼区間を出す。

使い方:
    # 単一相手（ランダムはサニティ用。強さ指標にしない）
    poetry run python scripts/run_eval.py greedy_first --vs random --games 200
    # 多様プール（過学習対策の主力。相手あたり N 戦）
    poetry run python scripts/run_eval.py greedy_first --pool --games-per-opponent 400 --restart-every 100
    # プールを JSON で指定
    poetry run python scripts/run_eval.py greedy_first --pool --pool-config configs/eval_pool.json

評価規律（Discussion 713608 / 717697 / 724187）:
- ランダム単体の勝率は当てにならない（あるチームは 66% → 実戦 0.6%）。
- 改善判定は相手あたり >=400 戦、±1.4pp 以下はノイズ床。
- ローカルはスクリーニング。最終判断は実ラダーの COMPLETE スコア。
- --restart-every で子プロセス分割し C++ シミュレータのメモリリークを回避。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from _bootstrap import AGENTS_DIR, REPO_ROOT

from ptcg import engine
from ptcg.agents.base import RandomAgent
from ptcg.cards import read_deck_csv
from ptcg.eval import (
    EvalResult,
    Opponent,
    PoolMember,
    PoolResult,
    default_pool,
    evaluate,
    load_pool_config,
    resolve_deck_path,
)

THIS = Path(__file__)


# ---- 単一マッチアップの実行（in-process） ----

def run_matchup(
    agent_name: str,
    opp_behavior: str,
    opp_deck_path: str,
    games: int,
    alternate: bool = True,
) -> EvalResult:
    # ML エージェントは decide 毎に weights.npz 等を相対パスで読む。run_eval は
    # cwd=REPO_ROOT のままなので、env 経由で資産ディレクトリを教える
    # （教えないと無言でルール縮退が評価される事故になる）。
    os.environ["PTCG_AGENT_DIR"] = str(AGENTS_DIR / agent_name)
    agent = engine.load_agent(AGENTS_DIR / agent_name / "main.py")
    agent_deck = read_deck_csv(AGENTS_DIR / agent_name / "deck.csv")
    opp_deck = read_deck_csv(opp_deck_path)
    if opp_behavior == "random":
        opponent = RandomAgent(opp_deck)
    else:
        opponent = engine.load_agent(AGENTS_DIR / opp_behavior / "main.py")
    return evaluate(agent, opponent, agent_deck, opp_deck, games=games, alternate_first=alternate)


def run_matchup_batched(
    agent_name: str,
    opp_behavior: str,
    opp_deck_path: str,
    games: int,
    restart_every: int,
    alternate: bool = True,
) -> EvalResult:
    """restart_every>0 なら子プロセスに分割して合算（メモリリーク対策）。"""
    if restart_every <= 0:
        return run_matchup(agent_name, opp_behavior, opp_deck_path, games, alternate)
    total = EvalResult(0, 0, 0, 0)
    remaining = games
    while remaining > 0:
        batch = min(restart_every, remaining)
        cmd = [
            sys.executable, str(THIS), agent_name, "--_child",
            "--opp-behavior", opp_behavior, "--opp-deck", str(opp_deck_path),
            "--games", str(batch),
        ]
        if not alternate:
            cmd.append("--no-alternate")
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr)
            raise RuntimeError(f"子プロセス失敗 (opp={opp_behavior}@{Path(opp_deck_path).stem})")
        w, l, d, n = (int(x) for x in proc.stdout.strip().splitlines()[-1].split())
        total = EvalResult(total.games + n, total.wins + w, total.losses + l, total.draws + d)
        remaining -= batch
    return total


# ---- モード ----

def mode_single(args) -> int:
    if args.vs == "random":
        opp_behavior, opp_deck = "random", str(AGENTS_DIR / args.agent / "deck.csv")
    else:
        opp_behavior, opp_deck = args.vs, str(AGENTS_DIR / args.vs / "deck.csv")
    res = run_matchup_batched(
        args.agent, opp_behavior, opp_deck, args.games, args.restart_every, not args.no_alternate
    )
    print(f"{args.agent} vs {args.vs}: {res}")
    return 0


def mode_pool(args) -> int:
    pool = load_pool_config(args.pool_config) if args.pool_config else default_pool()
    result = PoolResult(agent_label=args.agent)
    for opp in pool:
        deck_path = str(resolve_deck_path(opp.deck))
        res = run_matchup_batched(
            args.agent, opp.behavior, deck_path,
            args.games_per_opponent, args.restart_every, not args.no_alternate,
        )
        result.members.append(PoolMember(opponent=opp, result=res))
        print(f"  [done] {opp.name}: {res}")
    print()
    print(result.report())
    return 0


def mode_child(args) -> int:
    res = run_matchup(
        args.agent, args.opp_behavior, args.opp_deck, args.games, not args.no_alternate
    )
    # 親がパースする機械可読な最終行
    print(f"{res.wins} {res.losses} {res.draws} {res.games}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("agent", help="評価対象のエージェント名")
    ap.add_argument("--vs", default="random", help="単一相手（エージェント名 or 'random'）")
    ap.add_argument("--pool", action="store_true", help="多様プール評価を行う")
    ap.add_argument("--pool-config", default=None, help="プール定義 JSON（省略時は既定プール）")
    ap.add_argument("--games", type=int, default=200, help="単一相手モードの総対戦数")
    ap.add_argument("--games-per-opponent", type=int, default=200, help="プールの相手あたり対戦数")
    ap.add_argument("--restart-every", type=int, default=0, help=">0 で N 戦ごとに子プロセス分割")
    ap.add_argument("--no-alternate", action="store_true", help="先攻/後攻の交互入れ替えを無効化")
    # 内部用
    ap.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--opp-behavior", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--opp-deck", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    try:
        if getattr(args, "_child", False):
            return mode_child(args)
        if args.pool:
            return mode_pool(args)
        return mode_single(args)
    except Exception as e:  # noqa: BLE001
        print(f"[error] {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
