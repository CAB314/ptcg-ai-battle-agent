#!/usr/bin/env python
"""提出前スモークテスト（隔離ハーネス模倣）。

Kaggle と同じ流儀（素の名前空間で exec、`__file__` なし、deck.csv は
エージェントディレクトリから読む）でエージェントをロードし、実ゲームを
数回完走させてクラッシュ・違法手・デッキ読込事故を洗い出す。

使い方:
    poetry run python scripts/smoke_test.py random_baseline
    poetry run python scripts/smoke_test.py meta_a --games 20

全部 green なら本番でも「ポケモンの意思決定で負ける」ところまでは到達できる。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from _bootstrap import AGENTS_DIR

from ptcg import engine
from ptcg.agents.base import random_move, validate_move
from ptcg.cards import read_deck_csv


def _check_deck(deck) -> bool:
    if len(deck) != 60:
        print(f"[FAIL] deck.csv が {len(deck)}枚（60枚必要）")
        return False
    ok, err = engine.check_deck(deck)
    if not ok:
        print(f"[FAIL] デッキ不正: {engine.DECK_ERROR_MESSAGES.get(err, err)}")
        return False
    print("[ok] デッキ合法（60枚）")
    return True


def _check_load(main_py: Path):
    """__file__ 罠・import 時クラッシュを検出。"""
    try:
        agent = engine.load_agent(main_py)
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] agent のロードで例外: {type(e).__name__}: {e}")
        print("       （import 時に __file__ 等を触っていないか確認）")
        return None
    print("[ok] agent ロード成功（__file__ 罠なし）")
    return agent


def _check_initial_selection(agent, deck) -> bool:
    """初手（select=None）で 60枚デッキを返すか。deck.csv 読込経路の検証。"""
    obs = {"select": None, "logs": [], "current": None}
    try:
        ret = agent(obs)
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] 初手選択で例外: {type(e).__name__}: {e}")
        return False
    if not isinstance(ret, list) or len(ret) != 60:
        print(f"[FAIL] 初手が 60枚のデッキを返さない（返り値長 {len(ret) if isinstance(ret, list) else '非list'}）")
        return False
    if list(ret) != list(deck):
        print("[warn] 初手デッキが deck.csv と一致しない（意図的なら可）")
    print("[ok] 初手で 60枚デッキを返す")
    return True


def _survive(agent, deck, games: int) -> bool:
    engine.ensure_cg_importable()
    from cg.api import to_observation_class
    from cg.game import battle_finish, battle_select, battle_start

    crashes = illegal = 0
    for g in range(games):
        obs, start = battle_start(list(deck), list(deck))
        if start.errorPlayer >= 0:
            print("[FAIL] スモーク中にデッキ不正が発生")
            return False
        me = g % 2
        try:
            while obs["current"]["result"] < 0:
                who = obs["current"]["yourIndex"]
                o = to_observation_class(obs)
                if who == me:
                    try:
                        mv = agent(obs)
                    except Exception:  # noqa: BLE001
                        crashes += 1
                        mv = random_move(o.select)
                    else:
                        if not validate_move(mv, o.select):
                            illegal += 1
                            mv = random_move(o.select)
                else:
                    mv = random_move(o.select)
                obs = battle_select(mv)
        finally:
            battle_finish()
    status = "ok" if crashes == 0 and illegal == 0 else "FAIL"
    print(f"[{status}] {games} ゲーム完走 — crashes={crashes}, illegal={illegal}")
    return crashes == 0 and illegal == 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("agent", help="agents/ 配下のエージェント名")
    ap.add_argument("--games", type=int, default=5, help="スモーク対戦数")
    args = ap.parse_args()

    agent_dir = AGENTS_DIR / args.agent
    main_py = agent_dir / "main.py"
    deck_csv = agent_dir / "deck.csv"
    if not main_py.exists() or not deck_csv.exists():
        print(f"[FAIL] {agent_dir} に main.py / deck.csv がありません")
        return 1

    deck = read_deck_csv(deck_csv)
    # Kaggle 同様、エージェントは deck.csv と同じ場所で動く
    os.chdir(agent_dir)

    results = [
        _check_deck(deck),
    ]
    agent = _check_load(main_py)
    results.append(agent is not None)
    if agent is not None:
        results.append(_check_initial_selection(agent, deck))
        results.append(_survive(agent, deck, args.games))

    ok = all(results)
    print("\n=== " + ("ALL GREEN ✅ 提出OK" if ok else "赤あり ❌ 修正して再実行") + " ===")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
