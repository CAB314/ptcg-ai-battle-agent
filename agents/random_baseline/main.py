"""random_baseline — 合法手をランダムに返す self-contained なエージェント。

このファイルは **提出物そのもの**。以下を厳守している:
- `__file__` を使わない（Kaggle は素の名前空間で exec するため未定義になる）。
- `agent()` を最後の関数として定義する。
- 例外を投げず必ず合法手を返す（fail-closed）。1手でも例外を投げると即負け。
- 初手（obs.select is None）は deck.csv の 60枚を返す。

戦略を作るときは decide() の中身だけ差し替える。安全網（agent の try/except と
検証）はそのまま残すこと。src/ptcg には import 依存しない（同梱されないため）。
"""

import os
import random

from cg.api import to_observation_class

_DECK_CACHE = None


def _load_deck():
    global _DECK_CACHE
    if _DECK_CACHE is not None:
        return _DECK_CACHE
    path = "deck.csv"
    if not os.path.exists(path):
        path = "/kaggle_simulations/agent/deck.csv"
    with open(path, "r", encoding="utf-8") as f:
        _DECK_CACHE = [int(line) for line in f.read().splitlines() if line.strip()]
    return _DECK_CACHE


def _random_move(select):
    n = len(select.option)
    k = min(select.maxCount, n)
    k = max(k, select.minCount)
    return random.sample(range(n), k)


def _is_valid(move, select):
    n = len(select.option)
    return (
        isinstance(move, list)
        and all(isinstance(x, int) and 0 <= x < n for x in move)
        and len(set(move)) == len(move)
        and select.minCount <= len(move) <= select.maxCount
    )


def decide(obs):
    """戦略本体。obs.select は None でない。ここを実装していく。

    現状: 合法手をランダムに選ぶだけ。
    """
    return _random_move(obs.select)


def agent(obs_dict):
    try:
        obs = to_observation_class(obs_dict)
        if obs.select is None:
            return _load_deck()
        move = decide(obs)
        if _is_valid(move, obs.select):
            return move
        return _random_move(obs.select)
    except Exception:
        # 何があっても合法手を返す
        try:
            obs = to_observation_class(obs_dict)
            if obs.select is None:
                return _load_deck()
            return _random_move(obs.select)
        except Exception:
            return []
