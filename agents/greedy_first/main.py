"""greedy_first — エンジンの「最良→最悪」オプション順を利用する簡易エージェント。

cabt エンジンは合法手を概ね強い順に列挙する（Discussion 713608）。そこで
各選択で先頭（=最善とされる）インデックスを必要枚数だけ選ぶ。ランダムより
大幅に強く、ルールベースを積み上げる際のたたき台になる。

このファイルは提出物そのもの。self-contained・`__file__` 非依存・fail-closed を厳守。
戦略を足すときは decide() を拡張し、安全網（agent の try/except と検証）は残すこと。
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
    k = max(min(select.maxCount, n), select.minCount)
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
    """先頭（最善とされる）オプションを必要数だけ選ぶ貪欲手。

    - maxCount==0: 何も選ばない（[]）
    - minCount>0: 先頭 minCount 個
    - それ以外（任意選択）: 先頭 1 個だけ取る（行動をスキップせず最善を打つ）
    """
    s = obs.select
    n = len(s.option)
    if n == 0 or s.maxCount == 0:
        return []
    k = s.minCount if s.minCount > 0 else 1
    k = min(k, s.maxCount, n)
    return list(range(k))


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
        try:
            obs = to_observation_class(obs_dict)
            if obs.select is None:
                return _load_deck()
            return _random_move(obs.select)
        except Exception:
            return []
