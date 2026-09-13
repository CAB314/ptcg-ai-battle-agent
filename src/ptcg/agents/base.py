"""fail-closed なエージェント基底（ローカル評価用）。

重要な契約（Discussion より）:
- 例外を投げてはいけない。投げるとそのゲームは即負け → 必ず合法手を返す。
- 返り値は list[int]、要素は 0<=x<len(option)、重複なし、
  minCount <= len <= maxCount。
- 初手（obs.select is None）はデッキ 60枚の card ID を返す。

提出用 main.py はこのクラスに import 依存せず、同じ安全パターンを
インラインで持たせる（Kaggle の import 事故を避けるため self-contained にする）。
本ファイルはその「正典」であり、評価ハーネスや単体テストで使う。
"""

from __future__ import annotations

import random
from typing import Sequence

from ..engine import ensure_cg_importable

ensure_cg_importable()
from cg.api import Observation, SelectData, to_observation_class  # noqa: E402


def validate_move(move, select: SelectData) -> bool:
    """返り値が選択契約を満たすか。"""
    if not isinstance(move, list) or not all(isinstance(x, int) for x in move):
        return False
    n = len(select.option)
    if not all(0 <= x < n for x in move):
        return False
    if len(set(move)) != len(move):
        return False
    return select.minCount <= len(move) <= select.maxCount


def random_move(select: SelectData) -> list[int]:
    """契約を満たすランダムな選択。maxCount 個を重複なく選ぶ。"""
    n = len(select.option)
    k = min(select.maxCount, n)
    k = max(k, select.minCount)
    return random.sample(range(n), k)


class Agent:
    """decide() を実装するだけで fail-closed になる基底クラス。"""

    def __init__(self, deck: Sequence[int]):
        self.deck = list(deck)

    def decide(self, obs: Observation) -> list[int]:
        """戦略本体。obs.select は None でない。サブクラスで実装。"""
        raise NotImplementedError

    def __call__(self, obs_dict: dict) -> list[int]:
        try:
            obs = to_observation_class(obs_dict)
            if obs.select is None:
                return list(self.deck)
            move = self.decide(obs)
            if validate_move(move, obs.select):
                return move
            return random_move(obs.select)
        except Exception:
            # 何が起きても合法手を返す（fail-closed）
            try:
                obs = to_observation_class(obs_dict)
                if obs.select is None:
                    return list(self.deck)
                return random_move(obs.select)
            except Exception:
                return []


class RandomAgent(Agent):
    """全選択をランダムに行うベースライン。評価の下限比較に使う。"""

    def decide(self, obs: Observation) -> list[int]:
        return random_move(obs.select)
