"""自己対戦による勝率評価。

評価規律（Discussion 713608 より）を意識すること:
- 短時間評価は過大に出る。改善判定は **>=400 ゲーム** で再検証。
- ノイズ床は概ね ±1.4pp（単一マッチアップ N~1200）。それ以下の差はノイズ。
- エンジンに CRN（共通乱数）は無く全対戦が独立サンプル。
- C++ シミュレータに長時間実行のメモリリークあり。大量対戦時は
  scripts/run_eval.py の --restart-every で子プロセスを分割する。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from ..engine import Agent, play_game


def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """勝率の点推定と Wilson 95% 信頼区間 (p, lo, hi)。"""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = wins / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, center - half, center + half


@dataclass
class EvalResult:
    games: int
    wins: int
    losses: int
    draws: int

    @property
    def winrate(self) -> float:
        return self.wins / self.games if self.games else 0.0

    @property
    def wilson(self) -> tuple[float, float, float]:
        # 引き分けは分母に含めるが勝ちには数えない
        return wilson_interval(self.wins, self.games)

    def __str__(self) -> str:
        p, lo, hi = self.wilson
        return (
            f"{self.wins}W-{self.losses}L-{self.draws}D / {self.games}  "
            f"winrate={p * 100:.1f}% [95% CI {lo * 100:.1f}–{hi * 100:.1f}]"
        )


def evaluate(
    agent: Agent,
    opponent: Agent,
    deck: Sequence[int],
    opponent_deck: Sequence[int] | None = None,
    games: int = 100,
    alternate_first: bool = True,
) -> EvalResult:
    """`agent` を `opponent` と対戦させ、`agent` 視点の成績を返す。

    alternate_first=True で先攻/後攻を交互に入れ替え、手番の有利不利を平均化する。
    """
    opponent_deck = list(opponent_deck) if opponent_deck is not None else list(deck)
    deck = list(deck)
    wins = losses = draws = 0
    for g in range(games):
        agent_is_first = (g % 2 == 0) or not alternate_first
        if agent_is_first:
            result = play_game(agent, opponent, deck, opponent_deck)
            agent_seat = 0
        else:
            result = play_game(opponent, agent, opponent_deck, deck)
            agent_seat = 1
        if result == 2:
            draws += 1
        elif result == agent_seat:
            wins += 1
        else:
            losses += 1
    return EvalResult(games=games, wins=wins, losses=losses, draws=draws)
