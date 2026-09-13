"""多様な相手プールでの評価。

過学習対策の中核（Discussion 717697 / 713608 / 724187）:
- 単一相手（特にランダム単体）に合わせ込むとラダーに転移しない。
- そこで「複数の振る舞い × 複数デッキ（アーキタイプ）」の相手プールで評価し、
  相手ごとの内訳と集計勝率を見る。
- **これはあくまでスクリーニング**。採否の最終判断は実ラダーの COMPLETE スコアで。

相手 (Opponent) = (behavior, deck)。
- behavior: "random"（内蔵ランダム）または agents/<name>（その main.py の振る舞い）
- deck: decks/ 配下の csv 名 or パス

ローカル対戦では battle_start に渡すデッキと、エージェントの振る舞い（main.py）は
独立に指定できる（エージェントが deck.csv を読むのは初手選択のみで、ローカルでは
その分岐を通らない）。そのため 1 つの振る舞いを複数デッキと組ませて多様な相手を作れる。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

from ..agents.base import RandomAgent
from ..cards import read_deck_csv
from ..engine import REPO_ROOT, Agent, load_agent
from .harness import EvalResult, evaluate

AGENTS_DIR = REPO_ROOT / "agents"
DECKS_DIR = REPO_ROOT / "decks"

# 既定プールで使う「基準となる振る舞い」。いまはこれが唯一の非自明な手なので、
# デッキだけを振ってアーキタイプ横断の相性を測る。強い振る舞いが増えたら拡張する。
DEFAULT_BEHAVIOR = "greedy_first"


@dataclass
class Opponent:
    behavior: str  # "random" or agents/<name>
    deck: str  # decks/ の csv 名 or パス
    name: str = ""  # 表示名（未指定なら behavior@deck）

    def __post_init__(self):
        if not self.name:
            self.name = f"{self.behavior}@{Path(self.deck).stem}"


def resolve_deck_path(deck: str) -> Path:
    """デッキ指定をパスに解決する（絶対/相対、decks/ 名、拡張子省略に対応）。"""
    p = Path(deck)
    if p.exists():
        return p
    for cand in (DECKS_DIR / deck, DECKS_DIR / f"{deck}.csv"):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"デッキが見つかりません: {deck}")


def resolve_opponent(opp: Opponent) -> tuple[Agent, list[int]]:
    """Opponent から (呼び出し可能エージェント, デッキ) を得る。"""
    deck = read_deck_csv(resolve_deck_path(opp.deck))
    if opp.behavior == "random":
        return RandomAgent(deck), deck
    main_py = AGENTS_DIR / opp.behavior / "main.py"
    if not main_py.exists():
        raise FileNotFoundError(f"振る舞いが見つかりません: {main_py}")
    return load_agent(main_py), deck


def default_pool(behavior: str = DEFAULT_BEHAVIOR) -> list[Opponent]:
    """decks/ 配下の全デッキ × 基準振る舞い。デッキ横断のアーキタイプ場を作る。"""
    decks = sorted(DECKS_DIR.glob("*.csv"))
    if not decks:
        raise FileNotFoundError(f"{DECKS_DIR} にデッキがありません")
    return [Opponent(behavior=behavior, deck=str(d)) for d in decks]


def load_pool_config(path: str | Path) -> list[Opponent]:
    """JSON からプールを読む。形式: [{"behavior": "...", "deck": "...", "name": "..."}, ...]"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for item in data:
        behavior = item.get("behavior") or item.get("agent")
        out.append(Opponent(behavior=behavior, deck=item["deck"], name=item.get("name", "")))
    return out


@dataclass
class PoolMember:
    opponent: Opponent
    result: EvalResult


@dataclass
class PoolResult:
    agent_label: str
    members: list[PoolMember] = field(default_factory=list)

    @property
    def aggregate(self) -> EvalResult:
        return EvalResult(
            games=sum(m.result.games for m in self.members),
            wins=sum(m.result.wins for m in self.members),
            losses=sum(m.result.losses for m in self.members),
            draws=sum(m.result.draws for m in self.members),
        )

    def report(self) -> str:
        w = max([len(m.opponent.name) for m in self.members] + [len("opponent")])
        lines = [f"pool eval: {self.agent_label}", ""]
        lines.append(f"  {'opponent'.ljust(w)}  {'record':>12}  {'winrate':>8}  {'95% CI':>16}")
        lines.append(f"  {'-' * w}  {'-' * 12}  {'-' * 8}  {'-' * 16}")
        # 弱い相手順に見やすく（勝率降順）
        for m in sorted(self.members, key=lambda m: m.result.winrate, reverse=True):
            r = m.result
            p, lo, hi = r.wilson
            rec = f"{r.wins}-{r.losses}-{r.draws}"
            lines.append(
                f"  {m.opponent.name.ljust(w)}  {rec:>12}  {p * 100:>7.1f}%  "
                f"[{lo * 100:>5.1f}–{hi * 100:>5.1f}]"
            )
        agg = self.aggregate
        p, lo, hi = agg.wilson
        lines.append(f"  {'-' * w}  {'-' * 12}  {'-' * 8}  {'-' * 16}")
        lines.append(
            f"  {'AGGREGATE (equal-weight field)'.ljust(w)}  "
            f"{f'{agg.wins}-{agg.losses}-{agg.draws}':>12}  {p * 100:>7.1f}%  "
            f"[{lo * 100:>5.1f}–{hi * 100:>5.1f}]"
        )
        lines.append("")
        lines.append("  注: スクリーニング用。採否は実ラダーの COMPLETE スコアで判断。")
        lines.append("      改善判定は相手あたり >=400 戦、±1.4pp 以下はノイズとして無視。")
        return "\n".join(lines)


def evaluate_pool(
    agent_label: str,
    pool: Sequence[Opponent],
    games_per_opponent: int = 100,
    *,
    agent: Optional[Agent] = None,
    agent_deck: Optional[Sequence[int]] = None,
    matchup_runner: Optional[Callable[[Opponent], EvalResult]] = None,
    alternate_first: bool = True,
) -> PoolResult:
    """プール全体で評価する。

    in-process の場合は agent と agent_deck を渡す。サブプロセス分割など外部で
    対戦を回す場合は matchup_runner(opp)->EvalResult を渡す（agent は不要）。
    """
    if matchup_runner is None:
        if agent is None or agent_deck is None:
            raise ValueError("agent と agent_deck、または matchup_runner が必要です")
        agent_deck = list(agent_deck)

        def matchup_runner(opp: Opponent) -> EvalResult:  # noqa: F811
            opp_agent, opp_deck = resolve_opponent(opp)
            return evaluate(
                agent,
                opp_agent,
                agent_deck,
                opp_deck,
                games=games_per_opponent,
                alternate_first=alternate_first,
            )

    result = PoolResult(agent_label=agent_label)
    for opp in pool:
        result.members.append(PoolMember(opponent=opp, result=matchup_runner(opp)))
    return result
