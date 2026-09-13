# AUTO-VENDORED from src/ptcg/meta/archetypes.py — edit the source there and re-run scripts/vendor.py
"""相手アーキタイプ推定（観測の公開情報から）。

規約OK（711741）: 相手デッキ/アーキタイプの推定と、それに応じた打ち回しの変更は
ホスト推奨の "robust in-game decision-making"。固定なのは自分のデッキだけ。

このモジュールは **cg に依存しない純ロジック**にしてある（stdlib のみ）。そのため
開発・精度検証（src 側）でも、提出エージェントへ **そのまま vendoring** しても使える。
入力は cg.api.Observation 相当のオブジェクト（属性アクセスのみ、duck-typing）。

観測で見える相手情報: バトル場/ベンチのポケモン（進化元・付与エネ/道具含む）と捨札は
可視。手札は枚数のみ・サイドは伏せ。logs には PLAY/ATTACH/EVOLVE/ATTACK 等の cardId が
逐次流れる。試合が進むほど相手の署名カードが判明する。
"""

from __future__ import annotations

from dataclasses import dataclass, field

# アーキタイプの「署名カード」→ 重み。エース(ex/主力)ほど高重み。
# 同名でも収録違いで別 cardId のものは全て列挙する（相手がどの版を使うか不明なため）。
# 重み 3=エース/デッキの核, 2=進化中間/主要, 1=たね/準主要。
SIGNATURES: dict[str, dict[int, float]] = {
    "crustle_lo": {58: 3, 344: 2, 532: 2, 345: 2, 533: 2, 607: 1},          # Great Tusk / Dwebble / Crustle / Terrakion
    "archaludon": {190: 3, 170: 2, 840: 2, 169: 1, 839: 1, 992: 1, 666: 2, 57: 1},  # Archaludon ex / Duraludon / Cinderace / Relicanth
    "abomasnow": {723: 3, 419: 2, 418: 1, 722: 1, 721: 1},                   # Mega Abomasnow ex / Snover / Kyogre
    "lucario": {678: 3, 333: 1, 677: 1, 974: 1},                             # Mega Lucario ex / Riolu
    "dragapult": {121: 3, 120: 2, 119: 1},                                   # Dragapult ex / Drakloak / Dreepy
    "gardevoir": {747: 3, 746: 2, 745: 1},                                   # Mega Gardevoir ex / Kirlia / Ralts
    "charizard": {790: 3, 928: 3, 789: 2, 927: 2, 788: 1, 926: 1},           # Mega Charizard X/Y ex / Charmeleon / Charmander
    "gholdengo": {191: 2, 700: 2, 186: 1, 668: 1},                           # Gholdengo / Gimmighoul
    "miraidon": {313: 3, 957: 3, 87: 1},                                     # Miraidon ex / Miraidon
    "raging_bolt": {63: 3, 171: 1},                                          # Raging Bolt ex / Raging Bolt
}

# エース級（重み3）が見えたら確信度 1.0 とする正規化係数。
_ACE_WEIGHT = 3.0
# 「確信した」とみなす閾値（gating 用）。エースが見えている状態。
CONFIDENT_THRESHOLD = 0.99


@dataclass
class ArchetypeGuess:
    name: str  # 推定アーキタイプ名（判別不能は "unknown"）
    confidence: float  # 0..1（エースが見えれば ~1.0）
    score: float  # 最良アーキタイプの一致重み
    margin: float  # 最良 - 次点（曖昧さの指標）
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def confident(self) -> bool:
        return self.name != "unknown" and self.confidence >= CONFIDENT_THRESHOLD


def classify(card_ids) -> ArchetypeGuess:
    """観測された相手カード ID 集合からアーキタイプを推定する（純関数）。"""
    ids = set(card_ids)
    scores = {
        name: sum(w for cid, w in sig.items() if cid in ids)
        for name, sig in SIGNATURES.items()
    }
    best = max(scores, key=scores.get) if scores else "unknown"
    best_score = scores.get(best, 0.0)
    if best_score <= 0:
        return ArchetypeGuess("unknown", 0.0, 0.0, 0.0, scores)
    ordered = sorted(scores.values(), reverse=True)
    margin = ordered[0] - (ordered[1] if len(ordered) > 1 else 0.0)
    confidence = min(1.0, best_score / _ACE_WEIGHT)
    return ArchetypeGuess(best, confidence, best_score, margin, scores)


# ---- 観測からの相手カード抽出（cg.api.Observation を duck-typing） ----

def _mon_card_ids(mon) -> set:
    ids = {mon.id}
    for c in getattr(mon, "energyCards", None) or []:
        ids.add(c.id)
    for c in getattr(mon, "tools", None) or []:
        ids.add(c.id)
    for c in getattr(mon, "preEvolution", None) or []:
        ids.add(c.id)
    return ids


def visible_opponent_card_ids(obs, self_index: int | None = None) -> set:
    """現観測で可視な相手カード ID（バトル場/ベンチ/進化元/付与/捨札）。"""
    st = getattr(obs, "current", None)
    if st is None:
        return set()
    si = self_index if self_index is not None else st.yourIndex
    opp = 1 - si
    p = st.players[opp]
    ids: set = set()
    for mon in (p.active or []):
        if mon is not None:
            ids |= _mon_card_ids(mon)
    for mon in (p.bench or []):
        ids |= _mon_card_ids(mon)
    for c in (p.discard or []):
        ids.add(c.id)
    return ids


def opponent_log_card_ids(obs, self_index: int | None = None) -> set:
    """今回の logs に現れた相手 cardId（PLAY/ATTACH/EVOLVE/ATTACK など）。"""
    st = getattr(obs, "current", None)
    si = self_index if self_index is not None else (st.yourIndex if st else 0)
    opp = 1 - si
    ids: set = set()
    for lg in (getattr(obs, "logs", None) or []):
        if getattr(lg, "playerIndex", None) == opp and getattr(lg, "cardId", None):
            ids.add(lg.cardId)
    return ids


class ArchetypeTracker:
    """試合を通じて相手の可視カードを蓄積し、アーキタイプ推定を更新する。

    エージェントは自分の手番ごとに update(obs) を呼ぶだけ。obs は
    cg.api.to_observation_class(obs_dict) で得た Observation。
    """

    def __init__(self):
        self.seen: set = set()
        self.last: ArchetypeGuess = ArchetypeGuess("unknown", 0.0, 0.0, 0.0, {})

    def update(self, obs) -> ArchetypeGuess:
        self.seen |= visible_opponent_card_ids(obs)
        self.seen |= opponent_log_card_ids(obs)
        self.last = classify(self.seen)
        return self.last
