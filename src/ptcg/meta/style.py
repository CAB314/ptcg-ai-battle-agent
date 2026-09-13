"""デッキのスタイル特徴量と Aggro/Midrange/Control 分類（Issue #1 抜粋の実装）。

出典（docs/ptcg-usable-excerpts.md #5）:
- ろし: Aggro=「速いダメージソースで削りきる」/ Control=「負けない盤面を作った後に
  数少ない勝ち手段で長いターンを使う」
- Flipside: 特徴量=平均ワザ打点/進化段数/妨害・ドローサポ枚数/回収手段
- Designing a Deck: タイプ構成から傾向推定

archetypes.py と同じく **cg に依存しない純ロジック**（duck-typing）。カード/ワザの
lookup は呼び出し側が渡す（エンジン由来でも CSV 由来でも良い）。エージェントへの
vendoring 可。episodes は両者の完全60枚デッキを含むため、実メタ分析にも使える。
"""

from __future__ import annotations

from dataclasses import dataclass, field

# トレーナーテキストの役割キーワード（JustInBasil の役割4分類を拡張）
_KW_DRAW = ("draw",)
_KW_SEARCH = ("search your deck",)
_KW_DISRUPT = (
    "your opponent discards",
    "opponent's hand",
    "your opponent's deck",
    "can't play",
    "can't use",
)
_KW_GUST = ("switch in 1 of your opponent",)
_KW_HEAL = ("heal",)
_KW_RECOVER = ("from your discard pile",)


@dataclass
class StyleFeatures:
    n_pokemon: int = 0
    n_basic: int = 0
    n_stage1: int = 0
    n_stage2: int = 0
    n_evo_lines: int = 0
    n_rule_box: int = 0          # ex/megaEx（2-3プライズ risk）
    n_energy: int = 0
    avg_attacker_damage: float = 0.0  # アタッカー（打点>0のワザ持ち）の最大素点の平均
    top_damage: int = 0
    max_hp: int = 0
    n_draw: int = 0              # ドローサポ/グッズ
    n_search: int = 0
    n_disrupt: int = 0           # 手札/山への妨害
    n_gust: int = 0              # 呼び出し
    n_heal: int = 0
    n_recover: int = 0           # 回収
    n_mill: int = 0              # 相手の山を削るワザ/効果（LO シグナル）
    scores: dict = field(default_factory=dict)


def _norm(text: str) -> str:
    # エンジンのカードテキストは Unicode アポストロフィ（’）を使う。ASCII に正規化
    # しないとキーワードが一切ヒットしない（過去に grep でも踏んだ罠）。
    return (text or "").lower().replace("’", "'")


def _texts_of(card, attacks):
    out = []
    for s in getattr(card, "skills", None) or []:
        out.append(_norm(getattr(s, "text", "")))
    for aid in getattr(card, "attacks", None) or []:
        a = attacks.get(aid)
        if a is not None:
            out.append(_norm(getattr(a, "text", "")))
    return out


def style_features(deck_ids, cards, attacks) -> StyleFeatures:
    """60枚デッキ（またはカードID列）からスタイル特徴量を計算する純関数。

    cards: {cardId: CardData 相当}, attacks: {attackId: Attack 相当}。
    """
    f = StyleFeatures()
    attacker_damages = []
    stage1_names = set()
    stage2_names = set()
    seen_ids = set()
    for cid in deck_ids:
        c = cards.get(cid)
        if c is None:
            continue
        ctype = int(getattr(c, "cardType", -1))
        if ctype == 0:  # POKEMON
            f.n_pokemon += 1
            if getattr(c, "basic", False):
                f.n_basic += 1
            if getattr(c, "stage1", False):
                f.n_stage1 += 1
                stage1_names.add(getattr(c, "name", ""))
            if getattr(c, "stage2", False):
                f.n_stage2 += 1
                stage2_names.add(getattr(c, "evolvesFrom", "") or getattr(c, "name", ""))
            if getattr(c, "ex", False) or getattr(c, "megaEx", False):
                f.n_rule_box += 1
            f.max_hp = max(f.max_hp, int(getattr(c, "hp", 0) or 0))
            if cid not in seen_ids:
                dmg = max(
                    (int(getattr(attacks.get(aid), "damage", 0) or 0)
                     for aid in (getattr(c, "attacks", None) or [])
                     if attacks.get(aid) is not None),
                    default=0,
                )
                if dmg > 0:
                    attacker_damages.append(dmg)
                f.top_damage = max(f.top_damage, dmg)
        elif ctype in (5, 6):  # ENERGY
            f.n_energy += 1
        texts = _texts_of(c, attacks)
        for t in texts:
            if any(k in t for k in _KW_DRAW):
                f.n_draw += 1
            if any(k in t for k in _KW_SEARCH):
                f.n_search += 1
            if any(k in t for k in _KW_DISRUPT):
                f.n_disrupt += 1
            if any(k in t for k in _KW_GUST):
                f.n_gust += 1
            if any(k in t for k in _KW_HEAL):
                f.n_heal += 1
            if any(k in t for k in _KW_RECOVER):
                f.n_recover += 1
            if "your opponent's deck" in t and "discard" in t:
                f.n_mill += 1
        seen_ids.add(cid)
    # 進化ライン数 ≒ 1進化の種類数 + （対応1進化を持たない）2進化の系統数
    f.n_evo_lines = len(stage1_names | stage2_names)
    if attacker_damages:
        f.avg_attacker_damage = sum(attacker_damages) / len(attacker_damages)
    return f


def classify_style(deck_ids, cards, attacks):
    """(label, StyleFeatures) を返す。label は aggro / midrange / control。

    閾値は透明なヒューリスティック（実メタの再計測で調整する前提）:
    - control: ミル/妨害/回復が厚く、打点に依存しない勝ち筋を持つ
    - aggro: たね中心・高素点・進化が浅い（速いダメージソースで削りきる）
    - midrange: それ以外（進化で盤面を作り中打点で殴る）
    """
    f = style_features(deck_ids, cards, attacks)
    control_pts = 2.0 * f.n_mill + 1.0 * f.n_disrupt + 0.5 * f.n_heal + 0.5 * f.n_gust
    aggro_pts = (
        (2.0 if f.avg_attacker_damage >= 120 else 0.0)
        + (2.0 if f.n_stage2 == 0 else 0.0)
        + (1.0 if f.n_basic >= f.n_pokemon * 0.8 else 0.0)
        + (1.0 if f.top_damage >= 200 else 0.0)
    )
    f.scores = {"control": control_pts, "aggro": aggro_pts}
    if control_pts >= 8 and control_pts > aggro_pts:
        return "control", f
    if aggro_pts >= 4:
        return "aggro", f
    return "midrange", f
