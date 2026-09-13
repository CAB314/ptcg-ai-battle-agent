"""lucario_playbook — 提案書D7メガルカリオを「回し方」プレイブックで回すエージェント。

設計方針（実験ログの教訓: 強い greedy 基準を不確実な推測で上書きすると退化する）:
- 行動タイプの選択（出す/進化/攻撃/エンド等）は従来通り **エンジン最良手順(greedy) + 正確打点
  リーサル**（弱点×2/抵抗−30/Maximum Belt/ex被ダメ無効の考慮）に任せる。
- プレイブック(playbook.py)が上書きするのは **エンジンの静的順序では知り得ない
  「デッキ依存のターゲティング」だけ**:
    1. 手貼り/効果でのエネルギーの付け先 → 主力アタッカー優先（ATTACH の inPlay 対象、
       ATTACH_FROM 選択）。ルナトーン等のエンジン部品には付けない。
    2. バトル場に出すポケモン（SWITCH / TO_ACTIVE）→ アタッカー優先・部品回避。
    3. 相手を対象に取る効果（ボスの指令等の EFFECT_TARGET）とダメージ/ダメカン配分
       → 「KOできる相手 > 残りHPの低い相手」。
- メガブレイブの連続使用不可は、エンジンが使用不能な攻撃を候補に出さないため対処不要。
  2体交互運用のためのリトリート最適化は v1 では見送り（退化リスク > 期待値）。

self-contained / __file__ 非依存 / fail-closed。playbook.py を同梱（tar トップレベル）。
"""

import os
import random

from cg.api import (
    AreaType,
    OptionType,
    SelectContext,
    SelectType,
    all_attack,
    all_card_data,
    to_observation_class,
)

import playbook
from playbook import ATTACKER_PRIORITY, SUPPORT_POKEMON

# プレイブックの適用範囲フラグ（playbook.py で上書き可能）。
# ALLY = エネ付け先/バトル場に出す先の上書き（デッキが直線的なときだけ有効にする）
# ENEMY = 相手対象（ボス/ダメージ配分）の「KOできる>低HP」上書き
OVERRIDE_ALLY = getattr(playbook, "OVERRIDE_ALLY", True)
OVERRIDE_ENEMY = getattr(playbook, "OVERRIDE_ENEMY", True)

MAXIMUM_BELT = 1158
RESISTANCE_AMOUNT = 30

_DECK_CACHE = None
_ATK_DMG = None
_CARDS = None

# ターゲティングを上書きする CARD 選択の context
_PROMOTE_CONTEXTS = {SelectContext.SWITCH, SelectContext.TO_ACTIVE}
_ATTACH_CONTEXTS = {SelectContext.ATTACH_FROM}
_OFFENSE_TARGET_CONTEXTS = {
    SelectContext.EFFECT_TARGET,
    SelectContext.DAMAGE,
    SelectContext.DAMAGE_COUNTER,
    SelectContext.DAMAGE_COUNTER_ANY,
}


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


def _attack_damage():
    global _ATK_DMG
    if _ATK_DMG is None:
        _ATK_DMG = {a.attackId: (a.damage or 0) for a in all_attack()}
    return _ATK_DMG


def _card(cid):
    global _CARDS
    if _CARDS is None:
        _CARDS = {c.cardId: c for c in all_card_data()}
    return _CARDS.get(cid)


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


def _pick_k(select):
    """greedy と同じ選択数ポリシー（実績ある挙動を変えない）。"""
    n = len(select.option)
    if n == 0 or select.maxCount == 0:
        return 0
    k = select.minCount if select.minCount > 0 else 1
    return min(k, select.maxCount, n)


def _greedy(select):
    k = _pick_k(select)
    return list(range(k))


def _mon_at(obs, player_index, area, index):
    """option の (playerIndex, area, index) から Pokemon を引く。無ければ None。"""
    try:
        p = obs.current.players[player_index]
    except Exception:
        return None
    if area == AreaType.ACTIVE:
        arr = p.active or []
        return arr[0] if arr else None
    if area == AreaType.BENCH:
        arr = p.bench or []
        return arr[index] if index is not None and 0 <= index < len(arr) else None
    return None


def _my_active(obs):
    me = obs.current.players[obs.current.yourIndex]
    if me.active and me.active[0] is not None:
        return me.active[0]
    return None


def _opp_active(obs):
    opp = obs.current.players[1 - obs.current.yourIndex]
    if opp.active and opp.active[0] is not None:
        return opp.active[0]
    return None


def _is_ex(card):
    return bool(getattr(card, "ex", False) or getattr(card, "megaEx", False))


def _ex_damage_immune(def_card):
    if def_card is None:
        return False
    for s in (getattr(def_card, "skills", None) or []):
        t = (getattr(s, "text", "") or "").lower()
        prevents = "prevent all damage" in t or "no damage" in t
        vs_ex = "{ex}" in t or "pokémon ex" in t or "pokemon ex" in t
        if prevents and vs_ex:
            return True
    return False


def _effective_damage(base, attacker, defender):
    if base <= 0 or defender is None:
        return max(0, base)
    def_card = _card(defender.id)
    atk_card = _card(attacker.id) if attacker is not None else None
    if _is_ex(atk_card) and _ex_damage_immune(def_card):
        return 0
    dmg = base
    if attacker is not None:
        tools = {c.id for c in (attacker.tools or [])}
        if MAXIMUM_BELT in tools and def_card is not None and _is_ex(def_card):
            dmg += 50
    atk_type = getattr(atk_card, "energyType", None) if atk_card else None
    if def_card is not None and atk_type is not None:
        if getattr(def_card, "weakness", None) == atk_type:
            dmg *= 2
        elif getattr(def_card, "resistance", None) == atk_type:
            dmg = max(0, dmg - RESISTANCE_AMOUNT)
    return dmg


def _best_attack(obs):
    dmg_map = _attack_damage()
    attacker = _my_active(obs)
    defender = _opp_active(obs)
    opp_hp = defender.hp if defender is not None else None
    best = None
    for i, o in enumerate(obs.select.option):
        if o.type != OptionType.ATTACK or o.attackId is None:
            continue
        eff = _effective_damage(dmg_map.get(o.attackId, 0), attacker, defender)
        lethal = opp_hp is not None and eff > 0 and eff >= opp_hp
        key = (1 if lethal else 0, eff, -i)
        if best is None or key > best[0]:
            best = (key, i, eff, lethal)
    if best is None:
        return None
    _, i, eff, lethal = best
    return i, eff, lethal


# ---- プレイブック: デッキ依存ターゲティング ----

def _ally_priority(mon):
    """自分側ポケモンの優先度（エネ付け先/バトル場に出す先）。"""
    if mon is None:
        return -100
    if mon.id in SUPPORT_POKEMON:
        return -10  # エンジン部品には付けない/出さない
    base = ATTACKER_PRIORITY.get(mon.id, 0)
    # 進化元にエネが乗っている場合など、既に投資済みの個体をやや優先
    return base + min(len(mon.energies or []), 3)


def _reorder_main_attach(obs):
    """MAIN で先頭が ATTACH のとき、同タイプ内で付け先の良い option に差し替える。"""
    s = obs.select
    first = s.option[0]
    if first.type != OptionType.ATTACH:
        return None
    best = None
    for i, o in enumerate(s.option):
        if o.type != OptionType.ATTACH:
            continue
        mon = _mon_at(obs, obs.current.yourIndex, o.inPlayArea, o.inPlayIndex)
        score = (_ally_priority(mon), -i)
        if best is None or score > best[0]:
            best = (score, i)
    return [best[1]] if best is not None else None


def _score_ally_card_option(obs, o):
    mon = _mon_at(obs, obs.current.yourIndex, o.area, o.index)
    return _ally_priority(mon)


def _score_enemy_target(obs, o):
    """相手対象: KOできる相手を最優先、次に残りHPが低い順。"""
    opp_index = 1 - obs.current.yourIndex
    if o.playerIndex is not None and o.playerIndex != opp_index:
        return -1000  # 自分側が混ざる選択では相手を優先しない（安全側）
    mon = _mon_at(obs, opp_index, o.area, o.index)
    if mon is None:
        return -100
    dmg_map = _attack_damage()
    attacker = _my_active(obs)
    best_eff = 0
    if attacker is not None:
        atk_card = _card(attacker.id)
        for aid in (getattr(atk_card, "attacks", None) or []):
            best_eff = max(best_eff, _effective_damage(dmg_map.get(aid, 0), attacker, mon))
    ko = best_eff > 0 and best_eff >= mon.hp
    return (1000 if ko else 0) + (500 - min(mon.hp, 499))


def _reorder_card_select(obs):
    """CARD 選択で context に応じたターゲティング上書き。該当しなければ None。"""
    s = obs.select
    ctx = s.context
    if OVERRIDE_ALLY and (ctx in _PROMOTE_CONTEXTS or ctx in _ATTACH_CONTEXTS):
        scorer = lambda o: _score_ally_card_option(obs, o)  # noqa: E731
    elif OVERRIDE_ENEMY and ctx in _OFFENSE_TARGET_CONTEXTS:
        scorer = lambda o: _score_enemy_target(obs, o)  # noqa: E731
    else:
        return None
    k = _pick_k(s)
    if k <= 0:
        return []
    scored = sorted(
        range(len(s.option)), key=lambda i: (scorer(s.option[i]), -i), reverse=True
    )
    return scored[:k]


def decide(obs):
    s = obs.select
    # 1) 攻撃: 正確打点でリーサル/最大打点（従来ロジック）
    best = _best_attack(obs)
    if best is not None:
        i, eff, lethal = best
        if (s.type == SelectType.ATTACK or lethal) and s.minCount <= 1 <= s.maxCount:
            return [i]
    # 2) MAIN の手貼り: 付け先をプレイブックで選ぶ（ALLY 上書きが有効なときだけ）
    if OVERRIDE_ALLY and s.type == SelectType.MAIN and len(s.option) > 0:
        move = _reorder_main_attach(obs)
        if move is not None:
            return move
    # 3) CARD 選択のターゲティング（付け先/出す先/相手対象/配分）
    if s.type == SelectType.CARD:
        move = _reorder_card_select(obs)
        if move is not None:
            return move
    # 4) それ以外は greedy
    return _greedy(s)


def agent(obs_dict):
    try:
        obs = to_observation_class(obs_dict)
        if obs.select is None:
            try:
                _attack_damage()
                _card(0)
            except Exception:
                pass
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
