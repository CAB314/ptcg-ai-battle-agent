"""kangaskhan_playbook2 — kangaskhan(実メタデッキ) + D3移植パターン + 狭い条件分岐。

土台は実績ある greedy + 正確打点リーサル（弱点×2/抵抗−30/Maximum Belt/ex被ダメ無効）。
追加分岐（playbook.py のフラグで個別ON/OFF・各400戦A/Bで採否）:
  1. ガスト優先: 相手の Dwebble（壁の前身）と Dusknoir 線（天敵Starmie型のスナイプ）を
     「KOできる>低HP」コアの上に加点（D3のピッピ最優先の移植）。
  2. 壁検出時のみの前出し変更: 相手の場に ex技無効壁がいるときだけ、SWITCH/TO_ACTIVE で
     非exの自Crustle を優先しメガガルーラを減点。壁不在時は一切上書きしない
     （包括promoteは−22ptで失敗済み、狭い発火条件が差別化点）。
  3. 先攻選択: 計測の結果エンジン[0]=YES=先攻を既に選んでいる → 実装不要（確認済み）。
  4. 死にアクティブ入れ替え: 自アクティブの全ワザが相手アクティブに実効0（壁ロック等の
     証明可能な手詰まり）のときだけ、MAIN で ポケモンいれかえ/にげる を優先。

self-contained / __file__ 非依存 / fail-closed。
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

MAXIMUM_BELT = 1158
RESISTANCE_AMOUNT = 30

_DECK_CACHE = None
_ATK_DMG = None
_CARDS = None

_PROMOTE_CONTEXTS = {SelectContext.SWITCH, SelectContext.TO_ACTIVE}
_COUNTER_CONTEXTS = {SelectContext.DAMAGE_COUNTER, SelectContext.DAMAGE_COUNTER_ANY}
_OFFENSE_TARGET_CONTEXTS = {SelectContext.EFFECT_TARGET, SelectContext.DAMAGE} | _COUNTER_CONTEXTS


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
    n = len(select.option)
    if n == 0 or select.maxCount == 0:
        return 0
    k = select.minCount if select.minCount > 0 else 1
    return min(k, select.maxCount, n)


def _greedy(select):
    k = _pick_k(select)
    return list(range(k))


def _me(obs):
    return obs.current.players[obs.current.yourIndex]


def _opp(obs):
    return obs.current.players[1 - obs.current.yourIndex]


def _opp_mons(obs):
    op = _opp(obs)
    return [m for m in list(op.active or []) + list(op.bench or []) if m is not None]


def _mon_at(obs, player_index, area, index):
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
    me = _me(obs)
    if me.active and me.active[0] is not None:
        return me.active[0]
    return None


def _opp_active(obs):
    op = _opp(obs)
    if op.active and op.active[0] is not None:
        return op.active[0]
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


# ---- 分岐1: ガスト優先つき敵側スコア ----

def _enemy_score(obs, o, budget=None):
    opp_index = 1 - obs.current.yourIndex
    if o.playerIndex is not None and o.playerIndex != opp_index:
        return -1000
    mon = _mon_at(obs, opp_index, o.area, o.index)
    if mon is None:
        return -100
    if budget is None:
        dmg_map = _attack_damage()
        attacker = _my_active(obs)
        budget = 0
        if attacker is not None:
            atk_card = _card(attacker.id)
            for aid in (getattr(atk_card, "attacks", None) or []):
                budget = max(budget, _effective_damage(dmg_map.get(aid, 0), attacker, mon))
    ko = budget > 0 and budget >= mon.hp
    bonus = playbook.GUST_PRIORITY.get(mon.id, 0) if playbook.ENABLE_GUST_PRIORITY else 0
    return (1000 if ko else 0) + bonus + (500 - min(mon.hp, 499))


# ---- 分岐2: 壁検出時のみの前出し ----

def _wall_on_opp_board(obs):
    return any(_ex_damage_immune(_card(m.id)) for m in _opp_mons(obs))


def _wall_promote_score(obs, o):
    mon = _mon_at(obs, obs.current.yourIndex, o.area, o.index)
    if mon is None:
        return -100
    return playbook.WALL_PROMOTE_PRIORITY.get(mon.id, 0) + min(len(mon.energies or []), 3)


# ---- 分岐4: 死にアクティブの入れ替え ----

def _active_is_dead(obs):
    """真の壁ロックか: 相手アクティブが ex技無効 かつ 自アクティブが ex/megaEx。

    当初「全ワザ実効0」で判定したら、Dwebble の Ascension（打点0だが山から進化する
    優良ワザ）を手詰まりと誤判定して Switch を浪費し、対Starmieが 59.7→50.7% に悪化。
    打点0でも効果ワザには価値があるため、条件を証明可能な壁ロックのみに絞る。
    """
    attacker = _my_active(obs)
    defender = _opp_active(obs)
    if attacker is None or defender is None:
        return False
    if not _is_ex(_card(attacker.id)):
        return False
    return _ex_damage_immune(_card(defender.id))


def _dead_active_switch_move(obs):
    s = obs.select
    if s.type != SelectType.MAIN or not _active_is_dead(obs):
        return None
    hand = _me(obs).hand or []
    retreat_i = None
    for i, o in enumerate(s.option):
        if o.type == OptionType.PLAY and o.index is not None and 0 <= o.index < len(hand):
            if hand[o.index].id == playbook.SWITCH_CARD:
                return [i]  # ポケモンいれかえ（エネ消費なし）を最優先
        elif o.type == OptionType.RETREAT and retreat_i is None:
            retreat_i = i
    if retreat_i is not None:
        return [retreat_i]
    return None


def decide(obs):
    s = obs.select
    # 攻撃: 正確打点でリーサル/最大打点（実績パターン）
    best = _best_attack(obs)
    if best is not None:
        i, eff, lethal = best
        if (s.type == SelectType.ATTACK or lethal) and s.minCount <= 1 <= s.maxCount:
            return [i]
    # 分岐4: 死にアクティブなら入れ替えを優先
    if playbook.ENABLE_DEAD_ACTIVE_SWITCH:
        move = _dead_active_switch_move(obs)
        if move is not None:
            return move
    # CARD 選択
    if s.type == SelectType.CARD:
        ctx = s.context
        scorer = None
        if (
            playbook.ENABLE_WALL_PROMOTE
            and ctx in _PROMOTE_CONTEXTS
            and _wall_on_opp_board(obs)
        ):
            scorer = lambda o: _wall_promote_score(obs, o)  # noqa: E731
        elif ctx in _OFFENSE_TARGET_CONTEXTS:
            budget = None
            if ctx in _COUNTER_CONTEXTS:
                budget = max(0, (s.remainDamageCounter or 0)) * 10
            scorer = lambda o: _enemy_score(obs, o, budget)  # noqa: E731
        if scorer is not None:
            k = _pick_k(s)
            if k <= 0:
                return []
            order = sorted(range(len(s.option)), key=lambda i: (scorer(s.option[i]), -i), reverse=True)
            return order[:k]
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
