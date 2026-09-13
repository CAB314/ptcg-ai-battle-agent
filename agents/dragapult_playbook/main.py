"""dragapult_playbook — 提案D3ガルーラ・ドラパルトを詳細戦術シートに従って回すエージェント。

土台は実績ある greedy + 正確打点リーサル（弱点×2/抵抗−30/Maximum Belt/ex被ダメ無効）。
その上に D3 の詳細戦術（playbook.py 参照）をデッキ依存の選択にだけ適用する:
  1. バトル場に出す先: ガルーラ(2ドロー)/ラティアス(逃げ0)優先、部品は出さない。
     ピッピ(#272)が相手の場にいる間はドラゴンを前に出す優先度を下げる。
  2. 相手対象(ボス等): KOできる相手 > 低HP。ピッピは最優先(+2000)。
     ダメカン配分(ファントムダイブの6個/マシマシラ)は「残カウンタで取り切れる相手」を優先。
  3. 進化ベト: 最後のドロンチは（ドラパルトが既に立っていれば）進化させず継続ドローを残す。
  4. エネの付け先は多色のためエンジン既定に任せる（OVERRIDE_ATTACH=False）。

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
_ATTACH_CONTEXTS = {SelectContext.ATTACH_FROM}
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


def _me(obs):
    return obs.current.players[obs.current.yourIndex]


def _opp(obs):
    return obs.current.players[1 - obs.current.yourIndex]


def _my_mons(obs):
    me = _me(obs)
    return [m for m in list(me.active or []) + list(me.bench or []) if m is not None]


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


# ---- D3 デッキ知識 ----

def _clefairy_on_opp_board(obs):
    return any(m.id == playbook.CLEFAIRY_EX for m in _opp_mons(obs))


def _promote_score(obs, mon):
    if mon is None:
        return -100
    if mon.id in playbook.SUPPORT_POKEMON:
        return -10
    score = playbook.PROMOTE_PRIORITY.get(mon.id, 0)
    c = _card(mon.id)
    # ピッピ存在時はドラゴンを前に出さない（超弱点を突かれる）
    if (
        c is not None
        and getattr(c, "energyType", None) == playbook.DRAGON_TYPE
        and _clefairy_on_opp_board(obs)
    ):
        score -= playbook.DRAGON_PENALTY_WHEN_CLEFAIRY
    return score + min(len(mon.energies or []), 3)


def _enemy_score(obs, o, budget=None):
    """KOできる相手を最優先、次にガスト優先(ピッピ)、次に低HP。"""
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
    bonus = playbook.GUST_PRIORITY.get(mon.id, 0)
    return (1000 if ko else 0) + bonus + (500 - min(mon.hp, 499))


def _drakloak_evolve_veto(obs, o):
    """最後のドロンチは（ドラパルトが既に立っていれば）進化させない。"""
    if o.type != OptionType.EVOLVE:
        return False
    target = _mon_at(obs, obs.current.yourIndex, o.inPlayArea, o.inPlayIndex)
    if target is None or target.id != playbook.DRAKLOAK:
        return False
    mons = _my_mons(obs)
    n_drakloak = sum(1 for m in mons if m.id == playbook.DRAKLOAK)
    n_dragapult = sum(1 for m in mons if m.id == playbook.DRAGAPULT_EX)
    return n_drakloak <= 1 and n_dragapult >= 1


def _greedy_with_veto(obs):
    """greedy だが、ベトされた option はスキップ（代替が無ければ従来通り）。"""
    s = obs.select
    k = _pick_k(s)
    if k <= 0:
        return []
    if k == 1:
        for i, o in enumerate(s.option):
            if not _drakloak_evolve_veto(obs, o):
                return [i]
        return [0]
    return list(range(k))


def decide(obs):
    s = obs.select
    # 1) 攻撃: 正確打点でリーサル/最大打点
    best = _best_attack(obs)
    if best is not None:
        i, eff, lethal = best
        if (s.type == SelectType.ATTACK or lethal) and s.minCount <= 1 <= s.maxCount:
            return [i]
    # 2) CARD 選択のターゲティング
    if s.type == SelectType.CARD:
        ctx = s.context
        scorer = None
        if playbook.OVERRIDE_PROMOTE and ctx in _PROMOTE_CONTEXTS:
            scorer = lambda o: _promote_score(obs, _mon_at(obs, obs.current.yourIndex, o.area, o.index))  # noqa: E731
        elif playbook.OVERRIDE_ATTACH and ctx in _ATTACH_CONTEXTS:
            scorer = lambda o: _promote_score(obs, _mon_at(obs, obs.current.yourIndex, o.area, o.index))  # noqa: E731
        elif playbook.OVERRIDE_ENEMY and ctx in _OFFENSE_TARGET_CONTEXTS:
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
    # 3) それ以外は greedy（ドロンチ進化ベトつき）
    return _greedy_with_veto(obs)


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
