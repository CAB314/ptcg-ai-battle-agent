"""証明可能ガード + ルール fallback（kangaskhan_playbook2 から移植・デッキ非依存）。

- ガード: 正確打点リーサル（弱点×2/抵抗−30/ex被ダメ無効壁）の即取りのみ。
- rules_decide: greedy（エンジンの最良順 [0]）+ リーサル。numpy 不在/モデル失敗時の主戦力。
fail-closed の最終フォールバック（random_move/validate）もここに置く。
"""

import random

from cg.api import AreaType, OptionType, SelectType, all_attack, all_card_data

RESISTANCE_AMOUNT = 30

_ATKS = None
_CARDS = None


def attacks():
    global _ATKS
    if _ATKS is None:
        _ATKS = {a.attackId: a for a in all_attack()}
    return _ATKS


def card(cid):
    global _CARDS
    if _CARDS is None:
        _CARDS = {c.cardId: c for c in all_card_data()}
    return _CARDS.get(cid)


def random_move(select):
    n = len(select.option)
    k = max(min(select.maxCount, n), select.minCount)
    return random.sample(range(n), k)


def is_valid(move, select):
    n = len(select.option)
    return (
        isinstance(move, list)
        and all(isinstance(x, int) and 0 <= x < n for x in move)
        and len(set(move)) == len(move)
        and select.minCount <= len(move) <= select.maxCount
    )


def pick_k(select):
    n = len(select.option)
    if n == 0 or select.maxCount == 0:
        return 0
    k = select.minCount if select.minCount > 0 else 1
    return min(k, select.maxCount, n)


def greedy(select):
    return list(range(pick_k(select)))


def _me(obs):
    return obs.current.players[obs.current.yourIndex]


def _opp(obs):
    return obs.current.players[1 - obs.current.yourIndex]


def my_active(obs):
    me = _me(obs)
    if me.active and me.active[0] is not None:
        return me.active[0]
    return None


def opp_active(obs):
    op = _opp(obs)
    if op.active and op.active[0] is not None:
        return op.active[0]
    return None


def _is_ex(c):
    return bool(getattr(c, "ex", False) or getattr(c, "megaEx", False))


def _ex_damage_immune(def_card):
    """相手 ex のワザダメージを全て無効にする壁か（イワパレス型）。

    2026-08-02 修正: 限定語を見ていなかったため、Farigiraf ex（相手の**たね**ex のみ無効）
    や Shaymin（**ベンチ**を守る＝自分が殴られる話ではない）まで壁と誤判定していた。
    偽陽性は「取れるリーサルを取らない」に直結する。src/ptcg/ml/vocab.py の
    _ex_dmg_immune と同一条件に揃えてある。
    """
    if def_card is None:
        return False
    for s in (getattr(def_card, "skills", None) or []):
        t = (getattr(s, "text", "") or "").lower().replace("’", "'")
        prevents = "prevent all damage" in t or "no damage" in t
        vs_ex = "{ex}" in t or "pokémon ex" in t or "pokemon ex" in t
        to_self = "to this pok" in t
        limited = "basic pok" in t or "benched" in t
        if prevents and vs_ex and to_self and not limited:
            return True
    return False


def effective_damage(base, attacker, defender):
    if base <= 0 or defender is None:
        return max(0, base)
    def_card = card(defender.id)
    atk_card = card(attacker.id) if attacker is not None else None
    if _is_ex(atk_card) and _ex_damage_immune(def_card):
        return 0
    dmg = base
    atk_type = getattr(atk_card, "energyType", None) if atk_card else None
    if def_card is not None and atk_type is not None:
        if getattr(def_card, "weakness", None) == atk_type:
            dmg *= 2
        elif getattr(def_card, "resistance", None) == atk_type:
            dmg = max(0, dmg - RESISTANCE_AMOUNT)
    return dmg


def lethal_attack(obs):
    """正確打点で相手アクティブを一撃KOできる ATTACK option の index（無ければ None）。"""
    s = obs.select
    attacker = my_active(obs)
    defender = opp_active(obs)
    if attacker is None or defender is None:
        return None
    dmg_map = attacks()
    best = None
    for i, o in enumerate(s.option):
        if o.type != OptionType.ATTACK or o.attackId is None:
            continue
        a = dmg_map.get(o.attackId)
        base = (a.damage or 0) if a is not None else 0
        eff = effective_damage(base, attacker, defender)
        if eff > 0 and eff >= defender.hp:
            if best is None or eff > best[1]:
                best = (i, eff)
    return best[0] if best is not None else None


def rules_decide(obs):
    """ルール fallback: リーサル > greedy [0]（エンジンの最良順）。"""
    s = obs.select
    if s.type in (SelectType.MAIN, SelectType.ATTACK) and s.minCount <= 1 <= s.maxCount:
        i = lethal_attack(obs)
        if i is not None:
            return [i]
    return greedy(s)
