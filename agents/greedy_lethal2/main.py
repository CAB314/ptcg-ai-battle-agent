"""greedy_lethal2 — greedy_lethal の lethal 判定を「正確な打点」に強化したもの。

素点(all_attack の damage)だけでなく、カードデータから以下を反映して実効打点を出す:
- **弱点**: 攻撃側ポケモンのタイプ(energyType)＝相手の weakness なら ×2（SV仕様）。
- **抵抗力**: 攻撃側タイプ＝相手の resistance なら −30（SV標準値の仮定）。
- **ツール（Maximum Belt, id=1158）**: 攻撃側に付いていて相手が「ポケモンex」なら +50。
  ※ エンジンの実データでは `ex` フラグに Mega ex が含まれない（megaEx が別）ため、
    「ポケモンex」= `ex or megaEx` で判定する。
- 演算順: (素点 + ツール加算) → 弱点 ×2 → 抵抗 −30。

未対応（v1 の割り切り・要注意）: 上記以外のツール/特性/スタジアムによる増減、
「エネルギー1個につき+X」等の効果依存の可変打点、相手側の軽減効果。これらは
構造化データに無い（テキストのみ）ため未モデル化。よって実効打点は「タイプ＋Belt補正済みの素点」。

self-contained / __file__ 非依存 / fail-closed 厳守。src/ptcg には依存しない。
"""

import os
import random

from cg.api import OptionType, SelectType, all_attack, all_card_data, to_observation_class

MAXIMUM_BELT = 1158
RESISTANCE_AMOUNT = 30

_DECK_CACHE = None
_ATK_DMG = None
_CARDS = None


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


def _greedy(select):
    n = len(select.option)
    if n == 0 or select.maxCount == 0:
        return []
    k = select.minCount if select.minCount > 0 else 1
    k = min(k, select.maxCount, n)
    return list(range(k))


def _my_active(obs):
    st = obs.current
    me = st.players[st.yourIndex]
    if me.active and me.active[0] is not None:
        return me.active[0]
    return None


def _opp_active(obs):
    st = obs.current
    opp = st.players[1 - st.yourIndex]
    if opp.active and opp.active[0] is not None:
        return opp.active[0]
    return None


def _is_ex(card):
    return bool(getattr(card, "ex", False) or getattr(card, "megaEx", False))


def _effective_damage(base, attacker, defender):
    """タイプ相性・Maximum Belt を反映した実効打点。"""
    if base <= 0 or defender is None:
        return base
    def_card = _card(defender.id)
    atk_card = _card(attacker.id) if attacker is not None else None
    dmg = base
    # ツール加算（攻撃側）: Maximum Belt は相手が ex/megaEx なら +50
    if attacker is not None:
        tool_ids = {c.id for c in (attacker.tools or [])}
        if MAXIMUM_BELT in tool_ids and def_card is not None and _is_ex(def_card):
            dmg += 50
    # 弱点 / 抵抗力（攻撃側ポケモンのタイプ基準）
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


def decide(obs):
    s = obs.select
    best = _best_attack(obs)
    if best is not None:
        i, eff, lethal = best
        if (s.type == SelectType.ATTACK or lethal) and s.minCount <= 1 <= s.maxCount:
            return [i]
    return _greedy(s)


def agent(obs_dict):
    try:
        obs = to_observation_class(obs_dict)
        if obs.select is None:
            try:  # キャッシュを温めておく（初手選択は時間に余裕がある）
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
