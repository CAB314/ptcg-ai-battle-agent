"""greedy_lethal — greedy_first に「KOを取り逃さない攻撃選択」を足したエージェント。

方針（713608 の安全則: エンジンの option[0] は既に強い基準なので、確信できる場面だけ上書き）:
- 攻撃系オプション(OptionType.ATTACK)があるとき、all_attack() のダメージと相手バトル場の
  現HPを見て、**リーサル（打点 >= 相手現HP）を取れるなら最大打点の攻撃を選ぶ**。純粋な上積み。
- 専用の攻撃選択(SelectType.ATTACK)では、リーサルが無くても最大打点の攻撃を選ぶ。
- それ以外は greedy（エンジンの先頭＝最良手）にフォールバック。

注意（v1 の割り切り）: 弱点/抵抗力は未考慮の素点で判定。弱点は打点が増える方向なので
リーサルを「見逃す」ことはあっても「空振り」は稀（抵抗力持ち相手のみ注意）。効果ダメージ
（"エネルギー1個につき+10" 等）も素点のみ。ここは後で精緻化する。

self-contained / __file__ 非依存 / fail-closed は厳守。src/ptcg には依存しない。
"""

import os
import random

from cg.api import OptionType, SelectType, all_attack, to_observation_class

_DECK_CACHE = None
_ATK_DMG = None  # attackId -> damage


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


def _opponent_active_hp(obs):
    st = obs.current
    opp = st.players[1 - st.yourIndex]
    if opp.active and opp.active[0] is not None:
        return opp.active[0].hp
    return None


def _best_attack(obs):
    """(index, damage, is_lethal) を返す。攻撃オプションが無ければ None。"""
    dmg_map = _attack_damage()
    opp_hp = _opponent_active_hp(obs)
    best = None  # (is_lethal, damage, index)
    for i, o in enumerate(obs.select.option):
        if o.type != OptionType.ATTACK or o.attackId is None:
            continue
        dmg = dmg_map.get(o.attackId, 0)
        lethal = opp_hp is not None and dmg > 0 and dmg >= opp_hp
        key = (1 if lethal else 0, dmg, -i)
        if best is None or key > best[0]:
            best = (key, i, dmg, lethal)
    if best is None:
        return None
    _, i, dmg, lethal = best
    return i, dmg, lethal


def decide(obs):
    s = obs.select
    best = _best_attack(obs)
    if best is not None:
        i, dmg, lethal = best
        # 専用の攻撃選択、またはリーサルが見えているときだけ上書き
        if (s.type == SelectType.ATTACK or lethal) and s.minCount <= 1 <= s.maxCount:
            return [i]
    return _greedy(s)


def agent(obs_dict):
    try:
        obs = to_observation_class(obs_dict)
        if obs.select is None:
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
