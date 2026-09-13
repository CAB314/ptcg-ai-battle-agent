"""meta_aware — greedy_lethal に相手アーキタイプ推定によるゲーティングを足したもの。

- 相手の公開情報から ArchetypeTracker でアーキタイプを推定（規約OK・ホスト推奨）。
- ベースは greedy_lethal（KO取り逃さない攻撃 + エンジン最良手 greedy）。
- ゲーティング（v1）: 相手が **コントロール/LO 系（crustle_lo）** と確信できたら、
  リーサルが無い場面でも攻撃を選び **テンポ良くサイドを取りに行く**（遅いデッキには速攻が正着）。
  それ以外の相手では余計なことをせず greedy_lethal のまま。

推定器 archetypes.py は src/ptcg/meta から vendoring（scripts/vendor.py で同期）。
self-contained / __file__ 非依存 / fail-closed 厳守。tracker は「初手選択(select None)=新規ゲーム」で毎回リセット
（プロセス使い回しでも状態が混ざらないように）。
"""

import os
import random

from cg.api import OptionType, SelectType, all_attack, to_observation_class

from archetypes import ArchetypeTracker  # vendored (src/ptcg/meta/archetypes.py)

# 遅い/コントロール/ライブラリアウト系。
CONTROL_ARCHETYPES = {"crustle_lo"}

# v1 の「コントロール相手は無理攻め」ゲーティングは A/B で **退化** と判明したため既定 OFF。
#   対 crustle: greedy_lethal 93.5% → meta_aware(gating ON) 82%（200戦/相手, pool）。
#   head-to-head では相手が非コントロールで発火せず 50/50。
# 理由: greedy_lethal の [0]+lethal が既に正着で、そこを上書きすると悪化する（713608 と一致）。
# 推定器 (ArchetypeTracker) は毎ターン走らせており、より良いゲーティング/探索の土台として温存する。
ENABLE_TEMPO_GATING = False

_DECK_CACHE = None
_ATK_DMG = None
_TRACKER = None


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


def _tracker(reset=False):
    global _TRACKER
    if reset or _TRACKER is None:
        _TRACKER = ArchetypeTracker()
    return _TRACKER


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
    dmg_map = _attack_damage()
    opp_hp = _opponent_active_hp(obs)
    best = None
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


def decide(obs, guess):
    s = obs.select
    best = _best_attack(obs)
    if best is not None:
        i, dmg, lethal = best
        # 攻撃選択そのもの or リーサル → 最良の攻撃
        if (s.type == SelectType.ATTACK or lethal) and s.minCount <= 1 <= s.maxCount:
            return [i]
        # ゲーティング（既定OFF・退化のため）: 相手がコントロール系と確信 → テンポ重視で攻撃
        if (
            ENABLE_TEMPO_GATING
            and guess is not None
            and guess.confident
            and guess.name in CONTROL_ARCHETYPES
            and s.type == SelectType.MAIN
            and s.minCount <= 1 <= s.maxCount
        ):
            return [i]
    return _greedy(s)


def agent(obs_dict):
    try:
        obs = to_observation_class(obs_dict)
        if obs.select is None:
            _tracker(reset=True)  # 新規ゲーム開始 → 推定状態をリセット
            return _load_deck()
        guess = _tracker().update(obs)
        move = decide(obs, guess)
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
