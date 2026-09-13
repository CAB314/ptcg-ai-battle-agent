"""ML ハイブリッドエージェント雛形（BCポリシー主軸 + 証明可能ガード + ルール縮退）。

責任分界（docs/plans 承認済み設計）:
  import時: numpy+資産（cards/attacks/weights/config）のロードを試行。失敗は握りつぶして
            MODEL_OK=False（以後ルール専行）。**numpy 不在でも 0点にならない**。
  0. アンチループガード: 同一ターン内の選択回数が閾値超過 → END 強制（ミラー自己対戦の
     バリデーションで「状態が進まない行動サイクル」に嵌まると無限ゲーム=ERRORED になる。
     2026-07-17 の alakazam_bc ERROR（18,167ステップ・crashなし）の再発防止）
  1. 確定ガード: 正確打点リーサルの即取り（guards.lethal_attack）
  2. 時間予算ガード: remainingOverageTime<60s または forward>200ms 連続で以後ルール専行
  3. ルーティング: 順序依存 context / featurize 失敗 → rules_decide
  4. モデル層: featurize(生obs) → NpPolicy.choose
  5. rules_decide: リーサル+greedy（numpy不在時の主戦力）
最外殻: is_valid 検証 → NG なら rules → random（fail-closed）。

資産解決: env PTCG_AGENT_DIR → ./ → /kaggle_simulations/agent/
"""

import os
import time

from cg.api import OptionType, SelectType, to_observation_class

import guards

MODEL_OK = False
_POLICY = None
_TABLES = None
_DECK = None
_SLOW_STREAK = 0
_FORWARD_DISABLED = False
_STATS = {"model": 0, "rules": 0, "antiloop": 0}

# アンチループ: (turn, そのターン内に自分が受けた選択回数)。ゲーム跨ぎは turn 巻き戻りで自然リセット
_TURN_GUARD = {"turn": -1, "count": 0, "game_count": 0}
TURN_SELECT_CAP = 40      # 正常プレイの上限を大きく超える値（教師データの1試合片側最大~75選択）
GAME_SELECT_CAP = 800     # ゲーム全体の保険（通常~120選択/側。ミラーでは両席合算で数える）
# 注: エンジンには FixedList capacity:7 超過で C++ 例外死する内部バグがあり（実測）、
# 無限ループ級の長試合で蓄積して発火する。ループ遮断はエラー0点の直接の再発防止策。


def _resolve(name):
    for base in (os.environ.get("PTCG_AGENT_DIR"), ".", "/kaggle_simulations/agent"):
        if not base:
            continue
        p = os.path.join(base, name)
        if os.path.exists(p):
            return p
    return None


def _load_deck():
    global _DECK
    if _DECK is None:
        path = _resolve("deck.csv")
        with open(path, "r", encoding="utf-8") as f:
            _DECK = [int(line) for line in f.read().splitlines() if line.strip()]
    return _DECK


try:  # モデル資産のロード（失敗しても素通し）
    import numpy  # noqa: F401

    from features import FEATURE_VERSION, featurize
    from np_forward import NpPolicy
    from vocab import load_tables

    _TABLES = load_tables(_resolve("cards.npz"), _resolve("attacks.npz"))
    _POLICY = NpPolicy.load(_resolve("weights.npz"), _resolve("config.json"),
                            feature_version=FEATURE_VERSION)
    MODEL_OK = _TABLES is not None and _POLICY is not None
except Exception:
    MODEL_OK = False


def _warmup():
    """初手（デッキ選択）の時間にダミー forward を隠す。"""
    try:
        guards.attacks()
        guards.card(0)
        if MODEL_OK:
            deck = _load_deck()
            obs = {
                "current": {"turn": 1, "yourIndex": 0,
                            "players": [{"deckCount": 47, "hand": [{"id": deck[0]}]}, {"deckCount": 47}]},
                "select": {"type": 0, "context": 0, "minCount": 1, "maxCount": 1,
                           "option": [{"type": 14}]},
            }
            feats = featurize(obs, _TABLES)
            if feats is not None:
                _POLICY.score(feats)
    except Exception:
        pass


def _time_guard(obs_dict):
    global _FORWARD_DISABLED
    if _FORWARD_DISABLED:
        return False
    rot = obs_dict.get("remainingOverageTime") if isinstance(obs_dict, dict) else None
    if rot is not None and rot < 60:
        _FORWARD_DISABLED = True
        return False
    return True


def _model_move(obs_dict, obs):
    """モデル層。None を返したら呼び出し側がルールへ縮退。"""
    global _SLOW_STREAK, _FORWARD_DISABLED
    if not MODEL_OK or not _time_guard(obs_dict):
        return None
    feats = featurize(obs_dict, _TABLES)
    if feats is None:
        return None
    t0 = time.monotonic()
    move = _POLICY.choose(feats)
    dt = time.monotonic() - t0
    if dt > 0.2:
        _SLOW_STREAK += 1
        if _SLOW_STREAK >= 3:
            _FORWARD_DISABLED = True
    else:
        _SLOW_STREAK = 0
    return move


def _anti_loop_visits(obs):
    """同一ターン内で自分が受けた選択回数を数える（turn 巻き戻り=新ゲームでリセット）。"""
    t = obs.current.turn or 0
    if t < _TURN_GUARD["turn"]:
        _TURN_GUARD.update({"turn": t, "count": 0, "game_count": 0})
    if t != _TURN_GUARD["turn"]:
        _TURN_GUARD["turn"] = t
        _TURN_GUARD["count"] = 0
    _TURN_GUARD["count"] += 1
    _TURN_GUARD["game_count"] += 1
    return _TURN_GUARD["count"], _TURN_GUARD["game_count"]


def _force_end(s):
    for i, o in enumerate(s.option):
        if o.type == OptionType.END:
            return [i]
    return None


def decide(obs_dict, obs):
    s = obs.select
    # 0. アンチループ: ターン内選択が異常に多い → END 強制（無限ゲーム=ERRORED の再発防止）
    visits, game_visits = _anti_loop_visits(obs)
    if visits > TURN_SELECT_CAP or game_visits > GAME_SELECT_CAP:
        _STATS["antiloop"] += 1
        end = _force_end(s)
        if end is not None:
            return end
        if visits > TURN_SELECT_CAP + 20:
            return guards.random_move(s)  # END の無い選択が延々続く場合はランダムで攪乱
        return guards.rules_decide(obs)
    # 1. 確定ガード: 正確リーサル即取り
    if s.type in (SelectType.MAIN, SelectType.ATTACK) and s.minCount <= 1 <= s.maxCount:
        i = guards.lethal_attack(obs)
        if i is not None:
            return [i]
    # 2-4. モデル層（時間ガード・ルーティング込み）
    move = _model_move(obs_dict, obs)
    if move is not None and guards.is_valid(move, s):
        _STATS["model"] += 1
        return move
    # 5. ルール縮退
    _STATS["rules"] += 1
    return guards.rules_decide(obs)


def agent(obs_dict):
    try:
        obs = to_observation_class(obs_dict)
        if obs.select is None:
            _warmup()
            return _load_deck()
        move = decide(obs_dict, obs)
        if guards.is_valid(move, obs.select):
            return move
        return guards.random_move(obs.select)
    except Exception:
        try:
            obs = to_observation_class(obs_dict)
            if obs.select is None:
                return _load_deck()
            return guards.random_move(obs.select)
        except Exception:
            return []
