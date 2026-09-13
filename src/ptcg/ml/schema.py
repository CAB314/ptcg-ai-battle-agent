"""生 obs dict のアクセサと enum 定数（int 直書き）。

episode の steps[t][i].observation と本番 agent(obs_dict) の obs は同一スキーマ。
このモジュールは両方をそのまま扱う（cg 非依存・stdlib のみ）。エンジンの enum は
「競技中に要素追加があり得る」と明記されているため、**未知 int は必ず素通し**する
（IntEnum 変換をしない・one-hot 側は UNK バケットに落とす）。

vendor 対象: 提出エージェントにコピーされる（scripts/vendor.py）。
"""

from __future__ import annotations

SCHEMA_VERSION = 1

# ---- SelectType（cg.api.SelectType と同値の int）----
SEL_MAIN = 0
SEL_CARD = 1
SEL_ATTACHED_CARD = 2
SEL_CARD_OR_ATTACHED = 3
SEL_ENERGY = 4
SEL_SKILL = 5
SEL_ATTACK = 6
SEL_EVOLVE = 7
SEL_COUNT = 8
SEL_YES_NO = 9
SEL_SPECIAL_CONDITION = 10
N_SELECT_TYPES = 11  # one-hot 幅（未知は UNK）

# ---- OptionType ----
OPT_NUMBER = 0
OPT_YES = 1
OPT_NO = 2
OPT_CARD = 3
OPT_TOOL_CARD = 4
OPT_ENERGY_CARD = 5
OPT_ENERGY = 6
OPT_PLAY = 7
OPT_ATTACH = 8
OPT_EVOLVE = 9
OPT_ABILITY = 10
OPT_DISCARD = 11
OPT_RETREAT = 12
OPT_ATTACK = 13
OPT_END = 14
OPT_SKILL = 15
OPT_SPECIAL_CONDITION = 16
N_OPTION_TYPES = 17

# ---- AreaType ----
AREA_DECK = 1
AREA_HAND = 2
AREA_DISCARD = 3
AREA_ACTIVE = 4
AREA_BENCH = 5
AREA_PRIZE = 6
AREA_STADIUM = 7
AREA_ENERGY = 8
AREA_TOOL = 9
AREA_PRE_EVOLUTION = 10
AREA_PLAYER = 11
AREA_LOOKING = 12
N_AREAS = 13  # 0=なし を含む one-hot 幅

# ---- SelectContext（ルーティングに使う値のみ名前付き）----
CTX_SETUP_ACTIVE = 1
CTX_SETUP_BENCH = 2
CTX_SWITCH = 3
CTX_TO_ACTIVE = 4
CTX_TO_DECK_BOTTOM = 10
CTX_SKILL_ORDER = 34
CTX_EVOLVE = 37
CTX_MULLIGAN = 42
# モデル対象外（順序依存・希少）→ ルール層へルーティングする context
RULES_ONLY_CONTEXTS = {CTX_TO_DECK_BOTTOM, CTX_SKILL_ORDER, CTX_EVOLVE}

# select.context の one-hot バケット（~49種+今後の追加を 21 バケットに集約）
_CTX_BUCKET = {
    0: 0,                      # MAIN
    1: 1, 2: 2,                # SETUP_ACTIVE / SETUP_BENCH
    3: 3, 4: 3,                # SWITCH / TO_ACTIVE（前出し）
    5: 4, 6: 4,                # TO_BENCH / TO_FIELD
    7: 5,                      # TO_HAND
    8: 6, 29: 6,               # DISCARD 系
    9: 7, 10: 7, 32: 7,        # TO_DECK 系
    11: 8,                     # TO_PRIZE
    12: 9, 24: 9,              # NOT_MOVE / LOOK
    13: 10, 14: 10, 15: 10,    # ダメージ系
    16: 11, 17: 11, 40: 11,    # 回復系
    18: 12, 19: 12, 20: 12, 37: 12, 45: 12,  # 進化/退化
    21: 13, 22: 13, 23: 13, 28: 13, 33: 13,  # エネ付替
    25: 14,                    # EFFECT_TARGET
    26: 15, 27: 15, 30: 15, 31: 15,          # 付属カード操作
    34: 16,                    # SKILL_ORDER
    35: 17, 36: 17,            # ATTACK / DISABLE_ATTACK
    38: 18, 39: 18,            # COUNT 系
    41: 19, 42: 19, 43: 19, 44: 19, 46: 19,  # YES_NO 系
    47: 20, 48: 20,            # SPECIAL_CONDITION
}
N_CTX_BUCKETS = 22  # 21 + UNK


def ctx_bucket(context) -> int:
    try:
        return _CTX_BUCKET.get(int(context), N_CTX_BUCKETS - 1)
    except Exception:
        return N_CTX_BUCKETS - 1


# ---- EnergyType ----
N_ENERGY_TYPES = 12  # 0..11（COLORLESS..TEAM_ROCKET）


# ---- 生 dict アクセサ（欠損・None・型ゆらぎに耐える）----

def g(d, key, default=None):
    if isinstance(d, dict):
        return d.get(key, default)
    return getattr(d, key, default)  # 念のため dataclass にも耐える


def as_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default


def current_of(obs):
    return g(obs, "current") or {}


def select_of(obs):
    return g(obs, "select")


def players(cur):
    return g(cur, "players") or [{}, {}]


def your_index(cur):
    return as_int(g(cur, "yourIndex", 0))


def me_opp(cur):
    ps = players(cur)
    yi = your_index(cur)
    if yi < 0 or yi > 1 or len(ps) < 2:
        return (ps[0] if ps else {}, ps[1] if len(ps) > 1 else {})
    return ps[yi], ps[1 - yi]


def active_mon(p):
    arr = g(p, "active") or []
    return arr[0] if arr and arr[0] is not None else None


def bench(p):
    return [m for m in (g(p, "bench") or []) if m is not None]


def hand_cards(p):
    return [c for c in (g(p, "hand") or []) if c is not None]


def discard_cards(p):
    return [c for c in (g(p, "discard") or []) if c is not None]


def prize_count(p):
    return len(g(p, "prize") or [])


def stadium_card(cur):
    st = g(cur, "stadium") or []
    return st[0] if st and st[0] is not None else None


def looking_cards(cur):
    return [c for c in (g(cur, "looking") or []) if c is not None]


def mon_at(cur, player_index, area, index):
    """area/index/playerIndex から場のポケモン dict を解決（無ければ None）。"""
    ps = players(cur)
    if player_index is None or not (0 <= as_int(player_index, -1) < len(ps)):
        return None
    p = ps[as_int(player_index)]
    a = as_int(area, -1)
    if a == AREA_ACTIVE:
        return active_mon(p)
    if a == AREA_BENCH:
        b = g(p, "bench") or []
        i = as_int(index, -1)
        return b[i] if 0 <= i < len(b) and b[i] is not None else None
    return None


def option_card_id(obs, o):
    """CARD 系 option が指すカード ID（伏せ札・不明は 0）。"""
    cur = current_of(obs)
    sel = select_of(obs) or {}
    cid = g(o, "cardId")
    if cid:
        return as_int(cid)
    area = as_int(g(o, "area"), -1)
    idx = g(o, "index")
    pi = g(o, "playerIndex")
    pi = your_index(cur) if pi is None else as_int(pi)
    try:
        if area == AREA_DECK and g(sel, "deck") is not None:
            c = (g(sel, "deck") or [])[as_int(idx)]
            return as_int(g(c, "id"), 0) if c is not None else 0
        ps = players(cur)
        p = ps[pi] if 0 <= pi < len(ps) else {}
        if area == AREA_HAND:
            c = (g(p, "hand") or [])[as_int(idx)]
            return as_int(g(c, "id"), 0) if c is not None else 0
        if area in (AREA_ACTIVE, AREA_BENCH):
            m = mon_at(cur, pi, area, idx)
            return as_int(g(m, "id"), 0) if m is not None else 0
        if area == AREA_DISCARD:
            c = (g(p, "discard") or [])[as_int(idx)]
            return as_int(g(c, "id"), 0) if c is not None else 0
        if area == AREA_LOOKING:
            c = (g(cur, "looking") or [])[as_int(idx)]
            return as_int(g(c, "id"), 0) if c is not None else 0
        if area == AREA_STADIUM:
            c = stadium_card(cur)
            return as_int(g(c, "id"), 0) if c is not None else 0
    except Exception:
        return 0
    return 0
