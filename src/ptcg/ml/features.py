"""featurize(obs_dict, tables) — 生 obs dict → モデル入力テンソル群（numpy 純・vendor対象）。

episode の観測と本番 agent の観測は同一スキーマなので、この1ファイルを学習・PPO actor・
提出ランタイムで共用する（train/serve skew の排除）。失敗時は None を返し、呼び出し側が
ルール層へ縮退する。例外は投げない。

出力（dict of np.ndarray）:
  g   : (G,)        float32  グローバル特徴
  sid : (S,)        int32    state トークンのカードID（0=pad/UNK）
  sf  : (S, F_S)    float32  state トークン特徴
  oid : (K, 2)      int32    option の [cardId, attackId]
  of  : (K, F_O)    float32  option 特徴（導出打点含む）
  k_min, k_max     : int     選択数の下限/上限（推論の k 選択規則用）
  noop_allowed     : bool    minCount==0（no-op が合法）
"""

from __future__ import annotations

import numpy as np

try:  # パッケージ文脈（src/ptcg/ml/…）
    from .schema import (
        N_AREAS,
        N_ENERGY_TYPES,
        N_OPTION_TYPES,
        N_SELECT_TYPES,
        OPT_ATTACK,
        OPT_ATTACH,
        OPT_END,
        OPT_EVOLVE,
        OPT_PLAY,
        RULES_ONLY_CONTEXTS,
        active_mon,
        as_int,
        bench,
        ctx_bucket,
        current_of,
        discard_cards,
        g,
        hand_cards,
        looking_cards,
        me_opp,
        mon_at,
        option_card_id,
        prize_count,
        select_of,
        stadium_card,
        your_index,
    )
    from .vocab import (
        ATTACK_ROWS,
        C_CARD_TYPE,
        C_ENERGY_TYPE,
        C_EX,
        C_EX_DMG_IMMUNE,
        C_MEGA_EX,
        C_RESISTANCE,
        C_WEAKNESS,
        CARD_ROWS,
    )
except ImportError:  # vendor 文脈（提出エージェントのトップレベル）
    from schema import (  # type: ignore
        N_AREAS,
        N_ENERGY_TYPES,
        N_OPTION_TYPES,
        N_SELECT_TYPES,
        OPT_ATTACK,
        OPT_ATTACH,
        OPT_END,
        OPT_EVOLVE,
        OPT_PLAY,
        RULES_ONLY_CONTEXTS,
        active_mon,
        as_int,
        bench,
        ctx_bucket,
        current_of,
        discard_cards,
        g,
        hand_cards,
        looking_cards,
        me_opp,
        mon_at,
        option_card_id,
        prize_count,
        select_of,
        stadium_card,
        your_index,
    )
    from vocab import (  # type: ignore
        ATTACK_ROWS,
        C_CARD_TYPE,
        C_ENERGY_TYPE,
        C_EX,
        C_EX_DMG_IMMUNE,
        C_MEGA_EX,
        C_RESISTANCE,
        C_WEAKNESS,
        CARD_ROWS,
    )

FEATURE_VERSION = 1

# ---- 次元定義 ----
G_DIM = 72
S_TOKENS = 56
F_S = 40
F_O = 64

# state トークン配置
TOK_GLOBAL = 0
TOK_MY_ACTIVE = 1
TOK_MY_BENCH0 = 2      # 2..6
TOK_OPP_ACTIVE = 7
TOK_OPP_BENCH0 = 8     # 8..12
TOK_HAND0 = 13         # 13..36 (24枚)
N_HAND_TOKENS = 24
TOK_MY_DISCARD = 37
TOK_OPP_DISCARD = 38
TOK_STADIUM = 39
TOK_CONTEXT_CARD = 40
TOK_EFFECT_CARD = 41
TOK_LOOKING0 = 42      # 42..49 (8枚)
N_LOOKING_TOKENS = 8
# 50..55 予備

# state 特徴列（F_S=40）
SF_ZONE0 = 0           # 12: zone one-hot
_N_ZONES = 12
SF_HP_FRAC = 12
SF_MAXHP = 13
SF_DMG_FRAC = 14
SF_ENERGY0 = 15        # 12: タイプ別エネ個数/4
SF_N_TOOLS = 27
SF_APPEAR = 28
SF_N_PREEVO = 29
SF_COUNT = 30          # 集約トークンの枚数/10
SF_BENCH_IDX = 31
SF_PAD = 32
# 33..39 予備

ZONE_GLOBAL, ZONE_MY_ACTIVE, ZONE_MY_BENCH, ZONE_OPP_ACTIVE, ZONE_OPP_BENCH, ZONE_HAND, \
    ZONE_MY_DISCARD, ZONE_OPP_DISCARD, ZONE_STADIUM, ZONE_CTX, ZONE_EFFECT, ZONE_LOOKING = range(12)


def _clip01(x):
    return 0.0 if x < 0 else (1.0 if x > 1.0 else x)


def _card_row(tables, cid):
    cid = as_int(cid, 0)
    if 0 < cid < CARD_ROWS:
        return tables["cards"][cid]
    return tables["cards"][0]


def _mon_token(tables, mon, zone, bench_idx=0.0):
    """場のポケモン1体 → (cid, feat[F_S])。"""
    f = np.zeros(F_S, dtype=np.float32)
    f[SF_ZONE0 + zone] = 1.0
    cid = as_int(g(mon, "id"), 0)
    hp = float(as_int(g(mon, "hp"), 0))
    max_hp = float(as_int(g(mon, "maxHp"), 0)) or 1.0
    f[SF_HP_FRAC] = _clip01(hp / max_hp)
    f[SF_MAXHP] = _clip01(max_hp / 340.0)
    f[SF_DMG_FRAC] = _clip01((max_hp - hp) / max_hp)
    for e in g(mon, "energies") or []:
        e = as_int(e, -1)
        if 0 <= e < N_ENERGY_TYPES:
            f[SF_ENERGY0 + e] = min(f[SF_ENERGY0 + e] + 0.25, 1.0)
    f[SF_N_TOOLS] = _clip01(len(g(mon, "tools") or []) / 2.0)
    f[SF_APPEAR] = 1.0 if g(mon, "appearThisTurn") else 0.0
    f[SF_N_PREEVO] = _clip01(len(g(mon, "preEvolution") or []) / 2.0)
    f[SF_BENCH_IDX] = bench_idx
    return cid, f


def _card_token(cid, zone):
    f = np.zeros(F_S, dtype=np.float32)
    f[SF_ZONE0 + zone] = 1.0
    return as_int(cid, 0), f


def _agg_token(zone, count):
    f = np.zeros(F_S, dtype=np.float32)
    f[SF_ZONE0 + zone] = 1.0
    f[SF_COUNT] = _clip01(count / 10.0)
    return 0, f


def _payable(cost_vec, energies):
    """コスト(12次元カウント) を付いているエネで払えるかの近似（レインボー10=万能）。"""
    have = [0] * N_ENERGY_TYPES
    for e in energies or []:
        e = as_int(e, -1)
        if 0 <= e < N_ENERGY_TYPES:
            have[e] += 1
    wild = have[10]
    colorless_need = cost_vec[0]
    total_typed_need = 0.0
    for t in range(1, N_ENERGY_TYPES):
        need = cost_vec[t]
        if need <= 0:
            continue
        total_typed_need += need
        avail = have[t] + (have[11] if t in (5, 7) else 0)
        if avail < need:
            deficit = need - avail
            if wild >= deficit:
                wild -= deficit
            else:
                return False
    used_typed = sum(cost_vec[1:]) - 0  # 型付きは上で消費済み扱い
    remaining = sum(have) - used_typed  # 粗い近似（重複計上は許容）
    return remaining + wild >= colorless_need + used_typed - total_typed_need


def _effective_damage(tables, base, atk_card_id, def_mon):
    """弱点×2/抵抗−30/ex被ダメ無効の実効打点（guards と同一ロジックの純関数版）。"""
    if base <= 0 or def_mon is None:
        return max(0.0, base)
    atk_row = _card_row(tables, atk_card_id)
    def_row = _card_row(tables, as_int(g(def_mon, "id"), 0))
    atk_is_ex = atk_row[C_EX] > 0 or atk_row[C_MEGA_EX] > 0
    if atk_is_ex and def_row[C_EX_DMG_IMMUNE] > 0:
        return 0.0
    dmg = float(base)
    atk_type = atk_row[C_ENERGY_TYPE]
    if atk_type >= 0:
        if def_row[C_WEAKNESS] == atk_type:
            dmg *= 2
        elif def_row[C_RESISTANCE] == atk_type:
            dmg = max(0.0, dmg - 30.0)
    return dmg


# option 特徴列（F_O=64）
OF_TYPE0 = 0                      # 18: OptionType one-hot(+UNK)
OF_AREA0 = 18                     # 13: area one-hot（0=なし含む）
OF_OWNER0 = 31                    # 3: 対象の所有（自/敵/なし）
OF_INDEX = 34
OF_NUMBER = 35
OF_COUNT = 36
OF_HAS_ENERGY_IDX = 37
OF_HAS_TOOL_IDX = 38
OF_SPECIAL0 = 39                  # 6: specialConditionType(+なし)
OF_CARDTYPE0 = 45                 # 7: 対象カードの CardType one-hot
OF_EFF_DMG = 52
OF_LETHAL = 53
OF_OVERKILL = 54
OF_DEF_HP = 55
OF_PRIZE_REWARD = 56
OF_ZERO_DMG_ATTACK = 57
OF_TGT_ENERGY_N = 58              # ATTACH 対象のエネ数/4
OF_TGT_HP_FRAC = 59
OF_IS_END = 60
# 61..63 予備


def _option_row(tables, obs, cur, o, opp_active):
    oid_card = 0
    oid_attack = 0
    f = np.zeros(F_O, dtype=np.float32)
    ot = as_int(g(o, "type"), -1)
    f[OF_TYPE0 + (ot if 0 <= ot < N_OPTION_TYPES else N_OPTION_TYPES)] = 1.0
    area = as_int(g(o, "area"), 0)
    f[OF_AREA0 + (area if 0 <= area < N_AREAS else 0)] = 1.0
    pi = g(o, "playerIndex")
    yi = your_index(cur)
    if pi is None:
        f[OF_OWNER0 + 2] = 1.0
    elif as_int(pi) == yi:
        f[OF_OWNER0 + 0] = 1.0
    else:
        f[OF_OWNER0 + 1] = 1.0
    f[OF_INDEX] = _clip01(as_int(g(o, "index"), 0) / 10.0)
    f[OF_NUMBER] = _clip01(as_int(g(o, "number"), 0) / 10.0)
    f[OF_COUNT] = _clip01(as_int(g(o, "count"), 0) / 4.0)
    f[OF_HAS_ENERGY_IDX] = 1.0 if g(o, "energyIndex") is not None else 0.0
    f[OF_HAS_TOOL_IDX] = 1.0 if g(o, "toolIndex") is not None else 0.0
    sc = g(o, "specialConditionType")
    sc = as_int(sc, -1) if sc is not None else -1
    f[OF_SPECIAL0 + (sc if 0 <= sc < 5 else 5)] = 1.0
    if ot == OPT_END:
        f[OF_IS_END] = 1.0

    if ot == OPT_ATTACK:
        aid = as_int(g(o, "attackId"), 0)
        oid_attack = aid if 0 < aid < ATTACK_ROWS else 0
        me, _ = me_opp(cur)
        my_act = active_mon(me)
        oid_card = as_int(g(my_act, "id"), 0) if my_act else 0
        base = float(tables["atk_damage"][oid_attack]) if oid_attack else 0.0
        f[OF_ZERO_DMG_ATTACK] = 1.0 if base <= 0 else 0.0
        eff = _effective_damage(tables, base, oid_card, opp_active)
        f[OF_EFF_DMG] = _clip01(eff / 300.0)
        if opp_active is not None:
            dhp = float(as_int(g(opp_active, "hp"), 0))
            f[OF_DEF_HP] = _clip01(dhp / 340.0)
            if eff > 0 and dhp > 0:
                f[OF_LETHAL] = 1.0 if eff >= dhp else 0.0
                f[OF_OVERKILL] = _clip01((eff - dhp) / 300.0) if eff >= dhp else 0.0
            drow = _card_row(tables, as_int(g(opp_active, "id"), 0))
            reward = 3.0 if drow[C_MEGA_EX] > 0 else (2.0 if drow[C_EX] > 0 else 1.0)
            f[OF_PRIZE_REWARD] = reward / 3.0
    elif ot == OPT_ATTACH:
        hand = hand_cards(me_opp(cur)[0])
        idx = as_int(g(o, "index"), -1)
        if 0 <= idx < len(hand):
            oid_card = as_int(g(hand[idx], "id"), 0)
        tgt = mon_at(cur, yi, g(o, "inPlayArea"), g(o, "inPlayIndex"))
        if tgt is not None:
            f[OF_TGT_ENERGY_N] = _clip01(len(g(tgt, "energies") or []) / 4.0)
            mh = float(as_int(g(tgt, "maxHp"), 0)) or 1.0
            f[OF_TGT_HP_FRAC] = _clip01(float(as_int(g(tgt, "hp"), 0)) / mh)
    elif ot in (OPT_PLAY, OPT_EVOLVE):
        hand = hand_cards(me_opp(cur)[0])
        idx = as_int(g(o, "index"), -1)
        if 0 <= idx < len(hand):
            oid_card = as_int(g(hand[idx], "id"), 0)
    else:
        oid_card = option_card_id(obs, o)

    row = _card_row(tables, oid_card)
    ct = int(row[C_CARD_TYPE])
    if 0 <= ct < 7:
        f[OF_CARDTYPE0 + ct] = 1.0
    if not (0 < oid_card < CARD_ROWS):
        oid_card = 0
    return oid_card, oid_attack, f


def featurize(obs, tables):
    """生 obs dict → テンソル群。失敗時 None（例外は出さない）。"""
    try:
        sel = select_of(obs)
        cur = current_of(obs)
        if sel is None or not cur:
            return None
        if as_int(g(sel, "context"), -1) in RULES_ONLY_CONTEXTS:
            return None
        options = g(sel, "option") or []
        if not options:
            return None

        me, opp = me_opp(cur)
        my_act = active_mon(me)
        opp_act = active_mon(opp)

        # ---- global ----
        gv = np.zeros(G_DIM, dtype=np.float32)
        turn = as_int(g(cur, "turn"), 0)
        gv[0] = _clip01(turn / 30.0)
        if 1 <= turn <= 4:
            gv[turn] = 1.0  # 1..4
        gv[5] = 1.0 if as_int(g(cur, "firstPlayer"), -1) == your_index(cur) else 0.0
        gv[6] = _clip01(prize_count(me) / 6.0)
        gv[7] = _clip01(prize_count(opp) / 6.0)
        gv[8] = _clip01(as_int(g(me, "deckCount"), 0) / 60.0)
        gv[9] = _clip01(as_int(g(opp, "deckCount"), 0) / 60.0)
        gv[10] = _clip01(as_int(g(me, "handCount"), 0) / 24.0)
        gv[11] = _clip01(as_int(g(opp, "handCount"), 0) / 24.0)
        gv[12] = _clip01(len(bench(me)) / 5.0)
        gv[13] = _clip01(len(bench(opp)) / 5.0)
        gv[14] = 1.0 if g(cur, "supporterPlayed") else 0.0
        gv[15] = 1.0 if g(cur, "stadiumPlayed") else 0.0
        gv[16] = 1.0 if g(cur, "energyAttached") else 0.0
        gv[17] = 1.0 if g(cur, "retreated") else 0.0
        for j, k in enumerate(("poisoned", "burned", "asleep", "paralyzed", "confused")):
            gv[18 + j] = 1.0 if g(me, k) else 0.0
            gv[23 + j] = 1.0 if g(opp, k) else 0.0
        st = as_int(g(sel, "type"), -1)
        gv[28 + (st if 0 <= st < N_SELECT_TYPES else N_SELECT_TYPES)] = 1.0  # 28..39
        gv[40 + ctx_bucket(g(sel, "context"))] = 1.0                        # 40..61
        k_min = as_int(g(sel, "minCount"), 0)
        k_max = as_int(g(sel, "maxCount"), 1)
        gv[62] = _clip01(k_min / 6.0)
        gv[63] = _clip01(k_max / 6.0)
        gv[64] = _clip01(as_int(g(sel, "remainDamageCounter"), 0) / 10.0)
        gv[65] = _clip01(as_int(g(sel, "remainEnergyCost"), 0) / 4.0)
        gv[66] = _clip01(len(options) / 30.0)

        # ---- state tokens ----
        sid = np.zeros(S_TOKENS, dtype=np.int32)
        sf = np.zeros((S_TOKENS, F_S), dtype=np.float32)
        sf[:, SF_PAD] = 1.0

        def put(tok, cid, feat):
            sid[tok] = cid if 0 < cid < CARD_ROWS else 0
            sf[tok] = feat
            sf[tok, SF_PAD] = 0.0

        put(TOK_GLOBAL, 0, _agg_token(ZONE_GLOBAL, 0)[1])
        if my_act is not None:
            put(TOK_MY_ACTIVE, *_mon_token(tables, my_act, ZONE_MY_ACTIVE))
        for j, m in enumerate(bench(me)[:5]):
            put(TOK_MY_BENCH0 + j, *_mon_token(tables, m, ZONE_MY_BENCH, bench_idx=j / 5.0))
        if opp_act is not None:
            put(TOK_OPP_ACTIVE, *_mon_token(tables, opp_act, ZONE_OPP_ACTIVE))
        for j, m in enumerate(bench(opp)[:5]):
            put(TOK_OPP_BENCH0 + j, *_mon_token(tables, m, ZONE_OPP_BENCH, bench_idx=j / 5.0))
        hand = hand_cards(me)
        for j, c in enumerate(hand[:N_HAND_TOKENS]):
            put(TOK_HAND0 + j, *_card_token(g(c, "id"), ZONE_HAND))
        put(TOK_MY_DISCARD, *_agg_token(ZONE_MY_DISCARD, len(discard_cards(me))))
        put(TOK_OPP_DISCARD, *_agg_token(ZONE_OPP_DISCARD, len(discard_cards(opp))))
        stc = stadium_card(cur)
        if stc is not None:
            put(TOK_STADIUM, *_card_token(g(stc, "id"), ZONE_STADIUM))
        cc = g(sel, "contextCard")
        if cc is not None:
            put(TOK_CONTEXT_CARD, *_card_token(g(cc, "id"), ZONE_CTX))
        ec = g(sel, "effect")
        if ec is not None:
            put(TOK_EFFECT_CARD, *_card_token(g(ec, "id"), ZONE_EFFECT))
        for j, c in enumerate(looking_cards(cur)[:N_LOOKING_TOKENS]):
            put(TOK_LOOKING0 + j, *_card_token(g(c, "id"), ZONE_LOOKING))

        # ---- options ----
        K = len(options)
        oid = np.zeros((K, 2), dtype=np.int32)
        of = np.zeros((K, F_O), dtype=np.float32)
        for k, o in enumerate(options):
            c, a, row = _option_row(tables, obs, cur, o, opp_act)
            oid[k, 0] = c
            oid[k, 1] = a
            of[k] = row

        return {
            "g": gv,
            "sid": sid,
            "sf": sf,
            "oid": oid,
            "of": of,
            "k_min": k_min,
            "k_max": k_max,
            "noop_allowed": k_min == 0,
        }
    except Exception:
        return None
