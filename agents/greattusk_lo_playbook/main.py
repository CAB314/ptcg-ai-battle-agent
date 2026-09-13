"""greattusk_lo_playbook — イダイナキバ+イワパレスLO（Issue #4 対応）。

デッキ選定: Issue #4 の3候補のうちシムのプールで成立する唯一のLO＝イダイナキバLO。
deck.csv = decks/greattusk_crustle_lo.csv（Great Tusk 4 / Dwebble-Crustle 4-4 / Terrakion 1）。

勝ち筋: Land Collapse（古代サポ=Explorer's Guidance 併用で毎ターン相手山4枚ミル）で
相手を山切れ（reason 2）に追い込む。ダメージ勝ちは副次（正確打点リーサルは常時判定）。

汎用分岐（LOの骨格・常時ON）:
  - 正確打点リーサル（弱点×2/抵抗−30/ex被ダメ無効壁/NZ）と勝利直結手の即取り
  - ミル優先の攻撃選択（リーサル > Land Collapse > 素点）
  - EGシークエンシング（攻撃前に古代サポ→ミル1→4）と相手山4以下の詰め
  - 自滅LOガード（自山残量でEG/任意サーチを止める。reason 2 は自分にも適用される）
  - エネ付け先・サポーター選択木・取捨スコア（EG/ポケギア/ボール類）
局所分岐（発火条件が証明できる場面のみ・playbook.py のフラグで個別ON/OFF）:
  - 相手アクティブex & NZ不在 → Crustle壁前出し（NZ稼働で自動的にミル復帰）
  - コレスの気丈→NZサーチ、相手exを見てからNZ設置
  - 手札×ダメ型（Alakazam等、テキスト検出）へのクセロシキ＝ダメージ減衰
  - ボス/リーシアの足止めガスト（逃げ重・エネ無しを縛る）と詰めガスト
  - 前ターン被KO時のテラキオン報復130・Dwebble縛られ時のAscension脱出
  - まひ/ねむり/スタンス不一致のポケモンいれかえ

self-contained / __file__ 非依存 / fail-closed。
"""

import os
import random

from cg.api import (
    AreaType,
    LogType,
    OptionType,
    SelectContext,
    SelectType,
    all_attack,
    all_card_data,
    to_observation_class,
)

import playbook as pb

RESISTANCE_AMOUNT = 30

_DECK_CACHE = None
_ATKS = None
_CARDS = None

_PROMOTE_CONTEXTS = {SelectContext.SWITCH, SelectContext.TO_ACTIVE}
_COUNTER_CONTEXTS = {SelectContext.DAMAGE_COUNTER, SelectContext.DAMAGE_COUNTER_ANY}
_OFFENSE_TARGET_CONTEXTS = {SelectContext.EFFECT_TARGET, SelectContext.DAMAGE} | _COUNTER_CONTEXTS
_TAKE_MAX_CONTEXTS = {SelectContext.TO_HAND, SelectContext.TO_BENCH, SelectContext.TO_FIELD}
_GIVE_UP_CONTEXTS = {SelectContext.DISCARD, SelectContext.TO_DECK, SelectContext.TO_DECK_BOTTOM}


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


def _attacks():
    global _ATKS
    if _ATKS is None:
        _ATKS = {a.attackId: a for a in all_attack()}
    return _ATKS


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


def _option_card_id(obs, o, select=None):
    """CARD系オプションが指すカードIDを解決する（伏せ札は None）。"""
    try:
        if select is not None and select.deck is not None and o.area == AreaType.DECK:
            return select.deck[o.index].id
        pi = o.playerIndex if o.playerIndex is not None else obs.current.yourIndex
        p = obs.current.players[pi]
        if o.area == AreaType.HAND:
            c = (p.hand or [])[o.index]
            return c.id if c is not None else None
        if o.area == AreaType.ACTIVE:
            arr = p.active or []
            return arr[0].id if arr and arr[0] is not None else None
        if o.area == AreaType.BENCH:
            return p.bench[o.index].id
        if o.area == AreaType.DISCARD:
            return p.discard[o.index].id
        if o.area == AreaType.LOOKING:
            c = (obs.current.looking or [])[o.index]
            return c.id if c is not None else None
        if o.area == AreaType.STADIUM:
            st = obs.current.stadium
            return st[0].id if st else None
    except Exception:
        return None
    return None


# ---- 対戦内メモリ（ゲーム跨ぎは turn の巻き戻りでリセット）----

_MEM = {
    "turn": -1,
    "my_turn_seen": -1,
    "opp_prize_ref": None,
    "revenge": False,
    "eg_turn": -1,
    "prized_initial": None,  # サイド落ちの確定マップ {cid: 枚数}（山全観測で判明）
    "prize_taken": {},       # サイドから回収済みのカード {cid: 枚数}（logs で追跡）
}


def _reset_mem():
    _MEM.update(
        {
            "turn": -1,
            "my_turn_seen": -1,
            "opp_prize_ref": None,
            "revenge": False,
            "eg_turn": -1,
            "prized_initial": None,
            "prize_taken": {},
        }
    )


def _is_my_turn(obs):
    cur = obs.current
    t = cur.turn or 0
    if t <= 0 or cur.firstPlayer is None or cur.firstPlayer < 0:
        return True
    im_first = cur.firstPlayer == cur.yourIndex
    return (t % 2 == 1) == im_first


def _update_memory(obs):
    cur = obs.current
    if cur is None:
        return
    t = cur.turn or 0
    if t < _MEM["turn"]:
        _reset_mem()
    _MEM["turn"] = t
    # サイドから回収したカードを追跡（prized_initial の消し込み用）
    for lg in obs.logs or []:
        if (
            getattr(lg, "type", None) == LogType.MOVE_CARD
            and lg.playerIndex == cur.yourIndex
            and lg.fromArea == AreaType.PRIZE
            and lg.cardId is not None
        ):
            taken = _MEM["prize_taken"]
            taken[lg.cardId] = taken.get(lg.cardId, 0) + 1
    opl = len(_opp(obs).prize or [])
    if _is_my_turn(obs) and t != _MEM["my_turn_seen"]:
        # 自ターン開始境界: 前の自ターン以降に相手がサイドを取った=自分のポケモンが倒された
        _MEM["revenge"] = _MEM["opp_prize_ref"] is not None and opl < _MEM["opp_prize_ref"]
        _MEM["opp_prize_ref"] = opl
        _MEM["my_turn_seen"] = t


def _revenge(obs):
    """前の自ターン以降に自分のポケモンがKOされたか（テラキオン報復130の条件）。"""
    if _MEM["revenge"]:
        return True
    ref = _MEM["opp_prize_ref"]
    return ref is not None and len(_opp(obs).prize or []) < ref


# ---- 盤面・山札の勘定 ----

def _hand_cards(obs):
    return [c for c in (_me(obs).hand or []) if c is not None]


def _hand_count_of(obs, cid):
    return sum(1 for c in _hand_cards(obs) if c.id == cid)


def _board_count_of(obs, cid):
    n = 0
    for m in _my_mons(obs):
        if m.id == cid:
            n += 1
        n += sum(1 for c in (m.preEvolution or []) if c is not None and c.id == cid)
    return n


def _seen_elsewhere(obs, cid):
    """手札・トラッシュ・盤面・スタジアムで見えている cid の枚数。"""
    me = _me(obs)
    seen = _hand_count_of(obs, cid)
    seen += sum(1 for c in (me.discard or []) if c is not None and c.id == cid)
    for m in _my_mons(obs):
        if m.id == cid:
            seen += 1
        for arr in (m.preEvolution, m.energyCards, m.tools):
            seen += sum(1 for c in (arr or []) if c is not None and c.id == cid)
    st = obs.current.stadium
    if st and st[0] is not None and st[0].id == cid and st[0].playerIndex == obs.current.yourIndex:
        seen += 1
    return seen


def _prized_now(cid):
    """現時点でまだサイドに眠っている cid の枚数（山全観測前は 0=従来の過大側推定）。"""
    pi = _MEM["prized_initial"]
    if not pb.ENABLE_PRIZE_MAPPING or pi is None:
        return 0
    return max(0, pi.get(cid, 0) - _MEM["prize_taken"].get(cid, 0))


def _remaining_in_deck(obs, cid):
    """自分の山に残る cid の推定枚数。

    山全観測（サーチ解決時の select.deck）後はサイド落ちを差し引いた確定値。
    観測前はサイド落ちが見えないため過大側の近似。
    """
    total = sum(1 for x in _load_deck() if x == cid)
    return max(0, total - _seen_elsewhere(obs, cid) - _prized_now(cid))


def _snapshot_deck(obs, select):
    """サーチ解決時の山全観測からサイド落ちを確定する（プライズマッピング）。

    デッキリスト60 − 見えている札 − 山の実物 ＝ サイド落ち。サイドはゲーム開始時に
    固定なので、回収分（prize_taken）を足し戻した prized_initial は以後ずっと有効。

    注意: 解決中のサーチカード自身（select.effect）は手札にもトラッシュにも居ない
    「宙ぶらりん」状態でサイド落ちに誤計上される（実測+1）。Σサイド落ち＝サイド残枚数
    の整合性チェックで補正し、合わない観測は破棄する（既存の確定値を汚さない）。
    """
    if not pb.ENABLE_PRIZE_MAPPING:
        return
    me = _me(obs)
    deck_cards = [c for c in (select.deck or []) if c is not None]
    if len(deck_cards) != me.deckCount:
        return  # 山の一部しか見えていない観測は採用しない
    in_deck = {}
    for c in deck_cards:
        in_deck[c.id] = in_deck.get(c.id, 0) + 1
    prized = {}
    for cid in set(_load_deck()):
        total = sum(1 for x in _load_deck() if x == cid)
        prized[cid] = max(0, total - _seen_elsewhere(obs, cid) - in_deck.get(cid, 0))
    expect = len(me.prize or [])
    total_p = sum(prized.values())
    if total_p == expect + 1:
        eff = getattr(select, "effect", None)
        if eff is not None and prized.get(eff.id, 0) > 0:
            prized[eff.id] -= 1
            total_p -= 1
    if total_p != expect:
        return
    taken = _MEM["prize_taken"]
    _MEM["prized_initial"] = {cid: v + taken.get(cid, 0) for cid, v in prized.items()}


def _deck_spend_ok(obs, cost):
    """任意の山消費（サーチ/EG）後も自滅LO安全圏に留まるか。"""
    return _me(obs).deckCount - cost >= pb.SEARCH_MIN_SELF_DECK


# ---- ダメージモデル（汎用: 弱点/抵抗/ex無効壁/NZ）----

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


def _nz_in_play(obs):
    st = obs.current.stadium
    return bool(st) and st[0] is not None and st[0].id == pb.NEUTRALIZATION_ZONE


def _effective_damage(obs, base, atk_mon, def_mon):
    if base <= 0 or def_mon is None:
        return max(0, base)
    def_card = _card(def_mon.id)
    atk_card = _card(atk_mon.id) if atk_mon is not None else None
    if _is_ex(atk_card):
        if _ex_damage_immune(def_card):
            return 0
        if _nz_in_play(obs) and not _is_ex(def_card):
            return 0
    dmg = base
    atk_type = getattr(atk_card, "energyType", None) if atk_card else None
    if def_card is not None and atk_type is not None:
        if getattr(def_card, "weakness", None) == atk_type:
            dmg *= 2
        elif getattr(def_card, "resistance", None) == atk_type:
            dmg = max(0, dmg - RESISTANCE_AMOUNT)
    return dmg


def _payable(cost, energies):
    """ワザコストが付いているエネで払えるかの近似（レインボー10=万能, TR11=超/悪）。"""
    have = list(energies or [])
    for c in cost or []:
        if c == 0:
            continue
        if c in have:
            have.remove(c)
        elif 10 in have:
            have.remove(10)
        elif c in (5, 7) and 11 in have:
            have.remove(11)
        else:
            return False
    need_c = sum(1 for c in (cost or []) if c == 0)
    return need_c <= len(have)


def _payable_attacks(mon):
    card = _card(mon.id) if mon is not None else None
    if card is None:
        return []
    out = []
    for aid in (getattr(card, "attacks", None) or []):
        a = _attacks().get(aid)
        if a is not None and _payable(a.energies, mon.energies):
            out.append(a)
    return out


def _attack_base(obs, attack_id):
    a = _attacks().get(attack_id)
    base = (a.damage or 0) if a is not None else 0
    if (
        pb.ENABLE_RETALIATE_REVENGE
        and attack_id == pb.ATK_RETALIATE
        and _revenge(obs)
    ):
        base += 80
    return base


def _my_budget_vs(obs, target):
    """自アクティブが今払えるワザでの target への最大実効打点。"""
    attacker = _my_active(obs)
    if attacker is None:
        return 0
    best = 0
    for a in _payable_attacks(attacker):
        best = max(best, _effective_damage(obs, _attack_base(obs, a.attackId), attacker, target))
    return best


# ---- 相手の脅威推定・スタンス ----

def _hand_scaler(card):
    """「手札の枚数×ダメージ」型のワザを持つか（Alakazam系。テキスト検出で汎用化）。"""
    if card is None:
        return False
    for aid in (getattr(card, "attacks", None) or []):
        a = _attacks().get(aid)
        if a is not None and "for each card in your hand" in (a.text or "").lower():
            return True
    return False


def _opp_threat(obs):
    """相手アクティブが次ターン自アクティブへ出せる打点の推定。"""
    atk = _opp_active(obs)
    mine = _my_active(obs)
    if atk is None or mine is None:
        return 0
    best = 0
    for a in _payable_attacks(atk):
        best = max(best, _effective_damage(obs, a.damage or 0, atk, mine))
    card = _card(atk.id)
    if _hand_scaler(card) and (atk.energies or []):
        est = _opp(obs).handCount * 20
        if not (_is_ex(card) and (_ex_damage_immune(_card(mine.id)) or (_nz_in_play(obs) and not _is_ex(_card(mine.id))))):
            best = max(best, est)
    return best


def _wall_stance(obs):
    """局所分岐: 相手アクティブが ex/megaEx かつ NZ 不在なら Crustle 壁スタンス。"""
    if not pb.ENABLE_WALL_STANCE:
        return False
    if _nz_in_play(obs):
        return False
    op_act = _opp_active(obs)
    return op_act is not None and _is_ex(_card(op_act.id))


def _board_thin(obs):
    """盤面危機: 場のポケモンが残りわずか（全滅=reason 3 の負けが見えている）。"""
    return pb.ENABLE_BOARD_SURVIVAL and len(_my_mons(obs)) <= pb.BOARD_THIN


def _energy_starved(obs):
    """ミル機関の停止状態: 手札にエネが無く、場のGT/テラキオンがエネ不足で山にエネが残る。

    デッキのエネは8枚・サーチ手段はコレスの気丈のみ。この状態はENDターンの最頻原因
    （実測: END選択の約半分が GT前・エネ0-1・手札エネ0）。
    """
    if any(c.id in (pb.ROCK_FIGHTING_ENERGY, pb.MIST_ENERGY) for c in _hand_cards(obs)):
        return False
    if _remaining_in_deck(obs, pb.ROCK_FIGHTING_ENERGY) + _remaining_in_deck(obs, pb.MIST_ENERGY) <= 0:
        return False
    return any(
        m.id in (pb.GREAT_TUSK, pb.TERRAKION) and len(m.energies or []) < 2
        for m in _my_mons(obs)
    )


# ---- 取捨スコア（EGの2枚/ポケギア/ボール類/ディスカード共通）----

def _keep_value(obs, cid):
    if cid is None:
        return 0
    me = _me(obs)
    if cid in (pb.GREAT_TUSK, pb.DWEBBLE, pb.CRUSTLE, pb.TERRAKION) and _board_thin(obs):
        return 92  # 盤面危機（場2体以下）は頭数の確保が何より優先
    if cid == pb.EXPLORERS_GUIDANCE:
        return 88 if me.deckCount >= pb.EG_MIN_SELF_DECK else 30
    if cid == pb.GREAT_TUSK:
        board = _board_count_of(obs, pb.GREAT_TUSK)
        if board == 0:
            return 86
        return 68 if board + _hand_count_of(obs, pb.GREAT_TUSK) < 2 else 40
    if cid == pb.CRUSTLE:
        need = _board_count_of(obs, pb.DWEBBLE) > _board_count_of(obs, pb.CRUSTLE) + _hand_count_of(obs, pb.CRUSTLE)
        return 76 if need else 24
    if cid == pb.NEUTRALIZATION_ZONE:
        return 84 if not _nz_in_play(obs) else 5
    if cid == pb.COLRESS_TENACITY:
        useful = pb.ENABLE_NZ_LOGIC and not _nz_in_play(obs) and _remaining_in_deck(obs, pb.NEUTRALIZATION_ZONE) > 0
        useful = useful or (pb.ENABLE_COLRESS_ENERGY and _energy_starved(obs))
        return 80 if useful else 18
    if cid == pb.BOSS:
        return 70
    if cid == pb.SWITCH:
        return 66
    if cid == pb.LISIA:
        return 62
    if cid == pb.XEROSIC:
        return 58
    if cid == pb.DWEBBLE:
        line = _board_count_of(obs, pb.DWEBBLE) + _board_count_of(obs, pb.CRUSTLE)
        return 55 if line < 2 else 26
    if cid in (pb.ROCK_FIGHTING_ENERGY, pb.MIST_ENERGY):
        if _energy_starved(obs):
            return 82  # ミル機関停止中はエネ最優先で拾う
        attached = sum(len(m.energies or []) for m in _my_mons(obs))
        in_hand = sum(1 for c in _hand_cards(obs) if c.id in (pb.ROCK_FIGHTING_ENERGY, pb.MIST_ENERGY))
        return 64 if attached + in_hand < 4 else 36
    if cid == pb.POKEGEAR:
        return 46
    if cid == pb.POFFIN:
        return 44 if _remaining_in_deck(obs, pb.DWEBBLE) > 0 else 12
    if cid == pb.POKE_PAD:
        return 42
    if cid == pb.FIGHTING_GONG:
        return 38
    if cid == pb.TERRAKION:
        return 34
    if cid == pb.ULTRA_BALL:
        return 22
    if cid == pb.JUMBO_ICE_CREAM:
        return 20
    return 15


def _pick_by_keep(obs, select, take_max):
    """keep値で選ぶ: 獲得系は高い方から maxCount、手放す系は低い方から minCount。"""
    n = len(select.option)
    scored = sorted(
        range(n),
        key=lambda i: (_keep_value(obs, _option_card_id(obs, select.option[i], select)), -i),
        reverse=True,
    )
    if take_max:
        k = min(select.maxCount, n)
        return scored[:k]
    k = _pick_k(select)
    if select.minCount == 0:
        return []
    return scored[-k:] if k > 0 else []


# ---- 前出し（スタンス別）----

def _promote_score(obs, mon):
    if mon is None:
        return -100
    wall = _wall_stance(obs)
    table = pb.PROMOTE_WALL if wall else pb.PROMOTE_MILL
    sc = table.get(mon.id, 0) + min(len(mon.energies or []), 3) * 5
    if not wall:
        if mon.id == pb.GREAT_TUSK and len(mon.energies or []) >= 2:
            sc += 30  # 即ミル再開できる個体
        if (
            pb.ENABLE_RETALIATE_REVENGE
            and mon.id == pb.TERRAKION
            and _revenge(obs)
        ):
            a = _attacks().get(pb.ATK_RETALIATE)
            op_act = _opp_active(obs)
            if a is not None and _payable(a.energies, mon.energies) and op_act is not None:
                if _effective_damage(obs, 130, mon, op_act) >= op_act.hp:
                    sc += 150  # 報復リーサル
    return sc


# ---- ガスト対象（ボス/リーシア）----

def _prize_reward(card):
    return 3 if getattr(card, "megaEx", False) else (2 if getattr(card, "ex", False) else 1)


def _gust_score(obs, mon):
    if mon is None:
        return -100
    card = _card(mon.id)
    budget = _my_budget_vs(obs, mon)
    if budget >= mon.hp and (_is_ex(card) or len(mon.energies or []) >= 2 or mon.hp >= 90):
        if pb.ENABLE_PRIZE_EFFICIENT_GUST:
            # サイドプラン（ぽけおじ）: 6枚完走の総ダメージ最小化=報酬大・必要打点小を優先
            return 1000 + 300 * _prize_reward(card) + (300 - min(mon.hp, 299))
        return 1000 + mon.hp  # 呼んで倒せる価値ある的
    sc = 0
    if not _payable_attacks(mon):
        sc += 60  # 今は殴れない=縛れば足止め
        if not (mon.energies or []):
            sc += 40
    retreat = getattr(card, "retreatCost", 0) or 0
    sc += retreat * 20
    return sc


def _best_gust_target(obs):
    op = _opp(obs)
    best = None
    for m in op.bench or []:
        if m is None:
            continue
        s = _gust_score(obs, m)
        if best is None or s > best[0]:
            best = (s, m)
    return best  # (score, mon) | None


# ---- 攻撃ポリシー（汎用: リーサル > ミル > 素点）----

def _attack_options(select):
    return [(i, o.attackId) for i, o in enumerate(select.option) if o.type == OptionType.ATTACK and o.attackId is not None]


def _mill_now(obs):
    return 4 if _MEM["eg_turn"] == (obs.current.turn or 0) else 1


def _attack_policy(obs, atk_opts, winning_only=False):
    if not atk_opts:
        return None
    me, op = _me(obs), _opp(obs)
    attacker = _my_active(obs)
    defender = _opp_active(obs)
    my_prizes_left = len(me.prize or []) or 1
    lc = next((i for i, aid in atk_opts if aid == pb.ATK_LAND_COLLAPSE), None)
    asc = next((i for i, aid in atk_opts if aid == pb.ATK_ASCENSION), None)
    best_dmg = None  # (eff, i, lethal, reward)
    for i, aid in atk_opts:
        eff = _effective_damage(obs, _attack_base(obs, aid), attacker, defender)
        if defender is None or eff <= 0:
            continue
        lethal = eff >= defender.hp
        dc = _card(defender.id)
        reward = 3 if getattr(dc, "megaEx", False) else (2 if getattr(dc, "ex", False) else 1)
        key = (1 if lethal else 0, eff, -i)
        if best_dmg is None or key > best_dmg[0]:
            best_dmg = (key, i, eff, lethal, reward)
    win_ko = best_dmg is not None and best_dmg[3] and best_dmg[4] >= my_prizes_left
    mill_win = lc is not None and op.deckCount <= _mill_now(obs)
    if win_ko:
        return [best_dmg[1]]
    if mill_win:
        return [lc]
    if winning_only:
        return None
    if best_dmg is not None and best_dmg[3]:
        return [best_dmg[1]]  # リーサルは常に取る（脅威除去+サイドレース）
    if lc is not None:
        return [lc]  # LOデッキの基本手: ミル
    if asc is not None and pb.ENABLE_ASCENSION_ESCAPE:
        return [asc]  # Dwebble縛られ→山からCrustleへ（実質的な脱出+壁形成）
    if best_dmg is not None:
        return [best_dmg[1]]
    return None


# ---- MAIN ポリシー ----

def _index_main_options(obs, select):
    hand = _me(obs).hand or []
    plays = {}
    attaches = []
    evolves = []
    for i, o in enumerate(select.option):
        if o.type == OptionType.PLAY and o.index is not None and 0 <= o.index < len(hand):
            c = hand[o.index]
            if c is not None and c.id not in plays:
                plays[c.id] = i
        elif o.type == OptionType.ATTACH:
            attaches.append((i, o))
        elif o.type == OptionType.EVOLVE:
            evolves.append(i)
    return plays, attaches, evolves


def _attach_score(obs, o):
    hand = _me(obs).hand or []
    try:
        ecid = hand[o.index].id
    except Exception:
        return -1
    mon = _mon_at(obs, obs.current.yourIndex, o.inPlayArea, o.inPlayIndex)
    if mon is None:
        return -1
    n_energy = len(mon.energies or [])
    is_active = o.inPlayArea == AreaType.ACTIVE
    if mon.id == pb.GREAT_TUSK:
        if n_energy < 2:
            sc = 100 if is_active else 70
            if ecid == pb.ROCK_FIGHTING_ENERGY:
                sc += 5  # 闘ポケモンへの効果防御が乗る
            return sc
        return 8 if is_active and n_energy < 4 else 2  # 3枚目以降はほぼ不要
    if mon.id == pb.DWEBBLE and is_active and n_energy < 1 and pb.ENABLE_ASCENSION_ESCAPE:
        return 90  # 縛られたDwebbleにAscension用の1枚
    if mon.id == pb.TERRAKION:
        if n_energy < 2:
            has_f = 6 in (mon.energies or [])
            if ecid == pb.ROCK_FIGHTING_ENERGY and not has_f:
                return 55
            return 40 if has_f else 20
        return 3
    if mon.id == pb.CRUSTLE and ecid == pb.MIST_ENERGY and n_energy < 1 and _wall_stance(obs):
        return 25  # 壁へのワザ効果防御
    return 1


def _choose_supporter(obs, plays, atk_opts):
    """サポーター選択木。返り値=打つカードID or None。"""
    cur = obs.current
    if cur.supporterPlayed:
        return None
    me, op = _me(obs), _opp(obs)
    hand_ids = {c.id for c in _hand_cards(obs)}
    lc_ready = any(aid == pb.ATK_LAND_COLLAPSE for _, aid in atk_opts)

    # 1. 詰めEG: 相手山4以下なら EG→Land Collapse で今勝つ
    if (
        pb.ENABLE_MILL_FINISHER
        and pb.EXPLORERS_GUIDANCE in hand_ids
        and pb.EXPLORERS_GUIDANCE in plays
        and lc_ready
        and 1 < op.deckCount <= 4
        and me.deckCount >= pb.EG_FINISHER_MIN_DECK
    ):
        return pb.EXPLORERS_GUIDANCE
    # 2. コレスの気丈: NZ不在 & 相手にex → NZ+エネをサーチ
    if (
        pb.ENABLE_NZ_LOGIC
        and pb.COLRESS_TENACITY in hand_ids
        and pb.COLRESS_TENACITY in plays
        and not _nz_in_play(obs)
        and _remaining_in_deck(obs, pb.NEUTRALIZATION_ZONE) > 0
        and any(_is_ex(_card(m.id)) for m in _opp_mons(obs))
        and _deck_spend_ok(obs, 2)
    ):
        return pb.COLRESS_TENACITY
    # 2b. コレスの気丈=エネサーチ: ミル機関停止（エネ枯渇）の再始動
    if (
        pb.ENABLE_COLRESS_ENERGY
        and pb.COLRESS_TENACITY in hand_ids
        and pb.COLRESS_TENACITY in plays
        and _energy_starved(obs)
        and _deck_spend_ok(obs, 2)
    ):
        return pb.COLRESS_TENACITY
    # 3. 手札×ダメ型（Alakazam等）にはクセロシキ=ダメージ減衰+リソース破壊
    if (
        pb.ENABLE_HAND_SCALER_XEROSIC
        and pb.XEROSIC in hand_ids
        and pb.XEROSIC in plays
        and op.handCount >= pb.HAND_SCALER_MIN_HAND
        and _hand_scaler(_card(_opp_active(obs).id) if _opp_active(obs) else None)
    ):
        return pb.XEROSIC
    # 4. 大量手札へのクセロシキ（EGの+3ミルより価値が高い水準）
    if pb.XEROSIC in hand_ids and pb.XEROSIC in plays and op.handCount >= pb.XEROSIC_BIG_HAND:
        return pb.XEROSIC
    # 5. ガスト
    if pb.ENABLE_GUST_LOGIC:
        best = _best_gust_target(obs)
        if best is not None:
            score, target = best
            if score >= 1000 and pb.BOSS in hand_ids and pb.BOSS in plays:
                return pb.BOSS  # 詰めガスト（呼んで倒す）
            my_act = _my_active(obs)
            threatened = my_act is not None and _opp_threat(obs) >= my_act.hp
            if threatened and score >= pb.STALL_GUST_MIN_SCORE:
                tc = _card(target.id)
                if pb.LISIA in hand_ids and pb.LISIA in plays and getattr(tc, "basic", False):
                    return pb.LISIA  # たねなら呼び+こんらんで固く縛る
                if pb.BOSS in hand_ids and pb.BOSS in plays:
                    return pb.BOSS
    # 6. 既定のEG（ミルが撃てるターンのみ・自滅LOゲート）
    if (
        pb.EXPLORERS_GUIDANCE in hand_ids
        and pb.EXPLORERS_GUIDANCE in plays
        and lc_ready
        and (not pb.ENABLE_EG_GATE or me.deckCount >= pb.EG_MIN_SELF_DECK)
    ):
        return pb.EXPLORERS_GUIDANCE
    # 7. 代替クセロシキ（EGが打てない番の手なりの妨害）
    if pb.XEROSIC in hand_ids and pb.XEROSIC in plays and op.handCount >= pb.XEROSIC_FALLBACK_HAND:
        return pb.XEROSIC
    return None


def _switch_move(obs, plays):
    """局所分岐: ポケモンいれかえによる状態回復/スタンス修正。"""
    if pb.SWITCH not in plays:
        return None
    me = _me(obs)
    act = _my_active(obs)
    if act is None or not (me.bench or []):
        return None
    wall = _wall_stance(obs)
    bench = [m for m in me.bench if m is not None]
    # a. まひ/ねむりの解除（前出し先はスタンススコアが選ぶ）
    if pb.ENABLE_CONDITION_SWITCH and (me.paralyzed or me.asleep) and bench:
        return [plays[pb.SWITCH]]
    if not pb.ENABLE_STANCE_SWITCH:
        return None
    # b. 壁スタンスなのに前が壁でない → Crustleへ（「EX中にイワパレス」）
    if wall and act.id != pb.CRUSTLE and any(m.id == pb.CRUSTLE for m in bench):
        return [plays[pb.SWITCH]]
    # c. ミルスタンスなのに前が壁のまま → エネの乗ったGTへ戻してミル再開
    if not wall and act.id == pb.CRUSTLE and any(
        m.id == pb.GREAT_TUSK and len(m.energies or []) >= 2 for m in bench
    ):
        return [plays[pb.SWITCH]]
    return None


def _item_move(obs, plays):
    """アイテムの価値プレイ（条件はすべて盤面から証明できるものに限定）。"""
    me = _me(obs)
    bench_space = me.benchMax - len(me.bench or [])
    hand = _hand_cards(obs)
    # 盤面危機: 通常のゲートを外して頭数を最優先で補充（reason 3=場切れ対策）
    if _board_thin(obs):
        if pb.POFFIN in plays and bench_space > 0 and _remaining_in_deck(obs, pb.DWEBBLE) > 0 and _deck_spend_ok(obs, 2):
            return [plays[pb.POFFIN]]
        if pb.FIGHTING_GONG in plays and (
            _remaining_in_deck(obs, pb.GREAT_TUSK) + _remaining_in_deck(obs, pb.TERRAKION) > 0
        ) and _deck_spend_ok(obs, 1):
            return [plays[pb.FIGHTING_GONG]]
        if pb.POKE_PAD in plays and any(
            _remaining_in_deck(obs, cid) > 0
            for cid in (pb.GREAT_TUSK, pb.DWEBBLE, pb.CRUSTLE, pb.TERRAKION)
        ) and _deck_spend_ok(obs, 1):
            return [plays[pb.POKE_PAD]]
        if pb.ULTRA_BALL in plays and len(hand) >= 3 and (
            _remaining_in_deck(obs, pb.GREAT_TUSK) + _remaining_in_deck(obs, pb.TERRAKION) > 0
        ) and _deck_spend_ok(obs, 1):
            return [plays[pb.ULTRA_BALL]]
    # なかよしポフィン: Dwebble供給（壁パイプライン）
    if (
        pb.POFFIN in plays
        and bench_space > 0
        and _remaining_in_deck(obs, pb.DWEBBLE) > 0
        and _board_count_of(obs, pb.DWEBBLE) + _board_count_of(obs, pb.CRUSTLE) < 3
        and _deck_spend_ok(obs, 2)
    ):
        return [plays[pb.POFFIN]]
    # ファイトングゴング: GT線が細いときの補充
    if (
        pb.FIGHTING_GONG in plays
        and _remaining_in_deck(obs, pb.GREAT_TUSK) > 0
        and _board_count_of(obs, pb.GREAT_TUSK) + _hand_count_of(obs, pb.GREAT_TUSK) < 2
        and _deck_spend_ok(obs, 1)
    ):
        return [plays[pb.FIGHTING_GONG]]
    # ポケパッド: Crustle/GTの不足分をサーチ
    if pb.POKE_PAD in plays and _deck_spend_ok(obs, 1):
        need_crustle = (
            _board_count_of(obs, pb.DWEBBLE) > _board_count_of(obs, pb.CRUSTLE) + _hand_count_of(obs, pb.CRUSTLE)
            and _remaining_in_deck(obs, pb.CRUSTLE) > 0
        )
        need_gt = (
            _board_count_of(obs, pb.GREAT_TUSK) + _hand_count_of(obs, pb.GREAT_TUSK) < 2
            and _remaining_in_deck(obs, pb.GREAT_TUSK) > 0
        )
        if need_crustle or need_gt:
            return [plays[pb.POKE_PAD]]
    # ポケギア: EGが手に無ければ掘る（シャッフル戻しで山は減らない）
    if (
        pb.POKEGEAR in plays
        and _hand_count_of(obs, pb.EXPLORERS_GUIDANCE) == 0
        and _remaining_in_deck(obs, pb.EXPLORERS_GUIDANCE) > 0
        and _deck_spend_ok(obs, 1)
    ):
        return [plays[pb.POKEGEAR]]
    # ジャンボイスクリーム: エネ3以上の前を80回復
    act = _my_active(obs)
    if (
        pb.JUMBO_ICE_CREAM in plays
        and act is not None
        and len(act.energies or []) >= 3
        and act.maxHp - act.hp >= 80
    ):
        return [plays[pb.JUMBO_ICE_CREAM]]
    # ハイパーボール: 場にミル役が居ない緊急時のみ（手札2枚コストが重い）
    if (
        pb.ULTRA_BALL in plays
        and _board_count_of(obs, pb.GREAT_TUSK) == 0
        and _remaining_in_deck(obs, pb.GREAT_TUSK) > 0
        and len(hand) >= 3
        and _deck_spend_ok(obs, 1)
    ):
        return [plays[pb.ULTRA_BALL]]
    return None


def _bench_basic_move(obs, plays):
    me = _me(obs)
    if me.benchMax - len(me.bench or []) <= 0:
        return None
    for cid in (pb.DWEBBLE, pb.GREAT_TUSK, pb.TERRAKION):
        if cid in plays:
            return [plays[cid]]
    return None


def _main_policy(obs):
    s = obs.select
    cur = obs.current
    plays, attaches, evolves = _index_main_options(obs, s)
    atk_opts = _attack_options(s)

    # 1. 進化（Dwebble→Crustle）は常に取る
    if evolves:
        return [evolves[0]]
    # 2. 今勝てる手（勝利KO/ミル切れ）は即取り
    win = _attack_policy(obs, atk_opts, winning_only=True)
    if win is not None:
        move = win
        if s.minCount <= 1 <= s.maxCount:
            return move
    # 3. いれかえ（状態回復/スタンス修正）
    sw = _switch_move(obs, plays)
    if sw is not None:
        return sw
    # 4. エネルギー手張り（サポーター判断より先: LCの支払い可否が変わる）
    if pb.ENABLE_ATTACH_POLICY and attaches and not cur.energyAttached:
        best = max(attaches, key=lambda t: (_attach_score(obs, t[1]), -t[0]))
        if _attach_score(obs, best[1]) > 0:
            return [best[0]]
    # 5. NZ設置（相手にexが見えてから）
    if (
        pb.ENABLE_NZ_LOGIC
        and pb.NEUTRALIZATION_ZONE in plays
        and not cur.stadiumPlayed
        and not _nz_in_play(obs)
        and any(_is_ex(_card(m.id)) for m in _opp_mons(obs))
    ):
        return [plays[pb.NEUTRALIZATION_ZONE]]
    # 6. たねをベンチへ
    b = _bench_basic_move(obs, plays)
    if b is not None:
        return b
    # 7. アイテム（ポケギアのEG掘りはサポーター判断より先）
    it = _item_move(obs, plays)
    if it is not None:
        return it
    # 8. サポーター（詰めEGを含む選択木）
    if pb.ENABLE_SUPPORTER_TREE:
        sup = _choose_supporter(obs, plays, atk_opts)
        if sup is not None:
            if sup == pb.EXPLORERS_GUIDANCE:
                _MEM["eg_turn"] = cur.turn or 0
            return [plays[sup]]
    # 9. 攻撃（リーサル > Land Collapse > 素点）
    atk = _attack_policy(obs, atk_opts)
    if atk is not None:
        return atk
    # 10. 番を終える（攻撃できないターンのエンジン既定手＝[0]は攻撃を含まないので安全）
    for i, o in enumerate(s.option):
        if o.type == OptionType.END:
            return [i]
    return _greedy(s)


# ---- CARD 選択のディスパッチ ----

def _card_policy(obs):
    s = obs.select
    ctx = s.context
    n = len(s.option)
    if n == 0:
        return []
    # セットアップ
    if pb.ENABLE_SETUP_POLICY and ctx == SelectContext.SETUP_ACTIVE_POKEMON:
        order = sorted(
            range(n),
            key=lambda i: (pb.SETUP_ACTIVE_PRIORITY.get(_option_card_id(obs, s.option[i], s) or -1, 0), -i),
            reverse=True,
        )
        return order[: _pick_k(s)]
    if pb.ENABLE_SETUP_POLICY and ctx == SelectContext.SETUP_BENCH_POKEMON:
        order = sorted(
            range(n),
            key=lambda i: (pb.SETUP_BENCH_PRIORITY.get(_option_card_id(obs, s.option[i], s) or -1, 0), -i),
            reverse=True,
        )
        return order[: min(s.maxCount, n)]  # たねは全部並べる（LOは頭数=時間）
    # 前出し（スタンス別）
    if ctx in _PROMOTE_CONTEXTS:
        order = sorted(
            range(n),
            key=lambda i: (
                _promote_score(obs, _mon_at(obs, obs.current.yourIndex, s.option[i].area, s.option[i].index)),
                -i,
            ),
            reverse=True,
        )
        return order[: _pick_k(s)]
    # ポフィン等の「山からベンチへ」: Dwebbleのみ
    if ctx in (SelectContext.TO_BENCH, SelectContext.TO_FIELD) and s.deck is not None:
        picks = [i for i in range(n) if _option_card_id(obs, s.option[i], s) == pb.DWEBBLE]
        picks = picks[: s.maxCount]
        if len(picks) >= s.minCount:
            return picks
        return _greedy(s)
    # 獲得系（EGの2枚/ポケギア/サーチ先）: keep値の高い順に maxCount
    if pb.ENABLE_KEEP_POLICY and ctx in _TAKE_MAX_CONTEXTS:
        return _pick_by_keep(obs, s, take_max=True)
    # 手放す系（ハイパーボール/相手クセロシキ）: keep値の低い順に minCount
    if pb.ENABLE_KEEP_POLICY and ctx in _GIVE_UP_CONTEXTS:
        return _pick_by_keep(obs, s, take_max=False)
    # ガスト対象（自分のボス/リーシア発動時）
    eff = getattr(s, "effect", None)
    if (
        pb.ENABLE_GUST_LOGIC
        and ctx == SelectContext.EFFECT_TARGET
        and eff is not None
        and eff.id in (pb.BOSS, pb.LISIA)
    ):
        order = sorted(
            range(n),
            key=lambda i: (
                _gust_score(obs, _mon_at(obs, 1 - obs.current.yourIndex, s.option[i].area, s.option[i].index)),
                -i,
            ),
            reverse=True,
        )
        return order[: _pick_k(s)]
    # 敵側ターゲティングの汎用フォールバック（KOできる > 低HP）
    if ctx in _OFFENSE_TARGET_CONTEXTS:
        opp_index = 1 - obs.current.yourIndex

        def _enemy(i):
            o = s.option[i]
            if o.playerIndex is not None and o.playerIndex != opp_index:
                return -1000
            mon = _mon_at(obs, opp_index, o.area, o.index)
            if mon is None:
                return -100
            budget = _my_budget_vs(obs, mon)
            if ctx in _COUNTER_CONTEXTS:
                budget = max(0, (s.remainDamageCounter or 0)) * 10
            ko = budget > 0 and budget >= mon.hp
            bonus = 0
            if ko and pb.ENABLE_PRIZE_EFFICIENT_GUST:
                bonus = 200 * (_prize_reward(_card(mon.id)) - 1)  # 同KOならプライズ報酬大を優先
            return (1000 if ko else 0) + bonus + (500 - min(mon.hp, 499))

        order = sorted(range(n), key=lambda i: (_enemy(i), -i), reverse=True)
        return order[: _pick_k(s)]
    return _greedy(s)


def decide(obs):
    s = obs.select
    _update_memory(obs)
    if s.deck is not None:
        _snapshot_deck(obs, s)  # サーチで山が見えた瞬間にサイド落ちを確定
    if s.type == SelectType.MAIN:
        return _main_policy(obs)
    if s.type == SelectType.ATTACK:
        move = _attack_policy(obs, _attack_options(s))
        if move is not None and s.minCount <= 1 <= s.maxCount:
            return move
        return _greedy(s)
    if s.type == SelectType.CARD:
        return _card_policy(obs)
    return _greedy(s)


def agent(obs_dict):
    try:
        obs = to_observation_class(obs_dict)
        if obs.select is None:
            try:
                _attacks()
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
