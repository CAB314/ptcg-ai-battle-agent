"""カード/ワザ静的テーブルの生成（cg 必要）と読み込み（numpy のみ）。

学習・PPO actor・提出ランタイムの全経路で **同一の cards.npz / attacks.npz** を使い、
特徴量パスから cg 依存を排除する（設計原則2）。ID はヘッドルーム付き固定長配列の
添字（範囲外・未知IDは行0=UNKに落ちる）。

生成: poetry run python scripts/build_vocab.py

--- v2（2026-08-02）---
v1 ではカードの質的情報（弱点・タイプ・進化段階・効果テキスト）が
**学習ID埋め込みにしか入っていなかった**。実測の結果その埋め込みは初期値と区別が
つかず（未使用ID行を対照群にした4検定すべてで有意差なし）、モデルはカードの意味を
持っていなかった。v2 では:

  1. **静的属性ブロック** card_static / atk_static を one-hot 展開済みで持つ。
     モデルは sid/oid から gather するだけでよく、シャードの再生成が要らない。
  2. **効果テキストのキーワード特徴**（カード20 / ワザ16）を追加。正規表現で
     ドロー・サーチ・呼び出し・エネ加速・ダメカン操作・無効化・状態異常・山札破壊等を
     符号化する。ID 埋め込みと違い **未知カードにもそのまま効く**のが要点。
  3. 行数を実在ID数に寄せる（4096→1536 / 2048→1600）。v1 では card_emb の
     262,144 パラメータ（全体の29%）のうち 96% が一度も参照されていなかった。
"""

from __future__ import annotations

import re

import numpy as np

VOCAB_VERSION = 3
CARD_ROWS = 1536   # 現行 cardId 最大 1267 に対しヘッドルーム約20%
ATTACK_ROWS = 1600  # 現行 attackId 最大 1556

N_ENERGY_TYPES_TBL = 12

# cards.npz の生カラム（float32 行列 cards[CARD_ROWS, C_*]）— v1 と同一（features.py が参照）
C_HP = 0
C_RETREAT = 1
C_ENERGY_TYPE = 2   # -1=なし, 0..11
C_WEAKNESS = 3      # -1=なし
C_RESISTANCE = 4    # -1=なし
C_CARD_TYPE = 5     # cg.CardType 0..6, -1=不明
C_BASIC = 6
C_STAGE1 = 7
C_STAGE2 = 8
C_EX = 9
C_MEGA_EX = 10
C_TERA = 11
C_ACE_SPEC = 12
C_N_ATTACKS = 13
C_EX_DMG_IMMUNE = 14   # 「相手exのワザダメージを全て防ぐ」系スキル持ち
C_KNOWN = 15           # 1=実在ID
N_CARD_COLS = 16

# ---- 効果テキストのキーワード特徴 ----
# 設計方針: ゲーム機構の単位で切る（カード名ではなく「何をするカードか」）。
# 未知カードでもテキストさえあれば同じベクトルになるので、新セット・新型に汎化する。
CARD_EFFECT_PATTERNS: list[tuple[str, str]] = [
    ("search_deck",    r"search your deck"),
    # 三人称形も拾う（v2 初版は "draws 4 cards" を取りこぼし、Judge がフラグ全ゼロだった）
    ("draw",           r"\bdraws? \d+ card|\bdraws? a card|draws? cards"),
    ("hand_refresh",   r"shuffles? (your|their) hand"),
    ("look_top",       r"look at the top"),
    ("attach_energy",  r"attach .{0,40}energy"),
    ("from_discard",   r"from your discard pile"),
    ("gust",           r"switch in 1 of your opponent|switch out your opponent"),
    ("self_switch",    r"switch this pok|switch your active"),
    ("heal",           r"\bheal\b"),
    ("dmg_counter",    r"damage counter"),
    ("prevent_damage", r"prevent all damage|no damage"),
    ("opp_hand",       r"opponent'?s hand|opponent shuffles"),
    ("mill",           r"discard the top .{0,40}opponent'?s deck"),
    ("ability_lock",   r"have no abilit|abilities .{0,20}have no effect"),
    ("coin_flip",      r"flip a coin"),
    ("condition",      r"asleep|poisoned|burned|paralyzed|confused"),
    ("prize",          r"prize card"),
    ("retreat_mod",    r"retreat cost"),
    ("once_per_turn",  r"once during your turn"),
    ("on_evolve",      r"when you play this pok.{0,30}to evolve"),
]
ATTACK_EFFECT_PATTERNS: list[tuple[str, str]] = [
    # 「ベンチに打点が飛ぶ」に限定する。v2 初版の r"benched pok" は
    # ドロー枚数の参照（Nab 'n' Dash）やエネ加速（Leaflet Blessings）にも発火していた。
    ("bench_damage",   r"damage to .{0,40}benched pok|damage to each .{0,30}benched"),
    ("scaling",        r"more damage for each|damage for each"),
    ("discard_energy", r"discard .{0,40}energy"),
    ("self_damage",    r"does \d+ damage to itself|damage to this pok"),
    ("heal",           r"\bheal\b"),
    ("condition",      r"asleep|poisoned|burned|paralyzed|confused"),
    ("coin_flip",      r"flip a coin"),
    ("switch_opp",     r"switch out your opponent|switch in 1 of your opponent"),
    ("switch_self",    r"switch this pok"),
    ("draw_search",    r"\bdraws? |search your deck"),
    ("energy_accel",   r"attach .{0,40}energy"),
    # 「自分が守られる」効果に限定。v2 初版の r"during your opponent's next turn" は
    # 相手デバフ（次のターン逃げられない等）と自傷ペナルティも同じ1ビットに潰していた。
    ("protect",        r"prevent all .{0,30}damage|takes \d+ less damage|"
                       r"can'?t be knocked out|no damage .{0,30}to this pok"),
    # 相手の次ターンを縛る妨害（protect とは方向が逆なので分ける）
    ("opp_debuff",     r"during your opponent'?s next turn, (they|the defending|your opponent)"),
    ("opp_hand",       r"opponent'?s hand"),
    # 相手の山札を削るミルに限定（自分の山を削るコストと混ざっていた）
    ("mill",           r"discard the top .{0,40}opponent'?s deck"),
    ("dmg_counter",    r"damage counter"),
]
N_CARD_EFFECTS = len(CARD_EFFECT_PATTERNS)
N_ATTACK_EFFECTS = len(ATTACK_EFFECT_PATTERNS)

# ---- 静的属性ブロック（one-hot 展開済み。モデルが sid/oid から gather する）----
# card_static[CARD_ROWS, D_CARD_STATIC]
D_CARD_STATIC = (
    8          # cardType one-hot（0..6 + 不明）。v2初版は width=7 で CardType6 が不明と衝突していた
    + 13       # energyType one-hot（なし + 12）
    + 13       # weakness one-hot
    + 13       # resistance one-hot
    + 3        # basic / stage1 / stage2
    + 4        # ex / megaEx / tera / aceSpec
    + 1        # ex_dmg_immune
    + 1        # retreat/4
    + 1        # maxHP/340
    + 1        # ワザ数/4
    + 1        # 最大打点/300
    + 1        # known
    + N_CARD_EFFECTS
)
# atk_static[ATTACK_ROWS, D_ATK_STATIC]
D_ATK_STATIC = (
    1          # damage/300
    + 1        # 総コスト/5
    + N_ENERGY_TYPES_TBL   # タイプ別コスト/4
    + 1        # known
    + N_ATTACK_EFFECTS
)


def _norm_text(t) -> str:
    return (t or "").lower().replace("’", "'")


# ace(カードID) → 表示名。**エンジンの実データで確認済み（2026-08-04 修正）**。
# 注意: ace は「2枚以上採用のポケモンで最高HP」というヒューリスティックなので、
# 実際のアタッカーではなく高HPのサポートポケモンを拾うことがある。
#   ・66 = Dudunsparce(HP140) は Alakazam(743, HP140) と同HPで、
#     Alakazam デッキの ace として 66 が選ばれる（同HPのタイブレークの副作用）。
#     **66 と 743 は同じ Alakazam デッキ**なので集計時は合算すべき。
ACE_DISPLAY_NAMES = {
    648: "Grimmsnarl ex", 756: "M Kangaskhan ex", 849: "M Lopunny ex",
    431: "TR Mewtwo ex", 121: "Dragapult ex", 381: "Garchomp ex",
    96: "Teal Ogerpon ex", 117: "Cornerstone Ogerpon ex", 1031: "M Starmie ex",
    678: "M Lucario ex", 90: "Thwackey", 607: "Terrakion", 140: "Fezandipiti ex",
    304: "Hop's Snorlax", 66: "Alakazam", 245: "Alakazam", 743: "Alakazam",
}
# 同一アーキタイプとして合算する ace の別名（代表ID → 吸収するID）
ACE_MERGE = {66: (245, 743)}


def _ex_dmg_immune(t: str, card_type: int) -> bool:
    """「自分が ex のワザダメージを全て無効にする」壁か（イワパレス型）。

    この判定は **リーサル計算に直接効く**（features._effective_damage / guards.effective_damage）。
    偽陽性は「取れるリーサルを取らない」に直結するので条件を厳しく取る。
    v2 初版は限定語を見ておらず、Farigiraf ex（相手の**たね**exのみ無効）や
    Shaymin（**ベンチ**を守る＝自分が殴られる話ではない）、
    さらにポケモンですらないサポート/スタジアムまで拾っていた（6件中4件が誤り）。
    """
    if card_type != 0:               # ポケモン以外は防御側になり得ない
        return False
    prevents = "prevent all damage" in t or "no damage" in t
    vs_ex = "{ex}" in t or "pokémon ex" in t or "pokemon ex" in t
    to_self = "to this pok" in t
    limited = "basic pok" in t or "benched" in t   # 対象が限定される＝無条件の壁ではない
    return prevents and vs_ex and to_self and not limited


def _flags(text: str, patterns) -> np.ndarray:
    v = np.zeros(len(patterns), dtype=np.float32)
    for i, (_, pat) in enumerate(patterns):
        if re.search(pat, text):
            v[i] = 1.0
    return v


def _onehot(out: np.ndarray, base: int, idx, width: int) -> int:
    """idx が [0, width) なら立てる。-1/None は「なし」= 最終スロット。戻り値は次の基点。"""
    i = int(idx) if idx is not None else -1
    if 0 <= i < width - 1:
        out[base + i] = 1.0
    else:
        out[base + width - 1] = 1.0
    return base + width


def build_tables():
    """cg から静的テーブルを生成する（学習環境専用。提出物ではロードのみ）。"""
    from ptcg.engine import engine_attacks, engine_card_data

    atk_list = list(engine_attacks())
    atk_by_id = {int(a.attackId): a for a in atk_list}

    cards = np.zeros((CARD_ROWS, N_CARD_COLS), dtype=np.float32)
    cards[:, C_ENERGY_TYPE] = -1
    cards[:, C_WEAKNESS] = -1
    cards[:, C_RESISTANCE] = -1
    cards[:, C_CARD_TYPE] = -1
    card_attacks = np.zeros((CARD_ROWS, 4), dtype=np.int32)
    card_effects = np.zeros((CARD_ROWS, N_CARD_EFFECTS), dtype=np.float32)
    card_static = np.zeros((CARD_ROWS, D_CARD_STATIC), dtype=np.float32)

    for c in engine_card_data():
        i = int(c.cardId)
        if not (0 < i < CARD_ROWS):
            continue
        cards[i, C_HP] = float(c.hp or 0)
        cards[i, C_RETREAT] = float(c.retreatCost or 0)
        cards[i, C_ENERGY_TYPE] = float(-1 if c.energyType is None else int(c.energyType))
        cards[i, C_WEAKNESS] = float(-1 if c.weakness is None else int(c.weakness))
        cards[i, C_RESISTANCE] = float(-1 if c.resistance is None else int(c.resistance))
        cards[i, C_CARD_TYPE] = float(int(c.cardType))
        cards[i, C_BASIC] = float(bool(c.basic))
        cards[i, C_STAGE1] = float(bool(c.stage1))
        cards[i, C_STAGE2] = float(bool(c.stage2))
        cards[i, C_EX] = float(bool(c.ex))
        cards[i, C_MEGA_EX] = float(bool(c.megaEx))
        cards[i, C_TERA] = float(bool(c.tera))
        cards[i, C_ACE_SPEC] = float(bool(c.aceSpec))
        atk_ids = list(c.attacks or [])[:4]
        cards[i, C_N_ATTACKS] = float(len(atk_ids))
        card_attacks[i, : len(atk_ids)] = atk_ids

        skill_text = _norm_text(" ".join((s.text or "") for s in (c.skills or [])))
        for s in c.skills or []:
            if _ex_dmg_immune(_norm_text(s.text), int(c.cardType)):
                cards[i, C_EX_DMG_IMMUNE] = 1.0
        cards[i, C_KNOWN] = 1.0
        card_effects[i] = _flags(skill_text, CARD_EFFECT_PATTERNS)

        # --- 静的ブロックを one-hot 展開 ---
        v = card_static[i]
        b = 0
        b = _onehot(v, b, int(c.cardType) if c.cardType is not None else -1, 8)
        b = _onehot(v, b, -1 if c.energyType is None else int(c.energyType), 13)
        b = _onehot(v, b, -1 if c.weakness is None else int(c.weakness), 13)
        b = _onehot(v, b, -1 if c.resistance is None else int(c.resistance), 13)
        v[b] = float(bool(c.basic)); v[b + 1] = float(bool(c.stage1)); v[b + 2] = float(bool(c.stage2)); b += 3
        v[b] = float(bool(c.ex)); v[b + 1] = float(bool(c.megaEx))
        v[b + 2] = float(bool(c.tera)); v[b + 3] = float(bool(c.aceSpec)); b += 4
        v[b] = cards[i, C_EX_DMG_IMMUNE]; b += 1
        v[b] = min(float(c.retreatCost or 0) / 4.0, 1.0); b += 1
        v[b] = min(float(c.hp or 0) / 340.0, 1.0); b += 1
        v[b] = min(len(atk_ids) / 4.0, 1.0); b += 1
        maxdmg = max((float(atk_by_id[a].damage or 0) for a in atk_ids if a in atk_by_id), default=0.0)
        v[b] = min(maxdmg / 300.0, 1.0); b += 1
        v[b] = 1.0; b += 1  # known
        v[b : b + N_CARD_EFFECTS] = card_effects[i]

    damage = np.zeros((ATTACK_ROWS,), dtype=np.float32)
    cost = np.zeros((ATTACK_ROWS, N_ENERGY_TYPES_TBL), dtype=np.float32)
    known = np.zeros((ATTACK_ROWS,), dtype=np.float32)
    atk_effects = np.zeros((ATTACK_ROWS, N_ATTACK_EFFECTS), dtype=np.float32)
    atk_static = np.zeros((ATTACK_ROWS, D_ATK_STATIC), dtype=np.float32)
    for a in atk_list:
        i = int(a.attackId)
        if not (0 < i < ATTACK_ROWS):
            continue
        damage[i] = float(a.damage or 0)
        for e in a.energies or []:
            e = int(e)
            if 0 <= e < N_ENERGY_TYPES_TBL:
                cost[i, e] += 1.0
        known[i] = 1.0
        atk_effects[i] = _flags(_norm_text(a.text), ATTACK_EFFECT_PATTERNS)
        v = atk_static[i]
        v[0] = min(damage[i] / 300.0, 1.0)
        v[1] = min(float(cost[i].sum()) / 5.0, 1.0)
        v[2 : 2 + N_ENERGY_TYPES_TBL] = np.minimum(cost[i] / 4.0, 1.0)
        v[2 + N_ENERGY_TYPES_TBL] = 1.0
        v[3 + N_ENERGY_TYPES_TBL :] = atk_effects[i]

    return {
        "cards": cards,
        "card_attacks": card_attacks,
        "card_static": card_static,
        "atk_damage": damage,
        "atk_cost": cost,
        "atk_known": known,
        "atk_static": atk_static,
        "vocab_version": np.array([VOCAB_VERSION], dtype=np.int32),
    }


def save_tables(tables: dict, cards_path, attacks_path) -> None:
    np.savez_compressed(
        cards_path,
        cards=tables["cards"],
        card_attacks=tables["card_attacks"],
        card_static=tables["card_static"],
        vocab_version=tables["vocab_version"],
    )
    np.savez_compressed(
        attacks_path,
        atk_damage=tables["atk_damage"],
        atk_cost=tables["atk_cost"],
        atk_known=tables["atk_known"],
        atk_static=tables["atk_static"],
        vocab_version=tables["vocab_version"],
    )


def load_tables(cards_path, attacks_path) -> dict | None:
    """numpy のみで読み込み。失敗や版不一致は None（呼び出し側でルール層へ縮退）。"""
    try:
        cz = np.load(cards_path)
        az = np.load(attacks_path)
        if int(cz["vocab_version"][0]) != VOCAB_VERSION or int(az["vocab_version"][0]) != VOCAB_VERSION:
            return None
        return {
            "cards": cz["cards"].astype(np.float32),
            "card_attacks": cz["card_attacks"].astype(np.int32),
            "card_static": cz["card_static"].astype(np.float32),
            "atk_damage": az["atk_damage"].astype(np.float32),
            "atk_cost": az["atk_cost"].astype(np.float32),
            "atk_known": az["atk_known"].astype(np.float32),
            "atk_static": az["atk_static"].astype(np.float32),
        }
    except Exception:
        return None
