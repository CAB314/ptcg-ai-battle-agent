"""kangaskhan_playbook2 — D3で効いたパターンの移植 + 狭い条件分岐（各分岐は個別A/Bで採否）。

ベース: kangaskhan_playbook の「敵側ターゲティングのみ」構成（味方側の包括上書きは−22ptで失敗済み）。
追加分岐はすべて「エンジンが知り得ない・条件が証明できる場面」に限定し、フラグで個別にON/OFF。

メモ: 先攻選択(IS_FIRST)は計測の結果、エンジンの option[0]=YES=先攻 を greedy が既に
選んでいるため実装不要（options=[YES, NO] を10戦×複数selectで確認）。
"""

# ---- 分岐フラグ（A/Bで個別に検証して確定する）----
ENABLE_GUST_PRIORITY = False        # 分岐1: ガスト優先（敵側・低リスク）
ENABLE_WALL_PROMOTE = True         # 分岐2: 壁検出時のみの前出し変更（条件付き味方側）
ENABLE_DEAD_ACTIVE_SWITCH = True   # 分岐4: 死にアクティブの入れ替え（実験枠）

# ---- 分岐1: ガスト優先度（相手カードID -> ボーナス）----
# Dwebble: 進化すると Crustle 壁(ex技無効)になり、うちのメガガルーラが完全に止まる。壁になる前に狩る。
# Duskull/Dusclops/Dusknoir: 天敵 Starmie 型(対Kangaskhan 79%)のスナイプエンジン。早期に除去。
GUST_PRIORITY = {
    344: 300, 532: 300,          # 相手の Dwebble（壁の前身）
    131: 200, 132: 200, 133: 200,  # Duskull / Dusclops / Dusknoir
}

# ---- 分岐2: 壁検出時の前出し優先度 ----
# 相手の場に ex技無効の壁（Crustle系）がいるときだけ発火。
# 非exの自Crustleライン（壁に通常打点が通る）を前へ、メガガルーラ(0ダメージ)を下げる。
WALL_PROMOTE_PRIORITY = {
    345: 40,   # 自Crustle — 壁を割れる非ex（Superb Scissors 120）
    344: 20,   # 自Dwebble — Ascension で Crustle へ
    756: -30,  # メガガルーラex — 壁に0ダメージ
    343: -10,  # シェイミ — 部品
}

# ---- 分岐4: 死にアクティブ入れ替え ----
SWITCH_CARD = 1123  # グッズ「ポケモンいれかえ」(デッキに4枚)
