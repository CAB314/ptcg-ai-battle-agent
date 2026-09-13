"""greattusk_lo_playbook — イダイナキバ+イワパレスLO（Issue #4）の分岐テーブル。

Issue #4 の3候補（ピジョットex/カビゴン/イダイナキバ）のうち、シムのカードプールで
成立するのはイダイナキバLOのみ（Pidgeot ex・とおせんぼカビゴンはプール不在）。
デッキは検証済みの decks/greattusk_crustle_lo.csv。

方針（kangaskhan_playbook2 の運用則を踏襲）:
- 汎用分岐 = LOの骨格（ミル優先攻撃・EGシークエンシング・自滅LOガード・正確打点リーサル）。
- 局所分岐 = 発火条件が証明できる場面だけの狭い上書き。各フラグは個別A/Bで採否。
"""

# ---- カードID ----
GREAT_TUSK = 58        # たね闘140。Land Collapse(●●)=ミル1/古代サポ併用で4
DWEBBLE = 344          # たね草70。Ascension(●)=山からCrustleへ進化
CRUSTLE = 345          # 1進化草150。特性=相手exのワザダメージ全防御（「EX中にイワパレス」）
TERRAKION = 607        # たね闘140。Retaliate(F●)=50/前ターン被KOで+80

FIGHTING_GONG = 1142   # 山から闘たね（GT/テラキオン）をサーチ
POKE_PAD = 1152        # 山からルールボックス無しポケモンをサーチ
POFFIN = 1086          # 山からHP70以下たね（=Dwebble）を2体ベンチへ
POKEGEAR = 1122        # 山上7枚からサポートを1枚
ULTRA_BALL = 1121      # 手札2枚コストでポケモンサーチ
SWITCH = 1123          # ポケモンいれかえ（スタンス切替の主手段。にげる3は使わない）
JUMBO_ICE_CREAM = 1147  # エネ3以上のバトル場を80回復

EXPLORERS_GUIDANCE = 1185  # 古代サポ。山上6→2枚手札・4枚トラッシュ（自傷ミル）
XEROSIC = 1197             # 相手手札を3枚まで削る
BOSS = 1182                # ボスの指令
LISIA = 1204               # 相手ベンチのたねを呼び+こんらん
COLRESS_TENACITY = 1194    # 山からスタジアム+エネをサーチ

NEUTRALIZATION_ZONE = 1247  # 非ルールボックスへのex/Vダメージ全防御（トラッシュから回収不可）
ROCK_FIGHTING_ENERGY = 20   # {F}扱い+闘ポケモンへのワザ効果防御
MIST_ENERGY = 11            # {C}扱い+ワザ効果防御

# ---- ワザID ----
ATK_LAND_COLLAPSE = 62   # ミル（このデッキの勝ち筋）
ATK_GIANT_TUSK = 63      # 160。リーサル時のみ
ATK_ASCENSION = 478      # Dwebble縛られ脱出=山からCrustle進化
ATK_RETALIATE = 873      # 50/+80（報復）
ATK_LAND_CRUSH = 874     # 100

# ---- 分岐フラグ（A/Bで個別に検証して採否を決める）----
ENABLE_MILL_FINISHER = True       # 汎: 相手山4以下でEG+Land Collapseの詰め
ENABLE_EG_GATE = True             # 汎: 自山残量ゲート（自滅LO対策）
ENABLE_ATTACH_POLICY = True       # 汎: エネ付け先の優先度（GT2枚を最優先）
ENABLE_SUPPORTER_TREE = True      # 汎: サポーター選択木
ENABLE_KEEP_POLICY = True         # 汎: EG/ポケギア/ボール類の取捨スコア
ENABLE_SETUP_POLICY = True        # 汎: セットアップ配置（GT前・Dwebbleベンチ）
ENABLE_WALL_STANCE = True         # 局: 相手アクティブex & NZ不在 → Crustle前出し
ENABLE_NZ_LOGIC = True            # 局: NZ設置タイミング + コレスの気丈でサーチ
ENABLE_COLRESS_ENERGY = True      # 局: エネ枯渇時（手札エネ0&GTエネ不足）のコレス=エネサーチ
ENABLE_HAND_SCALER_XEROSIC = True  # 局: 手札枚数×ダメ型（Alakazam）へのクセロシキ減衰
ENABLE_GUST_LOGIC = True          # 局: ボス/リーシアの足止め・詰めガスト
ENABLE_RETALIATE_REVENGE = True   # 局: 前ターン被KO時のテラキオン報復(130)
ENABLE_ASCENSION_ESCAPE = True    # 局: Dwebble縛られ時のエネ付け+Ascension脱出
ENABLE_STANCE_SWITCH = True       # 局: いれかえ札によるスタンス修正（壁⇔ミル）
ENABLE_CONDITION_SWITCH = True    # 局: まひ/ねむりのバトル場をいれかえで解除
ENABLE_BOARD_SURVIVAL = False     # 局: 場が2体以下なら頭数補充を最優先（reason 3=場切れ対策）
# ↑ A/B（200戦/相手×19, pool）: ON=75.6% [74.2–76.9] vs OFF=76.8% [75.4–78.1] = −1.2pp。
#   改善なし（対TR Mewtwoも47.0%→43.0%）のためOFF。コードは再検証用に保持。
ENABLE_PRIZE_MAPPING = True       # 汎: サーチ時の山全観測からサイド落ちを確定し remaining を補正
# ↑ A/B（200戦/相手×19, pool）: ON=75.2%/76.2%（2回） vs OFF=76.8% [75.4–78.1] ＝ノイズ床内で中立。
#   確実な空振り回避・コレスのNZ誤誘導排除という正確性修正のため採用（greedy_lethal2 の前例に従う）。
#   整合性チェック（Σサイド落ち＝サイド残枚数）819回ズレ0。真価は弱プールでなく実ラダーで確認。
ENABLE_PRIZE_EFFICIENT_GUST = True  # 局: KO可能なガスト対象間でプライズ効率（報酬大・必要打点小）を優先
# ↑ LOデッキでは発火が稀（攻撃予算が通常0）で単独A/Bは測定解像度以下。
#   同KO対象間のタイブレークのみを変える狭い上書きとして採用（アタッカー型デッキで要再検証）。

BOARD_THIN = 2                    # 場のポケモンがこの数以下なら「盤面危機」

# ---- 自滅LOガード（reason 2: 山0でターン開始=敗北）----
EG_MIN_SELF_DECK = 10    # 自山がこれ未満ならEGを打たない（詰め時を除く）
EG_FINISHER_MIN_DECK = 7  # 詰めEG（相手山<=4）でも自山がこれ未満なら打たない
SEARCH_MIN_SELF_DECK = 3  # 自山がこれ以下なら任意サーチ（ポフィン/ゴング等）を止める

# ---- サポーター閾値 ----
XEROSIC_BIG_HAND = 7      # 相手手札がこれ以上ならクセロシキ最優先（>EGの+3ミル相当）
XEROSIC_FALLBACK_HAND = 5  # EGが打てない番の代替クセロシキ
HAND_SCALER_MIN_HAND = 5  # 手札×ダメ型相手にクセロシキを撃つ最小手札数

# ---- ガスト（足止め）----
STALL_GUST_MIN_SCORE = 60  # これ未満の足止め対象しかいなければガストを温存

# ---- セットアップ/前出し優先度 ----
SETUP_ACTIVE_PRIORITY = {GREAT_TUSK: 100, TERRAKION: 60, DWEBBLE: 40}
SETUP_BENCH_PRIORITY = {DWEBBLE: 60, GREAT_TUSK: 50, TERRAKION: 40}

# ミルスタンス（通常）: エネの乗ったGTを最優先で前へ
PROMOTE_MILL = {GREAT_TUSK: 100, TERRAKION: 40, CRUSTLE: 25, DWEBBLE: 10}
# 壁スタンス（相手アクティブex & NZ不在）: Crustleを前へ、GTを温存
PROMOTE_WALL = {CRUSTLE: 200, DWEBBLE: 40, TERRAKION: 25, GREAT_TUSK: 15}
