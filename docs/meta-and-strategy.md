# メタ分析と戦略（シムのラダー実データ + 現実SVメタ）

上位対戦ログ（Daily Top Episodes 2026-07-13, 全5,039戦を全数解析）と現実SVメタ調査の統合。
**我々にとっての地面はシムのラダー実メタ**（現実の大会メタとは乖離する。理由は §3）。

## TL;DR / 方針転換
- **sample_abomasnow は上位ラダーで採用 0/10,048 ＝ 降ろす**。ローカル総当たり78%は弱プール(greedy_first)由来の過大評価だった。
- シム上位メタは **Alakazam中心**（43.7%）。上位5で90.9%。
- **単一デッキを選ぶなら Alakazam が最有力**（単プライズ・エネ1火力・低スキルフロア＝「ボトルネックはピロット」の我々に最適）。
- エピソードに**両者の完全60枚デッキ**が入る → 実メタの正確なリストをカードIDで抽出可能（名前→ID変換不要）。
- 推定器 `ptcg.meta.classify()` は現メタで72%unknown ＝ **署名を現メタに更新**して opponent-gating を有効化。

## §1. シムのラダー実メタ（2026-07-13, 決着5,022戦, デッキ枠10,048）

判定=「2枚以上採用の最高HPポケモン」をエースとするヒューリスティック（デッキ全60枚が見えるため高精度）。

| シェア | アーキタイプ | 勝率(Wilson95%CI) | エース(ID) |
|---|---|---|---|
| **43.7%** | Alakazam | 50.1% [48.6,51.6] | 非ex Stage2 Alakazam(245) + Dudunsparce(66) + Fezandipiti ex(140) |
| 20.5% | Mega Kangaskhan ex | 50.3% [48.1,52.4] | Kangaskhan(756) 300HP + Crustle壁 |
| 14.4% | Marnie's Grimmsnarl ex | 51.4% [48.8,54.0] | Grimmsnarl(648) 320HP Stage2 |
| 7.2% | Mega Starmie ex | **53.4%** [49.8,57.0] | Starmie(1031) 330HP + Dusknoir |
| 5.1% | TR Mewtwo ex | **57.5%** [53.1,61.7] | Mewtwo(431) 280HP + TRエンジン |

- 上位帯は同型・巧者同士で人気デッキほど勝率50%に収束。その中で **TR Mewtwo(57.5%) / Starmie(53.4%) が有意に field超え**。
- **同型戦が全体の23.2%**、特に **Alakazamミラーだけで17.9%**。

### 三すくみ（相性グリッド, 行の対列勝率）
- **Mega Starmie** → 大型Basicを蹂躙: 対Kangaskhan **79%**, 対Mewtwo **83%**。だが 対Alakazam 40% / 対Grimmsnarl 40%（単プライズ・妨害に弱い）。
- **TR Mewtwo** → 対Kangaskhan 72% / 対Alakazam 58%。唯一の天敵 **対Starmie 17%**。
- **Mega Kangaskhan** → 対Grimmsnarl 66%。Starmie/Mewtwoに轢かれる。
- **Alakazam** → 対Starmie 60%、Kangaskhanと五分、Grimmsnarl(44%)・Mewtwo(43%)に負け越し＝**致命的不利なしの安定中心**。
- **Grimmsnarl** → 対Alakazam 56% / 対Starmie 60% だが 対Kangaskhan 34%。

### 試合長・手番
- 全体中央値 **151 step**。**Starmieは100（最速レース）**、Kangaskhan 142 < Alakazam 158 ≈ Grimmsnarl 160 < Mewtwo 166。
- **Kangaskhanは短期戦だと負ける**（勝ち157/負け116）＝速攻に轢かれる。Starmieは勝敗問わず~100＝完全レース特化。
- **先攻勝率 54.3% / 後攻 45.7%（+8.6pp）**。旧実測(55.2/44.8)と一致＝先攻優位は事実。

### 各デッキの勝ち筋（戦法）
- **Alakazam**: 単プライズ・トゥールボックス。手札枚数×20をエネ1で撃つ("Powerful Hand")+相手エネ×50でMega群を咎める。低コスト火力×耐久×妨害(Enhanced Hammer/Boss)。**スキルフロアが低い＝単純ピロット向き**。
- **Kangaskhan**: 300HP脳筋+Crustle壁+回復。長期志向（速攻に弱い）。
- **Grimmsnarl**: 進化時に闇エネ5枚サーチで爆加速→180+ベンチ30のミッドレンジ。
- **Starmie**: 最速。エネ1で120+ベンチ50、210(弱点/効果無視)、Dusknoir130スナイプ。大型Basicメタの筆頭カウンター。
- **TR Mewtwo**: TR4体展開が条件のエンジン型。160+捨てエネ×60。上振れ最強だが**重く・Starmieに激弱**でシンプルピロット不向き。

## §2. 現実SVメタ（文脈・出典付き / 2026年年央, Mega Evolution期「Perfect Order」）
2026/4ローテで Charizard/Gholdengo/Gardevoir が全滅、**Dragapult ex一強**（~20%）。以下 N's Zoroark / Slowking / Mega Greninja / Alakazam / Ogerpon-Meganium / Mega Lucario / Raging Bolt。妨害系(Honchkrow, Crustle, Slowking)がローテで相対上昇。現実の Mega Abomasnow ex は勝率~39%のファンジ。

## §3. なぜ現実とシムが乖離するか（重要）
1. **シムはローテ無しの固定プール**。現実で落ちた旧強豪が、シムのカードDBにあれば現役たり得る。
2. **シムの相手は人間でなくエージェント**。人間には難しいデッキ(Dragapultの妨害タイミング等)を各エージェントが回せず、逆に**単純に強いデッキ(Alakazam等)が上位を占める**。だから **Kangaskhan/Alakazam のようなシム固有メタ**が形成される。
3. → **意思決定はシムのラダー実メタ(§1)を最優先**。現実メタ(§2)は相手モデルの補助・カード理解に使う。

## §4. 我々の位置と、ローカル評価が誤誘導した理由
- 現ライブ: greedy_first 409.9 / greedy_lethal 600 / abomasnow_pilot(提出直後)。**全て sample_abomasnow デッキ**＝上位メタに存在しないデッキ。600点は「そこそこ」だが上を目指すなら**メタデッキが必要**。
- **ローカル総当たりで abomasnow が78%だったのは、相手プールが弱い greedy_first のみで、実メタの強デッキ(Alakazam等)が居なかったから**。＝ プールの代表性が無いと結論を誤る好例（過学習・非代表評価の教訓）。

## §5. 推奨アクション（優先度順）
1. **エピソードから実メタの正確な60枚デッキを抽出**（Alakazam最優先, 次いでKangaskhan/Grimmsnarl/Starmie/Mewtwo）→ `decks/` に追加。全カードIDが揃うので合法性も担保。
2. **Alakazam を greedy_lethal 系ピロットに載せる**（低スキルフロア＝期待値最大）→ smoke/A/B → 3枠目に提出して実ラダー検証。
3. **推定器 `SIGNATURES` を現メタで更新**（Alakazam245/Dudunsparce66/Fezandipiti140, Kangaskhan756, Grimmsnarl648, Starmie1031, Mewtwo431）→ opponent-gating を実メタで有効化。相性が20〜80%と急峻なので、**特に対Starmieは絶対にダレない**等のペース調整が効く。
4. **評価プールを実メタデッキで再構成**（greedy_first@実メタデッキ、または抽出した実エージェント風味）→ ローカル評価の代表性を回復（これで初めてピロット改善が測れる）。
5. コインで先攻を取りにいく（+8.6pp）が、後攻でも回る構築を優先（Alakazamは先攻依存が低い）。

## §6. 実メタ込み総当たり検証 → 採用デッキ訂正（Alakazam説は却下）
`scripts/compare_decks.py`（behavior=greedy_lethal2 固定, 12デッキ, 120戦/ペア）で、抽出した実メタデッキ込みで再検証。総合勝率:

| 順位 | デッキ | 勝率 |
|---|---|---|
| 1 | **meta_mega_kangaskhan_ex** | **78.8%** [76.5,80.9] |
| 2 | sample_abomasnow | 72.4% [70.0,74.8] |
| 3 | meta_mega_starmie_ex | 60.4% |
| 4 | meta_marnie_s_grimmsnarl_ex | 53.8% |
| 5 | meta_team_rocket_s_mewtwo_ex | 49.5% |
| 6-9 | archaludon / garchomp / crustle / lucario | 43–48% |
| 10 | **meta_alakazam** | **38.6%** |
| 11-12 | dragapult / iono | 27–31% |

- **訂正**: §5 で（episode解析の "低スキルフロア" 推定に基づき）Alakazam を薦めたが、**我々の単純ピロット(greedy_lethal2)では Alakazam は10位/12・38.6% ＝最弱級**。Alakazam の「手札枚数×火力・トゥールボックス・単プライズ交換」は巧いピロット前提で、greedy racing では回らない。"低スキルフロア" は人間基準でありAI（我々のピロット）基準ではなかった。→ **憶測でなく総当たりで選ぶべき、の実例**。
- **我々のピロットに最適＝big-Basic 脳筋ビート**: Kangaskhan 78.8% > abomasnow 72.4% > Starmie 60.4%。Kangaskhan は現ライブの abomasnow にも **75.8%** で勝つ。
- **実メタシェア加重の対field勝率(概算)**: Kangaskhan **≈65%** ≫ abomasnow **≈51%**。abomasnow は Kangas(24%)・Starmie(31%) が痛く、Kangas は 20% シェア。Kangaskhan の穴は Starmie(~50%) のみで Starmie は 7% しか居ない。
- **留保**: 相手も greedy_lethal2（実ラダーの巧いピロットより弱い）ため絶対値は過大評価。ただし**デッキ選択（ピロット固定）の相対比較としては妥当**で、Kangaskhan の優位は明確。
- **決定: 採用デッキ = `meta_mega_kangaskhan_ex`**（Alakazam説は却下）。次アクション: Kangaskhan にピロットを載せて A/B → 実ラダー提出。推定器署名の現メタ更新も並行。

## 留保
- 上位episode＝好成績帯の偏り、全ラダーの縮図ではない（10%ランダム相手は別デッキ）。1日分でメタは短期変動（Lucario 42%→0.3%）。エースは単一ラベル近似。

## ソース
- シムlog解析: `kaggle/pokemon-tcg-ai-battle-episodes-2026-07-13`（全数解析, 手法は本文）
- 現実メタ: https://play.limitlesstcg.com/decks?format=standard ／ ローテ https://www.pokemon.com/us/pokemon-news/2026-pokemon-tcg-standard-format-rotation-announcement ／ 各デッキ limitlesstcg.com/decks/{284 Dragapult, 320 Zoroark, 345 Lucario, 351 Ogerpon, 280 RagingBolt, 341 Crustle, 356 Honchkrow} 他（詳細は調査ログ）
