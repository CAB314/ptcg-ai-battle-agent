# 本文の数値台帳（report-draft.md v5 → 出典）

本文（`docs/report-draft.md` の `---\n---` 区間）に登場する全数値と出典。台帳に無い数値は本文から消す。
出典の `L####` は `docs/experiments-log.md` の行番号（2026-09-05 時点）。JSON は `docs/data/`。

## §1 The problem the ladder actually poses

| 数値 | 出典 |
|---|---|
| 75 days / 6 leader changes (7-day average) | `meta_timeline.csv`（6/16–8/30）→ `make_report_figures.fig1_meta_timeline` が7日移動平均で首位交代を計数 |
| Grimmsnarl 64% on Jul 28 → 1% on Aug 30 | 同上（fig1 注記、データから算出） |
| Dragapult 9% → 32% in 19 days | 同上 |
| 20 teams, identical list, 100+ games, 3 days, 43–57% | `clone_winrates.json`（dates 8/2–8/4, teams n≥100）→ `fig_clone_spread` |
| "gives each team two active slots and scores the better one" | 本戦 Evaluation ページ（"we only track the latest 2 submissions … only your best scoring agent will be shown"）・topic 714189 |

## §2 Why win rate cannot tell you what to play

| 数値 | 出典 |
|---|---|
| 53.7% (0–1 day since submission) / 50.6% (4–7 days) | L685-708（表: 0-1日 495,733決定 53.7% / 4-7日 53,943決定 50.6%）→ `fig3_rating_confound` |
| "Three of our own instruments" (v3 で3件に圧縮) | 下記3行。ガントレット起動競合（L599-602）は本文から外した |
| self-play evaluation bug, one decision reversed | L548-580（55.5→58.5% / 53.7→48.0%） |
| release gate green while model never fired | L604-607（既存バグ9件の #2,#3）・閾値 0.95 は `scripts/tar_smoke_test.py --min-model-rate` 既定値 |
| 36% of parameters dead | L471-495（旧 v1 モデル 902,915 中の埋め込み） |
| §8 "four broken instruments" | 上記3件 + ガントレット起動競合（L599-602） |

## §3 The agent

| 数値 | 出典 |
|---|---|
| 56 board tokens | `src/ptcg/ml/features.py` `S_TOKENS = 56` |
| 2 self-attention + 2 cross-attention layers, d=128, 4 heads | `runs/selfplay_dh/kangaskhan/config.json`（n_state_layers 2 / n_option_layers 2 / d_model 128 / n_heads 4） |
| 691,843 learned parameters / 172,480 static attributes / 2.6 MB | `weights.npz` 実測（plan-writeup §7.3; card_static 1536×80 + atk_static 1600×31）。2.6 MB は 9/13 に本文から外し Table 1 のみに |
| 3.4 ms per decision (mean) | `scripts/bench_inference.py` 9/3 実測（dh 3.40/7.44, k2 3.42/7.30; 各24試合） |
| 600-second game budget, no per-move limit | 本戦 topic 726708（運営回答 2026-07-16: "600s per game - no time limit per move"）。当方メモの「≈1秒/手」は目安で公式でないため本文から削除 |
| 31.96 million decisions / 47 days | `runs/bc_pretrain_v3.log`（train=31,958,161） |
| 1.5 GPU-hours | `runs/bc_pretrain_v3/metrics.jsonl`（2,736 + 2,757 s = 1.53 h） |
| 73.4% temporal hold-out agreement | 同上（tval acc 0.7344） |
| +6.6 / +7.3 points from shared pre-train | L264-265（alakazam 73.5→80.2 / trmewtwo 68.1→75.4）。**9/13 に語数のため本文から削除**（台帳に残す） |
| ≈100 GPU-hours total | plan-writeup §7.2（PPO 約95h + BC 1.5h ほか）→ `table_compute.png` |
| lethal guard applied as mask before sampling | `src/ptcg/ml/rl/masks.py guard_mask` / `actor.py _np_agent_argmax` |
| parity max |Δ| 7.2e-7, argmax 100% | L515 |
| infinite mirror game → ERROR; per-turn cap | L193（18,167 steps, 7/17 alakazam_bc）・`scripts/tar_smoke_test.py` docstring |
| no errored submission since | `kaggle competitions submissions` 9/3（全件 COMPLETE） |

## §4 The league

| 数値 | 出典 |
|---|---|
| 50% mirror / 30% snapshots / 20% fixed | `configs/ppo_opp3_lucario.json`（league_latest .5 / snapshot .3 / fixed .2） |
| 12 snapshots, one every 150 iterations | 同上（snapshot_pool_max 12 / snapshot_every_iters 150）。「150 iterations ごと」は 9/13 に本文から外した |
| decks 55% meta / 25% weak / 20% uniform | 同上（deck_mix） |
| PFSP: max(floor, 1 − recent win rate)²（recent = EMA） | `src/ptcg/ml/rl/league.py` docstring; floor = pfsp_floor 0.05 |
| KL anchor target 0.03, adaptive β | 同 config（kl_target 0.03, β ×1.5 刻み, [0.05, 20]） |
| hourly gauntlet with rollback | `jobs/*.sbatch --loop 3600`（公開版では削除）・`scripts/ppo_gauntlet.py`; rollback = `learner._check_rollback` |
| evaluator runs exported NumPy policy through the submission path | `scripts/selfplay_judge.py`（配備同一経路） |
| median **gauntlet** gain +10.6 / +7.3 / +4.5 | `league_rounds.json` → `fig_league_decay`（中央値は図中で算出）。ガントレット（固定対戦場）の値。§5 の field benchmark（他チームの実測）と区別するため 9/13 に "field gain" → "gauntlet gain"、§5 Case 1 の "field score" → "gauntlet score" に改称 |

## §5 Weak matchups

| 数値 | 出典 |
|---|---|
| field benchmark definition | `scripts/field_matchup_matrix.py`（他チームのエージェントのアーキ×アーキ勝率） |
| Hydrapple vs Kangaskhan: ours 4.0% | `curriculum_results.json` stages[0] |
| other teams ≈ half | `field_matchup_matrix_full.json` 48.1% (n=5,594) / 8/9–8/14窓 56.8% (n=340) / 最終窓 49.2% (n=3,862) |
| 4.0 → 18.7 → 29.3%; gate 25%; 64.4 → 72.6%; 4.7 GPU-h; zero regressions in 9 matchups | `curriculum_results.json` hydrapple_vs_kangaskhan（L1043, L1059-1063） |
| Ogerpon vs Kangaskhan: ours 2.2%; benchmark 22%; "left it at 1.3%" | ours 2.2%: commit 7b15cb4 / L933; benchmark: `field_matchup_matrix_full.json` 22.2% (n=933); 1.3%: L1045（vs kang_r3 プロキシで 1.2→1.3）。2.2% と 1.2% は別測定なので本文では並置しない（9/13） |
| Ogerpon vs Lopunny: ours 16.7%, benchmark 15.2% (n=877) | ours: L1020; benchmark: `field_matchup_matrix_lopunny-log-window.json`（ログ値 15.1 と一致） |
| archetype with 74.5% benchmark vs Lopunny (Hydrapple) | `field_matchup_matrix_lopunny-log-window.json`（ログ値 74.0） |
| five tech packages, 64,000 games, −15.7 to −23.1 | L1070-1093 |
| "55% were Ogerpon–Meganium hybrids" | L1083（直近6日の勝ちリスト 478 のうち 55% が Meganium 線） |
| seeding: −19 to −23 points; lighter dose −5 | L1143-1147 |
| Dragapult stage predicted 71.8% → ladder 78% | `curriculum_results.json` kangaskhan_dh.before_after / `final_pair_matchups.json`（77.8%, n=135） |
| Hydrapple stage predicted 82.8% → ladder 47% | 同上（47.4%, n=57） |

## §6 The portfolio

| 数値 | 出典 |
|---|---|
| "In the week before the freeze": Dragapult 8 → 22%, Grimmsnarl halved | L1119-1122（8/9→8/14 上位帯、検出 8/16、凍結 8/17）。旧「Two days before」は不正確（データ窓は 3〜8 日前）なので 9/13 に修正 |
| "which favoured the Kangaskhan agent" | `final_pair_matchups.json` dh 77.8% vs Dragapult, k2 51.4% |
| leaderboard = better of two agents; final 993.8 is kangaskhan_dh alone | 本戦 Evaluation ページ（"only your best scoring agent will be shown"）・submissions（dh 993.8 / k2 932.6） |
| Hydrapple "would have been" better slot on 46 of 75 days; +9 pts late July; Kangaskhan +12 late August; better-of-two 59% vs 56% | `portfolio_reweighting.json` summary（days_hydrapple_k2_better 46; late_july 62.81 vs 53.77; late_august 59.80 vs 47.36; mean_E_best 59.0 vs 55.6/56.6）→ `fig_portfolio_hedge`。反実仮想の再重み付け（最終10日の対面勝率を固定）であり historical backtest ではない |
| "the score is one agent's" / 932.6 | submissions: kangaskhan_dh 993.8, hydrapple_k2 932.6 |
| 56–57% over final 1,000 games each; "no errored game" | `final_pair_matchups.json`（dh 56.6% / k2 56.9%）; `replay_behavior_*.json` errors: 0 / 0 |
| n = 94–142 per cell（引用した3対面） | `final_pair_matchups.json`: dh Lopunny 111 / Kangaskhan 94 / Dragapult 135; k2 127 / 120 / 142。旧「94–249」は未引用の Grimmsnarl を含んでいた（9/13 修正） |
| 993.8 / 932.6, rank 138 of 6,807 | Kaggle LB CSV 2026-09-03（`cabbage patch` 138位）・submissions |
| Lopunny 35% / 68%; Kangaskhan 55% / 36%; Dragapult 78% / 51% | `final_pair_matchups.json`（35.1/67.7, 55.3/35.8, 77.8/51.4） |
| Alakazam 34% / 35%, 309 games, third most common | 同上（34.0% n=144 / 35.2% n=165） |
| Hydrapple vs Alakazam benchmark 33.5%; Kangaskhan vs Alakazam 44.7% | `field_matchup_matrix_final-pair-window.json`（n=1,622 / 1,881） |
| 7 of 11 predictions outside 95% CI, up to 35 points | `fig_internal_vs_ladder`（図中で算出; 最大 dh vs Hydrapple −35.4） |

## §7 Deck

| 数値 | 出典 |
|---|---|
| Kangaskhan deck: Mega Kangaskhan ex ×2, Crustle ×3 (**immune to damage from attacks by** Pokémon ex), Jumbo Ice Cream ×4, Hero's Cape ×1, Boss's Orders ×4 | `agents/kangaskhan_dh/deck.csv` × `data/EN_Card_Data.csv`（該当刷りの効果テキスト: Crustle DRI 12「Prevent all damage done to this Pokémon by attacks from your opponent's Pokémon ex」）→ `deck_roles.json` / `fig_deck_anatomy`。9/7 のユーザーレビューで「ワザによるダメージ」に限定 |
| Hydrapple deck: Meganium ×2 (**each Basic Grass on your board provides two**), Hydrapple ex ×2 (attach Basic Grass from hand), Teal Mask Ogerpon ex ×4, Mega Meganium ex ×1, Forest of Vitality ×4 | `agents/hydrapple_k2/deck.csv` × カード効果テキスト（Meganium MEG 10 / Hydrapple ex SCR 14 / Forest of Vitality MEG 117）→ 同上。Forest of Vitality は**両プレイヤー**に適用され初手ターンは除く |
| 20 single-card changes, 10,000 games each; stage 2 = 20,000 fresh games per arm | `ablation_results/ablation_stage1_cluster_results.json`（14）+ `ablation_mewtwo_results.json`（6）; `s2_*_eval.json` n=20,000 |
| +1.8 (z = 3.8) 3rd Boss's Orders for Tool Scrapper; +1.1 (z = 2.2) 3rd Mewtwo ex — directional | `s2_grim__TS__Boss_eval.json` 0.632 vs `s2_grim_base_eval.json` 0.614; `s2_mw_mw_mewtwo3_eval.json` 0.579 vs `s2_mw_base_eval.json` 0.568（各 n=20,000; z は図中で算出）; カード名は `ablation_candidates.json` の差分 |
| "z = 2.2 on the fresh confirmation set — suggestive rather than decisive" | 第2段は選択に使っていない新規 20,000 戦×2（`s2_*_eval.json`）。再検定した候補は 4 本なので、20 候補の多重性は当てはまらない（ラウンド2で表現修正） |
| "one to two points" (confirmed single-card change) | 第2段の +1.8 / +1.1 |
| "switching archetype moved individual matchups by 19–33 points" | `final_pair_matchups.json`: Lopunny 35.1 vs 67.7 (32.6) / Dragapult 77.8 vs 51.4 (26.4) / Grimmsnarl 52.6 vs 74.9 (22.3) / Kangaskhan 55.3 vs 35.8 (19.5) |
| "the most common top-rated list of their archetype when we adopted them, unchanged" | `scripts/mine_decklists.py`（最頻リスト抽出）。hydrapple_k2/deck.csv == decks/meta_hydrapple_ex.csv（8/10 採掘、9/6 に一致確認）; kangaskhan_dh/deck.csv = 7月採掘の当時の最頻リスト（L18「Mega Kangaskhan ex 実メタデッキ(最頻リスト)」; 8/10 刷新版とは不一致） |
| リプレイ実測すべて（§7「Did they play as designed?」と Fig. 11 の下段） | `scripts/analyze_replay_behavior.py` → `docs/data/replay_behavior_{kangaskhan_dh,hydrapple_k2}.json`（各1,000試合・`_matchups.json` の窓）。図の下段は JSON から動的に文を生成しており、本文の数値も同じ JSON |
| Crustle 在場時: ex（特性なし）96% が0ダメージ n=1,028 / 非ex 5% n=941 | `replay_behavior_kangaskhan_dh.json` `incoming_by_attacker`。ex は Card ID から `ex or megaEx`、特性は `skills` の有無（エンジンのカード表） |
| Cornerstone 在場時: 特性持ち 80% が0ダメージ n=970 / 特性なし 7% n=149 | 同上（ex/mega/非ex を合算して特性の有無で層別） |
| Mega Lopunny ex は 44% しか止まらない n=545 | 同上 `mega-ex` 行。機構: Spiky Hopper「This attack's damage isn't affected by any effects on your opponent's Active Pokémon」（エンジンのワザ表）。貫通の内訳は `leak_by_attack`（Spiky Hopper 300件・160ダメージ） |
| 残りの貫通は効果によるダメカン配置 | `leak_by_attack`: Phantom Dive は Crustle に 60（=ダメカン6個）だけ通り 200 は防がれている。Dragapult ex は特性なし（`skills` 空）なので Cornerstone では 200 が通る（35件） |
| Crustle が与ダメージの 65%（Superb Scissors）、Mega Kangaskhan ex 19%（Rapid-Fire Combo）、Cornerstone 16%（Demolish） | `replay_behavior_kangaskhan_dh.json` `our_attacks` の `total_damage` 比（65.3 / 18.6 / 16.1） |
| Hydrapple: Syrup Storm 68% / Teal Mask Ogerpon ex（Myriad Leaf Shower）27% | `replay_behavior_hydrapple_k2.json` `our_attacks`（67.8 / 26.9）。旧本文の「so Teal Mask Ogerpon ex reaches lethal damage fast」は主砲を誤っていたため 9/13 に修正 |
| Syrup Storm "30 plus 30 per Grass Energy on our side" | エンジンのワザ表: 30 + "30 more damage for each {G} Energy attached to all of your Pokémon"（Meganium の2倍化ぶんを含む提供数） |
| "we won 35% there while the second slot won 68%"（対 Lopunny） | `final_pair_matchups.json` 35.1 / 67.7。旧「our worst and the second slot's best」は誤り: dh の最低は Alakazam 34.0（n=144）、k2 の最高は Lucario 77.1 / Ogerpon 76.7 / Grimmsnarl 74.9（9/13 修正） |
| Meganium 90.0% 着地（勝率 58.8% vs 40.0%）/ Hydrapple ex 84.3%（61.3% vs 33.1%）/ Mega Meganium ex 7.3% | `replay_behavior_hydrapple_k2.json` `key_cards`。条件つき勝率は交絡あり（長引いた試合ほど着地しやすい）ため因果とは書かない |
| Wild Growth の2倍化: 2,691 盤面すべてで「提供エネルギー = 基本草の枚数×2」 | 同 JSON `wild_growth_doubling`。盤面の `energies` 配列（提供エネルギーの実体）と `energyCards`（カード枚数）の比較 |
| Syrup Storm 平均468・78% が「30+30×提供エネルギー」に一致（枚数一致は9%） | 同 JSON `damage_scaling`。**ユーザーのレビュー回答（現実ルールでも2個分として数える）とエンジン実装が一致**。残差は弱点2倍・Growl の −20・スナップショット間のエネルギー増減 |
| リプレイのイベント列の復元 | 各エージェントの `observation.logs` は「そのエージェントが前回促された時点以降」で促されない間は同一バッチが再掲される。**自エージェントが ACTIVE のステップのみを連結**する。相手視点から復元した列と件数が一致（差は最終ターン末尾1件以内）。壁の判定は自分のバトル場の serial 宛のダメージのみを数え、ベンチ被弾は別勘定 |
| Fig. 11 の役割・注記すべて | `data/EN_Card_Data.csv` の効果テキストから再導出（刷りは `deck.csv` の Card ID で確定: Crustle DRI 12 / Battle Cage PFL 85 / Mist TEF 161 / Spiky JTG 159 / Cornerstone TWM 112 / Bayleef MEG 9 / Dipplin SCR 13）。2026-09-07 にユーザーの PTCG レビューを受けて表現を厳密化（`deck_roles.json` の `verification.corrections` に11件） |
| Cornerstone Mask Ogerpon ex の役割 | **旧記述「闘弱点を突くアタッカー」は誤り**: Demolish は「damage isn't affected by Weakness or Resistance」。実体は第2の壁（Cornerstone Stance = 特性持ちポケモンのワザのダメージを防ぐ）。9/7 に自己検出・修正 |
| デッキリスト（添付用） | `scripts/export_decklists.py` → `docs/data/decklists/*.{txt,csv}`。60枚・`deck_roles.json` の枚数と差分0 |
| −6.1 zero-shot → −3.9 fine-tuned | `field_v3` −6.13 (stage1) / `s2_grim_field_v3_eval.json` 0.576 vs 0.614 |
| +1.4 zero-shot → −0.2 fine-tuned | `mw_night2` +1.37 / `s2_mw_mw_night2_eval.json` 0.566 vs 0.568 |

## §8

数値なし。
