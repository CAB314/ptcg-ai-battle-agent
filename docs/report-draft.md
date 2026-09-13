# レポート本文 v5（Kaggle Writeup / 2,000語上限）— 2026-09-13 最終仕上げ（論理性・データ正確性・平易性・評価指標）まで反映

- 提出先: `pokemon-tcg-ai-battle-challenge-strategy`（締切 2026-09-13 23:59 UTC = 9/14 08:59 JST・**1チーム1回のみ**）
- 配点: Model 70% / Deck 20% / Report 10%。本文は `---\n---` の区間のみが語数対象（図表・Media Gallery は語数外）
- 主軸 = **ポートフォリオ設計**（ユーザー決定 9/3）。§4 リーグ・§5 弱点対面を厚く（ユーザー指示 9/3）
- 用語: "human ceiling" でも "field ceiling" でもなく **field benchmark**（ラダー参加者は全員 AI エージェント。かつ §2 の交絡があるので上限ではない）
- 数値の出典台帳: `docs/report-numbers.md`

語数カウント: `poetry run python scripts/count_words.py docs/report-draft.md`

図番号 → ファイル（Media Gallery に添付する順）:
| 本文 | ファイル | 内容 |
|---|---|---|
| Fig. 1 | `fig1_meta_timeline.png` | メタシェア推移 6/16–8/30 |
| Fig. 2 | `fig_clone_spread.png` | 同一60枚クローンの勝率分布 |
| Fig. 3 | `fig3_rating_confound.png` | レート交絡（既存） |
| Fig. 4 | `fig_architecture.png` | 推論経路 + fail-closed 3層 |
| Fig. 5 | `fig_pair_coverage.png` | 最終ペアの被覆表 |
| Fig. 6 | `fig_internal_vs_ladder.png` | 内部予測 vs ラダー実測 vs field benchmark |
| Fig. 7 | `fig_ablation_two_stage.png` | 二段アブレーション |
| Fig. 8 | `fig_league_decay.png` | リーグ3ラウンドの減衰曲線 |
| Fig. 9 | `fig_curriculum_gates.png` | カリキュラム before/after と事前登録ゲート |
| Fig. 10 | `fig_field_matrix_final.png`（＋ `_full`） | field benchmark 行列 |
| Fig. 11 | `fig_deck_anatomy.png` | 最終2デッキの解剖表（カード・枚数・役割。`docs/data/deck_roles.json`、役割はユーザー検証済み 9/7。下段は 2,000 試合の実測 `replay_behavior_*.json`） |
| Fig. 12 | `fig_portfolio_hedge.png` | 2枠の保険価値（反実仮想の再重み付け、`docs/data/portfolio_reweighting.json`） |
| Table 1 | `table_compute.png` | 計算環境・学習コスト |

タイトル: **Two slots, one design: insuring against a moving metagame**（ユーザー決定 9/5・センテンスケース）

Kaggle Writeup 画面の制約（9/7 ユーザーが実機確認）: Title 80字 / Subtitle 140字 / Track は単一 "Main Track" が自動選択。
現状 Title 57字・Subtitle は 162字だったので **133字に短縮**（"after the freeze" は §6 に残る）。語数は `count_words.py` の kaggle 値（空白区切り）で管理。Kaggle 画面の実測は当方比 −7語（9/7）。
添付予定: `docs/data/decklists/{kangaskhan_dh,hydrapple_k2}.{txt,csv}`（標準デッキリスト形式。運営 topic 738657 が CSV 添付を明示的に許可）

改訂ラウンド1（9/5・外部採点 85 点への対応）: LB=2枠の最大値なので最終 993.8 は dh 単体の値であることを §6 で明記し、
第2枠を「メタ転換への保険」として Fig. 12 で定量化。field ceiling → field benchmark。§7 にキーカードと役割、
多重検定の注意。絶対表現（meaningless / not the differentiator / cannot）を緩和。バグ列挙は3件に。

改訂ラウンド2（9/6・外部再採点 89 点への対応）: §1 冒頭を公式仕様「two active slots, scores the better one」に、
「約1秒の予算」を公式の「600s/ゲーム・1手制限なし」に、Fig. 12 を "reweighting stress test, not a historical backtest"
として本文・副題・ファイル名を統一、§7 末尾にカード調整（1–2 pt）とアーキ切替（19–33 pt）の接続文、§6 冒頭を
最終レーティング・順位から、z=2.2 は "suggestive rather than decisive"。Fig. 11 の Cornerstone 注記の誤りを訂正。

最終仕上げ（9/13・ユーザー指示: 論理性・データ正確性・平易性・評価指標）: 実データと矛盾した記述を修正（Lopunny は dh の最悪対面でも
k2 の最良対面でもない／Hydrapple の主砲は Syrup Storm 68%、Ogerpon 27%／Crustle が与ダメージ 65%、Mega Kangaskhan ex を記述／
「凍結2日前」→「凍結前の週」／Case 2 の 1.2→1.3 の並置を解消／n=94–142）、ガントレット用語（gauntlet gain/score）を field benchmark
から分離、§4 第4ガードと §6 ペア選定の接続を明示、専門語を平易化（non-stationary / modal / zero-shot / EMA / staircase 等）、
各図に Fig. 番号タグ。提出物一覧と手順は `docs/submission-package.md`。

---
---

**Title:** Two slots, one design: insuring against a moving metagame

**Subtitle:** Behaviour cloning, a self-play league with two jobs, and two agents built to cover each other's holes, checked on 2,000 ladder games.

---

## 1. The problem the ladder actually poses

**The ladder gives each team two active slots and scores the better one.** We therefore designed the two slots jointly, choosing agents that fail in different places.

**The metagame did not stand still.** Over 75 days the leading archetype changed six times (7-day average): Grimmsnarl fell from 64% of top-rated decks on July 28 to 1% on August 30, Dragapult rose from 9% to 32% in the 19 days around the freeze (Fig. 1). Anything tuned to a fixed opponent mix has a shelf life of days.

**In this period, the pilot — the agent playing the list — mattered more than the list.** Twenty teams played a Grimmsnarl list identical to the most common one for 100+ games in the same three days; their win rates spread from 43% to 57% (Fig. 2). So we spent our effort on the pilot and treated deck choice as a portfolio problem.

## 2. Why win rate cannot tell you what to play

Reading archetype win rates off the public episodes is the obvious way to pick decks, but rating-matched pairing makes them a biased measure of deck strength: at equilibrium every archetype tends to 50%, and a high figure mostly measures *recency of submission* — agents submitted within a day sat at 53.7%, those four to seven days old at 50.6% (Fig. 3).

Three of our own instruments lied too, each with *plausible* numbers. Loading two trained agents in one process made the second inherit the first's weights, so every model-vs-model win rate we had collected was self-play; one decision reversed when fixed. A release gate printed green while every forward pass failed and the agent was playing its rule-based fallback. And 36% of the network's parameters were dead: card identity arrived only through an ID embedding whose never-indexed rows form a natural control group, and real cards were indistinguishable from them. We rebuilt the input around static card attributes.

The habit that runs through this report: **verify the instrument before trusting the reading.**

## 3. The agent

**Architecture.** The action space is a variable-length list of legal options, so we use a pointer network: 56 board tokens (two self-attention layers), each legal option attending to that state (two cross-attention layers), and a shared head scoring every option (Fig. 4). One network for every decision type: 691,843 learned parameters plus 172,480 static card and attack attributes, **3.4 ms per decision** in real games, against a 600-second game budget with no per-move limit — never a bottleneck.

**Training.** Behaviour cloning on 31.96 million decisions from all teams across 47 days of public episodes (1.5 GPU-hours; 73.4% agreement with held-out later games), then per-deck fine-tuning on that archetype's strongest pilots, then PPO in the league of Section 4 (Table 1: about 100 GPU-hours in all).

**Stability by construction.** Our one hard rule (take a proven lethal) is a mask applied *before* sampling, so the learner optimises exactly the policy that plays. Inference is fail-closed in three layers — model, rule fallback, legal random — behind an export gate that refuses weights unless the PyTorch and NumPy paths agree (max logit difference 7.2e-7, identical argmax). A per-turn selection cap ended the infinite mirror game that cost one early submission; none has errored since.

## 4. The league: two jobs, not one

Self-play alone teaches an agent to beat itself, so we gave the league two explicit jobs with separate machinery.

*Job 1 — breadth.* Each iteration drew opponents from three buckets: 50% the current policy, 30% a pool of twelve past snapshots, 20% frozen non-learning agents (cloned teacher, greedy baselines, hand-written playbooks). Decks came 55% in proportion to the *measured* metagame share, 25% from decks we had measured ourselves losing to, 20% uniformly from the whole library. That uniform fifth is the only evidence that a gain is not an artifact of one matchup.

*Job 2 — targeted repair.* Inside a bucket, opponent *i* is drawn with probability proportional to max(floor, 1 − recent win rate against *i*)². Opponents we lose to attract games, solved ones decay out, and the floor keeps them in rotation so regression is detected, not forgotten. This made curricula cheap: we weighted progressively stronger pilots for one archetype and the sampler allocated the games.

**Why we trust what came out.** Four guards, one per failure mode: a KL anchor to the cloned policy (target 0.03, adaptive β) against self-play-only exploits; an hourly gauntlet on a fixed field with rollback, shipping the best checkpoint, not the last; an evaluator running the *exported* NumPy policy through the submission path, so we measure what plays; and gates written *before* each run, so shipping was decided before the numbers.

**What it bought.** Across eight archetypes the returns decayed cleanly: median gauntlet gain +10.6 points for a first PPO round, +7.3 for the second, +4.5 for the third (Fig. 8).

## 5. Weak matchups: the procedure, and what happened

A losing matchup is one of two problems with opposite treatments; we told them apart by measurement, not judgement.

**The instrument: a field benchmark.** For any matchup, the public episodes give the win rate *other teams' agents* achieve with that archetype against that opponent (Fig. 10). It carries the confounds of Section 2 — a benchmark, not a ceiling — but it gives a decision rule. Far below it, the deck can do it and our policy cannot: a policy problem. *At* it, more training has low expected value: we treat the weakness as structural unless shown otherwise and hand it to the portfolio.

**Case 1 — policy problem, repaired.** Our Hydrapple agent won 4.0% against Kangaskhan; other teams' Hydrapple agents won about half. We seeded the policy from the games those teams *won* (cloning them before PPO) and ran two curriculum rounds: 4.0% → 18.7% → 29.3% against our Kangaskhan proxy, past a 25% gate written in advance, with no regression in nine matchups and a gauntlet score of 64.4% → 72.6%, for 4.7 GPU-hours (Fig. 9).

**Case 2 — structural, and we stopped.** Our Ogerpon agent won 2.2% against Kangaskhan under a benchmark of 22%; four methods (three league rounds, a seeded curriculum) left it at 1.3%, and we closed it as structural. Against Lopunny it won 16.7% under a benchmark of 15.2% (n = 877) — already *at* it — so we skipped training and added the archetype whose benchmark there is 74.5%.

**Case 3 — the fix that was already in the other slot.** On the deck side, five tech packages against the decks Ogerpon loses to, 64,000 games, policy frozen, all lost by 15.7 to 23.1 points. The winning lists showed why others win those matchups: 55% were Ogerpon–Meganium hybrids, and our second slot already *was* that hybrid.

**A negative result.** Seeding does not generalise: on an already-strong policy the same recipe cost 19 to 23 points, a lighter dose still 5; it works only where the win rate is near zero.

**Did the repairs transfer?** Where the sparring partners were diverse, yes: the Kangaskhan agent's Dragapult stage (a greedy bot, a hand-written playbook, our trained agent) was predicted at 71.8% and won 78% (n = 135). Where they were three versions of our own agent, no: its Hydrapple stage was predicted at 82.8% and won 47% (n = 57). Our hypothesis: transfer tracks the diversity of the sparring partners, not the win rate against them.

## 6. The portfolio, and what the ladder said

**Final result: 993.8, rank 138 of 6,807.** That is the Kangaskhan agent's rating; the Hydrapple agent finished at 932.6. In the week before the freeze the metagame turned again (Dragapult 8% → 22%, Grimmsnarl halved), which favoured the Kangaskhan agent; we paired it with the Hydrapple agent because their holes did not overlap: Hydrapple lost to Kangaskhan, Kangaskhan lost to Lopunny, and each covered the other.

**What the second slot is for.** Two copies of one agent insure against rating noise; two different agents insure against the metagame moving. After six leader changes we bought the second kind, and it did not pay out — the score is one agent's. Fig. 12 shows what it insured against — a stress test, not a backtest: holding each agent's Aug 21–31 matchup rates fixed and reweighting by each day's archetype mix, Hydrapple would have been the better slot on 46 of 75 days (by nine points in late July), Kangaskhan by twelve in late August; the better of the two averages 59% against 56% alone.

**Stability, the final 2,000 ladder games.** Both agents won 56–57% of their final 1,000 games, with no errored game — a stability diagnostic, not the competition metric. Coverage held where designed — Lopunny 35% / **68%** (Kangaskhan / Hydrapple agent), Kangaskhan **55%** / 36%, Dragapult **78%** / 51% (n = 94–142 per cell; intervals in Fig. 5) — and failed where we had not predicted: Alakazam, 34% and 35% over 309 games. The field benchmark splits that hole: other Hydrapple agents win only 33.5% of the matchup, so there it is the deck; other Kangaskhan agents win 44.7%, so there the pilot is eleven points short.

**What our predictions were worth.** Seven of eleven internal predictions fell outside the ladder's 95% interval, by up to 35 points (Fig. 6), with a sign: too high where our sparring partner was weaker than the real population, too low where stronger. Internal evaluation measures the proxy, not the matchup; the two predictions checked against the field benchmark beforehand, Dragapult and Kangaskhan, held.

## 7. Deck

**The two lists** (Fig. 11). The Kangaskhan deck hides behind Crustle: no damage from attacks by Pokémon ex, kept alive by Jumbo Ice Cream and Hero's Cape, and itself the main attacker (65% of the deck's damage; Mega Kangaskhan ex and Cornerstone Mask Ogerpon ex the rest), with four Boss's Orders to pick targets. The Hydrapple deck is a hybrid of the kind Case 3 found: Meganium doubles every Basic Grass Energy and Hydrapple ex attaches one from hand each turn; its Syrup Storm, 30 plus 30 per Grass Energy on our side, dealt 68% of the deck's damage, Teal Mask Ogerpon ex 27%.

**Did they play as designed?** Re-parsing the 2,000 replays: with Crustle Active, attacks by Pokémon ex with no Ability did zero damage in 96% of 1,028 attempts, non-ex attackers in 5% of 941, and Cornerstone, the second wall, stopped 80% of 970 attacks by Ability holders. Mega Lopunny ex's attack ignores effects on the defender, which is why we won 35% there while the second slot won 68% (Fig. 11).

**How we tested changes.** We screened 20 single-card changes to the most common Grimmsnarl and TR Mewtwo lists, 10,000 games each with the policy frozen, then fine-tuned each survivor and re-measured it on 20,000 fresh games not used for selection, separating "the card is weak" from "the policy is unfamiliar with it" (Fig. 7). Two changes survived: a third Boss's Orders in place of Tool Scrapper (+1.8 points, z = 3.8) and a third Mewtwo ex (+1.1, z = 2.2 on the confirmation set — suggestive, not decisive). The second stage mattered: a strong team's list went from −6.1 before fine-tuning to −3.9 after, another change from +1.4 to −0.2. Hence conservative lists: a confirmed card change was worth one to two points, an archetype switch moved single matchups by 19–33 points (Fig. 5). Both final lists are the most common top-rated list of their archetype when adopted, unchanged; the effort went into the pair.

## 8. What we would do differently

Build the field benchmark on day one and diagnose every matchup before training. Never trust an internal evaluation without a ladder-quality sparring partner per archetype from a *different* policy family. Keep the control-group habit: we found four broken instruments (three above, plus a gauntlet race) and assume there is a fifth.

---
---

## メモ（提出前に対応）

- [x] タイトル決定（9/5）
- [x] デッキリスト → 標準形式のテキスト/CSV を添付（`docs/data/decklists/`、運営 topic 738657）。ビジュアライザのスクショは任意
- [x] `docs/report-numbers.md` 台帳と本文の数値を突き合わせ（9/13 最終。実データと矛盾した箇所を修正）
- [ ] Kaggle に貼る際、見出しの番号・太字・イタリックが Writeup エディタで崩れないか確認（ユーザー・提出時。手順は `docs/submission-package.md`）
