#!/usr/bin/env python
"""リーダーボード上位チームのデッキ構成と戦術をまとめた Markdown レポートを生成する。

データソース（日付つき）:
- リーダーボード: `kaggle competitions leaderboard --show -v`（実行時点のスナップショット）
- デッキ/戦績: Daily Top Episodes の全エピソード
  （steps[0][0].visualize[0].action に両者の完全60枚、info.TeamNames にチーム名）

使い方:
    poetry run python scripts/report_top_teams.py \
        --episodes-dir data/episodes/2026-07-13 --top 20 \
        --out docs/top20-decks-2026-07-14.md
"""

from __future__ import annotations

import argparse
import csv
import glob
import io
import json
import os
import statistics
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401
from _bootstrap import REPO_ROOT

from ptcg import engine
from ptcg.cards import id_to_name
from ptcg.eval.harness import wilson_interval

# アーキタイプ別の戦術メモ（docs/meta-and-strategy.md の分析を要約したもの）
TACTICS = {
    "Alakazam": "単プライズ・トゥールボックス。エネ1個の低コスト打点（手札枚数/相手エネ依存）を軸に、"
    "高ドローと妨害（ハンマー/ボス）で耐えながら1対1交換を続け、サイドレースで上回る。",
    "Mega Kangaskhan ex": "300HP＋回復の耐久ビート。無色コストで事故が少なく、イワパレス壁（ex技無効）を"
    "添えて長期戦へ持ち込み、消耗戦で勝つ。速攻レースが弱点。",
    "Marnie's Grimmsnarl ex": "進化時に基本悪エネを大量サーチする爆発的加速から、180＋ベンチ30の"
    "ミッドレンジビート。立ち上がり明けの中盤が最強。",
    "Mega Starmie ex": "最速レース特化。低コストの120＋ベンチ50や効果無視の210、スナイプで"
    "中央値100stepで決着させる。大型たね主体のデッキを轢く。1プライズ/妨害系が苦手。",
    "Team Rocket's Mewtwo ex": "ロケット団ポケモンを並べるエンジン型。展開が整うと160＋捨てエネ比例の"
    "大打点が続く。上振れ最強だが立ち上げが重く、速攻に弱い。",
    "Dragapult ex": "本体200＋ベンチへのダメカンばら撒きスプレッド。ダメカン移動と併せ複数取りを狙う。",
    "Cynthia's Garchomp ex": "シンオウ軸のミッドレンジビート。",
    "Cornerstone Mask Ogerpon ex": "闘単の耐久アタッカー軸。",
    "Mega Lopunny ex": "メガシンカの高速アタッカー軸。",
    "Dudunsparce": "非ルール・ドロー特化のトゥールボックス。",
}


def leaderboard_top(n: int) -> list[dict]:
    env = dict(os.environ)
    env.setdefault("KAGGLE_CONFIG_DIR", str(REPO_ROOT / ".kaggle"))
    out = subprocess.run(
        ["poetry", "run", "kaggle", "competitions", "leaderboard",
         "-c", "pokemon-tcg-ai-battle", "--show", "-v"],
        cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, check=True,
    ).stdout
    # 先頭に "Next Page Token = ..." が付くことがあるので CSV ヘッダ行から読む
    lines = out.splitlines()
    start = next(i for i, l in enumerate(lines) if l.lower().startswith("teamid"))
    rows = list(csv.DictReader(io.StringIO("\n".join(lines[start:]))))
    return rows[:n]


def ace_label(deck, cd):
    counts = Counter(deck)
    best = None
    for cid, cnt in counts.items():
        c = cd.get(cid)
        if cnt < 2 or c is None or getattr(c, "cardType", None) != 0:
            continue
        key = (c.hp or 0, bool(c.megaEx), bool(c.stage2), bool(c.stage1), cid)
        if best is None or key > best[0]:
            best = (key, cid)
    return cd[best[1]].name if best else "no_ace"


def deck_summary(deck, cd, names, max_items=9) -> str:
    """ポケモンライン＋主要トレーナーズを『枚数x名前』で要約。"""
    counts = Counter(deck)
    pokes, trainers, energies = [], [], []
    for cid, n in counts.most_common():
        c = cd.get(cid)
        label = f"{n}×{names.get(cid, cid)}"
        if c is not None and c.cardType == 0:
            pokes.append((n, c.hp or 0, label))
        elif c is not None and c.cardType in (5, 6):
            energies.append(label)
        else:
            trainers.append((n, label))
    pokes.sort(key=lambda t: (-t[0], -t[1]))
    trainers.sort(key=lambda t: -t[0])
    parts = [l for _, _, l in pokes]
    parts += [l for _, l in trainers[: max(0, max_items - len(parts))]]
    extra = len(counts) - len(parts) - len(energies)
    tail = f" ／エネ: {', '.join(energies)}" if energies else ""
    more = f" …他{extra}種" if extra > 0 else ""
    return "、".join(parts) + more + tail


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes-dir", required=True)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    lb = leaderboard_top(args.top)
    now = datetime.now(timezone.utc)
    ep_date = Path(args.episodes_dir).name

    cd = {c.cardId: c for c in engine.engine_card_data()}
    names = id_to_name("EN")

    files = sorted(glob.glob(f"{args.episodes_dir}/**/*.json", recursive=True))
    print(f"parsing {len(files)} episodes ...")
    # team -> deck(tuple) -> {games,wins,steps[]}
    team_decks: dict[str, dict] = defaultdict(lambda: defaultdict(lambda: {"g": 0, "w": 0, "steps": []}))
    for k, f in enumerate(files, 1):
        if k % 1000 == 0:
            print(f"  {k}/{len(files)}")
        try:
            d = json.load(open(f, encoding="utf-8"))
            tn = (d.get("info") or {}).get("TeamNames") or []
            action = d["steps"][0][0]["visualize"][0]["action"]
            rew = d.get("rewards") or []
            nsteps = len(d.get("steps") or [])
        except Exception:
            continue
        if len(tn) != 2 or len(action) != 2 or len(rew) != 2:
            continue
        if rew[0] is None or rew[1] is None or rew[0] == rew[1]:
            continue
        winner = 0 if rew[0] > rew[1] else 1
        for seat in (0, 1):
            deck = tuple(sorted(int(x) for x in action[seat]))
            if len(deck) != 60:
                continue
            rec = team_decks[tn[seat]][deck]
            rec["g"] += 1
            rec["w"] += int(seat == winner)
            rec["steps"].append(nsteps)

    # Markdown 生成
    out = []
    out.append(f"# 上位{args.top}チームのデッキ構成と戦術")
    out.append("")
    out.append(f"- リーダーボード: **{now:%Y-%m-%d %H:%M} UTC** 時点のスナップショット")
    out.append(f"- デッキ/戦績: Daily Top Episodes **{ep_date}**（全{len(files)}エピソードを解析。"
               "上位帯の対戦のみ収録のため、戦績はその範囲のもの）")
    out.append("- デッキはエピソードに埋め込まれた完全60枚から抽出。アーキタイプは"
               "『2枚以上採用の最高HPポケモン』をエースとするヒューリスティック。")
    out.append("")
    for i, row in enumerate(lb, 1):
        team = row.get("teamName") or row.get("TeamName") or "?"
        score = row.get("score") or row.get("Score") or "?"
        sub_date = (row.get("submissionDate") or "")[:10]
        out.append(f"## {i}位 {team} — {score}点")
        out.append(f"（最終提出: {sub_date}）")
        decks = team_decks.get(team)
        if not decks:
            out.append("")
            out.append(f"- {ep_date} のトップエピソードに出現なし（新規提出・デッキ変更直後など）。")
            out.append("")
            continue
        for j, (deck, rec) in enumerate(sorted(decks.items(), key=lambda kv: -kv[1]["g"]), 1):
            arch = ace_label(deck, cd)
            g, w = rec["g"], rec["w"]
            p, lo, hi = wilson_interval(w, g)
            med = int(statistics.median(rec["steps"])) if rec["steps"] else "-"
            out.append("")
            out.append(f"### デッキ{j}: {arch}")
            out.append(f"- 戦績（{ep_date}, 上位帯）: {w}勝{g - w}敗 / {g}戦 "
                       f"勝率 {p*100:.0f}% [{lo*100:.0f}–{hi*100:.0f}] ／ 試合長中央値 {med} step")
            out.append(f"- 構成: {deck_summary(deck, cd, names)}")
            tactic = TACTICS.get(arch)
            if tactic:
                out.append(f"- 戦術: {tactic}")
        out.append("")
    Path(args.out).write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
