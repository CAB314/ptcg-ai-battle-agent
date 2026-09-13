#!/usr/bin/env python
"""ストラテジー部門レポート用の図表を生成する（docs/figures/*.png）。

Kaggle Writeup は本文2,000語上限だが Media Gallery の図表は語数に含まれない。
つまり図表に寄せるのが構造的に正しい。Report Score(10%) の評価項目が
「図表の効果的な使用」であり、Model Score(70%) の「技術的妥当性」「特定の対面・
状況的優位への依存を避けているか」にも図表で答える。

ここで生成する図は**すべて自前の実測データのプロット**。カード画像は運営回答（2026-08-24/09-01）で
公式ビジュアライザのスクショに限り使用可となったが、それはデッキリスト図（別途）だけに使う。

ラベルは英語（審査は国際パネル想定 + CJKフォント依存を避ける）。

使い方: poetry run python scripts/make_report_figures.py [--only fig2]
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import _bootstrap  # noqa: F401
from _bootstrap import REPO_ROOT

OUT = REPO_ROOT / "docs" / "figures"

# --- 検証済みパレット（dataviz skill の reference instance / light mode）---
# validate_palette.js: ALL CHECKS PASS（コントラストWARN=直接ラベルで relief）
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
GOOD = "#0ca30c"
CRIT = "#d03b3b"


def _style(ax, title=None, sub=None, ylab=None, xlab=None):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelsize=9, length=3, width=1.0)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    if title:
        # 副題の行数ぶんタイトルを持ち上げる（固定 pad だと2行副題と衝突する）
        nl = sub.count("\n") + 1 if sub else 0
        ax.set_title(title, color=INK, fontsize=13, fontweight="bold", loc="left",
                     pad=8 + 13.5 * nl)
    if sub:
        ax.text(0.0, 1.012, sub, transform=ax.transAxes, color=INK2, fontsize=9.5, va="bottom",
                linespacing=1.35)
    if ylab:
        ax.set_ylabel(ylab, color=INK2, fontsize=10)
    if xlab:
        ax.set_xlabel(xlab, color=INK2, fontsize=10)


# 本文の図番号 → ファイル名。Media Gallery ではキャプションが付かないので、画像自体の右上に番号を刻む
# （docs/report-draft.md ヘッダの表と一致させる。ここに無いファイルはギャラリー対象外で、タグを付けない）。
FIG_LABELS = {
    "fig1_meta_timeline": "Fig. 1", "fig_clone_spread": "Fig. 2", "fig3_rating_confound": "Fig. 3",
    "fig_architecture": "Fig. 4", "fig_pair_coverage": "Fig. 5", "fig_internal_vs_ladder": "Fig. 6",
    "fig_ablation_two_stage": "Fig. 7", "fig_league_decay": "Fig. 8", "fig_curriculum_gates": "Fig. 9",
    "fig_field_matrix_final": "Fig. 10", "fig_field_matrix_full": "Fig. 10 (full)",
    "fig_deck_anatomy": "Fig. 11", "fig_portfolio_hedge": "Fig. 12", "table_compute": "Table 1",
}


def _stamp(fig, name):
    """図の右下に本文の図番号を刻む（右上は2パネル図の右タイトルと衝突する。bbox_inches='tight' でも欠けない位置）。"""
    label = FIG_LABELS.get(name)
    if label:
        fig.text(0.995, 0.004, label, ha="right", va="bottom", color=MUTED, fontsize=9.5, fontweight="600")


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{name}.png"
    _stamp(fig, name)
    fig.savefig(p, dpi=200, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)
    print(f"[ok] {p}")


# ---------------------------------------------------------------- Fig 1
def fig1_meta_timeline():
    """メタシェア推移。非定常性＝静的な戦略が陳腐化することの一次証拠。"""
    from ptcg.ml.vocab import ACE_DISPLAY_NAMES

    rows = list(csv.DictReader(open(REPO_ROOT / "docs/data/meta_timeline.csv")))
    dates = [r["date"] for r in rows]
    cols = [c[:-6] for c in rows[0] if c.endswith("_share")]

    def raw(c):
        return np.array([float(r[f"{c}_share"] or 0) for r in rows]) * 100

    # CSV の列名は ace の表示名 or 生ID。生IDは正しい名前に直し、
    # 同一アーキタイプ（Alakazam = ace 66/245/743）は合算する。
    merged: dict[str, np.ndarray] = {}
    for c in cols:
        name = ACE_DISPLAY_NAMES.get(int(c), c) if c.isdigit() else c
        name = {"Alakazam(245)": "Alakazam", "Grimmsnarl ex": "Grimmsnarl ex"}.get(name, name)
        merged[name] = merged.get(name, np.zeros(len(rows))) + raw(c)

    keep = sorted(merged, key=lambda k: -merged[k].sum())[:7]
    keep.sort(key=lambda k: -merged[k][-1])          # 最終日の大きさ順＝凡例の並び
    ys = [merged[k] for k in keep]
    other = np.clip(100 - np.sum(ys, axis=0), 0, None)

    # 数値はデータから: 試合数（sides/2）、首位アーキの交代回数（7日移動平均で日次ノイズを除く）、
    # Grimmsnarl の最大シェアと最終シェア
    games = sum(int(r["sides"]) for r in rows) // 2
    k7 = np.ones(7) / 7
    smooth = {k: np.convolve(v, k7, mode="same") for k, v in merged.items()}
    leader = [max(smooth, key=lambda k: smooth[k][t]) for t in range(len(rows))]
    changes = sum(1 for t in range(1, len(rows)) if leader[t] != leader[t - 1])
    g = merged.get("Grimmsnarl ex")
    gi_peak = int(np.argmax(g)) if g is not None else None

    fig, ax = plt.subplots(figsize=(11, 5.4))
    x = np.arange(len(dates))
    labels = [k.replace(" ex", "") for k in keep]
    ax.stackplot(x, *ys, other, labels=labels + ["Other"],
                 colors=S[:len(keep)] + [GRID], edgecolor=SURFACE, linewidth=1.2, zorder=2)
    _style(ax, f"The metagame kept moving: the leading archetype changed {changes} times in {len(rows)} days",
           f"Daily archetype share among top-rated public episodes (n = {games:,} games, "
           f"{dates[0]} – {dates[-1]}). Leader = 7-day average. Stack order = share on the last day.",
           ylab="Share of decks (%)")
    ax.set_xlim(0, len(dates) - 1)
    ax.set_ylim(0, 100)
    step = max(1, len(dates) // 9)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([d[5:] for d in dates[::step]])
    # 最大の転換に注記（語数を使わずに主張を伝える）: Grimmsnarl のピーク → 最終日
    if gi_peak is not None and "Grimmsnarl ex" in keep:
        gi = keep.index("Grimmsnarl ex")
        base_peak = float(np.sum([ys[j][gi_peak] for j in range(gi)]))  # 積み上げの下端
        ax.annotate(f"Grimmsnarl {g[gi_peak]:.0f}% on {dates[gi_peak][5:]}\n→ {g[-1]:.0f}% on {dates[-1][5:]}",
                    xy=(gi_peak, base_peak + g[gi_peak] / 2), xytext=(gi_peak - 26, 90),
                    ha="left", va="top", color=INK, fontsize=10, fontweight="bold",
                    bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.85, pad=2.5),
                    arrowprops=dict(arrowstyle="->", color=INK2, lw=1.4))
    if "Dragapult ex" in keep:
        di = keep.index("Dragapult ex")
        d = merged["Dragapult ex"]
        t0 = next((t for t in range(len(rows) - 1, 0, -1) if d[t] < 10), None)
        if t0 is not None:
            base_last = float(np.sum([ys[j][-1] for j in range(di)]))
            ax.annotate(f"Dragapult {d[t0]:.0f}% → {d[-1]:.0f}%\nin {len(rows) - 1 - t0} days",
                        xy=(len(rows) - 1.3, base_last + d[-1] / 2), xytext=(len(rows) - 24, 62),
                        ha="left", va="center", color=INK, fontsize=10, fontweight="bold",
                        bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.85, pad=2.5),
                        arrowprops=dict(arrowstyle="->", color=INK2, lw=1.4))
    leg = ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False,
                    fontsize=9.5, labelcolor=INK2, handlelength=1.2, handleheight=1.0)
    leg.set_title("Archetype", prop={"size": 9.5, "weight": "600"})
    leg.get_title().set_color(INK)
    _save(fig, "fig1_meta_timeline")


# ---------------------------------------------------------------- Fig 2
def fig2_dead_embedding():
    """カードID埋め込みが未学習だった証拠。未使用ID行が完全な対照群になる。"""
    z = np.load(REPO_ROOT / "agents/grimmsnarl_ppo2/weights.npz")
    E = z["card_emb.weight"].astype(np.float64)
    # 実在カードID = 1..1267（エンジンの最大ID）／未参照ID = 1268..4095
    real = np.arange(1, 1268)
    ctrl = np.arange(1268, E.shape[0])
    nr = np.linalg.norm(E[real], axis=1)
    nc = np.linalg.norm(E[ctrl], axis=1)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6),
                                  gridspec_kw={"width_ratios": [1.45, 1]})
    bins = np.linspace(min(nr.min(), nc.min()), max(nr.max(), nc.max()), 46)
    ax.hist(nc, bins=bins, density=True, color=GRID, edgecolor=SURFACE, linewidth=0.6,
            label=f"Never-indexed IDs (control, n={len(ctrl)})", zorder=2)
    ax.hist(nr, bins=bins, density=True, histtype="step", color=S[0], linewidth=2.0,
            label=f"Real card IDs (n={len(real)})", zorder=3)
    _style(ax, "Card-ID embeddings never learned",
           "Row-norm distribution. Rows for IDs that no game can index keep their\ninitialisation — a perfect control group. The two are indistinguishable.",
           ylab="Density", xlab="L2 norm of embedding row")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.30)   # 凡例が棒に重ならないよう上に余白
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left")

    # 4検定は単位が違うので棒にしない（比率の棒＋生の値ラベルは読み違いを招く）。表で出す。
    rows = [
        ("Test", "Real", "Control", ""),
        ("Row-norm median", "5.965", "5.946", "no difference"),
        ("Cosine-sim. SD", "0.1249", "0.1251", "no difference"),
        ("Same-type pairs", "+0.9", "0", "z, not sig."),
        ("Same-ex pairs", "-0.6", "0", "z, not sig."),
        ("HP decode R²", "0.077", "0.040", "at null floor"),
    ]
    ax2.axis("off")
    ax2.set_title("Four independent tests, one verdict", color=INK, fontsize=13,
                  fontweight="bold", loc="left", pad=36)
    ax2.text(0.0, 1.012, "Real card rows carry no more information than rows the\ngame can never look up.",
             transform=ax2.transAxes, color=INK2, fontsize=9.5, va="bottom", linespacing=1.35)
    for r, (lab, a, b, note) in enumerate(rows):
        y = 0.86 - r * 0.148
        head = r == 0
        ax2.text(0.0, y, lab, transform=ax2.transAxes, fontsize=9.5, va="center",
                 color=INK if head else INK2, fontweight="bold" if head else "normal")
        for x, cell in ((0.52, a), (0.71, b)):
            ax2.text(x, y, cell, transform=ax2.transAxes, fontsize=9.5, va="center",
                     ha="right", color=INK, fontweight="bold" if head else "normal",
                     family=None if head else "monospace")
        ax2.text(0.75, y, note, transform=ax2.transAxes, fontsize=9, va="center",
                 color=MUTED)
        if head:
            ax2.plot([0, 1], [y - 0.058] * 2, transform=ax2.transAxes,
                     color=AXIS, linewidth=1.0)
    _save(fig, "fig2_dead_embedding")


# ---------------------------------------------------------------- Fig 3
def fig3_rating_confound():
    """勝率はレート帯マッチングで50%に収束する＝デッキ強度の指標にならない。"""
    buckets = ["0–1 days", "2–3 days", "4–7 days", "8+ days"]
    wr = [53.7, 51.4, 50.6, 52.4]
    n = [495733, 154944, 53943, 24201]
    # 「50%からのずれ」なので棒(面積符号化)は使わない。ゼロ基点でない棒は差を誇張する。
    # 基準線からのステム + ドット（位置符号化）にする。
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    x = np.arange(len(buckets))
    ax.axhline(50, color=AXIS, linewidth=1.4, zorder=2)
    ax.vlines(x, 50, wr, color=S[0], linewidth=2.0, zorder=3)
    ax.scatter(x, wr, s=132, color=S[0], zorder=4, edgecolor=SURFACE, linewidth=1.6)
    for xi, (w, ni) in enumerate(zip(wr, n)):
        ax.text(xi, w + 0.22, f"{w}%", ha="center", fontsize=11, color=INK, fontweight="bold")
        ax.text(xi, 49.62, f"n = {ni:,}", ha="center", fontsize=8.5, color=MUTED)
    _style(ax, "Field win rate measures recency, not deck strength",
           "The ladder pairs agents of similar rating, so win rate returns to 50%.\nOnly freshly submitted agents look strong — they are still climbing.",
           ylab="Win rate (%)", xlab="Days since the team's last submission")
    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.set_xlim(-0.55, len(buckets) - 0.45)
    ax.set_ylim(49.4, 54.5)
    ax.set_yticks([50, 51, 52, 53, 54])
    ax.text(-0.5, 50.1, "50% = matchmaking equilibrium", color=INK2, fontsize=9.5,
            ha="left", va="bottom")
    _save(fig, "fig3_rating_confound")


# ---------------------------------------------------------------- Fig 4
def fig4_matchup_profile():
    """実ラダーでの自エージェントの対面プロファイル。

    審査項目「特定の対面・状況的優位への依存を避けているか」に正面から答える図。
    自分の弱点を隠さずに出す。数値は公式CLIで取得した実リプレイ145戦の集計。
    """
    # (対面, 勝, 総) — trmewtwo_ppo2 / 145戦 / スコア931.7（収束済み・勝率49.0%）
    data = [
        ("Alakazam", 29, 34), ("M Kangaskhan", 9, 14), ("Dragapult", 2, 5),
        ("Teal Ogerpon", 1, 3), ("M Starmie", 1, 3), ("Mewtwo (mirror)", 2, 7),
        ("Garchomp", 2, 8), ("Grimmsnarl", 19, 58), ("M Lopunny", 0, 4),
    ]
    data.sort(key=lambda d: -(d[1] / d[2]))
    labs = [f"{d[0]}  (n={d[2]})" for d in data]
    wr = np.array([d[1] / d[2] * 100 for d in data])
    y = np.arange(len(data))[::-1]

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    # 50%を基準に、上回る/下回るで色を分ける（発散＝極性の符号化）
    cols = [S[0] if v >= 50 else S[7] for v in wr]
    ax.hlines(y, 50, wr, color=cols, linewidth=2.0, zorder=3)
    ax.scatter(wr, y, s=[26 + 2.3 * d[2] for d in data], color=cols, zorder=4,
               edgecolor=SURFACE, linewidth=1.6)
    ax.axvline(50, color=AXIS, linewidth=1.4, zorder=2)
    for yi, v in zip(y, wr):
        off = 2.6 if v >= 50 else -2.6
        ax.text(v + off, yi, f"{v:.0f}%", va="center",
                ha="left" if v >= 50 else "right", fontsize=9.5, color=INK)
    _style(ax, "Our best agent's real matchup profile — including where it loses",
           "Win rate per opponent archetype over 145 live ladder games (score 931.7).\nMarker area is games played; the vertical line is even (50%).",
           xlab="Win rate (%)")
    ax.set_yticks(y)
    ax.set_yticklabels(labs, fontsize=9.5)
    ax.tick_params(axis="y", labelcolor=INK2, length=0)
    ax.set_xlim(-8, 108)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.grid(True, axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.grid(False, axis="y")
    ax.annotate("40% of all games are the\nmatchup we lose most",
                xy=(33, y[labs.index("Grimmsnarl  (n=58)")]), xytext=(62, 1.4),
                fontsize=9.5, color=INK, fontweight="bold", ha="left",
                arrowprops=dict(arrowstyle="->", color=INK2, lw=1.3))
    _save(fig, "fig4_matchup_profile")


# ================================================================ 2026-09 追加分
# 元データはすべて docs/data/*.json（experiments-log からの転記は行番号つき）。
# 図の番号でなく内容でファイル名を付ける（plan-writeup の F番号との対応は計画書に記載）。

def _load_json(rel: str):
    return json.loads((REPO_ROOT / rel).read_text(encoding="utf-8"))


SHORT = {  # カード名 → 図ラベル
    "Marnie's Grimmsnarl ex": "Grimmsnarl", "Mega Kangaskhan ex": "Kangaskhan",
    "Mega Lopunny ex": "Lopunny", "Dragapult ex": "Dragapult", "Hydrapple ex": "Hydrapple",
    "Teal Mask Ogerpon ex": "Teal Ogerpon", "Mega Lucario ex": "Lucario", "Alakazam": "Alakazam",
    "Team Rocket's Mewtwo ex": "TR Mewtwo", "Cynthia's Garchomp ex": "Garchomp",
    "Mega Starmie ex": "Starmie", "Archaludon ex": "Archaludon", "Hop’s Snorlax": "Hop's Snorlax",
    "Thwackey": "Thwackey", "Seaking": "Seaking", "Arboliva ex": "Arboliva", "Sylveon": "Sylveon",
    "Latias ex": "Latias", "Crustle": "Crustle",
}
AGENT_ARCH = {"kangaskhan_dh": "Mega Kangaskhan ex", "hydrapple_k2": "Hydrapple ex"}
AGENT_COL = {"kangaskhan_dh": S[0], "hydrapple_k2": S[1]}
AGENT_LABEL = {"kangaskhan_dh": "kangaskhan_dh (rating 993.8)", "hydrapple_k2": "hydrapple_k2 (rating 932.6)"}


def _short(n: str) -> str:
    return SHORT.get(n, n.replace(" ex", ""))


def _hgrid_off(ax):
    ax.grid(False)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8, zorder=0)


# ---------------------------------------------------------------- pair coverage
def fig_pair_coverage():
    """最終ペアの被覆表: 対面ごとに2体の実戦勝率。片方が50%超なら被覆、両方未満なら未被覆。"""
    d = _load_json("docs/data/final_pair_matchups.json")
    A = d["agents"]
    per = {ag: {m["opponent"]: m for m in A[ag]["matchups"]} for ag in AGENT_ARCH}
    enc = lambda o: sum(per[ag].get(o, {}).get("n", 0) for ag in per)  # noqa: E731
    opps = sorted({o for ag in per for o in per[ag]}, key=lambda o: -enc(o))
    opps = [o for o in opps if enc(o) >= 60][:9]
    y = np.arange(len(opps))[::-1]

    fig, ax = plt.subplots(figsize=(10, 5.8))
    _style(ax, "Two slots, one design: each agent covers the other's losing matchups",
           "Real-ladder win rate by opponent archetype, final 10 days (Aug 21–31, 2026; 1,000 games per agent; "
           "95% Wilson CI).\nRows ordered by encounters. Shaded row: neither agent above 50% — the uncovered hole.",
           xlab="Win rate (%)")
    _hgrid_off(ax)
    ax.axvline(50, color=AXIS, lw=1.2, zorder=1)
    ax.text(50.6, len(opps) - 0.35, "50%", color=MUTED, fontsize=8.5, va="top")
    for i, o in enumerate(opps):
        vals = [per[ag][o]["wr"] for ag in per if o in per[ag]]
        if max(vals) < 50:
            ax.axhspan(y[i] - 0.46, y[i] + 0.46, color="#fbeaea", zorder=0)
            ax.text(99, y[i], "uncovered", color=CRIT, fontsize=9.5, va="center", ha="right", fontweight="600")
        for ag, dy in (("kangaskhan_dh", +0.17), ("hydrapple_k2", -0.17)):
            m = per[ag].get(o)
            if not m:
                continue
            lo, hi = m["ci95"]
            ax.plot([lo, hi], [y[i] + dy] * 2, color=AGENT_COL[ag], lw=1.4, alpha=0.5, zorder=2)
            ax.scatter([m["wr"]], [y[i] + dy], s=70, color=AGENT_COL[ag], edgecolor=SURFACE,
                       linewidth=1.5, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{_short(o)}  (n={enc(o)})" for o in opps], fontsize=10, color=INK)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, len(opps) - 0.4)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    hs = [plt.Line2D([], [], marker="o", ls="", ms=8, color=AGENT_COL[ag], mec=SURFACE, label=AGENT_LABEL[ag])
          for ag in AGENT_ARCH]
    ax.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2, frameon=False,
              fontsize=9.5, labelcolor=INK2)
    _save(fig, "fig_pair_coverage")


# ---------------------------------------------------------------- curriculum / gates
def fig_curriculum_gates():
    """(a) Hydrapple 対 Kangaskhan の穴埋め（3段＋事前登録ゲート＋実ラダー）
       (b) kangaskhan_dh の4ゲート（before→after ダンベル＋ゲート位置＋実ラダー）"""
    c = _load_json("docs/data/curriculum_results.json")
    fm = _load_json("docs/data/field_matchup_matrix_final-pair-window.json")
    cells = {(r["my"], r["opp"]): r for r in fm["cells"]}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.4), gridspec_kw={"width_ratios": [1, 1.3], "wspace": 0.32})

    # (a)
    h = c["hydrapple_vs_kangaskhan"]
    xs = np.arange(len(h["stages"]))
    ys = [s["vs_target"] for s in h["stages"]]
    gate = h["preregistered_gate_vs_target"]
    _style(a1, "One losing matchup, repaired in 4.7 GPU-hours",
           "Hydrapple vs Kangaskhan: win rate vs our own Kangaskhan\n"
           "proxy after each stage. Hollow = real ladder (Aug 21–31).", ylab="Win rate (%)")
    a1.plot(xs, ys, color=S[1], lw=2.2, marker="o", ms=9, mec=SURFACE, mew=1.5, zorder=3)
    for x_, y_ in zip(xs, ys):
        a1.text(x_, y_ + 2.2, f"{y_:.1f}", ha="center", color=INK, fontsize=10, fontweight="600")
    a1.axhline(gate, color=INK2, lw=1.1, ls=(0, (4, 3)), zorder=2)
    a1.text(-0.35, gate + 0.8, f"pre-registered gate {gate:.0f}%", color=INK2, fontsize=9, va="bottom")
    fc = cells.get(("Hydrapple ex", "Mega Kangaskhan ex"))
    if fc:
        a1.axhline(fc["wr"], color=AXIS, lw=1.1, zorder=1)
        a1.text(-0.35, fc["wr"] + 0.8, f"field benchmark {fc['wr']:.0f}% (other teams, n={fc['n']:,})",
                color=MUTED, fontsize=9, va="bottom")
    lad = h["ladder_actual"]
    xl = len(xs)
    a1.plot([xl] * 2, lad["ci95"], color=S[1], lw=1.4, alpha=0.6, zorder=2)
    a1.scatter([xl], [lad["wr"]], s=95, facecolor=SURFACE, edgecolor=S[1], linewidth=2.2, zorder=3)
    a1.text(xl, lad["wr"] + 2.2 + (lad["ci95"][1] - lad["wr"]), f"{lad['wr']:.1f}", ha="center", color=INK,
            fontsize=10, fontweight="600")
    a1.set_xticks(list(xs) + [xl])
    a1.set_xticklabels(["r3\n(before)", "k\nBC seed +\ncurriculum 1", "k2\ncurriculum 2", f"real ladder\n(n={lad['n']})"],
                       fontsize=9, color=INK2)
    a1.set_xlim(-0.45, xl + 0.45)
    a1.set_ylim(0, 62)

    # (b)
    dh = c["kangaskhan_dh"]
    rows = [r for r in dh["before_after"] if r["before"] is not None]
    y = np.arange(len(rows))[::-1]
    _style(a2, "kangaskhan_dh: four gates written before the run — all passed",
           "Win rate vs proxy opponents before → after curriculum PPO (812 iterations, 2.3 GPU-h).\n"
           "Orange tick = pre-registered gate; hollow = real ladder (Aug 21–31, 95% CI).", xlab="Win rate (%)")
    _hgrid_off(a2)
    for i, r in enumerate(rows):
        a2.plot([r["before"], r["after"]], [y[i]] * 2, color=GRID, lw=3.2, zorder=1, solid_capstyle="round")
        a2.scatter([r["before"]], [y[i]], s=60, color=MUTED, edgecolor=SURFACE, linewidth=1.2, zorder=2)
        a2.scatter([r["after"]], [y[i]], s=80, color=S[0], edgecolor=SURFACE, linewidth=1.5, zorder=3)
        a2.plot([r["gate"]] * 2, [y[i] - 0.3, y[i] + 0.3], color=S[1], lw=2.4, zorder=2)
        la = dh["ladder_actual"].get(r["matchup"])
        if la:
            a2.scatter([la["wr"]], [y[i]], s=95, facecolor=SURFACE, edgecolor=S[0], linewidth=2.0, zorder=3)
            a2.text(la["wr"], y[i] + 0.36, f"ladder {la['wr']:.0f}", ha="center", color=INK2, fontsize=8.5)
        a2.text(r["after"], y[i] - 0.42, f"{r['before']:.1f} → {r['after']:.1f}", ha="center", color=INK, fontsize=9)
    a2.set_yticks(y)
    a2.set_yticklabels([f"vs {_short(r['matchup'])}" for r in rows], fontsize=10, color=INK)
    a2.tick_params(axis="y", length=0)
    a2.spines["left"].set_visible(False)
    a2.set_xlim(0, 100)
    a2.set_ylim(-0.8, len(rows) - 0.3)
    hs = [plt.Line2D([], [], marker="o", ls="", ms=7, color=MUTED, mec=SURFACE, label="before"),
          plt.Line2D([], [], marker="o", ls="", ms=8, color=S[0], mec=SURFACE, label="after (internal)"),
          plt.Line2D([], [], marker="|", ls="", ms=12, mew=2.4, color=S[1], label="pre-registered gate"),
          plt.Line2D([], [], marker="o", ls="", ms=9, mfc=SURFACE, mec=S[0], mew=2, label="real ladder")]
    a2.legend(handles=hs, loc="lower right", frameon=False, fontsize=9, labelcolor=INK2, ncol=2)
    _save(fig, "fig_curriculum_gates")


# ---------------------------------------------------------------- internal vs ladder vs field
def fig_internal_vs_ladder():
    """対面ごとに 内部予測 → ラダー実測（CI）と field benchmark。内部評価は自作プロキシを測っていた、の一次証拠。"""
    d = _load_json("docs/data/final_pair_matchups.json")
    fm = _load_json("docs/data/field_matchup_matrix_final-pair-window.json")
    cells = {(r["my"], r["opp"]): r for r in fm["cells"]}
    rows = []
    for ag, A in d["agents"].items():
        for m in A["matchups"]:
            if m["n"] < 30:
                continue
            f = cells.get((AGENT_ARCH[ag], m["opponent"]))
            rows.append({"agent": ag, "opp": m["opponent"], "wr": m["wr"], "ci": m["ci95"], "n": m["n"],
                         "internal": m.get("internal"), "field": f["wr"] if f and f["n"] >= 30 else None})
    # 内部予測あり → 誤差の大きい順、その後に予測なし（n順）
    rows.sort(key=lambda r: (r["internal"] is None, -(abs(r["wr"] - r["internal"]) if r["internal"] is not None else r["n"])))
    y = np.arange(len(rows))[::-1]

    fig, ax = plt.subplots(figsize=(11, 7.2))
    n_out = sum(1 for r in rows if r["internal"] is not None
                and not (r["ci"][0] <= r["internal"] <= r["ci"][1]))
    n_pred = sum(1 for r in rows if r["internal"] is not None)
    _style(ax, "Our internal evaluation measured our sparring partners, not the matchup",
           "Per matchup: internal prediction before freeze (gray) → real ladder result (colored, 95% CI, Aug 21–31, "
           f"1,000 games/agent).\nBlack tick = field benchmark: what other teams' agents achieved in the same matchup over "
           f"the same 10 days. {n_out} of {n_pred} predictions fall outside the ladder CI.", xlab="Win rate (%)")
    _hgrid_off(ax)
    ax.axvline(50, color=AXIS, lw=1.0, zorder=1)
    split_done = False
    for i, r in enumerate(rows):
        col = AGENT_COL[r["agent"]]
        if r["internal"] is not None:
            ax.plot([r["internal"], r["wr"]], [y[i]] * 2, color=GRID, lw=3.2, zorder=1, solid_capstyle="round")
            ax.scatter([r["internal"]], [y[i]], s=60, facecolor=SURFACE, edgecolor=MUTED, linewidth=1.6, zorder=2)
            err = r["wr"] - r["internal"]
            if abs(err) >= 10:
                ax.text((r["internal"] + r["wr"]) / 2, y[i] + 0.3, f"{err:+.0f}", ha="center", color=INK,
                        fontsize=9, fontweight="600")
        elif not split_done:
            ax.axhline(y[i] + 0.5, color=GRID, lw=1.0)
            ax.text(1, y[i] + 0.55, "no internal prediction recorded", color=MUTED, fontsize=8.5,
                    ha="left", va="bottom")
            split_done = True
        lo, hi = r["ci"]
        ax.plot([lo, hi], [y[i]] * 2, color=col, lw=1.5, alpha=0.55, zorder=2)
        ax.scatter([r["wr"]], [y[i]], s=78, color=col, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        if r["field"] is not None:
            ax.plot([r["field"]] * 2, [y[i] - 0.34, y[i] + 0.34], color=INK, lw=2.0, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['agent'].split('_')[0]} vs {_short(r['opp'])}  (n={r['n']})" for r in rows],
                       fontsize=9.5, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    hs = [plt.Line2D([], [], marker="o", ls="", ms=7, mfc=SURFACE, mec=MUTED, mew=1.6, label="internal prediction"),
          plt.Line2D([], [], marker="o", ls="", ms=8, color=S[0], mec=SURFACE, label="real ladder — kangaskhan_dh"),
          plt.Line2D([], [], marker="o", ls="", ms=8, color=S[1], mec=SURFACE, label="real ladder — hydrapple_k2"),
          plt.Line2D([], [], marker="|", ls="", ms=12, mew=2.0, color=INK, label="field benchmark (other teams)")]
    ax.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=4, frameon=False,
              fontsize=9, labelcolor=INK2)
    _save(fig, "fig_internal_vs_ladder")


# ---------------------------------------------------------------- league decay
def fig_league_decay():
    """PPO ラウンドごとの field 勝率の伸び（8アーキ）。灰=各アーキ、青=中央値、白抜き=ゲート不合格。"""
    L = _load_json("docs/data/league_rounds.json")
    pts: dict[str, list] = {}
    retries: list = []  # 同じ ppo_index の再挑戦（alakazam の v1 起点 2回目）は別扱い
    for rd in L["rounds"]:
        for r in rd["results"]:
            if r.get("delta") is None:
                continue
            seq = pts.setdefault(r["arch"], [])
            if any(ix == r["ppo_index"] for ix, _, _ in seq):
                retries.append((r["ppo_index"], r["delta"], bool(r["pass"]), r["arch"]))
                continue
            seq.append((r["ppo_index"], r["delta"], bool(r["pass"])))
    fig, ax = plt.subplots(figsize=(9, 5.6))
    _style(ax, "Returns from league self-play fall off fast",
           "Gauntlet win-rate gain (percentage points) per PPO round, 8 archetypes. Gray = one archetype "
           "(hollow = rejected by the pre-registered gate);\nblue = median. The gauntlet's opponent field differs by calendar round "
           "(v0 / v1 / current pool), so cross-round deltas are approximate.", ylab="Gauntlet win-rate gain (pp)")
    ax.axhline(0, color=AXIS, lw=1.0, zorder=1)
    # ラベル衝突回避: 同じ x に来る末端ラベルを最小間隔で押し広げる
    ends: dict[int, list] = {}
    for arch, p in pts.items():
        p.sort()
        xs = [a for a, _, _ in p]
        ys = [b for _, b, _ in p]
        ax.plot(xs, ys, color=MUTED, lw=1.2, alpha=0.7, zorder=2)
        for x_, y_, ok in p:
            ax.scatter([x_], [y_], s=42, facecolor=(MUTED if ok else SURFACE), edgecolor=MUTED, linewidth=1.3, zorder=3)
        ends.setdefault(xs[-1], []).append((ys[-1], arch))
    for x_, items in ends.items():
        items.sort()
        placed = []
        for y_, arch in items:
            yy = y_ if not placed else max(y_, placed[-1] + 0.95)
            placed.append(yy)
            ax.text(x_ + 0.07, yy, arch, fontsize=8.5, color=INK2, va="center")
    for x_, y_, ok, arch in retries:
        ax.scatter([x_ + 0.12], [y_], s=42, facecolor=(MUTED if ok else SURFACE), edgecolor=MUTED,
                   linewidth=1.3, zorder=3)
        ax.text(x_ + 0.19, y_, f"{arch} (retry)", fontsize=8.5, color=INK2, va="center")
    med = {i: float(np.median([d for a in pts for (ix, d, _) in pts[a] if ix == i])) for i in (1, 2, 3)}
    ax.plot(list(med), list(med.values()), color=S[0], lw=2.8, marker="o", ms=10, mec=SURFACE, mew=1.5, zorder=4)
    for i, v in med.items():  # 0 線の下は空いているので、そこに中央値を置く（他ラベルと衝突しない）
        ax.text(i, -1.15, f"median {v:+.1f}", ha="center", va="center", color=S[0], fontsize=10,
                fontweight="600", zorder=5)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["1st PPO round\n(from BC)", "2nd round", "3rd round"], color=INK2, fontsize=10)
    ax.set_xlim(0.6, 3.75)
    ax.set_ylim(-2, 19)
    _save(fig, "fig_league_decay")


# ---------------------------------------------------------------- field benchmark matrix
def _field_matrix(tag: str, name: str, top: int = 10, title_note: str = ""):
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    fm = _load_json(f"docs/data/field_matchup_matrix_{tag}.json")
    archs = list(fm["archetype_share"].keys())[:top]
    cells = {(r["my"], r["opp"]): r for r in fm["cells"]}
    M = np.full((top, top), np.nan)
    N = np.zeros((top, top), dtype=int)
    for i, a in enumerate(archs):
        for j, b in enumerate(archs):
            r = cells.get((a, b))
            if r and r["n"] >= 30 and i != j:
                M[i, j] = r["wr"]
                N[i, j] = r["n"]
    cmap = LinearSegmentedColormap.from_list("div_rb", ["#b93535", "#f0efec", "#1c5cab"])
    norm = TwoSlopeNorm(vmin=20, vcenter=50, vmax=80)

    fig, ax = plt.subplots(figsize=(10.5, 9.2))
    ax.set_facecolor(SURFACE)
    im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, norm=norm, zorder=1)
    for k in range(top + 1):  # 2px surface gap
        ax.axhline(k - 0.5, color=SURFACE, lw=2.5, zorder=2)
        ax.axvline(k - 0.5, color=SURFACE, lw=2.5, zorder=2)
    for i in range(top):
        for j in range(top):
            if i == j:
                ax.text(j, i, "mirror", ha="center", va="center", color=MUTED, fontsize=8)
                continue
            if np.isnan(M[i, j]):
                r = cells.get((archs[i], archs[j]))
                ax.text(j, i, f"n={r['n']}" if r else "–", ha="center", va="center", color=MUTED, fontsize=7.5)
                continue
            dark = M[i, j] <= 33 or M[i, j] >= 67
            ax.text(j, i - 0.12, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=11, fontweight="600",
                    color=("white" if dark else INK))
            ax.text(j, i + 0.27, f"n={N[i, j]:,}", ha="center", va="center", fontsize=7,
                    color=("#e8e8e8" if dark else INK2))
    labels = [_short(a) for a in archs]
    ax.set_xticks(range(top))
    ax.set_xticklabels(labels, rotation=35, ha="left", fontsize=9.5, color=INK)
    ax.xaxis.tick_top()
    ax.set_yticks(range(top))
    ax.set_yticklabels(labels, fontsize=9.5, color=INK)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    w = fm["window"]
    fig.suptitle("Field benchmark: what other teams' agents achieve in each matchup" + title_note,
                 x=0.02, y=0.995, ha="left", color=INK, fontsize=13, fontweight="bold")
    fig.text(0.02, 0.955,
             f"Row archetype's win rate (%) vs column archetype, top-rated public episodes {w['from']} – {w['to']} "
             f"({fm['total_sides']:,} deck-sides).\nTop {top} archetypes by share. Cells with n<30 show n only. "
             "Archetype = highest-HP Pokémon with ≥2 copies (heuristic; hybrids fall to one side).",
             color=INK2, fontsize=9.5, va="top", linespacing=1.35)
    ax.set_xlabel("opponent archetype", color=INK2, fontsize=10, labelpad=8)
    ax.xaxis.set_label_position("top")
    ax.set_ylabel("own archetype", color=INK2, fontsize=10)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.035, pad=0.03, aspect=45)
    cb.set_label("win rate (%) — red: loses, blue: wins", color=INK2, fontsize=9)
    cb.ax.tick_params(colors=MUTED, labelsize=8.5)
    cb.outline.set_visible(False)
    fig.subplots_adjust(top=0.80, bottom=0.08, left=0.14, right=0.98)
    p = OUT / f"{name}.png"
    OUT.mkdir(parents=True, exist_ok=True)
    _stamp(fig, name)
    fig.savefig(p, dpi=200, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)
    print(f"[ok] {p}")


# ---------------------------------------------------------------- clone spread
def fig_clone_spread():
    """同一60枚（最頻 Grimmsnarl リスト）を使うチームの勝率分布。リストは差別化要因でない、の一次証拠。"""
    d = _load_json("docs/data/clone_winrates.json")
    min_n = 100
    teams = [t for t in d["teams"] if t["n"] >= min_n]
    wr = np.array([100 * t["wr"] for t in teams])
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    _style(ax, f"Same 60 cards, {len(teams)} teams: win rates spread from {wr.min():.0f}% to {wr.max():.0f}%",
           f"Teams whose deck is card-for-card identical to the most common Grimmsnarl list, top-rated public episodes "
           f"{d['dates'][0][5:]} – {d['dates'][-1][5:]}, 2026.\nEach bar = number of teams in a 2-pt win-rate bin "
           f"(teams with ≥{min_n} games; {len(d['teams'])} clone teams in total, {d['totals']['clone_sides']:,} games).",
           ylab="Teams", xlab="Team win rate on the identical list (%)")
    bins = np.arange(np.floor(wr.min() / 2) * 2, np.ceil(wr.max() / 2) * 2 + 2, 2)
    ax.hist(wr, bins=bins, color=S[0], edgecolor=SURFACE, linewidth=1.5, zorder=2, rwidth=0.96)
    ax.axvline(50, color=AXIS, lw=1.0, zorder=1)
    med = float(np.median(wr))
    ax.axvline(med, color=INK2, lw=1.2, ls=(0, (4, 3)), zorder=3)
    ax.text(med + 0.15, ax.get_ylim()[1] * 0.985, f"median {med:.1f}%", color=INK2, fontsize=9.5, va="top",
            bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))
    q10, q90 = np.percentile(wr, [10, 90])
    ax.annotate(f"10th–90th percentile: {q10:.0f}% – {q90:.0f}%\n(same list, same 3 days)",
                xy=(q90, ax.get_ylim()[1] * 0.35), xytext=(q90 + 2, ax.get_ylim()[1] * 0.6),
                color=INK, fontsize=9.5, fontweight="600",
                arrowprops=dict(arrowstyle="->", color=INK2, lw=1.2))
    _save(fig, "fig_clone_spread")


# ---------------------------------------------------------------- deck ablation (two-stage)
def fig_ablation_two_stage():
    """(a) 第1段: 方策固定で20候補をスクリーニング（Δpp, 95%CI）
       (b) 第2段: 上位候補を微調整して再評価 — 改善は消え、退行は残る"""
    from collections import Counter

    g1 = _load_json("docs/data/ablation_results/ablation_stage1_cluster_results.json")
    m1 = _load_json("docs/data/ablation_results/ablation_mewtwo_results.json")
    # 候補コード（-TS_+Boss 等）は、基準デッキとの差分をカード名で表示する
    cnames = {}
    with open(REPO_ROOT / "data/EN_Card_Data.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                cnames[int(r["Card ID"])] = r["Card Name"].strip()
            except (KeyError, ValueError):
                continue
    short = {"Boss’s Orders": "Boss's Orders", "Team Rocket's Mewtwo ex": "TR Mewtwo ex",
             "Team Rocket's Murkrow": "TR Murkrow", "Team Rocket's Factory": "TR Factory",
             "Team Rocket's Honchkrow": "TR Honchkrow", "Lillie's Determination": "Lillie's Det.",
             "Pokégear 3.0": "Pokégear", "Basic {G} Energy": "Grass Energy", "Basic {W} Energy": "Water Energy"}

    def diff_label(defs, name):
        base = Counter(defs["base"]["deck"])
        c = next((x for x in defs["candidates"] if x["name"] == name), None)
        if c is None:
            return name
        cd = Counter(c["deck"])
        rem, add = base - cd, cd - base
        fmt = lambda cnt, sign: " ".join(  # noqa: E731
            f"{sign}{n if n > 1 else ''}{'×' if n > 1 else ''}{short.get(cnames.get(i, str(i)), cnames.get(i, str(i)))}"
            for i, n in cnt.items())
        lab = (fmt(rem, "−") + "  " + fmt(add, "+")).strip()
        if name.startswith("field_"):
            lab = f"{name} (another team's full list)"
        return lab if len(lab) < 54 else lab[:52] + "…"

    gdef = _load_json("docs/data/ablation_candidates.json")
    mdef = _load_json("docs/data/ablation_mewtwo_candidates.json")
    cands = []
    for src, deck, defs in ((g1, "Grimmsnarl", gdef), (m1, "TR Mewtwo", mdef)):
        for r in src["rows"][1:]:
            se = abs(r["delta"] / r["z"]) if r["z"] else 0.0
            cands.append({"name": r["name"], "label": diff_label(defs, r["name"]), "deck": deck,
                          "d": 100 * r["delta"], "ci": 100 * 1.96 * se, "z": r["z"], "n": r["n"]})
    cands.sort(key=lambda c: c["d"])
    # 第2段（微調整後・各 20,000 戦）
    def s2(name):
        return _load_json(f"docs/data/ablation_results/{name}.json")["rows"][0]
    stage2 = [
        {"label": "Grimmsnarl: −Tool Scrapper +Boss's Orders\n(3rd gust — pick the benched target)",
         "zero": next(c for c in cands if c["name"] == "-TS_+Boss"),
         "base": s2("s2_grim_base_eval"), "cand": s2("s2_grim__TS__Boss_eval")},
        {"label": "Grimmsnarl: another team's full list (field_v3)\n(different engine, same attacker)",
         "zero": next(c for c in cands if c["name"] == "field_v3"),
         "base": s2("s2_grim_base_eval"), "cand": s2("s2_grim_field_v3_eval")},
        {"label": "TR Mewtwo: 3rd copy of Mewtwo ex\n(attacker redundancy vs prize-outs)",
         "zero": next(c for c in cands if c["name"] == "mw_mewtwo3"),
         "base": s2("s2_mw_base_eval"), "cand": s2("s2_mw_mw_mewtwo3_eval")},
        {"label": "TR Mewtwo: 2nd Night Stretcher\n(recover attacker / energy from discard)",
         "zero": next(c for c in cands if c["name"] == "mw_night2"),
         "base": s2("s2_mw_base_eval"), "cand": s2("s2_mw_mw_night2_eval")},
    ]
    for s in stage2:
        pb, pc, n = s["base"]["wr"], s["cand"]["wr"], s["base"]["n"]
        s["ft"] = 100 * (pc - pb)
        s["ft_ci"] = 100 * 1.96 * np.sqrt(pb * (1 - pb) / n + pc * (1 - pc) / n)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14.5, 6.4), gridspec_kw={"width_ratios": [1.2, 1], "wspace": 0.62})
    # (a)
    y = np.arange(len(cands))
    _style(a1, "Stage 1 — screen 20 single-card changes with the policy frozen",
           "Δ win rate vs the most common list (pp, 95% CI), 10,000 games per arm against meta-weighted opponents.\n"
           "Filled = sent to stage 2.", xlab="Δ win rate vs the most common list (pp)")
    _hgrid_off(a1)
    a1.axvline(0, color=AXIS, lw=1.0, zorder=1)
    sent = {s["zero"]["name"] for s in stage2}
    for i, c in enumerate(cands):
        col = S[0] if c["deck"] == "Grimmsnarl" else S[1]
        a1.plot([c["d"] - c["ci"], c["d"] + c["ci"]], [y[i]] * 2, color=col, lw=1.4, alpha=0.55, zorder=2)
        a1.scatter([c["d"]], [y[i]], s=58, facecolor=(col if c["name"] in sent else SURFACE), edgecolor=col,
                   linewidth=1.6, zorder=3)
    a1.set_yticks(y)
    a1.set_yticklabels([c["label"] for c in cands], fontsize=8.5, color=INK2)
    a1.tick_params(axis="y", length=0)
    a1.spines["left"].set_visible(False)
    a1.set_ylim(-0.7, len(cands) - 0.3)
    hs = [plt.Line2D([], [], marker="o", ls="", ms=8, color=S[0], mec=SURFACE, label="Grimmsnarl list (14 candidates)"),
          plt.Line2D([], [], marker="o", ls="", ms=8, color=S[1], mec=SURFACE, label="TR Mewtwo list (6 candidates)")]
    a1.legend(handles=hs, loc="lower right", frameon=False, fontsize=9, labelcolor=INK2)
    # (b)
    y2 = np.arange(len(stage2))[::-1]
    _style(a2, "Stage 2 — fine-tune each arm, then re-measure",
           "Δ vs the most common list after 1,650/1,150 PPO iterations per arm (20,000 fresh games per arm).\n"
           "Hollow = before fine-tuning (stage 1), filled = after. Gains shrink; losses persist.",
           xlab="Δ win rate vs the most common list (pp)")
    _hgrid_off(a2)
    a2.axvline(0, color=AXIS, lw=1.0, zorder=1)
    for i, s in enumerate(stage2):
        col = S[0] if s["label"].startswith("Grimmsnarl") else S[1]
        z = s["zero"]
        a2.plot([z["d"], s["ft"]], [y2[i]] * 2, color=GRID, lw=3.2, zorder=1, solid_capstyle="round")
        a2.scatter([z["d"]], [y2[i]], s=64, facecolor=SURFACE, edgecolor=col, linewidth=1.6, zorder=2)
        a2.plot([s["ft"] - s["ft_ci"], s["ft"] + s["ft_ci"]], [y2[i]] * 2, color=col, lw=1.4, alpha=0.55, zorder=2)
        a2.scatter([s["ft"]], [y2[i]], s=80, color=col, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        a2.text(max(z["d"], s["ft"]) + 0.45, y2[i], f"{z['d']:+.1f} → {s['ft']:+.1f}", va="center", color=INK,
                fontsize=9.5, fontweight="600")
    a2.set_yticks(y2)
    a2.set_yticklabels([s["label"] for s in stage2], fontsize=9, color=INK)
    a2.tick_params(axis="y", length=0)
    a2.spines["left"].set_visible(False)
    a2.set_xlim(-8, 5)
    a2.set_ylim(-0.7, len(stage2) - 0.3)
    _save(fig, "fig_ablation_two_stage")


# ---------------------------------------------------------------- compute / training cost table
def fig_compute_table():
    """計算環境と学習コストの表（語数外で出すため画像化）。数値は docs/plan-writeup-2026-09.md §7.2。"""
    rows = [
        ("Data", "Public daily top episodes, 47 days → 34.4M decisions (train 31,958,161 / temporal hold-out 756,908)"),
        ("Behaviour cloning", "All teams pooled, 2 epochs, batch 512, lr 3e-4 → 1.53 GPU-hours; hold-out agreement 73.4%"),
        ("Per-deck fine-tune", "Top pilots of that archetype only, 5 epochs (minutes per deck)"),
        ("PPO league", "1 GPU + 32 CPU cores, 28 actor processes; 49,152 decisions/iteration; 2–4 h per run"),
        ("PPO total", "≈95 GPU-hours across 3 league rounds (18.6 + 18.8 + 24.2 h) + repairs (4.7 + 2.3 h) + A/B (≈9 h)"),
        ("Deployed model", "691,843 learned parameters + 172,480 static card/attack attributes; 2.6 MB NumPy weights"),
        ("Inference", "3.4 ms per decision (mean), 7.4 ms max — measured in real games on a 256-thread host; game budget 600 s, no per-move limit"),
        ("Software", "Python 3.12, PyTorch for training, pure NumPy at inference; poetry.lock pins every version"),
    ]
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.set_axis_off()
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    fig.text(0.02, 0.96, "Compute footprint: the whole pipeline fits in about 100 GPU-hours",
             color=INK, fontsize=13, fontweight="bold", va="top")
    fig.text(0.02, 0.905, "What it took to train and ship the two submitted agents. The design leans on measurement "
             "and curriculum, not on scale.", color=INK2, fontsize=9.5, va="top")
    top, h = 0.84, 0.095
    for i, (k, v) in enumerate(rows):
        yy = top - i * h
        if i % 2 == 0:
            fig.patches.append(plt.Rectangle((0.02, yy - h + 0.012), 0.96, h, transform=fig.transFigure,
                                             facecolor="#f3f2ee", edgecolor="none", zorder=0))
        fig.text(0.035, yy - h / 2 + 0.012, k, color=INK, fontsize=10, fontweight="600", va="center")
        fig.text(0.215, yy - h / 2 + 0.012, v, color=INK2, fontsize=9.6, va="center")
    p = OUT / "table_compute.png"
    _stamp(fig, "table_compute")
    fig.savefig(p, dpi=200, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"[ok] {p}")


# ---------------------------------------------------------------- architecture / fail-closed
def fig_architecture():
    """推論パイプラインと fail-closed 3層の図。数値は runs/selfplay_*/config.json・bench_inference.py の実測。"""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(13, 6.0))
    ax.set_axis_off()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    fig.patch.set_facecolor(SURFACE)

    def box(x, y, w, h, title, body, fc="#f3f2ee", ec=AXIS, tc=INK):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    facecolor=fc, edgecolor=ec, linewidth=1.2, zorder=2))
        ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", color=tc, fontsize=10, fontweight="bold",
                zorder=3)
        ax.text(x + w / 2, y + h / 2 - 1.6, body, ha="center", va="center", color=INK2, fontsize=8.6,
                linespacing=1.35, zorder=3)

    def arrow(x0, y0, x1, y1, color=INK2):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14, color=color,
                                     linewidth=1.4, zorder=4))

    fig.text(0.02, 0.965, "One network for every decision type, wrapped in three layers that cannot crash",
             color=INK, fontsize=13, fontweight="bold", va="top")
    fig.text(0.02, 0.915, "Inference path of the submitted agents (pure NumPy, 3.4 ms per decision measured in real "
             "games). Training used the same network in PyTorch; an export gate refuses weights unless both paths agree.",
             color=INK2, fontsize=9.5, va="top")

    # --- main pipeline (top row) ---
    yb, hb = 36, 17
    box(1, yb, 16, hb, "Engine observation",
        "board, hands, prizes,\ndiscard pile, and the\nK enumerated legal options")
    box(21, yb, 21, hb, "Featurize",
        "56 board tokens × 40 features\nK option tokens × 64 features\ncard / attack IDs → static tables\n"
        "(1,536×80 card, 1,600×31 attack)")
    box(46, yb, 25, hb, "PTCGNet — pointer network",
        "2 self-attention layers over the board\n2 cross-attention layers: option → board\n"
        "shared scoring head, d=128, 4 heads\n691,843 learned parameters", fc="#e8f0fb", ec=S[0])
    box(75, yb, 24, hb, "Mask, then choose",
        "legal-move mask ∧ guard mask\n(take a proven lethal) applied\nBEFORE sampling — the learner\n"
        "optimises exactly what plays")
    for x0, x1 in ((17.4, 20.6), (42.4, 45.6), (71.4, 74.6)):
        arrow(x0, yb + hb / 2, x1, yb + hb / 2)

    # --- fail-closed ladder (bottom row) ---
    yl, hl = 8, 15
    box(1, yl, 30, hl, "Layer 1 — NumPy policy",
        "99.7–100% of decisions in measured games\nfeaturize → PTCGNet → mask → argmax", fc="#e8f0fb", ec=S[0])
    box(35, yl, 30, hl, "Layer 2 — rule fallback",
        "greedy-lethal heuristic on the same\nenumerated options; used if layer 1\nraises or returns an illegal move")
    box(69, yl, 30, hl, "Layer 3 — legal random",
        "any exception anywhere → a legal index\nper-turn selection cap ends infinite mirrors\n(the one ERROR we ever had)")
    arrow(31.4, yl + hl / 2, 34.6, yl + hl / 2, color=CRIT)
    arrow(65.4, yl + hl / 2, 68.6, yl + hl / 2, color=CRIT)
    ax.text(33, yl + hl / 2 + 2.3, "on failure", ha="center", color=CRIT, fontsize=8.5)
    ax.text(67, yl + hl / 2 + 2.3, "on failure", ha="center", color=CRIT, fontsize=8.5)
    # vertical link: pipeline is layer 1
    arrow(58.5, yb - 0.6, 16, yl + hl + 0.8, color=MUTED)
    ax.text(50, 27.5, "the pipeline above is layer 1", color=MUTED, fontsize=8.5, ha="left")

    # --- export gate note ---
    ax.text(99, 3.0, "Export gate: PyTorch vs NumPy logits max |Δ| = 7.2e-7, argmax agreement 100% on held-out decisions; "
            "weights are not shipped unless this passes.", ha="right", va="center", color=INK2, fontsize=8.6,
            style="italic")
    p = OUT / "fig_architecture.png"
    _stamp(fig, "fig_architecture")
    fig.savefig(p, dpi=200, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"[ok] {p}")


# ---------------------------------------------------------------- portfolio hedge (counterfactual)
def fig_portfolio_hedge():
    """2枠の保険価値: 最終10日の対面別実測勝率 × 日次メタシェア → 各エージェントの期待勝率と良い方の包絡線。"""
    d = _load_json("docs/data/portfolio_reweighting.json")
    s = d["series"]
    x = np.arange(len(s))
    dates = [r["date"] for r in s]
    e_dh = np.array([r["E_kangaskhan_dh"] for r in s])
    e_k2 = np.array([r["E_hydrapple_k2"] for r in s])
    e_best = np.maximum(e_dh, e_k2)
    cov = np.minimum(np.array([r["coverage_kangaskhan_dh"] for r in s]),
                     np.array([r["coverage_hydrapple_k2"] for r in s]))
    sm = d["summary"]
    lj, la = sm["late_july"], sm["late_august"]

    fig, (ax, ac) = plt.subplots(2, 1, figsize=(11, 6.6), sharex=True,
                                 gridspec_kw={"height_ratios": [3.2, 1], "hspace": 0.08})
    _style(ax, "The second slot was insurance: in the late-July metagame Hydrapple would have carried, in late August Kangaskhan did",
           "A reweighting stress test, not a historical backtest: each agent's matchup win rates measured on the real ladder "
           "Aug 21–31 (n ≥ 30 per matchup) are held fixed\nand reweighted by each day's archetype shares. Per-matchup rates drift "
           "with pilots and lists (Section 2); unmeasured archetypes take the agent's residual rate. Shaded band = gap between "
           "the two.",
           ylab="Expected win rate vs that day's field (%)")
    ax.fill_between(x, np.minimum(e_dh, e_k2), e_best, color=GRID, alpha=0.7, zorder=1)
    ax.plot(x, e_dh, color=S[0], lw=2.0, zorder=3, label="kangaskhan_dh")
    ax.plot(x, e_k2, color=S[1], lw=2.0, zorder=3, label="hydrapple_k2")
    ax.plot(x, e_best, color=INK, lw=1.2, ls=(0, (3, 2)), zorder=4, label="better of the two (leaderboard takes the max)")
    ax.axhline(50, color=AXIS, lw=1.0, zorder=2)
    ax.set_ylim(min(e_dh.min(), e_k2.min()) - 4, max(e_dh.max(), e_k2.max()) + 6)
    # 注記: 7月下旬と8月下旬
    def idx(date):
        return dates.index(date) if date in dates else None
    i_j, i_a = idx("2026-07-26"), idx("2026-08-26")
    if i_j is not None:
        ax.annotate(f"late July (Grimmsnarl era)\nHydrapple {lj['hydrapple_k2']:.0f}% vs Kangaskhan {lj['kangaskhan_dh']:.0f}%",
                    xy=(i_j, e_k2[i_j]), xytext=(i_j - 22, e_k2[i_j] + 4.5), ha="left", va="bottom", color=INK, fontsize=9.5,
                    fontweight="600", arrowprops=dict(arrowstyle="->", color=INK2, lw=1.2))
    if i_a is not None:
        ax.annotate(f"late August (Dragapult era)\nKangaskhan {la['kangaskhan_dh']:.0f}% vs Hydrapple {la['hydrapple_k2']:.0f}%",
                    xy=(i_a, e_dh[i_a]), xytext=(i_a - 26, e_dh[i_a] + 4.5), ha="left", va="bottom", color=INK, fontsize=9.5,
                    fontweight="600", arrowprops=dict(arrowstyle="->", color=INK2, lw=1.2))
    ax.legend(loc="lower left", frameon=False, fontsize=9, labelcolor=INK2)
    # 下段: 被覆率
    _style(ac, ylab="Coverage (%)")
    ac.fill_between(x, 0, cov, color=S[2], alpha=0.35, zorder=2)
    ac.plot(x, cov, color=S[2], lw=1.4, zorder=3)
    ac.set_ylim(0, 100)
    ac.text(0.5, 8, "share of that day's field covered by matchups we measured (rest = residual rate)", color=INK2,
            fontsize=8.5, va="bottom")
    step = max(1, len(dates) // 9)
    ac.set_xticks(x[::step])
    ac.set_xticklabels([dd[5:] for dd in dates[::step]])
    ac.set_xlim(0, len(dates) - 1)
    _save(fig, "fig_portfolio_hedge")


# ---------------------------------------------------------------- deck anatomy
def _ladder_lines(agent: str) -> list[str]:
    """`analyze_replay_behavior.py` の集計から、デッキの想定が実戦で成立していたかを文にする。

    数値はここでハードコードせず replay_behavior_<agent>.json から都度読む。
    """
    try:
        b = _load_json(f"docs/data/replay_behavior_{agent}.json")
    except FileNotFoundError:
        return []
    rows = b["incoming_by_attacker"]

    def grab(active, kinds=None, ability=None):
        n = z = 0
        for r in rows:
            if r["our_active"] != active:
                continue
            if kinds and r["attacker_ex"] not in kinds:
                continue
            if ability and r["attacker_ability"] != ability:
                continue
            n += r["attacks"]
            z += r["attacks"] * r["zero_damage_share"] / 100
        return n, (100 * z / n if n else 0.0)

    out = [f"What the {b['games']:,} post-freeze ladder games show (win rate {b['wr']}%):"]
    if agent == "kangaskhan_dh":
        k = b["key_cards"]["Crustle"]
        out.append(f"Crustle reached the Active Spot in {k['share']}% of games (median turn {k['median_turn']}) "
                   f"and stood there {b['wall_active_turns_mean']['Crustle']} of our {b['mean_our_turns']} turns.")
        n1, z1 = grab("Crustle", ("ex",), "no-ability")
        n2, z2 = grab("Crustle", ("non-ex",))
        out.append(f"Axis 1 — with Crustle Active, attacks by Pokémon ex with no Ability did zero damage "
                   f"{z1:.0f}% of the time (n = {n1:,}); attacks by non-ex Pokémon, {z2:.0f}% (n = {n2:,}).")
        n3, z3 = grab("Cornerstone", None, "ability")
        n4, z4 = grab("Cornerstone", None, "no-ability")
        out.append(f"Axis 2 — with Cornerstone Active, attackers that have an Ability did zero damage {z3:.0f}% "
                   f"(n = {n3:,}); attackers without one, {z4:.0f}% (n = {n4:,}).")
        n5, z5 = grab("Crustle", ("mega-ex",), "no-ability")
        fp = _load_json("docs/data/final_pair_matchups.json")["agents"]
        lop = {a: next(m["wr"] for m in fp[a]["matchups"] if m["opponent"] == "Mega Lopunny ex") for a in fp}
        out.append(f"The hole: Mega Lopunny ex's Spiky Hopper ignores effects on the defender, so only {z5:.0f}% "
                   f"of its {n5:,} attacks were stopped — a {lop['kangaskhan_dh']:.0f}% matchup for this agent, "
                   f"{lop['hydrapple_k2']:.0f}% for the other slot.")
        dmg = {a["attack"]: a["total_damage"] for a in b["our_attacks"]}
        tot = sum(dmg.values()) or 1
        out.append(f"The wall was also the offence: Superb Scissors dealt {100 * dmg.get('Superb Scissors', 0) / tot:.0f}% "
                   f"of our damage, Rapid-Fire Combo {100 * dmg.get('Rapid-Fire Combo', 0) / tot:.0f}%.")
    else:
        for name in ("Meganium", "Hydrapple ex"):
            k = b["key_cards"][name]
            out.append(f"{name} landed in {k['share']}% of games (median turn {k['median_turn']}): "
                       f"win rate {k['wr_landed']}% with it, {k['wr_not_landed']}% without.")
        w = b["wild_growth_doubling"]
        out.append(f"Wild Growth doubling held in {w['share']:.0f}% of the {w['observations']:,} board states with "
                   f"Meganium and Basic Grass in play (provided energy = twice the cards).")
        sc = b["damage_scaling"]["Syrup Storm"]
        out.append(f"Syrup Storm dealt {sc['mean_damage']:.0f} damage on average when it connected. "
                   f"{100 * sc['fits_provided_energy'] / sc['n']:.0f}% of its {sc['n']:,} damaging uses match "
                   f"30 + 30 per Grass Energy provided (the doubled count), {100 * sc['fits_card_count'] / sc['n']:.0f}% "
                   f"the raw card count — the engine counts the doubling.")
        dmg = {a["attack"]: a["total_damage"] for a in b["our_attacks"]}
        tot = sum(dmg.values()) or 1
        out.append(f"Syrup Storm (Hydrapple ex) dealt {100 * dmg.get('Syrup Storm', 0) / tot:.0f}% of our damage, "
                   f"Myriad Leaf Shower (Teal Mask Ogerpon ex) {100 * dmg.get('Myriad Leaf Shower', 0) / tot:.0f}%.")
        k = b["key_cards"]["Mega Meganium ex"]
        out.append(f"Mega Meganium ex landed in only {k['share']}% of games: an option, not the plan.")
    return out


def fig_deck_anatomy():
    """最終2デッキの解剖表（カード名・枚数・役割・注記）。docs/data/deck_roles.json から生成。画像なし。"""
    d = _load_json("docs/data/deck_roles.json")
    group_of = {"attacker": "Attack", "finisher": "Attack", "tech": "Tech", "wall": "Defence", "heal": "Defence",
                "accel": "Engine", "evo": "Engine", "energy": "Engine", "draw": "Draw / search", "search": "Draw / search",
                "gust": "Control", "disrupt": "Control", "stadium": "Control", "recovery": "Recovery"}
    group_col = {"Attack": S[1], "Defence": S[0], "Engine": S[2], "Draw / search": S[3], "Control": S[6],
                 "Recovery": S[4], "Tech": MUTED}
    order = ["Attack", "Defence", "Engine", "Control", "Draw / search", "Recovery", "Tech"]

    import textwrap

    decks = list(d["decks"].items())
    fig, axes = plt.subplots(1, 2, figsize=(17.5, 11.5), gridspec_kw={"wspace": 0.16})
    fig.patch.set_facecolor(SURFACE)
    fig.text(0.02, 0.985, "What each deck is built to do — the two submitted lists, card by card",
             color=INK, fontsize=13, fontweight="bold", va="top")
    fig.text(0.02, 0.96, "60 cards each. Roles are read from the card text of the exact printings, which are listed in the attached decklists. The measured lines come from re-parsing the 2,000 post-freeze replays. Colour = role group; no card images.",
             color=INK2, fontsize=9.5, va="top")
    for ax, (name, deck) in zip(axes, decks):
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.text(0.0, 0.995, textwrap.fill(deck["title"], 64), color=INK, fontsize=11.5, fontweight="bold", va="top",
                linespacing=1.2)
        plan = textwrap.fill(deck["game_plan"], 84)
        n_title_lines = deck["title"].count("\n") + (1 if len(deck["title"]) <= 64 else 2)
        y_plan = 0.995 - 0.03 * n_title_lines - 0.005
        ax.text(0.0, y_plan, plan, color=INK2, fontsize=8.6, va="top", linespacing=1.3)
        n_plan_lines = plan.count("\n") + 1
        y0 = y_plan - 0.021 * n_plan_lines - 0.028
        # Targets ブロックの高さを先に確保する（表がその上で終わるように）
        tg = deck.get("targets", {})
        tg_lines = [textwrap.fill(f"{k}: {v}", 96) for k, v in tg.items()]
        tg_h = (0.018 * (sum(t.count("\n") + 1 for t in tg_lines) + 1) + 0.03) if tg else 0.0
        lc_raw = _ladder_lines(name)
        lc_lines = [textwrap.fill(t, 100) for t in lc_raw[1:]]
        lc_h = (0.017 * (sum(t.count("\n") + 1 for t in lc_lines) + 1.6) + 0.03) if lc_lines else 0.0
        tg_h += lc_h
        cards = sorted(deck["cards"], key=lambda c: (order.index(group_of[c["role"]]), -c["n"], c["card"]))
        row_h = (y0 - tg_h - 0.02) / (len(cards) + len(order) + 1)
        y = y0
        # header
        ax.text(0.00, y, "Copies", color=MUTED, fontsize=8, va="top", fontweight="600")
        ax.text(0.09, y, "Card", color=MUTED, fontsize=8, va="top", fontweight="600")
        ax.text(0.40, y, "Role", color=MUTED, fontsize=8, va="top", fontweight="600")
        ax.text(0.50, y, "Note", color=MUTED, fontsize=8, va="top", fontweight="600")
        y -= row_h * 1.1
        last_group = None
        for c in cards:
            g = group_of[c["role"]]
            if g != last_group:
                ax.add_patch(plt.Rectangle((0.0, y - row_h * 0.85), 1.0, row_h * 0.9, color=group_col[g], alpha=0.12, lw=0))
                ax.text(0.0, y - row_h * 0.1, g, color=group_col[g] if g != "Tech" else INK2, fontsize=8.5,
                        fontweight="bold", va="top")
                y -= row_h
                last_group = g
            ax.add_patch(plt.Rectangle((0.0, y - row_h * 0.72), 0.012, row_h * 0.62, color=group_col[g], lw=0))
            ax.text(0.03, y, f"{c['n']}×", color=INK, fontsize=8.8, va="top", fontweight="600")
            ax.text(0.09, y, c["card"], color=INK, fontsize=8.8, va="top")
            ax.text(0.40, y, c["role"], color=INK2, fontsize=8.4, va="top")
            note = c.get("note", "")
            assert len(note) <= 62, f"note too long for the column: {c['card']} ({len(note)})"
            ax.text(0.50, y, note, color=INK2, fontsize=7.9, va="top")
            y -= row_h
        # targets（表の下、確保した領域に）
        if tg:
            ty = tg_h - 0.01
            ax.text(0.0, ty, "Targets", color=INK, fontsize=8.5, fontweight="bold", va="top")
            ty -= 0.02
            for t in tg_lines:
                ax.text(0.0, ty, t, color=INK2, fontsize=7.9, va="top", linespacing=1.25)
                ty -= 0.018 * (t.count("\n") + 1)
        if lc_lines:
            ly = lc_h - 0.012
            ax.text(0.0, ly, lc_raw[0], color=INK, fontsize=8.5, fontweight="bold", va="top")
            ly -= 0.021
            for t in lc_lines:
                ax.text(0.0, ly, t, color=INK2, fontsize=7.7, va="top", linespacing=1.25)
                ly -= 0.017 * (t.count("\n") + 1)
    p = OUT / "fig_deck_anatomy.png"
    _stamp(fig, "fig_deck_anatomy")
    fig.savefig(p, dpi=200, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"[ok] {p}")


def fig_field_matrix_final():
    _field_matrix("final-pair-window", "fig_field_matrix_final", top=8, title_note=" — the final 10 days")


def fig_field_matrix_full():
    _field_matrix("full", "fig_field_matrix_full", top=10, title_note=" — whole competition")


FIGS = {"fig1": fig1_meta_timeline, "fig2": fig2_dead_embedding,
        "fig3": fig3_rating_confound, "fig4": fig4_matchup_profile,
        "fig_pair_coverage": fig_pair_coverage, "fig_curriculum_gates": fig_curriculum_gates,
        "fig_internal_vs_ladder": fig_internal_vs_ladder, "fig_league_decay": fig_league_decay,
        "fig_field_matrix_final": fig_field_matrix_final, "fig_field_matrix_full": fig_field_matrix_full,
        "fig_clone_spread": fig_clone_spread, "fig_ablation_two_stage": fig_ablation_two_stage,
        "table_compute": fig_compute_table, "fig_architecture": fig_architecture,
        "fig_portfolio_hedge": fig_portfolio_hedge, "fig_deck_anatomy": fig_deck_anatomy}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", help="生成する図（既定=全部）")
    a = ap.parse_args()
    for k, f in FIGS.items():
        if a.only and k not in a.only:
            continue
        f()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
