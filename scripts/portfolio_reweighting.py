#!/usr/bin/env python
"""2枠ポートフォリオの「保険」を反実仮想の再重み付けで定量化する（レポート Fig. 12 / §6 の一次データ）。

これは **historical backtest ではない**（過去のエージェントの実成績を再生するものではない）。
LB スコアは2枠の最大値なので、最終 LB（993.8）は kangaskhan_dh 単体の値であり、第2枠は結果に
寄与していない。第2枠が何に対する保険だったかを示すため、**最終10日に実測した対面別勝率を固定し**、
**各日のアーキタイプシェアで再重み付け**して、各エージェントの日次期待勝率と「良い方」の包絡線を計算する
（メタ構成に対するストレステスト）。

  E_A(t) = Σ_k share_k(t) · wr_A(k) + share_other(t) · wr_A(other)
  wr_A(other) = A の総勝率から測定済み対面を除いた残差勝率
  coverage(t) = Σ_k share_k(t)   （測定済み対面がその日のメタをどれだけ覆うか）

これは反実仮想（対面別勝率が期間を通じて一定という仮定。実際の対戦帯は上位帯シェアと異なる）。
図と本文にその旨を明記する。

    poetry run python scripts/portfolio_reweighting.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from _bootstrap import REPO_ROOT

MATCHUPS = REPO_ROOT / "docs/data/final_pair_matchups.json"
TIMELINE = REPO_ROOT / "docs/data/meta_timeline.csv"
OUT = REPO_ROOT / "docs/data/portfolio_reweighting.json"
MIN_N = 30  # 対面別勝率を使う最小試合数

# meta_timeline.csv の列名（表示名 or 生ID）→ final_pair_matchups.json の対面名（カード名）
COL_TO_MATCHUP = {
    "Grimmsnarl ex": "Marnie's Grimmsnarl ex",
    "Alakazam": "Alakazam", "743": "Alakazam", "245": "Alakazam",   # 66/245/743 は同じ Alakazam デッキ
    "M Kangaskhan ex": "Mega Kangaskhan ex",
    "Dragapult ex": "Dragapult ex",
    "M Lucario ex": "Mega Lucario ex",
    "M Lopunny ex": "Mega Lopunny ex",
    "Hydrapple ex": "Hydrapple ex",
    "TR Mewtwo ex": "Team Rocket's Mewtwo ex",
    "Thwackey": "Thwackey", "90": "Thwackey",
    "Garchomp ex": "Cynthia's Garchomp ex", "381": "Cynthia's Garchomp ex",
    "96": "Teal Mask Ogerpon ex", "Teal Ogerpon ex": "Teal Mask Ogerpon ex",
    "Archaludon ex": "Archaludon ex", "190": "Archaludon ex",
    "M Starmie ex": "Mega Starmie ex", "1031": "Mega Starmie ex",
    "304": "Hop’s Snorlax", "Hop's Snorlax": "Hop’s Snorlax",
    "345": "Crustle", "Crustle": "Crustle",
}


def main() -> int:
    d = json.loads(MATCHUPS.read_text(encoding="utf-8"))
    agents = {}
    for name, a in d["agents"].items():
        prof = {m["opponent"]: m for m in a["matchups"] if m["n"] >= MIN_N}
        n_meas = sum(m["n"] for m in prof.values())
        w_meas = sum(m["wins"] for m in prof.values())
        other_n = a["games"] - n_meas
        other_wr = (a["wins"] - w_meas) / other_n if other_n > 0 else a["wins"] / a["games"]
        agents[name] = {"profile": {k: m["wr"] / 100 for k, m in prof.items()},
                        "other_wr": other_wr, "other_n": other_n, "overall": a["wr"] / 100}

    rows = list(csv.DictReader(open(TIMELINE, encoding="utf-8")))
    cols = [c[:-6] for c in rows[0] if c.endswith("_share")]
    unmapped = [c for c in cols if c not in COL_TO_MATCHUP]

    series = []
    for r in rows:
        share = {}
        for c in cols:
            v = r.get(f"{c}_share")
            if not v:
                continue
            m = COL_TO_MATCHUP.get(c)
            if m is None:
                continue
            share[m] = share.get(m, 0.0) + float(v)
        rec = {"date": r["date"], "sides": int(r["sides"])}
        for name, a in agents.items():
            covered = sum(s for k, s in share.items() if k in a["profile"])
            e = sum(s * a["profile"][k] for k, s in share.items() if k in a["profile"])
            e += (1.0 - covered) * a["other_wr"]
            rec[f"E_{name}"] = round(100 * e, 2)
            rec[f"coverage_{name}"] = round(100 * covered, 1)
        rec["E_best"] = round(max(rec[f"E_{n}"] for n in agents), 2)
        rec["better"] = max(agents, key=lambda n: rec[f"E_{n}"])
        series.append(rec)

    names = list(agents)
    a0, a1 = names[0], names[1]
    days_a1_better = sum(1 for s in series if s["better"] == a1)
    summary = {
        "days": len(series),
        "window_profile": d["agents"][a0]["window"],
        f"days_{a1}_better": days_a1_better,
        f"days_{a0}_better": len(series) - days_a1_better,
        **{f"min_E_{n}": min(s[f"E_{n}"] for s in series) for n in names},
        **{f"mean_E_{n}": round(sum(s[f"E_{n}"] for s in series) / len(series), 2) for n in names},
        "min_E_best": min(s["E_best"] for s in series),
        "mean_E_best": round(sum(s["E_best"] for s in series) / len(series), 2),
        **{f"mean_coverage_{n}": round(sum(s[f"coverage_{n}"] for s in series) / len(series), 1) for n in names},
    }
    # 7月下旬（Grimmsnarl 期）と 8月下旬（Dragapult 期）の対比
    def window_mean(d_from, d_to, key):
        xs = [s[key] for s in series if d_from <= s["date"] <= d_to]
        return round(sum(xs) / len(xs), 2) if xs else None
    for label, (f, t) in {"late_july": ("2026-07-21", "2026-07-31"), "late_august": ("2026-08-21", "2026-08-30")}.items():
        summary[label] = {n: window_mean(f, t, f"E_{n}") for n in names} | {"best": window_mean(f, t, "E_best")}

    out = {
        "method": __doc__.strip().split("\n\n")[1],
        "assumptions": ["対面別勝率は最終10日（Aug 21–31）のラダー実測を全期間に適用（定常性の仮定）",
                        "シェアは上位帯の公開エピソード。実際に当たった相手分布はレート帯で異なる",
                        f"n<{MIN_N} の対面は 'other' の残差勝率に含める"],
        "agents": {n: {"other_wr": round(100 * a["other_wr"], 1), "other_n": a["other_n"],
                       "profile_wr": {k: round(100 * v, 1) for k, v in a["profile"].items()}} for n, a in agents.items()},
        "unmapped_columns": unmapped,
        "summary": summary,
        "series": series,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"days={summary['days']}  {a0} better on {summary[f'days_{a0}_better']} days, {a1} on {days_a1_better}")
    for n in names:
        print(f"  {n:14s} mean {summary[f'mean_E_{n}']:5.1f}  min {summary[f'min_E_{n}']:5.1f}  coverage {summary[f'mean_coverage_{n}']:4.1f}%")
    print(f"  {'best-of-two':14s} mean {summary['mean_E_best']:5.1f}  min {summary['min_E_best']:5.1f}")
    print("  late July :", summary["late_july"]); print("  late Aug  :", summary["late_august"])
    if unmapped:
        print("  unmapped columns (→ other):", unmapped)
    print(f"書き出し: {OUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
