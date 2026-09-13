#!/usr/bin/env python
"""最終2提出のデッキリストを、対面のTCGプレイヤーが共有する標準形式で書き出す。

Writeup の添付用。運営回答（topic 738657, 2026-09-01）:
  「デッキリストは画像として載せるか、CSV/Kaggle データセットとして添付すれば十分で、
    語数制限の回避には当たらない。対面のTCGプレイヤーが共有する形式で構わない」

入力: agents/<agent>/deck.csv（カードID 60行）× data/EN_Card_Data.csv（カード名・エキスパンション）
出力: docs/data/decklists/<agent>.txt（標準形式）, <agent>.csv（枚数・カード名・型番・種別）

    poetry run python scripts/export_decklists.py
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401
from _bootstrap import REPO_ROOT

CARDS = REPO_ROOT / "data/EN_Card_Data.csv"
OUTDIR = REPO_ROOT / "docs/data/decklists"
AGENTS = ["kangaskhan_dh", "hydrapple_k2"]
CLASS_COL = "Stage (Pokémon)/Type (Energy and Trainer)"

STAGE_ORDER = {"Basic Pokémon": 0, "Stage 1 Pokémon": 1, "Stage 2 Pokémon": 2}
TRAINER_ORDER = {"Supporter": 0, "Item": 1, "Pokémon Tool": 2, "Stadium": 3}


def group_of(klass: str) -> str:
    if klass.endswith("Energy"):
        return "Energy"
    if "Pokémon" in klass and klass != "Pokémon Tool":
        return "Pokémon"
    return "Trainer"


def main() -> int:
    cards = {r["Card ID"]: r for r in csv.DictReader(CARDS.open(encoding="utf-8"))}
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for agent in AGENTS:
        ids = [l.strip() for l in (REPO_ROOT / f"agents/{agent}/deck.csv").read_text().splitlines() if l.strip()]
        assert len(ids) == 60, f"{agent}: {len(ids)} cards"
        counts = Counter(ids)
        entries = []
        for cid, n in counts.items():
            r = cards[cid]
            klass = r[CLASS_COL]
            entries.append({"n": n, "name": r["Card Name"], "exp": r["Expansion"],
                            "no": r["Collection No."], "class": klass, "group": group_of(klass)})
        assert sum(e["n"] for e in entries) == 60

        def key(e):
            if e["group"] == "Pokémon":
                return (STAGE_ORDER.get(e["class"], 9), e["name"])
            if e["group"] == "Trainer":
                return (TRAINER_ORDER.get(e["class"], 9), e["name"])
            return (0 if e["class"] == "Basic Energy" else 1, e["name"])

        lines = []
        for g in ("Pokémon", "Trainer", "Energy"):
            sel = sorted([e for e in entries if e["group"] == g], key=key)
            lines.append(f"{g}: {sum(e['n'] for e in sel)}")
            lines += [f"{e['n']} {e['name']} {e['exp']} {e['no']}" for e in sel]
            lines.append("")
        lines.append("Total: 60")
        (OUTDIR / f"{agent}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

        with (OUTDIR / f"{agent}.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["Count", "Card Name", "Expansion", "Collection No.", "Class"])
            for g in ("Pokémon", "Trainer", "Energy"):
                for e in sorted([x for x in entries if x["group"] == g], key=key):
                    w.writerow([e["n"], e["name"], e["exp"], e["no"], e["class"]])
        print(f"{agent}: {len(entries)} unique / 60 cards → {(OUTDIR / f'{agent}.txt').relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
