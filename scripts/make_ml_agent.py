#!/usr/bin/env python
"""ML エージェントの組み立て → 検証チェーン一括実行。

テンプレ（agents/_ml_template/）+ vendor（schema/vocab/features/np_forward）+
export 成果物（weights.npz/config.json）+ 静的テーブル（cards/attacks.npz）+
deck.csv（教師チームの最頻60枚 or 指定CSV）から agents/<name>/ を構築し、
smoke_test → build_submission → tar_smoke_test を順に実行する。

使い方:
    poetry run python scripts/make_ml_agent.py alakazam_bc \
        --export runs/bc_alakazam_v1/export --deck-from-team Majkel1337
    poetry run python scripts/make_ml_agent.py thwackey_bc \
        --export runs/bc_thwackey_v1/export --deck decks/thwackey.csv
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401
from _bootstrap import AGENTS_DIR, REPO_ROOT

from vendor import VENDOR_ML, check_sync, vendor

TEMPLATE = AGENTS_DIR / "_ml_template"
ML_DATA = REPO_ROOT / "data" / "ml"


def mine_team_deck(team: str, dates: list[str]) -> list[int]:
    """episodes から指定チームの最頻60枚を抽出する。"""
    from ptcg.ml.bc.extract import decks_of

    counter: Counter = Counter()
    for date in dates:
        for p in sorted((REPO_ROOT / "data" / "episodes" / date).glob("*.json")):
            try:
                ep = json.load(open(p))
            except Exception:
                continue
            teams = (ep.get("info") or {}).get("TeamNames") or []
            if team not in teams:
                continue
            i = teams.index(team)
            decks = decks_of(ep)
            if decks[i]:
                counter[tuple(sorted(decks[i]))] += 1
    if not counter:
        raise SystemExit(f"チーム {team} のデッキが {dates} に見つからない")
    deck, n = counter.most_common(1)[0]
    print(f"[ok] {team} の最頻デッキ（{n}戦で使用, 変種{len(counter)}種）")
    return list(deck)


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if proc.returncode != 0:
        raise SystemExit(f"[FAIL] {' '.join(cmd)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name")
    ap.add_argument("--export", required=True, help="export_policy.py の出力ディレクトリ")
    ap.add_argument("--deck", default=None, help="deck.csv のコピー元")
    ap.add_argument("--deck-from-team", default=None, help="episodes から最頻60枚を抽出するチーム名")
    ap.add_argument("--deck-dates", nargs="*", default=None, help="抽出対象日（既定=最新1日）")
    ap.add_argument("--games", type=int, default=20, help="smoke の対戦数")
    ap.add_argument("--skip-checks", action="store_true")
    args = ap.parse_args()

    export_dir = REPO_ROOT / args.export if not Path(args.export).is_absolute() else Path(args.export)
    for f in ("weights.npz", "config.json"):
        if not (export_dir / f).exists():
            raise SystemExit(f"{export_dir / f} が無い（先に export_policy.py）")

    dst = AGENTS_DIR / args.name
    dst.mkdir(parents=True, exist_ok=True)

    # 1. テンプレ + vendor + 資産
    for f in ("main.py", "guards.py"):
        shutil.copy2(TEMPLATE / f, dst / f)
    vendor(args.name, VENDOR_ML)
    for f in ("weights.npz", "config.json"):
        shutil.copy2(export_dir / f, dst / f)
    for f in ("cards.npz", "attacks.npz"):
        src = ML_DATA / f
        if not src.exists():
            raise SystemExit(f"{src} が無い（scripts/build_vocab.py を実行）")
        shutil.copy2(src, dst / f)

    # 2. deck.csv
    if args.deck:
        shutil.copy2(REPO_ROOT / args.deck, dst / "deck.csv")
    elif args.deck_from_team:
        dates = args.deck_dates
        if not dates:
            ep_root = REPO_ROOT / "data" / "episodes"
            dates = [sorted(p.name for p in ep_root.iterdir()
                            if p.is_dir() and (p / "_FETCHED").exists())[-1]]
        deck = mine_team_deck(args.deck_from_team, dates)
        (dst / "deck.csv").write_text("\n".join(str(x) for x in deck) + "\n")
    elif not (dst / "deck.csv").exists():
        raise SystemExit("--deck か --deck-from-team を指定（既存 deck.csv も無い）")

    bad = check_sync(args.name, VENDOR_ML)
    if bad:
        raise SystemExit(f"[FAIL] vendor 不一致: {bad}")
    print(f"[ok] agents/{args.name} 組み立て完了")

    # 3. 検証チェーン
    if not args.skip_checks:
        py = sys.executable
        run([py, "scripts/smoke_test.py", args.name, "--games", str(args.games)])
        run([py, "scripts/build_submission.py", args.name])
        run([py, "scripts/tar_smoke_test.py", args.name, "--games", "3"])
        print("\n=== make_ml_agent ALL GREEN ✅（pool 評価へ: scripts/run_eval.py）===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
