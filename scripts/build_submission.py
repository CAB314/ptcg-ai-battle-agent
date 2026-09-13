#!/usr/bin/env python
"""提出用 submission.tar.gz を組み立てる。

agents/<name>/ 配下（main.py, deck.csv, 補助 *.py）を **トップレベル** に置き、
配布データの cg/ を同梱した tar.gz を submission/ に出力する。

使い方:
    poetry run python scripts/build_submission.py random_baseline
    poetry run python scripts/build_submission.py meta_a --out submission --no-check

チェック内容（Discussion の 0点事故対策）:
- main.py と deck.csv が agents/<name>/ に存在するか
- deck.csv が 60枚・エンジン的に合法か（--no-check で省略）
- 生成 tar のトップレベルに main.py / deck.csv / cg/api.py があるか
"""

from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path

import _bootstrap  # noqa: F401
from _bootstrap import AGENTS_DIR, SUBMISSION_DIR

from ptcg import engine
from ptcg.cards import read_deck_csv


def build(agent_name: str, out_dir: Path, check: bool = True) -> Path:
    agent_dir = AGENTS_DIR / agent_name
    main_py = agent_dir / "main.py"
    deck_csv = agent_dir / "deck.csv"

    if not main_py.exists():
        raise FileNotFoundError(f"{main_py} がありません")
    if not deck_csv.exists():
        raise FileNotFoundError(f"{deck_csv} がありません")

    deck = read_deck_csv(deck_csv)
    if len(deck) != 60:
        raise ValueError(f"deck.csv は 60枚である必要があります（現在 {len(deck)}枚）")

    if check:
        ok, err = engine.check_deck(deck)
        if not ok:
            raise ValueError(
                "デッキが不正です: " + engine.DECK_ERROR_MESSAGES.get(err, str(err))
            )
        print(f"[ok] デッキ合法性チェック通過（{len(deck)}枚）")

    cg_dir = engine.find_cg_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{agent_name}.tar.gz"

    # 補助 .py（main.py 以外）+ モデル資産（*.npz / config.json）もトップレベルに同梱
    helpers = sorted(p for p in agent_dir.glob("*.py") if p.name != "main.py")
    assets = sorted(
        p for pat in ("*.npz", "config.json") for p in agent_dir.glob(pat)
    )

    with tarfile.open(out_path, "w:gz") as tar:
        tar.add(main_py, arcname="main.py")
        tar.add(deck_csv, arcname="deck.csv")
        for h in helpers + assets:
            tar.add(h, arcname=h.name)
        tar.add(cg_dir, arcname="cg")

    _verify_layout(out_path)
    size_mb = out_path.stat().st_size / 1e6
    if size_mb > 95:
        raise RuntimeError(f"tar が {size_mb:.1f}MB（Kaggle上限目安 100MB に接近）")
    print(f"[ok] 生成: {out_path} ({size_mb:.1f}MB)")
    if helpers:
        print("     同梱補助モジュール: " + ", ".join(h.name for h in helpers))
    if assets:
        print("     同梱資産: " + ", ".join(a.name for a in assets))
    return out_path


def _verify_layout(tar_path: Path) -> None:
    with tarfile.open(tar_path, "r:gz") as tar:
        names = set(tar.getnames())
    required = {"main.py", "deck.csv", "cg/api.py"}
    missing = required - names
    if missing:
        raise RuntimeError(f"tar に必須ファイルがありません: {sorted(missing)}")
    # main.py がトップレベルにあることを保証（フォルダ埋もれ = 0点の定番事故）
    if "main.py" not in names:
        raise RuntimeError("main.py がトップレベルにありません")
    print("[ok] tar レイアウト検証通過（main.py / deck.csv / cg/ がトップレベル）")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("agent", help="agents/ 配下のエージェント名")
    ap.add_argument("--out", default=str(SUBMISSION_DIR), help="出力先ディレクトリ")
    ap.add_argument(
        "--no-check", action="store_true", help="デッキ合法性チェックを省略"
    )
    args = ap.parse_args()
    try:
        build(args.agent, Path(args.out), check=not args.no_check)
    except Exception as e:  # noqa: BLE001
        print(f"[error] {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
