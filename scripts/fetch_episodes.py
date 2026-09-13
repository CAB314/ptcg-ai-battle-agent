#!/usr/bin/env python
"""デイリー Top Episodes データセットを取得する（メタ/リプレイ分析用）。

運営が毎日「平均レーティング最高の episode」を集めたデータセットを公開している。
JSON には対戦チーム名と勝敗(+1/-1)が入る（rating/agent_id は無い点に注意）。
リプレイの `selected` キーには 1 ステップの off-by-one バグがあるので、
行動クローニング等に使うときは補正すること（docs/competition-notes.md 参照）。

使い方:
    # 特定日のトップ episode
    poetry run python scripts/fetch_episodes.py --date 2026-07-13
    # 日付範囲のバックフィル（取得済み=_FETCHED はスキップ、失敗は1回リトライ）
    poetry run python scripts/fetch_episodes.py --since 2026-07-01 --until 2026-07-14
    # 全データセットの入口（index）
    poetry run python scripts/fetch_episodes.py --index

認証は KAGGLE_CONFIG_DIR=./.kaggle（access_token）を使う。
BC パイプラインの前段。取得完了マーカー _FETCHED を日付ディレクトリに置く
（build_bc_shards.py はこれを見て変換する）。
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401
from _bootstrap import REPO_ROOT

EPISODES_DIR = REPO_ROOT / "data" / "episodes"
INDEX_DATASET = "kaggle/pokemon-tcg-ai-battle-episodes-index"
DAILY_DATASET_FMT = "kaggle/pokemon-tcg-ai-battle-episodes-{date}"
FETCHED_MARKER = "_FETCHED"


def _kaggle_env() -> dict:
    env = dict(os.environ)
    env.setdefault("KAGGLE_CONFIG_DIR", str(REPO_ROOT / ".kaggle"))
    return env


def download_dataset(ref: str, dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [
        "poetry", "run", "kaggle", "datasets", "download",
        "-d", ref, "-p", str(dest), "--unzip",
    ]
    print("running:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=_kaggle_env())
    if proc.returncode != 0:
        print(f"[error] ダウンロード失敗: {ref}", file=sys.stderr)
    else:
        print(f"[ok] {ref} -> {dest}")
    return proc.returncode


def fetch_date(date: str, retries: int = 1) -> int:
    """1日分を取得して _FETCHED マーカーを置く（冪等）。"""
    dest = EPISODES_DIR / date
    marker = dest / FETCHED_MARKER
    if marker.exists():
        print(f"[skip] {date} は取得済み（{marker}）")
        return 0
    ref = DAILY_DATASET_FMT.format(date=date)
    rc = download_dataset(ref, dest)
    for _ in range(retries):
        if rc == 0:
            break
        time.sleep(10)
        rc = download_dataset(ref, dest)
    if rc == 0 and any(dest.glob("*.json")):
        marker.write_text(dt.datetime.now().isoformat())
    elif rc == 0:
        print(f"[warn] {date}: JSONが見つからないためマーカーを置かない", file=sys.stderr)
        rc = 1
    return rc


def _date_range(since: str, until: str) -> list[str]:
    d0 = dt.date.fromisoformat(since)
    d1 = dt.date.fromisoformat(until)
    return [(d0 + dt.timedelta(days=i)).isoformat() for i in range((d1 - d0).days + 1)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--date", help="YYYY-MM-DD 形式のデイリーデータセット")
    g.add_argument("--since", help="バックフィル開始日（--until と併用）")
    g.add_argument("--index", action="store_true", help="index データセットを取得")
    ap.add_argument("--until", default=None, help="バックフィル終了日（含む）")
    ap.add_argument("--dest", default=None, help="出力先（既定 data/episodes/<name>）")
    args = ap.parse_args()

    if args.index:
        return download_dataset(INDEX_DATASET, Path(args.dest) if args.dest else EPISODES_DIR / "index")
    if args.since:
        if not args.until:
            ap.error("--since には --until が必要")
        failed = []
        for date in _date_range(args.since, args.until):
            if fetch_date(date) != 0:
                failed.append(date)  # 存在しない日（データセット欠落日）はスキップ扱い
        if failed:
            print(f"[note] 取得できなかった日: {', '.join(failed)}", file=sys.stderr)
        return 0
    if args.dest:
        return download_dataset(DAILY_DATASET_FMT.format(date=args.date), Path(args.dest))
    return fetch_date(args.date)


if __name__ == "__main__":
    raise SystemExit(main())
