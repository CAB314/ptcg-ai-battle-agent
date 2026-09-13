#!/usr/bin/env python
"""episode JSON → BC シャード変換ドライバ（日付単位・冪等・STOP対応・nice実行）。

使い方:
    poetry run python scripts/build_bc_shards.py --dates 2026-07-15
    poetry run python scripts/build_bc_shards.py --all --workers 32
    touch runs/bc_shards/STOP   # 現在の episode 完了後に flush して安全に終了

- 入力: data/episodes/<date>/*.json（_FETCHED があるもの）
- 出力: data/bc_shards/v<FEATURE_VERSION>/<date>/shard_*.npz（_DONE.<hash> で冪等）
- --prune-raw: 変換完了後に生 JSON を削除（再取得は fetch_episodes.py で可能）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import _bootstrap  # noqa: F401
from _bootstrap import REPO_ROOT

from ptcg.ml.bc.extract import ace_of, decks_of, episode_ok, iter_decisions
from ptcg.ml.bc.shards import ShardWriter
from ptcg.ml.features import FEATURE_VERSION, featurize
from ptcg.ml.vocab import VOCAB_VERSION, load_tables

EPISODES_DIR = REPO_ROOT / "data" / "episodes"
SHARDS_ROOT = REPO_ROOT / "data" / "bc_shards" / f"v{FEATURE_VERSION}"
RUN_DIR = REPO_ROOT / "runs" / "bc_shards"
STOP_FILE = RUN_DIR / "STOP"
FILTER_VERSION = 1

_TABLES = None


def _config_hash() -> str:
    key = f"feat{FEATURE_VERSION}-vocab{VOCAB_VERSION}-filt{FILTER_VERSION}"
    return hashlib.sha1(key.encode()).hexdigest()[:10]


def _init_worker():
    global _TABLES
    os.nice(19)
    _TABLES = load_tables(REPO_ROOT / "data/ml/cards.npz", REPO_ROOT / "data/ml/attacks.npz")


def _process_episode(path_str: str):
    """1 episode → (records, drops, episode_id, teams) を返す（ワーカー側で featurize 済み）。"""
    global _TABLES
    try:
        ep = json.load(open(path_str))
    except Exception:
        return None, {"json_error": 1}, 0, ("?", "?")
    info = ep.get("info") or {}
    teams = tuple(info.get("TeamNames") or ["?", "?"])[:2]
    try:
        ep_id = int(info.get("EpisodeId") or 0)  # トップレベル id は UUID 文字列なので使わない
    except Exception:
        ep_id = int(hashlib.sha1(str(ep.get("id")).encode()).hexdigest()[:15], 16)
    if not episode_ok(ep):
        return None, {"not_done": 1}, ep_id, teams
    decks = decks_of(ep)
    if decks[0] is None or decks[1] is None:
        return None, {"no_deck": 1}, ep_id, teams
    aces = [ace_of(decks[0], _TABLES), ace_of(decks[1], _TABLES)]
    rewards = ep.get("rewards")
    records = []
    drops: dict[str, int] = {}
    for kind, reason, i, payload, turn in iter_decisions(ep):
        if kind == "drop":
            drops[reason] = drops.get(reason, 0) + 1
            continue
        obs, label = payload
        feats = featurize(obs, _TABLES)
        if feats is None:
            drops["featurize_none"] = drops.get("featurize_none", 0) + 1
            continue
        sel_type = int((obs.get("select") or {}).get("type") or -1)
        mover_reward = int(rewards[i])
        meta_row = [ep_id, i, mover_reward, -1, aces[i], aces[1 - i], turn, sel_type]
        records.append((feats, label, meta_row, i))
    return records, drops, ep_id, teams


def process_date(date: str, workers: int, prune_raw: bool) -> bool:
    src = EPISODES_DIR / date
    if not (src / "_FETCHED").exists():
        print(f"[skip] {date}: _FETCHED なし")
        return True
    out_dir = SHARDS_ROOT / date
    done = out_dir / f"_DONE.{_config_hash()}"
    if done.exists():
        print(f"[skip] {date}: 変換済み")
        return True
    files = sorted(src.glob("*.json"), key=lambda p: -p.stat().st_size)  # 大→小で偏り回避
    if not files:
        print(f"[warn] {date}: JSONなし", file=sys.stderr)
        return True
    writer = ShardWriter(out_dir)
    stopped = False
    with Pool(processes=workers, initializer=_init_worker, maxtasksperchild=64) as pool:
        for records, drops, ep_id, teams in pool.imap_unordered(
            _process_episode, [str(p) for p in files], chunksize=4
        ):
            for r, c in (drops or {}).items():
                writer.drops[r] = writer.drops.get(r, 0) + c
            if records:
                writer.episodes.append(ep_id)
                for feats, label, meta_row, agent_idx in records:
                    meta_row[3] = writer.team_idx(str(teams[agent_idx]))
                    writer.add(feats, label, meta_row)
            if STOP_FILE.exists():
                stopped = True
                break
    writer.close(extra={"date": date, "stopped": stopped, "n_files": len(files)})
    if stopped:
        print(f"[stop] {date}: STOP 検出。_DONE は置かない（再実行で最初から作り直し）")
        return False
    done.write_text("ok")
    n_dec = sum(1 for _ in ())  # writer 側で計上済み（manifest 参照）
    print(f"[ok] {date}: shards={writer.n_shards} episodes={len(writer.episodes)} drops={writer.drops}")
    if prune_raw:
        for p in files:
            p.unlink(missing_ok=True)
        print(f"[prune] {date}: 生JSON削除")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dates", nargs="+", help="変換する日付（YYYY-MM-DD ...）")
    g.add_argument("--all", action="store_true", help="_FETCHED のある全日付")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--prune-raw", action="store_true", help="変換完了後に生JSONを削除")
    args = ap.parse_args()

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if STOP_FILE.exists():
        STOP_FILE.unlink()
    dates = args.dates or sorted(
        p.name for p in EPISODES_DIR.iterdir() if p.is_dir() and (p / "_FETCHED").exists()
    )
    for date in dates:
        if not process_date(date, args.workers, args.prune_raw):
            return 1  # STOP
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
