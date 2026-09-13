"""特徴量化済み決定レコードのシャード書き込み/読み込み（ragged npz + manifest）。

シャード構成（data/bc_shards/v<FEATURE_VERSION>/<date>/）:
  shard_XXXX.npz  … 下記配列（~SHARD_SIZE 決定/個）
  teams.json      … チーム名 → team_idx（日内ローカル）
  manifest.jsonl  … シャードごとの統計（n, episodes, drops）
  _DONE.<config_hash>

配列（N=決定数, sumK=option総数, sumL=ラベル総数）:
  g (N,72) f16 / sid (N,56) i16 / sf (N,56,40) f16
  opt_off (N+1,) i64 / oid (sumK,2) i16 / of (sumK,64) f16
  lab_off (N+1,) i32 / lab (sumL,) i16
  kmin,kmax,noop (N,) i8
  meta (N,8) i64: [episode_id, agent_idx, mover_reward(+1/-1), team_idx,
                   my_ace, opp_ace, turn, sel_type]
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

SHARD_SIZE = 4096
META_COLS = 8


class ShardWriter:
    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.teams: dict[str, int] = {}
        self.n_shards = 0
        self.drops: dict[str, int] = {}
        self.episodes: list[int] = []
        self._reset_buf()
        self._manifest = open(self.out_dir / "manifest.jsonl", "a", encoding="utf-8")

    def _reset_buf(self):
        self.buf = {k: [] for k in ("g", "sid", "sf", "oid", "of", "lab", "kmin", "kmax", "noop", "meta")}
        self.opt_lens: list[int] = []
        self.lab_lens: list[int] = []

    def team_idx(self, name: str) -> int:
        if name not in self.teams:
            self.teams[name] = len(self.teams)
        return self.teams[name]

    def add_drop(self, reason: str):
        self.drops[reason] = self.drops.get(reason, 0) + 1

    def add(self, feats: dict, label, meta_row):
        self.buf["g"].append(feats["g"].astype(np.float16))
        self.buf["sid"].append(feats["sid"].astype(np.int16))
        self.buf["sf"].append(feats["sf"].astype(np.float16))
        self.buf["oid"].append(feats["oid"].astype(np.int16))
        self.buf["of"].append(feats["of"].astype(np.float16))
        self.buf["lab"].append(np.asarray(label, dtype=np.int16))
        self.buf["kmin"].append(np.int8(min(feats["k_min"], 127)))
        self.buf["kmax"].append(np.int8(min(feats["k_max"], 127)))
        self.buf["noop"].append(np.int8(1 if feats["noop_allowed"] else 0))
        self.buf["meta"].append(np.asarray(meta_row, dtype=np.int64))
        self.opt_lens.append(len(feats["oid"]))
        self.lab_lens.append(len(label))
        if len(self.opt_lens) >= SHARD_SIZE:
            self.flush()

    def flush(self):
        n = len(self.opt_lens)
        if n == 0:
            return
        path = self.out_dir / f"shard_{self.n_shards:04d}.npz"
        tmp = self.out_dir / f".tmp_shard_{self.n_shards:04d}.npz"
        opt_off = np.zeros(n + 1, dtype=np.int64)
        opt_off[1:] = np.cumsum(self.opt_lens)
        lab_off = np.zeros(n + 1, dtype=np.int32)
        lab_off[1:] = np.cumsum(self.lab_lens)
        np.savez_compressed(
            tmp,
            g=np.stack(self.buf["g"]),
            sid=np.stack(self.buf["sid"]),
            sf=np.stack(self.buf["sf"]),
            opt_off=opt_off,
            oid=np.concatenate(self.buf["oid"]) if opt_off[-1] else np.zeros((0, 2), np.int16),
            of=np.concatenate(self.buf["of"]) if opt_off[-1] else np.zeros((0, 64), np.float16),
            lab_off=lab_off,
            lab=np.concatenate(self.buf["lab"]) if lab_off[-1] else np.zeros((0,), np.int16),
            kmin=np.asarray(self.buf["kmin"], dtype=np.int8),
            kmax=np.asarray(self.buf["kmax"], dtype=np.int8),
            noop=np.asarray(self.buf["noop"], dtype=np.int8),
            meta=np.stack(self.buf["meta"]),
        )
        os.replace(tmp, path)
        self._manifest.write(json.dumps({"shard": path.name, "n": n}) + "\n")
        self._manifest.flush()
        self.n_shards += 1
        self._reset_buf()

    def close(self, extra: dict | None = None):
        self.flush()
        (self.out_dir / "teams.json").write_text(
            json.dumps(self.teams, ensure_ascii=False, indent=0), encoding="utf-8"
        )
        summary = {"summary": True, "n_shards": self.n_shards, "drops": self.drops,
                   "n_episodes": len(self.episodes)}
        if extra:
            summary.update(extra)
        self._manifest.write(json.dumps(summary) + "\n")
        self._manifest.close()


def load_shard(path):
    """shard npz → dict（memmap ではなく即ロード。f16 のまま返す）。"""
    z = np.load(path)
    return {k: z[k] for k in z.files}


def iter_shard_paths(root: Path):
    root = Path(root)
    for date_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for sp in sorted(date_dir.glob("shard_*.npz")):
            yield date_dir.name, sp
