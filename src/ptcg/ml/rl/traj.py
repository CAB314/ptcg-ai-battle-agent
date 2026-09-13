"""軌跡パック（actor→learner のファイルキュー）と GAE。

パック = npz **非圧縮**（~6.5KB/決定、書込みは tmp→rename で原子的）。
ファイル名: t<unix_ms>_a<actor_id>_v<policy_version>.npz

決定ごとの列（BC シャードと同列 + RL 列）:
  g/sid/sf/opt_off/oid/of/kmin/kmax/noop  … 特徴量（f16/i16）
  gmask_off/gmask (f16)   … actor が使った最終ガードマスク（learner は再構成しない）
  act (i16)               … 選択（K=noop スロット）
  behavior_logp (f32) / value_pred (f32) / trainable (i8)
  ep_seat (i32)           … パック内エピソード通番×2+席（GAE 復元用）
  done (i8) / reward (f32)… エピソード末の決定に終端報酬を記録（他は0）
ヘッダ: policy_version, opp_id, deck_id, results[]
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np


class PackWriter:
    def __init__(self, out_dir: Path, actor_id: int, policy_version: int,
                 flush_games: int = 16, flush_secs: float = 60.0):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.actor_id = actor_id
        self.policy_version = policy_version
        self.flush_games = flush_games
        self.flush_secs = flush_secs
        self._t0 = time.time()
        self._games = 0
        self._rows: list[dict] = []
        self._results: list[int] = []
        self._meta: list[tuple[int, int]] = []  # (opp_id, deck_id)

    def add_step(self, feats: dict, gmask, noop_allow, act: int, logp: float,
                 value: float, trainable: bool, ep_seat: int):
        self._rows.append({
            "g": feats["g"].astype(np.float16),
            "sid": feats["sid"].astype(np.int16),
            "sf": feats["sf"].astype(np.float16),
            "oid": feats["oid"].astype(np.int16),
            "of": feats["of"].astype(np.float16),
            "kmin": np.int8(min(int(feats["k_min"]), 127)),
            "kmax": np.int8(min(int(feats["k_max"]), 127)),
            "noop": np.int8(1 if noop_allow else 0),
            "gmask": np.asarray(gmask, dtype=np.float16),
            "act": np.int16(act),
            "behavior_logp": np.float32(logp),
            "value_pred": np.float32(value),
            "trainable": np.int8(1 if trainable else 0),
            "ep_seat": np.int32(ep_seat),
            "done": np.int8(0),
            "reward": np.float32(0.0),
        })

    def end_episode(self, ep_seat: int, reward: float):
        """その席の最後の決定に終端報酬を記す。"""
        for r in reversed(self._rows):
            if int(r["ep_seat"]) == ep_seat:
                r["done"] = np.int8(1)
                r["reward"] = np.float32(reward)
                break

    def end_game(self, result: int, opp_id: int, deck_id: int):
        self._games += 1
        self._results.append(result)
        self._meta.append((opp_id, deck_id))
        if self._games >= self.flush_games or (time.time() - self._t0) > self.flush_secs:
            self.flush()

    def flush(self):
        if not self._rows:
            self._games = 0
            self._t0 = time.time()
            return None
        rows = self._rows
        n = len(rows)
        opt_off = np.zeros(n + 1, dtype=np.int64)
        opt_off[1:] = np.cumsum([len(r["oid"]) for r in rows])
        arrays = {
            "g": np.stack([r["g"] for r in rows]),
            "sid": np.stack([r["sid"] for r in rows]),
            "sf": np.stack([r["sf"] for r in rows]),
            "opt_off": opt_off,
            "oid": np.concatenate([r["oid"] for r in rows]),
            "of": np.concatenate([r["of"] for r in rows]),
            "gmask": np.concatenate([r["gmask"] for r in rows]).astype(np.float16),
            "kmin": np.array([r["kmin"] for r in rows], dtype=np.int8),
            "kmax": np.array([r["kmax"] for r in rows], dtype=np.int8),
            "noop": np.array([r["noop"] for r in rows], dtype=np.int8),
            "act": np.array([r["act"] for r in rows], dtype=np.int16),
            "behavior_logp": np.array([r["behavior_logp"] for r in rows], dtype=np.float32),
            "value_pred": np.array([r["value_pred"] for r in rows], dtype=np.float32),
            "trainable": np.array([r["trainable"] for r in rows], dtype=np.int8),
            "ep_seat": np.array([r["ep_seat"] for r in rows], dtype=np.int32),
            "done": np.array([r["done"] for r in rows], dtype=np.int8),
            "reward": np.array([r["reward"] for r in rows], dtype=np.float32),
            "policy_version": np.array([self.policy_version], dtype=np.int64),
            "results": np.array(self._results, dtype=np.int8),
            "opp_deck": np.array(self._meta, dtype=np.int32).reshape(-1, 2),
        }
        name = f"t{int(time.time() * 1000)}_a{self.actor_id}_v{self.policy_version}.npz"
        tmp = self.out_dir / f".tmp_{name}"
        with open(tmp, "wb") as f:
            np.savez(f, **arrays)
        os.replace(tmp, self.out_dir / name)
        self._rows = []
        self._results = []
        self._meta = []
        self._games = 0
        self._t0 = time.time()
        return name

    def set_version(self, v: int):
        self.policy_version = v


def load_pack(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def compute_gae(pack: dict, gamma: float = 1.0, lam: float = 0.95):
    """席ごと（ep_seat グループ）に GAE を計算し (advantage[N], returns[N]) を返す。

    中間報酬 0・終端のみ ±1 の設計。V(s_{t+1}) は同一席の次決定の value_pred、
    終端は 0。
    """
    n = len(pack["act"])
    adv = np.zeros(n, dtype=np.float32)
    ret = np.zeros(n, dtype=np.float32)
    ep_seat = pack["ep_seat"]
    order = np.argsort(ep_seat, kind="stable")
    values = pack["value_pred"].astype(np.float32)
    rewards = pack["reward"].astype(np.float32)
    dones = pack["done"].astype(bool)
    # 席ごとに時系列（パックは追記順なので stable sort 内で元順序が保たれる）
    for seat in np.unique(ep_seat):
        idx = order[ep_seat[order] == seat]
        last_adv = 0.0
        next_value = 0.0
        for i in reversed(idx):
            r = rewards[i] if dones[i] else 0.0
            nv = 0.0 if dones[i] else next_value
            delta = r + gamma * nv - values[i]
            last_adv = delta + gamma * lam * (0.0 if dones[i] else last_adv)
            adv[i] = last_adv
            ret[i] = adv[i] + values[i]
            next_value = values[i]
    return adv, ret
