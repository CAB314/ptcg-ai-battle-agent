"""BC 用 torch Dataset（教師フィルタ / 重み / episode 単位 split / collate）。

- 分割は episode_id ハッシュ（決定単位はリーク）。val_iid=ep_id%100<5。
  temporal-val=最新日付をまるごと holdout（メタドリフト下の本命指標）。
- 教師フィルタ: teachers config（チーム名→重み）× ace_ids（アーキタイプ）。
  勝者側 w=1.0 / 敗者側 loser_weight。recency exp(-Δ日/τ)。
- シャードは f16 のまま RAM に保持し、collate で f32/torch へ。
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import torch

# meta 列
M_EP, M_AGENT, M_REWARD, M_TEAM, M_MY_ACE, M_OPP_ACE, M_TURN, M_SELTYPE = range(8)


def _load_day(date_dir: Path, store: dict | None = None):
    """1日分のシャードを読む。store を渡すと日単位でキャッシュして再利用する。

    train / val_iid は同じ日の同じシャードを見る（違いは episode%100<5 の振り分けだけ）
    なので、split ごとに読み直すと RAM が2倍になる。47日で片側196GB＝二重だと392GBとなり
    実機503GBでは危険なため、呼び出し側で store を共有する。
    """
    key = str(date_dir)
    if store is not None and key in store:
        return store[key]
    teams = json.loads((date_dir / "teams.json").read_text(encoding="utf-8"))
    shards = []
    for sp in sorted(date_dir.glob("shard_*.npz")):
        z = np.load(sp)
        shards.append({k: z[k] for k in z.files})
    if store is not None:
        store[key] = (teams, shards)
    return teams, shards


class BCDataset(torch.utils.data.Dataset):
    def __init__(self, shards_root, teachers_cfg: dict, split: str = "train",
                 temporal_holdout: str | None = "latest", max_options: int = 64,
                 shard_store: dict | None = None):
        root = Path(shards_root)
        self.max_options = max_options
        # teams: {チーム名: 重み} または "*"（全チーム重み1.0 = 共有プリトレイン用）
        team_weights: dict[str, float] | str = teachers_cfg["teams"]
        ace_ids = set(teachers_cfg.get("ace_ids") or [])
        # 弱点対面のBCシード用: 相手アーキで絞る（例 opp_ace_ids:[756] = 対Kangaskhan戦のみ）
        opp_ace_ids = set(teachers_cfg.get("opp_ace_ids") or [])
        loser_w = float(teachers_cfg.get("loser_weight", 0.7))
        tau = float(teachers_cfg.get("recency_tau_days", 7.0))
        # 重みがこれ未満のサンプルは採らない（uniform サンプリング + loss 乗算方式のため、
        # 古い日の重み~0 サンプルはバッチ枠の無駄。日レベルで先に足切りして RAM も節約）
        min_w = float(teachers_cfg.get("min_weight", 0.02))

        dates = sorted(p.name for p in root.iterdir() if p.is_dir() and (p / "teams.json").exists())
        if not dates:
            raise FileNotFoundError(f"シャードが無い: {root}")
        latest = dates[-1]
        self.items: list[tuple[int, int, float]] = []  # (shard_gid, row, weight)
        self.shards: list[dict] = []
        latest_d = dt.date.fromisoformat(latest)
        for date in dates:
            is_temporal = temporal_holdout == "latest" and date == latest and len(dates) > 1
            if split == "val_temporal" and not is_temporal:
                continue
            if split in ("train", "val_iid") and is_temporal:
                continue
            age = (latest_d - dt.date.fromisoformat(date)).days
            rec_w = float(np.exp(-age / tau))
            # この日の最大到達重みが足切り未満 → 読み込み自体をスキップ
            max_team_w = 1.0 if team_weights == "*" else max(team_weights.values())
            if max_team_w * rec_w < min_w:
                continue
            teams, shards = _load_day(root / date, shard_store)
            if team_weights == "*":
                allowed = {idx: 1.0 for idx in teams.values()}
            else:
                allowed = {}
                for name, w in team_weights.items():
                    if name in teams:
                        allowed[teams[name]] = float(w)
            if not allowed:
                continue
            for sh in shards:
                gid = len(self.shards)
                self.shards.append(sh)
                meta = sh["meta"]
                for row in range(meta.shape[0]):
                    tw = allowed.get(int(meta[row, M_TEAM]))
                    if tw is None:
                        continue
                    if ace_ids and int(meta[row, M_MY_ACE]) not in ace_ids:
                        continue
                    if opp_ace_ids and int(meta[row, M_OPP_ACE]) not in opp_ace_ids:
                        continue
                    ep = int(meta[row, M_EP])
                    in_val = (ep % 100) < 5
                    if split == "train" and in_val:
                        continue
                    if split == "val_iid" and not in_val:
                        continue
                    w = tw * rec_w * (1.0 if int(meta[row, M_REWARD]) > 0 else loser_w)
                    if w < min_w:
                        continue
                    self.items.append((gid, row, w))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        gid, row, w = self.items[i]
        z = self.shards[gid]
        o0, o1 = int(z["opt_off"][row]), int(z["opt_off"][row + 1])
        l0, l1 = int(z["lab_off"][row]), int(z["lab_off"][row + 1])
        K = min(o1 - o0, self.max_options)
        label = [int(x) for x in z["lab"][l0:l1] if int(x) < K]
        return {
            "g": z["g"][row].astype(np.float32),
            "sid": z["sid"][row].astype(np.int64),
            "sf": z["sf"][row].astype(np.float32),
            "oid": z["oid"][o0:o0 + K].astype(np.int64),
            "of": z["of"][o0:o0 + K].astype(np.float32),
            "label": label,
            "noop": bool(z["noop"][row]),
            "reward": float(z["meta"][row, M_REWARD]),
            "sel_type": int(z["meta"][row, M_SELTYPE]),
            "weight": float(w),
        }


def collate(batch, type_weights: dict[int, float] | None = None):
    B = len(batch)
    Kmax = max(len(r["oid"]) for r in batch)
    S = batch[0]["sid"].shape[0]
    out = {
        "g": torch.zeros(B, batch[0]["g"].shape[0]),
        "sid": torch.zeros(B, S, dtype=torch.long),
        "sf": torch.zeros(B, S, batch[0]["sf"].shape[1]),
        "oid": torch.zeros(B, Kmax, 2, dtype=torch.long),
        "of": torch.zeros(B, Kmax, batch[0]["of"].shape[1]),
        "opt_mask": torch.zeros(B, Kmax),
        "state_mask": torch.zeros(B, S),
        "lab_multi": torch.zeros(B, Kmax),
        "lab_single": torch.zeros(B, dtype=torch.long),
        "is_single": torch.zeros(B, dtype=torch.bool),
        "noop_allowed": torch.zeros(B),
        "reward": torch.zeros(B),
        "weight": torch.ones(B),
    }
    for b, r in enumerate(batch):
        K = len(r["oid"])
        out["g"][b] = torch.from_numpy(r["g"])
        out["sid"][b] = torch.from_numpy(r["sid"])
        out["sf"][b] = torch.from_numpy(r["sf"])
        out["oid"][b, :K] = torch.from_numpy(r["oid"])
        out["of"][b, :K] = torch.from_numpy(r["of"])
        out["opt_mask"][b, :K] = 1.0
        out["state_mask"][b] = 1.0 - out["sf"][b, :, 32]  # SF_PAD
        out["noop_allowed"][b] = 1.0 if r["noop"] else 0.0
        out["reward"][b] = r["reward"]
        w = r["weight"]
        if type_weights:
            w *= float(type_weights.get(r["sel_type"], 1.0))
        out["weight"][b] = w
        lab = r["label"]
        if len(lab) == 1:
            out["is_single"][b] = True
            out["lab_single"][b] = lab[0]
        elif len(lab) == 0:
            out["is_single"][b] = True
            out["lab_single"][b] = Kmax  # 仮想 noop スロット
        else:
            for x in lab:
                out["lab_multi"][b, x] = 1.0
    return out
