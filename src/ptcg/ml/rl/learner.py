"""PPO learner — inbox 消化 → GAE → 更新 → policy/latest 公開 → ckpt/metrics。

運用契約は train_bc.py と同型（STOP / --resume / metrics.jsonl / atomic ckpt）。
league_state.json の書き手は supervisor（EMA）なので、learner はスナップショット追加を
runs/<run>/snapshots/pending_*.json に置くだけ（単一書き手の維持）。

Phase 0（value較正）: value_head のみの param group を学習（trunk 凍結）。
EV ゲート通過で Phase 1（全損失）へ自動移行し、progress.json に記録する。
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from ..export import state_dict_to_npz
from ..model import ModelConfig, PTCGNet
from ..vocab import load_tables
from .config import PPOConfig
from .ppo import ppo_losses
from .traj import compute_gae, load_pack

REPO_ROOT = Path(__file__).resolve().parents[4]


def _pack_version(path: Path) -> int:
    try:
        return int(path.stem.split("_v")[-1])
    except Exception:
        return -1


class Learner:
    def __init__(self, run_dir: Path, resume: bool = False):
        self.run = Path(run_dir)
        self.cfg = PPOConfig.load(self.run / "config.json")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        anchor_ckpt = torch.load(REPO_ROOT / self.cfg.anchor_ckpt, map_location="cpu", weights_only=False)
        mc = ModelConfig(**anchor_ckpt["model_config"])
        mc.dropout = 0.0  # actor(numpy決定論)との分布一致のため dropout 無効
        # 静的テーブルは ckpt の buffer にも入っているが、明示的に渡して不一致を早期に落とす
        static = load_tables(REPO_ROOT / "data/ml/cards.npz", REPO_ROOT / "data/ml/attacks.npz")
        self.model = PTCGNet(mc, static=static).to(self.device)
        self.model.load_state_dict(anchor_ckpt["model"])
        self.anchor = PTCGNet(mc, static=static).to(self.device).eval()
        self.anchor.load_state_dict(anchor_ckpt["model"])
        for p in self.anchor.parameters():
            p.requires_grad_(False)

        self.opt_value = torch.optim.AdamW(self.model.value_head.parameters(), lr=1e-4, weight_decay=0.0)
        self.opt_full = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=0.0)
        self.iteration = 0
        self.version = 0
        self.kl_beta = self.cfg.kl_beta_init
        self.phase = 0
        self.best_marker = {"iter": 0}
        if resume and (self.run / "ckpt.pt").exists():
            ck = torch.load(self.run / "ckpt.pt", map_location="cpu", weights_only=False)
            self.model.load_state_dict(ck["model"])
            self.opt_value.load_state_dict(ck["opt_value"])
            self.opt_full.load_state_dict(ck["opt_full"])
            self.iteration = ck["iteration"]
            self.version = ck["version"]
            self.kl_beta = ck["kl_beta"]
            self.phase = ck["phase"]
            print(f"[resume] iter={self.iteration} version={self.version} phase={self.phase}")
        self.model.train()
        # 起動時に inbox を purge する。前回 run の未消化パックが残っていると 1回目の
        # iteration が version lag 1-2 のデータを掴み、epoch-0 の ratio チェック
        # （lag=0 前提・閾値1e-3）が誤発火して learner が即死する。
        # collect_iteration は lag<=max_version_lag を「新鮮」として通すので、
        # フィルタでは防げない。2026-07-30 the cluster job でこれを踏み、
        # learner 死亡後もアクターが4.5時間空回りして GPU 時間と 699GB を無駄にした。
        stale = list(self.run.glob("inbox/*.npz"))
        for p in stale:
            p.unlink(missing_ok=True)
        if stale:
            print(f"[learner] 起動時に残留パック {len(stale)} 個を破棄（1回目を lag=0 にするため）", flush=True)

    # ---- I/O ----
    def _log(self, obj):
        obj["t"] = round(time.time(), 1)
        with open(self.run / "metrics.jsonl", "a") as f:
            f.write(json.dumps(obj) + "\n")

    def publish(self):
        """policy/latest を atomic に更新（actor が hot-reload する）。"""
        self.version += 1
        pdir = self.run / "policy" / "latest"
        tmp = self.run / "policy" / ".tmp_latest"
        tmp.mkdir(parents=True, exist_ok=True)
        state_dict_to_npz(self.model.float().cpu(), tmp / "weights.npz")
        self.model.to(self.device)
        shutil.copy2(pdir / "config.json", tmp / "config.json")
        (tmp / "version.json").write_text(json.dumps({"version": self.version}))
        for f in tmp.iterdir():
            os.replace(f, pdir / f.name)
        tmp.rmdir()

    def save_ckpt(self):
        tmp = self.run / ".tmp_ckpt.pt"
        torch.save({
            "model": self.model.state_dict(),
            "opt_value": self.opt_value.state_dict(),
            "opt_full": self.opt_full.state_dict(),
            "iteration": self.iteration, "version": self.version,
            "kl_beta": self.kl_beta, "phase": self.phase,
            "model_config": self.model.cfg.to_dict(),
        }, tmp)
        os.replace(tmp, self.run / "ckpt.pt")

    def snapshot(self, tag: str, is_best: bool = False):
        sdir = self.run / "policy" / f"snap_{tag}"
        sdir.mkdir(parents=True, exist_ok=True)
        state_dict_to_npz(self.model.float().cpu(), sdir / "weights.npz")
        self.model.to(self.device)
        shutil.copy2(self.run / "policy" / "latest" / "config.json", sdir / "config.json")
        pend = self.run / "snapshots"
        pend.mkdir(exist_ok=True)
        (pend / f"pending_{tag}.json").write_text(json.dumps({
            "id": f"snap_{tag}", "kind": "np_snapshot", "path": str(sdir),
            "deck": None, "ema": 0.5, "games": 0, "is_best": is_best,
        }))

    # ---- データ ----
    def collect_iteration(self):
        """inbox から新鮮パックを iteration_decisions 分集めて連結・GAE 済みで返す。"""
        need = self.cfg.iteration_decisions
        rows = []
        got = 0
        t0 = time.time()
        while got < need:
            packs = sorted(self.run.glob("inbox/*.npz"), key=lambda p: p.name)
            fresh = [p for p in packs if _pack_version(p) >= self.version - self.cfg.max_version_lag]
            stale = [p for p in packs if p not in fresh]
            for p in stale:
                p.unlink(missing_ok=True)
            for p in fresh:
                try:
                    z = load_pack(p)
                except Exception:
                    p.unlink(missing_ok=True)
                    continue
                adv, ret = compute_gae(z, self.cfg.gamma, self.cfg.gae_lambda)
                z["adv"], z["ret"] = adv, ret
                rows.append(z)
                got += len(z["act"])
                p.unlink(missing_ok=True)
                if got >= need:
                    break
            if got < need:
                if (self.run / "STOP").exists():
                    return None
                time.sleep(2)
                if time.time() - t0 > 900:
                    print(f"[warn] 収集が遅い: {got}/{need}（actor 稼働を確認）")
                    t0 = time.time()
        return rows

    def _batch(self, rows, idx_pack, idx_row):
        """パック行列から learner ミニバッチ（padding 済み torch テンソル）を作る。"""
        recs = []
        for pi, ri in zip(idx_pack, idx_row):
            z = rows[pi]
            o0, o1 = int(z["opt_off"][ri]), int(z["opt_off"][ri + 1])
            recs.append((z, ri, o0, o1))
        B = len(recs)
        Kmax = max(o1 - o0 for _, _, o0, o1 in recs)
        S = recs[0][0]["sid"].shape[1]
        out = {
            "g": torch.zeros(B, recs[0][0]["g"].shape[1]),
            "sid": torch.zeros(B, S, dtype=torch.long),
            "sf": torch.zeros(B, S, recs[0][0]["sf"].shape[2]),
            "oid": torch.zeros(B, Kmax, 2, dtype=torch.long),
            "of": torch.zeros(B, Kmax, recs[0][0]["of"].shape[1]),
            "opt_mask": torch.zeros(B, Kmax),
            "gmask": torch.zeros(B, Kmax),
            "state_mask": torch.zeros(B, S),
            "noop_allow": torch.zeros(B),
            "act": torch.zeros(B, dtype=torch.long),
            "behavior_logp": torch.zeros(B),
            "adv": torch.zeros(B),
            "ret": torch.zeros(B),
            "value_old": torch.zeros(B),
            "trainable": torch.zeros(B),
        }
        for b, (z, ri, o0, o1) in enumerate(recs):
            K = o1 - o0
            out["g"][b] = torch.from_numpy(z["g"][ri].astype(np.float32))
            out["sid"][b] = torch.from_numpy(z["sid"][ri].astype(np.int64))
            sf = z["sf"][ri].astype(np.float32)
            out["sf"][b] = torch.from_numpy(sf)
            out["state_mask"][b] = torch.from_numpy(1.0 - sf[:, 32])
            out["oid"][b, :K] = torch.from_numpy(z["oid"][o0:o1].astype(np.int64))
            out["of"][b, :K] = torch.from_numpy(z["of"][o0:o1].astype(np.float32))
            out["opt_mask"][b, :K] = 1.0
            out["gmask"][b, :K] = torch.from_numpy(z["gmask"][o0:o1].astype(np.float32))
            out["noop_allow"][b] = float(z["noop"][ri])
            act = int(z["act"][ri])
            out["act"][b] = act if act < K else Kmax  # 仮想noopはバッチ内Kmaxへ
            out["behavior_logp"][b] = float(z["behavior_logp"][ri])
            out["adv"][b] = float(z["adv"][ri])
            out["ret"][b] = float(z["ret"][ri])
            out["value_old"][b] = float(z["value_pred"][ri])
            out["trainable"][b] = float(z["trainable"][ri])
        return {k: v.to(self.device) for k, v in out.items()}

    # ---- 学習 ----
    def run_iteration(self) -> bool:
        rows = self.collect_iteration()
        if rows is None:
            return False
        n = sum(len(z["act"]) for z in rows)
        idx_pack = np.concatenate([np.full(len(z["act"]), i, np.int32) for i, z in enumerate(rows)])
        idx_row = np.concatenate([np.arange(len(z["act"]), dtype=np.int32) for z in rows])
        # advantage 正規化（iteration 全体・trainable のみ）
        adv_all = np.concatenate([z["adv"] for z in rows])
        tr_all = np.concatenate([z["trainable"] for z in rows]).astype(bool)
        mu, sd = adv_all[tr_all].mean(), adv_all[tr_all].std() + 1e-8
        for z in rows:
            z["adv"] = (z["adv"] - mu) / sd

        perm = np.random.permutation(n)
        mb = self.cfg.minibatch
        diags = []
        stop_epochs = False
        phase0 = self.phase == 0
        for ep in range(1 if phase0 else self.cfg.epochs):
            if stop_epochs:
                break
            for s in range(0, n - mb + 1, mb):
                sel = perm[s:s + mb]
                batch = self._batch(rows, idx_pack[sel], idx_row[sel])
                model_out = self.model(batch)
                with torch.no_grad():
                    anchor_out = self.anchor(batch)
                loss, diag = ppo_losses(model_out, anchor_out, batch, self.cfg, self.kl_beta)
                if phase0:
                    loss = torch.nn.functional.mse_loss(model_out[2], batch["ret"])
                    diag = {"value": float(loss)}
                opt = self.opt_value if phase0 else self.opt_full
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                opt.step()
                diags.append(diag)
                if not phase0 and ep == 0 and s == 0 and not getattr(self, "_ratio_checked", False):
                    # Phase 1 最初のミニバッチ: 更新前なので ratio≈1 のはず（≠1なら方策不一致バグ）
                    self._ratio_checked = True
                    if diag["ratio_dev"] > 1e-3:
                        raise RuntimeError(f"train/serve 方策不一致の疑い: mean|ratio-1|={diag['ratio_dev']:.2e}")
                if not phase0 and diag.get("approx_kl_behavior", 0) > self.cfg.behavior_kl_stop:
                    stop_epochs = True
                    break
        # KL adaptive β
        if not phase0 and diags:
            kl = float(np.mean([d["kl_anchor"] for d in diags]))
            if kl > 1.5 * self.cfg.kl_target:
                self.kl_beta = min(self.kl_beta * 1.5, self.cfg.kl_beta_max)
            elif kl < self.cfg.kl_target / 1.5:
                self.kl_beta = max(self.kl_beta / 1.5, self.cfg.kl_beta_min)
        # Phase 0 → 1 の EV ゲート
        ev = None
        if phase0:
            ret_all = np.concatenate([z["ret"] for z in rows])
            v_all = np.concatenate([z["value_pred"] for z in rows])
            ev = 1.0 - float(np.var(ret_all - v_all)) / (float(np.var(ret_all)) + 1e-8)
            if (ev >= self.cfg.phase0_ev_gate and self.iteration >= 10) or \
               self.iteration >= self.cfg.phase0_iterations:
                self.phase = 1
                print(f"[phase] Phase 0 完了 (EV={ev:.3f}, iter={self.iteration}) → Phase 1")
        self.iteration += 1
        agg = {k: round(float(np.mean([d[k] for d in diags if k in d])), 5)
               for k in (diags[-1] if diags else {})}
        self._log({"iter": self.iteration, "phase": self.phase, "n": n, "beta": round(self.kl_beta, 3),
                   "ev": None if ev is None else round(ev, 4), **agg})
        self.publish()
        if self.iteration % self.cfg.ckpt_every_iters == 0:
            self.save_ckpt()
        if self.phase == 1 and self.iteration % self.cfg.snapshot_every_iters == 0:
            self.snapshot(f"i{self.iteration}")
        return True

    def _check_rollback(self):
        """ガントレットが置いた ROLLBACK を検知して best_gauntlet を復元する。"""
        rb = self.run / "ROLLBACK"
        if not rb.exists():
            return
        bdir = self.run / "policy" / "best_gauntlet"
        wz = bdir / "weights.npz"
        if wz.exists():
            z = np.load(wz)
            sd = {k: torch.from_numpy(z[k].copy()) for k in z.files}
            self.model.load_state_dict(sd)
            self.model.to(self.device)
            self.kl_beta = min(self.kl_beta * 2.0, self.cfg.kl_beta_max)
            for g in self.opt_full.param_groups:
                g["lr"] = max(g["lr"] * 0.5, 1e-6)
            self.publish()
            self.save_ckpt()
            self._log({"rollback": True, "iter": self.iteration, "beta": self.kl_beta})
            print(f"[rollback] best_gauntlet を復元（β={self.kl_beta:.2f}, lr×0.5）")
        rb.unlink(missing_ok=True)

    def loop(self):
        print(f"[learner] device={self.device} phase={self.phase} iter={self.iteration}")
        while not (self.run / "STOP").exists():
            self._check_rollback()
            if not self.run_iteration():
                break
        self.save_ckpt()
        print(f"[learner] 停止（iter={self.iteration}）。--resume で再開可")
