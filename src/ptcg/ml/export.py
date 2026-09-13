"""ckpt → weights.npz + config.json（export 時に torch/numpy パリティ検査を強制）。

パリティ検査: 実シャードから 100 決定を取り、PTCGNet と NpPolicy の logit を比較。
最大絶対差 < 1e-4 かつ argmax 一致 100% でなければ export 失敗（例外）。
TF32 は無効化して float32 同士で比較する。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .bc.shards import load_shard
from .features import FEATURE_VERSION
from .model import ModelConfig, PTCGNet
from .np_forward import NpPolicy
from .vocab import VOCAB_VERSION


def state_dict_to_npz(model: PTCGNet, path):
    arrs = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in model.state_dict().items()}
    np.savez_compressed(path, **arrs)


def _record_from_shard(z, i):
    o0, o1 = int(z["opt_off"][i]), int(z["opt_off"][i + 1])
    return {
        "g": z["g"][i].astype(np.float32),
        "sid": z["sid"][i].astype(np.int32),
        "sf": z["sf"][i].astype(np.float32),
        "oid": z["oid"][o0:o1].astype(np.int32),
        "of": z["of"][o0:o1].astype(np.float32),
        "k_min": int(z["kmin"][i]),
        "k_max": int(z["kmax"][i]),
        "noop_allowed": bool(z["noop"][i]),
    }


def parity_check(model: PTCGNet, np_policy: NpPolicy, shard_path, n=100) -> dict:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = model.float().cpu().eval()
    z = load_shard(shard_path)
    n = min(n, len(z["kmin"]))
    max_diff = 0.0
    n_argmax_ok = 0
    with torch.no_grad():
        for i in range(n):
            r = _record_from_shard(z, i)
            batch = {
                "g": torch.from_numpy(r["g"])[None],
                "sid": torch.from_numpy(r["sid"])[None],
                "sf": torch.from_numpy(r["sf"])[None],
                "oid": torch.from_numpy(r["oid"])[None],
                "of": torch.from_numpy(r["of"])[None],
                "opt_mask": torch.ones(1, len(r["oid"])),
                "state_mask": torch.from_numpy(1.0 - r["sf"][:, 32])[None],
            }
            tl, tn, tv = model(batch)
            out = np_policy.score(r)
            assert out is not None, "NpPolicy.score が None"
            nl, nn_, nv = out
            d = float(np.max(np.abs(tl[0].numpy() - nl)))
            d = max(d, abs(float(tn[0]) - nn_), abs(float(tv[0]) - nv))
            max_diff = max(max_diff, d)
            ta, na = int(tl[0].argmax()), int(np.argmax(nl))
            # ほぼ同値ロジットのタイは argmax 反転を許容（数値一致は max_diff 側で担保）
            if ta == na or float(tl[0, ta] - tl[0, na]) < 1e-4:
                n_argmax_ok += 1
    return {"n": n, "max_diff": max_diff, "argmax_match": n_argmax_ok / max(n, 1)}


def export_policy(ckpt_path, out_dir, shard_for_parity, extra_cfg: dict | None = None):
    """ckpt(torch) → out_dir/{weights.npz, config.json}。パリティ NG なら例外。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ckpt["model_config"])
    model = PTCGNet(cfg)
    model.load_state_dict(ckpt["model"])
    weights_path = out_dir / "weights.npz"
    state_dict_to_npz(model, weights_path)
    config = cfg.to_dict()
    config.update({
        "feature_version": FEATURE_VERSION,
        "vocab_version": VOCAB_VERSION,
        "train_step": int(ckpt.get("step", -1)),
    })
    if extra_cfg:
        config.update(extra_cfg)
    config_path = out_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=1), encoding="utf-8")
    pol = NpPolicy.load(weights_path, config_path, feature_version=FEATURE_VERSION)
    assert pol is not None, "NpPolicy.load 失敗"
    rep = parity_check(model, pol, shard_for_parity)
    if rep["max_diff"] >= 1e-4 or rep["argmax_match"] < 1.0:
        raise RuntimeError(f"パリティ検査 NG: {rep}")
    print(f"[ok] export {out_dir} parity={rep}")
    return rep
