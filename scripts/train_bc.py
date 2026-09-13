#!/usr/bin/env python
"""BC 学習ループ（config 駆動・--resume・STOP・metrics.jsonl・--init-from）。

使い方:
    poetry run python scripts/train_bc.py --config configs/bc_v1.json --run runs/bc_alakazam_v1
    touch runs/bc_alakazam_v1/STOP   # 現バッチ完了後に保存して終了（共有機器の明け渡し）
    poetry run python scripts/train_bc.py --config ... --run ... --resume

GPU は CUDA_VISIBLE_DEVICES（未設定なら "1"=48GB GPU）を使用。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

# 共有機器: 48GB GPU（nvidia-smi の 1 番）のみ使用。another GPU×2 は Compute Prohibited。
# CUDAランタイム既定の FASTEST_FIRST 順だと番号が入れ替わるため PCI_BUS_ID を明示する。
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import _bootstrap  # noqa: F401,E402
from _bootstrap import REPO_ROOT  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ptcg.ml.bc.dataset import BCDataset, collate  # noqa: E402
from ptcg.ml.features import FEATURE_VERSION  # noqa: E402
from ptcg.ml.model import ModelConfig, PTCGNet, bc_loss  # noqa: E402
from ptcg.ml.vocab import load_tables  # noqa: E402


def _log(run_dir: Path, obj: dict):
    with open(run_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


def evaluate(model, ds, device, type_weights, batch_size=512, max_batches=200):
    if len(ds) == 0:
        return {}
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=2,
        collate_fn=lambda b: collate(b, type_weights),
    )
    model.eval()
    tot = n = acc = 0.0
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if bi >= max_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            loss, m = bc_loss(out, batch)
            bs = batch["g"].shape[0]
            tot += float(loss) * bs
            acc += m["acc_single"] * bs
            n += bs
    model.train()
    return {"loss": tot / max(n, 1), "acc": acc / max(n, 1), "n": int(n)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init-from", default=None, help="共有プリトレイン ckpt から初期化")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    run_dir = REPO_ROOT / args.run if not Path(args.run).is_absolute() else Path(args.run)
    run_dir.mkdir(parents=True, exist_ok=True)
    stop_file = run_dir / "STOP"
    if stop_file.exists():
        stop_file.unlink()
    ckpt_path = run_dir / "ckpt.pt"

    seed = int(cfg.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    teachers = json.loads((REPO_ROOT / cfg["teachers"]).read_text(encoding="utf-8"))
    shards_root = REPO_ROOT / "data" / "bc_shards" / f"v{FEATURE_VERSION}"
    type_weights = {int(k): float(v) for k, v in (cfg.get("type_weights") or {}).items()}

    # 3 split で同じ日のシャードを共有する（split ごとに読むと RAM が約2倍になる）
    shard_store: dict = {}
    ds_train = BCDataset(shards_root, teachers, split="train", shard_store=shard_store)
    ds_val = BCDataset(shards_root, teachers, split="val_iid", shard_store=shard_store)
    ds_tval = BCDataset(shards_root, teachers, split="val_temporal", shard_store=shard_store)
    print(f"train={len(ds_train):,} val_iid={len(ds_val):,} val_temporal={len(ds_tval):,}")
    if len(ds_train) == 0:
        print("[error] 教師データ 0 件（teachers/ace_ids を確認）")
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_cfg = ModelConfig(**(cfg.get("model") or {}))
    static = load_tables(REPO_ROOT / "data/ml/cards.npz", REPO_ROOT / "data/ml/attacks.npz")
    if static is None:
        print("[error] cards.npz/attacks.npz が読めない（scripts/build_vocab.py を実行）")
        return 1
    model = PTCGNet(model_cfg, static=static).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 3e-4)),
                            weight_decay=float(cfg.get("weight_decay", 0.01)))
    start_epoch = 0
    step = 0
    best_val = float("inf")
    if args.init_from:
        init = torch.load(args.init_from, map_location="cpu", weights_only=False)
        model.load_state_dict(init["model"])
        print(f"[init] {args.init_from} から初期化")
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_epoch = int(ck.get("epoch", 0))
        step = int(ck.get("step", 0))
        best_val = float(ck.get("best_val", best_val))
        print(f"[resume] epoch={start_epoch} step={step}")

    def save(epoch):
        tmp = run_dir / ".tmp_ckpt.pt"
        torch.save({
            "model": model.state_dict(), "opt": opt.state_dict(),
            "model_config": model_cfg.to_dict(), "epoch": epoch, "step": step,
            "best_val": best_val, "config": cfg,
        }, tmp)
        os.replace(tmp, ckpt_path)

    loader = torch.utils.data.DataLoader(
        ds_train, batch_size=int(cfg.get("batch_size", 512)), shuffle=True,
        num_workers=4, drop_last=True, collate_fn=lambda b: collate(b, type_weights),
    )
    label_smoothing = float(cfg.get("label_smoothing", 0.05))
    value_coef = float(cfg.get("value_coef", 0.25))
    epochs = int(cfg.get("epochs", 6))
    patience = int(cfg.get("early_stop_patience", 2))
    bad_epochs = 0

    model.train()
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            out = model(batch)
            loss, m = bc_loss(out, batch, label_smoothing=label_smoothing, value_coef=value_coef)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 100 == 0:
                _log(run_dir, {"step": step, "epoch": epoch, "loss": float(loss), **m})
            if step % 500 == 0 and stop_file.exists():
                save(epoch)
                print(f"[stop] step={step} 保存して終了（--resume で再開）")
                return 0
        val = evaluate(model, ds_val, device, type_weights)
        tval = evaluate(model, ds_tval, device, type_weights)
        _log(run_dir, {"epoch_end": epoch, "step": step, "val": val, "tval": tval,
                       "sec": round(time.time() - t0, 1)})
        print(f"[epoch {epoch}] val={val} tval={tval} ({time.time() - t0:.0f}s)")
        crit = (tval or val).get("loss", float("inf"))
        if crit < best_val:
            best_val = crit
            bad_epochs = 0
            save(epoch + 1)
            torch.save(torch.load(ckpt_path, map_location="cpu", weights_only=False), run_dir / "best.pt")
        else:
            bad_epochs += 1
            save(epoch + 1)
            if bad_epochs >= patience:
                print(f"[early-stop] epoch={epoch}")
                break
    print(f"[done] best_val={best_val:.4f} → export: poetry run python scripts/export_policy.py --ckpt {run_dir}/best.pt --out {run_dir}/export")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
