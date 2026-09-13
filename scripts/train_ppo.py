#!/usr/bin/env python
"""PPO learner ドライバ（actor は scripts/run_actors.py で別途起動する）。

使い方:
    poetry run python scripts/train_ppo.py --run runs/ppo_alakazam_v1 [--resume]
    touch runs/ppo_alakazam_v1/STOP   # learner+actor とも安全停止

GPU: CUDA_DEVICE_ORDER=PCI_BUS_ID / CUDA_VISIBLE_DEVICES=1（48GB GPU）。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import _bootstrap  # noqa: F401,E402
from _bootstrap import REPO_ROOT  # noqa: E402

from ptcg.ml.rl.learner import Learner  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    run = (REPO_ROOT / args.run) if not Path(args.run).is_absolute() else Path(args.run)
    assert (run / "config.json").exists(), "先に run_actors.py で run を初期化"
    stop = run / "STOP"
    if stop.exists() and not args.resume:
        stop.unlink()
    Learner(run, resume=args.resume).loop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
