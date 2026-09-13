#!/usr/bin/env python
"""凍結 npz（state_dict 全キー）を learner の anchor_ckpt 用 .pt に変換する。

learner は anchor_ckpt に {"model": state_dict, "model_config": dict} を要求する
（learner.py __init__）。ガントレット best の npz は state_dict の全キー（buffer 含む）
を持つ（learner の ROLLBACK 復元と同一形式）ので、model_config を同 run の ckpt から
借りて包み直すだけでよい。

    python scripts/npz_to_anchor.py --npz runs/opp_frozen_v1/lopunny/weights.npz \
        --ref-ckpt runs/ppo_opp_lopunny_v1/ckpt.pt --out runs/anchors_v1/lopunny.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--ref-ckpt", required=True, help="model_config を借りる同構成の ckpt.pt")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    z = np.load(a.npz)
    sd = {k: torch.from_numpy(z[k].copy()) for k in z.files}
    ref = torch.load(a.ref_ckpt, map_location="cpu", weights_only=False)
    diff = set(ref["model"].keys()) ^ set(sd.keys())
    assert not diff, f"state_dict キー不一致: {sorted(diff)[:8]}"
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": sd, "model_config": ref["model_config"]}, out)
    print(f"[ok] {a.out} keys={len(sd)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
