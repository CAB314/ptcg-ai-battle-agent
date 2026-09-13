#!/usr/bin/env python
"""ckpt → weights.npz + config.json（torch/np パリティ検査込み）。

使い方:
    poetry run python scripts/export_policy.py --ckpt runs/bc_alakazam_v1/best.pt \
        --out runs/bc_alakazam_v1/export
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from _bootstrap import REPO_ROOT

from ptcg.ml.export import export_policy
from ptcg.ml.features import FEATURE_VERSION


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parity-shard", default=None)
    args = ap.parse_args()
    shard = args.parity_shard
    if shard is None:
        root = REPO_ROOT / "data" / "bc_shards" / f"v{FEATURE_VERSION}"
        cands = sorted(root.glob("*/shard_0000.npz"))
        assert cands, "パリティ用シャードが見つからない"
        shard = cands[-1]
    export_policy(args.ckpt, args.out, shard)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
