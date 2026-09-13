#!/usr/bin/env python
"""ガントレット実行（単発 or ループ）。学習と独立に回す・GPU不使用・nice実行。

使い方:
    poetry run python scripts/ppo_gauntlet.py --run runs/ppo_alakazam_v1              # 単発quick
    poetry run python scripts/ppo_gauntlet.py --run runs/ppo_alakazam_v1 --loop 7200  # 2時間毎
    poetry run python scripts/ppo_gauntlet.py --run ... --scale 2.0                   # nightly(2倍戦数)

判定: composite が best より 3pp 低い評価×2連続 → ROLLBACK を書く（learner が復元）。
STOP ファイル（run 直下）でループ終了。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.nice(19)

import _bootstrap  # noqa: F401,E402
from _bootstrap import REPO_ROOT  # noqa: E402

from ptcg.ml.rl.config import PPOConfig  # noqa: E402
from ptcg.ml.rl.gauntlet import gauntlet_step  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--loop", type=int, default=0, help="秒間隔でループ（0=単発）")
    ap.add_argument("--scale", type=float, default=1.0, help="戦数スケール（nightly=2.0）")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    run = (REPO_ROOT / args.run) if not Path(args.run).is_absolute() else Path(args.run)
    cfg = PPOConfig.load(run / "config.json")
    # sbatch は supervisor / learner / gauntlet を同時に起動するが、policy/latest を作るのは
    # supervisor の ensure_run。初回だけ競合するので待つ（2026-08-02: ここで即死し、
    # 4.3時間の学習が **自動巻き戻しの安全網なし** で走ってしまった）。
    for _ in range(120):
        if (run / "policy" / "latest" / "weights.npz").exists():
            break
        if (run / "STOP").exists():
            print("[gauntlet] STOP 検出 → 起動を中止")
            return 0
        time.sleep(5)
    else:
        print("[gauntlet] policy/latest が 10 分経っても現れない → 中止", file=sys.stderr)
        return 1
    while True:
        t0 = time.time()
        try:
            res = gauntlet_step(run, cfg.deck_csv, games_scale=args.scale)
        except Exception as e:
            # 評価の失敗で学習まで巻き込まない。次の周期で再試行する。
            print(f"[gauntlet] 評価に失敗（次周期で再試行）: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            if not args.loop:
                return 1
            time.sleep(max(60, args.loop))
            continue
        line = " ".join(f"{k}={v['wr']:.3f}" for k, v in res["matchups"].items())
        print(f"[gauntlet] v{res['version']} composite={res['composite']:.4f} "
              f"best={'YES' if res.get('is_best') else 'no'} streak={res.get('bad_streak', 0)} | {line}",
              flush=True)
        if res.get("rollback_triggered"):
            print("[gauntlet] ROLLBACK をトリガした")
        if not args.loop:
            return 0
        if (run / "STOP").exists():
            print("[gauntlet] STOP 検出 → 終了")
            return 0
        wait = max(60, args.loop - (time.time() - t0))
        time.sleep(wait)


if __name__ == "__main__":
    raise SystemExit(main())
