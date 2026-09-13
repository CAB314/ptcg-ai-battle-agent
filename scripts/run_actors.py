#!/usr/bin/env python
"""actor supervisor — actor_target ファイルを追従して子プロセス数を維持する。

使い方:
    poetry run python scripts/run_actors.py --run runs/ppo_alakazam_v1 --target 16
    echo 64 > runs/ppo_alakazam_v1/actor_target   # 実行中に増減（昼24/夜64の縮退運転）
    touch runs/ppo_alakazam_v1/STOP               # 全actorがゲーム境界で終了→supervisor終了

初回起動時に runs/<run>/ を初期化する:
  config.json（PPOConfig）/ policy/latest（BC best.pt から export）/ league_state.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401
from _bootstrap import REPO_ROOT

from ptcg.ml.rl.config import PPOConfig
from ptcg.ml.rl.league import default_state, load_state, sample_opponent, save_state

import numpy as np


def ensure_run(run: Path, cfg_path: str | None) -> PPOConfig:
    run.mkdir(parents=True, exist_ok=True)
    (run / "inbox").mkdir(exist_ok=True)
    cfg_file = run / "config.json"
    if not cfg_file.exists():
        src = Path(cfg_path) if cfg_path else None
        if src and src.exists():
            shutil.copy2(src, cfg_file)
        else:
            cfg_file.write_text(json.dumps(PPOConfig().__dict__, ensure_ascii=False, indent=1))
    cfg = PPOConfig.load(cfg_file)
    policy_dir = run / "policy" / "latest"
    if not (policy_dir / "weights.npz").exists():
        print("[init] BC ckpt から policy/latest を export ...")
        from ptcg.ml.export import export_policy

        export_policy(REPO_ROOT / cfg.anchor_ckpt, policy_dir,
                      sorted((REPO_ROOT / "data/bc_shards/v1").glob("*/shard_0000.npz"))[-1])
        (policy_dir / "version.json").write_text(json.dumps({"version": 0}))
    lg = run / "league_state.json"
    if not lg.exists():
        anchor_export = str((run / "policy" / "anchor").resolve())
        # 凍結BC = 初期 policy/latest のコピー（学習が進んでも不変）
        shutil.copytree(policy_dir, run / "policy" / "anchor", dirs_exist_ok=True)
        save_state(default_state(str(REPO_ROOT), anchor_export, cfg.deck_csv), lg)
    return cfg


# ace(カードID) → 代理デッキCSV。CSVの無い新興ace（例: Mega Lopunny 849）はスキップし、
# マップ済みaceのシェアで正規化する。新aceが5%超えたらCSVを採掘してここに足す。
ACE_TO_DECK = {
    648: "decks/meta_marnie_s_grimmsnarl_ex.csv",
    66: "decks/meta_alakazam.csv",
    431: "decks/meta_team_rocket_s_mewtwo_ex.csv",
    381: "decks/meta_cynthia_s_garchomp_ex.csv",
    756: "decks/meta_mega_kangaskhan_ex.csv",
    121: "decks/meta_dragapult_ex.csv",
    1031: "decks/meta_mega_starmie_ex.csv",
    # 2026-08-02 追加。未登録だったため refresh_meta が実メタから落としていた3種。
    # Thwackey 5.1% / Lopunny 4.7% / Teal Ogerpon 3.3%（7/31実測）。
    # 特に Ogerpon は greedy 操縦でも我々から 51-59% 取る実測済みの弱点。
    90: "decks/meta_thwackey.csv",
    849: "decks/meta_mega_lopunny_ex.csv",
    96: "decks/meta_teal_mask_ogerpon_ex.csv",
    # 2026-08-07 追加。両方とも実リプレイから採掘したネットデッキ最頻リスト。
    # Lucario は上位帯 3.0-3.6%（8/3-8/4実測）。Archaludon は上位帯 0.1% だが
    # 中位帯（700-900）で急拡散中（我々の対戦の11%・19チーム・うち10チーム同一リスト）。
    678: "decks/meta_mega_lucario_ex.csv",
    190: "decks/meta_archaludon_ex.csv",
}


def refresh_meta(state: dict, shares_file: Path) -> bool:
    """meta_shares_<date>.json から decks.meta を実メタ比例で再構成（変化があれば True）。"""
    shares = json.loads(shares_file.read_text())["shares"]
    mapped = []
    for ace, v in shares.items():
        deck = ACE_TO_DECK.get(int(ace))
        if deck and (REPO_ROOT / deck).exists():
            mapped.append((deck, float(v["share"])))
    tot = sum(w for _, w in mapped)
    if tot <= 0:
        return False
    new = [[d, round(w / tot, 4)] for d, w in sorted(mapped, key=lambda x: -x[1])]
    if new == state["decks"]["meta"]:
        return False
    state["decks"]["meta"] = new
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", default="configs/ppo_alakazam_v1.json")
    ap.add_argument("--target", type=int, default=None, help="初期 actor 数（actor_target を上書き）")
    ap.add_argument("--max-minutes", type=float, default=None, help="計測用: 指定分数で終了")
    args = ap.parse_args()
    run = (REPO_ROOT / args.run) if not Path(args.run).is_absolute() else Path(args.run)
    cfg = ensure_run(run, args.config)
    tfile = run / "actor_target"
    if args.target is not None or not tfile.exists():
        tfile.write_text(str(args.target if args.target is not None else 8))
    stop = run / "STOP"
    if stop.exists():
        stop.unlink()
    uniform_decks = sorted(str(p.relative_to(REPO_ROOT)) for p in (REPO_ROOT / "decks").glob("*.csv"))
    rng = np.random.default_rng(int(time.time()) % 2**32)

    procs: dict[int, subprocess.Popen] = {}
    next_id = 0
    t0 = time.time()
    last_report = t0
    packs_seen = 0
    last_meta_file = ""

    def spawn(aid: int):
        state = load_state(run / "league_state.json")
        spec = sample_opponent(state, uniform_decks, rng)
        env = dict(**__import__("os").environ)
        env["OMP_NUM_THREADS"] = "1"
        env["PYTHONPATH"] = str(REPO_ROOT / "src")
        env.pop("PTCG_AGENT_DIR", None)
        p = subprocess.Popen(
            [sys.executable, "-m", "ptcg.ml.rl.actor", str(run), str(aid), json.dumps(spec)],
            cwd=str(REPO_ROOT), env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        procs[aid] = p

    print(f"[supervisor] run={run} target={tfile.read_text().strip()}")
    try:
        while True:
            if stop.exists():
                print("[supervisor] STOP 検出 → respawn 停止、既存 actor の終了を待つ")
                break
            try:
                target = max(0, int(tfile.read_text().strip()))
            except Exception:
                target = 8
            # 死んだ子を回収 + 戦績ファイルをリーグ EMA に反映（league_state の書き手は supervisor のみ）
            for aid, p in list(procs.items()):
                if p.poll() is not None:
                    del procs[aid]
            res_files = sorted((run / "results").glob("r*.json")) if (run / "results").exists() else []
            pend = sorted((run / "snapshots").glob("pending_*.json")) if (run / "snapshots").exists() else []
            if res_files or pend:
                from ptcg.ml.rl.league import update_ema

                state = load_state(run / "league_state.json")
                for rf in res_files:
                    try:
                        r = json.loads(rf.read_text())
                        update_ema(state, r["opp"], r["wins"], r["games"])
                    except Exception:
                        pass
                    rf.unlink(missing_ok=True)
                # learner からのスナップショット追加要求をマージ（league_state の単一書き手を維持）
                for pf in pend:
                    try:
                        snap = json.loads(pf.read_text())
                        state["snapshots"].append(snap)
                        best_ids = {s["id"] for s in state["snapshots"] if s.get("is_best")}
                        while len(state["snapshots"]) > 12:
                            for i, s in enumerate(state["snapshots"]):
                                if s["id"] not in best_ids:
                                    state["snapshots"].pop(i)
                                    break
                            else:
                                state["snapshots"].pop(0)
                    except Exception:
                        pass
                    pf.unlink(missing_ok=True)
                save_state(state, run / "league_state.json")
            # 最新 meta_shares を自動反映（日次。デッキ分布の陳腐化防止・決定的処理）
            mfiles = sorted((REPO_ROOT / "runs" / "daily_data").glob("meta_shares_*.json"))
            if mfiles and mfiles[-1].name != last_meta_file:
                try:
                    state = load_state(run / "league_state.json")
                    if refresh_meta(state, mfiles[-1]):
                        save_state(state, run / "league_state.json")
                        print(f"[supervisor] metaデッキ分布を更新 <- {mfiles[-1].name}", flush=True)
                except Exception as e:
                    print(f"[supervisor] meta更新失敗（続行）: {e}", flush=True)
                last_meta_file = mfiles[-1].name
            # 追従
            while len(procs) < target:
                spawn(next_id)
                next_id += 1
            while len(procs) > target:
                aid, p = next(iter(procs.items()))
                p.terminate()
                del procs[aid]
            if time.time() - last_report > 30:
                n_packs = len(list((run / "inbox").glob("*.npz")))
                dp = n_packs - packs_seen
                packs_seen = n_packs
                print(f"[supervisor] actors={len(procs)} inbox={n_packs} (+{dp}/30s) elapsed={time.time()-t0:.0f}s",
                      flush=True)
                last_report = time.time()
            if args.max_minutes and (time.time() - t0) > args.max_minutes * 60:
                print("[supervisor] max-minutes 到達 → 終了")
                stop.touch()
                break
            time.sleep(5)
    finally:
        deadline = time.time() + 120
        for p in procs.values():
            if time.time() > deadline:
                p.kill()
            else:
                try:
                    p.wait(timeout=max(1, deadline - time.time()))
                except subprocess.TimeoutExpired:
                    p.kill()
    print("[supervisor] 終了")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
