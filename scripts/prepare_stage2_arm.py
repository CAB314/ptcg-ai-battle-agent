#!/usr/bin/env python
"""アブレーション第2段の1系列（arm）を用意する。

既存 PPO run の ckpt/policy/league をコピーし、**デッキだけ差し替えた**新 run を作る。
対照群（基準デッキ）も同じ手続きで作り、同じ iteration 数だけ追加学習することで
「微調整そのものによる改善」を交絡から排除する（差分の差分で判定）。

    python scripts/prepare_stage2_arm.py --src runs/ppo_grimmsnarl_v1 \
        --dst runs/s2_grim_control --candidates docs/data/ablation_candidates.json --arm base
    python scripts/prepare_stage2_arm.py --src runs/ppo_grimmsnarl_v1 \
        --dst runs/s2_grim_tsboss   --candidates docs/data/ablation_candidates.json --arm -TS_+Boss
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import _bootstrap  # noqa: F401
from _bootstrap import REPO_ROOT


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="コピー元 PPO run")
    ap.add_argument("--dst", required=True, help="作成する arm の run")
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--arm", required=True, help="候補名。'base' で基準デッキ（対照群）")
    ap.add_argument("--iters", type=int, default=150, help="この arm で回す追加 iteration 数")
    args = ap.parse_args()

    src = REPO_ROOT / args.src
    dst = REPO_ROOT / args.dst
    spec = json.loads((REPO_ROOT / args.candidates).read_text())
    if args.arm == "base":
        entry = spec["base"]
    else:
        entry = next((c for c in spec["candidates"] if c["name"] == args.arm), None)
        if entry is None:
            raise SystemExit(f"候補 '{args.arm}' が無い: {[c['name'] for c in spec['candidates']]}")

    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    # 学習再開に必要なものだけコピー（inbox は空で始める＝1回目を lag=0 にする）
    for name in ("ckpt.pt", "config.json", "league_state.json"):
        shutil.copy2(src / name, dst / name)
    shutil.copytree(src / "policy" / "latest", dst / "policy" / "latest")
    shutil.copytree(src / "policy" / "anchor", dst / "policy" / "anchor")
    (dst / "inbox").mkdir()
    (dst / "results").mkdir(exist_ok=True)
    (dst / "snapshots").mkdir(exist_ok=True)

    # この arm 専用のデッキ CSV を書き、config を差し替える
    deck_path = dst / "deck.csv"
    deck_path.write_text("\n".join(str(c) for c in entry["deck"]) + "\n")
    cfg = json.loads((dst / "config.json").read_text())
    cfg["run_name"] = dst.name
    cfg["deck_csv"] = str(deck_path.relative_to(REPO_ROOT))
    (dst / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1))

    # league の自デッキ参照も差し替え（ミラー相手が基準デッキのままだと比較が濁る）
    st = json.loads((dst / "league_state.json").read_text())
    st["my_deck"] = cfg["deck_csv"]
    for f in st.get("fixed", []):
        if f.get("deck") == json.loads((src / "config.json").read_text())["deck_csv"]:
            f["deck"] = cfg["deck_csv"]
    (dst / "league_state.json").write_text(json.dumps(st, ensure_ascii=False, indent=1))

    import torch
    ck = torch.load(dst / "ckpt.pt", map_location="cpu", weights_only=False)
    target = int(ck["iteration"]) + args.iters
    meta = {"arm": args.arm, "note": entry.get("note", ""), "src": args.src,
            "start_iter": int(ck["iteration"]), "target_iter": target,
            "deck": entry["deck"]}
    (dst / "stage2_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"[ok] {dst.name}: arm='{args.arm}' iter {meta['start_iter']}→{target} deck={cfg['deck_csv']}")
    print(f"     {entry.get('note','')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
