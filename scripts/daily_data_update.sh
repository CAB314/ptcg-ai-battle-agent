#!/usr/bin/env bash
# 日次データ更新（完全自動・LLM判断なし・冪等）。cron から毎朝 09:30 JST に実行する想定。
#
#   1. 前日(UTC)の Daily Top Episodes を取得（取得済みなら _FETCHED でスキップ）
#   2. BC シャード変換（変換済みなら _DONE でスキップ。32ワーカー・nice19・GPU不使用）
#   3. メタシェアを機械集計して runs/daily_data/meta_shares_<date>.json に出力
#      （PPO learner はこのファイルをイテレーション境界で読むだけ＝書き込み競合なし）
#
# ログ: runs/daily_data/log_<date>.txt / crontab 例:
#   30 9 * * * ./scripts/daily_data_update.sh
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
# cron の最小 PATH には poetry(~/.local/bin) が無い（7/17-19 の fetch 失敗の原因）
export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
export KAGGLE_CONFIG_DIR="$REPO/.kaggle"
# 2026-08-08: NAS がステイルマウント化して 8/6-8/8 の更新が止まったため、
# data/episodes,bc_shards をローカル実ディレクトリへ切替（履歴は NAS と the cluster に残存）。
# fail-fast はディスク残量チェックに置換（1日あたり episodes+shards で数GB消費）。
AVAIL_GB=$(df --output=avail -BG "$REPO/data" | tail -1 | tr -dc 0-9)
[ "${AVAIL_GB:-0}" -ge 50 ] || { echo "[error] disk残量 ${AVAIL_GB}GB < 50GB のため中止"; exit 1; }
DATE="${1:-$(date -u -d 'yesterday' +%F)}"
LOGDIR="$REPO/runs/daily_data"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/log_${DATE}.txt"
exec >> "$LOG" 2>&1
echo "=== daily_data_update $DATE start $(date -Iseconds) ==="

nice -n 19 poetry run python scripts/fetch_episodes.py --date "$DATE" || { echo "[error] fetch failed"; exit 1; }
nice -n 19 poetry run python scripts/build_bc_shards.py --dates "$DATE" --workers 32 || { echo "[error] shards failed"; exit 1; }

nice -n 19 poetry run python - "$DATE" <<'PYEOF'
import json, sys, glob
from collections import Counter
sys.path.insert(0, "src")
import numpy as np
date = sys.argv[1]
arch = Counter(); wins = Counter()
for sp in sorted(glob.glob(f"data/bc_shards/v1/{date}/shard_*.npz")):
    z = np.load(sp)
    meta = z["meta"]  # [ep, agent, reward, team, my_ace, opp_ace, turn, sel_type]
    eps = {}
    for row in meta:
        eps[(int(row[0]), int(row[1]))] = (int(row[4]), int(row[2]))
    for (ep, ag), (ace, rw) in eps.items():
        arch[ace] += 1
        if rw > 0:
            wins[ace] += 1
total = sum(arch.values()) or 1
out = {
    "date": date,
    "total_sides": total,
    "shares": {str(a): {"share": n / total, "win": wins[a] / n, "n": n}
               for a, n in arch.most_common(15)},
}
path = f"runs/daily_data/meta_shares_{date}.json"
open(path, "w").write(json.dumps(out, indent=1))
print(f"[ok] {path}")
for a, n in arch.most_common(8):
    print(f"  ace={a}: share {n/total:5.1%} win {wins[a]/n:5.1%} (n={n})")
PYEOF
echo "=== done $(date -Iseconds) ==="
