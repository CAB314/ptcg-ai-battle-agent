#!/usr/bin/env bash
# 一時停止した PPO 学習（両デッキ）の再開。cd 不要・多重起動ガード付き。
#   bash scripts/resume_training.sh
# 停止側は: touch runs/ppo_*/STOP（learnerはiteration境界で保存終了）+ gauntletループkill
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

for r in ppo_alakazam_v1 ppo_trmewtwo_v1; do
  if pgrep -f "run_actors.py --run runs/$r" >/dev/null || pgrep -f "train_ppo.py --run runs/$r" >/dev/null; then
    echo "[skip] $r: 既に稼働中"
    continue
  fi
  rm -f "runs/$r/STOP" "runs/$r/ROLLBACK"
  nohup nice -n 19 poetry run python scripts/run_actors.py --run "runs/$r" \
    >> "runs/$r/supervisor.log" 2>&1 &
  # --resume 必須（付け忘れるとフレッシュ開始で ckpt を上書きする）
  nohup nice -n 10 env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
    poetry run python scripts/train_ppo.py --run "runs/$r" --resume \
    >> "runs/$r/learner_restart.log" 2>&1 &
  nohup nice -n 19 poetry run python scripts/ppo_gauntlet.py --run "runs/$r" --loop 7200 --workers 4 \
    >> "runs/$r/gauntlet_loop.log" 2>&1 &
  echo "[ok] $r: actors + learner(--resume) + gauntlet を再開"
done
echo "[done] $(date -Iseconds)"
