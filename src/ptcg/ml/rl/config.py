"""PPO 設定（configs/ppo_*.json から読む）。deck/teachers/anchor も config 化して
2枠目のデッキへは JSON 差し替えだけで流用できるようにする。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PPOConfig:
    # 実験の同一性
    run_name: str = "ppo_alakazam_v1"
    anchor_ckpt: str = "runs/bc_alakazam_v1/best.pt"   # KLアンカー + warm start 元
    deck_csv: str = "agents/alakazam_bc/deck.csv"      # 自分のデッキ（固定）
    # 経験生成
    iteration_decisions: int = 49_152    # 1 update の新鮮決定数
    max_version_lag: int = 2             # これより古い behavior version のパックは破棄
    actor_games_per_proc: int = 25       # リーク対策の自然終了
    sample_temperature: float = 1.0
    # PPO
    minibatch: int = 4096
    epochs: int = 2
    lr: float = 7e-5
    clip: float = 0.2
    entropy_coef: float = 0.005
    value_coef: float = 0.5
    kl_target: float = 0.03
    kl_beta_init: float = 1.0
    kl_beta_min: float = 0.05
    kl_beta_max: float = 20.0
    behavior_kl_stop: float = 0.03       # epoch内 approx_kl 早期打切り
    gamma: float = 1.0
    gae_lambda: float = 0.95
    grad_clip: float = 1.0
    # Phase 0（value較正）
    phase0_iterations: int = 40
    phase0_ev_gate: float = 0.25
    # リーグ
    league_latest: float = 0.5
    league_snapshot: float = 0.3
    league_fixed: float = 0.2
    snapshot_every_iters: int = 150
    snapshot_pool_max: int = 12
    pfsp_floor: float = 0.05
    ema_alpha: float = 0.02
    # 運用
    ckpt_every_iters: int = 30
    metrics_every_iters: int = 1
    draw_reward: float = 0.0
    # 相手デッキ分布（league.py が使用）: {"meta": 0.55, "weak": 0.25, "uniform": 0.20}
    deck_mix: dict = field(default_factory=lambda: {"meta": 0.55, "weak": 0.25, "uniform": 0.20})

    @staticmethod
    def load(path: str | Path) -> "PPOConfig":
        cfg = json.loads(Path(path).read_text(encoding="utf-8"))
        return PPOConfig(**cfg)
