"""行動サンプリング — numpy（actor 側）と torch（learner 側）の**同一分布**実装。

分布の定義（bc_loss / NpPolicy.choose と同型）:
  logits_full = [opt_logits + log(opt_mask) + log(extra_mask), noop_logit + log(noop_allow)]
  （log(0) は −1e9 加算で表現。仮想 noop スロット = index K）

actor は sample_single でサンプルし (act, logp, entropy) を軌跡に保存。
learner は logp_entropy_torch で同じ logp を再計算する。**両者のパリティは
tests/test_rl_sample.py で固定**（PPO ratio≈1 の生命線）。
"""

from __future__ import annotations

import numpy as np

NEG = -1e9


def masked_full_logits(opt_logits, extra_mask, noop_logit, noop_allow):
    """numpy: (K,)+マスク → (K+1,) フルロジット（最後が noop）。"""
    full = np.concatenate([
        opt_logits.astype(np.float64) + (1.0 - extra_mask) * NEG,
        np.array([noop_logit + (0.0 if noop_allow else NEG)], dtype=np.float64),
    ])
    return full


def _log_softmax(x):
    m = x.max()
    z = x - m
    lse = np.log(np.exp(z).sum())
    return z - lse


def sample_single(opt_logits, extra_mask, noop_logit, noop_allow, rng, temperature=1.0):
    """マスク後分布からサンプル → (act∈[0..K], logp, entropy)。K=noop。"""
    full = masked_full_logits(opt_logits, extra_mask, noop_logit, noop_allow)
    logp_all = _log_softmax(full / max(temperature, 1e-6))
    p = np.exp(logp_all)
    act = int(rng.choice(len(p), p=p / p.sum()))
    # logp/entropy は temperature=1 の学習分布で記録（温度はexploration用の振る舞い側）
    logp1 = _log_softmax(full)
    ent = float(-(np.exp(logp1) * np.where(logp1 < -1e8, 0.0, logp1)).sum())
    return act, float(logp1[act]), ent


def logp_entropy_torch(opt_logits, noop_logit, opt_mask, extra_mask, noop_allow, act):
    """torch: バッチ版 (B,K)/(B,)/(B,K)/(B,K)/(B,)/(B,) → (logp[B], entropy[B])。

    opt_mask=実option、extra_mask=ガードマスク。act==K は noop。
    """
    import torch

    mask = opt_mask * extra_mask
    full = torch.cat([
        opt_logits + (1.0 - mask) * NEG,
        (noop_logit + (1.0 - noop_allow) * NEG)[:, None],
    ], dim=1)
    logp_all = torch.log_softmax(full, dim=1)
    logp = logp_all.gather(1, act[:, None].clamp(0, full.shape[1] - 1)).squeeze(1)
    p = logp_all.exp()
    ent = -(p * logp_all.clamp(min=-1e8)).sum(dim=1)
    return logp, ent
