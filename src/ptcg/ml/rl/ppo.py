"""PPO 損失（純関数・torch）— clip + value clip + entropy + KL-to-anchor。

分布は sample.logp_entropy_torch と同一構成（options+仮想noopスロット、
opt_mask×gmask の合成マスク後 softmax）。behavior_logp / gmask は actor が保存した値を
そのまま使う（learner 側で再構成しない = train/serve 方策一致の要）。
"""

from __future__ import annotations

import torch

NEG = -1e9


def full_logits(opt_logits, noop_logit, opt_mask, gmask, noop_allow):
    mask = opt_mask * gmask
    return torch.cat([
        opt_logits + (1.0 - mask) * NEG,
        (noop_logit + (1.0 - noop_allow) * NEG)[:, None],
    ], dim=1)


def dist_stats(opt_logits, noop_logit, opt_mask, gmask, noop_allow, act):
    """(logp[B], entropy[B], logp_all[B,K+1]) を返す。"""
    fl = full_logits(opt_logits, noop_logit, opt_mask, gmask, noop_allow)
    logp_all = torch.log_softmax(fl, dim=1)
    logp = logp_all.gather(1, act[:, None].clamp(0, fl.shape[1] - 1)).squeeze(1)
    p = logp_all.exp()
    ent = -(p * logp_all.clamp(min=-1e8)).sum(dim=1)
    return logp, ent, logp_all


def kl_divergence(logp_all_new, logp_all_ref):
    """KL(new‖ref) をマスク後分布同士で計算（両者同形 [B,K+1]）。"""
    p_new = logp_all_new.exp()
    diff = (logp_all_new - logp_all_ref).clamp(min=-30, max=30)
    return (p_new * diff).sum(dim=1)


def ppo_losses(model_out, anchor_out, batch, cfg, kl_beta: float):
    """1ミニバッチの損失と診断値。

    batch: opt_mask/gmask/noop_allow/act/behavior_logp/adv/ret/value_old/trainable(float)
    cfg:   clip / entropy_coef / value_coef
    """
    opt_logits, noop_logit, value = model_out
    logp, ent, logp_all = dist_stats(
        opt_logits, noop_logit, batch["opt_mask"], batch["gmask"], batch["noop_allow"], batch["act"]
    )
    tr = batch["trainable"]
    n_tr = tr.sum().clamp(min=1.0)

    ratio = torch.exp((logp - batch["behavior_logp"]).clamp(min=-20, max=20))
    adv = batch["adv"]
    surr = torch.min(ratio * adv, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv)
    policy_loss = -(surr * tr).sum() / n_tr

    v_old = batch["value_old"]
    v_clip = v_old + (value - v_old).clamp(-cfg.clip, cfg.clip)
    v_loss = torch.max((value - batch["ret"]) ** 2, (v_clip - batch["ret"]) ** 2).mean()

    ent_loss = -(ent * tr).sum() / n_tr

    with torch.no_grad():
        a_logits, a_noop, _ = anchor_out
    _, _, logp_all_ref = dist_stats(
        a_logits, a_noop, batch["opt_mask"], batch["gmask"], batch["noop_allow"], batch["act"]
    )
    kl_anchor = (kl_divergence(logp_all, logp_all_ref) * tr).sum() / n_tr

    total = (policy_loss + cfg.value_coef * v_loss + cfg.entropy_coef * ent_loss
             + kl_beta * kl_anchor)
    with torch.no_grad():
        approx_kl_behavior = ((batch["behavior_logp"] - logp) * tr).sum() / n_tr
        clip_frac = (((ratio - 1).abs() > cfg.clip).float() * tr).sum() / n_tr
        diag = {
            "policy": float(policy_loss), "value": float(v_loss),
            "entropy": float((ent * tr).sum() / n_tr), "kl_anchor": float(kl_anchor),
            "approx_kl_behavior": float(approx_kl_behavior), "clip_frac": float(clip_frac),
            "ratio_dev": float(((ratio - 1.0).abs() * tr).sum() / n_tr),
        }
    return total, diag
