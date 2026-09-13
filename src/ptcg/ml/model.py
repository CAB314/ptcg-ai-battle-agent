"""PTCGNet（torch）— 状態エンコーダ + option ポインタ + value head。

numpy 推論（np_forward.py）と 1:1 対応させるため、attention は nn.MultiheadAttention を
使わず Linear で手書きする（重みの命名を平坦にし export を単純化）。float32 固定。

入出力:
  forward(batch) -> (opt_logits [B,K], noop_logit [B], value [B])
  batch: g [B,72] / sid [B,56] / sf [B,56,40] / oid [B,K,2] / of [B,K,64]
         opt_mask [B,K] (1=実option) / state_mask [B,56] (1=実トークン)
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    d_model: int = 128
    n_heads: int = 4
    n_state_layers: int = 2
    n_option_layers: int = 2
    d_card_emb: int = 32     # v2: 静的ブロックが意味を担うので 64→32
    d_atk_emb: int = 32
    card_rows: int = 1536    # v2: 4096 は 96% が未参照だった
    atk_rows: int = 1600
    g_dim: int = 72
    f_s: int = 40
    f_o: int = 64
    d_card_static: int = 80  # vocab.D_CARD_STATIC（sid/oid から gather）
    d_atk_static: int = 31   # vocab.D_ATK_STATIC
    ff_mult: int = 2
    dropout: float = 0.1

    def to_dict(self):
        return asdict(self)


def _attn(q, k, v, n_heads, mask=None):
    """q:[B,Lq,D] k,v:[B,Lk,D] mask:[B,Lk] (1=有効)。手書き scaled dot-product。"""
    B, Lq, D = q.shape
    Lk = k.shape[1]
    h = n_heads
    dh = D // h
    q = q.view(B, Lq, h, dh).transpose(1, 2)  # [B,h,Lq,dh]
    k = k.view(B, Lk, h, dh).transpose(1, 2)
    v = v.view(B, Lk, h, dh).transpose(1, 2)
    scores = q @ k.transpose(-1, -2) / math.sqrt(dh)  # [B,h,Lq,Lk]
    if mask is not None:
        scores = scores + (1.0 - mask[:, None, None, :]) * -1e9
    w = torch.softmax(scores, dim=-1)
    out = (w @ v).transpose(1, 2).reshape(B, Lq, D)
    return out


class Block(nn.Module):
    """pre-LN の (self|cross)-attention + FF ブロック。"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        self.n_heads = cfg.n_heads
        self.ln_q = nn.LayerNorm(d)
        self.ln_kv = nn.LayerNorm(d)
        self.wq = nn.Linear(d, d)
        self.wk = nn.Linear(d, d)
        self.wv = nn.Linear(d, d)
        self.wo = nn.Linear(d, d)
        self.ln_ff = nn.LayerNorm(d)
        self.ff1 = nn.Linear(d, d * cfg.ff_mult)
        self.ff2 = nn.Linear(d * cfg.ff_mult, d)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, kv=None, kv_mask=None):
        src = x if kv is None else kv
        qn = self.ln_q(x)
        kn = self.ln_kv(src)
        a = _attn(self.wq(qn), self.wk(kn), self.wv(kn), self.n_heads, kv_mask)
        x = x + self.drop(self.wo(a))
        f = self.ff2(F.gelu(self.ff1(self.ln_ff(x)), approximate="tanh"))
        return x + self.drop(f)


class PTCGNet(nn.Module):
    """v2: カードの質的情報を **静的ブロックの gather** で直接与える。

    v1 では弱点・進化段階・効果テキストが学習ID埋め込みにしか入っておらず、実測でその
    埋め込みは初期値と区別がつかなかった（＝モデルはカードの意味を持っていなかった）。
    card_static/atk_static は学習しない buffer なので state_dict 経由で weights.npz に
    そのまま載り、numpy 推論側も同じ経路で読める。**未知カードにも効く**のが要点。

    static: {"card_static": [card_rows, d_card_static], "atk_static": [atk_rows, d_atk_static]}
    省略時はゼロで作る（ckpt ロードで上書きされる前提）。
    """

    def __init__(self, cfg: ModelConfig | None = None, static: dict | None = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        c = self.cfg
        d = c.d_model
        self.card_emb = nn.Embedding(c.card_rows, c.d_card_emb, padding_idx=0)
        self.atk_emb = nn.Embedding(c.atk_rows, c.d_atk_emb, padding_idx=0)
        cs = torch.zeros(c.card_rows, c.d_card_static)
        as_ = torch.zeros(c.atk_rows, c.d_atk_static)
        if static is not None:
            cs = torch.as_tensor(static["card_static"][: c.card_rows], dtype=torch.float32)
            as_ = torch.as_tensor(static["atk_static"][: c.atk_rows], dtype=torch.float32)
        self.register_buffer("card_static", cs)
        self.register_buffer("atk_static", as_)
        self.g_proj = nn.Linear(c.g_dim, d)
        self.s_proj = nn.Linear(c.d_card_emb + c.d_card_static + c.f_s, d)
        self.o_proj = nn.Linear(
            c.d_card_emb + c.d_atk_emb + c.d_card_static + c.d_atk_static + c.f_o, d
        )
        self.state_blocks = nn.ModuleList(Block(c) for _ in range(c.n_state_layers))
        self.option_blocks = nn.ModuleList(Block(c) for _ in range(c.n_option_layers))
        self.ln_out = nn.LayerNorm(d)
        self.logit_head = nn.Linear(d, 1)
        self.ln_state_out = nn.LayerNorm(d)
        self.noop_head = nn.Linear(d, 1)
        self.value_head = nn.Linear(d, 1)

    def encode_state(self, batch):
        sid = batch["sid"].long().clamp(0, self.cfg.card_rows - 1)
        s = torch.cat([self.card_emb(sid), self.card_static[sid], batch["sf"]], dim=-1)
        x = self.s_proj(s)
        # グローバルトークン(0番)にグローバル特徴を加算注入
        x = torch.cat([(x[:, :1] + self.g_proj(batch["g"])[:, None]), x[:, 1:]], dim=1)
        mask = batch["state_mask"]
        for blk in self.state_blocks:
            x = blk(x, kv_mask=mask)
        return x, mask

    def forward(self, batch):
        state, smask = self.encode_state(batch)
        oid = batch["oid"].long()
        ci = oid[..., 0].clamp(0, self.cfg.card_rows - 1)
        ai = oid[..., 1].clamp(0, self.cfg.atk_rows - 1)
        o = self.o_proj(torch.cat([
            self.card_emb(ci), self.atk_emb(ai),
            self.card_static[ci], self.atk_static[ai], batch["of"],
        ], dim=-1))
        for blk in self.option_blocks:
            o = blk(o, kv=state, kv_mask=smask)
        opt_logits = self.logit_head(self.ln_out(o)).squeeze(-1)  # [B,K]
        g_tok = self.ln_state_out(state[:, 0])
        noop_logit = self.noop_head(g_tok).squeeze(-1)            # [B]
        value = torch.tanh(self.value_head(g_tok)).squeeze(-1)    # [B]
        return opt_logits, noop_logit, value


def bc_loss(model_out, batch, label_smoothing=0.05, value_coef=0.25, type_weights=None):
    """BC 損失。単一選択= (option+noop) 上の softmax CE / 複数選択= per-option BCE。

    batch 追加キー:
      lab_multi [B,K] float(0/1) / lab_single [B] long（単一選択の正解 index、
      noop は K 位置=opt_logits の後ろに концат した仮想 index K）/ is_single [B] bool /
      noop_allowed [B] float / weight [B] float（教師・recency 重み）/ reward [B] float
    """
    opt_logits, noop_logit, value = model_out
    B, K = opt_logits.shape
    mask = batch["opt_mask"]  # [B,K] 1=実option
    neg = -1e9

    # ---- 単一選択（softmax CE, noop を仮想スロット K に連結）----
    logits_full = torch.cat([opt_logits + (1 - mask) * neg,
                             noop_logit[:, None] + (1 - batch["noop_allowed"][:, None]) * neg], dim=1)
    logp = F.log_softmax(logits_full, dim=1)
    is_single = batch["is_single"].float()
    tgt = batch["lab_single"].clamp(0, K)
    n_legal = mask.sum(1) + batch["noop_allowed"]
    ce = -logp.gather(1, tgt[:, None]).squeeze(1)
    if label_smoothing > 0:
        smooth = -(logp * torch.cat([mask, batch["noop_allowed"][:, None]], dim=1)).sum(1) / n_legal.clamp(min=1)
        ce = (1 - label_smoothing) * ce + label_smoothing * smooth
    # ---- 複数選択（BCE）----
    bce = F.binary_cross_entropy_with_logits(
        opt_logits, batch["lab_multi"], reduction="none"
    )
    bce = (bce * mask).sum(1) / mask.sum(1).clamp(min=1)

    w = batch.get("weight")
    w = torch.ones(B, device=opt_logits.device) if w is None else w
    if type_weights is not None:
        w = w * type_weights
    policy = (is_single * ce + (1 - is_single) * bce)
    policy = (policy * w).sum() / w.sum().clamp(min=1e-6)
    v_loss = F.mse_loss(value, batch["reward"])
    total = policy + value_coef * v_loss

    with torch.no_grad():
        pred = logits_full.argmax(1)
        acc = ((pred == tgt).float() * is_single * w).sum() / (is_single * w).sum().clamp(min=1e-6)
    return total, {"policy": float(policy), "value": float(v_loss), "acc_single": float(acc)}
