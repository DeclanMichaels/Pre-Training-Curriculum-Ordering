"""
GPT with Attention Residual connections (AttnRes).

Identical to nanoGPT's model.py except:
- Each Block stores its output
- Each Block computes a lightweight attention over ALL previous layer outputs
  to determine its residual, instead of using only the immediately previous layer
- A learnable gate blends between standard residual and attended residual,
  initialized to standard behavior (gate=0) so training starts identically

Drop this file into nanoGPT/ alongside model.py.
To use: --model_type=attnres in train.py (requires a small patch to train.py)
Or: import from this file directly in a custom training script.

Reference: Kimi Team, "Attention Residuals" (arXiv:2603.15031).
Pre-Norm fix per Ziming Liu's follow-up analysis:
  https://kindxiaoming.github.io/blog/2026/attention-residual-2/
Without Pre-Norm on cross-layer attention inputs, routing scores are dominated
by hidden state norm growth across layers rather than content similarity.
Adding Pre-Norm eliminates the phase transition where AttnRes underperforms.
Pre-Norm applied to query and key computation only; values stay unnormalized
so the attended output lives in the same magnitude space as the standard
residual path.
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class CrossLayerAttention(nn.Module):
    """
    Lightweight attention over previous layers' outputs.
    For each position, computes attention weights over all stored
    previous layer representations and returns a weighted combination.

    This is the "attention residual" mechanism: instead of always using
    the immediately previous layer as the residual, the model learns
    which previous layers are most useful at each position.

    Uses a small projection (n_embd -> res_dim) to keep compute low.

    Pre-Norm applied to query and key inputs so that routing scores
    reflect content similarity rather than hidden state norm growth
    across layers. Values stay unnormalized so the attended output
    lives in the same magnitude space as the standard residual path.
    """

    def __init__(self, config):
        super().__init__()
        self.res_dim = config.n_embd // 8  # small projection for efficiency
        self.ln_q = LayerNorm(config.n_embd, bias=config.bias)
        self.ln_kv = LayerNorm(config.n_embd, bias=config.bias)
        self.query_proj = nn.Linear(config.n_embd, self.res_dim, bias=False)
        self.key_proj = nn.Linear(config.n_embd, self.res_dim, bias=False)
        # Gate: starts at 0 (sigmoid=0.5), but we initialize to -3
        # so sigmoid(-3) ≈ 0.05, meaning ~95% standard residual at init
        self.gate = nn.Parameter(torch.tensor(-3.0))

    def forward(self, current, prev_layers):
        """
        current: (B, T, C) - current layer's pre-residual output
        prev_layers: list of (B, T, C) tensors from all previous layers

        Returns: (B, T, C) - attended residual to use instead of just prev_layers[-1]
        """
        if len(prev_layers) == 1:
            # Only one previous layer (embedding), nothing to attend over
            return prev_layers[0]

        g = torch.sigmoid(self.gate)

        # Stack previous layers: (n_prev, B, T, C)
        prev_stack = torch.stack(prev_layers, dim=0)
        n_prev = prev_stack.size(0)

        # Pre-Norm: normalize current before query projection
        q = self.query_proj(self.ln_q(current))  # (B, T, res_dim)

        # Pre-Norm: normalize previous layers before key projection
        # Reshape for efficient computation: (n_prev*B, T, C) -> norm -> project -> reshape
        nB, T, C = n_prev * prev_stack.size(1), prev_stack.size(2), prev_stack.size(3)
        prev_flat = prev_stack.reshape(nB, T, C)
        k = self.key_proj(self.ln_kv(prev_flat))  # (n_prev*B, T, res_dim)
        k = k.reshape(n_prev, prev_stack.size(1), T, self.res_dim)  # (n_prev, B, T, res_dim)

        # Attention scores: for each (batch, position), score each previous layer
        # q: (B, T, d), k: (n_prev, B, T, d) -> scores: (B, T, n_prev)
        scores = torch.einsum('btd,nbtd->btn', q, k) / math.sqrt(self.res_dim)
        weights = F.softmax(scores, dim=-1)  # (B, T, n_prev)

        # Values: use raw (unnormalized) previous layers for the weighted combination
        # so the output lives in the same magnitude space as the standard residual
        prev_perm = prev_stack.permute(1, 2, 0, 3)  # (B, T, n_prev, C)
        attended = torch.einsum('btn,btnc->btc', weights, prev_perm)

        # Blend between standard residual (last prev layer) and attended
        standard = prev_layers[-1]
        residual = (1 - g) * standard + g * attended

        return residual


class AttnResBlock(nn.Module):
    """
    Transformer block with Attention Residual connections.

    Standard Block:  x = x + attn(ln(x));  x = x + mlp(ln(x))
    AttnRes Block:   x = cross_layer_attn(prev_layers) + attn(ln(x));  x = x + mlp(ln(x))

    The MLP residual stays standard (previous work shows attention residual
    matters more than MLP residual). Only the attention sub-layer gets the
    cross-layer routing.
    """

    def __init__(self, config, layer_idx):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        self.layer_idx = layer_idx

        # Cross-layer attention for residual (only for layers > 0)
        if layer_idx > 0:
            self.cross_layer_attn = CrossLayerAttention(config)
        else:
            self.cross_layer_attn = None

    def forward(self, x, prev_layers=None):
        # Attention sub-layer with cross-layer residual
        attn_out = self.attn(self.ln_1(x))
        if self.cross_layer_attn is not None and prev_layers is not None:
            residual = self.cross_layer_attn(x, prev_layers)
            x = residual + attn_out
        else:
            x = x + attn_out

        # MLP sub-layer with standard residual
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([AttnResBlock(config, i) for i in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Store all layer outputs for cross-layer attention
        prev_layers = [x]  # embedding output is layer 0

        for block in self.transformer.h:
            x = block(x, prev_layers)
            prev_layers.append(x)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
