"""EEG encoder backbones — architecture-inspired by LaBraM / EEGPT / BIOT.

These are from-scratch, right-sized encoders that borrow each paper's core
architectural idea (patchify per (channel, time) token, ViT-style backbone),
NOT their original pretrained weights or bespoke SSL recipes (LaBraM's VQ
codebook, EEGPT's masked+alignment objective, BIOT's linear attention). All
four backbones here are trained with the same eb_jepa two-view objective
(`EEGSSL` in `main.py`), so the encoder architecture is the only variable in
the comparison.

Shared contract (used by `EEGSSL` / `eval.py`):
    .represent(x: [B, C, T]) -> [B, D]
    .frames(x: [B, C, T])    -> [B, F, D]
    .out_dim, .n_frames

Windows are always a fixed length (`window_len`, e.g. 2000 = 10s @ 200Hz), so all
parameters (positional embeddings included) are sized once in `__init__` — never
created lazily inside `forward`/`frames`, which would otherwise dodge the optimizer
(built from `ssl.parameters()` right after construction, before any forward pass).
"""
import torch
import torch.nn as nn

from eb_jepa.nn_utils import init_module_weights


class _PatchTransformerBase(nn.Module):
    """Shared patchify + channel/positional embedding + transformer machinery."""

    def __init__(self, in_channels, window_len, patch_len, embed_dim, tr_depth,
                 n_heads, mlp_ratio, dropout, out_dim):
        super().__init__()
        assert window_len % patch_len == 0, \
            f"window_len={window_len} must be a multiple of patch_len={patch_len}"
        self.in_channels = in_channels
        self.patch_len = patch_len
        self.n_patches = window_len // patch_len
        self.embed_dim = embed_dim
        self.out_dim = out_dim

        self.channel_embed = nn.Embedding(in_channels, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(self.n_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=tr_depth)
        self.out_proj = nn.Linear(embed_dim, out_dim)

    def represent(self, x):
        return self.frames(x).mean(dim=1)


class LaBraMEncoder(_PatchTransformerBase):
    """ViT-style patch transformer: every (channel, time-patch) is its own token,
    flattened into one sequence. Mirrors LaBraM's patchify-per-(channel,time) + ViT
    backbone, without its VQ-codebook neural tokenizer / masked-code pretraining."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patch_embed = nn.Linear(self.patch_len, self.embed_dim)
        self.apply(init_module_weights)

    def frames(self, x):
        B, C, T = x.shape
        P = self.n_patches
        patches = x.reshape(B, C, P, self.patch_len)           # [B, C, P, patch_len]
        tok = self.patch_embed(patches)                        # [B, C, P, D]
        chan_ids = torch.arange(C, device=x.device)
        tok = tok + self.channel_embed(chan_ids).view(1, C, 1, -1)
        tok = tok + self.pos_embed.view(1, 1, P, -1)
        tok = tok.reshape(B, C * P, self.embed_dim)            # flat token sequence

        h = self.transformer(tok)                              # [B, C*P, D]
        self.n_frames = h.shape[1]
        return self.out_proj(h)                                # [B, C*P, out_dim]


class EEGPTEncoder(_PatchTransformerBase):
    """Hierarchical local (spatial) + global (temporal) encoder: a channel-mixing
    attention pool first collapses the 19 channel tokens within each time-patch into
    one summary token, then a temporal transformer runs over those (n_patches tokens,
    not n_channels*n_patches) — captures EEGPT's spatio-temporal token design."""

    def __init__(self, *args, n_heads=4, dropout=0.1, **kwargs):
        super().__init__(*args, n_heads=n_heads, dropout=dropout, **kwargs)
        self.patch_embed = nn.Linear(self.patch_len, self.embed_dim)
        self.chan_query = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.chan_pool = nn.MultiheadAttention(
            self.embed_dim, n_heads, dropout=dropout, batch_first=True)
        nn.init.trunc_normal_(self.chan_query, std=0.02)
        self.apply(init_module_weights)

    def frames(self, x):
        B, C, T = x.shape
        P = self.n_patches
        patches = x.reshape(B, C, P, self.patch_len)            # [B, C, P, patch_len]
        tok = self.patch_embed(patches)                         # [B, C, P, D]
        chan_ids = torch.arange(C, device=x.device)
        tok = tok + self.channel_embed(chan_ids).view(1, C, 1, -1)

        # spatial pool: collapse C channel-tokens into 1 summary token, per time-patch
        tok = tok.permute(0, 2, 1, 3).reshape(B * P, C, self.embed_dim)  # [B*P, C, D]
        query = self.chan_query.expand(B * P, 1, -1)
        summary, _ = self.chan_pool(query, tok, tok)                    # [B*P, 1, D]
        summary = summary.reshape(B, P, self.embed_dim)                 # [B, P, D]
        summary = summary + self.pos_embed.view(1, P, -1)

        h = self.transformer(summary)                          # [B, P, D]
        self.n_frames = h.shape[1]
        return self.out_proj(h)                                # [B, P, out_dim]


class BIOTEncoder(_PatchTransformerBase):
    """Same (channel, time-patch) tokenization as LaBraM-style, but each token is
    embedded from its frequency-domain magnitude spectrum (rfft) instead of raw
    time samples — keeps BIOT's frequency tokenization; uses standard full attention
    (simplification of BIOT's linear-attention transformer)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        n_freq = self.patch_len // 2 + 1
        self.patch_embed = nn.Linear(n_freq, self.embed_dim)
        self.apply(init_module_weights)

    def frames(self, x):
        B, C, T = x.shape
        P = self.n_patches
        patches = x.reshape(B, C, P, self.patch_len)            # [B, C, P, patch_len]
        spec = torch.fft.rfft(patches, dim=-1).abs()            # [B, C, P, patch_len//2+1]
        tok = self.patch_embed(spec)                            # [B, C, P, D]
        chan_ids = torch.arange(C, device=x.device)
        tok = tok + self.channel_embed(chan_ids).view(1, C, 1, -1)
        tok = tok + self.pos_embed.view(1, 1, P, -1)
        tok = tok.reshape(B, C * P, self.embed_dim)

        h = self.transformer(tok)
        self.n_frames = h.shape[1]
        return self.out_proj(h)
