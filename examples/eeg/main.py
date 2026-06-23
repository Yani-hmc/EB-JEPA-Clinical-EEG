"""EEG — SSL pretraining entrypoint (self-supervised representation learning).

Research question: can two-view invariance learning on unlabeled EEG learn
features that linearly separate *normal vs abnormal* recordings, generalizing
to held-out (patient-disjoint) subjects?

The DATA + TRAINING LOOP are provided. The two modelling pieces you implement
are marked `# TODO` below — that is the whole point of the track:
  1. the 1D encoder over [B, C=19, T]
  2. the SSL objective (two-view VICReg  *or*  predictive JEPA)
The downstream probe + metric is the third `# TODO`, in eval.py.

Run:  python -m examples.eeg.main --fname examples/eeg/cfgs/train.yaml
"""
import os
import sys

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from eb_jepa.datasets.eeg.dataset import EEGConfig, make_loader

# Reuse the eb_jepa core — DO NOT reimplement these:
#   eb_jepa.architectures: Projector (MLP), RNNPredictor (GRU)
#   eb_jepa.losses:        VICRegLoss (inv+var+cov), VCLoss (variance+covariance)
from eb_jepa.architectures import Projector
from eb_jepa.losses import BCS, VICRegLoss
from examples.eeg.spectral_losses import (MultiScaleSpectralLoss,
                                          FFTMagConsistencyLoss)


# --------------------------------------------------------------------------- #
# 1) ENCODER  — team-written (not organizer-provided)
# --------------------------------------------------------------------------- #
class EEG1DEncoder(nn.Module):
    """1D conv encoder over EEG windows [B, C=19, T].

    A stack of strided Conv1d blocks (kernel 7, stride 2, BatchNorm + GELU)
    halves the time axis each block; global average pooling then yields a
    [B, out_dim] representation. `frames()` exposes the pre-pool latent
    sequence [B, L, out_dim] for the optional predictive-JEPA objective.
    """

    def __init__(self, in_channels=19, hidden=64, out_dim=256, depth=4, kernel=7):
        super().__init__()
        widths = [hidden * (2 ** i) for i in range(depth - 1)] + [out_dim]
        blocks, c_in = [], in_channels
        for c_out in widths:
            blocks += [
                nn.Conv1d(c_in, c_out, kernel_size=kernel, stride=2,
                          padding=kernel // 2, bias=False),
                nn.BatchNorm1d(c_out),
                nn.GELU(),
            ]
            c_in = c_out
        self.backbone = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_dim = out_dim

    def feature_map(self, x):       # [B, C, T] -> [B, out_dim, L]
        return self.backbone(x)

    def represent(self, x):         # [B, C, T] -> [B, out_dim]
        return self.pool(self.feature_map(x)).flatten(1)

    def frames(self, x):            # [B, C, T] -> [B, L, out_dim]
        return self.feature_map(x).transpose(1, 2)

    def forward(self, x):
        return self.represent(x)


def build_encoder(cfg):
    """EEG encoder. cfg.encoder_type = "conv" (default, EEG1DEncoder) | "transformer"
    (ViT-style attention encoder — the capacity upgrade for closing the SOTA gap)."""
    if getattr(cfg, "encoder_type", "conv") == "transformer":
        from examples.eeg.transformer_encoder import build_transformer_encoder
        return build_transformer_encoder(cfg)
    return EEG1DEncoder(
        in_channels=cfg.in_channels,
        hidden=getattr(cfg, "hidden", 64),
        out_dim=cfg.out_dim,
        depth=getattr(cfg, "depth", 4),
        kernel=getattr(cfg, "kernel", 7),
    )


# --------------------------------------------------------------------------- #
# 2) SSL OBJECTIVE  — team-written (not organizer-provided)
# --------------------------------------------------------------------------- #
class TwoViewVICReg(nn.Module):
    """Two-view VICReg ("EB time-series JEPA", invariance form).

    The dataset returns two independently-corrupted views (v1, v2) of the SAME
    window (masking + channel-drop + noise + scale jitter — set their strengths
    in the data config). We encode both, project, and apply VICReg:
        invariance  : pull the two views' embeddings together (learn what is
                      invariant to the corruption)
        variance    : keep each dim's std above a margin  (anti-collapse)
        covariance  : decorrelate dims                    (anti-collapse)
    The var+cov terms are what stop the encoder from mapping every window to the
    same point.
    """

    def __init__(self, encoder, cfg):
        super().__init__()
        self.encoder = encoder
        spec = getattr(cfg, "projector_spec", f"{cfg.out_dim}-1024-1024-1024")
        self.projector = Projector(spec)
        self.ssl_loss = getattr(cfg, "ssl_loss", "vicreg")
        if self.ssl_loss == "vicreg":
            self.criterion = VICRegLoss(
                inv_coeff=getattr(cfg, "inv_coeff", 25.0),
                std_coeff=getattr(cfg, "std_coeff", 25.0),
                cov_coeff=getattr(cfg, "cov_coeff", 1.0),
            )
        elif self.ssl_loss == "sigreg":
            # LeJEPA / BCS: invariance + Epps-Pulley Gaussianity test as anti-collapse
            self.criterion = BCS(
                num_slices=getattr(cfg, "num_slices", 256),
                lmbd=getattr(cfg, "lmbd", 10.0),
            )
        else:
            raise ValueError(f"unknown model.ssl_loss={self.ssl_loss!r}")
        # --- Phase 5: optional auxiliary spectral terms (default 0 = base run) ---
        # Both compare the two corrupted views (v1, v2) directly in the spectral
        # domain. With coeff 0 they are not even evaluated, so the base pipeline is
        # byte-for-byte unchanged.
        self.spectral_coeff = float(getattr(cfg, "spectral_coeff", 0.0))
        self.fft_consistency_coeff = float(getattr(cfg, "fft_consistency_coeff", 0.0))
        fft_sizes = getattr(cfg, "spectral_fft_sizes", [256, 128, 64, 32])
        self.spectral_loss = MultiScaleSpectralLoss(
            fft_sizes=list(fft_sizes),
            alpha=float(getattr(cfg, "spectral_log_alpha", 1.0)),
        )
        self.fft_consistency_loss = FFTMagConsistencyLoss(
            log=bool(getattr(cfg, "fft_consistency_log", True)),
        )

    def compute_loss(self, batch):
        v1, v2 = batch
        # One encoder forward per view; the feature map [B, D, L] is reused for
        # BOTH the pooled VICReg representation and the spectral terms.
        fm1 = self.encoder.feature_map(v1)            # [B, D, L]
        fm2 = self.encoder.feature_map(v2)
        z1 = self.projector(fm1.mean(dim=2))          # global avg pool == represent()
        z2 = self.projector(fm2.mean(dim=2))
        out = self.criterion(z1, z2)
        loss = out["loss"]
        if self.ssl_loss == "vicreg":
            logs = {
                "inv": round(out["invariance_loss"].item(), 4),
                "var": round(out["var_loss"].item(), 4),
                "cov": round(out["cov_loss"].item(), 4),
            }
        else:  # sigreg: invariance + bcs (Epps-Pulley Gaussianity)
            logs = {
                "inv": round(out["invariance_loss"].item(), 4),
                "bcs": round(out["bcs_loss"].item(), 4),
            }
        # Phase 5: the spectral terms operate on the ENCODER FEATURE MAPS of the
        # two views (latent sequences [B, D, L]), NOT the raw inputs. Comparing
        # raw v1/v2 spectra is constant w.r.t. the model -> zero gradient -> no
        # effect on training; using the feature maps makes the term encoder-
        # dependent so it actually backprops into the representation.
        if self.spectral_coeff > 0:
            spec = self.spectral_loss(fm1, fm2)
            loss = loss + self.spectral_coeff * spec
            logs["spec"] = round(spec.item(), 4)
        if self.fft_consistency_coeff > 0:
            fftc = self.fft_consistency_loss(fm1, fm2)
            loss = loss + self.fft_consistency_coeff * fftc
            logs["fftc"] = round(fftc.item(), 4)
        return loss, logs


def build_ssl(encoder, cfg):
    """SSL objective: two-view VICReg (default) or masked-prediction JEPA.

    cfg.ssl_type = "vicreg" (two-view invariance, default) | "masked"
    (latent masked-prediction JEPA with an EMA target — the faithful JEPA form).
    """
    if getattr(cfg, "ssl_type", "vicreg") == "masked":
        from examples.eeg.masked_jepa import MaskedJEPA
        return MaskedJEPA(encoder, cfg)
    return TwoViewVICReg(encoder, cfg)


# --------------------------------------------------------------------------- #
# TRAINING LOOP  — provided
# --------------------------------------------------------------------------- #
def run(fname="examples/eeg/cfgs/train.yaml", cfg=None, folder=None, **overrides):
    if cfg is None:
        cfg = OmegaConf.load(fname)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([f"{k}={v}" for k, v in overrides.items()]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.meta.seed)

    dcfg = EEGConfig(**OmegaConf.to_container(cfg.data, resolve=True))
    dcfg.mode = "ssl"
    loader = make_loader(dcfg)

    encoder = build_encoder(cfg.model).to(device)
    ssl = build_ssl(encoder, cfg.model).to(device)
    opt = torch.optim.AdamW(ssl.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)

    ckpt_dir = folder or cfg.meta.ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    for epoch in range(cfg.optim.epochs):
        ssl.train()
        for batch in loader:
            batch = batch.to(device) if torch.is_tensor(batch) else [b.to(device) for b in batch]
            opt.zero_grad(set_to_none=True)
            loss, logs = ssl.compute_loss(batch)
            loss.backward(); opt.step()
        print(f"[eeg] epoch {epoch} loss={loss.item():.4f} {logs}", flush=True)
        torch.save({"epoch": epoch, "encoder": encoder.state_dict(),
                    "cfg": OmegaConf.to_container(cfg, resolve=True)},
                   os.path.join(ckpt_dir, "latest.pth.tar"))
    print(f"[eeg] done -> {ckpt_dir}/latest.pth.tar")


if __name__ == "__main__":
    fname = sys.argv[sys.argv.index("--fname") + 1] if "--fname" in sys.argv \
        else "examples/eeg/cfgs/train.yaml"
    # Dotted CLI overrides, e.g. `model.spectral_coeff=0.1 data.aug_exact_corruption=true`.
    overrides, skip = {}, False
    for tok in sys.argv[1:]:
        if skip:
            skip = False; continue
        if tok == "--fname":
            skip = True; continue
        if "=" in tok:
            k, v = tok.split("=", 1)
            overrides[k] = v
    run(fname=fname, **overrides)
