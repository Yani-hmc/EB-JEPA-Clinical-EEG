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

from eb_jepa.architectures import Projector
from eb_jepa.datasets.eeg.dataset import EEGConfig, make_loader
from eb_jepa.losses import BCS, VICRegLoss
from eb_jepa.nn_utils import init_module_weights
from examples.eeg.encoders import BIOTEncoder, EEGPTEncoder, LaBraMEncoder

# Reuse the eb_jepa core — DO NOT reimplement these:
#   eb_jepa.architectures: Projector (MLP), RNNPredictor (GRU)
#   eb_jepa.losses:        VICRegLoss (inv+var+cov), VCLoss (variance+covariance)


# --------------------------------------------------------------------------- #
# 1) ENCODER
# --------------------------------------------------------------------------- #
class EEG1DEncoder(nn.Module):
    """Strided Conv1d stack over [B, C=n_channels, T] -> pooled [B, D].

    Each block halves the time axis (kernel 7, stride 2, BatchNorm1d + GELU).
    `represent()` global-average-pools the final conv map; `frames()` exposes
    the pre-pool sequence for a future predictive-JEPA objective.
    """

    def __init__(self, in_channels: int, hidden: int, depth: int, out_dim: int,
                 kernel_size: int = 7):
        super().__init__()
        widths = [in_channels] + [hidden * (2 ** i) for i in range(depth)]
        blocks = []
        for c_in, c_out in zip(widths[:-1], widths[1:]):
            blocks.append(nn.Conv1d(c_in, c_out, kernel_size=kernel_size, stride=2,
                                     padding=kernel_size // 2, bias=False))
            blocks.append(nn.BatchNorm1d(c_out))
            blocks.append(nn.GELU())
        self.conv = nn.Sequential(*blocks)
        self.out_proj = nn.Conv1d(widths[-1], out_dim, kernel_size=1)
        self.out_dim = out_dim
        self.apply(init_module_weights)

    def frames(self, x):
        """[B, C, T] -> [B, F, D] latent sequence (pre pooling)."""
        h = self.conv(x)            # [B, C', T']
        h = self.out_proj(h)        # [B, D, T']
        self.n_frames = h.shape[-1]
        return h.transpose(1, 2)    # [B, T', D]

    def represent(self, x):
        """[B, C, T] -> [B, D] global-average-pooled representation."""
        return self.frames(x).mean(dim=1)


_PATCH_ENCODERS = {"labram": LaBraMEncoder, "eegpt": EEGPTEncoder, "biot": BIOTEncoder}


def build_encoder(cfg):
    """Build the EEG encoder selected by `cfg.encoder_type`:
      * "conv" (default)        -> EEG1DEncoder (strided Conv1d stack)
      * "labram"/"eegpt"/"biot" -> architecture-inspired patch transformers
                                    (see examples/eeg/encoders.py)
    """
    encoder_type = cfg.get("encoder_type", "conv")
    if encoder_type == "conv":
        return EEG1DEncoder(
            in_channels=cfg.in_channels,
            hidden=cfg.get("hidden", 64),
            depth=cfg.get("depth", 4),
            out_dim=cfg.out_dim,
            kernel_size=cfg.get("kernel_size", 7),
        )
    if encoder_type in _PATCH_ENCODERS:
        return _PATCH_ENCODERS[encoder_type](
            in_channels=cfg.in_channels,
            window_len=cfg.get("window_len", 2000),
            patch_len=cfg.get("patch_len", 200),
            embed_dim=cfg.get("embed_dim", 128),
            tr_depth=cfg.get("tr_depth", 4),
            n_heads=cfg.get("n_heads", 4),
            mlp_ratio=cfg.get("mlp_ratio", 4.0),
            dropout=cfg.get("dropout", 0.1),
            out_dim=cfg.out_dim,
        )
    raise ValueError(f"unknown model.encoder_type={encoder_type!r}")


# --------------------------------------------------------------------------- #
# 2) SSL OBJECTIVE
# --------------------------------------------------------------------------- #
class EEGSSL(nn.Module):
    """Two-view invariance objective: encoder -> Projector -> {VICReg | SIGReg/BCS}.

    `cfg.ssl_loss` selects the anti-collapse term:
      * "vicreg" (default): eb_jepa.losses.VICRegLoss (variance + covariance)
      * "sigreg":           eb_jepa.losses.BCS (LeJEPA's Epps-Pulley Gaussianity test)
    Both losses take the same (z1, z2) projected-view signature, so swapping is a
    one-line config change (`model.ssl_loss=sigreg`).
    """

    def __init__(self, encoder, cfg):
        super().__init__()
        self.encoder = encoder
        spec = cfg.get("projector_spec", f"{encoder.out_dim}-1024-1024")
        self.projector = Projector(spec)
        self.ssl_loss = cfg.get("ssl_loss", "vicreg")
        if self.ssl_loss == "vicreg":
            self.loss_fn = VICRegLoss(inv_coeff=cfg.get("inv_coeff", 25.0),
                                       std_coeff=cfg.get("std_coeff", 25.0),
                                       cov_coeff=cfg.get("cov_coeff", 1.0))
        elif self.ssl_loss == "sigreg":
            self.loss_fn = BCS(num_slices=cfg.get("num_slices", 256),
                                lmbd=cfg.get("lmbd", 10.0))
        else:
            raise ValueError(f"unknown model.ssl_loss={self.ssl_loss!r}")

    def compute_loss(self, batch):
        v1, v2 = batch
        z1 = self.projector(self.encoder.represent(v1))
        z2 = self.projector(self.encoder.represent(v2))
        out = self.loss_fn(z1, z2)
        loss = out["loss"]
        logs = {k: float(v.item() if torch.is_tensor(v) else v)
                for k, v in out.items() if k != "loss"}
        return loss, logs


def build_ssl(encoder, cfg):
    """Build the two-view SSL objective (VICReg or SIGReg/BCS, per cfg.ssl_loss)."""
    return EEGSSL(encoder, cfg)


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
    run(fname=fname)
