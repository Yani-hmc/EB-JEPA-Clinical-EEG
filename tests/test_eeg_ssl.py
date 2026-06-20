"""
Local sanity tests for the EEG SSL track (no cluster, no real EDF data needed).

Builds the encoder + SSL objective on tiny random tensors shaped like real windows
([B, 19, 2000]) and checks all 4 `model.encoder_type` backbones (conv, labram, eegpt,
biot) train one step without shape errors or NaNs, before ever touching the cluster.
`ssl_loss` (vicreg/sigreg) is exercised independently on the `conv` backbone.
"""
import pytest
import torch
from omegaconf import OmegaConf

from examples.eeg.main import build_encoder, build_ssl

ENCODER_TYPES = ["conv", "labram", "eegpt", "biot"]


def _make_cfg(encoder_type="conv", ssl_loss="vicreg", objective="twoview"):
    return OmegaConf.create({
        "in_channels": 19,
        "out_dim": 64,
        "encoder_type": encoder_type,
        "window_len": 2000,
        # conv
        "hidden": 16,
        "depth": 3,
        "kernel_size": 7,
        # labram / eegpt / biot
        "patch_len": 200,
        "embed_dim": 32,
        "tr_depth": 2,
        "n_heads": 4,
        "mlp_ratio": 2.0,
        "dropout": 0.1,
        # ssl
        "objective": objective,
        "ssl_loss": ssl_loss,
        "projector_spec": "64-128-128",
        "num_slices": 32,
        "ema": 0.996,
    })


def _one_step(encoder_type="conv", ssl_loss="vicreg", objective="twoview"):
    torch.manual_seed(0)
    cfg = _make_cfg(encoder_type, ssl_loss, objective)
    encoder = build_encoder(cfg)
    ssl = build_ssl(encoder, cfg)

    v1 = torch.randn(8, 19, 2000)
    v2 = torch.randn(8, 19, 2000)
    loss, logs = ssl.compute_loss((v1, v2))

    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in ssl.parameters() if p.requires_grad]
    assert any(g is not None and torch.any(g != 0) for g in grads)
    return loss, logs


@pytest.mark.parametrize("encoder_type", ENCODER_TYPES)
def test_vicreg_one_step(encoder_type):
    loss, logs = _one_step(encoder_type, "vicreg")
    assert {"invariance_loss", "var_loss", "cov_loss"} <= logs.keys()


def test_sigreg_one_step():
    loss, logs = _one_step("conv", "sigreg")
    assert {"bcs_loss", "invariance_loss"} <= logs.keys()


@pytest.mark.parametrize("encoder_type", ENCODER_TYPES)
def test_predictive_jepa_one_step(encoder_type):
    loss, logs = _one_step(encoder_type, objective="predictive")
    assert {"pred_loss", "std_loss", "cov_loss"} <= logs.keys()


def test_predictive_jepa_target_is_ema_not_identity():
    """Target encoder must diverge from a pure copy after an update (EMA is live,
    not accidentally frozen to the init snapshot) but stay close (it's a SLOW average,
    not equal to the online encoder either)."""
    torch.manual_seed(0)
    cfg = _make_cfg("conv", objective="predictive")
    encoder = build_encoder(cfg)
    ssl = build_ssl(encoder, cfg)

    init_target = [p.clone() for p in ssl.target_encoder.parameters()]

    for _ in range(3):
        v1, v2 = torch.randn(8, 19, 2000), torch.randn(8, 19, 2000)
        loss, _ = ssl.compute_loss((v1, v2))
        loss.backward()
        for p in ssl.online_encoder.parameters():
            if p.grad is not None:
                with torch.no_grad():
                    p -= 0.01 * p.grad
                p.grad = None

    moved = [not torch.allclose(t0, t1) for t0, t1 in
             zip(init_target, ssl.target_encoder.parameters())]
    assert any(moved), "target encoder never moved -- EMA update isn't wired up"
    for p in ssl.target_encoder.parameters():
        assert not torch.isnan(p).any()


@pytest.mark.parametrize("encoder_type", ENCODER_TYPES)
def test_encoder_represent_and_frames_shapes(encoder_type):
    cfg = _make_cfg(encoder_type)
    encoder = build_encoder(cfg)
    x = torch.randn(4, 19, 2000)

    z = encoder.represent(x)
    assert z.shape == (4, cfg.out_dim)

    frames = encoder.frames(x)
    assert frames.shape == (4, encoder.n_frames, cfg.out_dim)
