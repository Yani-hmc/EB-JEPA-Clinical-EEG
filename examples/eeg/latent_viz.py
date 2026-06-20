"""Latent-space visualization of the frozen EB-JEPA encoder ("world model") on TUAB.

Loads a trained checkpoint, extracts the frozen latent for every TUAB-eval window (or
mean-pooled recording), and projects it to 2-D with PCA, t-SNE and UMAP, coloured by the
clinical label (normal vs abnormal). Saves one PNG per method.

Run:
  python -m examples.eeg.latent_viz --ckpt <.../latest.pth.tar> --out reports/latent_viz --level window
"""
import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from examples.eeg.main import build_encoder
from examples.eeg.eval import extract_features

NORMAL, ABN = "#4C72B0", "#E1574C"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="reports/latent_viz")
    ap.add_argument("--level", default="window", choices=["window", "recording"])
    ap.add_argument("--n_windows", type=int, default=16)
    ap.add_argument("--tag", default="seed0")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- load the frozen encoder from the checkpoint ---
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = OmegaConf.create(state["cfg"])
    encoder = build_encoder(cfg.model).to(device)
    encoder.load_state_dict(state["encoder"]); encoder.eval()

    # --- extract the latent for the TUAB eval split ---
    pool = (args.level == "recording")
    with torch.no_grad():
        X, y = extract_features(encoder, "eval", device, pool=pool, n_windows=args.n_windows)
    y = np.asarray(y)
    print(f"[viz] X={X.shape} y={y.shape} abnormal_frac={float(y.mean()):.3f} level={args.level}")

    from sklearn.preprocessing import StandardScaler
    Xs = StandardScaler().fit_transform(X)

    # --- 2-D projections ---
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    embs = {}
    embs["PCA"] = PCA(n_components=2, random_state=0).fit_transform(Xs)
    perp = max(5, min(30, (len(Xs) - 1) // 3))
    embs["t-SNE"] = TSNE(n_components=2, random_state=0, init="pca",
                         perplexity=perp).fit_transform(Xs)
    try:
        import umap
        embs["UMAP"] = umap.UMAP(n_components=2, random_state=0).fit_transform(Xs)
    except Exception as e:                       # pragma: no cover
        print("[viz] UMAP unavailable, skipping:", e)

    # --- plot ---
    for name, Z in embs.items():
        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        for lbl, col, nm in [(0, NORMAL, "normal"), (1, ABN, "abnormal")]:
            m = y == lbl
            ax.scatter(Z[m, 0], Z[m, 1], s=7, c=col, alpha=0.45,
                       label=f"{nm} (n={int(m.sum())})", edgecolors="none")
        ax.set_title(f"{name} — EB-JEPA latent (tuned, {args.tag}) · TUAB eval {args.level}",
                     fontsize=10.5)
        ax.set_xlabel(f"{name}-1"); ax.set_ylabel(f"{name}-2")
        ax.legend(markerscale=2.2, fontsize=9, framealpha=0.9)
        fig.tight_layout()
        slug = name.lower().replace("-", "")
        p = os.path.join(args.out, f"latent_{slug}_{args.level}_{args.tag}.png")
        fig.savefig(p, dpi=150); plt.close(fig)
        print("[viz] wrote", p)


if __name__ == "__main__":
    main()
