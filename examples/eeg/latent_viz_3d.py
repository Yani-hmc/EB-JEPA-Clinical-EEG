"""3D UMAP of the frozen EB-JEPA encoder latent ("world model") on TUAB.

Loads a checkpoint, extracts the frozen latent for the TUAB-eval split, runs UMAP to 3-D and
renders (a) static 3-D scatter PNGs from several view angles and (b) an interactive, rotatable
Plotly HTML, coloured by normal/abnormal.

Run:
  python -m examples.eeg.latent_viz_3d --ckpt <.../latest.pth.tar> --out reports/latent_viz --level window
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

    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = OmegaConf.create(state["cfg"])
    enc = build_encoder(cfg.model).to(device)
    enc.load_state_dict(state["encoder"]); enc.eval()

    pool = (args.level == "recording")
    with torch.no_grad():
        X, y = extract_features(enc, "eval", device, pool=pool, n_windows=args.n_windows)
    y = np.asarray(y)
    print(f"[viz3d] X={X.shape} abnormal_frac={float(y.mean()):.3f} level={args.level}")

    from sklearn.preprocessing import StandardScaler
    Xs = StandardScaler().fit_transform(X)
    import umap
    Z = umap.UMAP(n_components=3, random_state=0).fit_transform(Xs)

    # --- static 3-D scatter from several angles ---
    angles = [(20, -60), (20, 40), (65, -85)]
    for i, (elev, azim) in enumerate(angles):
        fig = plt.figure(figsize=(6.6, 5.6))
        ax = fig.add_subplot(projection="3d")
        for lbl, col, nm in [(0, NORMAL, "normal"), (1, ABN, "abnormal")]:
            m = y == lbl
            ax.scatter(Z[m, 0], Z[m, 1], Z[m, 2], s=6, c=col, alpha=0.4,
                       label=f"{nm} (n={int(m.sum())})", edgecolors="none", depthshade=True)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"3D UMAP — EB-JEPA latent ({args.tag}) · TUAB eval {args.level}", fontsize=10)
        ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2"); ax.set_zlabel("UMAP-3")
        ax.legend(markerscale=2, fontsize=9, loc="upper left")
        fig.tight_layout()
        p = os.path.join(args.out, f"latent_umap3d_{args.level}_{args.tag}_a{i}.png")
        fig.savefig(p, dpi=150); plt.close(fig); print("[viz3d] wrote", p)

    # --- interactive, rotatable HTML (Plotly) ---
    try:
        import plotly.graph_objects as go
        cols = np.where(y == 1, ABN, NORMAL)
        fig = go.Figure(data=[go.Scatter3d(
            x=Z[:, 0], y=Z[:, 1], z=Z[:, 2], mode="markers",
            marker=dict(size=2.5, color=cols, opacity=0.6),
            text=np.where(y == 1, "abnormal", "normal"), hoverinfo="text")])
        fig.update_layout(
            title=f"3D UMAP — EB-JEPA latent ({args.tag}) · TUAB eval {args.level}",
            scene=dict(xaxis_title="UMAP-1", yaxis_title="UMAP-2", zaxis_title="UMAP-3"))
        h = os.path.join(args.out, f"latent_umap3d_{args.level}_{args.tag}.html")
        fig.write_html(h); print("[viz3d] wrote", h)
    except Exception as e:                       # pragma: no cover
        print("[viz3d] plotly unavailable, skipping HTML:", e)


if __name__ == "__main__":
    main()
