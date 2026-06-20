"""Visualisation des latents EB-JEPA : PCA + UMAP sur TUAB et TUEV.

Usage:
    python viz_latents.py --ckpt <path/to/latest.pth.tar> [--out <dir>] [--tuev]

Produit dans <out>/ :
  pca_tuab.png         — PCA 2D, coloré normal/abnormal (eval split, 276 enregistrements)
  umap_tuab.png        — UMAP 2D, coloré normal/abnormal (eval split)
  pca_tuab_train.png   — PCA 2D sur le train split (2717 enregistrements)
  umap_tuab_train.png  — idem UMAP
  [si --tuev]
  pca_tuev.png         — PCA 2D, 6 classes TUEV (eval split)
  umap_tuev.png        — UMAP 2D, 6 classes TUEV

Lancé sur dalia via submit_viz.sh (un GPU, ~10 min).
"""

import argparse
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap

# ─── repo path (adapter si besoin) ──────────────────────────────────────────
REPO = "/lustre/work/vivatech-slightlyunawarefc/tvasnier/eb_jepa"
sys.path.insert(0, REPO)
os.chdir(REPO)

from omegaconf import OmegaConf
from eb_jepa.datasets.eeg.dataset import EEGConfig, EEGDataset
from examples.eeg.main import build_encoder

# ─── couleurs ────────────────────────────────────────────────────────────────
TUAB_COLORS = {0: "#2196F3", 1: "#F44336"}   # bleu=normal, rouge=abnormal
TUAB_LABELS = {0: "Normal", 1: "Abnormal"}

TUEV_COLORS = {
    0: "#E53935",  # SPSW  — rouge
    1: "#FB8C00",  # GPED  — orange
    2: "#FDD835",  # PLED  — jaune
    3: "#43A047",  # EYEM  — vert
    4: "#1E88E5",  # ARTF  — bleu
    5: "#8E24AA",  # BCKG  — violet
}
TUEV_LABELS = {
    0: "SPSW (spike-slow-wave)",
    1: "GPED (gen. periodic discharge)",
    2: "PLED (lat. periodic discharge)",
    3: "EYEM (eye movement)",
    4: "ARTF (artefact)",
    5: "BCKG (background)",
}


# ─── extraction de features ──────────────────────────────────────────────────
@torch.no_grad()
def extract_tuab(encoder, split, device, n_windows=16):
    """Retourne [N_rec, D] embeddings + labels binaires depuis TUAB."""
    ds = EEGDataset(EEGConfig(split=split, mode="probe"))
    loader = torch.utils.data.DataLoader(
        ds, batch_size=8, shuffle=False, num_workers=8, pin_memory=True)
    X, y = [], []
    for wins, labels, ok in loader:
        B, N = wins.shape[:2]
        idx = np.linspace(0, N - 1, min(n_windows, N), dtype=int)
        flat = wins[:, idx].reshape(B * len(idx), *wins.shape[2:]).to(device)
        z = encoder.represent(flat).reshape(B, len(idx), -1).mean(1)
        z = z.cpu().numpy()
        for k in range(B):
            if bool(ok[k]):
                X.append(z[k]); y.append(int(labels[k]))
    return np.stack(X), np.array(y)


@torch.no_grad()
def extract_tuev(encoder, device,
                 tuev_pkl="/lustre/work/vivatech-slightlyunawarefc/tvasnier/tuev_biot/processed"):
    """Retourne [N_win, D] embeddings + labels 6-class depuis TUEV eval."""
    import pickle, glob
    files = sorted(glob.glob(os.path.join(tuev_pkl, "eval", "*.pkl")))
    if not files:
        print(f"[viz] TUEV pkl introuvable dans {tuev_pkl}/eval/ — ignoré", flush=True)
        return None, None

    X, y = [], []
    for f in files:
        with open(f, "rb") as fh:
            data = pickle.load(fh)
        sig = torch.tensor(data["signal"], dtype=torch.float32).unsqueeze(0).to(device)
        # z-score par canal (comme le pipeline SSL)
        sig = (sig - sig.mean(-1, keepdim=True)) / (sig.std(-1, keepdim=True) + 1e-6)
        if sig.shape[-1] < 100:
            continue
        z = encoder.represent(sig).cpu().numpy()[0]
        X.append(z); y.append(int(data["label"]))

    if not X:
        return None, None
    return np.stack(X), np.array(y)


# ─── utilitaires plot ────────────────────────────────────────────────────────
def scatter(ax, coords, labels, color_map, label_map, title, alpha=0.7, s=18):
    for cls in sorted(np.unique(labels)):
        mask = labels == cls
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=color_map[cls], label=label_map[cls],
                   s=s, alpha=alpha, linewidths=0)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Composante 1"); ax.set_ylabel("Composante 2")
    ax.legend(fontsize=8, markerscale=1.5, framealpha=0.6)
    ax.grid(True, alpha=0.3)


def run_pca(X):
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(Xs)
    return coords, pca.explained_variance_ratio_


def run_umap(X, n_neighbors=15, min_dist=0.1):
    Xs = StandardScaler().fit_transform(X)
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                        min_dist=min_dist, random_state=42, metric="cosine")
    return reducer.fit_transform(Xs)


def save_fig(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] sauvegardé → {path}", flush=True)


# ─── main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="chemin vers latest.pth.tar")
    parser.add_argument("--out",  default="./viz_out", help="dossier de sortie")
    parser.add_argument("--tuev", action="store_true", help="aussi extraire TUEV")
    parser.add_argument("--no-train", action="store_true",
                        help="sauter les plots sur le train split (plus lent)")
    parser.add_argument("--tuev-pkl",
                        default="/lustre/work/vivatech-slightlyunawarefc/tvasnier/tuev_biot/processed")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[viz] device={device}  out={args.out}", flush=True)

    # chargement de l'encodeur
    state   = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg     = OmegaConf.create(state["cfg"])
    encoder = build_encoder(cfg.model).to(device).eval()
    encoder.load_state_dict(state["encoder"])
    print(f"[viz] encodeur chargé depuis {args.ckpt}", flush=True)

    # ── TUAB eval (276 enregistrements) ─────────────────────────────────────
    print("[viz] extraction TUAB eval…", flush=True)
    X_ev, y_ev = extract_tuab(encoder, "eval", device)
    print(f"[viz] TUAB eval : {X_ev.shape}  classes={np.bincount(y_ev)}", flush=True)

    coords_pca, var = run_pca(X_ev)
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter(ax, coords_pca, y_ev, TUAB_COLORS, TUAB_LABELS,
            f"PCA — TUAB eval (n={len(y_ev)})\n"
            f"PC1={var[0]:.1%}  PC2={var[1]:.1%}")
    save_fig(fig, os.path.join(args.out, "pca_tuab.png"))

    print("[viz] UMAP TUAB eval…", flush=True)
    coords_umap = run_umap(X_ev)
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter(ax, coords_umap, y_ev, TUAB_COLORS, TUAB_LABELS,
            f"UMAP — TUAB eval (n={len(y_ev)})")
    save_fig(fig, os.path.join(args.out, "umap_tuab.png"))

    # ── TUAB train (2717 enregistrements) ───────────────────────────────────
    if not args.no_train:
        print("[viz] extraction TUAB train (2717 enregistrements)…", flush=True)
        X_tr, y_tr = extract_tuab(encoder, "train", device)
        print(f"[viz] TUAB train : {X_tr.shape}  classes={np.bincount(y_tr)}", flush=True)

        coords_pca_tr, var_tr = run_pca(X_tr)
        fig, ax = plt.subplots(figsize=(7, 5))
        scatter(ax, coords_pca_tr, y_tr, TUAB_COLORS, TUAB_LABELS,
                f"PCA — TUAB train (n={len(y_tr)})\n"
                f"PC1={var_tr[0]:.1%}  PC2={var_tr[1]:.1%}",
                alpha=0.5, s=10)
        save_fig(fig, os.path.join(args.out, "pca_tuab_train.png"))

        print("[viz] UMAP TUAB train…", flush=True)
        coords_umap_tr = run_umap(X_tr, n_neighbors=30)
        fig, ax = plt.subplots(figsize=(7, 5))
        scatter(ax, coords_umap_tr, y_tr, TUAB_COLORS, TUAB_LABELS,
                f"UMAP — TUAB train (n={len(y_tr)})",
                alpha=0.4, s=8)
        save_fig(fig, os.path.join(args.out, "umap_tuab_train.png"))

    # ── TUEV eval ────────────────────────────────────────────────────────────
    if args.tuev:
        print("[viz] extraction TUEV eval…", flush=True)
        X_tv, y_tv = extract_tuev(encoder, device, args.tuev_pkl)
        if X_tv is not None:
            print(f"[viz] TUEV : {X_tv.shape}  classes={np.bincount(y_tv)}", flush=True)

            coords_pca_tv, var_tv = run_pca(X_tv)
            fig, ax = plt.subplots(figsize=(8, 5))
            scatter(ax, coords_pca_tv, y_tv, TUEV_COLORS, TUEV_LABELS,
                    f"PCA — TUEV eval (n={len(y_tv)})\n"
                    f"PC1={var_tv[0]:.1%}  PC2={var_tv[1]:.1%}",
                    alpha=0.4, s=6)
            save_fig(fig, os.path.join(args.out, "pca_tuev.png"))

            print("[viz] UMAP TUEV…", flush=True)
            coords_umap_tv = run_umap(X_tv, n_neighbors=30, min_dist=0.05)
            fig, ax = plt.subplots(figsize=(8, 5))
            scatter(ax, coords_umap_tv, y_tv, TUEV_COLORS, TUEV_LABELS,
                    f"UMAP — TUEV eval (n={len(y_tv)})",
                    alpha=0.4, s=6)
            save_fig(fig, os.path.join(args.out, "umap_tuev.png"))

    print("[viz] terminé.", flush=True)


if __name__ == "__main__":
    main()
