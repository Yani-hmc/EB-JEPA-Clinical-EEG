"""Visualisation de l'espace latent EB-JEPA : PCA, UMAP, t-SNE.

Prend le meilleur encodeur entraîné, extrait ses embeddings sur TUAB eval
(276 enregistrements, mean-pool 16 fenêtres → vecteur 256-dim par patient),
puis projette en 2D avec trois méthodes et sauvegarde les figures.

Usage:
    python viz_latents.py --ckpt <chemin/latest.pth.tar> [--out <dossier>]

Sortie dans <out>/ :
    pca_tuab.png    — PCA linéaire
    umap_tuab.png   — UMAP non-linéaire (préserve la topologie globale)
    tsne_tuab.png   — t-SNE (préserve les voisinages locaux)

Dépendances (toutes dans eb_jepa_x86_64) :
    scikit-learn, umap-learn, matplotlib, torch, omegaconf
"""

import argparse
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap

# ── repo EB-JEPA ─────────────────────────────────────────────────────────────
REPO = "/lustre/work/vivatech-slightlyunawarefc/tvasnier/eb_jepa"
sys.path.insert(0, REPO)
os.chdir(REPO)

from omegaconf import OmegaConf
from eb_jepa.datasets.eeg.dataset import EEGConfig, EEGDataset
from examples.eeg.main import build_encoder

# ── style ─────────────────────────────────────────────────────────────────────
COLORS = {0: "#2196F3", 1: "#F44336"}   # bleu = Normal, rouge = Abnormal
NAMES  = {0: "Normal", 1: "Abnormal"}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#F8F8F8",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "font.size":        11,
})


# ── extraction ────────────────────────────────────────────────────────────────
@torch.no_grad()
def extract_embeddings(encoder, split, device, n_windows=16):
    """Retourne (X [N, D], y [N]) — un vecteur par enregistrement TUAB."""
    ds = EEGDataset(EEGConfig(split=split, mode="probe"))
    loader = torch.utils.data.DataLoader(
        ds, batch_size=8, shuffle=False, num_workers=8, pin_memory=True)
    X, y = [], []
    for wins, labels, ok in loader:
        B, N = wins.shape[:2]
        idx  = np.linspace(0, N - 1, min(n_windows, N), dtype=int)
        flat = wins[:, idx].reshape(B * len(idx), *wins.shape[2:]).to(device)
        z    = encoder.represent(flat).reshape(B, len(idx), -1).mean(1)
        z    = z.cpu().numpy()
        for k in range(B):
            if bool(ok[k]):
                X.append(z[k]); y.append(int(labels[k]))
    return np.stack(X), np.array(y)


# ── projections ───────────────────────────────────────────────────────────────
def do_pca(X):
    Xs  = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=0)
    C   = pca.fit_transform(Xs)
    return C, pca.explained_variance_ratio_


def do_umap(X, n_neighbors=15, min_dist=0.1):
    Xs = StandardScaler().fit_transform(X)
    return umap.UMAP(
        n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
        metric="cosine", random_state=42, verbose=False
    ).fit_transform(Xs)


def do_tsne(X, perplexity=30, n_iter=1000):
    # t-SNE tourne sur un PCA-50 pour la vitesse (pratique standard)
    Xs    = StandardScaler().fit_transform(X)
    n_pca = min(50, Xs.shape[1], Xs.shape[0] - 1)
    Xpca  = PCA(n_components=n_pca, random_state=0).fit_transform(Xs)
    return TSNE(
        n_components=2, perplexity=min(perplexity, len(Xpca) - 1),
        max_iter=n_iter, random_state=42, init="pca", learning_rate="auto"
    ).fit_transform(Xpca)


def do_lda(X, y):
    """Project onto the single LDA axis (binary: 1 discriminant).
    Returns 1-D scores — the direction that maximally separates the two classes."""
    Xs = StandardScaler().fit_transform(X)
    return LinearDiscriminantAnalysis(n_components=1).fit_transform(Xs, y).ravel()


def make_lda_figure(scores, y, title):
    """Density histogram of LDA scores — the honest separability plot."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cls in [0, 1]:
        mask = y == cls
        ax.hist(scores[mask], bins=28, alpha=0.55, color=COLORS[cls],
                label=f"{NAMES[cls]} (n={mask.sum()})", density=True, edgecolor="none")
    # overlap shading already handled by alpha; add vertical means
    for cls in [0, 1]:
        mask = y == cls
        ax.axvline(scores[mask].mean(), color=COLORS[cls], linewidth=1.8,
                   linestyle="--", alpha=0.9)
    ax.set_xlabel("LDA score (supervised projection onto max-separation axis)", labelpad=6)
    ax.set_ylabel("Density", labelpad=6)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.legend(fontsize=10, framealpha=0.8, edgecolor="#CCCCCC")
    fig.tight_layout()
    return fig


# ── figure ────────────────────────────────────────────────────────────────────
def make_figure(coords, y, title, subtitle=""):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for cls in [0, 1]:
        mask = y == cls
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=COLORS[cls], label=f"{NAMES[cls]} (n={mask.sum()})",
            s=30, alpha=0.75, linewidths=0, zorder=3
        )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    if subtitle:
        ax.set_title(f"{title}\n{subtitle}", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Dim 1", labelpad=6)
    ax.set_ylabel("Dim 2", labelpad=6)
    ax.legend(fontsize=10, markerscale=1.8, framealpha=0.8,
              edgecolor="#CCCCCC", loc="best")
    fig.tight_layout()
    return fig


def save(fig, path):
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] ✓  {path}", flush=True)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True,
                        help="Chemin vers latest.pth.tar (meilleur modèle)")
    parser.add_argument("--out",  default="./viz_out",
                        help="Dossier de sortie pour les PNG")
    parser.add_argument("--split", default="eval",
                        choices=["eval", "train"],
                        help="Split TUAB à visualiser (eval=276 rec, train=2717 rec)")
    parser.add_argument("--n-windows", type=int, default=16,
                        help="Nombre de fenêtres à mean-pooler par enregistrement")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[viz] device={device}   ckpt={args.ckpt}", flush=True)

    # Chargement encodeur
    state   = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg     = OmegaConf.create(state["cfg"])
    encoder = build_encoder(cfg.model).to(device).eval()
    encoder.load_state_dict(state["encoder"])
    obj = cfg.model.get("ssl_loss", cfg.model.get("objective", "?"))
    print(f"[viz] encodeur : {cfg.model.get('encoder_type','conv')}  obj={obj}  "
          f"dim={cfg.model.out_dim}", flush=True)

    # Extraction
    print(f"[viz] extraction TUAB {args.split} …", flush=True)
    X, y = extract_embeddings(encoder, args.split, device, args.n_windows)
    n_normal, n_abnormal = (y == 0).sum(), (y == 1).sum()
    print(f"[viz] X={X.shape}  Normal={n_normal}  Abnormal={n_abnormal}", flush=True)

    tag = f"TUAB {args.split} (n={len(y)})"

    # ── PCA ──────────────────────────────────────────────────────────────────
    print("[viz] PCA …", flush=True)
    C_pca, var = do_pca(X)
    fig = make_figure(C_pca, y,
                      f"PCA — {tag}",
                      f"PC1={var[0]:.1%}  PC2={var[1]:.1%}  "
                      f"(variance expliquée totale: {sum(var):.1%})")
    save(fig, os.path.join(args.out, "pca_tuab.png"))

    # ── UMAP ─────────────────────────────────────────────────────────────────
    print("[viz] UMAP …", flush=True)
    C_umap = do_umap(X)
    fig = make_figure(C_umap, y,
                      f"UMAP — {tag}",
                      "metric=cosine  n_neighbors=15  min_dist=0.1")
    save(fig, os.path.join(args.out, "umap_tuab.png"))

    # ── t-SNE ────────────────────────────────────────────────────────────────
    print("[viz] t-SNE … (peut prendre ~1-2 min)", flush=True)
    C_tsne = do_tsne(X)
    fig = make_figure(C_tsne, y,
                      f"t-SNE — {tag}",
                      "perplexity=30  init=PCA  learning_rate=auto")
    save(fig, os.path.join(args.out, "tsne_tuab.png"))

    # ── LDA ──────────────────────────────────────────────────────────────────
    print("[viz] LDA …", flush=True)
    scores_lda = do_lda(X, y)
    fig = make_lda_figure(scores_lda, y,
                          f"LDA — {tag}\n(supervised projection onto max-separation axis)")
    save(fig, os.path.join(args.out, "lda_tuab.png"))

    # ── figure comparative 4-en-1 ────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 5))
    # 3 scatter panels
    scatter_specs = [
        (C_pca,  f"PCA\n(PC1={var[0]:.1%}, PC2={var[1]:.1%})"),
        (C_umap, "UMAP\n(cosine, n_neighbors=15)"),
        (C_tsne, "t-SNE\n(perplexity=30)"),
    ]
    for i, (C, subtitle) in enumerate(scatter_specs):
        ax = fig.add_subplot(1, 4, i + 1)
        for cls in [0, 1]:
            mask = y == cls
            ax.scatter(C[mask, 0], C[mask, 1],
                       c=COLORS[cls], label=f"{NAMES[cls]} (n={mask.sum()})",
                       s=20, alpha=0.7, linewidths=0)
        ax.set_title(subtitle, fontsize=11, fontweight="bold")
        ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")
        ax.legend(fontsize=9, markerscale=1.5, framealpha=0.7)
        ax.grid(True, alpha=0.3)
    # LDA density panel
    ax4 = fig.add_subplot(1, 4, 4)
    for cls in [0, 1]:
        mask = y == cls
        ax4.hist(scores_lda[mask], bins=28, alpha=0.55, color=COLORS[cls],
                 label=f"{NAMES[cls]} (n={mask.sum()})", density=True, edgecolor="none")
        ax4.axvline(scores_lda[mask].mean(), color=COLORS[cls],
                    linewidth=1.8, linestyle="--", alpha=0.9)
    ax4.set_title("LDA\n(max-separation axis, uses labels)", fontsize=11, fontweight="bold")
    ax4.set_xlabel("LDA score"); ax4.set_ylabel("Density")
    ax4.legend(fontsize=9, framealpha=0.7)
    ax4.grid(True, alpha=0.3)
    fig.suptitle(f"Espace latent EB-JEPA — {tag}", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, os.path.join(args.out, "comparison_3methods.png"))

    print(f"\n[viz] Figures sauvegardées dans : {args.out}/", flush=True)
    print("[viz] Fichiers :", flush=True)
    for f in sorted(os.listdir(args.out)):
        if f.endswith(".png"):
            print(f"       {f}", flush=True)


if __name__ == "__main__":
    main()
