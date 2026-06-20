"""Visualisation de l'espace latent BIOT (baseline) : PCA, UMAP, t-SNE, LDA.

Charge le checkpoint BIOT préentraîné (EEG-SHHS+PREST-18-channels),
extrait les embeddings sur TUAB eval (276 enregistrements, 18 canaux sur 19,
mean-pool 16 fenêtres → vecteur 256-dim par patient), puis projette en 2D.

LDA honnête : fit sur le train set (2717 rec), projeté sur eval.

Sortie dans <out>/ :
    pca_biot.png
    umap_biot.png
    tsne_biot.png
    lda_honest_biot.png
    comparison_4methods_biot.png
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap

# ── paths ─────────────────────────────────────────────────────────────────────
BIOT_REPO  = "/lustre/work/vivatech-slightlyunawarefc/tvasnier/external/BIOT"
EBJJEPA_REPO = "/lustre/work/vivatech-slightlyunawarefc/tvasnier/eb_jepa"
CKPT_DEFAULT = os.path.join(BIOT_REPO, "pretrained-models/EEG-SHHS+PREST-18-channels.ckpt")

sys.path.insert(0, BIOT_REPO)
sys.path.insert(0, EBJJEPA_REPO)
os.chdir(EBJJEPA_REPO)

from model.biot import BIOTEncoder
from eb_jepa.datasets.eeg.dataset import EEGConfig, EEGDataset

# ── style ─────────────────────────────────────────────────────────────────────
COLORS = {0: "#2196F3", 1: "#F44336"}
NAMES  = {0: "Normal",  1: "Abnormal"}

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
    """Retourne (X [N, 256], y [N]) — un vecteur par enregistrement TUAB.
    Utilise les 18 premiers canaux sur 19 (BIOT checkpoint 18-channel)."""
    ds = EEGDataset(EEGConfig(split=split, mode="probe"))
    loader = torch.utils.data.DataLoader(
        ds, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
    X, y = [], []
    for wins, labels, ok in loader:
        B, N = wins.shape[:2]
        idx  = np.linspace(0, N - 1, min(n_windows, N), dtype=int)
        # [B*n_windows, 18, T]  — drop channel 19
        flat = wins[:, idx].reshape(B * len(idx), *wins.shape[2:]).to(device)
        flat = flat[:, :18, :]
        z = encoder(flat).reshape(B, len(idx), -1).mean(1).cpu().numpy()
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


def do_umap(X):
    Xs = StandardScaler().fit_transform(X)
    return umap.UMAP(
        n_components=2, n_neighbors=15, min_dist=0.1,
        metric="cosine", random_state=42, verbose=False
    ).fit_transform(Xs)


def do_tsne(X):
    Xs    = StandardScaler().fit_transform(X)
    n_pca = min(50, Xs.shape[1], Xs.shape[0] - 1)
    Xpca  = PCA(n_components=n_pca, random_state=0).fit_transform(Xs)
    return TSNE(
        n_components=2, perplexity=min(30, len(Xpca) - 1),
        max_iter=1000, random_state=42, init="pca", learning_rate="auto"
    ).fit_transform(Xpca)


def do_lda_honest(encoder, device, n_windows=16):
    """Fit LDA on train, project eval — honest out-of-sample."""
    def embed(split):
        ds = EEGDataset(EEGConfig(split=split, mode="probe"))
        loader = torch.utils.data.DataLoader(
            ds, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
        X, y = [], []
        with torch.no_grad():
            for wins, labels, ok in loader:
                B, N = wins.shape[:2]
                idx = np.linspace(0, N - 1, min(n_windows, N), dtype=int)
                flat = wins[:, idx].reshape(B * len(idx), *wins.shape[2:]).to(device)
                flat = flat[:, :18, :]
                z = encoder(flat).reshape(B, len(idx), -1).mean(1).cpu().numpy()
                for k in range(B):
                    if bool(ok[k]):
                        X.append(z[k]); y.append(int(labels[k]))
        return np.stack(X), np.array(y)

    print("[viz_biot] LDA: extraction train …", flush=True)
    X_train, y_train = embed("train")
    print(f"[viz_biot] LDA: train X={X_train.shape}", flush=True)
    print("[viz_biot] LDA: extraction eval …", flush=True)
    X_eval,  y_eval  = embed("eval")

    scaler = StandardScaler().fit(X_train)
    lda    = LinearDiscriminantAnalysis(n_components=1).fit(scaler.transform(X_train), y_train)
    scores = lda.transform(scaler.transform(X_eval)).ravel()
    return scores, y_eval


# ── figures ───────────────────────────────────────────────────────────────────
def make_scatter(coords, y, title, subtitle=""):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for cls in [0, 1]:
        mask = y == cls
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=COLORS[cls], label=f"{NAMES[cls]} (n={mask.sum()})",
                   s=30, alpha=0.75, linewidths=0, zorder=3)
    full = f"{title}\n{subtitle}" if subtitle else title
    ax.set_title(full, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")
    ax.legend(fontsize=10, markerscale=1.8, framealpha=0.8, edgecolor="#CCCCCC")
    fig.tight_layout()
    return fig


def make_lda_fig(scores, y, title):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cls in [0, 1]:
        mask = y == cls
        ax.hist(scores[mask], bins=28, alpha=0.55, color=COLORS[cls],
                label=f"{NAMES[cls]} (n={mask.sum()})", density=True, edgecolor="none")
        ax.axvline(scores[mask].mean(), color=COLORS[cls], linewidth=1.8,
                   linestyle="--", alpha=0.9)
    ax.set_xlabel("LDA score", labelpad=6)
    ax.set_ylabel("Density", labelpad=6)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.legend(fontsize=10, framealpha=0.8, edgecolor="#CCCCCC")
    fig.tight_layout()
    return fig


def save(fig, path):
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz_biot] ✓  {path}", flush=True)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=CKPT_DEFAULT)
    parser.add_argument("--out",  default="./viz_out/biot")
    parser.add_argument("--n-windows", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[viz_biot] device={device}  ckpt={args.ckpt}", flush=True)

    # ── charger BIOT encoder ─────────────────────────────────────────────────
    encoder = BIOTEncoder(
        emb_size=256, heads=8, depth=4,
        n_channels=18, n_fft=200, hop_length=100
    ).to(device).eval()
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    encoder.load_state_dict(state)
    print(f"[viz_biot] BIOT encoder loaded (18-ch, 256-dim)", flush=True)

    # ── extraction eval ───────────────────────────────────────────────────────
    print("[viz_biot] extraction TUAB eval …", flush=True)
    X, y = extract_embeddings(encoder, "eval", device, args.n_windows)
    tag  = f"BIOT — TUAB eval (n={len(y)})"
    print(f"[viz_biot] X={X.shape}  Normal={(y==0).sum()}  Abnormal={(y==1).sum()}", flush=True)

    # ── PCA ───────────────────────────────────────────────────────────────────
    print("[viz_biot] PCA …", flush=True)
    C_pca, var = do_pca(X)
    save(make_scatter(C_pca, y, f"PCA — {tag}",
                      f"PC1={var[0]:.1%}  PC2={var[1]:.1%}  (total {sum(var):.1%})"),
         os.path.join(args.out, "pca_biot.png"))

    # ── UMAP ──────────────────────────────────────────────────────────────────
    print("[viz_biot] UMAP …", flush=True)
    C_umap = do_umap(X)
    save(make_scatter(C_umap, y, f"UMAP — {tag}",
                      "metric=cosine  n_neighbors=15  min_dist=0.1"),
         os.path.join(args.out, "umap_biot.png"))

    # ── t-SNE ─────────────────────────────────────────────────────────────────
    print("[viz_biot] t-SNE …", flush=True)
    C_tsne = do_tsne(X)
    save(make_scatter(C_tsne, y, f"t-SNE — {tag}",
                      "perplexity=30  init=PCA"),
         os.path.join(args.out, "tsne_biot.png"))

    # ── LDA (honnête) ─────────────────────────────────────────────────────────
    print("[viz_biot] LDA (fit train → eval) …", flush=True)
    scores_lda, y_lda = do_lda_honest(encoder, device, args.n_windows)
    save(make_lda_fig(scores_lda, y_lda,
                      f"LDA — BIOT — TUAB eval (n={len(y_lda)})\n"
                      "fit on train (2717 rec) → projected on eval (honest)"),
         os.path.join(args.out, "lda_honest_biot.png"))

    # ── figure 4-en-1 ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 5))
    for i, (C, subtitle) in enumerate([
        (C_pca,  f"PCA (PC1={var[0]:.1%}, PC2={var[1]:.1%})\nBIOT baseline"),
        (C_umap, "UMAP (cosine)\nBIOT baseline"),
        (C_tsne, "t-SNE (perp=30)\nBIOT baseline"),
    ]):
        ax = fig.add_subplot(1, 4, i + 1)
        for cls in [0, 1]:
            mask = y == cls
            ax.scatter(C[mask, 0], C[mask, 1], c=COLORS[cls],
                       label=f"{NAMES[cls]} (n={mask.sum()})",
                       s=20, alpha=0.7, linewidths=0)
        ax.set_title(subtitle, fontsize=10, fontweight="bold")
        ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")
        ax.legend(fontsize=8, markerscale=1.5, framealpha=0.7)
        ax.grid(True, alpha=0.3)
    ax4 = fig.add_subplot(1, 4, 4)
    for cls in [0, 1]:
        mask = y_lda == cls
        ax4.hist(scores_lda[mask], bins=28, alpha=0.55, color=COLORS[cls],
                 label=f"{NAMES[cls]} (n={mask.sum()})", density=True, edgecolor="none")
        ax4.axvline(scores_lda[mask].mean(), color=COLORS[cls],
                    linewidth=1.8, linestyle="--", alpha=0.9)
    ax4.set_title("LDA (fit train→eval, honest)\nBIOT baseline", fontsize=10, fontweight="bold")
    ax4.set_xlabel("LDA score"); ax4.set_ylabel("Density")
    ax4.legend(fontsize=8, framealpha=0.7); ax4.grid(True, alpha=0.3)
    fig.suptitle(f"Espace latent BIOT préentraîné — {tag}", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, os.path.join(args.out, "comparison_4methods_biot.png"))

    print(f"\n[viz_biot] Figures dans : {args.out}/", flush=True)
    for f in sorted(os.listdir(args.out)):
        if f.endswith(".png"):
            print(f"       {f}", flush=True)


if __name__ == "__main__":
    main()
