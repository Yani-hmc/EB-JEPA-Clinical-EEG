"""
Deep4Net baseline on TUAB (TUH Abnormal EEG Corpus).
Supervised training with labels — the ceiling EB-JEPA frozen probe must beat.

Usage:
    python train_deep4net.py --tuab_path /path/to/TUAB --epochs 20

Reference:
    Schirrmeister et al. 2017 — arxiv 1708.08012
    Target: 85.4% accuracy / BACC ~0.816 (patient-disjoint eval)
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch.optim import Adam
from torch.utils.data import DataLoader

from braindecode.datasets import TUHAbnormal
from braindecode.models import Deep4Net
from braindecode.preprocessing import Preprocessor, create_fixed_length_windows, preprocess

SFREQ      = 200          # resample to match EB-JEPA
WINDOW_S   = 10           # seconds
N_TIMES    = SFREQ * WINDOW_S   # 2000 samples
N_CHANS    = 19
N_CLASSES  = 2
BATCH_SIZE = 32


def load_tuab(tuab_path, split, n_jobs=4):
    ds = TUHAbnormal(
        path=str(Path(tuab_path) / split),
        target_name="pathological",
        preload=False,
    )
    preprocess(ds, [
        Preprocessor("pick_types", eeg=True),
        Preprocessor(lambda x: x * 1e6, apply_on_array=True),  # V → µV
        Preprocessor("resample", sfreq=SFREQ),
    ], n_jobs=n_jobs)
    windows = create_fixed_length_windows(
        ds,
        window_size_samples=N_TIMES,
        window_stride_samples=N_TIMES,
        drop_last_window=True,
    )
    return windows


def evaluate(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for X, y, _ in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            all_logits.append(logits.cpu())
            all_labels.append(y.cpu())
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels).numpy()
    probs  = torch.softmax(logits, dim=1)[:, 1].numpy()
    preds  = logits.argmax(dim=1).numpy()
    acc    = (preds == labels).mean()
    bacc   = balanced_accuracy_score(labels, preds)
    auroc  = roc_auc_score(labels, probs)
    model.train()
    return acc, bacc, auroc


def main(tuab_path, epochs=20, lr=1e-3, n_jobs=4, out_dir="results"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading TUAB train split...")
    train_windows = load_tuab(tuab_path, "train", n_jobs)
    print("Loading TUAB eval split...")
    eval_windows  = load_tuab(tuab_path, "eval",  n_jobs)

    train_loader = DataLoader(train_windows, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
    eval_loader  = DataLoader(eval_windows,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"Train windows: {len(train_windows)} | Eval windows: {len(eval_windows)}")

    model = Deep4Net(n_chans=N_CHANS, n_outputs=N_CLASSES, n_times=N_TIMES).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    log_file = out_path / "training_log.csv"

    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "eval_acc", "eval_bacc", "eval_auroc"])

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for X, y, _ in train_loader:
            X, y = X.to(device), y.long().to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        train_loss = np.mean(losses)
        acc, bacc, auroc = evaluate(model, eval_loader, device)

        print(f"Epoch {epoch:02d} | loss {train_loss:.4f} | acc {acc:.4f} | bacc {bacc:.4f} | auroc {auroc:.4f}")

        with open(log_file, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, acc, bacc, auroc])

    # Save final checkpoint
    torch.save(model.state_dict(), out_path / "deep4net_final.pth")
    print(f"\nDone. Results saved to {out_path}/")
    print(f"Final eval BACC: {bacc:.4f} | AUROC: {auroc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tuab_path", required=True, help="Path to TUAB root (contains train/ and eval/)")
    parser.add_argument("--epochs",    type=int, default=20)
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--n_jobs",    type=int, default=4)
    parser.add_argument("--out_dir",   default="results/deep4net")
    args = parser.parse_args()
    main(args.tuab_path, args.epochs, args.lr, args.n_jobs, args.out_dir)
