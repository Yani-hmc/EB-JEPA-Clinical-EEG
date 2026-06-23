"""EEG — downstream evaluation (the patient-disjoint abnormality probe).

The feature-extraction harness is provided: per recording, encode N evenly-spaced
10 s windows with the FROZEN encoder and mean-pool them into ONE embedding. What
you implement (`# TODO`) is the probe + metric.

GOLDEN RULE — patient-disjoint split: fit the probe on `train` patients, score on
`eval` patients (no subject overlap). A probe that scores well *within* a subject
but collapses across subjects is measuring identity, not pathology — so the held-
out-patient number is the only one that answers the transferability question.

Run:  python -m examples.eeg.eval --ckpt <.../latest.pth.tar>
      python -m examples.eeg.eval --ckpt <.../latest.pth.tar> --probe mlp
"""
import argparse

import numpy as np
import torch
from omegaconf import OmegaConf

from eb_jepa.datasets.eeg.dataset import EEGConfig, EEGDataset
from examples.eeg.main import build_encoder


@torch.no_grad()
def extract_features(encoder, split, device, pool=True):
    """Frozen encoder -> features + labels.

    pool=True  : ONE mean-pooled embedding per recording (RECORDING-level; easier,
                 inflates metrics — what we report as the headline).
    pool=False : every 10 s window kept as its own sample (WINDOW-level; label =
                 the recording's label). This is the convention the TUAB literature
                 (BIOT, LaBraM, EEGNet benchmarks) scores on — use it for a fair,
                 apples-to-apples comparison.
    """
    ds = EEGDataset(EEGConfig(split=split, mode="probe"))
    loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False, num_workers=16,
                                         pin_memory=True)
    X, y = [], []
    for wins, labels, ok in loader:          # wins: [B, N, C, T]
        B, N = wins.shape[0], wins.shape[1]
        flat = wins.reshape(B * N, *wins.shape[2:]).to(device, non_blocking=True)
        zr = encoder.represent(flat).reshape(B, N, -1)            # [B, N, D]
        for k in range(B):
            if not bool(ok[k]):              # drop unreadable recordings
                continue
            if pool:
                X.append(zr[k].mean(dim=0).cpu().numpy()); y.append(int(labels[k]))
            else:
                zk = zr[k].cpu().numpy()
                for n in range(N):
                    X.append(zk[n]); y.append(int(labels[k]))
    return np.stack(X), np.array(y)


# --------------------------------------------------------------------------- #
# PROBE + METRIC  — team-written (not organizer-provided)
# --------------------------------------------------------------------------- #
def _metrics(yev, pred, proba, ytr):
    """Shared metric set: accuracy / balanced-acc / precision / recall / F1 / AUROC."""
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 f1_score, precision_score, recall_score,
                                 roc_auc_score)
    return {
        "accuracy": round(float(accuracy_score(yev, pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(yev, pred)), 4),
        "precision": round(float(precision_score(yev, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(yev, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(yev, pred, zero_division=0)), 4),
        "auroc": round(float(roc_auc_score(yev, proba)), 4),
        "n_train": int(len(ytr)), "n_eval": int(len(yev)),
        "frac_abnormal_eval": round(float(np.mean(yev)), 4),
    }


def probe(Xtr, ytr, Xev, yev):
    """Patient-disjoint frozen-feature probe.

    Standardize on TRAIN stats only (no leakage), fit a class-balanced
    LogisticRegression on the train-patient embeddings, and score on the
    held-out-patient eval embeddings. normal=0 vs abnormal=1.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    scaler = StandardScaler().fit(Xtr)             # TRAIN stats only
    Xtr_s, Xev_s = scaler.transform(Xtr), scaler.transform(Xev)

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(Xtr_s, ytr)

    pred = clf.predict(Xev_s)
    proba = clf.predict_proba(Xev_s)[:, 1]
    return _metrics(yev, pred, proba, ytr)


def mlp_probe(Xtr, ytr, Xev, yev, hidden=256, epochs=200, lr=1e-3,
              weight_decay=1e-4, seed=0):
    """Phase 3b — small MLP head trained on the FROZEN features (patient-disjoint).

    Same leakage discipline as the logreg probe: standardize on TRAIN stats only,
    fit a 1-hidden-layer MLP with class-balanced loss, score on held-out patients.
    Reports the same metric set as ``probe``.
    """
    import torch.nn as nn
    from sklearn.preprocessing import StandardScaler

    torch.manual_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = torch.tensor(scaler.transform(Xtr), dtype=torch.float32, device=dev)
    Xev_s = torch.tensor(scaler.transform(Xev), dtype=torch.float32, device=dev)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=dev)

    # class-balanced weights (counter abnormal/normal imbalance)
    counts = np.bincount(ytr, minlength=2).astype(np.float64)
    w = (counts.sum() / (2.0 * np.maximum(counts, 1.0)))
    class_w = torch.tensor(w, dtype=torch.float32, device=dev)

    net = nn.Sequential(
        nn.Linear(Xtr.shape[1], hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
        nn.Dropout(0.3), nn.Linear(hidden, 2),
    ).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss(weight=class_w)
    net.train()
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = crit(net(Xtr_s), ytr_t)
        loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        logits = net(Xev_s)
        proba = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        pred = logits.argmax(dim=1).cpu().numpy()
    return _metrics(yev, pred, proba, ytr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--probe", default="logreg", choices=["logreg", "mlp", "both"],
                    help="frozen-feature classifier head")
    ap.add_argument("--level", default="recording",
                    choices=["recording", "window", "both"],
                    help="recording-level (mean-pooled, headline) or window-level "
                         "(per 10s window, the TUAB-literature convention) or both")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = OmegaConf.create(state["cfg"])
    encoder = build_encoder(cfg.model).to(device)
    encoder.load_state_dict(state["encoder"]); encoder.eval()

    levels = ["recording", "window"] if args.level == "both" else [args.level]
    for lvl in levels:
        pool = (lvl == "recording")
        print(f"[eeg-eval] extracting {lvl}-level features (pool={pool})...", flush=True)
        Xtr, ytr = extract_features(encoder, "train", device, pool=pool)
        Xev, yev = extract_features(encoder, "eval", device, pool=pool)
        if args.probe in ("logreg", "both"):
            print(f"[eeg-eval][{lvl}][logreg]", probe(Xtr, ytr, Xev, yev))
        if args.probe in ("mlp", "both"):
            print(f"[eeg-eval][{lvl}][mlp]   ", mlp_probe(Xtr, ytr, Xev, yev))


if __name__ == "__main__":
    main()
