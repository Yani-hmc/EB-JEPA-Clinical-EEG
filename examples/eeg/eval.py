"""EEG — downstream evaluation (the patient-disjoint abnormality probe).

The feature-extraction harness is provided: per recording, encode N evenly-spaced
10 s windows with the FROZEN encoder and mean-pool them into ONE embedding. What
you implement (`# TODO`) is the probe + metric.

GOLDEN RULE — patient-disjoint split: fit the probe on `train` patients, score on
`eval` patients (no subject overlap). A probe that scores well *within* a subject
but collapses across subjects is measuring identity, not pathology — so the held-
out-patient number is the only one that answers the transferability question.

Run:  python -m examples.eeg.eval --ckpt <.../latest.pth.tar>
"""
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                              precision_score, recall_score, roc_auc_score)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from eb_jepa.datasets.eeg.dataset import EEGConfig, EEGDataset
from examples.eeg.main import build_encoder


@torch.no_grad()
def extract_features(encoder, split, device, subset_frac=None, subset_seed=0):
    """Provided: frozen encoder -> [N_rec, D] recording-level features + labels.

    One embedding per recording: encode its N windows and mean-pool them.

    `subset_frac`/`subset_seed` mirror the training run's data config — e.g. for a
    fast 5%-of-the-data smoke test, probing should score on the SAME balanced subset
    pretraining saw, not silently fall back to the full split.
    """
    ds = EEGDataset(EEGConfig(split=split, mode="probe",
                               subset_frac=subset_frac, subset_seed=subset_seed))
    loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False, num_workers=16,
                                         pin_memory=True)
    X, y = [], []
    for wins, labels, ok in loader:          # wins: [B, N, C, T]
        B, N = wins.shape[0], wins.shape[1]
        flat = wins.reshape(B * N, *wins.shape[2:]).to(device, non_blocking=True)
        z = encoder.represent(flat).reshape(B, N, -1).mean(dim=1)  # [B, D]
        z = z.cpu().numpy()
        for k in range(B):
            if bool(ok[k]):                  # drop unreadable recordings
                X.append(z[k]); y.append(int(labels[k]))
    return np.stack(X), np.array(y)


# --------------------------------------------------------------------------- #
# PROBE + METRIC
# --------------------------------------------------------------------------- #
def probe(Xtr, ytr, Xev, yev):
    """Fit a PATIENT-DISJOINT MLP probe on the FROZEN train features and score
    on the held-out-patient eval features. Returns a metrics dict.

    No leakage: standardize features on TRAIN stats only, then fit an
    MLPClassifier and score on the eval embeddings (abnormal=1 = positive class).

    Reports balanced_accuracy + AUROC alongside accuracy/f1/recall/precision —
    TUAB is class-imbalanced, and BACC/AUROC are what published baselines (LaBraM
    0.814 BACC, BIOT 0.796 BACC) report, so they're required for a fair comparison.
    """
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xev_s = scaler.transform(Xtr), scaler.transform(Xev)

    clf = MLPClassifier(hidden_layer_sizes=(128, 64), early_stopping=True,
                         max_iter=500, random_state=0)
    clf.fit(Xtr_s, ytr)

    pred = clf.predict(Xev_s)
    prob = clf.predict_proba(Xev_s)[:, 1]
    return {
        "accuracy": accuracy_score(yev, pred),
        "balanced_accuracy": balanced_accuracy_score(yev, pred),
        "auroc": roc_auc_score(yev, prob),
        "f1": f1_score(yev, pred, pos_label=1, zero_division=0),
        "recall": recall_score(yev, pred, pos_label=1, zero_division=0),
        "precision": precision_score(yev, pred, pos_label=1, zero_division=0),
    }


def main():
    ckpt = sys.argv[sys.argv.index("--ckpt") + 1]
    random_floor = "--random-floor" in sys.argv
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = OmegaConf.create(state["cfg"])
    encoder = build_encoder(cfg.model).to(device)
    encoder.load_state_dict(state["encoder"]); encoder.eval()

    subset_frac = cfg.data.get("subset_frac", None)
    subset_seed = cfg.data.get("subset_seed", 0)
    if subset_frac:
        print(f"[eeg-eval] using the same {subset_frac:.0%} balanced subset as training "
              f"(subset_seed={subset_seed})", flush=True)

    print("[eeg-eval] extracting TRAIN embeddings (fit set)...", flush=True)
    Xtr, ytr = extract_features(encoder, "train", device, subset_frac, subset_seed)
    print("[eeg-eval] extracting EVAL embeddings (held-out patients)...", flush=True)
    Xev, yev = extract_features(encoder, "eval", device, subset_frac, subset_seed)
    print("[eeg-eval] trained encoder:", probe(Xtr, ytr, Xev, yev))

    if random_floor:
        # Same probe on a freshly-initialized (untrained) encoder — the floor a
        # trained encoder must clear to show the SSL pretraining did anything.
        torch.manual_seed(0)
        rand_encoder = build_encoder(cfg.model).to(device).eval()
        Xtr_r, ytr_r = extract_features(rand_encoder, "train", device, subset_frac, subset_seed)
        Xev_r, yev_r = extract_features(rand_encoder, "eval", device, subset_frac, subset_seed)
        print("[eeg-eval] random floor:  ", probe(Xtr_r, ytr_r, Xev_r, yev_r))


if __name__ == "__main__":
    main()
