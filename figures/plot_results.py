"""Generate result figures for the EB-JEPA EEG hackathon."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT = "/home/timothy/Bureau/Hackathon 1/Hack_the_worlds/figures"

# ─────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────

# tvasnier ablations — all conv encoder, recording-level logreg BACC
# (best of logreg/mlp per run; using logreg as primary per SSL standard)
tvasnier = {
    "base":           {"bacc": 0.796, "auroc": 0.888},
    "fftc (0.05)":    {"bacc": 0.807, "auroc": 0.887},
    "fftc (0.1)":     {"bacc": 0.798, "auroc": 0.881},
    "fftc (0.3)":     {"bacc": 0.802, "auroc": 0.889},
    "spec (0.05)":    {"bacc": 0.821, "auroc": 0.884},
    "spec (0.3)":     {"bacc": 0.789, "auroc": 0.877},
    "spec (1.0)":     {"bacc": 0.785, "auroc": 0.874},
    "corrupt":        {"bacc": 0.825, "auroc": 0.904},
    "corrupt_s1":     {"bacc": 0.816, "auroc": 0.891},
    "corrupt_s2":     {"bacc": 0.818, "auroc": 0.906},
    "corrupt_big":    {"bacc": 0.805, "auroc": 0.892},
    "corrupt_spec":   {"bacc": 0.802, "auroc": 0.899},
    "corrupt_sigreg": {"bacc": 0.825, "auroc": 0.904},
    "masked_big":     {"bacc": 0.764, "auroc": 0.841},
    "multicorpus":    {"bacc": 0.812, "auroc": 0.883},
    "spec_fixed":     {"bacc": 0.836, "auroc": 0.887},  # ← best
}

# yhammache encoder comparison — accuracy only (old eval.py)
# random floors from eval_all_full_74861.out
yhammache = {
    "conv\n+vicreg":   {"acc": 0.837, "random": 0.746},
    "conv\n+sigreg":   {"acc": 0.812, "random": 0.746},
    "biot\n+sigreg":   {"acc": 0.783, "random": 0.797},
    "biot\n+vicreg":   {"acc": 0.775, "random": 0.797},
    "labram\n+vicreg": {"acc": 0.775, "random": 0.652},
    "labram\n+sigreg": {"acc": 0.725, "random": 0.652},
    "eegpt\n+vicreg":  {"acc": 0.641, "random": 0.674},
    "eegpt\n+sigreg":  {"acc": 0.627, "random": 0.674},
}

# Published baselines (BACC where available, else accuracy ≈ BACC for balanced sets)
baselines = {
    "Random\ninit":         {"bacc": 0.500, "auroc": 0.500, "type": "floor"},
    "BIOT\n(SSL frozen)":   {"bacc": 0.796, "auroc": None,  "type": "baseline"},
    "LaBraM\n(SSL frozen)": {"bacc": 0.814, "auroc": None,  "type": "baseline"},
    "Deep4Net\n(supervised)": {"bacc": 0.854, "auroc": None, "type": "supervised"},
    "Ours best\n(spec_fixed)": {"bacc": 0.836, "auroc": 0.887, "type": "ours"},
    "Ours\n(corrupt)":      {"bacc": 0.825, "auroc": 0.904, "type": "ours"},
    "Ours conv\n(no BACC)": {"bacc": None,  "auroc": None,  "type": "ours"},  # placeholder
}

COLORS = {
    "floor":      "#cccccc",
    "baseline":   "#7bafd4",
    "supervised": "#e07b54",
    "ours":       "#5db46b",
    "vicreg":     "#4e79a7",
    "sigreg":     "#f28e2b",
    "random":     "#d3d3d3",
}

# ─────────────────────────────────────────────────────────────
# FIG 1 — Ablation heatmap: BACC vs AUROC for tvasnier runs
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

labels = list(tvasnier.keys())
baccs  = [tvasnier[k]["bacc"]  for k in labels]
aurocs = [tvasnier[k]["auroc"] for k in labels]

colors = ["#5db46b" if k == "spec_fixed" else
          "#f28e2b" if k == "corrupt" or k.startswith("corrupt") else
          "#7bafd4" for k in labels]

sc = ax.scatter(aurocs, baccs, c=colors, s=120, zorder=3, edgecolors="white", linewidth=0.8)

for i, lbl in enumerate(labels):
    ax.annotate(lbl, (aurocs[i], baccs[i]),
                textcoords="offset points", xytext=(6, 3),
                fontsize=7, color="#333333")

# Baseline lines
ax.axhline(0.814, color="#7bafd4", linestyle="--", linewidth=1, label="LaBraM (0.814)")
ax.axhline(0.796, color="#aaaaaa", linestyle=":",  linewidth=1, label="BIOT (0.796)")
ax.axhline(0.854, color="#e07b54", linestyle="--", linewidth=1, label="Deep4Net supervised (0.854)")

ax.set_xlabel("AUROC", fontsize=11)
ax.set_ylabel("BACC (balanced accuracy)", fontsize=11)
ax.set_title("tvasnier ablations — conv encoder, recording-level logreg probe", fontsize=11)
ax.legend(fontsize=8, loc="lower right")
ax.set_xlim(0.830, 0.920)
ax.set_ylim(0.750, 0.870)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT}/ablation_scatter.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved ablation_scatter.png")

# ─────────────────────────────────────────────────────────────
# FIG 2 — Encoder comparison (yhammache)
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

enc_labels = list(yhammache.keys())
accs    = [yhammache[k]["acc"]    for k in enc_labels]
randoms = [yhammache[k]["random"] for k in enc_labels]

x = np.arange(len(enc_labels))
w = 0.35

bars1 = ax.bar(x - w/2, accs, w, label="Trained encoder (acc)", zorder=3,
               color=[COLORS["vicreg"] if "vicreg" in k else COLORS["sigreg"] for k in enc_labels],
               edgecolor="white", linewidth=0.5)
bars2 = ax.bar(x + w/2, randoms, w, label="Random init floor (acc)", zorder=3,
               color=COLORS["random"], edgecolor="white", linewidth=0.5)

for bar, val in zip(bars1, accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)
for bar, val in zip(bars2, randoms):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f"{val:.3f}", ha="center", va="bottom", fontsize=7.5, color="#888888")

# Baseline lines
ax.axhline(0.814, color="#7bafd4", linestyle="--", linewidth=1, label="LaBraM BACC (0.814)")
ax.axhline(0.854, color="#e07b54", linestyle="--", linewidth=1, label="Deep4Net acc (0.854)")

vicreg_patch = mpatches.Patch(color=COLORS["vicreg"], label="VICReg")
sigreg_patch = mpatches.Patch(color=COLORS["sigreg"], label="SIGReg")
random_patch = mpatches.Patch(color=COLORS["random"], label="Random init floor")
ax.legend(handles=[vicreg_patch, sigreg_patch, random_patch,
                   mpatches.Patch(color="#7bafd4", label="LaBraM BACC baseline"),
                   mpatches.Patch(color="#e07b54", label="Deep4Net acc (supervised)")],
          fontsize=8, loc="lower right")

ax.set_xticks(x)
ax.set_xticklabels(enc_labels, fontsize=9)
ax.set_ylabel("Accuracy", fontsize=11)
ax.set_title("Encoder architecture comparison (yhammache) — TUAB patient-disjoint\n"
             "Note: accuracy only (BACC unavailable for these runs)", fontsize=10)
ax.set_ylim(0.55, 0.90)
ax.grid(True, axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT}/encoder_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved encoder_comparison.png")

# ─────────────────────────────────────────────────────────────
# FIG 3 — Summary: our best vs baselines
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

summary = [
    ("Random init",          0.500, "#dddddd", "floor"),
    ("BIOT (SSL frozen)",    0.796, "#7bafd4", "baseline"),
    ("LaBraM (SSL frozen)",  0.814, "#7bafd4", "baseline"),
    ("Ours — corrupt",       0.825, "#5db46b", "ours"),
    ("Ours — corrupt_sigreg",0.825, "#5db46b", "ours"),
    ("Ours — spec_fixed",    0.836, "#2d8a45", "ours_best"),
    ("Deep4Net (supervised)",0.854, "#e07b54", "supervised"),
]

names  = [s[0] for s in summary]
values = [s[1] for s in summary]
cols   = [s[2] for s in summary]

y = np.arange(len(names))
bars = ax.barh(y, values, color=cols, edgecolor="white", linewidth=0.6, zorder=3)

for bar, val, name in zip(bars, values, names):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=9,
            fontweight="bold" if "spec_fixed" in name else "normal")

ax.set_yticks(y)
ax.set_yticklabels(names, fontsize=9)
ax.set_xlabel("Balanced Accuracy (BACC)", fontsize=11)
ax.set_title("TUAB patient-disjoint — recording-level logreg probe\n"
             "Frozen SSL encoder vs supervised baseline", fontsize=11)
ax.set_xlim(0.45, 0.89)
ax.axvline(0.814, color="#7bafd4", linestyle="--", linewidth=1, alpha=0.7)
ax.grid(True, axis="x", alpha=0.3)

legend_handles = [
    mpatches.Patch(color="#dddddd", label="Random init floor"),
    mpatches.Patch(color="#7bafd4", label="Published SSL baselines"),
    mpatches.Patch(color="#5db46b", label="Ours (EB-JEPA conv)"),
    mpatches.Patch(color="#2d8a45", label="Ours — best (spec_fixed)"),
    mpatches.Patch(color="#e07b54", label="Supervised upper bound"),
]
ax.legend(handles=legend_handles, fontsize=8, loc="lower right")

plt.tight_layout()
plt.savefig(f"{OUT}/summary_vs_baselines.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved summary_vs_baselines.png")

# ─────────────────────────────────────────────────────────────
# FIG 4 — Training loss curves: EEGPT vs conv (inv_loss focus)
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

epochs = list(range(20))

conv_inv = [0.073,0.069,0.071,0.068,0.066,0.054,0.054,0.053,0.047,0.045,
            0.053,0.057,0.044,0.047,0.052,0.047,0.040,0.044,0.039,0.038]
conv_loss = [0.316,0.272,0.252,0.229,0.203,0.184,0.177,0.167,0.158,0.151,
             0.162,0.164,0.142,0.140,0.146,0.141,0.136,0.135,0.124,0.132]

eegpt_inv = [0.214,0.153,0.121,0.105,0.114,0.131,0.135,0.192,0.117,0.128,
             0.168,0.102,0.144,0.117,0.144,0.113,0.089,0.122,0.132,0.103]
eegpt_loss= [0.472,0.360,0.295,0.271,0.286,0.307,0.301,0.353,0.280,0.289,
             0.322,0.252,0.303,0.269,0.300,0.271,0.231,0.260,0.274,0.247]

ax1, ax2 = axes

ax1.plot(epochs, conv_loss,  color="#4e79a7", label="conv total loss")
ax1.plot(epochs, conv_inv,   color="#4e79a7", linestyle="--", label="conv inv loss")
ax1.plot(epochs, eegpt_loss, color="#e05c4b", label="eegpt total loss")
ax1.plot(epochs, eegpt_inv,  color="#e05c4b", linestyle="--", label="eegpt inv loss")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss (last batch)")
ax1.set_title("Training loss — VICReg (1,1,1) — old coefficients")
ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
ax1.annotate("inv stuck\nat 0.103", xy=(19, 0.103), xytext=(15, 0.16),
             arrowprops=dict(arrowstyle="->", color="#e05c4b"), color="#e05c4b", fontsize=8)
ax1.annotate("inv → 0.038", xy=(19, 0.038), xytext=(14, 0.06),
             arrowprops=dict(arrowstyle="->", color="#4e79a7"), color="#4e79a7", fontsize=8)

# Right: expected with fixed coefficients (schematic)
ax2.plot(epochs, conv_inv, color="#4e79a7", linestyle="--", label="conv inv (old, ref)")
expected_eegpt_inv = [v * 0.45 + (0.038 - 0.103*0.45)*(i/19) for i, v in enumerate(eegpt_inv)]
expected_eegpt_inv = np.clip(expected_eegpt_inv, 0.03, 0.25)
ax2.plot(epochs, eegpt_inv,          color="#e05c4b", linestyle="--", alpha=0.3, label="eegpt inv (old)")
ax2.plot(epochs, expected_eegpt_inv, color="#e05c4b", linewidth=2,   label="eegpt inv (expected, fixed coeff)")
ax2.axhline(0.038, color="#4e79a7", linestyle=":", linewidth=1, alpha=0.5)
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Invariance loss")
ax2.set_title("Expected effect of VICReg fix (25,25,1)\n— job 75490 running")
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 0.25)

plt.tight_layout()
plt.savefig(f"{OUT}/training_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved training_curves.png")

print("\nAll figures saved to", OUT)
