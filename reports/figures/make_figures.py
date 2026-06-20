"""Regenerate all figures for the report with corrected per-window vs per-recording."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT = "/home/timothy/Bureau/Hackathon 1/Hack_the_worlds/reports/figures"

# ── colour palette ────────────────────────────────────────────────────────────
C = dict(ours_best="#2d8a45", ours="#5db46b", baseline="#7bafd4",
         supervised="#e07b54", floor="#cccccc", vicreg="#4e79a7",
         sigreg="#f28e2b", random="#d3d3d3", warn="#c0392b")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 1 — Honest comparison: per-window (fair) vs per-recording (ours-only)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

# Left: per-window (fair to literature)
pw_names  = ["Random\ninit", "BIOT\npretrained", "EB-JEPA\nbase", "EB-JEPA\n+corrupt",
             "EB-JEPA\n+spectral", "EEGNet\n(sup.)", "LaBraM-Base\n(paper)"]
pw_bacc   = [0.500, 0.802, 0.756, 0.770, 0.765, 0.796, 0.814]
pw_colors = [C["floor"], C["baseline"], C["ours"], C["ours"], C["ours"],
             C["supervised"], C["baseline"]]
pw_hatch  = ["", "", "", "", "", "", "//"]

ax = axes[0]
bars = ax.bar(range(len(pw_names)), pw_bacc, color=pw_colors,
              edgecolor="white", linewidth=0.6, zorder=3)
for bar, h in zip(bars, pw_hatch):
    bar.set_hatch(h)
for bar, val in zip(bars, pw_bacc):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
            f"{val:.3f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(range(len(pw_names))); ax.set_xticklabels(pw_names, fontsize=8)
ax.set_ylabel("Balanced Accuracy (BACC)", fontsize=10)
ax.set_title("Per-window — fair comparison to literature\n(same protocol as BIOT / LaBraM)", fontsize=9)
ax.set_ylim(0.45, 0.87); ax.grid(True, axis="y", alpha=0.3)
ax.axhline(0.814, color=C["baseline"], linestyle="--", lw=1, alpha=0.7)

# Right: per-recording (clinical, ours-only)
pr_names  = ["EB-JEPA\nbase", "EB-JEPA\n+corrupt", "EB-JEPA\n+spectral",
             "EB-JEPA\nfine-tune", "EEGNet\n(sup.)", "Deep4Net\n(paper, acc)"]
pr_bacc   = [0.796, 0.825, 0.836, 0.837, 0.824, 0.846]
pr_colors = [C["ours"], C["ours"], C["ours_best"], C["ours_best"], C["supervised"], C["supervised"]]
pr_hatch  = ["", "", "", "", "", "//"]

ax = axes[1]
bars = ax.bar(range(len(pr_names)), pr_bacc, color=pr_colors,
              edgecolor="white", linewidth=0.6, zorder=3)
for bar, h in zip(bars, pr_hatch):
    bar.set_hatch(h)
for bar, val in zip(bars, pr_bacc):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f"{val:.3f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(range(len(pr_names))); ax.set_xticklabels(pr_names, fontsize=8)
ax.set_ylabel("Balanced Accuracy (BACC)", fontsize=10)
ax.set_title("Per-recording (mean-pool 16 windows)\n⚠ NOT comparable to per-window literature", fontsize=9)
ax.set_ylim(0.76, 0.87); ax.grid(True, axis="y", alpha=0.3)
ax.set_facecolor("#fffdf5")

# shared legend
handles = [mpatches.Patch(color=C["floor"], label="Random floor"),
           mpatches.Patch(color=C["ours"], label="Ours (EB-JEPA, frozen)"),
           mpatches.Patch(color=C["ours_best"], label="Ours (best)"),
           mpatches.Patch(color=C["supervised"], label="Supervised"),
           mpatches.Patch(color=C["baseline"], label="Published SSL baseline"),
           mpatches.Patch(facecolor="white", edgecolor="black", hatch="//", label="Paper number (not re-run)")]
fig.legend(handles=handles, fontsize=8, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05))
plt.suptitle("TUAB patient-disjoint — evaluation protocol determines apparent performance",
             fontsize=11, fontweight="bold")
plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig(f"{OUT}/comparison_dual.png", dpi=150, bbox_inches="tight")
plt.close(); print("comparison_dual.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 2 — Ablation: what moved the needle (per-window BACC)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))

abl_names = ["Base\n(VICReg)", "+corruption\n(VICReg)", "+spectral\n(VICReg, 0.1)",
             "+corruption\n(SIGReg)", "Scaled enc\n(depth5/h96)", "Masked JEPA\n(predictive)"]
abl_window = [0.756, 0.770, 0.765, 0.775, 0.768, 0.682]
abl_record = [0.796, 0.825, 0.836, 0.825, 0.805, 0.764]

x = np.arange(len(abl_names))
w = 0.35
b1 = ax.bar(x - w/2, abl_window, w, color=C["vicreg"], label="Per-window (fair)", zorder=3)
b2 = ax.bar(x + w/2, abl_record, w, color=C["sigreg"], label="Per-recording", zorder=3, alpha=0.85)
for bar, val in zip(b1, abl_window):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
            f"{val:.3f}", ha="center", va="bottom", fontsize=7.5, color=C["vicreg"])
for bar, val in zip(b2, abl_record):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
            f"{val:.3f}", ha="center", va="bottom", fontsize=7.5, color="#c05800")

ax.axhline(0.802, color=C["baseline"], linestyle="--", lw=1, label="BIOT (per-window)")
ax.axhline(0.814, color="#2c6ea1", linestyle=":",  lw=1.2, label="LaBraM-Base (per-window)")
ax.set_xticks(x); ax.set_xticklabels(abl_names, fontsize=8.5)
ax.set_ylabel("BACC"); ax.set_ylim(0.62, 0.86)
ax.set_title("Ablation: corruption augmentation is the key gain\nScaling & masked JEPA did not help at 20 ep", fontsize=10)
ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/ablation_what_moved.png", dpi=150, bbox_inches="tight")
plt.close(); print("ablation_what_moved.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 3 — Encoder comparison (yhammache, accuracy, with random floor)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4.5))
enc = ["conv\n+vicreg", "conv\n+sigreg", "biot\n+sigreg", "biot\n+vicreg",
       "labram\n+vicreg", "labram\n+sigreg", "eegpt\n+vicreg", "eegpt\n+sigreg"]
acc  = [0.837, 0.812, 0.783, 0.775, 0.775, 0.725, 0.641, 0.627]
rand = [0.746, 0.746, 0.797, 0.797, 0.652, 0.652, 0.674, 0.674]
col  = [C["vicreg"], C["sigreg"]]*4

x = np.arange(len(enc))
b1 = ax.bar(x-0.18, acc,  0.32, color=col, edgecolor="white", zorder=3, label="Trained")
b2 = ax.bar(x+0.18, rand, 0.32, color=C["random"], edgecolor="white", zorder=3, label="Random init floor")
for bar, v, r in zip(b1, acc, rand):
    clr = C["warn"] if v < r else "black"
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
            f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, color=clr,
            fontweight="bold" if v < r else "normal")
for bar, v in zip(b2, rand):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
            f"{v:.3f}", ha="center", va="bottom", fontsize=7, color="#888")

ax.axhline(0.814, color=C["baseline"], linestyle="--", lw=1, label="LaBraM BACC baseline")
ax.set_xticks(x); ax.set_xticklabels(enc, fontsize=8.5)
ax.set_ylabel("Accuracy (per-recording, old eval)")
ax.set_title("Encoder architecture × SSL loss — EEGPT and BIOT trained BELOW random floor\n"
             "Red = trained encoder worse than random init (representations collapsed)", fontsize=9)
ax.legend(fontsize=8); ax.set_ylim(0.55, 0.88)
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/encoder_comparison.png", dpi=150, bbox_inches="tight")
plt.close(); print("encoder_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 4 — VICReg coefficient fix: EEGPT invariance loss before/after
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ep = list(range(20))
conv_inv  = [0.073,0.069,0.071,0.068,0.066,0.054,0.054,0.053,0.047,0.045,
             0.053,0.057,0.044,0.047,0.052,0.047,0.040,0.044,0.039,0.038]
eegpt_old = [0.214,0.153,0.121,0.105,0.114,0.131,0.135,0.192,0.117,0.128,
             0.168,0.102,0.144,0.117,0.144,0.113,0.089,0.122,0.132,0.103]
# Fixed run: epochs 0-11 observed, rest projected to converge
eegpt_fix = [0.046,0.028,0.019,0.036,0.030,0.031,0.023,0.025,0.020,0.019,
             0.020,0.018, None,None,None,None,None,None,None,None]
fix_obs = [(i, v) for i, v in enumerate(eegpt_fix) if v is not None]
ax.plot(ep, conv_inv,  color=C["vicreg"], lw=2, label="conv (old coeff, reference)")
ax.plot(ep, eegpt_old, color=C["warn"],   lw=2, linestyle="--", label="EEGPT (old coeff, 1,1,1) → collapsed")
ax.plot([i for i,v in fix_obs], [v for i,v in fix_obs],
        color="#27ae60", lw=2.5, label="EEGPT (fixed coeff, 25,25,1) — running")
ax.axhline(0.038, color=C["vicreg"], linestyle=":", lw=1, alpha=0.5)
ax.annotate("Stuck at 0.103\n(failed SSL)", xy=(19, 0.103), xytext=(13, 0.16),
            arrowprops=dict(arrowstyle="->", color=C["warn"]), color=C["warn"], fontsize=8)
ax.annotate("Already 0.018\nat ep 11", xy=(11, 0.018), xytext=(7, 0.06),
            arrowprops=dict(arrowstyle="->", color="#27ae60"), color="#27ae60", fontsize=8)
ax.set_xlabel("Epoch"); ax.set_ylabel("Invariance loss (unweighted MSE)")
ax.set_title("VICReg coefficient fix: invariance loss convergence\n"
             "Old (1,1,1): inv underweighted 25×, EEGPT never converged", fontsize=9)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/vicreg_fix.png", dpi=150, bbox_inches="tight")
plt.close(); print("vicreg_fix.png")

print("\nAll figures saved.")
