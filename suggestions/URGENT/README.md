# ⚠️ URGENT — Action Items (read this first)

Three critical gaps identified from the current Dalia results.
Do these before running anything new.

## Priority order

1. **eval_yhammache_encoders.md** — yhammache's 8 checkpoints have never been evaluated. This is the biggest gap.
2. **fix_ft_comparison.md** — fine-tuning results are being mixed with frozen probe results. They are not comparable.
3. **fix_metrics.md** — BACC and AUROC are missing from the repo's eval.py (tvasnier has a patched version, the repo does not).
