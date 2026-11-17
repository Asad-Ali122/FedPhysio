# Per-window predictions

Run, from the repository root, after the notebook has finished:

    bash collect_predictions.sh /path/to/fedphysio_v2_results

This copies, for every reported run, `S<subject>.csv` (columns `y_true, y_pred, p0, p1, p2, y_pred_best, y_pred_tau0, t_start`)
and `S<subject>_hist.json` (training history, validation score, selected prior-correction strength), the per-fold
sensor-configuration choice (`selection_r2_seed42/S<subject>.json`) and the calibrated-protocol predictions
(`calibrated_60s_r2_seed42.pkl`, `calibrated_120s_r2_seed42.pkl`).

Then verify every published number:

    python audit/recompute_metrics.py --runs results/predictions --tables results/tables
