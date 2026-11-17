#!/usr/bin/env bash
# Copy the saved per-window predictions of every reported run into results/predictions/.
# Run from the repository root after the notebook has finished:
#   bash collect_predictions.sh /path/to/fedphysio_v2_results
set -e
SRC="${1:-fedphysio_v2_results}"
DST="results/predictions"
mkdir -p "$DST"
for d in "$SRC"/runs/*_r2_seed42; do
  mkdir -p "$DST/$(basename "$d")"
  cp "$d"/S*.csv "$d"/S*_hist.json "$DST/$(basename "$d")/" 2>/dev/null || cp "$d"/S*.json "$DST/$(basename "$d")/"
done
cp "$SRC"/runs/calibrated_*_r2_seed42.pkl "$DST/" 2>/dev/null || true
cp "$SRC"/logs/config.json "$DST/" 2>/dev/null || true
echo "copied predictions to $DST"
du -sh "$DST"
