#!/bin/bash

# Batch run both analysis scripts for all 5 seeds
# Total time: ~20-25 minutes (2 scripts × 5 seeds × 2-2.5 min/seed)

set -e

VENV="/c/Users/asus/PycharmProjects/mcm/.venv/Scripts/python.exe"
ROOT="/c/Users/asus/Desktop/damage-factorized-robot-arm"
cd "$ROOT"

echo "=========================================="
echo "Batch Running Depth-Stratified Calibration"
echo "=========================================="
for seed in 7 17 27 37 47; do
  echo ""
  echo "[$(date '+%H:%M:%S')] Seed $seed ..."
  "$VENV" scripts/analyze_depth_stratified_calibration.py --seed "$seed"
done

echo ""
echo "=========================================="
echo "Batch Running Selective Prediction"
echo "=========================================="
for seed in 7 17 27 37 47; do
  echo ""
  echo "[$(date '+%H:%M:%S')] Seed $seed ..."
  "$VENV" scripts/analyze_selective_prediction.py --seed "$seed"
done

echo ""
echo "=========================================="
echo "All analyses complete! Generate summaries..."
echo "=========================================="
