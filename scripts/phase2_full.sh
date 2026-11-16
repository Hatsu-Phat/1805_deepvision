#!/usr/bin/env bash
# Phase 2 on the server, end to end; every step resumes where it stopped, so the script can be rerun.
#   nohup nice -n 10 bash scripts/phase2_full.sh > ~/deepvision_runs/p2_logs/main.log 2>&1 &
# 1. extract test features (+ upload)  2. extract train features  3. heads x seeds + summary
# 4. upload small results (no checkpoints) to gdrive:DeepVision/runs  5. upload train features
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"
export HF_HOME="$HOME/deepvision_cache/hf" TORCH_HOME="$HOME/deepvision_cache/torch"
L="$HOME/deepvision_runs/p2_logs"
mkdir -p "$L"
echo "start $(date) commit $(git rev-parse --short HEAD)"

python -m deepvision.features.extract extract --split test --upload > "$L/extract_test.log" 2>&1
echo "test extract+upload exit $? $(date)"

python -m deepvision.features.extract extract --split train > "$L/extract_train.log" 2>&1 \
    || { echo "train extract failed $(date)"; exit 1; }
echo "train extracted $(date)"
cp "$HOME/deepvision_features/encoder_costs.csv" "$L/" 2>/dev/null

python -m deepvision.features.heads > "$L/heads.log" 2>&1 || { echo "heads failed $(date)"; exit 1; }
python -m deepvision.features.heads --summary > "$L/summary.txt" 2>&1
echo "heads done $(date)"

rclone copy "$HOME/deepvision_runs" gdrive:DeepVision/runs --filter "- p2_*/ckpt/**" --filter "+ p2_*/**" \
    --filter "+ p2_summary.csv" --filter "- **" > "$L/upload_runs.log" 2>&1
echo "runs upload exit $? $(date)"

python -m deepvision.features.extract upload --split train > "$L/upload_train.log" 2>&1
echo "train features upload exit $? $(date)"
echo "all done $(date)"
