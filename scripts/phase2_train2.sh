#!/usr/bin/env bash
# Phase 2 with the extra Youku reals (split train2, 2026-11-06): extract the train2 features, train the heads on
# splits_v3 (train + train2, checkpoints selected and evaluated on test), upload the small results, then upload
# the train2 features. Every step resumes where it stopped, so the script can be rerun.
#   nohup nice -n 10 bash scripts/phase2_train2.sh > ~/deepvision_runs/p2_logs/main_p2y.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"
export HF_HOME="$HOME/deepvision_cache/hf" TORCH_HOME="$HOME/deepvision_cache/torch"
L="$HOME/deepvision_runs/p2_logs"
mkdir -p "$L"
echo "start $(date) commit $(git rev-parse --short HEAD)"

python -m deepvision.features.extract extract --split train2 > "$L/extract_train2.log" 2>&1 \
    || { echo "train2 extract failed $(date)"; exit 1; }
echo "train2 extracted $(date)"

python -m deepvision.features.heads --splits gdrive:DeepVision/dataset_v1/manifests/splits_v3.csv --prefix p2y \
    > "$L/heads_p2y.log" 2>&1 || { echo "heads failed $(date)"; exit 1; }
python -m deepvision.features.heads --prefix p2y --summary > "$L/summary_p2y.txt" 2>&1
echo "heads done $(date)"

rclone copy "$HOME/deepvision_runs" gdrive:DeepVision/runs --filter "- p2y_*/ckpt/**" --filter "+ p2y_*/**" \
    --filter "+ p2y_summary.csv" --filter "+ p2_logs/*p2y*" --filter "+ p2_logs/*train2*" --filter "- **" \
    > "$L/upload_runs_p2y.log" 2>&1
echo "runs upload exit $? $(date)"

python -m deepvision.features.extract upload --split train2 > "$L/upload_train2.log" 2>&1
echo "train2 features upload exit $? $(date)"
echo "all done $(date)"
