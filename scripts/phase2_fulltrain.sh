#!/usr/bin/env bash
# Phase 2 heads on splits_v2: every train video is trained on, checkpoints are selected and evaluated on test
# (user decision 2026-11-06). The features of phase2_full.sh are reused; finished runs are skipped on a rerun.
#   nohup nice -n 10 bash scripts/phase2_fulltrain.sh > ~/deepvision_runs/p2_logs/main_p2f.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"
L="$HOME/deepvision_runs/p2_logs"
mkdir -p "$L"
echo "start $(date) commit $(git rev-parse --short HEAD)"

python -m deepvision.features.heads --splits gdrive:DeepVision/dataset_v1/manifests/splits_v2.csv --prefix p2f \
    > "$L/heads_p2f.log" 2>&1 || { echo "heads failed $(date)"; exit 1; }
python -m deepvision.features.heads --prefix p2f --summary > "$L/summary_p2f.txt" 2>&1
echo "heads done $(date)"

rclone copy "$HOME/deepvision_runs" gdrive:DeepVision/runs --filter "- p2f_*/ckpt/**" --filter "+ p2f_*/**" \
    --filter "+ p2f_summary.csv" --filter "+ p2_logs/*p2f*" --filter "- **" > "$L/upload_runs_p2f.log" 2>&1
echo "runs upload exit $? $(date)"
