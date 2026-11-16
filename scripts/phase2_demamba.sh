#!/usr/bin/env bash
# Phase 2 under the GenVideo protocol of DeMamba (2026-11-09): MLP head on the frozen features of every encoder,
# splits_v3 (train + train2, test only), checkpoint picked on overall test accuracy at 0.5 as upstream does, and the
# per-generator R / F1 / AP averaged over the generators (eval.metrics.demamba_protocol). Finished runs are skipped.
#   nohup nice -n 10 bash scripts/phase2_demamba.sh > ~/deepvision_runs/p2_logs/main_p2d.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"
L="$HOME/deepvision_runs/p2_logs"
mkdir -p "$L"
echo "start $(date) commit $(git rev-parse --short HEAD)"
python -m deepvision.features.heads --splits gdrive:DeepVision/dataset_v1/manifests/splits_v3.csv --heads mlp \
    --set select=test/acc --prefix p2d > "$L/heads_p2d.log" 2>&1 || { echo "heads failed $(date)"; exit 1; }
python -m deepvision.features.heads --prefix p2d --summary > "$L/summary_p2d.txt" 2>&1
echo "heads done $(date)"
rclone copy "$HOME/deepvision_runs" gdrive:DeepVision/runs --filter "- p2d_*/ckpt/**" --filter "+ p2d_*/**" \
    --filter "+ p2d_summary.csv" --filter "+ p2_logs/*p2d*" --filter "- **" > "$L/upload_runs_p2d.log" 2>&1
echo "runs upload exit $? $(date)"
echo "all done $(date)"
