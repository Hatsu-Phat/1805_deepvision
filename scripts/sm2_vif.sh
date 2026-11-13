#!/usr/bin/env bash
# Side branch SM, round 2 (2026-11-13): Scout with replicate boundary padding, end to end (scout_attn) and with the
# S0 -> S1 schedule of pipeline v3 (scout_s0s1); u8_mean / rand_attn rerun as references in the same runs root.
# Needs the stores fetched by scripts/sm_vif.sh. Finished runs are skipped, rerun to continue.
#   nohup nice -n 10 bash scripts/sm2_vif.sh > ~/deepvision_runs/p2_logs/main_sm2.log 2>&1 < /dev/null & disown
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"
L="$HOME/deepvision_runs/p2_logs"
R="$HOME/deepvision_runs/sm2"
echo "start $(date) commit $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
python scripts/run_tests.py tests/test_scout.py > "$L/sm2_tests.log" 2>&1 || { echo "tests failed $(date)"; exit 1; }
tail -1 "$L/sm2_tests.log"
python -m deepvision.features.scout --methods u8_mean rand_attn scout_attn scout_s0s1 --runs-root "$R" \
    > "$L/sm2_train.log" 2>&1 || { echo "train failed $(date)"; exit 1; }
python -m deepvision.features.scout --summary --runs-root "$R" > "$L/sm2_summary.log" 2>&1 \
    || { echo "summary failed $(date)"; exit 1; }
rclone copy "$R" gdrive:DeepVision/runs/sm2 --exclude "*.pt" > "$L/sm2_upload.log" 2>&1
rclone copy "$L" gdrive:DeepVision/runs/p2_logs --include "*sm2_*" >> "$L/sm2_upload.log" 2>&1
echo "upload exit $? $(date)"
echo "all done $(date)"
