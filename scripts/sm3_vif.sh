#!/usr/bin/env bash
# Side branch SM, round 3 (2026-11-13): S0 -> S1 with cross-fitting (scout_xfit, head and Scout on disjoint halves
# of the train names); u8_mean and scout_s0s1 rerun as references in the same runs root.
# Needs the stores fetched by scripts/sm_vif.sh. Finished runs are skipped, rerun to continue.
#   nohup nice -n 10 bash scripts/sm3_vif.sh > ~/deepvision_runs/p2_logs/main_sm3.log 2>&1 < /dev/null & disown
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"
L="$HOME/deepvision_runs/p2_logs"
R="$HOME/deepvision_runs/sm3"
echo "start $(date) commit $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
python scripts/run_tests.py tests/test_scout.py > "$L/sm3_tests.log" 2>&1 || { echo "tests failed $(date)"; exit 1; }
tail -1 "$L/sm3_tests.log"
python -m deepvision.features.scout --methods u8_mean scout_s0s1 scout_xfit --runs-root "$R" \
    > "$L/sm3_train.log" 2>&1 || { echo "train failed $(date)"; exit 1; }
python -m deepvision.features.scout --summary --runs-root "$R" > "$L/sm3_summary.log" 2>&1 \
    || { echo "summary failed $(date)"; exit 1; }
rclone copy "$R" gdrive:DeepVision/runs/sm3 --exclude "*.pt" > "$L/sm3_upload.log" 2>&1
rclone copy "$L" gdrive:DeepVision/runs/p2_logs --include "*sm3_*" >> "$L/sm3_upload.log" 2>&1
echo "upload exit $? $(date)"
echo "all done $(date)"
