#!/usr/bin/env bash
# Side branch SM (2026-11-13): Scout selection + MIL on frozen vit_b16 (16 frames), ViF only.
# Needs the even store (~/deepvision_features) and the odd store (~/deepvision_features_odd) of vit_b16 vif_{train,test}
# (made by scripts/t0_vif.sh); fetched from Drive when missing. Finished runs are skipped, rerun to continue.
#   nohup nice -n 10 bash scripts/sm_vif.sh > ~/deepvision_runs/p2_logs/main_sm.log 2>&1 < /dev/null & disown
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"
L="$HOME/deepvision_runs/p2_logs"
mkdir -p "$L"
echo "start $(date) commit $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
for s in vif_train vif_test; do
    for pair in "deepvision_features features_v1" "deepvision_features_odd features_odd_v1"; do
        set -- $pair
        d="$HOME/$1/vit_b16/$s"
        if ! ls "$d"/"$s"-*.csv > /dev/null 2>&1; then
            rclone copy "gdrive:DeepVision/$2/vit_b16/$s" "$d" && echo "fetched $2/vit_b16/$s"
        fi
    done
done

python scripts/run_tests.py tests/test_scout.py > "$L/sm_tests.log" 2>&1 || { echo "tests failed $(date)"; exit 1; }
tail -1 "$L/sm_tests.log"
python -m deepvision.features.scout --seeds 0 1 2 > "$L/sm_train.log" 2>&1 || { echo "train failed $(date)"; exit 1; }
python -m deepvision.features.scout --summary > "$L/sm_summary.log" 2>&1 || { echo "summary failed $(date)"; exit 1; }
rclone copy "$HOME/deepvision_runs/sm" gdrive:DeepVision/runs/sm --exclude "*.pt" > "$L/sm_upload.log" 2>&1
rclone copy "$L" gdrive:DeepVision/runs/p2_logs --include "*sm_*" >> "$L/sm_upload.log" 2>&1
echo "upload exit $? $(date)"
echo "all done $(date)"
