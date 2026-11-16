#!/usr/bin/env bash
# T0 (2026-11-13): structure function S(dt) of frozen features on ViF, real vs fake.
#  1. extract the 8 odd clip frames (1,3,5,7,8,10,12,14) of vif_test / vif_train for 3 encoders into
#     ~/deepvision_features_odd, upload to gdrive:DeepVision/features_odd_v1 (+ rclone check)
#  2. make sure the even store (features_v1) of the same encoders/splits is local
#  3. features.structure -> ~/deepvision_runs/t0, upload to gdrive:DeepVision/runs/t0
# Every step resumes, rerun to continue.
#   nohup nice -n 10 bash scripts/t0_vif.sh > ~/deepvision_runs/p2_logs/main_t0.log 2>&1 < /dev/null & disown
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"
export HF_HOME="$HOME/deepvision_cache/hf" TORCH_HOME="$HOME/deepvision_cache/torch"
L="$HOME/deepvision_runs/p2_logs"
ENC="vit_b16 dinov2_s14 xclip_b32"
mkdir -p "$L"
echo "start $(date) commit $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
df -h "$HOME" | tail -1

for s in vif_test vif_train; do
    python -m deepvision.features.extract extract --frames odd --out ~/deepvision_features_odd \
        --root gdrive:DeepVision/vif_v1 --split "$s" --encoders $ENC > "$L/t0_extract_odd_$s.log" 2>&1 \
        || { echo "$s odd extract failed $(date)"; exit 1; }
    python -m deepvision.features.extract upload --out ~/deepvision_features_odd \
        --remote gdrive:DeepVision/features_odd_v1 --split "$s" --encoders $ENC > "$L/t0_upload_odd_$s.log" 2>&1
    echo "$s odd extracted, upload exit $? $(date)"
    for e in $ENC; do
        d="$HOME/deepvision_features/$e/$s"
        if ! ls "$d"/"$s"-*.csv > /dev/null 2>&1; then
            rclone copy "gdrive:DeepVision/features_v1/$e/$s" "$d" && echo "fetched even $e/$s"
        fi
    done
done

python -m deepvision.features.structure --encoders $ENC --split vif_test vif_train > "$L/t0_structure.log" 2>&1 \
    || { echo "structure failed $(date)"; exit 1; }
rclone copy "$HOME/deepvision_runs/t0" gdrive:DeepVision/runs/t0 > "$L/t0_upload_runs.log" 2>&1
rclone copy "$L" gdrive:DeepVision/runs/p2_logs --include "*t0*" >> "$L/t0_upload_runs.log" 2>&1
echo "runs upload exit $? $(date)"
echo "all done $(date)"
