#!/usr/bin/env bash
# T0' (2026-11-13): S(dt) at the source frame rate on ViF-Bench. Shards: gdrive:DeepVision/vif_native_v1
# (configs/vif_native.yaml, 16 consecutive native frames). Even + odd frames of 3 encoders ->
# ~/deepvision_features_native{,_odd} (uploaded to gdrive:DeepVision/features_native_v1{,_odd}),
# then features.structure -> ~/deepvision_runs/t0n -> gdrive:DeepVision/runs/t0n. Rerun to resume.
#   nohup nice -n 10 bash scripts/t0n_vif.sh > ~/deepvision_runs/p2_logs/main_t0n.log 2>&1 < /dev/null & disown
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"
export HF_HOME="$HOME/deepvision_cache/hf" TORCH_HOME="$HOME/deepvision_cache/torch"
L="$HOME/deepvision_runs/p2_logs"
ENC="vit_b16 dinov2_s14 xclip_b32"
ROOT=gdrive:DeepVision/vif_native_v1
mkdir -p "$L"
echo "start $(date) commit $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

for f in even odd; do
    sfx=$([ "$f" = odd ] && echo _odd || echo "")
    python -m deepvision.features.extract extract --frames "$f" --out ~/deepvision_features_native$sfx \
        --root "$ROOT" --split vif_test --encoders $ENC > "$L/t0n_extract_$f.log" 2>&1 \
        || { echo "$f extract failed $(date)"; exit 1; }
    python -m deepvision.features.extract upload --out ~/deepvision_features_native$sfx \
        --remote gdrive:DeepVision/features_native_v1$sfx --split vif_test --encoders $ENC > "$L/t0n_upload_$f.log" 2>&1
    echo "$f extracted, upload exit $? $(date)"
done

python -m deepvision.features.structure --encoders $ENC --split vif_test --root "$ROOT" \
    --even ~/deepvision_features_native --odd ~/deepvision_features_native_odd --dest ~/deepvision_runs/t0n \
    > "$L/t0n_structure.log" 2>&1 || { echo "structure failed $(date)"; exit 1; }
rclone copy "$HOME/deepvision_runs/t0n" gdrive:DeepVision/runs/t0n > "$L/t0n_upload_runs.log" 2>&1
rclone copy "$L" gdrive:DeepVision/runs/p2_logs --include "*t0n*" >> "$L/t0n_upload_runs.log" 2>&1
echo "runs upload exit $? $(date)"
echo "all done $(date)"
