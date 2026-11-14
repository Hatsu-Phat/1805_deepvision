#!/usr/bin/env bash
# C2 v1 (2026-11-14): frozen features of the C2 clips (configs/{vif,aigvd,aigvd_int}_c2.yaml: 8 fps grid, short side
# -> 224, square crop, C2 floor on every split) for the C2 gates; every step resumes.
#   1. 3 encoders x splits vif_train, vif_test (gdrive:DeepVision/vif_c2_v1), aigvd_test (aigvd_c2_v1),
#      aigvd_int (aigvd_int_c2_v1), 8 even frames -> ~/deepvision_features_c2 -> gdrive:DeepVision/features_c2_v1
#   2. D3 (XCLIP-16, no training) on aigvd_c2 -> ~/deepvision_runs/c2/d3_aigvd_c2 (residual temporal leak check)
#   nohup nice -n 10 bash scripts/c2_features.sh > ~/deepvision_runs/p2_logs/main_c2.log 2>&1 < /dev/null & disown
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"
export HF_HOME="$HOME/deepvision_cache/hf" TORCH_HOME="$HOME/deepvision_cache/torch"
L="$HOME/deepvision_runs/p2_logs"
ENC="vit_b16 dinov2_s14 xclip_b32"
mkdir -p "$L" "$HOME/deepvision_runs/c2"
echo "start $(date) commit $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
df -h "$HOME" | tail -1
for pair in vif_c2:vif_train vif_c2:vif_test aigvd_int_c2:aigvd_int aigvd_c2:aigvd_test; do
    root=gdrive:DeepVision/${pair%%:*}_v1
    S2=${pair##*:}
    until [ -n "$(rclone lsf "$root/shards/${S2}_index.csv" 2>/dev/null)" ]; do
        echo "$S2: waiting for shards on $root $(date)"; sleep 600
    done
    python -m deepvision.features.extract extract --root "$root" --split $S2 --encoders $ENC \
        --out ~/deepvision_features_c2 > "$L/c2_extract_$S2.log" 2>&1 \
        || { echo "$S2 extract failed $(date)"; tail -3 "$L/c2_extract_$S2.log"; continue; }
    python -m deepvision.features.extract upload --out ~/deepvision_features_c2 --split $S2 --encoders $ENC \
        --remote gdrive:DeepVision/features_c2_v1 > "$L/c2_upload_$S2.log" 2>&1
    echo "$S2 extracted, upload exit $? $(date)"
done
out="$HOME/deepvision_runs/c2/d3_aigvd_c2"
[ -f "$out/summary.json" ] || python -m deepvision.baselines.d3 --root gdrive:DeepVision/aigvd_c2_v1 --split aigvd_test \
    --no-features --out "$out" > "$L/c2_d3.log" 2>&1
echo "d3: $(cat "$out/summary.json" 2>/dev/null | tr -d '\n ')"
rclone copy "$HOME/deepvision_runs/c2" gdrive:DeepVision/runs/c2 --exclude "*.npy" > "$L/c2_upload_runs.log" 2>&1
rclone copy "$L" gdrive:DeepVision/runs/p2_logs --include "*c2*" >> "$L/c2_upload_runs.log" 2>&1
echo "all done $(date)"
