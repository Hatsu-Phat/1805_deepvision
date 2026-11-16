#!/usr/bin/env bash
# GenVideo under C2 v2 (2026-11-15; configs/genvideo_c2v2.yaml, configs/genvideo_int_c2v2.yaml). Every step resumes.
#   1. 3 encoders x {GenVideo-C2 test (split test of gdrive:DeepVision/genvideo_c2v2_v1), GenVideo intervention on the
#      C2 grid (split genvideo_int of genvideo_int_c2v2_v1)} -> ~/deepvision_features_c2v2_gv -> features_c2v2_v1/<enc>/
#      {genvideo_test, genvideo_int} (renamed on upload so they never mix with the GenVideo dataset_v1 features)
#   2. D3 (XCLIP-16, no training) on both -> ~/deepvision_runs/c2v2/d3_genvideo_{test,int}
#   nohup nice -n 10 bash scripts/c2_genvideo.sh > ~/deepvision_runs/p2_logs/main_c2v2_gv.log 2>&1 < /dev/null & disown
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"
export HF_HOME="$HOME/deepvision_cache/hf" TORCH_HOME="$HOME/deepvision_cache/torch"
L="$HOME/deepvision_runs/p2_logs"
R="$HOME/deepvision_runs"
ENC="vit_b16 dinov2_s14 xclip_b32"
F="$HOME/deepvision_features_c2v2_gv"
mkdir -p "$L" "$R/c2v2"
echo "start $(date) commit $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
df -h "$HOME" | tail -1
for item in genvideo_c2v2:test:genvideo_test genvideo_int_c2v2:genvideo_int:genvideo_int; do
    IFS=: read -r ds S2 name <<< "$item"
    root=gdrive:DeepVision/${ds}_v1
    until [ -n "$(rclone lsf "$root/shards/${S2}_index.csv" 2>/dev/null)" ]; do
        echo "$name: waiting for shards on $root $(date)"; sleep 600
    done
    python -m deepvision.features.extract extract --root "$root" --split $S2 --encoders $ENC --out "$F" \
        > "$L/c2v2_gv_extract_$name.log" 2>&1 || { echo "$name extract failed $(date)"; continue; }
    for e in $ENC; do
        rclone copy "$F/$e/$S2" "gdrive:DeepVision/features_c2v2_v1/$e/$name" >> "$L/c2v2_gv_upload_$name.log" 2>&1
        rclone check "$F/$e/$S2" "gdrive:DeepVision/features_c2v2_v1/$e/$name" --one-way >> "$L/c2v2_gv_upload_$name.log" 2>&1 \
            && echo "$name $e uploaded+checked" || echo "$name $e upload check FAILED"
    done
    out="$R/c2v2/d3_$name"
    [ -f "$out/summary.json" ] || python -m deepvision.baselines.d3 --root "$root" --split $S2 --no-features --out "$out" \
        > "$L/c2v2_gv_d3_$name.log" 2>&1
    echo "$name d3: $(cat "$out/summary.json" 2>/dev/null | tr -d '\n ')"
done
rclone copy "$R/c2v2" gdrive:DeepVision/runs/c2v2 --exclude "*.npy" > "$L/c2v2_gv_upload_runs.log" 2>&1
rclone copy "$L" gdrive:DeepVision/runs/p2_logs --include "*c2v2_gv*" >> "$L/c2v2_gv_upload_runs.log" 2>&1
echo "all done $(date)"
