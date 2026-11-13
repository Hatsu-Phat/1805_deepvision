#!/usr/bin/env bash
# G3 (2026-11-14): frozen features for the standard vs processing-controlled ViF clips, both on the 8 fps grid:
#   8fps  = gdrive:DeepVision/vif_8fps_v1  (DeMamba 10 % crop + squash resize, as P2)
#   ctrl8 = gdrive:DeepVision/vif_ctrl8_v1 (centered square crop -> 224)
# 3 encoders x {vif_train, vif_test}, 8 even frames -> ~/deepvision_features_<tag> -> gdrive:DeepVision/features_<tag>_v1.
# Probes run on the laptop (reports/dataset_stats/scripts/g3_probes.py). D3 on ctrl8 vif_test -> runs/g1/d3_ctrl8.
#   nohup nice -n 10 bash scripts/g3_vif.sh > ~/deepvision_runs/p2_logs/main_g3.log 2>&1 < /dev/null & disown
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"
export HF_HOME="$HOME/deepvision_cache/hf" TORCH_HOME="$HOME/deepvision_cache/torch"
L="$HOME/deepvision_runs/p2_logs"
ENC="vit_b16 dinov2_s14 xclip_b32"
mkdir -p "$L"
echo "start $(date) commit $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
for tag in ctrl8 8fps; do
    root=gdrive:DeepVision/vif_${tag}_v1
    for s in vif_test vif_train; do
        if ! rclone lsf "$root/shards/${s}_index.csv" > /dev/null 2>&1 || [ -z "$(rclone lsf "$root/shards/${s}_index.csv")" ]; then
            echo "$tag $s: no shards on Drive, skipped"; continue
        fi
        python -m deepvision.features.extract extract --root "$root" --split "$s" --encoders $ENC \
            --out ~/deepvision_features_$tag > "$L/g3_extract_${tag}_$s.log" 2>&1 \
            || { echo "$tag $s extract failed $(date)"; tail -3 "$L/g3_extract_${tag}_$s.log"; continue; }
        python -m deepvision.features.extract upload --out ~/deepvision_features_$tag --split "$s" --encoders $ENC \
            --remote gdrive:DeepVision/features_${tag}_v1 > "$L/g3_upload_${tag}_$s.log" 2>&1
        echo "$tag $s extracted, upload exit $? $(date)"
    done
done
out="$HOME/deepvision_runs/g1/d3_ctrl8"
[ -f "$out/summary.json" ] || python -m deepvision.baselines.d3 --root gdrive:DeepVision/vif_ctrl8_v1 --split vif_test \
    --out "$out" > "$L/g1_d3_ctrl8.log" 2>&1
echo "d3 ctrl8: $(cat "$out/summary.json" 2>/dev/null | tr -d '\n ')"
rclone copy "$HOME/deepvision_runs/g1" gdrive:DeepVision/runs/g1 > "$L/g3_upload_runs.log" 2>&1
rclone copy "$L" gdrive:DeepVision/runs/p2_logs --include "*g3*" --include "*g1*" >> "$L/g3_upload_runs.log" 2>&1
echo "all done $(date)"
