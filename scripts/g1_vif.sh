#!/usr/bin/env bash
# G1' (2026-11-14): D3 (XCLIP-16, training-free) on ViF-Bench under three time grids:
#   8 fps (D3 protocol, gdrive:DeepVision/vif_8fps_v1), native 16 consecutive frames (vif_native_v1),
#   4 fps clips of P2 (vif_v1). Results -> ~/deepvision_runs/g1/d3_<grid> -> gdrive:DeepVision/runs/g1.
#   nohup nice -n 10 bash scripts/g1_vif.sh > ~/deepvision_runs/p2_logs/main_g1.log 2>&1 < /dev/null & disown
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"
export HF_HOME="$HOME/deepvision_cache/hf" TORCH_HOME="$HOME/deepvision_cache/torch"
L="$HOME/deepvision_runs/p2_logs"
mkdir -p "$L" "$HOME/deepvision_runs/g1"
echo "start $(date) commit $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
for grid in 8fps native 4fps; do
    case $grid in
        8fps) root=gdrive:DeepVision/vif_8fps_v1 ;;
        native) root=gdrive:DeepVision/vif_native_v1 ;;
        4fps) root=gdrive:DeepVision/vif_v1 ;;
    esac
    out="$HOME/deepvision_runs/g1/d3_$grid"
    if [ -f "$out/summary.json" ]; then echo "$grid done already"; continue; fi
    python -m deepvision.baselines.d3 --root "$root" --split vif_test --out "$out" > "$L/g1_d3_$grid.log" 2>&1 \
        || { echo "$grid failed $(date)"; tail -5 "$L/g1_d3_$grid.log"; continue; }
    echo "$grid ok $(date): $(cat "$out/summary.json" | tr -d '\n ')"
done
rclone copy "$HOME/deepvision_runs/g1" gdrive:DeepVision/runs/g1 > "$L/g1_upload.log" 2>&1
rclone copy "$L" gdrive:DeepVision/runs/p2_logs --include "*g1*" >> "$L/g1_upload.log" 2>&1
echo "upload exit $? $(date)"
echo "all done $(date)"
