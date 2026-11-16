#!/usr/bin/env bash
# GenVideo processing intervention (2026-11-15, configs/genvideo_int.yaml): 500 train2 Youku reals as orig / reenc /
# fps3 / cut2s / cut2s_fps8, same clip path as dataset_v1. Every step resumes.
#   1. frozen features (GenVideo grid, 8 even frames) for the encoders of the p2f heads -> ~/deepvision_features
#   2. zero-shot p2f MLP heads (trained on GenVideo splits_v2) -> p2f_*/predictions_genvideo_int.csv
#   3. D3 (XCLIP-16, no training) -> ~/deepvision_runs/c1g/d3_genvideo_int
#   4. small results -> gdrive:DeepVision/runs/{c1g,p2_logs} and the crosseval files of p2f_*
#   nohup nice -n 10 bash scripts/genvideo_int.sh > ~/deepvision_runs/p2_logs/main_c1g.log 2>&1 < /dev/null & disown
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"
export HF_HOME="$HOME/deepvision_cache/hf" TORCH_HOME="$HOME/deepvision_cache/torch"
L="$HOME/deepvision_runs/p2_logs"
R="$HOME/deepvision_runs"
ENC="resnet50 dinov2_s14 xclip_b32 vit_ti16"
S=genvideo_int
ROOT=gdrive:DeepVision/genvideo_int_v1
mkdir -p "$L" "$R/c1g"
echo "start $(date) commit $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
df -h "$HOME" | tail -1
until [ -n "$(rclone lsf "$ROOT/shards/${S}_index.csv" 2>/dev/null)" ]; do
    echo "waiting for shards on $ROOT $(date)"; sleep 600
done
busy() {  # 4 GiB RAM cgroup: never run two GPU jobs of ours at once (the C2 script may still be running)
    while pgrep -u "$USER" -f "deepvision.(features.extract extract|baselines.d3|features.heads)" > /dev/null; do
        echo "waiting for our other GPU job $(date)"; sleep 300
    done
}
busy
python -m deepvision.features.extract extract --root "$ROOT" --split $S --encoders $ENC --out ~/deepvision_features \
    > "$L/c1g_extract.log" 2>&1 || { echo "extract failed $(date)"; tail -3 "$L/c1g_extract.log"; }
python -m deepvision.features.extract upload --out ~/deepvision_features --split $S --encoders $ENC \
    --remote gdrive:DeepVision/features_v1 > "$L/c1g_upload.log" 2>&1
echo "features exit $? $(date)"
H="python -m deepvision.features.heads"
busy
$H --crosseval p2f --test-on $S --crosseval-tag $S --heads mlp --features $ENC --out ~/deepvision_features \
    > "$L/c1g_crosseval.log" 2>&1 || echo "crosseval failed $(date)"
$H --summary --prefix p2f --crosseval-tag $S >> "$L/c1g_crosseval.log" 2>&1
echo "crosseval exit $? $(date)"
out="$R/c1g/d3_$S"
busy
[ -f "$out/summary.json" ] || python -m deepvision.baselines.d3 --root "$ROOT" --split $S --no-features --out "$out" \
    > "$L/c1g_d3.log" 2>&1
echo "d3: $(cat "$out/summary.json" 2>/dev/null | tr -d '\n ')"
rclone copy "$R/c1g" gdrive:DeepVision/runs/c1g --exclude "*.npy" > "$L/c1g_upload_runs.log" 2>&1
rclone copy "$R" gdrive:DeepVision/runs --include "p2f_*/*_${S}*" --include "p2f_${S}_*summary.csv" \
    >> "$L/c1g_upload_runs.log" 2>&1
rclone copy "$L" gdrive:DeepVision/runs/p2_logs --include "*c1g*" >> "$L/c1g_upload_runs.log" 2>&1
echo "all done $(date)"
