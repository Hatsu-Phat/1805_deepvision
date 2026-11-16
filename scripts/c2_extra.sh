#!/usr/bin/env bash
# C2 v2 extra originals (2026-11-16, configs/aigvd_c2v2x.yaml): the 144 AIGVDBench intervention reals whose original
# clip was missing from aigvd_c2v2 -> frozen features of the C2 encoders -> gdrive:DeepVision/features_c2v2x_v1.
# Starts after the queued GenVideo jobs (4 GiB RAM cgroup: one GPU job of ours at a time). Resumes.
#   nohup nice -n 10 bash scripts/c2_extra.sh > ~/deepvision_runs/p2_logs/main_c2v2x.log 2>&1 < /dev/null & disown
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"
export HF_HOME="$HOME/deepvision_cache/hf" TORCH_HOME="$HOME/deepvision_cache/torch"
L="$HOME/deepvision_runs/p2_logs"
ENC="vit_b16 dinov2_s14 xclip_b32"
ROOT=gdrive:DeepVision/aigvd_c2v2x_v1
OUT="$HOME/deepvision_features_c2v2x"
echo "start $(date) commit $(git rev-parse --short HEAD)"
until grep -q "all done" "$L/main_c2v2_gv.log" 2>/dev/null && [ -n "$(rclone lsf "$ROOT/shards/aigvd_test_index.csv" 2>/dev/null)" ]; do
    echo "waiting for the GenVideo-C2 job and the shards $(date)"; sleep 600
done
while pgrep -u "$USER" -f "deepvision.(features.extract extract|baselines.d3|features.heads)" > /dev/null; do
    echo "waiting for our other GPU job $(date)"; sleep 300
done
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
python -m deepvision.features.extract extract --root "$ROOT" --split aigvd_test --encoders $ENC --out "$OUT" \
    > "$L/c2v2x_extract.log" 2>&1 || { echo "extract failed $(date)"; tail -3 "$L/c2v2x_extract.log"; }
python -m deepvision.features.extract upload --out "$OUT" --split aigvd_test --encoders $ENC \
    --remote gdrive:DeepVision/features_c2v2x_v1 > "$L/c2v2x_upload.log" 2>&1
echo "upload exit $? $(date)"
rclone copy "$L" gdrive:DeepVision/runs/p2_logs --include "*c2v2x*" > /dev/null 2>&1
echo "all done $(date)"
