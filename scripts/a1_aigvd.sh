#!/usr/bin/env bash
# A1 (2026-11-14): AIGVDBench test split (configs/aigvd_{8fps,ctrl8}.yaml) on the server, every step resumes.
#   1. frozen features, 3 encoders x {8fps (DeMamba 10 % crop), ctrl8 (centered square crop)}, 8 even frames,
#      splits aigvd_test and aigvd_int (reals re-encoded / downscaled, gdrive:DeepVision/aigvd_int_<tag>_v1)
#      -> ~/deepvision_features_<tag>/<enc>/aigvd_test -> gdrive:DeepVision/features_<tag>_v1
#   2. zero-shot heads on AIGVDBench (8fps features): p2d_* (trained on GenVideo) and p2v_* (trained on ViF-CoT-4K)
#   3. D3 (XCLIP-16, no training) on aigvd_8fps -> ~/deepvision_runs/a1/d3_aigvd_8fps
#   4. small results -> gdrive:DeepVision/runs/{a1,p2_logs} and the crosseval files of p2d_*/p2v_*
#   nohup nice -n 10 bash scripts/a1_aigvd.sh > ~/deepvision_runs/p2_logs/main_a1.log 2>&1 < /dev/null & disown
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"
export HF_HOME="$HOME/deepvision_cache/hf" TORCH_HOME="$HOME/deepvision_cache/torch"
L="$HOME/deepvision_runs/p2_logs"
ENC="vit_b16 dinov2_s14 xclip_b32"
S=aigvd_test
mkdir -p "$L" "$HOME/deepvision_runs/a1"
echo "start $(date) commit $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
df -h "$HOME" | tail -1
for tag in 8fps ctrl8; do
    for S2 in aigvd_test aigvd_int; do
        root=gdrive:DeepVision/aigvd_${tag}_v1
        [ $S2 = aigvd_int ] && root=gdrive:DeepVision/aigvd_int_${tag}_v1
        if [ -z "$(rclone lsf "$root/shards/${S2}_index.csv" 2>/dev/null)" ]; then
            echo "$tag $S2: no shards on Drive, skipped"; continue
        fi
        python -m deepvision.features.extract extract --root "$root" --split $S2 --encoders $ENC \
            --out ~/deepvision_features_$tag > "$L/a1_extract_${tag}_$S2.log" 2>&1 \
            || { echo "$tag $S2 extract failed $(date)"; tail -3 "$L/a1_extract_${tag}_$S2.log"; continue; }
        python -m deepvision.features.extract upload --out ~/deepvision_features_$tag --split $S2 --encoders $ENC \
            --remote gdrive:DeepVision/features_${tag}_v1 > "$L/a1_upload_${tag}_$S2.log" 2>&1
        echo "$tag $S2 extracted, upload exit $? $(date)"
    done
done
H="python -m deepvision.features.heads"
for S2 in aigvd_test aigvd_int; do
    for p in p2d p2v; do
        $H --crosseval $p --test-on $S2 --crosseval-tag $S2 --heads mlp --features $ENC --out ~/deepvision_features_8fps \
            > "$L/a1_crosseval_${p}_$S2.log" 2>&1 || echo "crosseval $p $S2 failed $(date)"
        $H --summary --prefix $p --crosseval-tag $S2 >> "$L/a1_crosseval_${p}_$S2.log" 2>&1
        echo "crosseval $p $S2 exit $? $(date)"
    done
done
out="$HOME/deepvision_runs/a1/d3_aigvd_8fps"
[ -f "$out/summary.json" ] || python -m deepvision.baselines.d3 --root gdrive:DeepVision/aigvd_8fps_v1 --split $S --no-features \
    --out "$out" > "$L/a1_d3.log" 2>&1
echo "d3: $(cat "$out/summary.json" 2>/dev/null | tr -d '\n ')"
R="$HOME/deepvision_runs"
rclone copy "$R/a1" gdrive:DeepVision/runs/a1 --exclude "*.npy" > "$L/a1_upload_runs.log" 2>&1
rclone copy "$R" gdrive:DeepVision/runs --include "p2[dv]_*/*_aigvd_*" --include "p2[dv]_aigvd_*_summary.csv" \
    >> "$L/a1_upload_runs.log" 2>&1
rclone copy "$L" gdrive:DeepVision/runs/p2_logs --include "*a1*" >> "$L/a1_upload_runs.log" 2>&1
echo "all done $(date)"
