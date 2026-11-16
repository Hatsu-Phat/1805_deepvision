#!/usr/bin/env bash
# Add ViT-B/16 (vit_b16, 2026-11-11) to every Phase 2 table. Every step resumes, rerun to continue.
#   1. features of vit_b16 on test, vif_test, vif_train (small) + cost row, upload each split to features_v1
#   2. p2v_vit_b16 (ViF-CoT-4K -> ViF-Bench), then every p2v head on the GenVideo test (crosseval genvideo_test)
#   3. features of vit_b16 on train + train2 (~210 GiB read from Drive), upload
#   4. p2d_vit_b16 (GenVideo splits_v3 -> GenVideo test, DeMamba protocol), then p2d_vit_b16 on ViF-Bench (zero-shot)
#   5. summaries, upload the small results to gdrive:DeepVision/runs
#   nohup nice -n 10 bash scripts/phase2_vitb.sh > ~/deepvision_runs/p2_logs/main_vitb.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"
export HF_HOME="$HOME/deepvision_cache/hf" TORCH_HOME="$HOME/deepvision_cache/torch"
L="$HOME/deepvision_runs/p2_logs"
mkdir -p "$L"
E=vit_b16
H="python -m deepvision.features.heads"
echo "start $(date) commit $(git rev-parse --short HEAD)"

extract() {  # split root
    python -m deepvision.features.extract extract --root "$2" --split "$1" --encoders $E > "$L/extract_${E}_$1.log" 2>&1 \
        || { echo "$1 extract failed $(date)"; exit 1; }
    python -m deepvision.features.extract upload --split "$1" --encoders $E > "$L/upload_${E}_$1.log" 2>&1
    echo "$1 extracted, upload exit $? $(date)"
}

extract test gdrive:DeepVision/dataset_v1
extract vif_test gdrive:DeepVision/vif_v1
extract vif_train gdrive:DeepVision/vif_v1
python -m deepvision.features.extract cost --encoders $E > "$L/cost_$E.log" 2>&1
echo "cost exit $? $(date)"

$H --train-on vif_train --test-on vif_test --paired --heads mlp --features $E --set select=test/vif_avg_bacc \
    --set batch_size=256 --prefix p2v > "$L/heads_p2v_$E.log" 2>&1 || { echo "p2v failed $(date)"; exit 1; }
$H --crosseval p2v --test-on test --crosseval-tag genvideo_test --heads mlp > "$L/heads_p2v_genvideo_test.log" 2>&1 \
    || { echo "p2v -> genvideo failed $(date)"; exit 1; }
echo "vif heads done $(date)"

extract train gdrive:DeepVision/dataset_v1
extract train2 gdrive:DeepVision/dataset_v1

$H --splits gdrive:DeepVision/dataset_v1/manifests/splits_v3.csv --heads mlp --features $E --set select=test/acc \
    --prefix p2d > "$L/heads_p2d_$E.log" 2>&1 || { echo "p2d failed $(date)"; exit 1; }
$H --crosseval p2d --test-on vif_test --paired --heads mlp --features $E > "$L/heads_p2d_vif_test_$E.log" 2>&1 \
    || { echo "p2d -> vif failed $(date)"; exit 1; }
echo "genvideo heads done $(date)"

$H --summary --prefix p2d > "$L/summary_p2d.txt" 2>&1
$H --summary --prefix p2v > "$L/summary_p2v.txt" 2>&1
$H --summary --prefix p2d --crosseval-tag vif_test > "$L/summary_p2d_vif_test.txt" 2>&1
$H --summary --prefix p2v --crosseval-tag genvideo_test > "$L/summary_p2v_genvideo_test.txt" 2>&1
rclone copy "$HOME/deepvision_runs" gdrive:DeepVision/runs --filter "- */ckpt/**" --filter "+ p2d_vit_b16_*/**" \
    --filter "+ p2v_*/**" --filter "+ p2d_*/*vif_test*" --filter "+ p2d_summary.csv" --filter "+ p2v_summary.csv" \
    --filter "+ p2d_vif_test_summary.csv" --filter "+ p2v_genvideo_test_summary.csv" --filter "+ p2_logs/*vit_b16*" \
    --filter "+ p2_logs/*genvideo_test*" --filter "+ p2_logs/main_vitb.log" --filter "+ p2_logs/summary_*" --filter "- **" \
    > "$L/upload_runs_vitb.log" 2>&1
echo "runs upload exit $? $(date)"
rclone copy "$HOME/deepvision_features/encoder_costs.csv" gdrive:DeepVision/features_v1/ > /dev/null 2>&1
echo "all done $(date)"
