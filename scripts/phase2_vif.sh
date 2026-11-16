#!/usr/bin/env bash
# Cross-dataset test on ViF (Skyra, 2026-11-09). Shards: gdrive:DeepVision/vif_v1 (vif_train = ViF-CoT-4K,
# vif_test = ViF-Bench). Extract the features of the 7 encoders, upload them to features_v1/<encoder>/vif_*, then
#   p2d -> vif_test  zero-shot: the MLP heads trained on GenVideo (p2d) scored on ViF-Bench, no training
#   p2v              MLP trained on ViF-CoT-4K, tested on ViF-Bench (checkpoint picked on its paired mean bAcc)
# both with the paired protocol of Skyra (eval.metrics.vif_paired_protocol). Every step resumes, rerun to continue.
#   nohup nice -n 10 bash scripts/phase2_vif.sh > ~/deepvision_runs/p2_logs/main_p2v.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"
export HF_HOME="$HOME/deepvision_cache/hf" TORCH_HOME="$HOME/deepvision_cache/torch"
L="$HOME/deepvision_runs/p2_logs"
mkdir -p "$L"
echo "start $(date) commit $(git rev-parse --short HEAD)"

for s in vif_test vif_train; do
    python -m deepvision.features.extract extract --root gdrive:DeepVision/vif_v1 --split "$s" > "$L/extract_$s.log" 2>&1 \
        || { echo "$s extract failed $(date)"; exit 1; }
    python -m deepvision.features.extract upload --split "$s" > "$L/upload_$s.log" 2>&1
    echo "$s extracted, upload exit $? $(date)"
done

python -m deepvision.features.heads --crosseval p2d --test-on vif_test --paired --heads mlp \
    > "$L/heads_p2d_vif_test.log" 2>&1 || { echo "zero-shot failed $(date)"; exit 1; }
python -m deepvision.features.heads --summary --prefix p2d --crosseval-tag vif_test > "$L/summary_p2d_vif_test.txt" 2>&1
echo "zero-shot done $(date)"

python -m deepvision.features.heads --train-on vif_train --test-on vif_test --paired --heads mlp \
    --set select=test/vif_avg_bacc --set batch_size=256 --prefix p2v > "$L/heads_p2v.log" 2>&1 \
    || { echo "p2v heads failed $(date)"; exit 1; }
python -m deepvision.features.heads --summary --prefix p2v > "$L/summary_p2v.txt" 2>&1
echo "p2v done $(date)"

rclone copy "$HOME/deepvision_runs" gdrive:DeepVision/runs --filter "- */ckpt/**" --filter "+ p2v_*/**" \
    --filter "+ p2d_*/*vif_test*" --filter "+ p2v_summary.csv" --filter "+ p2d_vif_test_summary.csv" \
    --filter "+ p2_logs/*vif*" --filter "+ p2_logs/*p2v*" --filter "- **" > "$L/upload_runs_p2v.log" 2>&1
echo "runs upload exit $? $(date)"
echo "all done $(date)"
