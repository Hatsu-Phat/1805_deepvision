#!/usr/bin/env bash
# Redo the C2 v2 AIGVDBench test features (2026-11-16): the first aigvd_c2v2 manifest kept only 4,875 of 15,155 rows
# (de-duplicated on video_id, which repeats across generators). Waits for the queued GenVideo-C2 job and for the rebuilt
# shards (index >= 15,000 rows), moves the old aigvd_test features / D3 aside (not deleted), then reruns
# scripts/c2_features.sh c2v2 (the vif / aigvd_int features are complete and are skipped).
#   nohup nice -n 10 bash scripts/c2_redo_aigvd.sh > ~/deepvision_runs/p2_logs/main_c2v2_redo.log 2>&1 < /dev/null & disown
set -u
cd "$(dirname "$0")/.."
L="$HOME/deepvision_runs/p2_logs"
IDX=gdrive:DeepVision/aigvd_c2v2_v1/shards/aigvd_test_index.csv
echo "start $(date) commit $(git rev-parse --short HEAD)"
until grep -q "all done" "$L/main_c2v2_gv.log" 2>/dev/null && [ "$(rclone cat "$IDX" 2>/dev/null | wc -l)" -gt 15000 ]; do
    echo "waiting for the GenVideo-C2 job and the rebuilt shards $(date)"; sleep 600
done
B="_bad_4875_$(date +%Y%m%d)"
for e in vit_b16 dinov2_s14 xclip_b32; do
    d="$HOME/deepvision_features_c2v2/$e/aigvd_test"
    [ -d "$d" ] && mv "$d" "$d$B"
done
d="$HOME/deepvision_runs/c2v2/d3_aigvd_c2v2"
[ -d "$d" ] && mv "$d" "$d$B"
rclone moveto gdrive:DeepVision/runs/c2v2/d3_aigvd_c2v2 "gdrive:DeepVision/runs/c2v2/d3_aigvd_c2v2$B" 2>/dev/null
bash scripts/c2_features.sh c2v2
echo "all done $(date)"
