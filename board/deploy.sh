#!/bin/bash
# deploy.sh — push all board artifacts and activate services
# Usage:  bash board/deploy.sh [serial]
# e.g.:   bash board/deploy.sh a9ef4ffe

set -e
ADB="C:/platform-tools/platform-tools/adb.exe"
SER=${1:-}
[ -n "$SER" ] && ADB="$ADB -s $SER"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Deploying to board ($SER) ==="

# ---------- scripts ----------
for f in edge_art_genai.py edge_art.py pose_worker.py seg_worker.py face_worker.py \
          probe_seg.py run_genai_demo.sh run_edge_art.sh test_htp_seg.sh \
          seg_inf.json seg_pre.json seg_post.json yolo26_parallel.sh; do
    $ADB push "$ROOT/board/scripts/$f" "/data/local/tmp/$f"
done

# ---------- systemd services ----------
for svc in demo-picker.service edge-art.service imdt-deepx-demo.service; do
    $ADB push "$ROOT/board/systemd/$svc" "/lib/systemd/system/$svc"
done

# ---------- permissions ----------
$ADB shell chmod +x /data/local/tmp/demo_picker \
                    /data/local/tmp/run_genai_demo.sh \
                    /data/local/tmp/run_edge_art.sh \
                    /data/local/tmp/test_htp_seg.sh \
                    /data/local/tmp/yolo26_parallel.sh

# ---------- reload & enable ----------
$ADB shell systemctl daemon-reload
$ADB shell systemctl enable demo-picker.service
$ADB shell systemctl restart demo-picker.service

echo "=== Done. demo-picker is running. ==="
echo "    journalctl -u demo-picker.service -f"
