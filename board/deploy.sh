#!/bin/bash
# deploy.sh — push all board artifacts and activate services
# Usage:  bash board/deploy.sh [serial]
# e.g.:   bash board/deploy.sh a9ef4ffe

set -e
export MSYS_NO_PATHCONV=1   # prevent Git Bash from converting /data/... to Windows paths
ADB="C:/platform-tools/platform-tools/adb.exe"
SER=${1:-}
[ -n "$SER" ] && ADB="$ADB -s $SER"

ROOT="$(cd "$(dirname "$0")/.." && pwd -W)"  # pwd -W gives C:/... on Git Bash

echo "=== Deploying to board ($SER) ==="

# Ensure adb runs as root (needed for /lib/systemd/system writes)
$ADB root && sleep 1

# ---------- scripts ----------
for f in deepx_dual_demo.py dual_deepx_worker.py easyocr_worker.py easyocr_post.py \
          pose_worker.py seg_worker.py face_worker.py probe_seg.py test_htp_seg.sh \
          seg_inf.json seg_pre.json seg_post.json yolo26_parallel.sh deepx_dual.sh; do
    $ADB push "$ROOT/board/scripts/$f" "/data/local/tmp/$f"
done

# ---------- demo picker ----------
$ADB push "$ROOT/src/demo_picker.py" "/data/local/tmp/demo_picker"
$ADB push "$ROOT/src/demo_picker.py" "/data/local/tmp/demo_picker.py"

# ---------- systemd services ----------
# Push to /etc/systemd/system/ — takes precedence over /lib/systemd/system/ on this board
for svc in demo-picker.service imdt-deepx-demo.service yolo26-parallel.service deepx-dual.service; do
    $ADB push "$ROOT/board/systemd/$svc" "/etc/systemd/system/$svc"
done

# ---------- permissions ----------
$ADB shell chmod +x /data/local/tmp/demo_picker \
                    /data/local/tmp/test_htp_seg.sh \
                    /data/local/tmp/yolo26_parallel.sh \
                    /data/local/tmp/deepx_dual.sh

# ---------- reload & enable ----------
$ADB shell systemctl daemon-reload
$ADB shell systemctl enable demo-picker.service
$ADB shell systemctl restart demo-picker.service

echo "=== Done. demo-picker is running. ==="
echo "    journalctl -u demo-picker.service -f"
