#!/bin/sh
# Capture one JPEG frame from qtiqmmfsrc.
set +e
export XDG_RUNTIME_DIR=/run/user/root
export WAYLAND_DISPLAY=wayland-1

pkill -9 gst-launch-1.0 2>/dev/null
sleep 1

rm -f /tmp/cam_grab.jpg
echo "[capture] running gst-launch ..."
timeout 6 gst-launch-1.0 -e \
    qtiqmmfsrc camera=0 ! \
    qtivtransform ! \
    "video/x-raw,format=NV12,width=1280,height=720,framerate=15/1" ! \
    videoconvert ! \
    jpegenc quality=90 ! \
    multifilesink location=/tmp/cam_grab.jpg max-files=1 \
    > /tmp/gst_capture.log 2>&1
ec=$?
echo "[capture] exit=$ec"

echo "=== gst-launch log (last 15 lines) ==="
tail -15 /tmp/gst_capture.log

echo "=== file ==="
ls -la /tmp/cam_grab.jpg 2>&1
