#!/bin/sh
export XDG_RUNTIME_DIR=/run/user/root
export WAYLAND_DISPLAY=wayland-1

pkill -9 gst-launch-1.0 2>/dev/null
sleep 1

echo "[diag] testing BGR pipe (5 sec capture to file)"
timeout 5 gst-launch-1.0 -e \
    qtiqmmfsrc camera=0 ! \
    qtivtransform ! \
    "video/x-raw,format=NV12,width=1280,height=720,framerate=30/1" ! \
    videoconvert ! \
    "video/x-raw,format=BGR" ! \
    filesink location=/tmp/test.bgr \
    > /tmp/diag_cam.log 2>&1
echo "exit=$?"

echo "=== log ==="
tail -25 /tmp/diag_cam.log

echo "=== file ==="
ls -la /tmp/test.bgr
echo "expected bytes per frame: $((1280*720*3)) = 2764800"
