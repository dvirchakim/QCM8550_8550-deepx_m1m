#!/bin/sh
export ADSP_LIBRARY_PATH="/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp"
export XDG_RUNTIME_DIR=/run/user/root/
export WAYLAND_DISPLAY=wayland-1

pkill -9 gst-launch-1.0 2>/dev/null
sleep 1

echo "=== Camera default caps probe ==="
timeout 4 gst-launch-1.0 -v qtiqmmfsrc camera=0 ! fakesink num-buffers=2 silent=false 2>&1 | grep -E "caps|Pipeline|ERROR" | head -20

sleep 2
pkill -9 gst-launch-1.0 2>/dev/null

echo ""
echo "=== Test: qtivtransform with smaller res ==="
timeout 5 gst-launch-1.0 qtiqmmfsrc camera=0 ! qtivtransform ! \
    "video/x-raw,format=NV12,width=1280,height=720,framerate=30/1" ! fakesink num-buffers=10 silent=false 2>&1 | tail -8

sleep 2
pkill -9 gst-launch-1.0 2>/dev/null

echo ""
echo "=== Test: requesting matching native: image/jpeg 640x480 30 ==="
timeout 5 gst-launch-1.0 qtiqmmfsrc camera=0 ! \
    "image/jpeg,width=640,height=480,framerate=30/1" ! \
    jpegdec ! videoconvert ! "video/x-raw,format=NV12" ! fakesink num-buffers=10 silent=false 2>&1 | tail -8

sleep 2
pkill -9 gst-launch-1.0 2>/dev/null

echo ""
echo "=== Cameras enumerated ==="
ls /dev/video*
