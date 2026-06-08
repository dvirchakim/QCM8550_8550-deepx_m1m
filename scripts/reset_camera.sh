#!/bin/sh
echo "[reset] killing any pipeline / camera consumers ..."
pkill -9 gst-launch-1.0 2>/dev/null || true
pkill -9 -f imdt-deepx-demo 2>/dev/null || true
pkill -9 -f live_demo 2>/dev/null || true
pkill -9 -f capture_and_pose 2>/dev/null || true
pkill -9 python3 2>/dev/null || true
sleep 2

echo "[reset] stopping qmmf-related services ..."
systemctl stop qmmf-webserver.service 2>/dev/null
systemctl stop qmmf-server.service
sleep 2

echo "[reset] killing any leftover qmmf-server ..."
pkill -9 -f qmmf-server 2>/dev/null || true
sleep 1

echo "[reset] checking init_qti_graphics ..."
systemctl status init_qti_graphics.service --no-pager 2>&1 | head -5
systemctl restart init_qti_graphics.service 2>&1
sleep 2

echo "[reset] starting qmmf-server fresh ..."
systemctl start qmmf-server.service
sleep 3
systemctl status qmmf-server.service --no-pager 2>&1 | head -10

echo
echo "[reset] testing minimal camera capture ..."
export ADSP_LIBRARY_PATH="/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp"
export XDG_RUNTIME_DIR=/run/user/root/
export WAYLAND_DISPLAY=wayland-1

timeout 6 gst-launch-1.0 -e qtiqmmfsrc camera=0 ! \
    qtivtransform ! video/x-raw,format=NV12,width=1920,height=1080,framerate=15/1 ! \
    fakesink num-buffers=10 silent=false > /tmp/test_post.log 2>&1
echo "exit=$?"
tail -8 /tmp/test_post.log
