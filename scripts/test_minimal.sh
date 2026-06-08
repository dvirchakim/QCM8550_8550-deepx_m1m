#!/bin/sh
export ADSP_LIBRARY_PATH="/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp"
export XDG_RUNTIME_DIR=/run/user/root/
export WAYLAND_DISPLAY=wayland-1

pkill -9 gst-launch-1.0 2>/dev/null || true
sleep 1

echo "=== Test 1: minimal camera 0 -> fakesink ==="
timeout 5 gst-launch-1.0 -v qtiqmmfsrc camera=0 ! fakesink > /tmp/test1.log 2>&1
echo "exit=$?"
tail -10 /tmp/test1.log
echo

sleep 2
pkill -9 gst-launch-1.0 2>/dev/null

echo "=== Test 2: original imdt-deepx-demo.sh ==="
echo "(running for 5 seconds)"
timeout 5 sh /usr/share/dx-stream/imdt-deepx-demo.sh > /tmp/test2.log 2>&1
echo "exit=$?"
tail -20 /tmp/test2.log
