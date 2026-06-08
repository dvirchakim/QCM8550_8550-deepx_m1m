#!/bin/sh
pkill -9 gst-launch-1.0 2>/dev/null
sleep 1
echo "=== Running ORIGINAL imdt-deepx-demo.sh for 15 seconds ==="
timeout 15 sh /usr/share/dx-stream/imdt-deepx-demo.sh > /tmp/test_orig.log 2>&1
echo "exit=$?"
echo
echo "=== Last 30 lines of log ==="
tail -30 /tmp/test_orig.log
echo
echo "=== gst-launch still alive? ==="
ps -ef | awk '/gst-launch/&&!/awk/{print}'
pkill -9 gst-launch-1.0 2>/dev/null
