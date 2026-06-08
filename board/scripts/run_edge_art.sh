#!/bin/sh
# Run on device: sh /data/local/tmp/run_edge_art.sh
export XDG_RUNTIME_DIR=/run/user/root
export WAYLAND_DISPLAY=wayland-1
export QT_QPA_PLATFORM=wayland-egl
export QT_WAYLAND_SHELL_INTEGRATION=wl-shell
export ADSP_LIBRARY_PATH="/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp"

# Load DEEPX driver
cd /lib/modules/$(uname -r)/extra/
insmod dxrt_driver.ko 2>/dev/null || true

# Stop service that holds cameras
systemctl stop imdt-deepx-demo.service 2>/dev/null || true
pkill -9 gst-launch-1.0 2>/dev/null || true
sleep 1

# Restart camera server
systemctl restart qmmf-server.service 2>/dev/null || true
sleep 2

exec /data/local/tmp/edge_art
