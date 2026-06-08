#!/bin/sh
# YOLO26 Parallel Demo

export XDG_RUNTIME_DIR=/run/user/root
export WAYLAND_DISPLAY=wayland-1
export QT_QPA_PLATFORM=wayland-egl
export QT_WAYLAND_SHELL_INTEGRATION=wl-shell
export ADSP_LIBRARY_PATH="/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp"

insmod /lib/modules/$(uname -r)/extra/dxrt_driver.ko 2>/dev/null || true

cd /usr/share/dx-stream

while true; do
    /data/local/tmp/yolo26_launcher
    echo "Launcher exited, restarting in 2s..."
    sleep 2
done
