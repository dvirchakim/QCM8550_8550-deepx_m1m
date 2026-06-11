#!/bin/sh
# DeepX Dual Demo launcher

export XDG_RUNTIME_DIR=/run/user/root
export WAYLAND_DISPLAY=wayland-1
export QT_QPA_PLATFORM=wayland-egl
export QT_WAYLAND_SHELL_INTEGRATION=wl-shell
export ADSP_LIBRARY_PATH="/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp"

insmod /lib/modules/$(uname -r)/extra/dxrt_driver.ko 2>/dev/null || true

cleanup() {
    kill -9 $child 2>/dev/null
    exit 0
}
trap cleanup TERM INT

while true; do
    python3 /data/local/tmp/deepx_dual_demo.py &
    child=$!
    wait $child
    echo "[deepx-dual] demo exited, restarting in 3s..."
    sleep 3
done
