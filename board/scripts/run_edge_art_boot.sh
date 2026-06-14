#!/bin/sh
# Boot-time launch of the C++ edge_art binary (launched by demo-picker tile).
#
# IMPORTANT: this script deliberately does NOT restart qmmf-server.
# On this board, a *manual* `systemctl restart qmmf-server` puts the camera
# HAL into a state where running qtiqmmfsrc (camera) and waylandsink (display)
# as separate GStreamer contexts crashes qmmf-server with:
#     g_strlcat: assertion 'src != NULL' failed   (libqmmf_camera_adaptor.so)
#     UndefinedBehaviorSanitizer: SEGV  (NULL function pointer)
# At clean boot the camera HAL is freshly initialized by the normal systemd
# ordering (After=qmmf-server.service), and edge_art's camera+display coexist
# fine. We preserve that fresh state by never restarting qmmf-server here.
#
# If the camera ever gets stuck after repeated qmmf crashes, only a full
# physical power-cycle clears it (an `adb reboot` is not always enough).
export XDG_RUNTIME_DIR=/run/user/root
export WAYLAND_DISPLAY=wayland-1
export QT_QPA_PLATFORM=wayland-egl
export QT_WAYLAND_SHELL_INTEGRATION=wl-shell
export SLOT_SUFFIX=_a
export ADSP_LIBRARY_PATH="/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp"

# Load DEEPX driver (no-op if already loaded)
cd /lib/modules/$(uname -r)/extra/
insmod dxrt_driver.ko 2>/dev/null || true

# Make sure nothing else is holding the cameras (do NOT touch qmmf-server)
systemctl stop imdt-deepx-demo.service 2>/dev/null || true
systemctl stop yolo26-parallel.service 2>/dev/null || true
pkill -9 gst-launch-1.0 2>/dev/null || true
sleep 1

exec /data/local/tmp/edge_art
