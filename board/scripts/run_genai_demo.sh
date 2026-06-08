#!/bin/sh
###############################################################################
#  Edge-Art GenAI Demo – deploy & run script
#  Push everything with:
#    adb push C:/edge_art_models/edge_art_genai.py  /data/local/tmp/edge_art_genai.py
#    adb push C:/edge_art_models/run_genai_demo.sh  /data/local/tmp/run_genai_demo.sh
#    adb push C:/edge_art_models/models/            /data/local/tmp/models/
###############################################################################

export XDG_RUNTIME_DIR=/run/user/root
export WAYLAND_DISPLAY=wayland-1
export QT_QPA_PLATFORM=wayland-egl
export QT_WAYLAND_SHELL_INTEGRATION=wl-shell
export ADSP_LIBRARY_PATH="/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp"

# load deepx driver if not already loaded
cd /lib/modules/$(uname -r)/extra/
insmod dxrt_driver.ko 2>/dev/null || true

# kill any leftover gst-launch processes
pkill -9 gst-launch-1.0 2>/dev/null || true
sleep 1

echo "[run] Starting Edge-Art GenAI Demo ..."
python3 /data/local/tmp/edge_art_genai.py
