#!/bin/sh
# Launch the Edge-Art live demo on the QCS8550 board.

export XDG_RUNTIME_DIR=/run/user/root
export WAYLAND_DISPLAY=wayland-1
export QT_QPA_PLATFORM=wayland-egl
export QT_WAYLAND_SHELL_INTEGRATION=wl-shell

pkill -9 gst-launch-1.0 2>/dev/null
sleep 1

echo "[run_live] starting live demo ..."
python3 /tmp/live_demo.py \
    --camera 0 \
    --width 1280 --height 720 --fps 30 \
    --style neon \
    --style-cycle-sec 8 \
    "$@"
