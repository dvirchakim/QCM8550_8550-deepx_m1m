#!/bin/sh
export XDG_RUNTIME_DIR=/run/user/root
export WAYLAND_DISPLAY=wayland-1

rm -f /tmp/wayland-screenshot-*.png /tmp/screen.png

# weston-screenshooter writes to /tmp/wayland-screenshot-<timestamp>.png
# in the current directory. Run it from /tmp.
cd /tmp
weston-screenshooter 2>&1
ls -la /tmp/wayland-screenshot-*.png 2>&1
