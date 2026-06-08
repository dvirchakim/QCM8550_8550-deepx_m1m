#!/bin/sh
# Probe what's available on the QCS8550 board for the new demo.
set +e

echo "=== Python =="
python3 --version
echo
echo "=== dx_engine / dx_rt =="
python3 -c "import dx_engine; print('dx_engine version:', dx_engine.__version__)"
echo
echo "=== OpenCV =="
python3 -c "import cv2; print('cv2', cv2.__version__); print('gstreamer:', 'YES' if 'GStreamer' in cv2.getBuildInformation() and 'YES' in cv2.getBuildInformation().split('GStreamer')[1].split('\n')[0] else 'NO')"
echo
echo "=== GStreamer Python (gi) =="
python3 -c "import gi; gi.require_version('Gst','1.0'); from gi.repository import Gst; Gst.init(None); print('gst-py OK')" 2>&1 | head -3
echo
echo "=== Qt5/6 =="
python3 -c "import PyQt5" 2>&1 | head -1
python3 -c "import PyQt6" 2>&1 | head -1
python3 -c "import PySide6" 2>&1 | head -1
echo
echo "=== pip =="
which pip pip3 python3-pip 2>&1
echo
echo "=== opkg python pkgs =="
opkg list-installed 2>/dev/null | grep python3 | head -20
echo
echo "=== /dev/video =="
ls -la /dev/video* 2>&1 | head -10
echo
echo "=== Wayland =="
echo "WAYLAND_DISPLAY=$WAYLAND_DISPLAY"
ls /run/user/0/ 2>&1
ls /run/user/root/ 2>&1
echo
echo "=== Mosquitto =="
systemctl is-active mosquitto
echo
echo "=== Demo service =="
systemctl is-active imdt-deepx-demo
echo
echo "=== Pose model =="
ls -la /usr/share/dx-stream/dx_stream/samples/models/YOLOV5Pose640_1.dxnn
echo
echo "=== DONE =="
