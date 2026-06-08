#!/bin/sh
set +e
echo "=== opkg python-pip availability =="
opkg list 2>/dev/null | grep -E "python3-pip|python3-evdev|python3-paho|paho-mqtt"

echo
echo "=== framebuffer =="
ls /dev/fb* 2>&1

echo
echo "=== weston processes =="
ps -ef | grep -E "weston|wayland" | grep -v grep

echo
echo "=== xdg runtime =="
ls -la /run/user/ 2>&1
ls /tmp/wayland-* 2>&1

echo
echo "=== gst plugins for waylandsink/appsrc =="
gst-inspect-1.0 waylandsink 2>&1 | head -3
gst-inspect-1.0 appsrc 2>&1 | head -3
gst-inspect-1.0 v4l2src 2>&1 | head -3

echo
echo "=== InferenceOption help =="
python3 -c "from dx_engine import InferenceEngine, InferenceOption; help(InferenceOption)" 2>&1 | head -40

echo
echo "=== Quick model load test =="
python3 -c "
from dx_engine import InferenceEngine
import time
t = time.time()
eng = InferenceEngine('/usr/share/dx-stream/dx_stream/samples/models/YOLOV5Pose640_1.dxnn')
print(f'Loaded in {(time.time()-t)*1000:.1f} ms')
print('Inputs :', eng.get_input_tensors_info())
print('Outputs:', eng.get_output_tensors_info())
print('PPU?  :', eng.is_ppu())
" 2>&1

echo
echo "=== DONE =="
