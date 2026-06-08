#!/bin/sh
# Validate HTP DeepLabV3 segmentation with videotestsrc (no camera needed)
# Push and run: adb push scripts/test_htp_seg.sh /data/local/tmp/test_htp_seg.sh
#               adb shell sh /data/local/tmp/test_htp_seg.sh

export XDG_RUNTIME_DIR=/run/user/root
export WAYLAND_DISPLAY=wayland-1
export QT_QPA_PLATFORM=wayland-egl
export QT_WAYLAND_SHELL_INTEGRATION=wl-shell
export ADSP_LIBRARY_PATH="/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp"

SEG_MODEL=/opt/deeplabv3_plus_mobilenet_quantized.tflite
SEG_LABELS=/opt/deeplabv3_resnet50.labels
SEG_CONSTS="deeplab,q-offsets=<8.0>,q-scales=<0.0040499246679246426>;"

echo "[test] Starting HTP segmentation with videotestsrc ..."

gst-launch-1.0 -v \
    videotestsrc num-buffers=30 ! \
    video/x-raw,width=512,height=512,format=RGB ! \
    videoconvert ! \
    video/x-raw,format=NV12 ! \
    qtimlvconverter ! \
    qtimltflite model=${SEG_MODEL} \
        external-delegate-path=libQnnTFLiteDelegate.so \
        external-delegate-options="QNNExternalDelegate,backend_type=htp,htp_performance_mode=(string)2;" ! \
    qtimlvsegmentation labels=${SEG_LABELS} module=deeplab-argmax \
        constants="${SEG_CONSTS}" ! \
    videoconvert ! \
    waylandsink sync=false

echo "[test] Done. Exit: $?"
