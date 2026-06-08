#!/bin/sh
###############################################################################
#  Edge-Art 4-pane demo using videotestsrc instead of cameras.
#
#  Validates that:
#   - DEEPX YOLOv5-Pose plugin chain works (no camera needed; pose still runs)
#   - Qualcomm HTP YOLOv8 chain works
#   - qtivcomposer + waylandsink display path is alive
#
#  When qmmf-server is healthy again, switch back to the camera version.
###############################################################################
set -e

export ADSP_LIBRARY_PATH="/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp"
export XDG_RUNTIME_DIR=/run/user/root/
export WAYLAND_DISPLAY=wayland-1
export QT_QPA_PLATFORM=wayland-egl
export QT_WAYLAND_SHELL_INTEGRATION=wl-shell

cd /lib/modules/$(uname -r)/extra/
insmod dxrt_driver.ko 2>/dev/null || true

pkill -9 gst-launch-1.0 2>/dev/null || true
sleep 1

SRC_DIR=/usr/share/dx-stream
cd $SRC_DIR

QUEUE="identity"
QUEUE_CAMERA="queue max-size-buffers=5 leaky=downstream"
FRAMERATE="15/1"
VIDEO_TRANSFORM="qtivtransform"
CAPS="video/x-raw,width=1280,height=720,format=NV12,framerate=${FRAMERATE}"

# Two distinct test patterns, one per "camera"
SRC_0="videotestsrc is-live=true pattern=ball"
SRC_1="videotestsrc is-live=true pattern=smpte100"

YOLO_V8_MODEL=/opt/YOLOv8-Detection-Quantized.tflite
YOLO_V8_LABELS=/opt/yolov8.labels
YOLO_V8_CONSTANTS="YOLOv8,q-offsets=<21.0, 0.0, 0.0>,q-scales=<3.0935,0.00390625,1.0>"
YOLO_V8_POST="qtimlvdetection threshold=50 results=10 labels=${YOLO_V8_LABELS} module=yolov8 constants=\"${YOLO_V8_CONSTANTS}\""

POSE_CFG=$SRC_DIR/configs/Pose_Estimation/YOLOV5Pose640_1

set -x

gst-launch-1.0 -e \
    ${SRC_0} ! ${VIDEO_TRANSFORM} ! ${CAPS} ! ${QUEUE_CAMERA} ! tee name=t0 \
    t0. ! $QUEUE ! ${VIDEO_TRANSFORM} ! ${CAPS} ! \
        dxpreprocess  config-file-path=$POSE_CFG/preprocess_config.json  ! $QUEUE ! \
        dxinfer       config-file-path=$POSE_CFG/inference_config.json   ! $QUEUE ! \
        dxpostprocess config-file-path=$POSE_CFG/postprocess_config.json ! $QUEUE ! \
        dxosd width=512 height=300 ! videoconvert ! videoscale ! \
        video/x-raw,format=RGBA,width=512,height=300 ! $QUEUE ! mixer.sink_0 \
    \
    t0. ! $QUEUE ! qtimlvconverter ! \
        qtimltflite model=${YOLO_V8_MODEL} \
            external-delegate-path=libQnnTFLiteDelegate.so \
            external-delegate-options="QNNExternalDelegate,backend_type=htp,htp_performance_mode=(string)2;" ! \
        ${YOLO_V8_POST} ! $QUEUE ! \
        video/x-raw,format=BGRA,width=512,height=300 ! $QUEUE ! mixer.sink_2 \
    \
    ${SRC_1} ! ${VIDEO_TRANSFORM} ! ${CAPS} ! ${QUEUE_CAMERA} ! tee name=t1 \
    t1. ! $QUEUE ! ${VIDEO_TRANSFORM} ! ${CAPS} ! \
        dxpreprocess  config-file-path=$POSE_CFG/preprocess_config.json  ! $QUEUE ! \
        dxinfer       config-file-path=$POSE_CFG/inference_config.json   ! $QUEUE ! \
        dxpostprocess config-file-path=$POSE_CFG/postprocess_config.json ! $QUEUE ! \
        dxosd width=512 height=300 ! videoconvert ! videoscale ! \
        video/x-raw,format=RGBA,width=512,height=300,colorimetry=sRGB ! $QUEUE ! mixer.sink_1 \
    \
    t1. ! $QUEUE ! qtimlvconverter ! \
        qtimltflite model=${YOLO_V8_MODEL} \
            external-delegate-path=libQnnTFLiteDelegate.so \
            external-delegate-options="QNNExternalDelegate,backend_type=htp,htp_performance_mode=(string)2;" ! \
        ${YOLO_V8_POST} ! $QUEUE ! \
        video/x-raw,format=BGRA,width=512,height=300 ! $QUEUE ! mixer.sink_3 \
    \
  qtivcomposer name=mixer \
    sink_0::position="<0, 0>"     sink_0::dimensions="<512, 300>" \
    sink_1::position="<512, 0>"   sink_1::dimensions="<512, 300>" \
    sink_2::position="<0, 300>"   sink_2::dimensions="<512, 300>" \
    sink_3::position="<512, 300>" sink_3::dimensions="<512, 300>" \
  ! $QUEUE ! waylandsink fullscreen=true sync=false
