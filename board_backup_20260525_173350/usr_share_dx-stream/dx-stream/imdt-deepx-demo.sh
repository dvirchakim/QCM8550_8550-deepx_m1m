#!/bin/sh
# Pose using DeepX and Object detection using YOLOv8 on HTP
export GST_DEBUG_DUMP_DOT_DIR=build/pipeline2/
SRC_DIR=/usr/share/dx-stream
cd /lib/modules/$(uname -r)/extra/
insmod dxrt_driver.ko
cd $SRC_DIR
SRC_DIR=/usr/share/dx-stream
cd /lib/modules/$(uname -r)/extra/
insmod dxrt_driver.ko
SRC_0="qtiqmmfsrc camera=0"
SRC_1="qtiqmmfsrc camera=1"
# Using queues can cause crashes and I think its due to more parallelization happening - I am not too sure
QUEUE="identity"
# Use a leaky queue to avoid buffers being held too long
QUEUE_CAMERA="queue max-size-buffers=5 leaky=downstream"
VIDEO0="video0.mp4"
# SRC_0="videotestsrc pattern=smpte100"
# SRC_1="videotestsrc pattern=smpte100"
YOLO_V8_CONSTANTS="YOLOv8,q-offsets=<21.0, 0.0, 0.0>,q-scales=<3.0935,0.00390625,1.0>"
YOLO_V8_MODEL=/opt/YOLOv8-Detection-Quantized.tflite 
YOLO_V8_LABELS=/opt/yolov8.labels
YOLO_V8_POST="qtimlvdetection threshold=50 results=10 labels=${YOLO_V8_LABELS} module=yolov8 constants=\"${YOLO_V8_CONSTANTS}\""
POSE_LABELS=/opt/hrnet_pose.labels
POSE_MODEL=/opt/hrnet_pose_quantized.tflite
POSE_CONSTANTS="hrnet,q-offsets=<8.0>,q-scales=<0.0040499246679246426>;"
POSE_SETTINGS="threshold=50 results=2 module=hrnet"
POSE_POST="qtimlvpose ${POSE_SETTINGS} labels=${POSE_LABELS} constants=\"${POSE_CONSTANTS}\""
FRAMERATE="15/1"
VIDEO_TRANSFORM="qtivtransform"
echo "Post process config: ${POSE_POST}"
cd $SRC_DIR
export XDG_RUNTIME_DIR=/run/user/root/; export WAYLAND_DISPLAY=wayland-1; export QT_QPA_PLATFORM=wayland-egl; 
export QT_WAYLAND_SHELL_INTEGRATION=wl-shell
gst-launch-1.0 -e    ${SRC_0} ! ${VIDEO_TRANSFORM} ! video/x-raw,width=1920,height=1080,format=NV12,framerate=${FRAMERATE} ! ${QUEUE_CAMERA} ! tee name=t  t. !  $QUEUE ! \
        ${VIDEO_TRANSFORM} ! video/x-raw,width=1920,height=1080,format=NV12,framerate=${FRAMERATE} ! dxpreprocess config-file-path=$SRC_DIR/configs/Pose_Estimation/YOLOV5Pose640_1/preprocess_config.json ! $QUEUE ! \
        dxinfer config-file-path=$SRC_DIR/configs/Pose_Estimation/YOLOV5Pose640_1/inference_config.json ! $QUEUE ! \
        dxpostprocess config-file-path=$SRC_DIR/configs/Pose_Estimation/YOLOV5Pose640_1/postprocess_config.json ! $QUEUE ! \
        dxosd width=512 height=300  ! videoconvert ! videoscale ! video/x-raw,format=RGBA,width=512,height=300 ! $QUEUE ! \
        mixer.sink_0  \
    ${SRC_1} ! ${VIDEO_TRANSFORM} ! video/x-raw,width=1920,height=1080,format=NV12,framerate=${FRAMERATE} ! ${QUEUE_CAMERA} ! \
        ${VIDEO_TRANSFORM} ! video/x-raw,width=1920,height=1080,format=NV12,framerate=${FRAMERATE} ! dxpreprocess config-file-path=$SRC_DIR/configs/Pose_Estimation/YOLOV5Pose640_1/preprocess_config.json ! $QUEUE ! \
        dxinfer config-file-path=$SRC_DIR/configs/Pose_Estimation/YOLOV5Pose640_1/inference_config.json ! $QUEUE ! \
        dxpostprocess config-file-path=$SRC_DIR/configs/Pose_Estimation/YOLOV5Pose640_1/postprocess_config.json ! $QUEUE ! \
        dxosd width=512 height=300  ! videoconvert ! videoscale ! video/x-raw,format=RGBA,width=512,height=300,colorimetry=sRGB   !  $QUEUE ! \
        mixer.sink_1  \
    t.   ! ${QUEUE}   ! \
    tee name=split \
    split. ! $QUEUE  !  identity sync=false ! ${VIDEO_TRANSFORM} ! mixer.sink_2  \
    split. ! $QUEUE ! qtimlvconverter ! \
    qtimltflite model=${YOLO_V8_MODEL}  \
    external-delegate-path=libQnnTFLiteDelegate.so \
    external-delegate-options="QNNExternalDelegate,backend_type=htp,htp_performance_mode=(string)2;" ! \
    ${YOLO_V8_POST} ! ${QUEUE} ! \
    video/x-raw,format=BGRA,width=512,height=300 ! mixer.sink_3 \
    filesrc location=logo.png ! pngdec ! imagefreeze  ! queue max-size-buffers=10 ! videoconvert ! videoscale ! queue max-size-buffers=10 ! video/x-raw,format=RGB,width=512,height=300,framerate=${FRAMERATE},colorimetry=sRGB ! ${QUEUE} ! mixer.sink_4 \
  qtivcomposer name=mixer \
    sink_0::position="<0, 0>" sink_0::dimensions="<512, 300>" \
    sink_1::position="<512, 0>" sink_1::dimensions="<512, 300>" \
    sink_2::position="<0, 300>" sink_2::dimensions="<512, 300>" \
    sink_3::position="<0, 300>" sink_3::dimensions="<512, 300>" \
    sink_4::position="<512, 300>" sink_4::dimensions="<512, 300>" \
  ! $QUEUE ! waylandsink fullscreen=true sync=false