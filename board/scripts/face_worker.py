#!/usr/bin/env python3
"""
Persistent QCS8550 HTP object/face detection worker — TFLite + QNN delegate.

Protocol (binary):
  C++ -> worker : 1 byte  (0x01 = run, 0x00 = quit)
  worker -> C++ : int32 n_dets
                  for each det: float x1,y1,x2,y2,conf + int32 cls_id  (24 bytes)
                  coordinates in original camera frame (CAM_W x CAM_H)
"""
import os, signal, struct, sys, ctypes
import numpy as np
import cv2

# HTP driver prints diagnostic lines to fd-1 (our pipe) during model init.
# Redirect fd-1 to /dev/null for the entire loading + warmup phase,
# then restore before writing READY so C++ parent receives a clean byte stream.
_pipe_wfd = os.dup(1)
_devnull  = os.open('/dev/null', os.O_WRONLY)
os.dup2(_devnull, 1)
os.close(_devnull)

signal.signal(signal.SIGTERM, lambda *_: os._exit(0))

MODEL      = '/opt/yolov8_det_quantized.tflite'
FRAME_FILE = '/tmp/yp_frame0.bin'
CAM_W, CAM_H = 1280, 720
INF_SIZE   = 640
CONF_THR   = 0.80

# quantization params probed at build time
BOX_SCALE  = 3.093529
BOX_ZP     = 21
CONF_SCALE = 0.003906
CONF_ZP    = 0

COCO_CLASSES = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
    'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat',
    'dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack',
    'umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball',
    'kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket',
    'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
    'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair',
    'couch','potted plant','bed','dining table','toilet','tv','laptop','mouse','remote',
    'keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book',
    'clock','vase','scissors','teddy bear','hair drier','toothbrush',
]

# ─── TFLite C API ────────────────────────────────────────────────────────────

tfl = ctypes.CDLL('/usr/lib/libtensorflowlite_c.so')
qnn = ctypes.CDLL('/usr/lib/libQnnTFLiteDelegate.so')

class _QP(ctypes.Structure):
    _fields_ = [('scale', ctypes.c_float), ('zero_point', ctypes.c_int32)]

def _setup_tfl():
    for name, res, args in [
        ('TfLiteModelCreateFromFile',         ctypes.c_void_p, [ctypes.c_char_p]),
        ('TfLiteInterpreterOptionsCreate',    ctypes.c_void_p, []),
        ('TfLiteInterpreterCreate',           ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_void_p]),
        ('TfLiteInterpreterAllocateTensors',  ctypes.c_int,    [ctypes.c_void_p]),
        ('TfLiteInterpreterInvoke',           ctypes.c_int,    [ctypes.c_void_p]),
        ('TfLiteInterpreterGetInputTensor',   ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_int]),
        ('TfLiteInterpreterGetOutputTensor',  ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_int]),
        ('TfLiteTensorByteSize',              ctypes.c_size_t, [ctypes.c_void_p]),
        ('TfLiteTensorData',                  ctypes.c_void_p, [ctypes.c_void_p]),
        ('TfLiteTensorNumDims',               ctypes.c_int,    [ctypes.c_void_p]),
        ('TfLiteTensorDim',                   ctypes.c_int32,  [ctypes.c_void_p, ctypes.c_int32]),
        ('TfLiteInterpreterOptionsAddDelegate', None,          [ctypes.c_void_p, ctypes.c_void_p]),
    ]:
        fn = getattr(tfl, name)
        fn.restype  = res
        fn.argtypes = args
    tfl.TfLiteTensorQuantizationParams.restype  = _QP
    tfl.TfLiteTensorQuantizationParams.argtypes = [ctypes.c_void_p]

_setup_tfl()

_ERRFN = ctypes.CFUNCTYPE(None, ctypes.c_char_p)
_err_fn = _ERRFN(lambda msg: None)
_keys = (ctypes.c_char_p * 2)(b'backend_type', b'htp_performance_mode')
_vals = (ctypes.c_char_p * 2)(b'htp', b'3')
qnn.tflite_plugin_create_delegate.restype  = ctypes.c_void_p
qnn.tflite_plugin_create_delegate.argtypes = [
    ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_char_p),
    ctypes.c_size_t, _ERRFN,
]
_delegate = qnn.tflite_plugin_create_delegate(_keys, _vals, 2, _err_fn)

_init_ok = False
_interp = _in_ptr = _in_sz = None
_out0_t = _out1_t = _out2_t = None
_out0_sz = _out1_sz = _out2_sz = 0
try:
    _model = tfl.TfLiteModelCreateFromFile(MODEL.encode())
    if not _model:
        raise RuntimeError("TfLiteModelCreateFromFile returned null")
    _opts = tfl.TfLiteInterpreterOptionsCreate()
    tfl.TfLiteInterpreterOptionsAddDelegate(_opts, _delegate)
    _interp = tfl.TfLiteInterpreterCreate(_model, _opts)
    if not _interp:
        raise RuntimeError("TfLiteInterpreterCreate returned null")
    tfl.TfLiteInterpreterAllocateTensors(_interp)

    _in_t   = tfl.TfLiteInterpreterGetInputTensor(_interp, 0)
    _in_sz  = tfl.TfLiteTensorByteSize(_in_t)
    _in_ptr = tfl.TfLiteTensorData(_in_t)

    _out0_t  = tfl.TfLiteInterpreterGetOutputTensor(_interp, 0)  # boxes  [8400,4] uint8
    _out1_t  = tfl.TfLiteInterpreterGetOutputTensor(_interp, 1)  # scores [8400]   uint8
    _out2_t  = tfl.TfLiteInterpreterGetOutputTensor(_interp, 2)  # cls    [8400]   uint8
    _out0_sz = tfl.TfLiteTensorByteSize(_out0_t)
    _out1_sz = tfl.TfLiteTensorByteSize(_out1_t)
    _out2_sz = tfl.TfLiteTensorByteSize(_out2_t)

    # warmup (compile HTP graph on first invoke)
    _dummy = (ctypes.c_uint8 * _in_sz)()
    ctypes.memmove(_in_ptr, _dummy, _in_sz)
    tfl.TfLiteInterpreterInvoke(_interp)
    _init_ok = True
except Exception as e:
    sys.stderr.write(f"[face_worker] init failed: {e}\n")
    sys.stderr.flush()
finally:
    # restore pipe regardless of init success so parent always gets READY
    os.dup2(_pipe_wfd, 1)
    os.close(_pipe_wfd)

sys.stdout.buffer.write(b'READY\n')
sys.stdout.buffer.flush()

# ─── helpers ─────────────────────────────────────────────────────────────────

# Pre-allocate once — avoids np.full() on every frame
_lb_canvas = np.full((INF_SIZE, INF_SIZE, 3), 114, np.uint8)

def letterbox(img, size=640, pad=114):
    h, w = img.shape[:2]
    r = min(size / w, size / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    _lb_canvas[:] = pad
    res = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    dx, dy = (size - nw) // 2, (size - nh) // 2
    _lb_canvas[dy:dy + nh, dx:dx + nw] = res
    return _lb_canvas, r, dx, dy

# ─── main loop ───────────────────────────────────────────────────────────────

stdin_fd = sys.stdin.buffer

while True:
    cmd = stdin_fd.read(1)
    if not cmd or cmd == b'\x00':
        os._exit(0)  # skip ctypes teardown to avoid SIGSEGV
    if cmd != b'\x01':
        continue

    if not _init_ok:
        sys.stdout.buffer.write(struct.pack('<i', 0))
        sys.stdout.buffer.flush()
        continue

    try:
        raw = np.fromfile(FRAME_FILE, dtype=np.uint8).reshape(CAM_H, CAM_W, 3)
    except Exception:
        sys.stdout.buffer.write(struct.pack('<i', 0))
        sys.stdout.buffer.flush()
        continue

    try:
        lb, r, dx, dy = letterbox(raw, INF_SIZE)
        rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
        # write into TFLite input tensor
        ctypes.memmove(_in_ptr, rgb.ctypes.data, _in_sz)
        tfl.TfLiteInterpreterInvoke(_interp)

        # read outputs
        boxes_u8  = np.frombuffer(ctypes.string_at(tfl.TfLiteTensorData(_out0_t), _out0_sz),
                                  dtype=np.uint8).reshape(8400, 4).astype(np.float32)
        scores_u8 = np.frombuffer(ctypes.string_at(tfl.TfLiteTensorData(_out1_t), _out1_sz),
                                  dtype=np.uint8).astype(np.float32)
        cls_u8    = np.frombuffer(ctypes.string_at(tfl.TfLiteTensorData(_out2_t), _out2_sz),
                                  dtype=np.uint8).astype(np.int32)

        # dequantize
        boxes  = (boxes_u8 - BOX_ZP)  * BOX_SCALE   # letterbox pixel coords
        scores = scores_u8 * CONF_SCALE               # confidence [0,1]

        mask = scores > CONF_THR
        if not mask.any():
            sys.stdout.buffer.write(struct.pack('<i', 0))
            sys.stdout.buffer.flush()
            continue

        boxes   = boxes[mask]
        scores  = scores[mask]
        cls_ids = cls_u8[mask]

        # map boxes from letterbox (INF_SIZE×INF_SIZE) back to camera frame
        boxes[:, 0] = np.clip((boxes[:, 0] - dx) / r, 0, CAM_W - 1)
        boxes[:, 1] = np.clip((boxes[:, 1] - dy) / r, 0, CAM_H - 1)
        boxes[:, 2] = np.clip((boxes[:, 2] - dx) / r, 0, CAM_W - 1)
        boxes[:, 3] = np.clip((boxes[:, 3] - dy) / r, 0, CAM_H - 1)

        # NMS to suppress overlapping detections
        boxes_xywh = np.column_stack([
            boxes[:, 0], boxes[:, 1],
            boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
        ])
        indices = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), scores.tolist(), CONF_THR, 0.45)
        if len(indices) == 0:
            sys.stdout.buffer.write(struct.pack('<i', 0))
            sys.stdout.buffer.flush()
            continue
        keep = np.array(indices).reshape(-1)
        boxes   = boxes[keep]
        scores  = scores[keep]
        cls_ids = cls_ids[keep]

        n = len(boxes)
        buf = struct.pack('<i', n)
        for i in range(n):
            buf += struct.pack('<5fi',
                float(boxes[i, 0]), float(boxes[i, 1]),
                float(boxes[i, 2]), float(boxes[i, 3]),
                float(scores[i]),   int(cls_ids[i]))
        sys.stdout.buffer.write(buf)
        sys.stdout.buffer.flush()

    except Exception:
        sys.stdout.buffer.write(struct.pack('<i', 0))
        sys.stdout.buffer.flush()
