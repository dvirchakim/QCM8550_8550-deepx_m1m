#!/usr/bin/env python3
"""
EasyOCR text-recognition worker — PERSISTENT (replaces trocr_worker.py).

Loads the CRAFT detector + CRNN recognizer ONNX models ONCE (onnxruntime, CPU)
and serves recognition requests over stdin/stdout. Unlike TrOCR (single text
line), EasyOCR detects multiple text regions anywhere in the frame and reads
each — far better suited to a live camera scene.

Run:
    PYTHONPATH=/data/local/tmp/ort181 python3 easyocr_worker.py

Binary protocol (little-endian), main <-> worker (UNCHANGED from trocr_worker):
  worker -> main : b'READY\n' once both models are loaded
  main -> worker : 0x01 + (CAM_H*CAM_W*3) uint8 BGR frame  -> recognize
                   0x00                                      -> quit
  worker -> main : 0x01 + int32 text_len + text_len bytes (utf-8, newline-joined)

Model files (exported by export_easyocr.py):
    /data/local/tmp/easyocr/easyocr_detector.onnx
    /data/local/tmp/easyocr/easyocr_recognizer.onnx
    /data/local/tmp/easyocr/easyocr_meta.json   {"character": "...", "imgH": 64}
"""
import glob, json, os, signal, struct, subprocess, sys, time
import numpy as np
import cv2
cv2.setNumThreads(0)   # avoid OpenCV/onnxruntime threadpool clashes

sys.path.insert(0, '/data/local/tmp')
import easyocr_post as ep

ORT_PATH    = '/data/local/tmp/ort181'
OCR_DIR     = '/data/local/tmp/easyocr'
DETECTOR    = f'{OCR_DIR}/easyocr_detector.onnx'
RECOGNIZER  = f'{OCR_DIR}/easyocr_recognizer.onnx'
META_FILE   = f'{OCR_DIR}/easyocr_meta.json'

# HTP-accelerated detector (QNN context binary). If present, the CRAFT detector
# runs on the Hexagon HTP via qnn-net-run (~0.5s incl. context load) instead of
# the CPU (~4s). The recognizer stays on CPU (fast on small crops).
QNN_DETECTOR = f'{OCR_DIR}/easyocr_detector_qcs8550.bin'
QNN_BACKEND  = '/usr/lib/libQnnHtp.so'
QNN_IO_DIR   = '/tmp/qnn_ocr_io'
QNN_OUT_DIR  = '/tmp/qnn_ocr_out'
DET_H, DET_W = 480, 800   # fixed QNN detector input (matches 1280x720 cam canvas)

CAM_W, CAM_H = 1280, 720
FRAME_BYTES  = CAM_H * CAM_W * 3

SQUARE_SIZE  = 800        # detector canvas (larger = better small text, slower)
MAX_BOXES    = 16         # cap recognitions per frame to bound latency
MIN_CONF     = 0.10       # drop low-confidence reads
TEXT_TH, LINK_TH, LOW_TH = 0.7, 0.4, 0.4

# Keep the binary reply channel clean (onnxruntime may print to stdout).
_pipe_wfd = os.dup(1)
_devnull  = os.open('/dev/null', os.O_WRONLY)
os.dup2(_devnull, 1)
os.close(_devnull)
_out = os.fdopen(_pipe_wfd, 'wb', buffering=0)
_in  = sys.stdin.buffer

signal.signal(signal.SIGTERM, lambda *_: os._exit(0))


def _log(m):
    sys.stderr.write(f'[easyocr] {m}\n'); sys.stderr.flush()


def read_exact(fp, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = fp.read(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


# ── Load models once ───────────────────────────────────────────────────────────
if ORT_PATH not in sys.path:
    sys.path.insert(0, ORT_PATH)
try:
    import onnxruntime as ort
except ImportError as e:
    _log(f'onnxruntime missing (PYTHONPATH={ORT_PATH}): {e}')
    os._exit(1)

try:
    meta = json.load(open(META_FILE, encoding='utf-8'))
    CHARACTER = meta['character']
    IMG_H     = int(meta.get('imgH', 64))
except Exception as e:
    _log(f'meta load failed: {e}')
    os._exit(1)

_opts = ort.SessionOptions()
_opts.inter_op_num_threads = 4
_opts.intra_op_num_threads = 4

USE_QNN = os.path.exists(QNN_DETECTOR)
det_sess = None
try:
    if USE_QNN:
        os.makedirs(QNN_IO_DIR, exist_ok=True)
        os.makedirs(f'{QNN_OUT_DIR}/Result_0', exist_ok=True)
        _log(f'detector: HTP (qnn) {os.path.basename(QNN_DETECTOR)}')
    else:
        _log('detector: CPU (onnx)')
        det_sess = ort.InferenceSession(DETECTOR, sess_options=_opts,
                                        providers=['CPUExecutionProvider'])
    _log('loading recognizer ...')
    rec_sess = ort.InferenceSession(RECOGNIZER, sess_options=_opts,
                                    providers=['CPUExecutionProvider'])
except Exception as e:
    _log(f'model load failed: {e}')
    os._exit(1)

_det_in = det_sess.get_inputs()[0].name if det_sess is not None else 'image'
_rec_in = rec_sess.get_inputs()[0].name
_qnn_env = os.environ.copy()
_log(f'models ready (n_char={len(CHARACTER)} imgH={IMG_H} '
     f'backend={"htp" if USE_QNN else "cpu"})')
_out.write(b'READY\n')


def _run_qnn_detector(nchw):
    nchw.astype(np.float32).tofile(f'{QNN_IO_DIR}/image.bin')
    with open(f'{QNN_IO_DIR}/input_list.txt', 'w') as f:
        f.write(f'image:={QNN_IO_DIR}/image.bin\n')
    for fn in glob.glob(f'{QNN_OUT_DIR}/Result_0/*'):
        os.unlink(fn)
    cmd = ['qnn-net-run', f'--retrieve_context={QNN_DETECTOR}',
           f'--backend={QNN_BACKEND}', f'--input_list={QNN_IO_DIR}/input_list.txt',
           f'--output_dir={QNN_OUT_DIR}', '--use_native_input_files',
           '--use_native_output_files']
    r = subprocess.run(cmd, capture_output=True, env=_qnn_env, timeout=60)
    if r.returncode != 0:
        raise RuntimeError('qnn-net-run failed: ' +
                           r.stderr.decode(errors='replace')[-400:])
    res = glob.glob(f'{QNN_OUT_DIR}/Result_0/*.raw')
    if not res:
        raise RuntimeError('qnn-net-run produced no output')
    return np.fromfile(res[0], np.float32).reshape(1, DET_H // 2, DET_W // 2, 2)


def _detect(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    canvas, ratio = ep.resize_aspect_ratio(rgb, SQUARE_SIZE)
    if canvas.shape[:2] != (DET_H, DET_W):
        padded = np.zeros((DET_H, DET_W, 3), canvas.dtype)
        h, w = min(DET_H, canvas.shape[0]), min(DET_W, canvas.shape[1])
        padded[:h, :w] = canvas[:h, :w]
        canvas = padded
    x = ep.normalize_mean_variance(canvas).transpose(2, 0, 1)[None].astype(np.float32)
    y = _run_qnn_detector(x) if USE_QNN else det_sess.run(None, {_det_in: x})[0]
    score_text = np.ascontiguousarray(y[0, :, :, 0])
    score_link = np.ascontiguousarray(y[0, :, :, 1])
    boxes = ep.get_det_boxes(score_text, score_link, TEXT_TH, LINK_TH, LOW_TH)
    return ep.adjust_result_coordinates(boxes, ratio)


def _recognize_box(frame_bgr, box):
    rect = np.array(box, np.float32)
    w = int(max(np.linalg.norm(rect[0] - rect[1]), np.linalg.norm(rect[2] - rect[3])))
    h = int(max(np.linalg.norm(rect[0] - rect[3]), np.linalg.norm(rect[1] - rect[2])))
    if w < 4 or h < 4:
        return None
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    crop = cv2.warpPerspective(frame_bgr, M, (w, h))
    inp = ep.recog_preprocess(crop, IMG_H).astype(np.float32)
    preds = rec_sess.run(None, {_rec_in: inp})[0][0]   # [T, n_class]
    text, conf = ep.ctc_greedy_decode(preds, CHARACTER)
    return text.strip(), conf


def recognize(frame_bgr):
    boxes = _detect(frame_bgr)
    # largest boxes first, then read; finally order top-to-bottom for display
    boxes = sorted(boxes, key=lambda b: -cv2.contourArea(np.array(b, np.float32)))
    boxes = boxes[:MAX_BOXES]
    reads = []
    for b in boxes:
        r = _recognize_box(frame_bgr, b)
        if r and r[0] and r[1] >= MIN_CONF:
            ymin = float(np.array(b)[:, 1].min())
            xmin = float(np.array(b)[:, 0].min())
            reads.append((ymin, xmin, r[0]))
    reads.sort(key=lambda t: (round(t[0] / 20), t[1]))   # rows top->bottom, L->R
    return '\n'.join(t[2] for t in reads)


# ── Serve requests ───────────────────────────────────────────────────────────
while True:
    cmd = _in.read(1)
    if not cmd or cmd == b'\x00':
        os._exit(0)
    if cmd != b'\x01':
        continue
    payload = read_exact(_in, FRAME_BYTES)
    if payload is None:
        os._exit(0)
    try:
        frame = np.frombuffer(payload, np.uint8).reshape(CAM_H, CAM_W, 3)
        t0 = time.time()
        text = recognize(frame)
        n = text.count('\n') + 1 if text else 0
        _log(f'{n} lines ({time.time()-t0:.2f}s): {text!r}')
    except Exception as e:
        _log(f'recognize error: {e}')
        text = ''
    data = text.encode('utf-8')[:1024]
    _out.write(b'\x01' + struct.pack('<i', len(data)) + data)
