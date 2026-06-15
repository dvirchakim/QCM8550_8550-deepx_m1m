#!/usr/bin/env python3
"""
Combined DEEPX worker — SCDepthV3 (depth) + YOLOv5s-Face (detect) + 3DDFA_V2
(3D face alignment), all in ONE process.

Why one process: DXRT does not allow two processes to access the device
concurrently. A single worker owns all three models. Depth and face-detection
are submitted with run_async() so they execute IN PARALLEL across the DX-M1
NPU cores (no frame alternating); 3DDFA then runs on the cropped face.

INFERENCE-ONLY: receives raw uint8 model inputs over stdin, returns raw results
over stdout. All rendering lives in the main demo process. No /tmp files.

Binary protocol (little-endian), main <-> worker:
  main -> worker:
      0x01 + (DEPTH_H*DEPTH_W*3) uint8   depth input
           + (FACE*FACE*3)       uint8   letterboxed face input (BGR, 640)
      0x00 -> quit
  worker -> main:
      0x01 + float32 depth_ms + float32 face_ms
           + (DEPTH_H*DEPTH_W) float32 depth map
           + int32 has_face
           + if has_face: 5*float32 (x1,y1,x2,y2,score) [in FACE coords]
                          + 136*float32 landmarks (x,y)*68 [in FACE coords]
  On error the worker still replies with a well-formed (zeroed) response.
"""
import os, signal, struct, sys, threading, time
import numpy as np
import cv2

# DXRT prints diagnostics to stdout; redirect real stdout to /dev/null and keep
# a private duplicate for the binary result channel so it is never corrupted.
_pipe_wfd = os.dup(1)
_devnull  = os.open('/dev/null', os.O_WRONLY)
os.dup2(_devnull, 1)
os.close(_devnull)
_out = os.fdopen(_pipe_wfd, 'wb', buffering=0)
_in  = sys.stdin.buffer

signal.signal(signal.SIGTERM, lambda *_: os._exit(0))

from dx_engine import InferenceEngine, InferenceOption

DEPTH_MODEL      = '/data/local/tmp/scdepthv3.dxnn'
FACE_MODEL       = '/data/local/tmp/yolov5s_face.dxnn'
TDDFA_MODEL      = '/data/local/tmp/3ddfa_v2.dxnn'
BASES_NPZ        = '/data/local/tmp/face3d_bases.npz'

DEPTH_W, DEPTH_H = 320, 256
FACE             = 640                # YOLOv5s-Face input
TDDFA_SIZE       = 120                # 3DDFA input
N_LMK            = 68

DEPTH_IN_BYTES = DEPTH_H * DEPTH_W * 3
FACE_IN_BYTES  = FACE * FACE * 3

FACE_CONF_THR  = 0.40
FACE_NMS_THR   = 0.45


def _log(msg):
    sys.stderr.write(f'[dual_deepx] {msg}\n')
    sys.stderr.flush()


def read_exact(fp, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = fp.read(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


# ── 3DDFA reconstruction bases (sliced 68-keypoint BFM) ────────────────────────
try:
    _b = np.load(BASES_NPZ)
    U_BASE     = _b['u_base'].astype(np.float32)        # (204, 1)
    W_SHP_BASE = _b['w_shp_base'].astype(np.float32)    # (204, 40)
    W_EXP_BASE = _b['w_exp_base'].astype(np.float32)    # (204, 10)
    PARAM_MEAN = _b['param_mean'].astype(np.float32)    # (62,)
    PARAM_STD  = _b['param_std'].astype(np.float32)     # (62,)
    _log('face3d bases loaded')
except Exception as e:
    U_BASE = W_SHP_BASE = W_EXP_BASE = PARAM_MEAN = PARAM_STD = None
    _log(f'face3d bases load failed: {e}')


def parse_roi_box_from_bbox(bbox):
    left, top, right, bottom = bbox[:4]
    old_size = (right - left + bottom - top) / 2
    cx = right - (right - left) / 2.0
    cy = bottom - (bottom - top) / 2.0 + old_size * 0.14
    size = int(old_size * 1.58)
    return [cx - size / 2, cy - size / 2, cx - size / 2 + size, cy - size / 2 + size]


def crop_img(img, roi_box):
    h, w = img.shape[:2]
    sx, sy, ex, ey = [int(round(v)) for v in roi_box]
    dh, dw = ey - sy, ex - sx
    res = np.zeros((dh, dw, 3), dtype=np.uint8)
    dsx = -sx if sx < 0 else 0
    sx  = max(sx, 0)
    if ex > w: dex = dw - (ex - w); ex = w
    else:      dex = dw
    dsy = -sy if sy < 0 else 0
    sy  = max(sy, 0)
    if ey > h: dey = dh - (ey - h); ey = h
    else:      dey = dh
    res[dsy:dey, dsx:dex] = img[sy:ey, sx:ex]
    return res


def _parse_param(param):
    R_ = param[:12].reshape(3, -1)
    R = R_[:, :3]
    offset = R_[:, -1].reshape(3, 1)
    alpha_shp = param[12:52].reshape(-1, 1)
    alpha_exp = param[52:].reshape(-1, 1)
    return R, offset, alpha_shp, alpha_exp


def similar_transform(pts3d, roi_box, size):
    pts3d[0, :] -= 1
    pts3d[2, :] -= 1
    pts3d[1, :] = size - pts3d[1, :]
    sx, sy, ex, ey = roi_box
    scale_x = (ex - sx) / size
    scale_y = (ey - sy) / size
    pts3d[0, :] = pts3d[0, :] * scale_x + sx
    pts3d[1, :] = pts3d[1, :] * scale_y + sy
    s = (scale_x + scale_y) / 2
    pts3d[2, :] *= s
    pts3d[2, :] -= np.min(pts3d[2, :])
    return pts3d.astype(np.float32)


def recon_landmarks(param62, roi_box):
    param = param62 * PARAM_STD + PARAM_MEAN
    R, offset, a_shp, a_exp = _parse_param(param)
    pts = (R @ (U_BASE + W_SHP_BASE @ a_shp + W_EXP_BASE @ a_exp)
           .reshape(3, -1, order='F') + offset)
    pts = similar_transform(pts, roi_box, TDDFA_SIZE)
    return pts[:2].T.copy()      # (68, 2)


def nms(boxes, scores, thr):
    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]; keep.append(int(i))
        if order.size == 1: break
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou < thr]
    return keep


def decode_faces(raw):
    """raw (25200,16): cx,cy,w,h,obj,lm(10),cls -> best face box [x1,y1,x2,y2], score."""
    raw = raw.reshape(-1, 16)
    conf = raw[:, 4] * raw[:, 15]
    m = conf > FACE_CONF_THR
    if not np.any(m):
        return None
    cand = raw[m]; sc = conf[m]
    cx, cy, bw, bh = cand[:, 0], cand[:, 1], cand[:, 2], cand[:, 3]
    boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
    keep = nms(boxes, sc, FACE_NMS_THR)
    bi = keep[int(np.argmax(sc[keep]))]
    return boxes[bi], float(sc[bi])


# ── Load engines (each with its own option so DXRT can spread across cores) ─────
def _mk_opt():
    o = InferenceOption()
    try: o.set_buffer_count(2)
    except Exception: pass
    return o


def _warmup(engine, dummy, label, timeout_s=90):
    done = threading.Event()
    def _run():
        try: engine.run(dummy)
        except Exception: pass
        done.set()
    threading.Thread(target=_run, daemon=True).start()
    if not done.wait(timeout=timeout_s):
        _log(f'{label} warmup HUNG (>{timeout_s}s) — disabled')
        return False
    _log(f'{label} warmup OK')
    return True


depth_engine = face_engine = tddfa_engine = None
try:
    depth_engine = InferenceEngine(DEPTH_MODEL, _mk_opt())
    _log('SCDepthV3 loaded — warming up')
    if not _warmup(depth_engine, np.zeros((1, DEPTH_H, DEPTH_W, 3), np.uint8), 'SCDepthV3'):
        depth_engine = None
except Exception as e:
    _log(f'SCDepthV3 load failed: {e}')

try:
    face_engine = InferenceEngine(FACE_MODEL, _mk_opt())
    _log('YOLOv5s-Face loaded — warming up')
    if not _warmup(face_engine, np.zeros((1, FACE, FACE, 3), np.uint8), 'YOLOv5s-Face'):
        face_engine = None
except Exception as e:
    _log(f'YOLOv5s-Face load failed: {e}')

try:
    tddfa_engine = InferenceEngine(TDDFA_MODEL, _mk_opt())
    _log('3DDFA_V2 loaded — warming up')
    if not _warmup(tddfa_engine, np.zeros((1, TDDFA_SIZE, TDDFA_SIZE, 3), np.uint8), '3DDFA_V2'):
        tddfa_engine = None
except Exception as e:
    _log(f'3DDFA_V2 load failed: {e}')

_log(f'depth={depth_engine is not None} face={face_engine is not None} '
     f'tddfa={tddfa_engine is not None} bases={U_BASE is not None}')
_out.write(b'READY\n')


# ── Main request loop ──────────────────────────────────────────────────────────
_zeros_depth = np.zeros(DEPTH_H * DEPTH_W, np.float32)


def _reply(depth_ms, face_ms, depth_flat, face_box, score, landmarks):
    resp = bytearray(b'\x01')
    resp += struct.pack('<ff', float(depth_ms), float(face_ms))
    resp += np.ascontiguousarray(depth_flat, np.float32).tobytes()
    if face_box is None:
        resp += struct.pack('<i', 0)
    else:
        resp += struct.pack('<i', 1)
        resp += struct.pack('<5f', float(face_box[0]), float(face_box[1]),
                            float(face_box[2]), float(face_box[3]), float(score))
        resp += np.ascontiguousarray(landmarks.ravel(), np.float32).tobytes()
    _out.write(bytes(resp))


while True:
    cmd = _in.read(1)
    if not cmd or cmd == b'\x00':
        os._exit(0)
    if cmd != b'\x01':
        continue

    depth_payload = read_exact(_in, DEPTH_IN_BYTES)
    face_payload  = read_exact(_in, FACE_IN_BYTES)
    if depth_payload is None or face_payload is None:
        os._exit(0)

    depth_flat = _zeros_depth
    depth_ms = face_ms = 0.0
    face_box = None; score = 0.0; landmarks = None

    try:
        depth_in = np.frombuffer(depth_payload, np.uint8).reshape(1, DEPTH_H, DEPTH_W, 3)
        face_img = np.frombuffer(face_payload, np.uint8).reshape(FACE, FACE, 3)

        # ── Submit depth + face IN PARALLEL across NPU cores ──────────────────
        t0 = time.time()
        depth_job = depth_engine.run_async(depth_in) if depth_engine else None
        face_job  = face_engine.run_async(np.expand_dims(face_img, 0)) if face_engine else None

        if depth_job is not None:
            douts = depth_engine.wait(depth_job)
            depth_flat = (douts[0].reshape(-1, DEPTH_H, DEPTH_W)[0]
                          .astype(np.float32).ravel())
        depth_ms = (time.time() - t0) * 1000.0

        if face_job is not None:
            fouts = face_engine.wait(face_job)
            face_ms = (time.time() - t0) * 1000.0
            det = decode_faces(fouts[0].astype(np.float32))
            if det is not None and tddfa_engine is not None and U_BASE is not None:
                box, score = det
                roi = parse_roi_box_from_bbox(box)
                crop = crop_img(face_img, roi)
                crop = cv2.resize(crop, (TDDFA_SIZE, TDDFA_SIZE),
                                  interpolation=cv2.INTER_LINEAR)
                touts = tddfa_engine.run(np.expand_dims(crop, 0))
                param62 = touts[0].reshape(-1)[:62].astype(np.float32)
                landmarks = recon_landmarks(param62, roi)
                face_box = box
    except Exception as e:
        _log(f'infer error: {e}')

    _reply(depth_ms, face_ms, depth_flat, face_box, score, landmarks)
