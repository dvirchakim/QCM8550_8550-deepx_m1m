#!/usr/bin/env python3
"""
Persistent DEEPX DX-M1 segmentation worker — yolo26n-seg via dx_engine.

Post-processing follows the official DEEPX dx_app InstanceSegPostprocessor:
  output[0]: [1, 300, 38]  →  post-NMS: 4 box(x1,y1,x2,y2) + score + cls_id + 32 mask_coeff
  output[1]: [1, 32, 160, 160]  →  prototype masks

Protocol (binary):
  C++ → worker : 1 byte  (0x01 = run, 0x00 = quit)
  worker → C++ : 1 byte  (0x01 = done; rendered pane written to SEG_OUT_FILE)
                 int32   n_dets
                 float   inf_ms
"""
import os, signal, struct, sys, time
import numpy as np
import cv2

# DXRT runtime writes [DXRT][OutputHandler] lines to fd-1 on every inference.
# Permanently redirect fd-1 to /dev/null and communicate on the saved pipe fd.
_pipe_wfd = os.dup(1)
_devnull  = os.open('/dev/null', os.O_WRONLY)
os.dup2(_devnull, 1)
os.close(_devnull)
_pipe_out = os.fdopen(_pipe_wfd, 'wb', buffering=0)

signal.signal(signal.SIGTERM, lambda *_: os._exit(0))

from dx_engine import InferenceEngine, InferenceOption

MODEL        = '/data/local/tmp/yolo26n-seg.dxnn'
FRAME_FILE   = '/tmp/yp_frame1.bin'
SEG_OUT_FILE = '/tmp/yp_seg_out.bin'
CAM_W, CAM_H = 1280, 720
PANE_W, PANE_H = 640, 720
INF_SIZE     = 640
SCORE_THR    = 0.30
NMS_THR      = 0.45
NUM_MASKS    = 32
PROTO_H      = 160
PROTO_W      = 160

_PALETTE = [
    (255, 56, 56),(255,157, 51),(255,112, 31),(255,178, 29),(207,210, 49),
    ( 72,249, 10),(146,204, 23),( 61,219,134),( 26,147, 52),(  0,212,187),
    ( 44,153,168),(  0,194,255),( 52, 69,147),(100, 45,144),(142, 27, 27),
    (224, 13, 99),(209, 26, 42),(  0,248,252),(187, 85,170),(175, 49,122),
]
def _bgr(i):
    r, g, b = _PALETTE[i % len(_PALETTE)]
    return (b, g, r)

# ── init engine ───────────────────────────────────────────────────────────────
opt = InferenceOption()
try:
    opt.set_buffer_count(1)
except Exception:
    pass
engine = None
try:
    engine = InferenceEngine(MODEL, opt)
except Exception as e:
    sys.stderr.write(f"[seg_worker] DEEPX init failed: {e}\n")
    sys.stderr.flush()

_pipe_out.write(b'READY\n')

# ── helpers ───────────────────────────────────────────────────────────────────

# Pre-allocate once — avoids repeated large allocations per frame
_lb_canvas = np.full((INF_SIZE, INF_SIZE, 3), 114, np.uint8)
_pane_buf  = np.empty((PANE_H, PANE_W, 3), np.uint8)

def letterbox(img, size=640, pad_val=114):
    h, w = img.shape[:2]
    r = min(size / w, size / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    _lb_canvas[:] = pad_val
    res = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    dx, dy = (size - nw) // 2, (size - nh) // 2
    _lb_canvas[dy:dy + nh, dx:dx + nw] = res
    return _lb_canvas, r, dx, dy

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))

# ── main loop ─────────────────────────────────────────────────────────────────

stdin_fd = sys.stdin.buffer
while True:
    cmd = stdin_fd.read(1)
    if not cmd or cmd == b'\x00':
        os._exit(0)
    if cmd != b'\x01':
        continue

    try:
        raw = np.fromfile(FRAME_FILE, dtype=np.uint8).reshape(CAM_H, CAM_W, 3)
    except Exception:
        pane = np.zeros((PANE_H, PANE_W, 3), np.uint8)
        pane.tofile(SEG_OUT_FILE)
        _pipe_out.write(b'\x01' + struct.pack('<if', 0, 0.0))
        continue

    try:
        cv2.resize(raw, (PANE_W, PANE_H), dst=_pane_buf, interpolation=cv2.INTER_AREA)
        pane = _pane_buf

        # Preprocess: letterbox 640×640, raw uint8 BGR (matches working pose_worker;
        # the dxnn model has normalization compiled in and expects uint8).
        lb, r, dx, dy = letterbox(raw, INF_SIZE)

        t0 = time.time()
        outs = engine.run(np.expand_dims(lb, 0))
        inf_ms = (time.time() - t0) * 1000.0

        # Post-NMS detection tensor: [1, 300, 38]
        dets   = np.squeeze(outs[0])          # (300, 38)
        protos = np.squeeze(outs[1])          # (32, 160, 160)
        if protos.shape[-1] == NUM_MASKS:     # HWC → CHW if needed
            protos = np.transpose(protos, (2, 0, 1))

        # yolo26n-seg post-NMS format: col4=score, col5=cls_id, cols6-37=mask_coeff
        scores   = dets[:, 4]
        cls_ids  = dets[:, 5].astype(int)
        mask_coeff = dets[:, 6:6 + NUM_MASKS]    # (300, 32)
        boxes    = dets[:, :4]                    # (300, 4) [x1,y1,x2,y2] in letterbox space

        # Filter by score threshold
        keep_mask = scores > SCORE_THR
        if not keep_mask.any():
            pane.tofile(SEG_OUT_FILE)
            _pipe_out.write(b'\x01' + struct.pack('<if', 0, float(inf_ms)))
            continue

        scores     = scores[keep_mask]
        cls_ids    = cls_ids[keep_mask]
        mask_coeff = mask_coeff[keep_mask]
        boxes      = boxes[keep_mask]

        # NMS (boxes are [x1,y1,x2,y2] → convert to [x,y,w,h] for OpenCV)
        boxes_xywh = np.column_stack([
            boxes[:, 0], boxes[:, 1],
            boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
        ])
        indices = cv2.dnn.NMSBoxes(
            boxes_xywh.tolist(), scores.tolist(), SCORE_THR, NMS_THR)
        if len(indices) == 0:
            pane.tofile(SEG_OUT_FILE)
            _pipe_out.write(b'\x01' + struct.pack('<if', 0, float(inf_ms)))
            continue
        keep = np.array(indices).reshape(-1)

        # Draw masks and boxes on pane
        # Scale factors: letterbox → cam frame → pane
        # cam_x = (lb_x - dx) / r ;  pane_x = cam_x * (PANE_W / CAM_W)
        sx = (1.0 / r) * (PANE_W / CAM_W)   # = (1/0.5) * (640/1280) = 1.0
        sy = (1.0 / r) * (PANE_H / CAM_H)   # = 2.0 * 1.0 = 2.0
        ox = -dx * sx                         # x offset = 0
        oy = -dy * sy                         # y offset = -140 * 2.0 = -280

        # Batch all mask coefficients in one matmul: (K,32) @ (32,25600) → (K,160,160)
        proto_flat  = protos.reshape(NUM_MASKS, -1)          # (32, 25600)
        raw_masks   = sigmoid(mask_coeff[keep] @ proto_flat).reshape(len(keep), PROTO_H, PROTO_W)
        ps = INF_SIZE / PROTO_H                              # = 4.0

        for idx, i in enumerate(keep):
            col = _bgr(int(cls_ids[i]))

            # Bounding box in pane coords
            x1p = int(np.clip(boxes[i, 0] * sx + ox, 0, PANE_W - 1))
            y1p = int(np.clip(boxes[i, 1] * sy + oy, 0, PANE_H - 1))
            x2p = int(np.clip(boxes[i, 2] * sx + ox, 0, PANE_W - 1))
            y2p = int(np.clip(boxes[i, 3] * sy + oy, 0, PANE_H - 1))

            if x2p <= x1p or y2p <= y1p:
                continue

            raw_mask = raw_masks[idx]

            # Crop mask to bounding box region in proto space
            mx1 = int(np.clip(boxes[i, 0] / ps, 0, PROTO_W))
            my1 = int(np.clip(boxes[i, 1] / ps, 0, PROTO_H))
            mx2 = int(np.clip(boxes[i, 2] / ps, 0, PROTO_W))
            my2 = int(np.clip(boxes[i, 3] / ps, 0, PROTO_H))
            if mx2 > mx1 and my2 > my1:
                crop = raw_mask[my1:my2, mx1:mx2]
            else:
                crop = raw_mask

            h_dst = max(1, y2p - y1p)
            w_dst = max(1, x2p - x1p)
            bin_mask = (cv2.resize(crop, (w_dst, h_dst),
                                   interpolation=cv2.INTER_LINEAR) > 0.5).astype(np.uint8)

            # Colored overlay on pane
            region = pane[y1p:y2p, x1p:x2p]
            if region.shape[:2] == bin_mask.shape:
                overlay = region.copy()
                overlay[bin_mask == 1] = col
                cv2.addWeighted(overlay, 0.50, region, 0.50, 0, region)
                pane[y1p:y2p, x1p:x2p] = region

            # Contour
            cnts, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in cnts:
                cnt += np.array([[[x1p, y1p]]])
                cv2.drawContours(pane, [cnt], -1, col, 1)

            # Box + score label
            cv2.rectangle(pane, (x1p, y1p), (x2p, y2p), col, 2)
            cv2.putText(pane, f'{scores[i]:.2f}', (x1p + 4, y1p + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

        pane.tofile(SEG_OUT_FILE)
        _pipe_out.write(b'\x01' + struct.pack('<if', len(keep), float(inf_ms)))

    except Exception:
        try:
            cv2.resize(raw, (PANE_W, PANE_H), dst=_pane_buf, interpolation=cv2.INTER_AREA)
            _pane_buf.tofile(SEG_OUT_FILE)
        except Exception:
            np.zeros((PANE_H, PANE_W, 3), np.uint8).tofile(SEG_OUT_FILE)
        _pipe_out.write(b'\x01' + struct.pack('<if', 0, 0.0))
