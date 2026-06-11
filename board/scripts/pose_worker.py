#!/usr/bin/env python3
"""
Persistent DEEPX pose worker — spawned once by the C++ app.

Protocol (binary):
  C++ → worker : 1 byte  (0x01 = run, 0x00 = quit)
  worker → C++ : int32 n_persons
                 for each person: 52 floats  [score, kp0_x, kp0_y, kp0_c, kp1_x ...]
"""
import os, struct, sys
import numpy as np
import cv2
from dx_engine import InferenceEngine, InferenceOption

POSE_MODEL = "/usr/share/dx-stream/dx_stream/samples/models/YOLOV5Pose640_1.dxnn"
FRAME_FILE = "/tmp/ea_frame.bin"
CAM_W, CAM_H = 1280, 720
POSE_SIZE    = 640
CONF_THR     = 0.30
NMS_THR      = 0.45

# ---------- init engine ----------
opt = InferenceOption()
try:
    opt.set_buffer_count(4)
except Exception:
    pass
engine = None
try:
    engine = InferenceEngine(POSE_MODEL, opt)
except Exception as e:
    sys.stderr.write(f"[pose_worker] DEEPX init failed: {e}\n")
    sys.stderr.flush()

sys.stdout.buffer.write(b"READY\n")
sys.stdout.buffer.flush()

# ---------- helpers ----------
def letterbox(img, size=640, pad=114):
    h, w = img.shape[:2]
    r = min(size/w, size/h)
    nw, nh = int(round(w*r)), int(round(h*r))
    canvas = np.full((size, size, 3), pad, np.uint8)
    res = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dx, dy = (size-nw)//2, (size-nh)//2
    canvas[dy:dy+nh, dx:dx+nw] = res
    return canvas, r, dx, dy

def nms(boxes, scores, thr):
    if not len(boxes):
        return []
    x1 = boxes[:,0]-boxes[:,2]/2;  x2 = boxes[:,0]+boxes[:,2]/2
    y1 = boxes[:,1]-boxes[:,3]/2;  y2 = boxes[:,1]+boxes[:,3]/2
    areas = (x2-x1)*(y2-y1)
    order = scores.argsort()[::-1]
    keep  = []
    while len(order):
        i = order[0]; keep.append(int(i))
        if len(order) == 1: break
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2-xx1)*np.maximum(0, yy2-yy1)
        iou   = inter/(areas[i]+areas[order[1:]]-inter+1e-9)
        order = order[1:][iou < thr]
    return keep

def decode_pose(raw, r, dx, dy):
    obj  = raw[:,4]; mask = obj > CONF_THR; cand = raw[mask]
    if not len(cand):
        return []
    boxes  = cand[:,:4]
    scores = cand[:,4]*cand[:,5]
    kps    = cand[:,6:].reshape(-1, 17, 3)
    keep   = nms(boxes, scores, NMS_THR)
    persons = []
    for i in keep:
        b = boxes[i].copy(); k = kps[i].copy()
        b[0] = (b[0]-dx)/r;  b[1] = (b[1]-dy)/r
        b[2] /= r;            b[3] /= r
        k[:,0] = (k[:,0]-dx)/r
        k[:,1] = (k[:,1]-dy)/r
        persons.append({"score": float(scores[i]), "kps": k})
    return persons

# ---------- main loop ----------
stdin_fd = sys.stdin.buffer

while True:
    cmd = stdin_fd.read(1)
    if not cmd or cmd == b'\x00':
        break
    if cmd != b'\x01':
        continue

    try:
        frame = np.fromfile(FRAME_FILE, dtype=np.uint8).reshape(CAM_H, CAM_W, 3)
    except Exception:
        sys.stdout.buffer.write(struct.pack('<i', 0))
        sys.stdout.buffer.flush()
        continue

    try:
        lb, r, dx, dy = letterbox(frame, POSE_SIZE)
        outs    = engine.run(np.expand_dims(lb, 0))
        persons = decode_pose(outs[0].reshape(-1, 57), r, dx, dy)
    except Exception as e:
        persons = []

    n = len(persons)
    buf = struct.pack('<i', n)
    for p in persons:
        vals = [p['score']]
        for x, y, c in p['kps']:
            vals += [float(x), float(y), float(c)]
        buf += struct.pack('<52f', *vals)
    sys.stdout.buffer.write(buf)
    sys.stdout.buffer.flush()
