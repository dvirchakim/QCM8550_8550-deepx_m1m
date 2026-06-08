"""
Capture a real frame from camera 0, run YOLOv5-Pose on the DX-M1,
decode 17-KP, draw skeleton, save annotated PNG.

Validates the full edge pipeline: sensor -> DEEPX -> Python -> overlay.

Output:  /tmp/pose_annotated.png  (pull to host with adb)
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from dx_engine import InferenceEngine, InferenceOption

MODEL  = "/usr/share/dx-stream/dx_stream/samples/models/YOLOV5Pose640_1.dxnn"
JPG    = Path("/tmp/cam_grab.jpg")
OUT    = Path("/tmp/pose_annotated.png")
CONF   = 0.35
NMS_IO = 0.5

# COCO skeleton edges
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


# ---------------------------------------------------------------------------
def grab_frame(camera: int = 0) -> np.ndarray:
    """Run qtiqmmfsrc with a short timeout; multifilesink writes a JPEG."""
    if JPG.exists():
        JPG.unlink()
    env = {
        "XDG_RUNTIME_DIR": "/run/user/root",
        "WAYLAND_DISPLAY": "wayland-1",
        "PATH":            "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "HOME":            "/home/root",
    }
    cmd = [
        "timeout", "5",
        "gst-launch-1.0", "-e",
        "qtiqmmfsrc", f"camera={camera}", "!",
        "qtivtransform", "!",
        "video/x-raw,format=NV12,width=1280,height=720,framerate=15/1", "!",
        "videoconvert", "!",
        "jpegenc", "quality=90", "!",
        "multifilesink", f"location={JPG}", "max-files=1",
    ]
    # timeout exits with 124/143; that is expected because qtiqmmfsrc is live
    subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not JPG.exists():
        raise RuntimeError("gst-launch produced no JPEG (camera busy?)")
    img = cv2.imread(str(JPG))
    if img is None:
        raise RuntimeError(f"Failed to decode {JPG}")
    return img


def letterbox(img: np.ndarray, size: int = 640, pad: int = 114
              ) -> tuple[np.ndarray, float, tuple[int, int]]:
    h, w = img.shape[:2]
    r = min(size / w, size / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), pad, dtype=np.uint8)
    dx, dy = (size - nw) // 2, (size - nh) // 2
    canvas[dy:dy + nh, dx:dx + nw] = resized
    return canvas, r, (dx, dy)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> list[int]:
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while len(order):
        i = order[0]
        keep.append(int(i))
        if len(order) == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w_ = np.maximum(0, xx2 - xx1); h_ = np.maximum(0, yy2 - yy1)
        inter = w_ * h_
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou < iou_thresh]
    return keep


def decode(raw: np.ndarray, conf: float, r: float, dx: int, dy: int
           ) -> list[dict]:
    """raw: (25500, 57) float32. Returns list of {'bbox': xywh, 'kps': (17,3)}."""
    obj = raw[:, 4]
    mask = obj > conf
    cand = raw[mask]
    if len(cand) == 0:
        return []
    boxes  = cand[:, :4]
    # 57 channels = 4 bbox + 1 objectness + 1 class score (person) + 17*3 keypoints
    scores = cand[:, 4] * cand[:, 5]      # obj * cls
    kps    = cand[:, 6:].reshape(-1, 17, 3)
    keep   = nms(boxes, scores, NMS_IO)
    out = []
    for i in keep:
        b = boxes[i].copy()
        k = kps[i].copy()
        # Reverse letterbox: subtract pad then divide by ratio
        b[0] = (b[0] - dx) / r
        b[1] = (b[1] - dy) / r
        b[2] = b[2] / r
        b[3] = b[3] / r
        k[:, 0] = (k[:, 0] - dx) / r
        k[:, 1] = (k[:, 1] - dy) / r
        out.append({"bbox": b.tolist(), "kps": k.tolist(), "score": float(scores[i])})
    return out


def draw(frame: np.ndarray, persons: list[dict]) -> np.ndarray:
    out = frame.copy()
    for p in persons:
        x, y, w, h = p["bbox"]
        x1, y1 = int(x - w / 2), int(y - h / 2)
        x2, y2 = int(x + w / 2), int(y + h / 2)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
        kps = p["kps"]
        for kx, ky, kc in kps:
            if kc > 0.3:
                cv2.circle(out, (int(kx), int(ky)), 5, (0, 229, 255), -1)
        for a, b in SKELETON:
            if kps[a][2] > 0.3 and kps[b][2] > 0.3:
                cv2.line(out,
                         (int(kps[a][0]), int(kps[a][1])),
                         (int(kps[b][0]), int(kps[b][1])),
                         (255, 0, 255), 3, lineType=cv2.LINE_AA)
        cv2.putText(out, f"score {p['score']:.2f}", (x1, max(20, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return out


# ---------------------------------------------------------------------------
def main() -> int:
    print("[capture] grabbing frame from camera 0 ...")
    t = time.time()
    frame = grab_frame()
    print(f"[capture] got {frame.shape} in {(time.time() - t) * 1000:.0f} ms")

    print("[capture] preparing input (letterbox 640x640) ...")
    inp, r, (dx, dy) = letterbox(frame, 640)
    inp_batched = np.expand_dims(inp, 0).astype(np.uint8)  # (1,640,640,3)

    print("[capture] loading model ...")
    eng = InferenceEngine(MODEL, InferenceOption())

    print("[capture] running inference x10 for warm timing ...")
    for _ in range(3):
        eng.run(inp_batched)
    t = time.time()
    outs = None
    for _ in range(10):
        outs = eng.run(inp_batched)
    print(f"[capture] 10 runs = {(time.time() - t) * 1000:.1f} ms "
          f"({10 / (time.time() - t):.1f} FPS)")

    raw = outs[0].reshape(-1, 57)
    persons = decode(raw, CONF, r, dx, dy)
    print(f"[capture] decoded {len(persons)} persons")
    for i, p in enumerate(persons):
        print(f"  #{i}  score={p['score']:.2f}  bbox={[round(v,1) for v in p['bbox']]}")

    out = draw(frame, persons)
    cv2.imwrite(str(OUT), out)
    print(f"[capture] wrote {OUT}  ({out.shape})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
