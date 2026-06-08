"""
Edge-Art live demo - continuous pose-driven overlay on Wayland.

Architecture (single Python process):

    [gst-launch IN]                    Python                    [gst-launch OUT]
    qtiqmmfsrc cam=0   --raw BGR-->   read frame   ---draw--->   fdsrc fd=0
    qtivtransform                     letterbox 640                rawvideoparse
    video/x-raw,BGR                   dx_engine.run                videoconvert
       fdsink fd=1                    decode 17-KP                 waylandsink
                                      compose overlay              (fullscreen)
                                      write to display

Run on board:
    sh /tmp/run_live_demo.sh
or
    python3 /tmp/live_demo.py --camera 0 --width 1280 --height 720

Stop with Ctrl+C.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from collections import deque

import cv2
import numpy as np
from dx_engine import InferenceEngine, InferenceOption


MODEL = "/usr/share/dx-stream/dx_stream/samples/models/YOLOV5Pose640_1.dxnn"
CONF  = 0.35
NMS_  = 0.5

SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

STYLES = {
    # bone color (BGR), glow color (BGR), thickness, joint radius, glow blur kernel
    "neon":    dict(bone=(255,   0, 255), glow=(0,   229, 255), th=4, jr=6, blur=15),
    "vangogh": dict(bone=( 50, 130, 255), glow=(60,  200, 255), th=8, jr=8, blur=21),
    "comic":   dict(bone=(  0,   0,   0), glow=(50,  180, 255), th=6, jr=8, blur=0),
    "noir":    dict(bone=(255, 255, 255), glow=(120, 120, 120), th=3, jr=5, blur=9),
}


# ---------------------------------------------------------------------------
# GStreamer subprocesses
# ---------------------------------------------------------------------------
def _gst_env() -> dict:
    return {
        **os.environ,
        "XDG_RUNTIME_DIR": "/run/user/root",
        "WAYLAND_DISPLAY": "wayland-1",
        "QT_QPA_PLATFORM": "wayland-egl",
        "GST_DEBUG":       "1",
    }


def spawn_camera(cam_id: int, w: int, h: int, fps: int) -> subprocess.Popen:
    """qtiqmmfsrc → BGR raw → stdout."""
    pipeline = (
        f"qtiqmmfsrc camera={cam_id} ! "
        f"qtivtransform ! "
        f"video/x-raw,format=NV12,width={w},height={h},framerate={fps}/1 ! "
        f"videoconvert ! "
        f"video/x-raw,format=BGR ! "
        f"fdsink fd=1 sync=false"
    )
    stderr_log = open("/tmp/live_cam.log", "w")
    return subprocess.Popen(
        ["gst-launch-1.0", "-q", *pipeline.split()],
        stdout=subprocess.PIPE,
        stderr=stderr_log,
        env=_gst_env(),
        bufsize=0,
    )


def spawn_display(w: int, h: int, fps: int) -> subprocess.Popen:
    """stdin → BGR raw → waylandsink fullscreen."""
    pipeline = (
        f"fdsrc fd=0 ! "
        f"rawvideoparse format=bgr width={w} height={h} framerate={fps}/1 ! "
        f"videoconvert ! "
        f"waylandsink sync=false fullscreen=true"
    )
    stderr_log = open("/tmp/live_disp.log", "w")
    return subprocess.Popen(
        ["gst-launch-1.0", "-q", *pipeline.split()],
        stdin=subprocess.PIPE,
        stderr=stderr_log,
        env=_gst_env(),
        bufsize=0,
    )


def read_exact(fp, n: int) -> bytes | None:
    """Block-read exactly n bytes from a pipe. Returns None on EOF."""
    buf = bytearray()
    while len(buf) < n:
        chunk = fp.read(n - len(buf))
        if not chunk:
            return None if not buf else bytes(buf)   # EOF
        buf.extend(chunk)
    return bytes(buf)


# ---------------------------------------------------------------------------
# YOLOv5-Pose decode (matches capture_and_pose.py)
# ---------------------------------------------------------------------------
def letterbox(img: np.ndarray, size: int = 640, pad: int = 114):
    h, w = img.shape[:2]
    r = min(size / w, size / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), pad, dtype=np.uint8)
    dx, dy = (size - nw) // 2, (size - nh) // 2
    canvas[dy:dy + nh, dx:dx + nw] = resized
    return canvas, r, dx, dy


def nms(boxes, scores, thr):
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    a = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while len(order):
        i = order[0]; keep.append(int(i))
        if len(order) == 1: break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (a[i] + a[order[1:]] - inter + 1e-9)
        order = order[1:][iou < thr]
    return keep


def decode(raw, r, dx, dy):
    obj = raw[:, 4]
    mask = obj > CONF
    cand = raw[mask]
    if len(cand) == 0:
        return []
    boxes  = cand[:, :4]
    scores = cand[:, 4] * cand[:, 5]
    kps    = cand[:, 6:].reshape(-1, 17, 3)
    keep   = nms(boxes, scores, NMS_)
    persons = []
    for i in keep:
        b = boxes[i].copy(); k = kps[i].copy()
        b[0] = (b[0] - dx) / r; b[1] = (b[1] - dy) / r
        b[2] = b[2] / r;        b[3] = b[3] / r
        k[:, 0] = (k[:, 0] - dx) / r
        k[:, 1] = (k[:, 1] - dy) / r
        persons.append({"bbox": b, "kps": k, "score": float(scores[i])})
    return persons


# ---------------------------------------------------------------------------
# Stylized overlay (writes directly onto the camera frame)
# ---------------------------------------------------------------------------
def overlay(frame: np.ndarray, persons: list, style: str) -> np.ndarray:
    s = STYLES.get(style, STYLES["neon"])

    if s["blur"] > 0 and persons:
        glow = np.zeros_like(frame)
        for p in persons:
            _draw_skel(glow, p, s["glow"], s["th"] + 4, s["jr"] + 3)
        glow = cv2.GaussianBlur(glow, (s["blur"], s["blur"]), 0)
        frame = cv2.addWeighted(frame, 1.0, glow, 0.85, 0)

    for p in persons:
        _draw_skel(frame, p, s["bone"], s["th"], s["jr"])

    return frame


def _draw_skel(img, p, color, th, jr):
    kps = p["kps"]
    pts = []
    for x, y, c in kps:
        pts.append((int(x), int(y)) if c > 0.3 else None)
    for a, b in SKELETON:
        if pts[a] and pts[b]:
            cv2.line(img, pts[a], pts[b], color, th, lineType=cv2.LINE_AA)
    for pt in pts:
        if pt is not None:
            cv2.circle(img, pt, jr, color, -1, lineType=cv2.LINE_AA)


def hud(frame, fps_in, fps_npu, n_persons, style, npu_ms):
    h, w = frame.shape[:2]
    bar = 38
    cv2.rectangle(frame, (0, 0), (w, bar), (10, 12, 22), -1)
    cv2.rectangle(frame, (0, bar - 2), (w, bar), (0, 229, 255), -1)
    txt = (f"IMDT QCS8550 + DEEPX  |  STYLE {style.upper()}  |  "
           f"PEOPLE {n_persons}  |  NPU {npu_ms:5.1f}ms  |  "
           f"CAM {fps_in:4.1f}fps  NPU {fps_npu:4.1f}fps")
    cv2.putText(frame, txt, (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 229, 255), 1, lineType=cv2.LINE_AA)
    return frame


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width",  type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps",    type=int, default=30)
    ap.add_argument("--style",  default="neon", choices=list(STYLES))
    ap.add_argument("--style-cycle-sec", type=float, default=0.0,
                    help="Auto-cycle styles every N s (0 = off)")
    ap.add_argument("--no-display", action="store_true")
    args = ap.parse_args()

    w, h, fps = args.width, args.height, args.fps
    frame_bytes = w * h * 3
    print(f"[live] frame {w}x{h} BGR  fps {fps}  bytes/frame {frame_bytes}")
    print(f"[live] style {args.style}  cycle {args.style_cycle_sec}s")

    print("[live] loading model ...")
    opt = InferenceOption()
    try: opt.set_buffer_count(4)
    except Exception: pass
    eng = InferenceEngine(MODEL, opt)

    print("[live] spawning camera gst-launch ...")
    cam = spawn_camera(args.camera, w, h, fps)
    time.sleep(0.5)
    if cam.poll() is not None:
        print(f"[live] camera process died early (rc={cam.returncode})")
        return 1

    disp = None
    if not args.no_display:
        print("[live] spawning waylandsink gst-launch ...")
        disp = spawn_display(w, h, fps)
        time.sleep(0.5)
        if disp.poll() is not None:
            print(f"[live] display process died early (rc={disp.returncode})")
            cam.terminate()
            return 1

    style = args.style
    style_t0 = time.time()
    style_keys = list(STYLES.keys())
    style_idx = style_keys.index(style)

    in_t = deque(maxlen=30)
    npu_t = deque(maxlen=30)
    last_npu_ms = 0.0
    n_persons = 0
    n_frames = 0

    print("[live] running. Ctrl+C to stop")
    try:
        while True:
            # ---- read one BGR frame from camera subprocess --------------
            t_in = time.time()
            raw = read_exact(cam.stdout, frame_bytes)
            if raw is None or len(raw) != frame_bytes:
                got = 0 if raw is None else len(raw)
                print(f"[live] short read ({got}/{frame_bytes}); camera ended")
                if cam.poll() is not None:
                    print(f"[live] cam rc={cam.returncode} - check /tmp/live_cam.log")
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
            in_t.append(time.time() - t_in)

            # ---- inference ----------------------------------------------
            t_n = time.time()
            inp, r, dx, dy = letterbox(frame, 640)
            outs = eng.run(np.expand_dims(inp, 0))
            last_npu_ms = (time.time() - t_n) * 1000.0
            npu_t.append(last_npu_ms / 1000.0)

            persons = decode(outs[0].reshape(-1, 57), r, dx, dy)
            n_persons = len(persons)

            # ---- overlay -------------------------------------------------
            out = overlay(frame, persons, style)
            fps_in  = 1.0 / (sum(in_t) / len(in_t) + 1e-9)
            fps_npu = 1.0 / (sum(npu_t) / len(npu_t) + 1e-9)
            out = hud(out, fps_in, fps_npu, n_persons, style, last_npu_ms)

            # ---- display -------------------------------------------------
            if disp is not None:
                try:
                    disp.stdin.write(out.tobytes())
                except BrokenPipeError:
                    print("[live] display pipe closed; stopping")
                    break

            n_frames += 1
            if n_frames % 30 == 0:
                print(f"[live] {n_frames:5d} frames  "
                      f"cam {fps_in:4.1f}fps  npu {fps_npu:4.1f}fps  "
                      f"npu {last_npu_ms:5.1f}ms  people {n_persons}")

            # ---- style auto-cycle ---------------------------------------
            if args.style_cycle_sec > 0 and time.time() - style_t0 > args.style_cycle_sec:
                style_idx = (style_idx + 1) % len(style_keys)
                style = style_keys[style_idx]
                style_t0 = time.time()
                print(f"[live] style -> {style}")
    except KeyboardInterrupt:
        print("\n[live] Ctrl+C")
    finally:
        print("[live] shutting down ...")
        for p in (cam, disp):
            if p is None: continue
            try:
                p.terminate(); p.wait(timeout=2)
            except Exception:
                p.kill()
        print(f"[live] processed {n_frames} frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())
