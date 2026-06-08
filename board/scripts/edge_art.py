#!/usr/bin/env python3
"""
Edge-Art Demo – IMDT QCS8550 + DEEPX DX-M1
============================================
Left  pane : Camera 0  +  DEEPX YOLOv5-Pose  (DX-M1 NPU)
Right pane : HTP DeepLabV3 segmentation  ->  OpenCV generative art  (Qualcomm DSP/HTP)

Deploy:
    adb push scripts/edge_art.py /data/local/tmp/edge_art.py
    adb shell python3 /data/local/tmp/edge_art.py

Stop:
    Ctrl-C (or kill from adb)
"""
from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import time
from collections import deque

import shlex
import cv2
import numpy as np
from dx_engine import InferenceEngine, InferenceOption

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CAM_W, CAM_H, CAM_FPS = 1280, 720, 15
PANE_W, PANE_H        = 640, 720           # each half-pane
SEG_W,  SEG_H         = 512, 512           # HTP model input size
SEG_OUT_W, SEG_OUT_H  = 512, 520           # actual qtimlvsegmentation output (BGRA)
SEG_RATE              = 3                  # run HTP every N camera frames

POSE_MODEL = "/usr/share/dx-stream/dx_stream/samples/models/YOLOV5Pose640_1.dxnn"
SEG_MODEL  = "/opt/deeplabv3_plus_mobilenet_quantized.tflite"
SEG_LABELS = "/opt/deeplabv3_resnet50.labels"
SEG_CONSTS = "deeplab,q-offsets=<8.0>,q-scales=<0.0040499246679246426>;"

POSE_CONF  = 0.30
NMS_THR    = 0.45

SKELETON = [
    (0,1),(0,2),(1,3),(2,4),
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]

STYLES = {
    "neon":    dict(bone=(255,0,255),   glow=(0,229,255),   th=4, jr=6, blur=15),
    "vangogh": dict(bone=(50,130,255),  glow=(60,200,255),  th=8, jr=8, blur=21),
    "comic":   dict(bone=(0,0,0),       glow=(50,180,255),  th=6, jr=8, blur=0),
    "noir":    dict(bone=(255,255,255), glow=(120,120,120), th=3, jr=5, blur=9),
}
STYLE_CYCLE_SEC = 30.0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _gst_env() -> dict:
    e = os.environ.copy()
    e["XDG_RUNTIME_DIR"]               = "/run/user/root"
    e["WAYLAND_DISPLAY"]               = "wayland-1"
    e["QT_QPA_PLATFORM"]               = "wayland-egl"
    e["QT_WAYLAND_SHELL_INTEGRATION"]  = "wl-shell"
    e["ADSP_LIBRARY_PATH"]             = "/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp"
    return e


def read_exact(fp, n: int):
    buf = bytearray()
    while len(buf) < n:
        chunk = fp.read(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


# ---------------------------------------------------------------------------
# GStreamer subprocesses
# ---------------------------------------------------------------------------
def spawn_camera() -> subprocess.Popen:
    """qtiqmmfsrc -> BGR 1280x720 -> stdout."""
    pipe = (
        f"gst-launch-1.0 -q "
        f"qtiqmmfsrc camera=0 ! "
        f"qtivtransform ! "
        f"video/x-raw,width={CAM_W},height={CAM_H},format=NV12,framerate={CAM_FPS}/1 ! "
        f"videoconvert ! "
        f"video/x-raw,format=BGR ! "
        f"fdsink fd=1 sync=false"
    )
    return subprocess.Popen(
        shlex.split(pipe),
        stdout=subprocess.PIPE,
        stderr=open("/tmp/ea_cam.log", "w"),
        env=_gst_env(), bufsize=0,
    )


def spawn_seg() -> subprocess.Popen:
    """stdin BGR 512x512 -> HTP DeepLabV3 -> stdout BGRA 512x520."""
    pipe = (
        f"gst-launch-1.0 -q "
        f"fdsrc fd=0 ! "
        f"rawvideoparse format=bgr width={SEG_W} height={SEG_H} "
        f"framerate=5/1 ! "
        f"qtimlvconverter ! "
        f"qtimltflite model={SEG_MODEL} "
        f"external-delegate-path=libQnnTFLiteDelegate.so "
        f"external-delegate-options=QNNExternalDelegate,backend_type=htp,"
        f"htp_performance_mode=(string)2; ! "
        f"qtimlvsegmentation labels={SEG_LABELS} module=deeplab-argmax "
        f"constants=\"{SEG_CONSTS}\" ! "
        f"fdsink fd=1 sync=false"
    )
    return subprocess.Popen(
        shlex.split(pipe),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=open("/tmp/ea_seg.log", "w"),
        env=_gst_env(), bufsize=0,
    )


def spawn_display() -> subprocess.Popen:
    """stdin BGR 1280x720 -> waylandsink fullscreen."""
    w, h = CAM_W, CAM_H
    pipe = (
        f"gst-launch-1.0 -q "
        f"fdsrc fd=0 ! "
        f"rawvideoparse format=bgr width={w} height={h} "
        f"framerate={CAM_FPS}/1 ! "
        f"videoconvert ! "
        f"waylandsink sync=false fullscreen=true"
    )
    return subprocess.Popen(
        shlex.split(pipe),
        stdin=subprocess.PIPE,
        stderr=open("/tmp/ea_disp.log", "w"),
        env=_gst_env(), bufsize=0,
    )


# ---------------------------------------------------------------------------
# Pose decode
# ---------------------------------------------------------------------------
def letterbox(img: np.ndarray, size: int = 640, pad: int = 114):
    h, w = img.shape[:2]
    r = min(size / w, size / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    canvas = np.full((size, size, 3), pad, dtype=np.uint8)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dx, dy = (size - nw) // 2, (size - nh) // 2
    canvas[dy:dy + nh, dx:dx + nw] = resized
    return canvas, r, dx, dy


def nms(boxes, scores, thr):
    if len(boxes) == 0:
        return []
    x1 = boxes[:,0] - boxes[:,2]/2;  x2 = boxes[:,0] + boxes[:,2]/2
    y1 = boxes[:,1] - boxes[:,3]/2;  y2 = boxes[:,1] + boxes[:,3]/2
    areas = (x2-x1)*(y2-y1); order = scores.argsort()[::-1]; keep = []
    while len(order):
        i = order[0]; keep.append(int(i))
        if len(order) == 1: break
        xx1=np.maximum(x1[i],x1[order[1:]]); yy1=np.maximum(y1[i],y1[order[1:]])
        xx2=np.minimum(x2[i],x2[order[1:]]); yy2=np.minimum(y2[i],y2[order[1:]])
        inter=np.maximum(0,xx2-xx1)*np.maximum(0,yy2-yy1)
        iou=inter/(areas[i]+areas[order[1:]]-inter+1e-9)
        order=order[1:][iou<thr]
    return keep


def decode(raw, r, dx, dy):
    obj  = raw[:, 4]; mask = obj > POSE_CONF; cand = raw[mask]
    if len(cand) == 0: return []
    boxes=cand[:,:4]; scores=cand[:,4]*cand[:,5]; kps=cand[:,6:].reshape(-1,17,3)
    keep=nms(boxes,scores,NMS_THR); persons=[]
    for i in keep:
        b=boxes[i].copy(); k=kps[i].copy()
        b[0]=(b[0]-dx)/r; b[1]=(b[1]-dy)/r; b[2]/=r; b[3]/=r
        k[:,0]=(k[:,0]-dx)/r; k[:,1]=(k[:,1]-dy)/r
        persons.append({"bbox":b,"kps":k,"score":float(scores[i])})
    return persons


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------
def draw_pose(frame: np.ndarray, persons: list, style: str) -> np.ndarray:
    s = STYLES.get(style, STYLES["neon"])
    out = frame.copy()
    if s["blur"] > 0 and persons:
        glow = np.zeros_like(out)
        for p in persons: _skel(glow, p, s["glow"], s["th"]+4, s["jr"]+3)
        glow = cv2.GaussianBlur(glow, (s["blur"], s["blur"]), 0)
        out  = cv2.addWeighted(out, 1.0, glow, 0.85, 0)
    for p in persons: _skel(out, p, s["bone"], s["th"], s["jr"])
    return out


def _skel(img, p, color, th, jr):
    kps = p["kps"]
    pts = [(int(x), int(y)) if c > 0.3 else None for x, y, c in kps]
    for a, b in SKELETON:
        if pts[a] and pts[b]: cv2.line(img, pts[a], pts[b], color, th, cv2.LINE_AA)
    for pt in pts:
        if pt: cv2.circle(img, pt, jr, color, -1, cv2.LINE_AA)


def stylize_seg(seg_bgra: np.ndarray, style: str) -> np.ndarray:
    """Apply generative art effect to the HTP segmentation output (BGRA input)."""
    seg_bgr = cv2.cvtColor(seg_bgra, cv2.COLOR_BGRA2BGR)
    if style == "neon":
        hsv  = cv2.cvtColor(seg_bgr, cv2.COLOR_BGR2HSV)
        hsv[:,:,1] = np.clip(hsv[:,:,1].astype(int) * 2, 0, 255).astype(np.uint8)
        hsv[:,:,2] = np.clip(hsv[:,:,2].astype(int) * 1.4, 0, 255).astype(np.uint8)
        out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        glow = cv2.GaussianBlur(out, (21, 21), 0)
        return cv2.addWeighted(out, 1.0, glow, 0.7, 0)
    elif style == "vangogh":
        noise = np.random.randint(0, 40, seg_bgr.shape, dtype=np.uint8)
        out   = cv2.add(seg_bgr, noise)
        return cv2.medianBlur(out, 7)
    elif style == "comic":
        gray  = cv2.cvtColor(seg_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 40, 120)
        out   = seg_bgr.copy()
        out[edges > 0] = (0, 0, 0)
        return out
    elif style == "noir":
        gray = cv2.cvtColor(seg_bgr, cv2.COLOR_BGR2GRAY)
        out  = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        rain = np.zeros_like(out)
        ry   = np.random.randint(0, CAM_H, 200)
        rx   = np.random.randint(0, PANE_W, 200)
        for y, x in zip(ry, rx):
            cv2.line(out, (x, y), (x+2, min(y+15, CAM_H-1)), (180,180,180), 1)
        return out
    return seg_bgr


def hud_left(frame, fps, npu_ms, n_persons, style):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 36), (10,12,22), -1)
    cv2.rectangle(frame, (0, 34), (w, 36), (0,229,255), -1)
    txt = f"DEEPX DX-M1  POSE  {n_persons}P  {npu_ms:.0f}ms  {fps:.0f}fps  [{style.upper()}]"
    cv2.putText(frame, txt, (8,24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0,229,255), 1, cv2.LINE_AA)


def hud_right(frame, fps):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 36), (10,22,12), -1)
    cv2.rectangle(frame, (0, 34), (w, 36), (0,255,100), -1)
    txt = f"QUALCOMM HTP  DEEPLABV3  GENERATIVE ART  {fps:.0f}fps"
    cv2.putText(frame, txt, (8,24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0,255,100), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# HTP segmentation reader thread
# ---------------------------------------------------------------------------
class SegReader:
    """Non-blocking reader for the HTP segmentation subprocess stdout."""
    def __init__(self, proc: subprocess.Popen):
        self._proc    = proc
        self._q       : queue.Queue = queue.Queue(maxsize=2)
        self._stopped = threading.Event()
        self._thread  = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._fps_t   = deque(maxlen=10)

    def _run(self):
        nbytes = SEG_OUT_W * SEG_OUT_H * 4  # BGRA 512x520
        while not self._stopped.is_set():
            data = read_exact(self._proc.stdout, nbytes)
            if data is None:
                break
            frame = np.frombuffer(data, dtype=np.uint8).reshape(SEG_OUT_H, SEG_OUT_W, 4)
            self._fps_t.append(time.time())
            try:
                self._q.get_nowait()   # drop old frame if consumer is slow
            except queue.Empty:
                pass
            self._q.put(frame)

    def latest(self):
        try:
            return self._q.get_nowait()
        except queue.Empty:
            return None

    @property
    def fps(self):
        ts = list(self._fps_t)
        if len(ts) < 2: return 0.0
        return (len(ts)-1) / max(ts[-1]-ts[0], 1e-9)

    def stop(self):
        self._stopped.set()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    style_keys = list(STYLES.keys())
    style_idx  = 0
    style      = style_keys[style_idx]
    style_t0   = time.time()

    print("[edge-art] Loading DEEPX pose model ...")
    opt = InferenceOption()
    try: opt.set_buffer_count(4)
    except Exception: pass
    engine = InferenceEngine(POSE_MODEL, opt)
    print("[edge-art] Model loaded.")

    print("[edge-art] Restarting qmmf-server ...")
    subprocess.run(["systemctl", "restart", "qmmf-server.service"],
                   check=False, stderr=subprocess.DEVNULL)
    time.sleep(3.0)

    print("[edge-art] Spawning camera gst-launch ...")
    cam_proc = spawn_camera()
    time.sleep(1.0)
    if cam_proc.poll() is not None:
        print(f"[edge-art] Camera process died (rc={cam_proc.returncode}). Check /tmp/ea_cam.log")
        return 1

    print("[edge-art] Spawning HTP segmentation gst-launch ...")
    seg_proc = spawn_seg()
    time.sleep(6.0)   # HTP delegate needs extra init time
    if seg_proc.poll() is not None:
        print(f"[edge-art] Segmentation process died (rc={seg_proc.returncode}). Check /tmp/ea_seg.log")
        cam_proc.terminate()
        return 1
    seg_reader = SegReader(seg_proc)

    print("[edge-art] Spawning display gst-launch ...")
    disp_proc = spawn_display()
    time.sleep(0.5)
    if disp_proc.poll() is not None:
        print(f"[edge-art] Display process died (rc={disp_proc.returncode}). Check /tmp/ea_disp.log")
        cam_proc.terminate(); seg_proc.terminate()
        return 1

    cam_bytes  = CAM_W * CAM_H * 3
    seg_bytes  = SEG_OUT_W * SEG_OUT_H * 4  # BGRA 512x520

    in_t      = deque(maxlen=30)
    npu_t     = deque(maxlen=30)
    n_persons = 0
    last_npu  = 0.0
    n_frames  = 0
    latest_art: np.ndarray | None = None
    seg_placeholder = np.zeros((PANE_H, PANE_W, 3), dtype=np.uint8)
    cv2.putText(seg_placeholder, "HTP warming up...", (60, PANE_H//2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,100), 2, cv2.LINE_AA)

    print("[edge-art] Running.  Ctrl-C to stop.")
    try:
        while True:
            # -------- read camera frame --------------------------------
            t0 = time.time()
            raw = read_exact(cam_proc.stdout, cam_bytes)
            if raw is None:
                print("[edge-art] Camera pipe closed.")
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(CAM_H, CAM_W, 3)
            in_t.append(time.time() - t0)

            # -------- DEEPX pose inference -----------------------------
            t1 = time.time()
            inp, r, dx, dy = letterbox(frame, 640)
            outs = engine.run(np.expand_dims(inp, 0))
            last_npu = (time.time() - t1) * 1000.0
            npu_t.append(last_npu / 1000.0)
            persons   = decode(outs[0].reshape(-1, 57), r, dx, dy)
            n_persons = len(persons)

            # -------- left pane: camera + pose -------------------------
            left = draw_pose(frame, persons, style)
            left = cv2.resize(left, (PANE_W, PANE_H))
            hud_left(left,
                     1.0 / (sum(in_t)/len(in_t)+1e-9),
                     last_npu, n_persons, style)

            # -------- send frame to HTP segmentation (every N frames) --
            if n_frames % SEG_RATE == 0:
                seg_in = cv2.resize(frame, (SEG_W, SEG_H))
                try:
                    seg_proc.stdin.write(seg_in.tobytes())
                    seg_proc.stdin.flush()
                except BrokenPipeError:
                    print("[edge-art] Seg pipe broken — check /tmp/ea_seg.log")

            # -------- right pane: HTP segmentation art -----------------
            seg_frame = seg_reader.latest()
            if seg_frame is not None:
                art = stylize_seg(seg_frame, style)
                art = cv2.resize(art, (PANE_W, PANE_H))
                hud_right(art, seg_reader.fps)
                latest_art = art

            right = latest_art if latest_art is not None else seg_placeholder

            # -------- compose & display --------------------------------
            composed = np.hstack([left, right])
            try:
                disp_proc.stdin.write(composed.tobytes())
            except BrokenPipeError:
                print("[edge-art] Display pipe broken.")
                break

            # -------- style auto-cycle ---------------------------------
            if time.time() - style_t0 > STYLE_CYCLE_SEC:
                style_idx = (style_idx + 1) % len(style_keys)
                style     = style_keys[style_idx]
                style_t0  = time.time()
                print(f"[edge-art] style -> {style}")

            n_frames += 1
            if n_frames % 30 == 0:
                fps_cam = 1.0/(sum(in_t)/len(in_t)+1e-9)
                fps_npu = 1.0/(sum(npu_t)/len(npu_t)+1e-9)
                print(f"[edge-art] {n_frames:5d} frames  "
                      f"cam {fps_cam:4.1f}fps  deepx {fps_npu:4.1f}fps/{last_npu:.0f}ms  "
                      f"htp {seg_reader.fps:.1f}fps  people {n_persons}")

    except KeyboardInterrupt:
        print("\n[edge-art] Ctrl-C")
    finally:
        print("[edge-art] Shutting down ...")
        seg_reader.stop()
        for p in (cam_proc, seg_proc, disp_proc):
            if p is None: continue
            try: p.terminate(); p.wait(timeout=3)
            except Exception: p.kill()
        print(f"[edge-art] Done — {n_frames} frames processed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
