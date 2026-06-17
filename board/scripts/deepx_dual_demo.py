#!/usr/bin/env python3
"""
DeepX Dual Demo  —  IMDT QCS8550 + DEEPX DX-M1   (clean rebuild)
=================================================================
2x2 quadrant layout (1920x1080), each quadrant 960x540:

  ┌────────────────────┬────────────────────┐
  │ cam0  RGB           │ cam1  RGB + class   │   top
  ├────────────────────┼────────────────────┤
  │ cam0  SCDepthV3     │ TrOCR  text         │   bottom
  └────────────────────┴────────────────────┘
   DEEPX NPE-0 (depth)    DEEPX NPE-1 (cls)
                          ONNX Runtime (TrOCR)

Architecture (hard-won constraints on this board):
  * BOTH cameras are captured by ONE gst-launch (qtivcomposer cam0|cam1).
    Two separate qtiqmmfsrc client processes crash the shared qmmf-server.
  * ONE DeepX worker process owns both .dxnn models (DXRT is single-access);
    the main loop alternates depth/cls requests.
  * A separate, PERSISTENT TrOCR worker (ONNX Runtime, CPU) loads once and
    serves frames on demand — no per-call model reload.
  * All IPC is via pipes (no /tmp files); the main process owns ALL rendering.
  * Camera/display are separate gst processes — this works from a CLEAN boot
    (fresh camera HAL); never restart qmmf-server while streaming.
"""
from __future__ import annotations
import os, select, shlex, signal, struct, subprocess, sys, threading, time
import cv2
import numpy as np

sys.path.insert(0, '/data/local/tmp')

# ── Config ─────────────────────────────────────────────────────────────────────
CAM_W, CAM_H, CAM_FPS = 1280, 720, 15
DISP_W, DISP_H        = 1920, 1080
QUAD_W, QUAD_H        = 960, 540          # each of the four panels

DEPTH_W, DEPTH_H = 320, 256               # SCDepthV3 input
FACE             = 640                    # YOLOv5s-Face input
N_LMK            = 68                     # 3DDFA landmarks

COMBO_W, COMBO_H = CAM_W * 2, CAM_H       # 2560 x 720 (cam0|cam1)
COMBO_BYTES      = COMBO_W * COMBO_H * 3

DEEPX_WORKER = '/data/local/tmp/dual_deepx_worker.py'
OCR_WORKER   = '/data/local/tmp/easyocr_worker.py'
ADSP_PATH    = '/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp'
ORT_PATH     = '/data/local/tmp/ort181'

OCR_INTERVAL_S    = 3.0
CAM_STALL_TIMEOUT = 5.0
DEPTH_MAP_BYTES   = DEPTH_H * DEPTH_W * 4   # float32 reply
DEPTH_IN_BYTES    = DEPTH_H * DEPTH_W * 3
FACE_IN_BYTES     = FACE * FACE * 3

# 68-landmark polyline groups (jaw, brows, nose, eyes, lips)
FACE_GROUPS = [
    list(range(0, 17)),    # jaw
    list(range(17, 22)),   # right brow
    list(range(22, 27)),   # left brow
    list(range(27, 31)),   # nose bridge
    list(range(31, 36)),   # nose base
    list(range(36, 42)) + [36],   # right eye (closed)
    list(range(42, 48)) + [42],   # left eye (closed)
    list(range(48, 60)) + [48],   # outer lips (closed)
    list(range(60, 68)) + [60],   # inner lips (closed)
]

# ── Environment ────────────────────────────────────────────────────────────────
def _env():
    e = os.environ.copy()
    e['XDG_RUNTIME_DIR'] = '/run/user/root'
    e['WAYLAND_DISPLAY'] = 'wayland-1'
    e['QT_QPA_PLATFORM'] = 'wayland-egl'
    e['QT_WAYLAND_SHELL_INTEGRATION'] = 'wl-shell'
    e['ADSP_LIBRARY_PATH'] = ADSP_PATH
    return e

# ── GStreamer processes ──────────────────────────────────────────────────────--
def spawn_cameras():
    """One process, both cameras, hardware-composited side-by-side (cam0|cam1)."""
    q = 'queue max-size-buffers=5 leaky=downstream'
    cam = (lambda i: f'qtiqmmfsrc camera={i} ! qtivtransform ! '
                     f'video/x-raw,width={CAM_W},height={CAM_H},format=NV12,'
                     f'framerate={CAM_FPS}/1 ! {q} ! comp.sink_{i}')
    pipe = (
        f'gst-launch-1.0 -q {cam(0)} {cam(1)} '
        f'qtivcomposer name=comp '
        f'sink_0::position="<0,0>" sink_0::dimensions="<{CAM_W},{CAM_H}>" '
        f'sink_1::position="<{CAM_W},0>" sink_1::dimensions="<{CAM_W},{CAM_H}>" ! '
        f'qtivtransform ! video/x-raw,format=NV12,width={COMBO_W},height={COMBO_H} ! '
        f'videoconvert ! video/x-raw,format=BGR ! '
        f'queue max-size-buffers=2 leaky=downstream ! fdsink fd=1 sync=false'
    )
    return subprocess.Popen(shlex.split(pipe), stdout=subprocess.PIPE,
                            stderr=open('/tmp/dd_cam.log', 'w'),
                            env=_env(), bufsize=0)


def spawn_display():
    pipe = (f'gst-launch-1.0 -q fdsrc fd=0 ! '
            f'rawvideoparse format=bgr width={DISP_W} height={DISP_H} framerate={CAM_FPS}/1 ! '
            f'videoconvert ! video/x-raw,format=BGRx ! waylandsink sync=false fullscreen=true')
    return subprocess.Popen(shlex.split(pipe), stdin=subprocess.PIPE,
                            stderr=open('/tmp/dd_disp.log', 'w'),
                            env=_env(), bufsize=0)


def read_exact(fp, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = fp.read(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)

# ── Worker management ──────────────────────────────────────────────────────────
def spawn_deepx_worker():
    return subprocess.Popen(['python3', DEEPX_WORKER],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=sys.stderr, bufsize=0)


def spawn_ocr_worker():
    env = os.environ.copy()
    pp = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = f'{ORT_PATH}:{pp}' if pp else ORT_PATH
    return subprocess.Popen(['python3', OCR_WORKER],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=sys.stderr, bufsize=0, env=env)


def wait_ready(proc, tag, timeout=180):
    deadline = time.time() + timeout
    buf = b''
    while time.time() < deadline:
        ch = proc.stdout.read(1)
        if not ch:
            return False
        buf += ch
        if buf.endswith(b'READY\n'):
            print(f'[{tag}] worker ready.', flush=True)
            return True
    return False

# ── Camera capture thread (in-memory, no files) ────────────────────────────────
_frame0 = None
_frame1 = None
_frame_lock = threading.Lock()


def cam_thread(cam_ref, restart_fn):
    proc = cam_ref[0]
    fd   = proc.stdout.fileno()
    buf  = bytearray()
    last = time.time()

    def restart(reason):
        nonlocal proc, fd, last
        print(f'[cam] {reason} — restarting cameras ...', flush=True)
        try:
            proc.terminate(); proc.wait(timeout=4)
        except Exception:
            try: proc.kill(); proc.wait(timeout=2)
            except Exception: pass
        time.sleep(1.5)
        proc = restart_fn(); cam_ref[0] = proc
        fd = proc.stdout.fileno(); buf.clear(); last = time.time()
        time.sleep(1.5)

    global _frame0, _frame1
    while True:
        r, _, _ = select.select([fd], [], [], 1.0)
        if r:
            try:
                chunk = os.read(fd, COMBO_BYTES - len(buf))
            except OSError:
                chunk = b''
            if chunk:
                last = time.time()
                buf.extend(chunk)
                if len(buf) >= COMBO_BYTES:
                    combo = np.frombuffer(bytes(buf[:COMBO_BYTES]),
                                          np.uint8).reshape(COMBO_H, COMBO_W, 3)
                    f0 = combo[:, :CAM_W].copy()
                    f1 = combo[:, CAM_W:].copy()
                    with _frame_lock:
                        _frame0, _frame1 = f0, f1
                    del buf[:COMBO_BYTES]
            else:
                restart('pipe closed (EOF)')
        else:
            if proc.poll() is not None:
                restart('process exited')
            elif time.time() - last > CAM_STALL_TIMEOUT:
                restart('no frames (camera service died)')

# ── DeepX request helpers ──────────────────────────────────────────────────────
def letterbox(img, size=FACE, pad=114):
    """Resize keeping aspect ratio into a square `size` canvas; return scale+pads."""
    h, w = img.shape[:2]
    r = min(size / w, size / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    canvas = np.full((size, size, 3), pad, np.uint8)
    res = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dx, dy = (size - nw) // 2, (size - nh) // 2
    canvas[dy:dy+nh, dx:dx+nw] = res
    return canvas, r, dx, dy


def req_infer(worker, frame0, frame1):
    """Send depth (cam0) + face (cam1) inputs; get depth map + cam1 landmarks.

    Returns: (depth[H,W] or None, depth_ms, face_ms, landmarks[N,2] in cam1
    pixel coords or None, score)."""
    depth_in = cv2.resize(frame0, (DEPTH_W, DEPTH_H), interpolation=cv2.INTER_AREA)
    face_in, r, dx, dy = letterbox(frame1, FACE)

    worker.stdin.write(b'\x01'
                       + np.ascontiguousarray(depth_in).tobytes()
                       + np.ascontiguousarray(face_in).tobytes())
    worker.stdin.flush()

    hdr = read_exact(worker.stdout, 9)            # tag + depth_ms + face_ms
    if hdr is None:
        return None, 0.0, 0.0, None, 0.0
    depth_ms, face_ms = struct.unpack('<ff', hdr[1:9])

    data = read_exact(worker.stdout, DEPTH_MAP_BYTES)
    depth = (np.frombuffer(data, np.float32).reshape(DEPTH_H, DEPTH_W)
             if data is not None else None)

    nf = read_exact(worker.stdout, 4)
    if nf is None:
        return depth, depth_ms, face_ms, None, 0.0
    has_face = struct.unpack('<i', nf)[0]
    landmarks = None; score = 0.0
    if has_face:
        meta = read_exact(worker.stdout, 5 * 4)   # x1,y1,x2,y2,score
        lmb  = read_exact(worker.stdout, N_LMK * 2 * 4)
        if meta is not None and lmb is not None:
            score = struct.unpack('<5f', meta)[4]
            lm = np.frombuffer(lmb, np.float32).reshape(N_LMK, 2).copy()
            # un-letterbox: FACE-space -> cam1 pixel coords
            lm[:, 0] = (lm[:, 0] - dx) / r
            lm[:, 1] = (lm[:, 1] - dy) / r
            landmarks = lm
    return depth, depth_ms, face_ms, landmarks, score

# ── EasyOCR thread (persistent worker) ─────────────────────────────────────────
_ocr_text = 'Initializing EasyOCR ...'
_ocr_lock = threading.Lock()


def ocr_thread(ocr_ref):
    global _ocr_text
    time.sleep(5.0)
    while True:
        w = ocr_ref[0]
        with _frame_lock:
            f1 = None if _frame1 is None else _frame1.copy()
        if w is not None and w.poll() is None and f1 is not None:
            try:
                w.stdin.write(b'\x01' + np.ascontiguousarray(f1).tobytes())
                w.stdin.flush()
                hdr = read_exact(w.stdout, 5)
                if hdr is not None:
                    n = struct.unpack('<i', hdr[1:5])[0]
                    data = read_exact(w.stdout, n) if n > 0 else b''
                    txt = (data or b'').decode('utf-8', 'replace').strip()
                    with _ocr_lock:
                        _ocr_text = txt if txt else '[no text detected]'
            except Exception as e:
                print(f'[easyocr] error: {e}', flush=True)
        time.sleep(OCR_INTERVAL_S)

# ── Rendering ──────────────────────────────────────────────────────────────────
FONT        = cv2.FONT_HERSHEY_SIMPLEX
_DEPTH_CMAP = getattr(cv2, 'COLORMAP_TURBO', cv2.COLORMAP_PLASMA)


def colorize_depth(depth):
    lo, hi = np.percentile(depth, 2.0), np.percentile(depth, 98.0)
    if hi > lo:
        norm = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
        u8   = (norm * 255.0).astype(np.uint8)
    else:
        u8 = np.zeros((DEPTH_H, DEPTH_W), np.uint8)
    return cv2.applyColorMap(u8, _DEPTH_CMAP)


def _title(img, text, sub=None):
    cv2.rectangle(img, (0, 0), (QUAD_W, 52), (0, 0, 0), -1)
    cv2.putText(img, text, (16, 36), FONT, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    if sub:
        (tw, _), _ = cv2.getTextSize(sub, FONT, 0.55, 1)
        cv2.putText(img, sub, (QUAD_W - tw - 16, 34), FONT, 0.55,
                    (170, 170, 170), 1, cv2.LINE_AA)


def panel_rgb(frame, title, sub):
    p = cv2.resize(frame, (QUAD_W, QUAD_H), interpolation=cv2.INTER_AREA)
    _title(p, title, sub)
    return p


def panel_depth(depth, ms):
    if depth is None:
        p = np.full((QUAD_H, QUAD_W, 3), 20, np.uint8)
    else:
        p = cv2.resize(colorize_depth(depth), (QUAD_W, QUAD_H),
                       interpolation=cv2.INTER_LINEAR)
    _title(p, 'SCDepthV3 depth', f'DEEPX NPE-0  {ms:.0f}ms')
    return p


def panel_face(frame1, landmarks, score, ms):
    p = cv2.resize(frame1, (QUAD_W, QUAD_H), interpolation=cv2.INTER_AREA)
    _title(p, '3DDFA_V2 face mesh', f'DEEPX NPE-1  {ms:.0f}ms')
    if landmarks is not None:
        sx, sy = QUAD_W / CAM_W, QUAD_H / CAM_H
        pts = landmarks.copy()
        pts[:, 0] *= sx
        pts[:, 1] *= sy
        pts = pts.astype(np.int32)
        for grp in FACE_GROUPS:
            poly = pts[grp]
            cv2.polylines(p, [poly], False, (0, 230, 120), 1, cv2.LINE_AA)
        for (x, y) in pts:
            cv2.circle(p, (int(x), int(y)), 2, (90, 230, 255), -1, cv2.LINE_AA)
        cv2.putText(p, f'face {score:.0%}', (16, QUAD_H - 24), FONT, 0.7,
                    (90, 230, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(p, 'no face detected', (16, QUAD_H - 24), FONT, 0.7,
                    (180, 180, 180), 2, cv2.LINE_AA)
    return p


def panel_trocr(text):
    p = np.full((QUAD_H, QUAD_W, 3), 18, np.uint8)
    _title(p, 'EasyOCR text recognition', 'ONNX Runtime')
    # strip non-ASCII — cv2.putText cannot render Unicode
    text_safe = ''.join(c for c in text if 32 <= ord(c) < 127)
    # word-wrap: ~40 chars fits comfortably at scale 1.0 on 960px panel
    words, line, lines = text_safe.split(), '', []
    for w in words:
        test = (line + ' ' + w).strip()
        if len(test) > 40:
            if line:
                lines.append(line)
            line = w
        else:
            line = test
    if line:
        lines.append(line)
    if not lines:
        lines = ['[no text detected]']
    y = 160
    for ln in lines[:6]:
        cv2.putText(p, ln, (28, y), FONT, 1.0, (235, 235, 235), 2, cv2.LINE_AA)
        y += 58
    return p


def compose(tl, tr, bl, br):
    canvas = np.empty((DISP_H, DISP_W, 3), np.uint8)
    canvas[0:QUAD_H,      0:QUAD_W]      = tl
    canvas[0:QUAD_H,      QUAD_W:DISP_W] = tr
    canvas[QUAD_H:DISP_H, 0:QUAD_W]      = bl
    canvas[QUAD_H:DISP_H, QUAD_W:DISP_W] = br
    # divider lines
    cv2.line(canvas, (QUAD_W, 0), (QUAD_W, DISP_H), (60, 60, 60), 2)
    cv2.line(canvas, (0, QUAD_H), (DISP_W, QUAD_H), (60, 60, 60), 2)
    return canvas

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print('[deepx-dual] Starting DeepX Dual demo (clean rebuild)', flush=True)

    deepx_w = spawn_deepx_worker()
    print('[deepx-dual] loading DeepX models (depth + cls) ...', flush=True)
    if not wait_ready(deepx_w, 'deepx', timeout=180):
        print('[deepx-dual] DeepX worker failed to become ready', flush=True)

    ocr_ref = [spawn_ocr_worker()]
    print('[deepx-dual] loading EasyOCR ...', flush=True)
    if not wait_ready(ocr_ref[0], 'easyocr', timeout=180):
        print('[deepx-dual] EasyOCR worker failed — text panel will be empty', flush=True)

    cam_ref = [spawn_cameras()]
    time.sleep(2.0)
    threading.Thread(target=cam_thread, args=(cam_ref, spawn_cameras), daemon=True).start()
    threading.Thread(target=ocr_thread, args=(ocr_ref,), daemon=True).start()

    disp = spawn_display()
    time.sleep(0.5)

    def cleanup(*_):
        print('[deepx-dual] shutting down ...', flush=True)
        for p in [deepx_w, ocr_ref[0], cam_ref[0], disp]:
            try: p.terminate()
            except Exception: pass
        time.sleep(2.0)
        for p in [deepx_w, ocr_ref[0], cam_ref[0], disp]:
            try:
                if p.poll() is None: p.kill()
            except Exception: pass
        sys.exit(0)

    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)

    print('[deepx-dual] Running. Ctrl-C to stop.', flush=True)

    last_depth = None
    last_lmk   = None
    last_score = 0.0
    depth_ms = face_ms = 0.0
    frame_idx = 0
    time.sleep(2.0)

    while True:
        t0 = time.time()
        with _frame_lock:
            f0 = None if _frame0 is None else _frame0
            f1 = None if _frame1 is None else _frame1

        if f0 is None or f1 is None:
            time.sleep(0.05)
            continue

        # depth + face-detect run IN PARALLEL on the DX-M1; 3DDFA on the crop.
        if deepx_w.poll() is None:
            try:
                d, depth_ms, face_ms, lmk, score = req_infer(deepx_w, f0, f1)
                if d is not None:
                    last_depth = d
                last_lmk, last_score = lmk, score
            except (BrokenPipeError, OSError):
                pass

        with _ocr_lock:
            ocr = _ocr_text

        tl = panel_rgb(f0, 'cam0  RGB', 'reference')
        tr = panel_face(f1, last_lmk, last_score, face_ms)
        bl = panel_depth(last_depth, depth_ms)
        br = panel_trocr(ocr)
        canvas = compose(tl, tr, bl, br)

        try:
            disp.stdin.write(canvas.tobytes())
            disp.stdin.flush()
        except BrokenPipeError:
            print('[deepx-dual] display pipe broken — restarting', flush=True)
            try: disp.terminate(); disp.wait(timeout=2)
            except Exception: disp.kill()
            time.sleep(0.5)
            disp = spawn_display()
            time.sleep(0.5)

        elapsed = time.time() - t0
        time.sleep(max(0.0, 1.0/CAM_FPS - elapsed))
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f'[deepx-dual] depth={depth_ms:.1f}ms face={face_ms:.1f}ms '
                  f'fps≈{1.0/(time.time()-t0+1e-9):.1f}', flush=True)


if __name__ == '__main__':
    main()
