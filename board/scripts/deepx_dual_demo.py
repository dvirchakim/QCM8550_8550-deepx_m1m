#!/usr/bin/env python3
"""
DeepX Dual Demo  —  IMDT QCS8550 + DEEPX DX-M1
================================================
Left  960x1080 : SCDepthV3 depth heatmap           [cam0, DEEPX NPE-0]
Right 960x1080 : YOLO26m-cls top-5 classification  [cam1, DEEPX NPE-1]
                 + TrOCR text recognition           [cam1, Qualcomm HTP/CPU]

Deploy:
    adb push deepx_dual_demo.py  /data/local/tmp/
    adb push depth_worker.py     /data/local/tmp/
    adb push cls_worker.py       /data/local/tmp/
    adb push trocr_worker.py     /data/local/tmp/
    adb push scdepthv3.dxnn      /data/local/tmp/
    adb push yolo26m-cls.dxnn    /data/local/tmp/
    (push trocr/ dir separately)
"""
from __future__ import annotations
import os, shlex, signal, struct, subprocess, sys, threading, time
import cv2
import numpy as np

# ── Config ─────────────────────────────────────────────────────────────────────
CAM_W, CAM_H, CAM_FPS = 1280, 720, 15
PANE_W, PANE_H        = 960, 1080
DISP_W, DISP_H        = 1920, 1080

FRAME0_FILE     = '/tmp/dd_frame0.bin'
FRAME1_FILE     = '/tmp/dd_frame1.bin'
DEPTH_PANE_FILE = '/tmp/dd_depth_pane.bin'
CLS_PANE_FILE   = '/tmp/dd_cls_pane.bin'
OCR_TEXT_FILE   = '/tmp/dd_ocr.txt'

DUAL_DEEPX_WORKER = '/data/local/tmp/dual_deepx_worker.py'
TROCR_WORKER      = '/data/local/tmp/trocr_worker.py'

ADSP_PATH = '/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp'
TROCR_INTERVAL_S = 8.0    # seconds between TrOCR inference runs

CAM_BYTES = CAM_W * CAM_H * 3
PANE_BYTES = PANE_W * PANE_H * 3

# ── Environment ────────────────────────────────────────────────────────────────
def _env():
    e = os.environ.copy()
    e['XDG_RUNTIME_DIR'] = '/run/user/root'
    e['WAYLAND_DISPLAY']  = 'wayland-1'
    e['QT_QPA_PLATFORM']  = 'wayland-egl'
    e['QT_WAYLAND_SHELL_INTEGRATION'] = 'wl-shell'
    e['ADSP_LIBRARY_PATH'] = ADSP_PATH
    return e

# ── GStreamer helpers ──────────────────────────────────────────────────────────
def spawn_camera(cam_idx):
    pipe = (f'gst-launch-1.0 -q qtiqmmfsrc camera={cam_idx} ! qtivtransform ! '
            f'video/x-raw,width={CAM_W},height={CAM_H},format=NV12,framerate={CAM_FPS}/1 ! '
            f'videoconvert ! video/x-raw,format=BGR ! fdsink fd=1 sync=false')
    return subprocess.Popen(shlex.split(pipe), stdout=subprocess.PIPE,
                            stderr=open(f'/tmp/dd_cam{cam_idx}.log', 'w'),
                            env=_env(), bufsize=0)


def spawn_display():
    pipe = (f'gst-launch-1.0 -q fdsrc fd=0 ! '
            f'rawvideoparse format=bgr width={DISP_W} height={DISP_H} framerate={CAM_FPS}/1 ! '
            f'videoconvert ! waylandsink sync=false fullscreen=true')
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

# ── Worker process management ──────────────────────────────────────────────────
def spawn_worker(script):
    return subprocess.Popen(
        ['python3', script],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=sys.stderr, bufsize=0)


def wait_ready(proc, tag, timeout=60):
    deadline = time.time() + timeout
    buf = b''
    while time.time() < deadline:
        ch = proc.stdout.read(1)
        if not ch:
            return False
        buf += ch
        if buf.endswith(b'READY\n'):
            print(f'[{tag}] worker ready.')
            return True
    return False


def trigger_worker(proc):
    try:
        proc.stdin.write(b'\x01')
        proc.stdin.flush()
    except OSError:
        pass


def read_worker_response(proc):
    try:
        hdr = proc.stdout.read(5)   # b'\x01' + float32
        if len(hdr) == 5:
            return struct.unpack('<xf', hdr)[0]   # inf_ms
    except OSError:
        pass
    return 0.0


def ensure_worker(proc, script, tag):
    if proc is None or proc.poll() is not None:
        print(f'[{tag}] (re)starting worker ...')
        try:
            if proc:
                proc.kill()
        except OSError:
            pass
        proc = spawn_worker(script)
        if not wait_ready(proc, tag, timeout=90):
            print(f'[{tag}] worker failed to become ready')
        return proc
    return proc

# ── Camera capture threads ─────────────────────────────────────────────────────
_frame0 = None
_frame1 = None
_frame_lock = threading.Lock()


def cam_thread(cam_proc_ref, frame_file, cam_idx, lock, restart_fn):
    while True:
        proc = cam_proc_ref[0]
        raw = read_exact(proc.stdout, CAM_BYTES)
        if raw is None:
            print(f'[cam{cam_idx}] pipe closed — restarting ...')
            try: proc.terminate(); proc.wait(timeout=2)
            except Exception: proc.kill()
            time.sleep(1.5)
            cam_proc_ref[0] = restart_fn()
            time.sleep(1.5)
            continue
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(CAM_H, CAM_W, 3)
        arr.tofile(frame_file)

# ── TrOCR background thread ────────────────────────────────────────────────────
_ocr_text = 'Initializing TrOCR ...'
_ocr_lock = threading.Lock()


def trocr_thread():
    global _ocr_text
    time.sleep(5.0)   # let cameras settle first
    while True:
        try:
            r = subprocess.run(
                ['python3', TROCR_WORKER, FRAME1_FILE, OCR_TEXT_FILE],
                capture_output=False, timeout=120)
            if r.returncode == 0:
                try:
                    with open(OCR_TEXT_FILE) as f:
                        txt = f.read().strip()
                    with _ocr_lock:
                        _ocr_text = txt if txt else '[no text detected]'
                except Exception:
                    pass
        except subprocess.TimeoutExpired:
            print('[trocr] inference timed out (>120s)')
        except Exception as e:
            print(f'[trocr] error: {e}')
        time.sleep(TROCR_INTERVAL_S)

# ── Placeholder pane ───────────────────────────────────────────────────────────
def make_placeholder(label):
    p = np.full((PANE_H, PANE_W, 3), 20, np.uint8)
    cv2.putText(p, label, (30, PANE_H // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 100, 100), 2, cv2.LINE_AA)
    return p


def read_pane(path, placeholder_label):
    try:
        if os.path.exists(path) and os.path.getsize(path) == PANE_BYTES:
            return np.fromfile(path, dtype=np.uint8).reshape(PANE_H, PANE_W, 3)
    except Exception:
        pass
    return make_placeholder(placeholder_label)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print('[deepx-dual] Starting DeepX Dual + TrOCR demo')

    # Single combined DeepX worker (avoids concurrent DXRT device access)
    deepx_w = spawn_worker(DUAL_DEEPX_WORKER)

    print('[deepx-dual] Waiting for DEEPX worker (loading both models) ...')
    if not wait_ready(deepx_w, 'deepx', timeout=180):
        print('[deepx-dual] dual_deepx_worker failed to start')

    # Start cameras
    cam0_ref = [spawn_camera(0)]
    cam1_ref = [spawn_camera(1)]
    time.sleep(1.5)

    # Camera capture threads (write to frame files)
    threading.Thread(
        target=cam_thread,
        args=(cam0_ref, FRAME0_FILE, 0, _frame_lock, lambda: spawn_camera(0)),
        daemon=True).start()
    threading.Thread(
        target=cam_thread,
        args=(cam1_ref, FRAME1_FILE, 1, _frame_lock, lambda: spawn_camera(1)),
        daemon=True).start()

    # TrOCR background thread
    threading.Thread(target=trocr_thread, daemon=True).start()

    # Display
    disp = spawn_display()
    time.sleep(0.5)

    def cleanup(*_):
        print('[deepx-dual] shutting down ...')
        for p in [deepx_w, cam0_ref[0], cam1_ref[0], disp]:
            try: p.terminate()
            except Exception: pass
        time.sleep(0.5)
        for p in [deepx_w, cam0_ref[0], cam1_ref[0], disp]:
            try: p.kill()
            except Exception: pass
        sys.exit(0)

    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)

    print('[deepx-dual] Running. Ctrl-C to stop.')

    depth_ms = cls_ms = 0.0
    frame_idx = 0

    # Wait for first frames to arrive
    time.sleep(2.0)

    while True:
        t0 = time.time()

        # Alternate: even frames = depth, odd frames = cls (DXRT is single-access)
        if frame_idx % 2 == 0:
            deepx_w.stdin.write(b'\x01')   # depth
            deepx_w.stdin.flush()
            depth_ms = read_worker_response(deepx_w) or depth_ms
        else:
            deepx_w.stdin.write(b'\x02')   # cls
            deepx_w.stdin.flush()
            cls_ms = read_worker_response(deepx_w) or cls_ms

        deepx_w = ensure_worker(deepx_w, DUAL_DEEPX_WORKER, 'deepx')

        # Read rendered panes
        depth_pane = read_pane(DEPTH_PANE_FILE, 'depth loading...')
        cls_pane   = read_pane(CLS_PANE_FILE,   'cls loading...')

        # Overlay TrOCR text on cls_pane
        with _ocr_lock:
            ocr = _ocr_text
        if ocr:
            # Dark band at bottom for text
            y0 = PANE_H - 160
            cv2.rectangle(cls_pane, (0, y0), (PANE_W, PANE_H), (0, 0, 0), -1)
            cv2.putText(cls_pane, 'TrOCR  (Qualcomm HTP):', (16, y0 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 220, 255), 1, cv2.LINE_AA)
            # Word-wrap
            words = ocr.split()
            line, lines = '', []
            for w in words:
                test = (line + ' ' + w).strip()
                if len(test) > 38:
                    if line: lines.append(line)
                    line = w
                else:
                    line = test
            if line:
                lines.append(line)
            for i, ln in enumerate(lines[:3]):
                cv2.putText(cls_pane, ln, (16, y0 + 66 + i * 34),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 1, cv2.LINE_AA)

        # Compose 1920×1080
        canvas = np.concatenate([depth_pane, cls_pane], axis=1)

        # Push to display
        try:
            disp.stdin.write(canvas.tobytes())
            disp.stdin.flush()
        except BrokenPipeError:
            print('[deepx-dual] display pipe broken — restarting')
            try: disp.terminate(); disp.wait(timeout=2)
            except Exception: disp.kill()
            time.sleep(0.5)
            disp = spawn_display()
            time.sleep(0.5)

        # Target frame rate
        elapsed = time.time() - t0
        sleep_t = max(0.0, 1.0/CAM_FPS - elapsed)
        time.sleep(sleep_t)
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f'[deepx-dual] depth={depth_ms:.1f}ms cls={cls_ms:.1f}ms  '
                  f'fps≈{1.0/(elapsed+sleep_t+1e-9):.1f}', flush=True)


if __name__ == '__main__':
    main()
