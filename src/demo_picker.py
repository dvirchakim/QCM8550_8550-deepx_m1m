#!/usr/bin/env python3
"""
demo_picker  --  Touchscreen demo selector for QCS8550 + DeepX DX-M1

Tiles (2x2):
  [Edge Art]        [OEM Reference]
  [YOLO26 Parallel] [          ]

Display: GStreamer appsrc -> waylandsink (1920x1080 fullscreen)
Touch:   /dev/input/event* ABS_X/Y + BTN_TOUCH
Launch:  systemctl start/stop <service>
"""
import os, sys, signal, struct, subprocess, threading, time, ctypes
import numpy as np
import cv2

_libc = ctypes.CDLL("libc.so.6", use_errno=True)

W, H = 1920, 1080

TILES = [
    {"label": "Edge Art",
     "sub":   "DeepX Pose  +  Qualcomm GenAI",
     "svc":   "edge-art.service",
     "color": (210,  60, 180)},   # purple  (BGR)
    {"label": "OEM Reference",
     "sub":   "Qualcomm  +  DeepX  Pipeline",
     "svc":   "imdt-deepx-demo.service",
     "color": (180, 120,   0)},   # blue
    {"label": "YOLO26 Parallel",
     "sub":   "DeepX Det  ||  DeepX Seg",
     "svc":   "yolo26-parallel.service",
     "color": ( 30, 160,  30)},   # green
]

_running_svc = ""


def _svc(cmd, svc):
    subprocess.run(["systemctl", cmd, svc],
                   capture_output=True, timeout=10)


def launch(idx):
    global _running_svc
    if _running_svc:
        _svc("stop", _running_svc)
    _running_svc = TILES[idx]["svc"]
    _svc("start", _running_svc)


# ── Rendering ─────────────────────────────────────────────────────────────────

def render(highlighted: int) -> np.ndarray:
    frame = np.full((H, W, 3), 20, dtype=np.uint8)

    TW, TH = W // 2, H // 2
    PAD = 20

    for i, t in enumerate(TILES):
        col = i % 2
        row = i // 2
        x0, y0 = col * TW + PAD, row * TH + PAD
        x1, y1 = x0 + TW - 2*PAD, y0 + TH - 2*PAD

        active = (_running_svc == t["svc"])
        hi     = (highlighted == i)
        scale  = 1.35 if hi else (0.85 if active else 1.0)
        bgr    = tuple(min(255, int(c * scale)) for c in t["color"])

        cv2.rectangle(frame, (x0, y0), (x1, y1), bgr, -1)
        bdr = (255, 255, 255) if active else t["color"]
        thick = 5 if active else 3
        cv2.rectangle(frame, (x0, y0), (x1, y1), bdr, thick)

        # Label
        fs = 1.6
        (tw, th), _ = cv2.getTextSize(t["label"], cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
        tx = x0 + (TW - 2*PAD - tw) // 2
        ty = y0 + (TH - 2*PAD) // 2 - 10
        cv2.putText(frame, t["label"], (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, fs,
                    (255, 255, 255), 2, cv2.LINE_AA)

        # Subtitle
        fs2 = 0.58
        (sw, _), _ = cv2.getTextSize(t["sub"], cv2.FONT_HERSHEY_SIMPLEX, fs2, 1)
        sx = x0 + (TW - 2*PAD - sw) // 2
        sy = ty + 36
        cv2.putText(frame, t["sub"], (sx, sy),
                    cv2.FONT_HERSHEY_SIMPLEX, fs2,
                    t["color"], 1, cv2.LINE_AA)

    # Header
    cv2.rectangle(frame, (0, 0), (W, 44), (12, 12, 12), -1)
    cv2.putText(frame,
                "Touch a panel to launch the chosen demo.",
                (22, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (160, 160, 160), 1, cv2.LINE_AA)

    return frame


# ── GStreamer display ─────────────────────────────────────────────────────────

def open_display():
    cmd = (
        "gst-launch-1.0 -q fdsrc "
        "! rawvideoparse width=1920 height=1080 format=bgr framerate=30/1 "
        "! videoconvert "
        "! video/x-raw,format=BGRx "
        "! waylandsink fullscreen=true sync=false"
    )
    return subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE)


# ── Touch input ───────────────────────────────────────────────────────────────

_tap_evt = threading.Event()
_tap_xy  = [0, 0]

EV_KEY  = 0x01
EV_ABS  = 0x03
BTN_TOUCH = 0x14A
ABS_X   = 0x00
ABS_Y   = 0x01
ABS_MT_X = 0x35
ABS_MT_Y = 0x36
EVIOCGBIT_EV = 0x80084500   # EVIOCGBIT(0, 8) on aarch64

def _find_touch():
    for i in range(3):   # only event0/1/2 on this board
        p = "/dev/input/event%d" % i
        try:
            fd = os.open(p, os.O_RDONLY | os.O_NONBLOCK)
            buf = (ctypes.c_uint8 * 8)()
            ret = _libc.ioctl(fd, EVIOCGBIT_EV, buf)
            if ret >= 0:
                bits = int.from_bytes(bytes(buf), "little")
                if bits & (1 << EV_ABS):
                    print("[touch] found touch at event%d" % i, flush=True)
                    return fd
            os.close(fd)
        except Exception:
            pass
    # Fallback: return event0
    try:
        return os.open("/dev/input/event0", os.O_RDONLY | os.O_NONBLOCK)
    except Exception:
        return -1


def touch_thread():
    fd = _find_touch()
    if fd < 0:
        print("[touch] no device found", flush=True)
        return
    tx, ty = 0, 0
    fmt = "llHHi"
    sz  = struct.calcsize(fmt)
    buf = bytearray(sz)
    while True:
        try:
            n = os.read(fd, sz)
        except BlockingIOError:
            time.sleep(0.005)
            continue
        if len(n) < sz:
            continue
        _, _, evtype, code, value = struct.unpack(fmt, n)
        if evtype == EV_ABS:
            if code in (ABS_X, ABS_MT_X):   tx = value
            if code in (ABS_Y, ABS_MT_Y):   ty = value
        if evtype == EV_KEY and code == BTN_TOUCH and value == 1:
            _tap_xy[0] = tx
            _tap_xy[1] = ty
            _tap_evt.set()


TOUCH_MAX_X = 1023
TOUCH_MAX_Y = 599

def tap_to_tile(tx, ty):
    # Map raw touch coords (0-1023, 0-599) to screen tile index
    x = tx * W // TOUCH_MAX_X
    y = ty * H // TOUCH_MAX_Y
    col = min(x // (W // 2), 1)
    row = min(y // (H // 2), 1)
    return row * 2 + col


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    disp = open_display()
    time.sleep(0.5)

    t = threading.Thread(target=touch_thread, daemon=True)
    t.start()

    highlighted = -1
    hi_until    = 0.0
    frame_bytes = W * H * 3

    def cleanup(*_):
        disp.terminate()
        sys.exit(0)
    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT,  cleanup)

    print("[demo_picker] running", flush=True)

    while True:
        if _tap_evt.is_set():
            _tap_evt.clear()
            tx, ty = _tap_xy
            tile = tap_to_tile(tx, ty)
            if 0 <= tile < len(TILES):
                highlighted = tile
                hi_until    = time.time() + 0.6
                print("[demo_picker] launching tile %d: %s" % (tile, TILES[tile]["svc"]), flush=True)
                threading.Thread(target=launch, args=(tile,), daemon=True).start()

        if highlighted >= 0 and time.time() > hi_until:
            highlighted = -1

        frame = render(highlighted)
        try:
            disp.stdin.write(frame.tobytes())
            disp.stdin.flush()
        except BrokenPipeError:
            print("[demo_picker] display pipe broken, restarting", flush=True)
            disp = open_display()
            time.sleep(0.5)

        time.sleep(1.0 / 30.0)


if __name__ == "__main__":
    main()
