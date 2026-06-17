#!/usr/bin/env python3
"""
demo_picker  --  Touchscreen demo selector for QCS8550 + DeepX DX-M1

Three full-width demo cards (what actually runs on the board):
  01  Pose Estimation        DeepX YOLOv5-Pose  +  HTP YOLOv8
  02  Instance Segmentation  DeepX YOLO26-SEG   +  HTP YOLOv8
  03  Depth / Face / OCR     DeepX Depth+Face   +  HTP EasyOCR

Display: GStreamer appsrc -> waylandsink (1920x1080 fullscreen)
Touch:   /dev/input/event* ABS_X/Y + BTN_TOUCH
Launch:  systemctl start/stop <service>  (one demo at a time)
"""
import os, sys, signal, struct, subprocess, threading, time, ctypes
import numpy as np
import cv2

_libc = ctypes.CDLL("libc.so.6", use_errno=True)

W, H = 1920, 1080
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Accent colours are BGR. HTP chips share one amber accent; each demo owns a hue.
HTP_COLOR = (60, 170, 255)   # amber

TILES = [
    {"label": "Pose Estimation",
     "sub":   "Real-time human pose skeletons on two cameras",
     "svc":   "imdt-deepx-demo.service",
     "accel": ("DeepX  YOLOv5-Pose", "HTP  YOLOv8"),
     "color": (210, 140,  40)},   # blue
    {"label": "Instance Segmentation",
     "sub":   "Object masks on DeepX, detection on the Hexagon HTP",
     "svc":   "yolo26-parallel.service",
     "accel": ("DeepX  YOLO26-SEG", "HTP  YOLOv8"),
     "color": ( 70, 175,  60)},   # green
    {"label": "Depth / Face / OCR",
     "sub":   "Depth map + 3D face mesh on DeepX, live OCR on the HTP",
     "svc":   "deepx-dual.service",
     "accel": ("DeepX  Depth+Face", "HTP  EasyOCR"),
     "color": ( 90, 185, 220)},   # teal/amber
]

# ── Layout ──────────────────────────────────────────────────────────────────
HEADER_H = 110
FOOTER_H = 70
MARGIN_X = 60
GAP      = 26
AREA_TOP = HEADER_H + 24
AREA_BOT = H - FOOTER_H - 14
CARD_H   = (AREA_BOT - AREA_TOP - (len(TILES) - 1) * GAP) // len(TILES)

_running_svc = ""
_disp_close_evt  = threading.Event()   # signals main loop to close display
_disp_reopen_evt = threading.Event()   # signals main loop to reopen display


def _svc(cmd, svc):
    try:
        subprocess.run(["systemctl", cmd, svc],
                       capture_output=True, timeout=15)
    except subprocess.TimeoutExpired:
        if cmd == "stop":
            subprocess.run(["pkill", "-9", "-f", svc.replace(".service", "")],
                           capture_output=True)


def _stop_all():
    for t in TILES:
        _svc("stop", t["svc"])


def launch(idx):
    global _running_svc
    _stop_all()
    _running_svc = TILES[idx]["svc"]
    _disp_close_evt.set()
    time.sleep(0.3)
    _svc("start", _running_svc)


def stop_demo():
    global _running_svc
    _stop_all()
    _running_svc = ""
    _disp_reopen_evt.set()   # tell main loop to reopen display


# ── Rendering ─────────────────────────────────────────────────────────────────

def _rrect(img, p0, p1, color, r=22, thickness=-1):
    """Filled or outlined rounded rectangle."""
    x0, y0 = p0; x1, y1 = p1
    r = max(1, min(r, (x1 - x0) // 2, (y1 - y0) // 2))
    if thickness < 0:
        cv2.rectangle(img, (x0 + r, y0), (x1 - r, y1), color, -1)
        cv2.rectangle(img, (x0, y0 + r), (x1, y1 - r), color, -1)
        for cx, cy in ((x0+r, y0+r), (x1-r, y0+r), (x0+r, y1-r), (x1-r, y1-r)):
            cv2.circle(img, (cx, cy), r, color, -1)
    else:
        t = thickness
        cv2.line(img, (x0+r, y0), (x1-r, y0), color, t, cv2.LINE_AA)
        cv2.line(img, (x0+r, y1), (x1-r, y1), color, t, cv2.LINE_AA)
        cv2.line(img, (x0, y0+r), (x0, y1-r), color, t, cv2.LINE_AA)
        cv2.line(img, (x1, y0+r), (x1, y1-r), color, t, cv2.LINE_AA)
        cv2.ellipse(img, (x0+r, y0+r), (r, r), 180, 0, 90, color, t, cv2.LINE_AA)
        cv2.ellipse(img, (x1-r, y0+r), (r, r), 270, 0, 90, color, t, cv2.LINE_AA)
        cv2.ellipse(img, (x0+r, y1-r), (r, r),  90, 0, 90, color, t, cv2.LINE_AA)
        cv2.ellipse(img, (x1-r, y1-r), (r, r),   0, 0, 90, color, t, cv2.LINE_AA)


def _chip(img, x_right, y_center, text, color):
    """Right-aligned pill badge. Returns its left x (for stacking)."""
    fs, th = 0.6, 1
    (tw, tht), _ = cv2.getTextSize(text, FONT, fs, th)
    padx, pady = 18, 12
    w, h = tw + 2*padx, tht + 2*pady
    x1, x0 = x_right, x_right - w
    y0, y1 = y_center - h//2, y_center - h//2 + h
    _rrect(img, (x0, y0), (x1, y1), (44, 40, 48), r=h//2, thickness=-1)
    _rrect(img, (x0, y0), (x1, y1), color, r=h//2, thickness=2)
    cv2.putText(img, text, (x0+padx, y1-pady-1), FONT, fs, color, th, cv2.LINE_AA)
    return x0


def render(highlighted: int) -> np.ndarray:
    frame = np.full((H, W, 3), 16, dtype=np.uint8)

    # Header
    cv2.rectangle(frame, (0, 0), (W, HEADER_H), (26, 22, 21), -1)
    cv2.line(frame, (0, HEADER_H), (W, HEADER_H), (62, 54, 52), 1)
    cv2.putText(frame, "AI Demo Station", (MARGIN_X, 70),
                FONT, 1.5, (242, 242, 248), 2, cv2.LINE_AA)
    sub = "Qualcomm QCS8550    .    DeepX DX-M1 NPU    .    Hexagon HTP"
    (sw, _), _ = cv2.getTextSize(sub, FONT, 0.7, 1)
    cv2.putText(frame, sub, (W - MARGIN_X - sw, 66),
                FONT, 0.7, (150, 150, 165), 1, cv2.LINE_AA)

    for i, t in enumerate(TILES):
        x0, x1 = MARGIN_X, W - MARGIN_X
        y0 = AREA_TOP + i * (CARD_H + GAP)
        y1 = y0 + CARD_H
        active = (_running_svc == t["svc"])
        hi     = (highlighted == i)

        if hi:
            base = tuple(min(255, int(c*0.45) + 32) for c in t["color"])
        elif active:
            base = tuple(min(255, int(c*0.28) + 24) for c in t["color"])
        else:
            base = (40, 34, 33)
        _rrect(frame, (x0, y0), (x1, y1), base, r=22, thickness=-1)
        _rrect(frame, (x0, y0), (x1, y1),
               (255, 255, 255) if active else t["color"], r=22,
               thickness=3 if active else 2)

        # Left accent bar
        cv2.rectangle(frame, (x0+7, y0+22), (x0+17, y1-22), t["color"], -1)

        cy = y0 + CARD_H // 2
        cv2.putText(frame, "%02d" % (i+1), (x0+48, cy+26),
                    FONT, 2.4, t["color"], 3, cv2.LINE_AA)

        tx = x0 + 210
        cv2.putText(frame, t["label"], (tx, cy-4),
                    FONT, 1.35, (245, 245, 250), 2, cv2.LINE_AA)
        cv2.putText(frame, t["sub"], (tx, cy+44),
                    FONT, 0.7, (168, 168, 182), 1, cv2.LINE_AA)
        if active:
            cv2.putText(frame, "RUNNING", (tx, y0+42),
                        FONT, 0.62, (120, 232, 140), 1, cv2.LINE_AA)

        # Accelerator chips (right-aligned: DeepX hue + HTP amber)
        left = _chip(frame, x1 - 30, cy, t["accel"][1], HTP_COLOR)
        _chip(frame, left - 16, cy, t["accel"][0], t["color"])

    # Footer
    cv2.rectangle(frame, (0, H-FOOTER_H), (W, H), (26, 22, 21), -1)
    cv2.line(frame, (0, H-FOOTER_H), (W, H-FOOTER_H), (62, 54, 52), 1)
    cv2.putText(frame,
                "Tap a card to launch a demo        HOME (top-left corner) to stop and return",
                (MARGIN_X, H-26), FONT, 0.7, (150, 150, 165), 1, cv2.LINE_AA)
    return frame


# ── GStreamer display ─────────────────────────────────────────────────────────

def open_display():
    import shlex
    cmd = (
        "gst-launch-1.0 -q fdsrc "
        "! rawvideoparse width=1920 height=1080 format=bgr framerate=30/1 "
        "! videoconvert "
        "! video/x-raw,format=BGRx "
        "! waylandsink fullscreen=true sync=false"
    )
    return subprocess.Popen(shlex.split(cmd), stdin=subprocess.PIPE)


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
    for i in range(10):  # scan event0 through event9
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
        except OSError:
            time.sleep(0.1)
            continue
        if len(n) < sz:
            continue
        _, _, evtype, code, value = struct.unpack(fmt, n)
        if evtype == EV_ABS:
            if code in (ABS_X, ABS_MT_X):   tx = value
            if code in (ABS_Y, ABS_MT_Y):   ty = value
        if evtype == EV_KEY and code == BTN_TOUCH and value == 1:
            print("[touch] BTN_TOUCH tx=%d ty=%d" % (tx, ty), flush=True)
            _tap_xy[0] = tx
            _tap_xy[1] = ty
            _tap_evt.set()


TOUCH_MAX_X = 1023
TOUCH_MAX_Y = 599

def tap_to_tile(tx, ty):
    # HOME button physically sits at top-left corner of touch panel (tx<80, ty<80)
    if tx < 80 and ty < 80:
        return -1
    y = ty * H // TOUCH_MAX_Y
    if y < AREA_TOP - 10:
        return -2          # header tap — ignore
    idx = (y - AREA_TOP) // (CARD_H + GAP)
    if 0 <= idx < len(TILES):
        return int(idx)
    return -2              # gap / footer — ignore


# ── Main ──────────────────────────────────────────────────────────────────────

def _detect_running_svc():
    """Return the svc name of any demo that is currently active, or ''."""
    for t in TILES:
        try:
            r = subprocess.run(
                ["systemctl", "is-active", t["svc"]],
                capture_output=True, timeout=3)
            if r.returncode == 0:
                return t["svc"]
        except Exception:
            pass
    return ""


def main():
    global _running_svc
    _running_svc = _detect_running_svc()

    # Don't open our display if a demo is already on screen
    disp = None if _running_svc else open_display()
    if disp:
        time.sleep(0.5)

    t = threading.Thread(target=touch_thread, daemon=True)
    t.start()

    highlighted  = -1
    hi_until     = 0.0
    last_launch  = 0.0
    DEBOUNCE     = 2.0

    def _close_disp(d):
        if d is None:
            return
        try: d.terminate(); d.wait(timeout=2)
        except Exception:
            try: d.kill()
            except Exception: pass

    def cleanup(*_):
        _close_disp(disp)
        sys.exit(0)
    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT,  cleanup)

    print("[demo_picker] running", flush=True)

    while True:
        # Handle display close/reopen requests from launch()/stop_demo()
        if _disp_close_evt.is_set():
            _disp_close_evt.clear()
            _close_disp(disp)
            disp = None
        if _disp_reopen_evt.is_set():
            _disp_reopen_evt.clear()
            _close_disp(disp)
            time.sleep(0.5)
            disp = open_display()
            time.sleep(0.5)

        if _tap_evt.is_set():
            _tap_evt.clear()
            now = time.time()
            if now - last_launch > DEBOUNCE:
                tx, ty = _tap_xy
                tile = tap_to_tile(tx, ty)
                if 0 <= tile < len(TILES):
                    last_launch = now
                    highlighted = tile
                    hi_until    = now + 0.6
                    print("[demo_picker] launching tile %d: %s" % (tile, TILES[tile]["svc"]), flush=True)
                    threading.Thread(target=launch, args=(tile,), daemon=True).start()
                elif tile == -1:
                    # HOME button: stop active demo and show picker again
                    print("[demo_picker] HOME → stopping demo", flush=True)
                    last_launch = now
                    threading.Thread(target=stop_demo, daemon=True).start()

        if highlighted >= 0 and time.time() > hi_until:
            highlighted = -1

        if disp is None:
            time.sleep(1.0 / 30.0)
            continue

        frame = render(highlighted)
        try:
            disp.stdin.write(frame.tobytes())
            disp.stdin.flush()
        except BrokenPipeError:
            print("[demo_picker] display pipe broken, restarting", flush=True)
            _close_disp(disp)
            time.sleep(0.5)
            disp = open_display()
            time.sleep(0.5)

        time.sleep(1.0 / 30.0)


if __name__ == "__main__":
    main()
