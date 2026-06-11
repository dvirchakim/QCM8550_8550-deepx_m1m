#!/usr/bin/env python3
"""
Persistent DEEPX DX-M1 depth estimation worker — SCDepthV3 via dx_engine.

Protocol (binary):
  main → worker : 1 byte  (0x01 = run, 0x00 = quit)
  worker → main : 1 byte 0x01 (done; depth pane written to DEPTH_PANE_FILE)
                  float  inf_ms
"""
import os, signal, struct, sys, time
import numpy as np
import cv2

# Suppress DXRT noise on fd-1
_pipe_wfd = os.dup(1)
_devnull  = os.open('/dev/null', os.O_WRONLY)
os.dup2(_devnull, 1)
os.close(_devnull)
_pipe_out = os.fdopen(_pipe_wfd, 'wb', buffering=0)

signal.signal(signal.SIGTERM, lambda *_: os._exit(0))

from dx_engine import InferenceEngine, InferenceOption

MODEL          = '/data/local/tmp/scdepthv3.dxnn'
FRAME_FILE     = '/tmp/dd_frame0.bin'
DEPTH_PANE_FILE = '/tmp/dd_depth_pane.bin'
CAM_W, CAM_H   = 1280, 720
INF_W, INF_H   = 320, 256    # SCDepthV3 input size
PANE_W, PANE_H = 960, 1080

# ── init ──────────────────────────────────────────────────────────────────────
opt = InferenceOption()
try:
    opt.set_buffer_count(1)
except Exception:
    pass

engine = None
try:
    engine = InferenceEngine(MODEL, opt)
except Exception as e:
    sys.stderr.write(f"[depth_worker] DEEPX init failed: {e}\n")
    sys.stderr.flush()

_pipe_out.write(b'READY\n')

# ── pre-allocate ───────────────────────────────────────────────────────────────
_inp_buf  = np.empty((INF_H, INF_W, 3), np.uint8)
_pane_buf = np.empty((PANE_H, PANE_W, 3), np.uint8)

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
        np.zeros((PANE_H, PANE_W, 3), np.uint8).tofile(DEPTH_PANE_FILE)
        _pipe_out.write(b'\x01' + struct.pack('<f', 0.0))
        continue

    try:
        cv2.resize(raw, (INF_W, INF_H), dst=_inp_buf, interpolation=cv2.INTER_AREA)

        t0 = time.time()
        outs = engine.run(np.expand_dims(_inp_buf, 0))
        inf_ms = (time.time() - t0) * 1000.0

        depth = outs[0].reshape(INF_H, INF_W).astype(np.float32)

        # Normalize to [0, 255] — inverse depth, so near objects are bright
        d_min, d_max = depth.min(), depth.max()
        if d_max > d_min:
            depth_u8 = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        else:
            depth_u8 = np.zeros((INF_H, INF_W), np.uint8)

        colored = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)  # (256,320,3)

        # Resize to output pane
        cv2.resize(colored, (PANE_W, PANE_H), dst=_pane_buf, interpolation=cv2.INTER_LINEAR)

        # HUD
        cv2.putText(_pane_buf, 'SCDepthV3  (DEEPX NPE-0)', (16, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(_pane_buf, f'cam0  {inf_ms:.1f} ms', (16, 76),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA)

        _pane_buf.tofile(DEPTH_PANE_FILE)
        _pipe_out.write(b'\x01' + struct.pack('<f', float(inf_ms)))

    except Exception as e:
        sys.stderr.write(f"[depth_worker] inference error: {e}\n")
        sys.stderr.flush()
        try:
            np.zeros((PANE_H, PANE_W, 3), np.uint8).tofile(DEPTH_PANE_FILE)
        except Exception:
            pass
        _pipe_out.write(b'\x01' + struct.pack('<f', 0.0))
