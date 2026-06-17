# Heterogeneous AI Edge Demos — Qualcomm QCS8550 + DeepX DX-M1

**Board:** Qualcomm QCS8550 SBC (Snapdragon 8 Gen 2) + DeepX DX-M1 NPU module
**OS / Kernel:** Linux 5.15.148 (qki-consolidate) · aarch64 · Weston/Wayland

<img width="1664" height="2043" alt="Demo Picker on QCS8550" src="https://github.com/user-attachments/assets/432fa13d-b010-46c0-9248-fdce374f1bec" />

A touchscreen demo station that showcases **three real-time computer-vision pipelines**, each one
splitting work across the board's two AI accelerators — the **DeepX DX-M1 NPU** and the **Qualcomm
Hexagon HTP (DSP)** — with the Kryo CPU handling glue logic.

> **Source of truth:** this README documents *what actually runs on the board today*. The systemd
> services and scripts under [`board/`](board/) are the deployed artifacts; [`src/demo_picker.py`](src/demo_picker.py)
> is the deployed boot menu.

---

## Demo Picker

A full-screen touchscreen menu (`demo-picker.service`) launches **at boot** and lets you pick one demo
at a time. Tap a card to launch it; tap **HOME** (top-left corner of the panel) to stop the demo and
return to the menu.

| # | Demo | Service | DeepX DX-M1 | Qualcomm HTP |
|---|---|---|---|---|
| 01 | **Pose Estimation** | `imdt-deepx-demo.service` | YOLOv5-Pose (dual camera) | YOLOv8 Detection |
| 02 | **Instance Segmentation** | `yolo26-parallel.service` | YOLO26L-SEG masks | YOLOv8 Detection |
| 03 | **Depth / Face / OCR** | `deepx-dual.service` | SCDepthV3 depth + 3D face mesh | EasyOCR text detector* |

\* The EasyOCR CRAFT **detector** runs on the HTP (QNN context binary); the CRNN **recognizer** runs on
the CPU via ONNX Runtime.

### One demo at a time

Exactly one demo runs at any moment. This is enforced in two places:

- **Picker** — `launch()` stops every other demo service before starting the selected one.
- **Each service** — an `ExecStartPre` hook stops the other two demo services (and kills any stray
  workers) before it starts, so launching a demo from the shell is also mutually exclusive.

---

## Demos

### 01 · Pose Estimation — `imdt-deepx-demo.service`

The stock IMDT/dx-stream reference pipeline (`/usr/share/dx-stream/imdt-deepx-demo.sh`).

```
Camera 0 ─┬─▶ DeepX DX-M1 · YOLOv5-Pose 640 (dxpreprocess→dxinfer→dxpostprocess) ─▶ pane
          └─▶ Qualcomm HTP · YOLOv8 Detection (qtimltflite + QNN delegate)        ─▶ pane
Camera 1 ───▶ DeepX DX-M1 · YOLOv5-Pose 640                                        ─▶ pane
                                   │
                              qtivcomposer grid ─▶ waylandsink (fullscreen)
```

Real-time human-pose skeletons from two cameras on the DeepX NPU, with a YOLOv8 object-detection
overlay computed on the Hexagon HTP, composited into a single grid.

### 02 · Instance Segmentation — `yolo26-parallel.service`

Launcher: `/data/local/tmp/yolo26_launcher` (driven by `board/scripts/yolo26_parallel.sh`).

```
Camera 0 ───▶ DeepX DX-M1 · YOLO26L-SEG (instance masks) ─▶ left pane
Camera 1 ───▶ Qualcomm HTP · YOLOv8 Detection            ─▶ right pane
                                   │
                              2-pane display ─▶ waylandsink (fullscreen)
```

Dual-camera, heterogeneous: per-object instance masks on the DeepX NPU alongside bounding-box
detection on the HTP.

### 03 · Depth / Face / OCR — `deepx-dual.service`

Orchestrator: [`board/scripts/deepx_dual_demo.py`](board/scripts/deepx_dual_demo.py), which drives two
persistent worker processes over a stdin/stdout binary protocol.

```
Camera ─┬─▶ DeepX DX-M1 (dual_deepx_worker.py)
        │      • SCDepthV3            → depth map  (TURBO colormap)
        │      • YOLOv5s-Face + 3DDFA → 3D face mesh
        │
        └─▶ Qualcomm HTP (easyocr_worker.py)
               • EasyOCR CRAFT detector  → qnn-net-run (QNN context binary)
               • EasyOCR CRNN recognizer → CPU / ONNX Runtime
                                   │
                          2×2 panel display ─▶ waylandsink (fullscreen)
```

| Pane | Content | Where it runs |
|---|---|---|
| RGB | Live camera | — |
| Depth | SCDepthV3 inverse-depth, colormapped | DeepX DX-M1 |
| Face | YOLOv5s-Face box + 3DDFA 68-pt mesh | DeepX DX-M1 |
| OCR | Detected text boxes + recognized strings | HTP (detect) + CPU (recognize) |

**Performance (1280×720):** depth ≈ 20 ms, face ≈ 40 ms (DeepX, ~10 fps combined); OCR ≈ 0.5 s per
cycle (HTP detector + CPU recognizer), down from ~4.8 s when the detector ran on CPU.

---

## Systemd Services

| Service | Restart | Role |
|---|---|---|
| `demo-picker.service` | `always` | Boot menu — owns the display, launches/stops demos |
| `imdt-deepx-demo.service` | `on-failure` | Demo 01 — Pose (DeepX) + Detection (HTP) |
| `yolo26-parallel.service` | `on-failure` | Demo 02 — Segmentation (DeepX) + Detection (HTP) |
| `deepx-dual.service` | `on-failure` | Demo 03 — Depth + Face (DeepX) + EasyOCR (HTP/CPU) |
| `qmmf-server.service` | — | Qualcomm camera server (dependency) |

Service files live in `/etc/systemd/system/` on the board (this path takes precedence over
`/lib/systemd/system/`). The repo copies are under [`board/systemd/`](board/systemd/).

---

## Deploying / Updating

```sh
# One-shot deploy of all scripts, services, and the picker (Git Bash on Windows, ADB connected):
bash board/deploy.sh [serial]

# Or update a single service manually:
adb root
adb push board/systemd/deepx-dual.service /etc/systemd/system/deepx-dual.service
adb shell systemctl daemon-reload
adb shell systemctl restart demo-picker.service
```

The picker executable on the board is the Python script pushed to `/data/local/tmp/demo_picker`
(run via its `#!/usr/bin/env python3` shebang).

---

## Repository Structure

```
board/
  deploy.sh                     # One-shot deploy script
  systemd/
    demo-picker.service         # Boot menu (Restart=always)
    imdt-deepx-demo.service     # Demo 01 — pose + detection
    yolo26-parallel.service     # Demo 02 — segmentation + detection
    deepx-dual.service          # Demo 03 — depth + face + OCR
  scripts/
    deepx_dual_demo.py          # Demo 03 orchestrator
    dual_deepx_worker.py        # DeepX worker: depth + face
    easyocr_worker.py           # HTP CRAFT detector + CPU CRNN recognizer
    easyocr_post.py             # CRAFT box post-processing
    compile_easyocr_qnn.py      # Compile detector ONNX → QNN context binary
    pose_worker.py / seg_worker.py / face_worker.py
    yolo26_parallel.sh          # Demo 02 launcher wrapper
  models_info/
    QNN_MODELS.md               # QNN binary docs + re-deploy steps

src/
  demo_picker.py                # Deployed touchscreen boot menu (source of truth)

QUALCOMM_BOARD_FINDINGS.md      # GStreamer plugins, models, env vars, gotchas
```

---

## Key Environment Variables

```sh
export XDG_RUNTIME_DIR=/run/user/root
export WAYLAND_DISPLAY=wayland-1
export QT_QPA_PLATFORM=wayland-egl
export QT_WAYLAND_SHELL_INTEGRATION=wl-shell
export ADSP_LIBRARY_PATH="/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp"
```
