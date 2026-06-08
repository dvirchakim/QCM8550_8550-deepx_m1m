# IMDT QCS8550 + DeepX M1 — Heterogeneous AI Edge Demo

**Board:** Qualcomm QCS8550 SBC (Snapdragon 8 Gen 2) + DeepX DX-M1 NPU module  
**Kernel:** Linux 5.15.148-qki-consolidate · aarch64

<img width="1536" height="2048" alt="Demo Picker on QCS8550" src="https://github.com/user-attachments/assets/1f4d4cc9-9de0-4ed7-84e8-9e17cf4b4dc9" />

---

## Demo Picker

A touchscreen menu that runs **at boot** and lets you select which demo to launch.  
Press **HOME** at any time to stop the current demo and return to the picker.

### Boot behaviour
`demo-picker.service` is enabled at startup and launches automatically after Weston (Wayland display server) is ready. It stops any previously running demo, then shows the selection menu on the attached display.

### Available demos

| Demo | Service | Description |
|---|---|---|
| **IMDT DeepX** | `imdt-deepx-demo.service` | Dual-camera YOLOv5-Pose + YOLOX-S detection on DeepX DX-M1 NPU |
| **Edge-Art GenAI** | `edge-art.service` | Pose skeleton (DeepX) + ControlNet/SD 1.5 generative art (Qualcomm HTP) |

---

## Hardware Architecture

### Demo 1 — IMDT DeepX (dual-camera)

```
┌──────────────────────────────────────────────────────┐
│                    QCS8550 SBC                       │
│                                                      │
│  ┌─────────┐    ┌──────────────────────────────────┐ │
│  │Camera 0 │───▶│ DeepX DX-M1 NPU                  │ │
│  └─────────┘    │  YOLOv5-Pose  →  left pane        │ │
│                 └──────────────────────────────────┘ │
│  ┌─────────┐    ┌──────────────────────────────────┐ │
│  │Camera 1 │───▶│ Qualcomm HTP (DSP)               │ │
│  └─────────┘    │  YOLOv8 Detection → right pane   │ │
│                 └──────────────────────────────────┘ │
│                         │                            │
│                 ┌────────▼─────────┐                 │
│                 │  Weston/Wayland  │                 │
│                 │  2-pane display  │                 │
│                 └──────────────────┘                 │
└──────────────────────────────────────────────────────┘
```

### Demo 2 — Edge-Art GenAI (single camera → two NPUs)

```
┌──────────────────────────────────────────────────────┐
│                    QCS8550 SBC                       │
│                                                      │
│              ┌─────────┐                             │
│              │Camera 0 │                             │
│              └────┬────┘                             │
│                   │ 1280×720 BGR                     │
│         ┌─────────┴──────────┐                       │
│         ▼                    ▼                       │
│  ┌─────────────┐    ┌──────────────────────────────┐ │
│  │DeepX DX-M1  │    │ Qualcomm HTP (DSP)           │ │
│  │YOLOv5-Pose  │    │  ControlNet-Canny            │ │
│  │skeleton     │    │  SD 1.5 UNet  ×4 DDIM steps  │ │
│  │left pane    │    │  VAE Decoder                 │ │
│  └──────┬──────┘    │  qnn-net-run + QAIRT         │ │
│         │           │  right pane  512×512         │ │
│         │           └──────────────┬───────────────┘ │
│         │                          │                  │
│         └──────────┬───────────────┘                 │
│                    ▼                                  │
│           ┌─────────────────┐                        │
│           │  Weston/Wayland │                        │
│           │  2-pane display │                        │
│           └─────────────────┘                        │
└──────────────────────────────────────────────────────┘
```

---

## Edge-Art GenAI Demo

**Left pane** — DeepX DX-M1 runs YOLOv5-Pose in real time, overlaying a skeleton on the live camera feed.  
**Right pane** — Qualcomm HTP runs ControlNet-Canny + Stable Diffusion 1.5, generating stylized art conditioned on the pose Canny edge map.

### Style presets (auto-cycle every 20 s)

| Style | Description |
|---|---|
| `neon` | Neon Cyberpunk |
| `vangogh` | Van Gogh painterly |
| `comic` | Comic Ink |
| `noir` | Film Noir |

### QNN Model files on board (`/data/local/tmp/models/`)

| File | Size |
|---|---|
| `unet_qairt_context.bin` | 881 MB |
| `controlnet_qairt_context.bin` | 369 MB |
| `text_encoder_qairt_context.bin` | 163 MB |
| `vae_qairt_context.bin` | 65 MB |

> Model files are not tracked in git (too large). See [`board/models_info/QNN_MODELS.md`](board/models_info/QNN_MODELS.md) for re-deployment instructions.

---

## Systemd Services

| Service | Enabled | Restart | Purpose |
|---|---|---|---|
| `demo-picker.service` | ✅ | `always` | Boot menu — owns the display |
| `edge-art.service` | ✅ | `on-failure` | GenAI demo — launched by picker |
| `imdt-deepx-demo.service` | ✅ | — | Original DeepX demo |
| `qmmf-server.service` | ✅ | — | Camera server (required) |

---

## Deploying / Updating

```sh
# Push all scripts and services (from Windows, ADB connected):
bash board/deploy.sh a9ef4ffe

# Or manually reload a single service:
adb root
adb push board/systemd/edge-art.service /lib/systemd/system/edge-art.service
adb shell systemctl daemon-reload
adb shell systemctl restart demo-picker.service
```

---

## Repo Structure

```
board/
  deploy.sh                  # One-shot deploy script
  systemd/
    demo-picker.service      # Boot menu service (Restart=always)
    edge-art.service         # GenAI demo service
    imdt-deepx-demo.service  # Original DeepX demo service
  models_info/
    QNN_MODELS.md            # QNN binary documentation + re-deploy steps

scripts/
  edge_art_genai.py          # Main GenAI demo (DEEPX pose + HTP ControlNet/SD)
  run_genai_demo.sh          # Launch wrapper for edge-art.service
  pose_worker.py             # Persistent DEEPX pose worker (C++ IPC protocol)
  live_demo.py               # Standalone DEEPX pose demo

board_results/               # Screenshots and captured frames
board_backup_20260525_*/     # Full board config/service backup

QUALCOMM_BOARD_FINDINGS.md   # GStreamer plugins, models, env vars, gotchas
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
