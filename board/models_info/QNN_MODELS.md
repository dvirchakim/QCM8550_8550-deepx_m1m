# QNN Pre-compiled Model Binaries

These files live on the board at `/data/local/tmp/models/` and are **not tracked in git** (too large).

## Files

| File | Size | Description |
|---|---|---|
| `unet_qairt_context.bin` | 881 MB | SD 1.5 UNet denoiser — QAIRT context for HTP |
| `controlnet_qairt_context.bin` | 369 MB | ControlNet Canny conditioner — QAIRT context for HTP |
| `text_encoder_qairt_context.bin` | 163 MB | CLIP text encoder — QAIRT context for HTP |
| `vae_qairt_context.bin` | 65 MB | VAE decoder (latent→512×512 RGB) — QAIRT context for HTP |

**Total on device:** ~1.48 GB in `/data/local/tmp/models/`

## DeepX Dual demo — EasyOCR detector on HTP

| File | Size | Description |
|---|---|---|
| `easyocr_detector_qcs8550.bin` | 40 MB | EasyOCR CRAFT text detector — QAIRT context for HTP (`/data/local/tmp/easyocr/`) |

- Input `image` `(1,3,480,800)` float32 NCHW; output `output_0` `(1,240,400,2)` float32 (text/link score maps).
- Runs via `qnn-net-run --retrieve_context=... --backend=/usr/lib/libQnnHtp.so` from `easyocr_worker.py`.
- **Must be compiled with QAIRT `2.42`** to match the board runtime (`qnn-net-run v2.41.0`); `2.45+` fails to load (`Create From Binary failure`).
- Detector on HTP ~0.5s/cycle (incl. context load) vs ~4.8s on CPU; recognizer stays on CPU.

Recompile:
```sh
python board/scripts/compile_easyocr_qnn.py <export_dir> 2.42
adb push <export_dir>/easyocr_detector_qcs8550.bin /data/local/tmp/easyocr/
```

## How to re-deploy if wiped

```sh
# From a Linux/WSL machine with qai-hub configured:
python -m qai_hub_models.models.controlnet_canny.export \
    --device "QCS8550 (Proxy)" \
    --output-dir /tmp/controlnet_out

# Then push all .bin files:
adb push /tmp/controlnet_out/*.bin /data/local/tmp/models/
```

## Runtime

Executed by `edge_art_genai.py` via `qnn-net-run` subprocess:
- Backend: `/usr/lib/libQnnHtp.so`
- I/O staging: `/tmp/qnn_io/` and `/tmp/qnn_out/`
- All quantization params hardcoded in `edge_art_genai.py` (uint16 affine)

## DEEPX DX-M1 Models

On board at `/usr/share/dx-stream/dx_stream/samples/models/`:
- `YOLOV5Pose640_1.dxnn` — YOLOv5-Pose 640px (used for left pane)
- `YOLOV5Pose_PPU.dxnn` — Pose post-processing unit
- `YOLOX-S_1.dxnn` — YOLOX-S detection

In `/data/local/tmp/`:
- `yolo26n-seg.dxnn` — YOLOv2.6 nano segmentation
- `yolo26l-seg.dxnn` — YOLOv2.6 large segmentation
