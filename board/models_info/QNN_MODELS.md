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
