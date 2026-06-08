# Qualcomm QCS8550 Board — Key Findings
> Updated: 2026-05-26  Session findings from live hardware exploration.

---

## 1. Hardware & Software Stack

| Item | Value |
|---|---|
| SoC | Qualcomm QCS8550 (≡ Snapdragon 8 Gen 2 / SM8550) |
| NPUs | DEEPX DX-M1 (on-module), Qualcomm HTP (DSP/Hexagon) |
| Display server | Weston (Wayland) |
| Python | 3.10.12 (aarch64) |
| GStreamer | 1.20.7 |
| Kernel modules | `dxrt_driver.ko` (DEEPX) |

---

## 2. HTP Inference — How It Works

HTP inference is exposed **only through GStreamer** (no Python tflite_runtime on board).

```sh
# Canonical HTP inference pipeline fragment:
qtimlvconverter !
qtimltflite model=/opt/model.tflite \
    external-delegate-path=libQnnTFLiteDelegate.so \
    external-delegate-options="QNNExternalDelegate,backend_type=htp,htp_performance_mode=(string)2;" !
<post-processor>
```

**There is NO `tflite_runtime` Python package on the board.**  
HTP is only reachable via `qtimltflite` GStreamer element.

---

## 3. GStreamer Plugins Available (`/usr/lib/gstreamer-1.0/`)

| Plugin | Purpose |
|---|---|
| `libgstqtiqmmfsrc.so` | Camera source (`qtiqmmfsrc`) |
| `libgstqtimltflite.so` | TFLite inference on HTP (`qtimltflite`) |
| `libgstqtimlqnn.so` | Direct QNN inference (`qtimlqnn`) |
| `libgstqtimlsnpe.so` | SNPE inference (`qtimlsnpe`) |
| `libgstqtimlvconverter.so` | Frame→ML tensor prep (`qtimlvconverter`) |
| `libgstqtimlvdetection.so` | YOLOv8/v5 detection overlay (`qtimlvdetection`) |
| `libgstqtimlvpose.so` | Pose overlay via HTP (`qtimlvpose`) |
| `libgstqtimlvsegmentation.so` | Segmentation overlay (`qtimlvsegmentation`) |
| `libgstqtimlvsuperresolution.so` | Super-resolution (`qtimlvsuperresolution`) |
| `libgstqtivcomposer.so` | Multi-stream compositor (`qtivcomposer`) |
| `libgstqtivtransform.so` | Hardware scaler/converter (`qtivtransform`) |
| `libgstqtivoverlay.so` | OSD overlay (`qtivoverlay`) |
| `libgstapp.so` | `appsrc` / `appsink` (Python pipe bridge) |
| `libgstshm.so` | Shared-memory transport |
| `libgstwaylandsink.so` | Wayland display sink |

---

## 4. `qtimlvsegmentation` — Exact Correct Usage

```sh
qtimlvsegmentation \
    labels=/opt/deeplabv3_resnet50.labels \
    module=deeplab-argmax \
    constants="deeplab,q-offsets=<8.0>,q-scales=<0.0040499246679246426>;"
```

**Gotchas discovered (hard way):**
- ❌ `module=deeplab` → rejected ("could not set property")
- ✅ `module=deeplab-argmax` → correct enum value
- ❌ `threshold=50` → no such property on this element
- ❌ `module=` is still required — omitting it gives "Module name not set" error

---

## 5. Models Pre-installed on Board (`/opt/`)

| File | Task | Notes |
|---|---|---|
| `YOLOv8-Detection-Quantized.tflite` | Object detection | Used in original demo |
| `deeplabv3_plus_mobilenet_quantized.tflite` | Semantic segmentation | Person mask |
| `hrnet_pose_quantized.tflite` | Pose estimation (HTP) | Alt to DEEPX pose |
| `quicksrnetsmall_quantized.tflite` | Super resolution | Already on board |
| `yolov5.tflite` | Detection | Float version |
| `midas_quantized.tflite` | Monocular depth | Artistic potential |
| `inception_v3_quantized.tflite` | Classification | |
| `deeplabv3_plus_mobilenet_quantized.tflite` | Segmentation | |

DEEPX models in `/usr/share/dx-stream/dx_stream/samples/models/`:
- `YOLOV5Pose640_1.dxnn` — pose (used by `dx_engine` Python API)
- `DeepLabV3PlusMobileNetV2_2.dxnn` — segmentation on DEEPX

---

## 6. DEEPX Python API (`dx_engine`)

```python
from dx_engine import InferenceEngine, InferenceOption
opt = InferenceOption(); opt.set_buffer_count(4)
eng = InferenceEngine("/usr/share/dx-stream/dx_stream/samples/models/YOLOV5Pose640_1.dxnn", opt)
out = eng.run(np.expand_dims(frame_640x640_bgr, 0))  # returns list of ndarrays
```

Output tensor: shape `(-1, 57)` → `4 bbox + 1 obj + 1 cls + 51 kps (17×3)`

---

## 7. Critical Environment Variables

```sh
export XDG_RUNTIME_DIR=/run/user/root
export WAYLAND_DISPLAY=wayland-1
export QT_QPA_PLATFORM=wayland-egl
export QT_WAYLAND_SHELL_INTEGRATION=wl-shell
export ADSP_LIBRARY_PATH="/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp"
```

Missing **any** of these causes GStreamer Wayland or ADSP/HTP init to silently fail.

---

## 8. Camera (`qmmf-server`) Lifecycle Rules

- `qmmf-server` **must be restarted** between separate `gst-launch` runs that use `qtiqmmfsrc`
- Stopping the systemd demo service and immediately starting a new `gst-launch` → "Camera service has died"
- Fix: `systemctl restart qmmf-server.service && sleep 3` before every new camera pipeline
- DRM resource leaks (`DRM_IOCTL_PRIME_FD_TO_HANDLE failed`) require **board reboot** to clear

---

## 9. ControlNet / Stable Diffusion — Export Blocker

The PRD targets `sd15_controlnet.dlc` (SD 1.5 + ControlNet-Canny on HTP).

| Step | Status |
|---|---|
| AI Hub account + API token | ✅ Configured |
| `qai-hub-models[controlnet_canny]` installed (Windows venv) | ✅ Done |
| WSL Ubuntu 22.04 available | ✅ Available |
| `aimet_onnx` (quantization toolkit) | ❌ Linux-only, not installable on Windows |
| Export via WSL | 🔲 Not yet attempted |

**Blocker**: `qai_hub_models.models.controlnet_canny` calls `aimet_onnx` during `from_pretrained()` before any cloud submission. This is a **local Linux-only quantization step** that must run in WSL or a Linux machine.

**Next step to unblock**: Run in WSL:
```bash
pip install 'qai-hub-models[controlnet_canny]'
qai-hub configure --api_token utdo8sllb29z06vawbf33rh53i0mc9i2gobee2k8
python -m qai_hub_models.models.controlnet_canny.export \
    --device "QCS8550 (Proxy)" \
    --output-dir /mnt/c/Users/dvir/CascadeProjects/qualcomm+deepx_m1/models/controlnet_canny_qcs8550
```

---

## 10. What Works Today (Validated)

- ✅ Original DEEPX demo (`imdt-deepx-demo.sh`) stable with 2 cameras
- ✅ `dx_engine` Python pose inference (live_demo.py camera → DEEPX → waylandsink)
- ✅ `qtimltflite` HTP detection in GStreamer (`qtimlvdetection module=yolov8`)
- ✅ `qtivcomposer` 4-pane layout → `waylandsink fullscreen`
- ✅ `appsrc`/`appsink` available for Python↔GStreamer pipe bridging
- ✅ `quicksrnetsmall_quantized.tflite` — super-resolution model on board, ready to use
- ✅ `deeplabv3_plus_mobilenet_quantized.tflite` — segmentation on HTP via GStreamer

---

## 11. Recommended Architecture (No ControlNet Blocker)

While ControlNet export is being resolved via WSL, the fastest path to a **live heterogeneous NPU demo** is:

```
Camera (qtiqmmfsrc)
  │
  ├─→ DEEPX (dx_engine Python)       → pose skeleton overlay   → left pane
  │
  └─→ HTP (GStreamer qtimltflite)    → DeepLabV3 segmentation  → stylized art → right pane
        via appsrc pipe ←──── Python sends frame
        via appsink pipe ────→ Python reads seg result
        OpenCV art effect applied on Python side
```

This uses **both NPUs** today with zero new model downloads.
