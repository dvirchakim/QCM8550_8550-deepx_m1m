#!/usr/bin/env python3
"""
Edge-Art GenAI Demo  –  IMDT QCS8550 + DEEPX DX-M1
====================================================
Pipeline:
  Camera  -->  [DEEPX DX-M1 NPU]  -->  YOLOv5-Pose skeleton  (left pane)
          -->  [Qualcomm HTP NPU]  -->  ControlNet Canny + SD 1.5 art  (right pane)

Inference on Qualcomm HTP uses qnn-net-run with pre-compiled QNN context binaries.

Deploy:
    adb push edge_art_genai.py  /data/local/tmp/edge_art_genai.py
    adb push run_genai_demo.sh  /data/local/tmp/run_genai_demo.sh
    adb push models/            /data/local/tmp/models/
    adb shell sh /data/local/tmp/run_genai_demo.sh
"""
from __future__ import annotations

import os, queue, struct, subprocess, sys, threading, time, shlex, ctypes
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from dx_engine import InferenceEngine, InferenceOption

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
POSE_MODEL   = "/usr/share/dx-stream/dx_stream/samples/models/YOLOV5Pose640_1.dxnn"
MODEL_DIR    = "/data/local/tmp/models"
TEXT_ENC_BIN = f"{MODEL_DIR}/text_encoder_qairt_context.bin"
CN_BIN       = f"{MODEL_DIR}/controlnet_qairt_context.bin"
UNET_BIN     = f"{MODEL_DIR}/unet_qairt_context.bin"
VAE_BIN      = f"{MODEL_DIR}/vae_qairt_context.bin"
QNN_BACKEND  = "/usr/lib/libQnnHtp.so"
QNN_IO_DIR   = "/tmp/qnn_io"
QNN_OUT_DIR  = "/tmp/qnn_out"

# ---------------------------------------------------------------------------
# Display + camera config
# ---------------------------------------------------------------------------
CAM_W, CAM_H, CAM_FPS = 1280, 720, 15
PANE_W,  PANE_H       = 640, 720
POSE_SIZE             = 640
POSE_CONF             = 0.30
NMS_THR               = 0.45

# SD config
SD_W, SD_H   = 512, 512
LAT_H, LAT_W = 64, 64
SD_STEPS     = 6      # denoising steps (6 = ~36s with CFG)
CFG_SCALE    = 7.5    # classifier-free guidance strength (0=off, 7.5=standard)
ADSP_PATH    = "/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp"

# ---------------------------------------------------------------------------
# Pre-tokenized CLIP prompts (computed offline, no transformers needed)
# CLIP vocab: BOS=49406, EOS/PAD=49407
# ---------------------------------------------------------------------------
TOKENIZED = {
    "neon":    [49406, 13919, 36896, 10317, 2338, 267, 18437, 3418, 267, 5031,
                1746, 537, 5496, 267, 33841, 4535, 49407] + [49407]*60,
    "vangogh": [49406, 2451, 19697, 1220, 26493, 27969, 26595, 267, 2870, 3086,
                267, 30963, 930, 1844, 267, 14270, 49407] + [49407]*60,
    "comic":   [49406, 4962, 1116, 967, 6052, 267, 8911, 29159, 267, 1296, 3391,
                637, 19019, 267, 2852, 794, 49407] + [49407]*60,
    "noir":    [49406, 12953, 1860, 26149, 267, 11240, 12971, 267, 1449, 537,
                1579, 267, 25602, 49407] + [49407]*63,
    "neg":     [49406] + [49407]*76,
}
STYLE_KEYS    = ["neon", "vangogh", "comic", "noir"]
STYLE_CYCLE_S = 30.0   # auto-cycle period (overridden by touch)
CROSSFADE_FRAMES = 12  # art blend-in duration (frames)

STYLE_LABELS = {
    "neon":    "NEON CYBERPUNK",
    "vangogh": "VAN GOGH",
    "comic":   "COMIC INK",
    "noir":    "FILM NOIR",
}

# ---------------------------------------------------------------------------
# Quantization params from metadata.json (scale, zero_point)
# Affine quantization: float = (uint16 - zero_point) * scale
#                      uint16 = clip(round(float / scale + zero_point), 0, 65535)
# ---------------------------------------------------------------------------
Q = {
    "te_text_emb_out": (0.0009303585393354297, 30063),

    "cn_latent":    (0.00018526687927078456, 34097),
    "cn_timestep":  (0.013639155775308609, 0),
    "cn_text_emb":  (0.0009331560577265918, 30103),
    "cn_image_cond": (0.000015259021893143654, 0),

    "cn_out_0":  (0.00003325078432681039,   36922),
    "cn_out_1":  (0.00006370455957949162,   35923),
    "cn_out_2":  (0.00032040136284194887,   35994),
    "cn_out_3":  (0.0003009315114468336,    35324),
    "cn_out_4":  (0.00025323042063973844,   42235),
    "cn_out_5":  (0.0003352258063387126,    31681),
    "cn_out_6":  (0.00039335372275672853,   30452),
    "cn_out_7":  (0.0002697439049370587,    39338),
    "cn_out_8":  (0.00045733703882433474,   30962),
    "cn_out_9":  (0.0004257652326487005,    34612),
    "cn_out_10": (0.0004820766334887594,    30938),
    "cn_out_11": (0.0004622728156391531,    27360),
    "cn_out_mid":(0.0007909731357358396,    26855),

    "unet_latent":    (0.00017259483865927905, 35671),
    "unet_timestep":  (0.015243763104081154, 0),
    "unet_text_emb":  (0.0009331560577265918, 30103),
    "unet_db0":  (0.000033097730920417234,  36785),
    "unet_db1":  (0.00006190238491399214,   36200),
    "unet_db2":  (0.00031561125069856644,   37431),
    "unet_db3":  (0.00031502419733442366,   34672),
    "unet_db4":  (0.00026047168648801744,   42271),
    "unet_db5":  (0.0003284156846348196,    32097),
    "unet_db6":  (0.00039996355189941823,   28820),
    "unet_db7":  (0.00025007667136378586,   35363),
    "unet_db8":  (0.00045629122178070247,   32159),
    "unet_db9":  (0.000442721153376624,     36521),
    "unet_db10": (0.0005222870386205614,    31060),
    "unet_db11": (0.0004957998171448708,    27021),
    "unet_mid":  (0.0008833248866721988,    26203),
    "unet_out":  (0.00013009473332203925,   33292),

    "vae_latent": (0.00020857801428064704,  33693),
    "vae_image":  (0.000015259021893143654, 0),
}


def quant(arr: np.ndarray, key: str) -> np.ndarray:
    sc, zp = Q[key]
    return np.clip(np.round(arr.astype(np.float32) / sc + zp), 0, 65535).astype(np.uint16)


def dequant(arr: np.ndarray, key: str) -> np.ndarray:
    sc, zp = Q[key]
    return (arr.astype(np.float32) - zp) * sc


# ---------------------------------------------------------------------------
# DDIM noise schedule (SD 1.5 linear beta schedule)
# ---------------------------------------------------------------------------
def _build_alphas_cumprod(n=1000, b0=0.00085, b1=0.012) -> np.ndarray:
    betas = np.linspace(b0**0.5, b1**0.5, n, dtype=np.float64) ** 2
    return np.cumprod(1.0 - betas).astype(np.float32)


ALPHAS_CUMPROD = _build_alphas_cumprod()


def get_ddim_timesteps(num_steps: int) -> list[int]:
    # Evenly spaced from t=999 down to t=0
    if num_steps == 1:
        return [999]
    step = 999 // (num_steps - 1)
    ts = sorted([min(i * step, 999) for i in range(num_steps)], reverse=True)
    return ts


def ddim_step(x_t: np.ndarray, noise_pred: np.ndarray,
              t: int, t_prev: int) -> np.ndarray:
    a_t = ALPHAS_CUMPROD[t]
    a_prev = ALPHAS_CUMPROD[max(t_prev, 0)] if t_prev >= 0 else 1.0
    x0 = np.clip((x_t - np.sqrt(1 - a_t) * noise_pred) / np.sqrt(a_t), -1, 1)
    return np.sqrt(a_prev) * x0 + np.sqrt(1 - a_prev) * noise_pred


# ---------------------------------------------------------------------------
# QNN runner (wraps qnn-net-run subprocess)
# ---------------------------------------------------------------------------
class QNNRunner:
    """Runs a single QNN context binary using qnn-net-run."""

    def __init__(self, bin_path: str, tag: str):
        self.bin_path = bin_path
        self.tag      = tag
        self.io_dir   = f"{QNN_IO_DIR}/{tag}"
        self.out_dir  = f"{QNN_OUT_DIR}/{tag}"
        self.res_dir  = f"{self.out_dir}/Result_0"
        os.makedirs(self.io_dir,  exist_ok=True)
        os.makedirs(self.res_dir, exist_ok=True)
        self._env = os.environ.copy()
        self._env["ADSP_LIBRARY_PATH"] = ADSP_PATH

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        # Write input tensors
        parts = []
        for name, arr in inputs.items():
            fpath = f"{self.io_dir}/{name}.bin"
            arr.tofile(fpath)
            parts.append(f"{name}:={fpath}")

        ilist = f"{self.io_dir}/input_list.txt"
        with open(ilist, "w") as f:
            f.write(" ".join(parts) + "\n")

        # Clear old output
        for fn in os.listdir(self.res_dir):
            os.unlink(f"{self.res_dir}/{fn}")

        cmd = [
            "qnn-net-run",
            f"--retrieve_context={self.bin_path}",
            f"--backend={QNN_BACKEND}",
            f"--input_list={ilist}",
            f"--output_dir={self.out_dir}",
            "--use_native_input_files",
            "--use_native_output_files",
        ]
        r = subprocess.run(cmd, capture_output=True, env=self._env)
        if r.returncode != 0:
            raise RuntimeError(
                f"[{self.tag}] qnn-net-run failed:\n{r.stderr.decode(errors='replace')[-600:]}"
            )

        # Read outputs (files: {name}_native.raw)
        outputs: dict[str, np.ndarray] = {}
        for fn in os.listdir(self.res_dir):
            if fn.endswith("_native.raw"):
                tname = fn[: -len("_native.raw")]
                outputs[tname] = np.fromfile(f"{self.res_dir}/{fn}", dtype=np.uint16)
        return outputs


# ---------------------------------------------------------------------------
# ControlNet + SD 1.5 pipeline
# ---------------------------------------------------------------------------
class ControlNetPipeline:
    """
    Full ControlNet Canny + SD 1.5 inference using QNN HTP.
    Runs asynchronously in a background thread.
    No CFG (guidance_scale=1) for speed. 4 DDIM steps by default.
    """

    def __init__(self):
        self._ready    = False
        self._q: queue.Queue   = queue.Queue(maxsize=1)
        self._lock             = threading.Lock()
        self._latest: np.ndarray | None = None
        self._fps_t: deque     = deque(maxlen=8)

        # Cached text embeddings (uint16, shape [1,77,768])
        self._text_emb_cache: dict[str, np.ndarray] = {}

        self._runner: dict[str, QNNRunner] = {}
        self._load_thread = threading.Thread(target=self._init, daemon=True)
        self._load_thread.start()
        self._work_thread = threading.Thread(target=self._work_loop, daemon=True)
        self._work_thread.start()

    # -----------------------------------------------------------------------
    def _init(self):
        try:
            for tag, path in [
                ("te",   TEXT_ENC_BIN),
                ("cn",   CN_BIN),
                ("unet", UNET_BIN),
                ("vae",  VAE_BIN),
            ]:
                if not os.path.exists(path):
                    print(f"[genai] MISSING: {path}")
                    return
                self._runner[tag] = QNNRunner(path, tag)
            self._ready = True
            print("[genai] QNN runners ready.")
        except Exception as e:
            print(f"[genai] Init failed: {e}")

    # -----------------------------------------------------------------------
    def _encode_text(self, style_key: str) -> np.ndarray:
        if style_key in self._text_emb_cache:
            return self._text_emb_cache[style_key]
        tokens = np.array([TOKENIZED[style_key]], dtype=np.int32)
        out = self._runner["te"].run({"tokens": tokens})
        emb = out["text_embedding"]  # uint16, flat
        self._text_emb_cache[style_key] = emb
        print(f"[genai] text emb cached for {style_key}")
        return emb

    # -----------------------------------------------------------------------
    def _run_controlnet(self, latent_u16: np.ndarray, timestep_u16: np.ndarray,
                        text_emb_u16: np.ndarray, canny_u16: np.ndarray
                        ) -> dict[str, np.ndarray]:
        inp = {
            "latent":     latent_u16,
            "timestep":   timestep_u16,
            "text_emb":   text_emb_u16,
            "image_cond": canny_u16,
        }
        return self._runner["cn"].run(inp)

    # -----------------------------------------------------------------------
    def _run_unet(self, latent_u16: np.ndarray, timestep_u16: np.ndarray,
                  text_emb_u16: np.ndarray, cn_out: dict[str, np.ndarray]
                  ) -> np.ndarray:
        # Map ControlNet output names → UNet input names, with re-quantization
        inp = {
            "latent":   latent_u16,
            "timestep": timestep_u16,
            "text_emb": text_emb_u16,
        }
        cn_to_unet = [
            ("down_block_0",  "controlnet_downblock0",  "cn_out_0",  "unet_db0"),
            ("down_block_1",  "controlnet_downblock1",  "cn_out_1",  "unet_db1"),
            ("down_block_2",  "controlnet_downblock2",  "cn_out_2",  "unet_db2"),
            ("down_block_3",  "controlnet_downblock3",  "cn_out_3",  "unet_db3"),
            ("down_block_4",  "controlnet_downblock4",  "cn_out_4",  "unet_db4"),
            ("down_block_5",  "controlnet_downblock5",  "cn_out_5",  "unet_db5"),
            ("down_block_6",  "controlnet_downblock6",  "cn_out_6",  "unet_db6"),
            ("down_block_7",  "controlnet_downblock7",  "cn_out_7",  "unet_db7"),
            ("down_block_8",  "controlnet_downblock8",  "cn_out_8",  "unet_db8"),
            ("down_block_9",  "controlnet_downblock9",  "cn_out_9",  "unet_db9"),
            ("down_block_10", "controlnet_downblock10", "cn_out_10", "unet_db10"),
            ("down_block_11", "controlnet_downblock11", "cn_out_11", "unet_db11"),
            ("mid_block",     "controlnet_midblock",    "cn_out_mid","unet_mid"),
        ]
        for cn_name, unet_name, cn_qkey, unet_qkey in cn_to_unet:
            if cn_name in cn_out:
                f32 = dequant(cn_out[cn_name], cn_qkey)
                inp[unet_name] = quant(f32, unet_qkey)

        out = self._runner["unet"].run(inp)
        return out.get("output_latent", np.zeros(LAT_H * LAT_W * 4, dtype=np.uint16))

    # -----------------------------------------------------------------------
    def _run_vae(self, latent_u16: np.ndarray) -> np.ndarray:
        out = self._runner["vae"].run({"latent": latent_u16})
        img_u16 = out.get("image", np.zeros(SD_H * SD_W * 3, dtype=np.uint16))
        # Dequantize → [0,1] float → [0,255] uint8 BGR
        img_f = dequant(img_u16, "vae_image").reshape(SD_H, SD_W, 3)
        img_f = np.clip(img_f, 0, 1)
        img_bgr = (img_f[:, :, ::-1] * 255.0).astype(np.uint8)
        return img_bgr

    # -----------------------------------------------------------------------
    def _denoise(self, canny_map: np.ndarray, style_key: str) -> np.ndarray:
        # 1. Text embeddings — cond (style) + uncond (negative) for CFG
        te_cond_u16  = self._encode_text(style_key)
        te_uncond_u16 = self._encode_text("neg")
        te_cond_f32   = dequant(te_cond_u16,  "te_text_emb_out").reshape(1, 77, 768)
        te_uncond_f32 = dequant(te_uncond_u16, "te_text_emb_out").reshape(1, 77, 768)
        # Re-quantize into CN / UNet input space
        te_cn_cond    = quant(te_cond_f32,   "cn_text_emb").flatten()
        te_unet_cond  = quant(te_cond_f32,   "unet_text_emb").flatten()
        te_unet_uncond = quant(te_uncond_f32, "unet_text_emb").flatten()

        # 2. Canny condition → quantize for ControlNet
        canny_resized = cv2.resize(canny_map, (SD_W, SD_H))
        canny_rgb = canny_resized[:, :, ::-1].astype(np.float32) / 255.0  # BGR→RGB, [0,1]
        canny_nhwc = canny_rgb[np.newaxis]  # (1, 512, 512, 3)
        canny_u16 = quant(canny_nhwc, "cn_image_cond").flatten()

        # 3. Initialize random latent
        timesteps = get_ddim_timesteps(SD_STEPS)
        latent_f32 = np.random.randn(1, LAT_H, LAT_W, 4).astype(np.float32)

        # 4. DDIM denoising loop with CFG
        #    Strategy: run CN once per step (conditioned), share residuals with both
        #    conditioned and unconditioned UNet passes → 1 CN + 2 UNet per step.
        for step_i, t in enumerate(timesteps):
            t_prev = timesteps[step_i + 1] if step_i + 1 < len(timesteps) else -1

            lat_cn   = quant(latent_f32, "cn_latent").flatten()
            lat_unet = quant(latent_f32, "unet_latent").flatten()
            ts_cn    = quant(np.array([[np.float32(t)]], np.float32), "cn_timestep").flatten()
            ts_unet  = quant(np.array([[np.float32(t)]], np.float32), "unet_timestep").flatten()

            # ControlNet (cond text + canny) — residuals shared with both UNet passes
            cn_out = self._run_controlnet(lat_cn, ts_cn, te_cn_cond, canny_u16)

            # UNet — conditioned
            noise_u16_cond  = self._run_unet(lat_unet, ts_unet, te_unet_cond,  cn_out)
            noise_cond      = dequant(noise_u16_cond,  "unet_out").reshape(1, LAT_H, LAT_W, 4)

            # UNet — unconditioned (same CN residuals, empty text)
            noise_u16_uncond = self._run_unet(lat_unet, ts_unet, te_unet_uncond, cn_out)
            noise_uncond     = dequant(noise_u16_uncond, "unet_out").reshape(1, LAT_H, LAT_W, 4)

            # Classifier-Free Guidance
            noise_guided = noise_uncond + CFG_SCALE * (noise_cond - noise_uncond)

            # DDIM step
            latent_f32 = ddim_step(latent_f32, noise_guided, t, t_prev)

        # 5. VAE decode
        lat_vae = quant(latent_f32, "vae_latent").flatten()
        img_bgr = self._run_vae(lat_vae)
        return img_bgr

    # -----------------------------------------------------------------------
    def _work_loop(self):
        while True:
            try:
                item = self._q.get(timeout=1.0)
            except queue.Empty:
                continue
            canny, style_key = item
            t0 = time.time()
            try:
                if self._ready:
                    img = self._denoise(canny, style_key)
                else:
                    img = self._artistic_fallback(canny, style_key)
            except Exception as e:
                print(f"[genai] inference error: {e}")
                img = self._artistic_fallback(canny, style_key)
            elapsed = time.time() - t0
            self._fps_t.append(time.time())
            with self._lock:
                self._latest = cv2.resize(img, (PANE_W, PANE_H))
            print(f"[genai] {elapsed:.1f}s  cfg={CFG_SCALE}  steps={SD_STEPS}  qnn={self._ready}  style={style_key}")

    # -----------------------------------------------------------------------
    @staticmethod
    def _artistic_fallback(canny: np.ndarray, style_key: str) -> np.ndarray:
        gray = cv2.cvtColor(canny, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        base = np.zeros((SD_H, SD_W, 3), np.uint8)
        if style_key == "neon":
            base[:] = (5, 2, 15)
            col = np.zeros_like(canny)
            col[mask > 0] = (255, 0, 230)
            glow = cv2.GaussianBlur(col, (21, 21), 0)
            base = cv2.addWeighted(base, 1.0, glow, 0.9, 0)
            col2 = np.zeros_like(canny)
            col2[mask > 0] = (0, 229, 255)
            base = cv2.addWeighted(base, 1.0, col2, 0.8, 0)
        elif style_key == "vangogh":
            base[:] = (40, 20, 80)
            stroke = np.zeros_like(canny)
            stroke[mask > 0] = (50, 150, 255)
            stroke = cv2.medianBlur(stroke, 9)
            base = cv2.addWeighted(base, 1.0, stroke, 1.2, 0)
        elif style_key == "comic":
            base[:] = 240
            base[mask > 0] = (0, 0, 0)
        else:  # noir
            base[mask > 0] = (200, 200, 200)
            base = cv2.GaussianBlur(base, (5, 5), 0)
        return base

    # -----------------------------------------------------------------------
    def submit(self, canny_map: np.ndarray, style_key: str):
        try:
            self._q.get_nowait()
        except queue.Empty:
            pass
        self._q.put_nowait((canny_map.copy(), style_key))

    def latest(self) -> np.ndarray | None:
        with self._lock:
            return self._latest

    @property
    def fps(self) -> float:
        ts = list(self._fps_t)
        if len(ts) < 2:
            return 0.0
        return (len(ts) - 1) / max(ts[-1] - ts[0], 1e-9)

    @property
    def ready(self) -> bool:
        return self._ready


# ---------------------------------------------------------------------------
# Skeleton / pose
# ---------------------------------------------------------------------------
SKELETON = [
    (0,1),(0,2),(1,3),(2,4),
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]
STYLE_COLORS = {
    "neon":    dict(bone=(255,0,230),   glow=(0,229,255),   th=4, jr=6, blur=15),
    "vangogh": dict(bone=(50,130,255),  glow=(60,200,255),  th=8, jr=8, blur=21),
    "comic":   dict(bone=(0,0,0),       glow=(50,180,255),  th=6, jr=8, blur=0),
    "noir":    dict(bone=(255,255,255), glow=(120,120,120), th=3, jr=5, blur=9),
}


def letterbox(img, size=640, pad=114):
    h, w = img.shape[:2]
    r = min(size / w, size / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    canvas = np.full((size, size, 3), pad, np.uint8)
    res = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dx, dy = (size - nw) // 2, (size - nh) // 2
    canvas[dy:dy+nh, dx:dx+nw] = res
    return canvas, r, dx, dy


def nms(boxes, scores, thr):
    if not len(boxes):
        return []
    x1=boxes[:,0]-boxes[:,2]/2; x2=boxes[:,0]+boxes[:,2]/2
    y1=boxes[:,1]-boxes[:,3]/2; y2=boxes[:,1]+boxes[:,3]/2
    areas=(x2-x1)*(y2-y1); order=scores.argsort()[::-1]; keep=[]
    while len(order):
        i=order[0]; keep.append(int(i))
        if len(order)==1: break
        xx1=np.maximum(x1[i],x1[order[1:]]); yy1=np.maximum(y1[i],y1[order[1:]])
        xx2=np.minimum(x2[i],x2[order[1:]]); yy2=np.minimum(y2[i],y2[order[1:]])
        inter=np.maximum(0,xx2-xx1)*np.maximum(0,yy2-yy1)
        iou=inter/(areas[i]+areas[order[1:]]-inter+1e-9)
        order=order[1:][iou<thr]
    return keep


def decode_pose(raw, r, dx, dy):
    obj=raw[:,4]; mask=obj>POSE_CONF; cand=raw[mask]
    if not len(cand):
        return []
    boxes=cand[:,:4]; scores=cand[:,4]*cand[:,5]; kps=cand[:,6:].reshape(-1,17,3)
    keep=nms(boxes,scores,NMS_THR); persons=[]
    for i in keep:
        b=boxes[i].copy(); k=kps[i].copy()
        b[0]=(b[0]-dx)/r; b[1]=(b[1]-dy)/r; b[2]/=r; b[3]/=r
        k[:,0]=(k[:,0]-dx)/r; k[:,1]=(k[:,1]-dy)/r
        persons.append({"bbox":b,"kps":k,"score":float(scores[i])})
    return persons


def draw_pose(frame, persons, style):
    s=STYLE_COLORS.get(style, STYLE_COLORS["neon"]); out=frame.copy()
    if s["blur"]>0 and persons:
        glow=np.zeros_like(out)
        for p in persons: _draw_skel(glow,p,s["glow"],s["th"]+4,s["jr"]+3)
        glow=cv2.GaussianBlur(glow,(s["blur"],s["blur"]),0)
        out=cv2.addWeighted(out,1.0,glow,0.85,0)
    for p in persons: _draw_skel(out,p,s["bone"],s["th"],s["jr"])
    return out


def _draw_skel(img, p, color, th, jr):
    kps=p["kps"]; pts=[(int(x),int(y)) if c>0.3 else None for x,y,c in kps]
    for a,b in SKELETON:
        if pts[a] and pts[b]: cv2.line(img,pts[a],pts[b],color,th,cv2.LINE_AA)
    for pt in pts:
        if pt: cv2.circle(img,pt,jr,color,-1,cv2.LINE_AA)


def pose_to_canny(persons, frame=None, w=SD_W, h=SD_H):
    """Hybrid canny: skeleton lines + camera-frame edges blended for richer ControlNet input."""
    canvas=np.zeros((h,w,3),np.uint8)
    sx=w/CAM_W; sy=h/CAM_H
    for p in persons:
        kps=p["kps"]; pts=[(int(x*sx),int(y*sy)) if c>0.3 else None for x,y,c in kps]
        for a,b in SKELETON:
            if pts[a] and pts[b]: cv2.line(canvas,pts[a],pts[b],(255,255,255),7,cv2.LINE_AA)
        for pt in pts:
            if pt: cv2.circle(canvas,pt,9,(255,255,255),-1,cv2.LINE_AA)
    # Dilate skeleton for thicker, more prominent ControlNet guidance
    canvas=cv2.dilate(canvas,np.ones((3,3),np.uint8),iterations=1)
    gray=cv2.cvtColor(canvas,cv2.COLOR_BGR2GRAY)
    skel_edges=cv2.Canny(gray,50,150)
    # Blend with camera-frame canny for structural detail
    if frame is not None and len(persons):
        frame_small=cv2.resize(frame,(w,h))
        frame_gray=cv2.cvtColor(frame_small,cv2.COLOR_BGR2GRAY)
        frame_blur=cv2.GaussianBlur(frame_gray,(5,5),0)
        frame_edges=cv2.Canny(frame_blur,40,100)
        # Weight: skeleton 70% + frame 30%
        blended=cv2.addWeighted(skel_edges,0.7,frame_edges,0.3,0)
    else:
        blended=skel_edges
    return cv2.cvtColor(blended,cv2.COLOR_GRAY2BGR)


def _pose_hash(persons):
    """Cheap fingerprint of current pose — only re-generate art when it changes."""
    if not persons:
        return 0
    pts=[]
    for p in persons[:2]:  # track up to 2 people
        for x,y,c in p["kps"]:
            if c>0.3: pts+=[int(x/20)*20, int(y/20)*20]  # 20px grid
    return hash(tuple(pts))


# ---------------------------------------------------------------------------
# HUD
# ---------------------------------------------------------------------------
def hud_header(frame, style, n_persons, deepx_ms, cam_fps, genai_fps, genai_ready):
    w=frame.shape[1]
    cv2.rectangle(frame,(0,0),(w,38),(8,12,28),-1)
    cv2.rectangle(frame,(0,36),(w,38),(0,180,255),-1)
    cv2.putText(frame,"IMDT QCS8550 + DEEPX DX-M1  |  HETEROGENEOUS AI EDGE DEMO",
                (10,25),cv2.FONT_HERSHEY_SIMPLEX,0.52,(0,180,255),1,cv2.LINE_AA)
    # Person count badge
    if n_persons>0:
        badge=f"{n_persons} person{'s' if n_persons>1 else ''}"
        cv2.putText(frame,badge,(w-220,25),cv2.FONT_HERSHEY_SIMPLEX,0.48,(0,255,150),1,cv2.LINE_AA)


def hud_left(pane, n_persons, deepx_ms, cam_fps, style):
    cv2.rectangle(pane,(0,38),(pane.shape[1],72),(10,12,22),-1)
    cv2.rectangle(pane,(0,70),(pane.shape[1],72),(0,229,255),-1)
    cv2.putText(pane,
        f"DEEPX DX-M1  YOLOv5-Pose  {deepx_ms:.0f}ms  {cam_fps:.0f}fps",
        (8,60),cv2.FONT_HERSHEY_SIMPLEX,0.46,(0,229,255),1,cv2.LINE_AA)


def hud_right(pane, genai_fps, genai_ready, style, gen_count):
    col=(0,255,100) if genai_ready else (0,180,255)
    label=f"QCS8550 HTP  ControlNet+SD1.5  {SD_STEPS}steps" if genai_ready else "LOADING QNN MODELS..."
    cv2.rectangle(pane,(0,38),(pane.shape[1],72),(10,22,12),-1)
    cv2.rectangle(pane,(0,70),(pane.shape[1],72),col,-1)
    gen_str=f"#{gen_count}" if gen_count else ""
    cv2.putText(pane,f"{label}  {gen_str}",
                (8,60),cv2.FONT_HERSHEY_SIMPLEX,0.42,col,1,cv2.LINE_AA)


# Style button hit-test regions (in right-pane coordinates)
STYLE_BTN_BH=44; STYLE_BTN_BW=160
def _btn_rect(i):
    x0=PANE_W-STYLE_BTN_BW-8; y=80+i*(STYLE_BTN_BH+6)
    return x0, y, x0+STYLE_BTN_BW, y+STYLE_BTN_BH

def draw_style_buttons(pane, current_style):
    for i,key in enumerate(STYLE_KEYS):
        x0,y0,x1,y1=_btn_rect(i); active=(key==current_style)
        bg=(0,180,255) if active else (30,30,50)
        border=(0,255,180) if active else (80,80,120)
        cv2.rectangle(pane,(x0,y0),(x1,y1),bg,-1)
        cv2.rectangle(pane,(x0,y0),(x1,y1),border,2)
        cv2.putText(pane,STYLE_LABELS[key],(x0+8,y0+27),
                    cv2.FONT_HERSHEY_SIMPLEX,0.42,
                    (255,255,255) if active else (160,160,200),1,cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Touch input (style switching)
# ---------------------------------------------------------------------------
_libc=ctypes.CDLL("libc.so.6",use_errno=True)
EV_KEY=0x01; EV_ABS=0x03; BTN_TOUCH=0x14A
ABS_X=0x00; ABS_Y=0x01; ABS_MT_X=0x35; ABS_MT_Y=0x36
EVIOCGBIT_EV=0x80084500
TOUCH_MAX_X=1023; TOUCH_MAX_Y=599

_touch_style=None  # set by touch thread when user taps a style button
_touch_lock=threading.Lock()

def _find_touch_dev():
    for i in range(4):
        p=f"/dev/input/event{i}"
        try:
            fd=os.open(p,os.O_RDONLY|os.O_NONBLOCK)
            buf=(ctypes.c_uint8*8)()
            if _libc.ioctl(fd,EVIOCGBIT_EV,buf)>=0:
                bits=int.from_bytes(bytes(buf),"little")
                if bits&(1<<EV_ABS): return fd
            os.close(fd)
        except Exception: pass
    return -1

def _touch_thread_fn():
    global _touch_style
    fd=_find_touch_dev()
    if fd<0: return
    fmt="llHHi"; sz=struct.calcsize(fmt)
    tx=ty=0
    while True:
        try: data=os.read(fd,sz)
        except BlockingIOError: time.sleep(0.005); continue
        if len(data)<sz: continue
        _,_,evtype,code,value=struct.unpack(fmt,data)
        if evtype==EV_ABS:
            if code in(ABS_X,ABS_MT_X): tx=value
            if code in(ABS_Y,ABS_MT_Y): ty=value
        if evtype==EV_KEY and code==BTN_TOUCH and value==1:
            # Map raw touch → screen (1920×1080 full, display is 1280×720 centred)
            sx=tx*1280//TOUCH_MAX_X; sy=ty*720//TOUCH_MAX_Y
            # Right pane starts at x=640
            if sx>=640:
                rpx=sx-640  # position within right pane
                for i,key in enumerate(STYLE_KEYS):
                    bx0,by0,bx1,by1=_btn_rect(i)
                    if bx0<=rpx<=bx1 and by0<=sy<=by1:
                        with _touch_lock: _touch_style=key
                        break


def make_placeholder(style):
    ph=np.zeros((PANE_H,PANE_W,3),np.uint8); ph[:] = (10,10,20)
    cv2.putText(ph,"LOADING CONTROLNET...",(40,PANE_H//2-20),
                cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,140,200),1,cv2.LINE_AA)
    cv2.putText(ph,"QNN HTP INITIALIZING",(55,PANE_H//2+20),
                cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,80,140),1,cv2.LINE_AA)
    return ph


def crossfade(prev, next_, alpha):
    """Alpha-blend two same-size BGR frames. alpha=0→prev, alpha=1→next_."""
    if prev is None: return next_
    if next_ is None: return prev
    a=np.clip(alpha,0.0,1.0)
    return cv2.addWeighted(prev,1.0-a,next_,a,0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Camera / display subprocesses
# ---------------------------------------------------------------------------
def _env():
    e=os.environ.copy()
    e["XDG_RUNTIME_DIR"]="/run/user/root"
    e["WAYLAND_DISPLAY"]="wayland-1"
    e["QT_QPA_PLATFORM"]="wayland-egl"
    e["QT_WAYLAND_SHELL_INTEGRATION"]="wl-shell"
    e["ADSP_LIBRARY_PATH"]=ADSP_PATH
    return e


def spawn_camera(cam_idx=0):
    pipe=(f"gst-launch-1.0 -q qtiqmmfsrc camera={cam_idx} ! qtivtransform ! "
          f"video/x-raw,width={CAM_W},height={CAM_H},format=NV12,framerate={CAM_FPS}/1 ! "
          f"videoconvert ! video/x-raw,format=BGR ! fdsink fd=1 sync=false")
    return subprocess.Popen(shlex.split(pipe),stdout=subprocess.PIPE,
                            stderr=open("/tmp/ea_cam.log","w"),env=_env(),bufsize=0)


def spawn_display():
    pipe=(f"gst-launch-1.0 -q fdsrc fd=0 ! "
          f"rawvideoparse format=bgr width={CAM_W} height={CAM_H} framerate={CAM_FPS}/1 ! "
          f"videoconvert ! waylandsink sync=false fullscreen=true")
    return subprocess.Popen(shlex.split(pipe),stdin=subprocess.PIPE,
                            stderr=open("/tmp/ea_disp.log","w"),env=_env(),bufsize=0)


def read_exact(fp, n):
    buf=bytearray()
    while len(buf)<n:
        chunk=fp.read(n-len(buf))
        if not chunk: return None
        buf.extend(chunk)
    return bytes(buf)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global _touch_style
    style_idx=0; style=STYLE_KEYS[0]; style_t0=time.time()

    print("[edge-art] Starting Edge-Art GenAI Demo")
    print(f"[edge-art] SD steps: {SD_STEPS}  |  Models: {MODEL_DIR}")

    opt=InferenceOption()
    try: opt.set_buffer_count(4)
    except: pass
    engine=InferenceEngine(POSE_MODEL, opt)
    print("[edge-art] DEEPX engine ready.")

    print("[edge-art] Starting ControlNet pipeline ...")
    pipeline=ControlNetPipeline()

    # Touch input thread (for interactive style switching)
    threading.Thread(target=_touch_thread_fn,daemon=True).start()

    subprocess.run(["systemctl","restart","qmmf-server.service"],
                   check=False,stderr=subprocess.DEVNULL)
    time.sleep(3.0)

    cam=spawn_camera(0); time.sleep(1.5)
    if cam.poll() is not None:
        print("[edge-art] Camera died — check /tmp/ea_cam.log"); return 1

    disp=spawn_display(); time.sleep(0.5)
    if disp.poll() is not None:
        print("[edge-art] Display died — check /tmp/ea_disp.log")
        cam.terminate(); return 1

    cam_bytes=CAM_W*CAM_H*3
    n_frames=0; persons=[]; last_deepx_ms=0.0
    displayed_art=None   # currently shown (possibly mid-crossfade)
    incoming_art=None    # newly arrived art to fade in
    crossfade_t=0        # frame counter for crossfade
    prev_pose_hash=None  # smart trigger: only re-gen when pose changes
    placeholder=make_placeholder(style)
    in_times=deque(maxlen=30); npu_times=deque(maxlen=30)
    gen_count=0

    print("[edge-art] Running — Ctrl-C to stop.")
    try:
        while True:
            t0=time.time()
            raw=read_exact(cam.stdout, cam_bytes)
            if raw is None:
                print("[edge-art] Camera pipe closed."); break
            frame=np.frombuffer(raw,np.uint8).reshape(CAM_H,CAM_W,3)
            in_times.append(time.time()-t0)

            # DEEPX pose
            t1=time.time()
            lb,r,dx,dy=letterbox(frame,POSE_SIZE)
            outs=engine.run(np.expand_dims(lb,0))
            last_deepx_ms=(time.time()-t1)*1000.0
            npu_times.append(last_deepx_ms/1000.0)
            persons=decode_pose(outs[0].reshape(-1,57),r,dx,dy)

            # Touch: interactive style switch
            with _touch_lock:
                if _touch_style is not None:
                    style=_touch_style; _touch_style=None
                    style_idx=STYLE_KEYS.index(style); style_t0=time.time()
                    placeholder=make_placeholder(style)
                    print(f"[edge-art] touch style → {style}")

            # Smart GenAI trigger: only re-submit when pose changed meaningfully
            cur_hash=_pose_hash(persons)
            if cur_hash!=prev_pose_hash and persons:
                canny=pose_to_canny(persons, frame)
                pipeline.submit(canny, style)
                prev_pose_hash=cur_hash

            # Art crossfade: detect new art arrival, blend it in over CROSSFADE_FRAMES
            new_art=pipeline.latest()
            if new_art is not None and new_art is not incoming_art and new_art is not displayed_art:
                incoming_art=new_art; crossfade_t=0; gen_count+=1

            if incoming_art is not None:
                alpha=crossfade_t/CROSSFADE_FRAMES
                blended=crossfade(displayed_art if displayed_art is not None else placeholder,
                                  incoming_art, alpha)
                if crossfade_t>=CROSSFADE_FRAMES:
                    displayed_art=incoming_art; incoming_art=None
                else:
                    crossfade_t+=1
                right=blended
            else:
                right=displayed_art if displayed_art is not None else placeholder.copy()

            # Compose frame
            left=draw_pose(frame,persons,style)
            left=cv2.resize(left,(PANE_W,PANE_H))
            right=cv2.resize(right,(PANE_W,PANE_H))
            composed=np.hstack([left,right])

            cam_fps=1.0/(sum(in_times)/len(in_times)+1e-9)
            hud_header(composed,style,len(persons),last_deepx_ms,cam_fps,pipeline.fps,pipeline.ready)
            hud_left(left,len(persons),last_deepx_ms,cam_fps,style)
            hud_right(right,pipeline.fps,pipeline.ready,style,gen_count)
            draw_style_buttons(right,style)
            composed[:,:PANE_W]=left
            composed[:,PANE_W:]=right

            try: disp.stdin.write(composed.tobytes())
            except BrokenPipeError:
                print("[edge-art] Display pipe broken."); break

            # Auto style cycle (only if user hasn't touched recently)
            if time.time()-style_t0>STYLE_CYCLE_S:
                style_idx=(style_idx+1)%len(STYLE_KEYS)
                style=STYLE_KEYS[style_idx]; style_t0=time.time()
                placeholder=make_placeholder(style)
                print(f"[edge-art] auto style → {style}")

            n_frames+=1
            if n_frames%30==0:
                deepx_fps=1.0/(sum(npu_times)/len(npu_times)+1e-9)
                print(f"[edge-art] {n_frames:5d}fr  cam {cam_fps:.1f}fps  "
                      f"deepx {deepx_fps:.1f}fps/{last_deepx_ms:.0f}ms  "
                      f"genai {pipeline.fps:.2f}fps  gen#{gen_count}  people {len(persons)}")

    except KeyboardInterrupt:
        print("\n[edge-art] Ctrl-C")
    finally:
        print("[edge-art] Shutting down ...")
        for p in (cam,disp):
            try: p.terminate(); p.wait(timeout=3)
            except: p.kill()
        print(f"[edge-art] Done — {n_frames} frames.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
