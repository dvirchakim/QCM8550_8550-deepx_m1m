"""
Generative stylization stage.

Phase-1 implementation uses OpenCV filters keyed off the pose silhouette
mask to produce a stylized "mural" image at ~1 Hz. This is a placeholder
for the eventual QNN-backed Stable Diffusion + ControlNet pipeline
(PRD 4.3) - the public API is the same so the swap is one-file-change.

Public API:
    stylizer = Stylizer()
    stylizer.start()
    stylizer.set_style("neon")
    stylizer.submit(silhouette_mask)  # uint8 single-channel HxW
    rgba = stylizer.latest()          # may be None until first result
    stylizer.stop()
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import cv2
import numpy as np

import config


def _palette(style_id: str) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
    """Return (bg, mid, fg) BGR colors for a style."""
    if style_id == "neon":
        return (10, 0, 30), (180, 0, 255), (0, 229, 255)
    if style_id == "vangogh":
        return (20, 40, 80), (40, 120, 220), (60, 200, 255)
    if style_id == "comic":
        return (250, 250, 230), (50, 50, 200), (0, 0, 0)
    if style_id == "noir":
        return (15, 15, 15), (80, 80, 80), (240, 240, 240)
    return (0, 0, 0), (128, 128, 128), (255, 255, 255)


def _stylize_neon(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    bg, mid, fg = _palette("neon")
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[:] = bg

    # Concentric glow halos
    soft = cv2.GaussianBlur(mask, (51, 51), 0)
    halo = cv2.GaussianBlur(mask, (21, 21), 0)

    out[..., 0] = np.clip(out[..., 0].astype(int) + (soft.astype(int) * mid[0] // 255), 0, 255)
    out[..., 1] = np.clip(out[..., 1].astype(int) + (soft.astype(int) * mid[1] // 255), 0, 255)
    out[..., 2] = np.clip(out[..., 2].astype(int) + (soft.astype(int) * mid[2] // 255), 0, 255)

    bone = cv2.cvtColor(halo, cv2.COLOR_GRAY2BGR)
    bone = (bone.astype(int) * np.array(fg, dtype=int) // 255).astype(np.uint8)
    return cv2.addWeighted(out, 1.0, bone, 0.9, 0)


def _stylize_vangogh(mask: np.ndarray) -> np.ndarray:
    bg, mid, fg = _palette("vangogh")
    h, w = mask.shape
    base = np.full((h, w, 3), bg, dtype=np.uint8)
    # Swirling texture
    noise = np.random.randint(0, 60, (h, w, 3), dtype=np.uint8)
    base = cv2.add(base, noise)
    base = cv2.GaussianBlur(base, (9, 9), 0)
    # Paint body in mid + fg edges
    body = cv2.dilate(mask, np.ones((9, 9), np.uint8), iterations=2)
    edges = cv2.Canny(mask, 30, 90)
    base[body > 0] = mid
    base[edges > 0] = fg
    # Brush-like blur
    return cv2.medianBlur(base, 7)


def _stylize_comic(mask: np.ndarray) -> np.ndarray:
    bg, mid, fg = _palette("comic")
    h, w = mask.shape
    out = np.full((h, w, 3), bg, dtype=np.uint8)
    # Halftone dots
    grid = np.zeros((h, w), dtype=np.uint8)
    step = 6
    for y in range(0, h, step):
        for x in range((y // step) % 2 * step // 2, w, step):
            grid[y, x] = 255
    halftone = cv2.bitwise_and(grid, cv2.dilate(mask, np.ones((7, 7), np.uint8)))
    out[halftone > 0] = mid
    # Bold ink outline
    thick = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    out[thick > 0] = fg
    return out


def _stylize_noir(mask: np.ndarray) -> np.ndarray:
    bg, mid, fg = _palette("noir")
    h, w = mask.shape
    out = np.full((h, w, 3), bg, dtype=np.uint8)
    # Rain streaks
    rain = np.random.randint(0, 255, (h, w), dtype=np.uint8)
    rain = (rain > 245).astype(np.uint8) * 80
    rain = cv2.warpAffine(rain, np.float32([[1, 0.5, 0], [0, 1, 0]]), (w, h))
    out[..., :] = cv2.add(out[..., :], cv2.cvtColor(rain, cv2.COLOR_GRAY2BGR))
    # White silhouette with hard edges
    body = cv2.dilate(mask, np.ones((7, 7), np.uint8))
    out[body > 0] = fg
    edges = cv2.Canny(body, 50, 150)
    out[edges > 0] = (0, 0, 0)
    return out


_STYLIZERS = {
    "neon":    _stylize_neon,
    "vangogh": _stylize_vangogh,
    "comic":   _stylize_comic,
    "noir":    _stylize_noir,
}


class Stylizer:
    """Background-thread stylizer. One-shot inference per submitted mask."""

    def __init__(self) -> None:
        self._lock      = threading.Lock()
        self._latest    : Optional[np.ndarray] = None
        self._pending   : Optional[np.ndarray] = None
        self._style_id  = config.STYLES[config.DEFAULT_STYLE_INDEX]["id"]
        self._stop_evt  = threading.Event()
        self._thread    : Optional[threading.Thread] = None
        self._last_run  = 0.0
        self._gen_count = 0

    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="stylizer")
        self._thread.start()

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None

    def set_style(self, style_id: str) -> None:
        if style_id in _STYLIZERS:
            with self._lock:
                self._style_id = style_id

    def submit(self, mask: np.ndarray) -> None:
        with self._lock:
            self._pending = mask

    def latest(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._latest is None else self._latest.copy()

    @property
    def generations(self) -> int:
        return self._gen_count

    # ------------------------------------------------------------------
    def _loop(self) -> None:
        while not self._stop_evt.is_set():
            now = time.time()
            if now - self._last_run < config.SD_REFRESH_INTERVAL_S:
                time.sleep(0.05)
                continue
            with self._lock:
                mask = None if self._pending is None else self._pending.copy()
                style = self._style_id
            if mask is None or mask.sum() == 0:
                time.sleep(0.05)
                continue
            try:
                img = _STYLIZERS[style](mask)
                if img.shape[0] != config.SD_OUTPUT_SIZE or img.shape[1] != config.SD_OUTPUT_SIZE:
                    img = cv2.resize(img, (config.SD_OUTPUT_SIZE, config.SD_OUTPUT_SIZE))
            except Exception as exc:
                print(f"[Stylizer] error: {exc}")
                time.sleep(0.1)
                continue
            with self._lock:
                self._latest = img
            self._gen_count += 1
            self._last_run = now
