"""
Edge-Art Agent - top-level orchestrator.

Responsibilities
----------------
* Start/stop the on-board GStreamer pipeline (BoardPipeline).
* Subscribe to pose data (MQTT or simulated).
* Pull camera frames - on the board we read directly from /dev/video for the
  Python compositor, on Windows we use whatever cv2.VideoCapture finds.
* Compose stylized overlay per camera (PoseRenderer @ 60 FPS).
* Drive the async Stylizer with the latest silhouette mask (1 Hz).
* Provide a `render_frames()` callback the UI polls.

The agent itself owns NO Qt - it is import-safe and unit-testable.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Optional

import cv2
import numpy as np

import config
from src.board_pipeline import BoardPipeline
from src.monitor        import HealthMonitor
from src.pose_mqtt      import PoseSource, make_source
from src.pose_renderer  import render_overlay, render_silhouette_mask
from src.stylizer       import Stylizer


def _open_camera(index: int) -> Optional[cv2.VideoCapture]:
    """Open a camera for the Python-side compositor.

    On the board the GStreamer pipeline already owns the qtiqmmfsrc handles
    for the visual layer; for the Python compositor we just want any frames
    matching the same camera index. On Windows / dev we use DSHOW for sane
    behaviour. Returns None if no camera is available - the UI then renders
    on a black panel with only the skeleton overlay.
    """
    api = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, api)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    return cap


class Agent:
    def __init__(self) -> None:
        print("[Agent] Initialising Edge-Art Interactive Silhouette Mural")

        self.monitor   = HealthMonitor()
        self.pipeline  = BoardPipeline()
        self.pose_src  : PoseSource = make_source()
        self.stylizer  = Stylizer()
        self.stylizer.start()

        # Cameras for the Python compositor (cosmetic; overlay still works
        # if these fail - skeleton paints on a dark canvas instead).
        self.cap0 = _open_camera(config.CAMERA_1_INDEX)
        self.cap1 = _open_camera(config.CAMERA_2_INDEX)

        self.active_style = config.STYLES[config.DEFAULT_STYLE_INDEX]["id"]
        self._tick = 0

    # ------------------------------------------------------------------
    def start_pipeline(self) -> None:
        try:
            self.pipeline.start()
        except Exception as exc:
            print(f"[Agent] pipeline start failed (continuing without): {exc}")

    # ------------------------------------------------------------------
    def on_style_change(self, style_id: str) -> None:
        print(f"[Agent] style -> {style_id}")
        self.active_style = style_id
        self.stylizer.set_style(style_id)

    # ------------------------------------------------------------------
    def _read(self, cap: Optional[cv2.VideoCapture]) -> Optional[np.ndarray]:
        if cap is None:
            return None
        ok, frame = cap.read()
        return frame if ok else None

    # ------------------------------------------------------------------
    def render_frames(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Return (cam0_view, cam1_view, generative_view) BGR images."""
        self._tick += 1

        f0 = self._read(self.cap0)
        f1 = self._read(self.cap1)

        persons0 = self.pose_src.latest(config.CAMERA_1_INDEX)
        persons1 = self.pose_src.latest(config.CAMERA_2_INDEX)

        view0 = render_overlay(f0, persons0, self.active_style, draw_bbox=False)
        view1 = render_overlay(f1, persons1, self.active_style, draw_bbox=False)

        # Push a fresh silhouette to the stylizer at ~1 Hz cadence (it
        # internally dedupes via SD_REFRESH_INTERVAL_S).
        all_persons = persons0 + persons1
        if all_persons:
            mask = render_silhouette_mask(
                all_persons, config.SD_OUTPUT_SIZE, config.SD_OUTPUT_SIZE,
            )
            self.stylizer.submit(mask)

        gen = self.stylizer.latest()
        return view0, view1, gen

    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        print("[Agent] shutting down")
        self.stylizer.stop()
        self.pose_src.stop()
        self.pipeline.stop()
        if self.cap0: self.cap0.release()
        if self.cap1: self.cap1.release()
