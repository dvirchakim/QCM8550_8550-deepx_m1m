"""
Board GStreamer pipeline manager.

Wraps the real on-board DEEPX gst-launch pipeline as a subprocess. The
pipeline performs:

    qtiqmmfsrc cam=0/1 (NV12 1920x1080)
        -> dxpreprocess -> dxinfer (YOLOV5Pose640_1.dxnn) -> dxpostprocess
            -> tee
                ├── dxosd -> waylandsink     (full-screen visual layer)
                └── dxmsgconv -> dxmsgbroker (mqtt://127.0.0.1:1883)

Python listens to MQTT for the 17-keypoint pose JSON; the visual neon
overlay is composed inside the Qt UI on top of the camera feed grabbed
via a parallel OpenCV VideoCapture (or the same qtiqmmfsrc via appsink
once we wire it up).
"""
from __future__ import annotations

import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Optional

import config


@dataclass
class PipelineHandle:
    proc: Optional[subprocess.Popen]
    cmd: str
    started_at: float


def _is_board() -> bool:
    """Heuristic: are we running on the Yocto board (Linux + dx-stream dir)."""
    return sys.platform.startswith("linux") and os.path.isdir(config.BOARD_DXSTREAM_ROOT)


def _build_pipeline_cmd(camera_id: int) -> str:
    """
    Build a single-camera gst-launch string. We keep two independent
    subprocesses (one per camera) so a crash in one viewport does not kill
    both. This matches the original demo's intent while exposing pose data
    via MQTT instead of only on-screen overlay.
    """
    src   = config.POSE_CFG_DIR
    model = config.POSE_MODEL_DXNN
    return (
        f"gst-launch-1.0 -e "
        f"qtiqmmfsrc camera={camera_id} ! "
        f"qtivtransform ! "
        f"video/x-raw,width={config.CAPTURE_WIDTH},height={config.CAPTURE_HEIGHT},"
        f"format=NV12,framerate={config.CAPTURE_FPS}/1 ! "
        f"queue max-size-buffers=5 leaky=downstream ! "
        f"dxpreprocess  config-file-path={src}/preprocess_config.json ! "
        f"dxinfer       config-file-path={src}/inference_config.json ! "
        f"dxpostprocess config-file-path={src}/postprocess_config.json ! "
        f"tee name=pose_t "
        f"  pose_t. ! queue ! dxosd width=960 height=540 ! "
        f"          videoconvert ! waylandsink fullscreen=false sync=false "
        f"  pose_t. ! queue ! dxmsgconv config-file-path={config.MSGCONV_CFG} "
        f"                       cam-id={camera_id} ! "
        f"          dxmsgbroker proto-lib=/usr/share/dx-stream/lib/libdx_msgbroker_mqtt.so "
        f"                      conn-str={config.MQTT_HOST};{config.MQTT_PORT};"
        f"{config.MQTT_TOPIC_POSE}/{camera_id} "
        f"                      config-file-path={config.MSGBROKER_CFG}"
    )


class BoardPipeline:
    """Lifecycle manager for the two on-board GStreamer pipelines."""

    def __init__(self) -> None:
        self.handles: list[PipelineHandle] = []
        self.simulated = not _is_board()

    # ------------------------------------------------------------------
    def start(self) -> None:
        if self.simulated:
            print("[BoardPipeline] Not on board - simulation mode (no gst-launch)")
            return

        # The DeepX driver must be loaded; idempotent insmod
        try:
            subprocess.run(
                ["modprobe", "dxrt_driver"],
                check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            pass

        env = os.environ.copy()
        env.setdefault("XDG_RUNTIME_DIR", "/run/user/root/")
        env.setdefault("WAYLAND_DISPLAY", "wayland-1")
        env.setdefault("QT_QPA_PLATFORM", "wayland-egl")

        for cam_id in (config.CAMERA_1_INDEX, config.CAMERA_2_INDEX):
            cmd = _build_pipeline_cmd(cam_id)
            print(f"[BoardPipeline] Launching cam={cam_id}:")
            print("  " + cmd)
            proc = subprocess.Popen(
                shlex.split(cmd),
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid if hasattr(os, "setsid") else None,
            )
            self.handles.append(PipelineHandle(proc=proc, cmd=cmd, started_at=time.time()))

    # ------------------------------------------------------------------
    def alive(self) -> bool:
        if self.simulated:
            return True
        return all(h.proc is not None and h.proc.poll() is None for h in self.handles)

    # ------------------------------------------------------------------
    def stop(self) -> None:
        for h in self.handles:
            if h.proc is None or h.proc.poll() is not None:
                continue
            try:
                os.killpg(os.getpgid(h.proc.pid), signal.SIGINT)
            except (ProcessLookupError, AttributeError):
                h.proc.terminate()
            try:
                h.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                h.proc.kill()
        self.handles.clear()
