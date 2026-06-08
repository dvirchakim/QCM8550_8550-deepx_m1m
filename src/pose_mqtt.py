"""
Pose data source.

Two implementations behind a single interface:

* `MqttPoseSource`  - subscribes to dxmsgbroker on `dxstream/pose/<cam>`.
                      Parses the dx-stream message envelope into a list of
                      Person records with 17 (x, y, conf) keypoints.

* `SimulatedPoseSource` - generates a moving "demo skeleton" for desktop
                          development on Windows where neither MQTT nor the
                          DEEPX runtime exist.

Both expose `.latest(cam_id) -> list[Person]` returning the most recent
pose frame for that camera. Old data older than `STALE_MS` is discarded.
"""
from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import config


STALE_MS = 500


@dataclass
class Keypoint:
    x: float       # normalized [0, 1] in the camera frame
    y: float
    conf: float


@dataclass
class Person:
    keypoints: list[Keypoint]
    bbox: tuple[float, float, float, float] = (0, 0, 0, 0)   # x, y, w, h normalized
    ts_ms: int = 0
    score: float = 1.0


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
class PoseSource:
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def latest(self, cam_id: int) -> list[Person]:
        return []


# ---------------------------------------------------------------------------
# MQTT implementation
# ---------------------------------------------------------------------------
class MqttPoseSource(PoseSource):
    """Subscribes to dx-stream broker. Tolerant of schema variations."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cache: dict[int, list[Person]] = {0: [], 1: []}
        self._client = None

    # ......................................................................
    def _parse_payload(self, payload: bytes) -> tuple[int, list[Person]]:
        """Parse a dx-stream MQTT payload. We accept multiple shapes."""
        msg = json.loads(payload)
        cam_id = int(msg.get("cam_id", msg.get("camera", 0)))
        objs = msg.get("objects") or msg.get("detections") or []
        persons: list[Person] = []
        for o in objs:
            kps_raw = o.get("keypoints") or o.get("pose") or []
            if len(kps_raw) < config.POSE_NUM_KEYPOINTS:
                continue
            kps: list[Keypoint] = []
            for k in kps_raw[: config.POSE_NUM_KEYPOINTS]:
                if isinstance(k, dict):
                    kps.append(Keypoint(float(k.get("x", 0)),
                                        float(k.get("y", 0)),
                                        float(k.get("conf", k.get("score", 1.0)))))
                elif isinstance(k, (list, tuple)) and len(k) >= 2:
                    kps.append(Keypoint(float(k[0]), float(k[1]),
                                        float(k[2] if len(k) > 2 else 1.0)))
            bb = o.get("bbox") or (0, 0, 0, 0)
            persons.append(Person(
                keypoints=kps,
                bbox=tuple(float(v) for v in bb[:4]),
                ts_ms=int(time.time() * 1000),
                score=float(o.get("score", 1.0)),
            ))
        return cam_id, persons

    # ......................................................................
    def _on_message(self, _client, _ud, message):  # paho callback
        try:
            cam_id, persons = self._parse_payload(message.payload)
        except Exception as exc:
            print(f"[MqttPoseSource] parse error: {exc}")
            return
        with self._lock:
            self._cache[cam_id] = persons

    # ......................................................................
    def start(self) -> None:
        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            raise RuntimeError("paho-mqtt not installed (pip install paho-mqtt)")

        self._client = mqtt.Client(client_id=f"edgeart-{int(time.time())}")
        if config.MQTT_USERNAME:
            self._client.username_pw_set(config.MQTT_USERNAME, config.MQTT_PASSWORD)
        if config.MQTT_USE_TLS:
            self._client.tls_set()
            self._client.tls_insecure_set(True)
        self._client.on_message = self._on_message
        self._client.connect(config.MQTT_HOST, config.MQTT_PORT, keepalive=30)
        self._client.subscribe(f"{config.MQTT_TOPIC_POSE}/#", qos=0)
        self._client.loop_start()
        print(f"[MqttPoseSource] subscribed to {config.MQTT_TOPIC_POSE}/#")

    def stop(self) -> None:
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()
            self._client = None

    def latest(self, cam_id: int) -> list[Person]:
        with self._lock:
            persons = self._cache.get(cam_id, [])
        now = int(time.time() * 1000)
        return [p for p in persons if now - p.ts_ms <= STALE_MS]


# ---------------------------------------------------------------------------
# Simulated implementation (Windows dev / fallback)
# ---------------------------------------------------------------------------
class SimulatedPoseSource(PoseSource):
    """Produces a stable, slowly-moving COCO skeleton in normalized coords."""

    def __init__(self) -> None:
        self._t0 = time.time()

    def start(self) -> None:
        print("[SimulatedPoseSource] running (no real DEEPX)")

    def stop(self) -> None:
        pass

    def _make_skeleton(self, phase: float) -> list[Keypoint]:
        # Anchor in the centre, slight breathing motion.
        cx, cy = 0.5 + 0.04 * math.sin(phase), 0.55
        # Approximate normalized COCO layout for a standing person
        # 0=nose 1=Leye 2=Reye 3=Lear 4=Rear 5=Lsh 6=Rsh 7=Lel 8=Rel 9=Lwr 10=Rwr
        # 11=Lhip 12=Rhip 13=Lkn 14=Rkn 15=Lank 16=Rank
        s = 0.18    # half shoulder width
        a = 0.05 * math.sin(phase * 1.7)  # arm swing
        layout = [
            (cx,        cy - 0.30),       # nose
            (cx - 0.02, cy - 0.32),       # L eye
            (cx + 0.02, cy - 0.32),       # R eye
            (cx - 0.04, cy - 0.30),       # L ear
            (cx + 0.04, cy - 0.30),       # R ear
            (cx - s,    cy - 0.18),       # L shoulder
            (cx + s,    cy - 0.18),       # R shoulder
            (cx - s - 0.03 + a, cy - 0.05),  # L elbow
            (cx + s + 0.03 - a, cy - 0.05),  # R elbow
            (cx - s - 0.06 + a, cy + 0.08),  # L wrist
            (cx + s + 0.06 - a, cy + 0.08),  # R wrist
            (cx - 0.10, cy + 0.05),       # L hip
            (cx + 0.10, cy + 0.05),       # R hip
            (cx - 0.10, cy + 0.20),       # L knee
            (cx + 0.10, cy + 0.20),       # R knee
            (cx - 0.10, cy + 0.38),       # L ankle
            (cx + 0.10, cy + 0.38),       # R ankle
        ]
        return [Keypoint(x, y, 0.9) for (x, y) in layout]

    def latest(self, cam_id: int) -> list[Person]:
        phase = (time.time() - self._t0) * 1.8 + cam_id * 0.7
        return [Person(
            keypoints=self._make_skeleton(phase),
            bbox=(0.3, 0.2, 0.4, 0.7),
            ts_ms=int(time.time() * 1000),
            score=1.0,
        )]


# ---------------------------------------------------------------------------
def make_source() -> PoseSource:
    """Factory - prefer MQTT on the board, else simulate."""
    import sys, os
    if sys.platform.startswith("linux") and os.path.isdir(config.BOARD_DXSTREAM_ROOT):
        try:
            src = MqttPoseSource()
            src.start()
            return src
        except Exception as exc:
            print(f"[pose_mqtt] MQTT unavailable ({exc}); falling back to sim")
    src = SimulatedPoseSource()
    src.start()
    return src
