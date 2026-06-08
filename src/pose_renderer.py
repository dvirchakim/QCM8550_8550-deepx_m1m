"""
Pose overlay renderer.

Given a list of `Person` records and a chosen style preset, paints a
stylized skeleton onto a frame (or a transparent canvas if no frame).

This is the deterministic 60 FPS visual layer of the demo - it runs on
every UI tick using only OpenCV draw calls, never the NPU.
"""
from __future__ import annotations

import cv2
import numpy as np

import config
from src.pose_mqtt import Person


# Style-specific draw parameters
_STYLE_DRAW = {
    "neon":    dict(bone_color=(255,   0, 255), glow=(0, 229, 255), thick=6, joint_r=8, glow_blur=21),
    "vangogh": dict(bone_color=( 50, 130, 255), glow=(255, 200, 60), thick=10, joint_r=10, glow_blur=31),
    "comic":   dict(bone_color=(  0,   0,   0), glow=(50, 180, 255), thick=8, joint_r=10, glow_blur=0),
    "noir":    dict(bone_color=(255, 255, 255), glow=(120, 120, 120), thick=4, joint_r=6, glow_blur=11),
}


def _denorm(kp_x: float, kp_y: float, w: int, h: int) -> tuple[int, int]:
    """Map normalized [0,1] keypoint to pixel coords. Tolerant of px-already values."""
    if kp_x <= 1.5 and kp_y <= 1.5:    # normalized
        return int(kp_x * w), int(kp_y * h)
    return int(kp_x), int(kp_y)


def render_overlay(
    frame: np.ndarray,
    persons: list[Person],
    style_id: str = "neon",
    draw_bbox: bool = False,
) -> np.ndarray:
    """
    Compose a stylized skeleton overlay on top of `frame` (BGR).
    Returns a new BGR image. If `frame` is None, paints onto a dark canvas
    sized to the configured viewport.
    """
    if frame is None:
        h, w = 540, 960
        out = np.zeros((h, w, 3), dtype=np.uint8)
        out[:] = (10, 14, 26)         # COLOR_PANEL_BG
    else:
        out = frame.copy()
        h, w = out.shape[:2]

    params = _STYLE_DRAW.get(style_id, _STYLE_DRAW["neon"])

    # ---- Glow layer (drawn first, blurred, then composited) -----------
    if params["glow_blur"] > 0:
        glow = np.zeros_like(out)
        for p in persons:
            _draw_skeleton(glow, p, params["glow"], params["thick"] + 6,
                            params["joint_r"] + 4, w, h)
        glow = cv2.GaussianBlur(glow, (params["glow_blur"], params["glow_blur"]), 0)
        out = cv2.addWeighted(out, 1.0, glow, 0.9, 0)

    # ---- Bone layer ---------------------------------------------------
    for p in persons:
        _draw_skeleton(out, p, params["bone_color"], params["thick"],
                       params["joint_r"], w, h)
        if draw_bbox:
            x, y, bw, bh = p.bbox
            if bw > 0 and bh > 0:
                p1 = _denorm(x, y, w, h)
                p2 = _denorm(x + bw, y + bh, w, h)
                cv2.rectangle(out, p1, p2, params["glow"], 2)

    return out


def _draw_skeleton(img, person: Person, color, thick: int, joint_r: int, w: int, h: int) -> None:
    kps = person.keypoints
    if not kps:
        return

    pts = [_denorm(k.x, k.y, w, h) if k.conf >= config.POSE_CONF_THRESHOLD else None
           for k in kps]

    # Bones
    for a, b in config.SKELETON_EDGES:
        if a < len(pts) and b < len(pts) and pts[a] and pts[b]:
            cv2.line(img, pts[a], pts[b], color, thick, lineType=cv2.LINE_AA)

    # Joints
    for pt in pts:
        if pt is not None:
            cv2.circle(img, pt, joint_r, color, -1, lineType=cv2.LINE_AA)
            cv2.circle(img, pt, joint_r + 2, (255, 255, 255), 1, lineType=cv2.LINE_AA)


def render_silhouette_mask(
    persons: list[Person],
    width: int,
    height: int,
) -> np.ndarray:
    """
    Build a single-channel mask of the skeleton outline. Used as the
    ControlNet conditioning input for the generative pass (PRD 4.3).
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    for p in persons:
        kps = p.keypoints
        pts = [_denorm(k.x, k.y, width, height) if k.conf >= config.POSE_CONF_THRESHOLD else None
               for k in kps]
        for a, b in config.SKELETON_EDGES:
            if a < len(pts) and b < len(pts) and pts[a] and pts[b]:
                cv2.line(mask, pts[a], pts[b], 255, 6, lineType=cv2.LINE_AA)
        for pt in pts:
            if pt is not None:
                cv2.circle(mask, pt, 4, 255, -1)
    return mask
