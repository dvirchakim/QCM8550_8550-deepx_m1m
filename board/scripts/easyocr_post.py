#!/usr/bin/env python3
"""
Vendored EasyOCR pre/post-processing (numpy + opencv only — no torch/scipy).

Ported from JaidedAI EasyOCR (MIT License):
  * imgproc.normalizeMeanVariance / resize_aspect_ratio
  * craft_utils.getDetBoxes_core / adjustResultCoordinates  (poly=False path)
  * utils.CTCLabelConverter.decode_greedy  (blank index 0)
  * recognition AlignCollate preprocessing  (imgH=64, normalize to [-1,1])

Used by easyocr_worker.py to run CRAFT + CRNN ONNX models on the board.
"""
import math
import cv2
import numpy as np

# ── Detector preprocessing ──────────────────────────────────────────────────
_MEAN = np.array([0.485 * 255.0, 0.456 * 255.0, 0.406 * 255.0], np.float32)
_STD  = np.array([0.229 * 255.0, 0.224 * 255.0, 0.225 * 255.0], np.float32)


def normalize_mean_variance(img_rgb):
    return (img_rgb.astype(np.float32) - _MEAN) / _STD


def resize_aspect_ratio(img, square_size, mag_ratio=1.0):
    """Resize so max side == square_size (capped), pad to multiple of 32."""
    h, w, c = img.shape
    target_size = mag_ratio * max(h, w)
    if target_size > square_size:
        target_size = square_size
    ratio = target_size / max(h, w)
    th, tw = int(h * ratio), int(w * ratio)
    proc = cv2.resize(img, (tw, th), interpolation=cv2.INTER_LINEAR)
    th32 = th + (32 - th % 32) if th % 32 else th
    tw32 = tw + (32 - tw % 32) if tw % 32 else tw
    canvas = np.zeros((th32, tw32, c), np.float32)
    canvas[:th, :tw] = proc
    return canvas, ratio


# ── Detector postprocessing (CRAFT getDetBoxes, poly=False) ───────────────────
def get_det_boxes(textmap, linkmap, text_threshold=0.7, link_threshold=0.4,
                  low_text=0.4):
    linkmap = linkmap.copy()
    textmap = textmap.copy()
    img_h, img_w = textmap.shape

    _, text_score = cv2.threshold(textmap, low_text, 1, 0)
    _, link_score = cv2.threshold(linkmap, link_threshold, 1, 0)
    text_score_comb = np.clip(text_score + link_score, 0, 1)
    # NOTE: connectivity=4 segfaults in this board's OpenCV build once
    # onnxruntime has run in-process; connectivity=8 is stable and the
    # grouping difference is negligible for text regions.
    nlab, labels, stats, _ = cv2.connectedComponentsWithStats(
        text_score_comb.astype(np.uint8), connectivity=8)

    det = []
    for k in range(1, nlab):
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10:
            continue
        if np.max(textmap[labels == k]) < text_threshold:
            continue

        segmap = np.zeros(textmap.shape, np.uint8)
        segmap[labels == k] = 255
        segmap[np.logical_and(link_score == 1, text_score == 0)] = 0  # remove link
        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
        sx, ex = max(0, x - niter), min(img_w, x + w + niter + 1)
        sy, ey = max(0, y - niter), min(img_h, y + h + niter + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter))
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)

        np_contours = (np.roll(np.array(np.where(segmap != 0)), 1, axis=0)
                       .transpose().reshape(-1, 2))
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)

        w_, h_ = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        box_ratio = max(w_, h_) / (min(w_, h_) + 1e-5)
        if abs(1 - box_ratio) <= 0.1:
            l, r = min(np_contours[:, 0]), max(np_contours[:, 0])
            t, b = min(np_contours[:, 1]), max(np_contours[:, 1])
            box = np.array([[l, t], [r, t], [r, b], [l, b]], np.float32)

        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - startidx, 0)
        det.append(box)
    return det


def adjust_result_coordinates(boxes, ratio, ratio_net=2):
    out = []
    for b in boxes:
        out.append(b * (1.0 / ratio) * ratio_net)
    return out


# ── Recognizer preprocessing (AlignCollate, imgH fixed, keep ratio) ───────────
def recog_preprocess(crop_bgr, img_h=64, img_w_max=512):
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    ratio = w / max(1, h)
    rw = min(img_w_max, max(1, int(math.ceil(img_h * ratio))))
    resized = cv2.resize(gray, (rw, img_h), interpolation=cv2.INTER_CUBIC)
    x = resized.astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5                       # -> [-1, 1]
    return x[np.newaxis, np.newaxis]          # [1,1,H,W]


# ── CTC greedy decode (blank == index 0) ──────────────────────────────────────
def ctc_greedy_decode(preds, character):
    """preds: [T, n_class] raw logits. character: list WITHOUT blank.
    Index 0 is blank; index i -> character[i-1]."""
    probs = _softmax(preds)
    idx = probs.argmax(axis=-1)                     # [T]
    keep = np.insert(idx[1:] != idx[:-1], 0, True)  # collapse repeats
    keep &= idx != 0                                # drop blank
    sel = idx[keep]
    chars = np.array([''] + list(character))
    return ''.join(chars[sel]), float(_conf(probs, idx))


def _softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def _conf(probs, idx):
    pm = probs.max(axis=-1)
    nz = pm[idx != 0]
    if nz.size == 0:
        return 0.0
    return float(nz.prod() ** (1.0 / len(nz)))      # geometric mean of char probs
