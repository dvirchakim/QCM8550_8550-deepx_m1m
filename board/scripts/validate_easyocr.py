#!/usr/bin/env python3
"""
PC-side validation: run the exported ONNX models + vendored post-processing
and compare against EasyOCR's own readtext() on a synthetic test image.

    python validate_easyocr.py <export_dir>
"""
import json
import sys

import cv2
import numpy as np
import onnxruntime as ort

import easyocr_post as ep


def make_test_image():
    img = np.full((300, 900, 3), 255, np.uint8)
    cv2.putText(img, 'Hello World', (40, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 2.2, (10, 10, 10), 5, cv2.LINE_AA)
    cv2.putText(img, 'EasyOCR 2026', (40, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (20, 20, 20), 4, cv2.LINE_AA)
    return img


def run_pipeline(det_sess, rec_sess, character, img_bgr,
                 square_size=800, img_h=64):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    canvas, ratio = ep.resize_aspect_ratio(rgb, square_size)
    x = ep.normalize_mean_variance(canvas).transpose(2, 0, 1)[None]
    y = det_sess.run(None, {det_sess.get_inputs()[0].name: x.astype(np.float32)})[0]
    score_text = y[0, :, :, 0]
    score_link = y[0, :, :, 1]
    boxes = ep.get_det_boxes(score_text, score_link)
    boxes = ep.adjust_result_coordinates(boxes, ratio)

    results = []
    for box in boxes:
        rect = np.array(box, np.float32)
        w = int(max(np.linalg.norm(rect[0] - rect[1]),
                    np.linalg.norm(rect[2] - rect[3])))
        h = int(max(np.linalg.norm(rect[0] - rect[3]),
                    np.linalg.norm(rect[1] - rect[2])))
        if w < 4 or h < 4:
            continue
        dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
        M = cv2.getPerspectiveTransform(rect, dst)
        crop = cv2.warpPerspective(img_bgr, M, (w, h))
        inp = ep.recog_preprocess(crop, img_h)
        preds = rec_sess.run(None, {rec_sess.get_inputs()[0].name:
                                    inp.astype(np.float32)})[0][0]
        text, conf = ep.ctc_greedy_decode(preds, character)
        if text.strip():
            results.append((box, text, conf))
    return results


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else '.'
    meta = json.load(open(f'{d}/easyocr_meta.json', encoding='utf-8'))
    character, img_h = meta['character'], meta['imgH']

    det_sess = ort.InferenceSession(f'{d}/easyocr_detector.onnx',
                                    providers=['CPUExecutionProvider'])
    rec_sess = ort.InferenceSession(f'{d}/easyocr_recognizer.onnx',
                                    providers=['CPUExecutionProvider'])

    img = make_test_image()
    print('=== ONNX + vendored post-processing ===')
    for box, text, conf in run_pipeline(det_sess, rec_sess, character, img, img_h=img_h):
        cx, cy = box[0]
        print(f'  [{cx:4.0f},{cy:4.0f}]  {conf:.2f}  "{text}"')

    print('=== EasyOCR reference (readtext) ===')
    import easyocr
    reader = easyocr.Reader(['en'], gpu=False, quantize=False)
    for (box, text, conf) in reader.readtext(img):
        print(f'  {conf:.2f}  "{text}"')


if __name__ == '__main__':
    main()
