#!/usr/bin/env python3
"""
PC-side tool: export EasyOCR (CRAFT detector + CRNN recognizer) to ONNX and
dump the character set, so the board can run OCR on onnxruntime with the
vendored numpy/opencv post-processing (easyocr_post.py).

Run on a PC that has `pip install easyocr torch onnx`:
    python export_easyocr.py <out_dir>

Outputs:
    easyocr_detector.onnx     (dynamic NCHW, H/W multiples of 32)
    easyocr_recognizer.onnx   (1x1x64xW, dynamic width)
    easyocr_meta.json         {"character": "...", "imgH": 64}
"""
import json
import sys

import torch
import torch.nn as nn
import easyocr


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else '.'
    reader = easyocr.Reader(['en'], gpu=False, quantize=False,
                            detect_network='craft', recog_network='standard')

    # ── Detector (CRAFT): returns (y, feature); keep y = [b, H/2, W/2, 2] ──────
    det = reader.detector.eval()

    class DetWrap(nn.Module):
        def __init__(self, m):
            super().__init__(); self.m = m

        def forward(self, x):
            out = self.m(x)
            y = out[0] if isinstance(out, (tuple, list)) else out
            return y

    detw = DetWrap(det).eval()
    dummy = torch.randn(1, 3, 608, 800)
    torch.onnx.export(
        detw, dummy, f'{out}/easyocr_detector.onnx',
        input_names=['image'], output_names=['y'], opset_version=12,
        dynamic_axes={'image': {0: 'b', 2: 'h', 3: 'w'},
                      'y': {0: 'b', 1: 'h2', 2: 'w2'}})
    print('exported detector')

    # ── Recognizer (CRNN/VGG): forward(image, text); text unused for CTC ───────
    rec = reader.recognizer.eval()

    class RecWrap(nn.Module):
        def __init__(self, m):
            super().__init__(); self.m = m

        def forward(self, x):
            # Reimplemented without AdaptiveAvgPool2d((None,1)) so dynamic width
            # exports cleanly. mean over feature-height == adaptive pool to 1.
            f = self.m.FeatureExtraction(x)        # [b, C, H', W']
            f = f.mean(dim=2)                       # [b, C, W']  (avg over H')
            f = f.permute(0, 2, 1)                  # [b, W', C]
            c = self.m.SequenceModeling(f)
            return self.m.Prediction(c.contiguous())

    recw = RecWrap(rec).eval()
    dummy_r = torch.randn(1, 1, 64, 256)
    torch.onnx.export(
        recw, dummy_r, f'{out}/easyocr_recognizer.onnx',
        input_names=['image'], output_names=['preds'], opset_version=12,
        dynamic_axes={'image': {3: 'w'}, 'preds': {1: 't'}})
    print('exported recognizer')

    # ── Charset + config ───────────────────────────────────────────────────────
    character = reader.character           # full recog charset (no blank token)
    img_h = getattr(reader, 'model_height', 64) or 64
    with open(f'{out}/easyocr_meta.json', 'w', encoding='utf-8') as f:
        json.dump({'character': character, 'imgH': int(img_h)}, f, ensure_ascii=False)
    print(f'meta: imgH={img_h} n_char={len(character)}')


if __name__ == '__main__':
    main()
