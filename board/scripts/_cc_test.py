import sys
sys.path.insert(0, '/data/local/tmp/ort181')
import cv2, numpy as np
print('cv2', cv2.__version__, flush=True)
a = (np.random.rand(144, 400) > 0.5).astype(np.uint8)
print('before-ort c4 nlab', cv2.connectedComponentsWithStats(a, connectivity=4)[0], flush=True)
import onnxruntime as ort
s = ort.InferenceSession('/data/local/tmp/easyocr/easyocr_detector.onnx', providers=['CPUExecutionProvider'])
x = np.random.rand(1, 3, 288, 800).astype(np.float32)
s.run(None, {s.get_inputs()[0].name: x})
print('ran detector', flush=True)
b = (a * 255).astype(np.uint8)
try:
    cnts, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print('findContours ok', len(cnts), flush=True)
except Exception as e:
    print('findContours ERR', e, flush=True)
try:
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cv2.dilate(b, k); print('dilate ok', flush=True)
except Exception as e:
    print('dilate ERR', e, flush=True)
try:
    pts = np.array([[10, 10], [50, 12], [48, 40], [8, 38]], np.float32)
    print('minAreaRect ok', cv2.minAreaRect(pts) is not None, flush=True)
except Exception as e:
    print('minAreaRect ERR', e, flush=True)
try:
    M = cv2.getPerspectiveTransform(pts, np.array([[0, 0], [40, 0], [40, 30], [0, 30]], np.float32))
    cv2.warpPerspective(np.zeros((50, 60, 3), np.uint8), M, (40, 30)); print('warp ok', flush=True)
except Exception as e:
    print('warp ERR', e, flush=True)
try:
    print('c8 nlab', cv2.connectedComponentsWithStats(a, connectivity=8)[0], flush=True)
except Exception as e:
    print('c8 ERR', e, flush=True)
