import sys, time, json
sys.path.insert(0, '/data/local/tmp')
sys.path.insert(0, '/data/local/tmp/ort181')
import numpy as np, cv2
cv2.setNumThreads(0)
import easyocr_post as ep
import onnxruntime as ort

D = '/data/local/tmp/easyocr'
d = ort.InferenceSession(f'{D}/easyocr_detector.onnx', providers=['CPUExecutionProvider'])
r = ort.InferenceSession(f'{D}/easyocr_recognizer.onnx', providers=['CPUExecutionProvider'])
m = json.load(open(f'{D}/easyocr_meta.json'))

img = np.full((300, 900, 3), 255, np.uint8)
cv2.putText(img, 'Hello World', (40, 110), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (10, 10, 10), 5)
cv2.putText(img, 'EasyOCR 2026', (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (20, 20, 20), 4)

rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
canvas, ratio = ep.resize_aspect_ratio(rgb, 800)
x = ep.normalize_mean_variance(canvas).transpose(2, 0, 1)[None].astype(np.float32)
t = time.time(); y = d.run(None, {d.get_inputs()[0].name: x})[0]
print('det', round(time.time() - t, 2), 's', y.shape)
boxes = ep.adjust_result_coordinates(ep.get_det_boxes(y[0, :, :, 0], y[0, :, :, 1]), ratio)
print('boxes', len(boxes), flush=True)
for b in boxes:
    rect = np.array(b, np.float32)
    w = int(max(np.linalg.norm(rect[0]-rect[1]), np.linalg.norm(rect[2]-rect[3])))
    h = int(max(np.linalg.norm(rect[0]-rect[3]), np.linalg.norm(rect[1]-rect[2])))
    if w < 4 or h < 4:
        continue
    M = cv2.getPerspectiveTransform(rect, np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32))
    crop = cv2.warpPerspective(img, M, (w, h))
    inp = ep.recog_preprocess(crop, m['imgH']).astype(np.float32)
    preds = r.run(None, {r.get_inputs()[0].name: inp})[0][0]
    txt, conf = ep.ctc_greedy_decode(preds, m['character'])
    print(round(conf, 2), repr(txt))
