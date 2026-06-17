import os, subprocess, sys, time, glob
sys.path.insert(0, '/data/local/tmp')
sys.path.insert(0, '/data/local/tmp/ort181')
import numpy as np, cv2
cv2.setNumThreads(0)
import easyocr_post as ep
import onnxruntime as ort

D = '/data/local/tmp/easyocr'
BIN = f'{D}/easyocr_detector_qcs8550.bin'
BACKEND = '/usr/lib/libQnnHtp.so'
IO = '/tmp/qnn_det_io'
OUT = '/tmp/qnn_det_out'
os.makedirs(IO, exist_ok=True); os.makedirs(OUT, exist_ok=True)

# 1280x720 synthetic scene with text
img = np.full((720, 1280, 3), 255, np.uint8)
cv2.putText(img, 'Hello World', (120, 300), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (10, 10, 10), 6)
cv2.putText(img, 'EasyOCR 2026', (120, 480), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (20, 20, 20), 6)

rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
canvas, ratio = ep.resize_aspect_ratio(rgb, 800)
print('canvas', canvas.shape, 'ratio', round(ratio, 4), flush=True)
norm = ep.normalize_mean_variance(canvas)            # HWC float32

# ── CPU ONNX reference ───────────────────────────────────────────────────────
sess = ort.InferenceSession(f'{D}/easyocr_detector.onnx', providers=['CPUExecutionProvider'])
xc = norm.transpose(2, 0, 1)[None].astype(np.float32)
yc = sess.run(None, {sess.get_inputs()[0].name: xc})[0]
print('cpu out', yc.shape, yc.dtype, 'text[min,max]', round(float(yc[0,:,:,0].min()),3), round(float(yc[0,:,:,0].max()),3), flush=True)

# ── QNN HTP run (NCHW input named 'image') ───────────────────────────────────
xc.astype(np.float32).tofile(f'{IO}/image.bin')
with open(f'{IO}/input_list.txt', 'w') as f:
    f.write(f'image:={IO}/image.bin\n')
for fn in glob.glob(f'{OUT}/Result_0/*'):
    os.unlink(fn)
cmd = ['qnn-net-run', f'--retrieve_context={BIN}', f'--backend={BACKEND}',
       f'--input_list={IO}/input_list.txt', f'--output_dir={OUT}',
       '--use_native_input_files', '--use_native_output_files']
t0 = time.time()
r = subprocess.run(cmd, capture_output=True, timeout=120)
dt = time.time() - t0
print('qnn rc', r.returncode, 'time', round(dt, 3), 's', flush=True)
if r.returncode != 0:
    print('STDERR', r.stderr.decode(errors='replace')[-1500:]); sys.exit(1)

res = glob.glob(f'{OUT}/Result_0/*')
print('result files', [os.path.basename(x) for x in res], flush=True)
for fp in res:
    raw32 = np.fromfile(fp, np.float32)
    print(' ', os.path.basename(fp), 'float32 count', raw32.size, flush=True)
    yq = raw32.reshape(yc.shape)
    # correlation with CPU
    a, b = yc.ravel(), yq.ravel()
    corr = float(np.corrcoef(a, b)[0, 1])
    print('   corr vs CPU', round(corr, 4), flush=True)
    boxes = ep.adjust_result_coordinates(ep.get_det_boxes(np.ascontiguousarray(yq[0,:,:,0]), np.ascontiguousarray(yq[0,:,:,1])), ratio)
    print('   boxes', len(boxes), flush=True)
