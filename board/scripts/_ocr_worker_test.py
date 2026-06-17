import subprocess, struct, sys, time, os
import numpy as np, cv2

CAM_W, CAM_H = 1280, 720
img = np.full((CAM_H, CAM_W, 3), 255, np.uint8)
cv2.putText(img, 'Hello World', (120, 300), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (10, 10, 10), 6)
cv2.putText(img, 'EasyOCR 2026', (120, 480), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (20, 20, 20), 6)

env = os.environ.copy()
env['PYTHONPATH'] = '/data/local/tmp/ort181'
p = subprocess.Popen(['python3', '-u', '/data/local/tmp/easyocr_worker.py'],
                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=sys.stderr, env=env, bufsize=0)

# wait for READY
line = p.stdout.readline()
print('handshake:', line.strip(), flush=True)

def read_exact(fp, n):
    b = bytearray()
    while len(b) < n:
        c = fp.read(n - len(b))
        if not c: return None
        b.extend(c)
    return bytes(b)

for i in range(3):
    t0 = time.time()
    p.stdin.write(b'\x01' + img.tobytes()); p.stdin.flush()
    tag = p.stdout.read(1)
    n = struct.unpack('<i', read_exact(p.stdout, 4))[0]
    txt = read_exact(p.stdout, n).decode('utf-8', 'replace')
    print(f'run {i}: {round(time.time()-t0,3)}s  text={txt!r}', flush=True)

p.stdin.write(b'\x00'); p.stdin.flush()
p.wait(timeout=5)
