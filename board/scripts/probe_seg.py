import numpy as np, os, cv2, sys, time
from dx_engine import InferenceEngine, InferenceOption

opt = InferenceOption()
try: opt.set_buffer_count(4)
except: pass
sys.stderr.write('loading...\n'); sys.stderr.flush()
engine = InferenceEngine('/data/local/tmp/yolo26n-seg.dxnn', opt)
sys.stderr.write('loaded\n'); sys.stderr.flush()

if os.path.exists('/tmp/yp_frame1.bin'):
    raw = np.fromfile('/tmp/yp_frame1.bin', dtype=np.uint8).reshape(720,1280,3)
    src='real'
else:
    raw = np.random.randint(50,200,(720,1280,3),dtype=np.uint8); src='random'

size=640; h,w=raw.shape[:2]; r=min(size/w,size/h)
nw,nh=int(round(w*r)),int(round(h*r))
canvas=np.full((size,size,3),114,np.uint8)
canvas[(size-nh)//2:(size-nh)//2+nh,(size-nw)//2:(size-nw)//2+nw]=cv2.resize(raw,(nw,nh))

# uint8 BGR, exactly like working pose_worker
t0=time.time()
outs = engine.run(np.expand_dims(canvas, 0))
dt=(time.time()-t0)*1000
s = np.squeeze(outs[0])[:,4]
sys.stderr.write(f'RESULT src={src} inf={dt:.0f}ms max={s.max():.4f} n03={int((s>0.3).sum())} n01={int((s>0.1).sum())}\n')
sys.stderr.flush()
