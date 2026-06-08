"""
DX-M1 smoke test - validates the heterogeneous AI pipeline works.

Loads YOLOV5Pose640_1.dxnn directly via dx_engine, runs N inferences,
reports latency. Proves PRD section 4.2 claim (sub-5ms NPU inference).

Run on board only:  python3 /tmp/dxm1_smoke.py
"""
import sys
import time

import numpy as np
from dx_engine import InferenceEngine, InferenceOption


MODEL = "/usr/share/dx-stream/dx_stream/samples/models/YOLOV5Pose640_1.dxnn"
RUNS  = 100


def main() -> int:
    print(f"[dxm1] Loading {MODEL}")
    t0 = time.time()
    opt = InferenceOption()
    eng = InferenceEngine(MODEL, opt)
    print(f"[dxm1] Loaded in {(time.time()-t0)*1000:.1f} ms")

    in_info  = eng.get_input_tensors_info()
    out_info = eng.get_output_tensors_info()
    print(f"[dxm1] is_ppu={eng.is_ppu()}  multi_input={eng.is_multi_input_model()}")
    print(f"[dxm1] inputs : {in_info}")
    print(f"[dxm1] outputs: {out_info}")

    # Build a synthetic input matching the first input tensor info
    info = in_info[0]
    shape = info["shape"]
    dtype = np.dtype(info["dtype"])
    print(f"[dxm1] synthetic input shape={shape} dtype={dtype}")
    x = (np.random.rand(*shape) * 255).astype(dtype)

    # Warmup
    for _ in range(5):
        eng.run(x)

    # Timed
    lat = []
    t0 = time.time()
    for _ in range(RUNS):
        ts = time.time()
        outs = eng.run(x)
        lat.append((time.time() - ts) * 1000)
    total = time.time() - t0

    lat = np.array(lat)
    print(f"[dxm1] {RUNS} runs in {total:.2f} s -> {RUNS/total:.1f} FPS")
    print(f"[dxm1] latency p50={np.percentile(lat,50):.2f} ms  "
          f"p95={np.percentile(lat,95):.2f} ms  "
          f"max={lat.max():.2f} ms  "
          f"min={lat.min():.2f} ms")
    print(f"[dxm1] dx_engine.inference_time={eng.inference_time()} us, "
          f"latency={eng.latency()} us")
    print(f"[dxm1] output shapes: {[o.shape for o in outs]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
