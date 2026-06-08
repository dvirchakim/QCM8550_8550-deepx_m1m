"""
Async + pipelined DEEPX inference - target the PRD 60 FPS pose stream.
"""
import sys
import time
import numpy as np
from dx_engine import InferenceEngine, InferenceOption

MODEL    = "/usr/share/dx-stream/dx_stream/samples/models/YOLOV5Pose640_1.dxnn"
PIPELINE = 4    # in-flight requests
RUNS     = 200


def main() -> int:
    opt = InferenceOption()
    try:
        opt.set_buffer_count(PIPELINE)
    except Exception as exc:
        print(f"[async] buffer_count not supported: {exc}")

    eng = InferenceEngine(MODEL, opt)
    in_info = eng.get_input_tensors_info()[0]
    shape   = in_info["shape"]
    dtype   = np.dtype(in_info["dtype"])
    x       = (np.random.rand(*shape) * 255).astype(dtype)

    print(f"[async] model {MODEL}")
    print(f"[async] pipeline depth = {PIPELINE}")
    print(f"[async] warmup ...")
    for _ in range(5):
        eng.run(x)

    # ------------------------------------------------------------------
    # Async: submit `PIPELINE` jobs ahead, then waitfor each in order.
    # ------------------------------------------------------------------
    in_flight: list[int] = []
    lat = []
    t0 = time.time()
    submitted = 0

    while submitted < RUNS or in_flight:
        # Top up the pipeline
        while submitted < RUNS and len(in_flight) < PIPELINE:
            job_id = eng.run_async(x, user_arg=time.time())
            in_flight.append(job_id)
            submitted += 1

        # Reap the oldest job (FIFO; dx_engine returns out in submit order)
        job_id = in_flight.pop(0)
        _outs = eng.wait(job_id)
        # We use NPU-reported timing for accuracy
    total = time.time() - t0

    print(f"[async] {RUNS} jobs in {total:.2f} s -> {RUNS/total:.1f} FPS")
    print(f"[async] eng.latency           = {eng.latency()} us")
    try:
        print(f"[async] eng.inference_time    = {eng.inference_time()} us")
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Batch sync (alternative API)
    # ------------------------------------------------------------------
    batch = [x] * PIPELINE
    out_bufs = [[np.zeros(o["shape"], dtype=np.dtype(o["dtype"]))
                 for o in eng.get_output_tensors_info()]
                for _ in range(PIPELINE)]
    t0 = time.time()
    for _ in range(RUNS // PIPELINE):
        eng.run_batch(batch, out_bufs)
    total = time.time() - t0
    print(f"[batch] {RUNS} via batch={PIPELINE} in {total:.2f} s -> "
          f"{RUNS/total:.1f} FPS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
