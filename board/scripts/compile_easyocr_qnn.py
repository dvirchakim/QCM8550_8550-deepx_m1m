#!/usr/bin/env python3
"""
PC-side: compile the EasyOCR CRAFT detector ONNX to a QNN context binary for
the Qualcomm QCS8550 HTP, via qai-hub (cloud compile). Keeps the recognizer +
post-processing on the CPU (they are fast on small crops).

Requires qai-hub configured (token in ~/.qai_hub/client.ini).

    python compile_easyocr_qnn.py <export_dir>

The detector input is deterministic for this demo: the camera is always
1280x720, and resize_aspect_ratio(square_size=800) -> canvas 480x800, so a
fixed (1,3,480,800) input is valid.
"""
import sys
import qai_hub as hub

DET_H, DET_W = 480, 800


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else '.'
    onnx_path = f'{d}/easyocr_detector.onnx'
    device = hub.Device('QCS8550 (Proxy)')

    print(f'Submitting compile job: {onnx_path} -> qnn_context_binary '
          f'for {device.name}  (input 1x3x{DET_H}x{DET_W})')
    # Pin QAIRT to match the board's runtime (qnn-net-run v2.41.0). A newer
    # context binary fails to load ("Create From Binary failure").
    qairt = sys.argv[2] if len(sys.argv) > 2 else '2.41'
    job = hub.submit_compile_job(
        model=onnx_path,
        device=device,
        input_specs={'image': ((1, 3, DET_H, DET_W), 'float32')},
        options=f'--target_runtime qnn_context_binary --qairt_version {qairt}',
        name='easyocr_craft_detector',
    )
    print('Compile job:', job.url, flush=True)
    target = job.get_target_model()          # blocks until compiled
    out = f'{d}/easyocr_detector_qcs8550.bin'
    target.download(out)
    print('Downloaded ->', out)


if __name__ == '__main__':
    main()
