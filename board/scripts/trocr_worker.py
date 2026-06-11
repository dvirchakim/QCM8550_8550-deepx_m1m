#!/usr/bin/env python3
"""
TrOCR text recognition worker — runs once per invocation.

Usage:
    PYTHONPATH=/data/local/tmp/ort python3 trocr_worker.py <frame_bin> <output_txt>

Reads a 1280x720 BGR uint8 frame binary, runs TrOCR encoder+decoder via
onnxruntime (CPU), writes recognized text to output_txt.

Model files (ONNX with external data):
    /data/local/tmp/trocr/encoder.onnx  + encoder.data  (~92 MB)
    /data/local/tmp/trocr/decoder.onnx  + decoder.data  (~153 MB)

Encoder input : pixel_values [1,3,384,384] float32 RGB [0,1]  (NCHW)
Encoder output: kv_cache_key/val_0..5  each [1,8,578,32]
Decoder inputs: input_ids [1,1] int32, index [1] int32,
                kv_{0-5}_attn_key/val [1,8,19,32] (self-attn, fixed window),
                kv_{0-5}_cross_attn_key/val [1,8,578,32] (cross-attn, fixed)
Decoder output: next_token [1] int32, kv_cache_key/val_0..5 [1,8,20,32]
"""
import json, os, sys, time
import numpy as np
import cv2

ORT_PATH   = '/data/local/tmp/ort181'
TROCR_DIR  = '/data/local/tmp/trocr'
ENCODER    = f'{TROCR_DIR}/encoder.onnx'
DECODER    = f'{TROCR_DIR}/decoder.onnx'
VOCAB_FILE = f'{TROCR_DIR}/vocab.json'

CAM_W, CAM_H  = 1280, 720
ENC_W, ENC_H  = 384, 384
BOS_ID        = 1
EOS_ID        = 2
MAX_TOKENS    = 20
NUM_LAYERS    = 6
NUM_HEADS     = 8
KV_SEQ_CROSS  = 578
KV_SEQ_SELF   = 19    # fixed self-attn window (decoder pads/slides)


def _setup_ort_path():
    if ORT_PATH not in sys.path:
        sys.path.insert(0, ORT_PATH)


def _load_vocab(path):
    try:
        with open(path) as f:
            data = json.load(f)
        # tokenizer.json has nested {'model': {'vocab': {...}}}
        if 'model' in data and isinstance(data['model'], dict):
            vocab = data['model'].get('vocab', {})
        else:
            vocab = data   # flat {token_str: token_id}
        return {int(v): k for k, v in vocab.items()}
    except Exception:
        return {}


def _bytes_to_text(tokens, id_to_tok):
    bs = list(range(ord('!'), ord('~')+1)) + list(range(ord('¡'), ord('¬')+1)) + list(range(ord('®'), ord('ÿ')+1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    byte_decoder = {chr(c): b for b, c in zip(bs, cs)}
    raw = []
    for tid in tokens:
        for ch in id_to_tok.get(tid, ''):
            raw.append(byte_decoder.get(ch, ord(ch) if ord(ch) < 256 else 63))
    try:
        return bytes(raw).decode('utf-8', errors='replace').strip()
    except Exception:
        return ''


def main():
    if len(sys.argv) < 3:
        print(f'Usage: {sys.argv[0]} <frame.bin> <output.txt>')
        sys.exit(1)

    frame_path = sys.argv[1]
    out_path   = sys.argv[2]

    _setup_ort_path()
    try:
        import onnxruntime as ort
    except ImportError as e:
        msg = f'[trocr] onnxruntime not found — set PYTHONPATH={ORT_PATH} ({e})'
        print(msg, file=sys.stderr)
        open(out_path, 'w').write(msg[:120])
        return

    t_start = time.time()

    id_to_tok = _load_vocab(VOCAB_FILE)

    try:
        raw = np.fromfile(frame_path, dtype=np.uint8).reshape(CAM_H, CAM_W, 3)
    except Exception as e:
        print(f'[trocr] frame read failed: {e}', file=sys.stderr)
        open(out_path, 'w').write('')
        return

    # Preprocess: resize 384×384, BGR→RGB, normalize, NHWC→NCHW [1,3,384,384]
    resized = cv2.resize(raw, (ENC_W, ENC_H), interpolation=cv2.INTER_AREA)
    rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    px_vals = rgb.transpose(2, 0, 1)[np.newaxis]   # [1,3,384,384]

    # ── Encoder ───────────────────────────────────────────────────────────────
    print('[trocr] loading encoder ...', flush=True)
    try:
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 4
        opts.intra_op_num_threads = 4
        enc_sess = ort.InferenceSession(ENCODER, sess_options=opts,
                                        providers=['CPUExecutionProvider'])
    except Exception as e:
        msg = f'[trocr] encoder load failed: {e}'
        print(msg, file=sys.stderr)
        open(out_path, 'w').write(msg[:120])
        return

    print('[trocr] running encoder ...', flush=True)
    try:
        enc_outs = enc_sess.run(None, {'pixel_values': px_vals})
    except Exception as e:
        msg = f'[trocr] encoder inference failed: {e}'
        print(msg, file=sys.stderr)
        open(out_path, 'w').write(msg[:120])
        return

    enc_out_names = [o.name for o in enc_sess.get_outputs()]
    cross_kv = {}
    for i, name in enumerate(enc_out_names):
        # names: kv_cache_key_0..5 / kv_cache_val_0..5
        # map to decoder input names: kv_{L}_cross_attn_{key|val}
        if 'key' in name:
            layer = name.split('_')[-1]
            cross_kv[f'kv_{layer}_cross_attn_key'] = enc_outs[i]
        else:
            layer = name.split('_')[-1]
            cross_kv[f'kv_{layer}_cross_attn_val'] = enc_outs[i]

    t_enc = time.time() - t_start
    print(f'[trocr] encoder done in {t_enc:.1f}s', flush=True)

    # ── Decoder ───────────────────────────────────────────────────────────────
    print('[trocr] loading decoder ...', flush=True)
    try:
        dec_sess = ort.InferenceSession(DECODER, sess_options=opts,
                                        providers=['CPUExecutionProvider'])
    except Exception as e:
        msg = f'[trocr] decoder load failed: {e}'
        print(msg, file=sys.stderr)
        open(out_path, 'w').write(msg[:120])
        return

    # Initialize self-attn KV cache (zeros)
    self_kv = {}
    for layer in range(NUM_LAYERS):
        for kv in ('key', 'val'):
            self_kv[f'kv_{layer}_attn_{kv}'] = np.zeros(
                (1, NUM_HEADS, KV_SEQ_SELF, 32), np.float32)

    generated   = []
    current_id  = BOS_ID

    for step in range(MAX_TOKENS):
        feed = {
            'input_ids': np.array([[current_id]], dtype=np.int32),
            'index':     np.array([step],         dtype=np.int32),
        }
        feed.update(self_kv)
        feed.update(cross_kv)

        try:
            dec_outs = dec_sess.run(None, feed)
        except Exception as e:
            print(f'[trocr] decoder step {step} failed: {e}', file=sys.stderr)
            break

        dec_names  = [o.name for o in dec_sess.get_outputs()]
        out_map    = dict(zip(dec_names, dec_outs))

        next_token = int(out_map['next_token'].flat[0])
        if next_token == EOS_ID:
            break
        generated.append(next_token)
        current_id = next_token

        # Update self-attn KV cache: output [1,8,20,32] → drop oldest → [1,8,19,32]
        for layer in range(NUM_LAYERS):
            for kv in ('key', 'val'):
                out_name = f'kv_cache_{kv}_{layer}'
                if out_name in out_map:
                    self_kv[f'kv_{layer}_attn_{kv}'] = out_map[out_name][:, :, 1:, :]

    text    = _bytes_to_text(generated, id_to_tok) if id_to_tok else str(generated)
    elapsed = time.time() - t_start
    print(f'[trocr] "{text}" ({elapsed:.1f}s, {len(generated)} tokens)', flush=True)

    with open(out_path, 'w') as f:
        f.write(text)


if __name__ == '__main__':
    main()
