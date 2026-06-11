#!/usr/bin/env python3
"""
TrOCR text recognition worker — runs once per invocation.

Usage:
    python3 trocr_worker.py <frame_bin> <output_txt>

Reads a 1280x720 BGR uint8 frame binary, runs TrOCR encoder+decoder
via snpe-net-run, writes recognized text to output_txt.

Model: encoder.dlc + decoder.dlc (QAIRT float32, run on CPU)
  encoder input : pixel_values [1,384,384,3] float32 RGB [0,1]
  encoder output: 12 cross-attention KV cache tensors
  decoder inputs: input_ids [1,1] int32, index [1] int32,
                  12 self-attn KV (grow each step), 12 cross-attn KV (fixed)
  decoder output: next_token [1] int32 + 12 updated self-attn KV
"""
import json, os, struct, subprocess, sys, tempfile, time
import numpy as np
import cv2

TROCR_DIR  = '/data/local/tmp/trocr'
ENCODER    = f'{TROCR_DIR}/encoder.dlc'
DECODER    = f'{TROCR_DIR}/decoder.dlc'
VOCAB_FILE = f'{TROCR_DIR}/vocab.json'
IO_DIR     = '/tmp/trocr_io'
CAM_W, CAM_H = 1280, 720
ENC_W, ENC_H = 384, 384
BOS_ID     = 1    # TrOCR uses decoder_start_token_id=1
EOS_ID     = 2
MAX_TOKENS = 20
NUM_LAYERS = 6
NUM_HEADS  = 8
KV_SEQ_CROSS = 578
KV_SEQ_SELF  = 19    # grows to 20 after step 0 — fixed-length KV cache

# ── Byte-level BPE decoder ─────────────────────────────────────────────────────

def _load_vocab(path):
    try:
        with open(path) as f:
            data = json.load(f)
        # Support both flat vocab.json {token: id} and tokenizer.json {model: {vocab: ...}}
        if 'model' in data:
            vocab = data['model'].get('vocab', {})
        else:
            vocab = data
        return {int(v): k for k, v in vocab.items()}   # id → token_str
    except Exception:
        return {}

def _bytes_to_text(tokens, id_to_tok):
    byte_decoder = {chr(i): i for i in range(256)}
    # GPT-2 byte-level BPE uses a specific character mapping
    bs = list(range(ord('!'), ord('~')+1)) + list(range(ord('¡'), ord('¬')+1)) + list(range(ord('®'), ord('ÿ')+1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    byte_decoder = {chr(c): b for b, c in zip(bs, cs)}

    raw = []
    for tid in tokens:
        tok = id_to_tok.get(tid, '')
        for c in tok:
            raw.append(byte_decoder.get(c, ord(c) if ord(c) < 256 else 63))
    try:
        return bytes(raw).decode('utf-8', errors='replace').strip()
    except Exception:
        return ''

# ── snpe-net-run helpers ───────────────────────────────────────────────────────

def _write_raw(path, arr):
    arr.astype(arr.dtype).tofile(path)

def _read_raw(path, dtype, shape):
    return np.fromfile(path, dtype=dtype).reshape(shape)

def _input_list(items):
    return ' '.join(f'{name}:={path}' for name, path in items)

def run_snpe(dlc, input_items, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    list_path = os.path.join(out_dir, 'input_list.txt')
    with open(list_path, 'w') as f:
        f.write(_input_list(input_items) + '\n')
    cmd = [
        'snpe-net-run',
        '--container', dlc,
        '--input_list', list_path,
        '--output_dir', out_dir,
    ]
    r = subprocess.run(cmd, capture_output=True, timeout=60)
    if r.returncode != 0:
        raise RuntimeError(f'snpe-net-run failed:\n{r.stderr.decode()[:500]}')

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 3:
        print(f'Usage: {sys.argv[0]} <frame.bin> <output.txt>')
        sys.exit(1)

    frame_path = sys.argv[1]
    out_path   = sys.argv[2]

    t_start = time.time()
    os.makedirs(IO_DIR, exist_ok=True)

    # Load vocab for decoding
    id_to_tok = _load_vocab(VOCAB_FILE)

    # Read and preprocess frame
    try:
        raw = np.fromfile(frame_path, dtype=np.uint8).reshape(CAM_H, CAM_W, 3)
    except Exception as e:
        print(f'[trocr] frame read failed: {e}', file=sys.stderr)
        open(out_path, 'w').write('')
        return

    # Preprocess: resize to 384×384, BGR→RGB, normalize [0,1]
    resized = cv2.resize(raw, (ENC_W, ENC_H), interpolation=cv2.INTER_AREA)
    rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    px_vals = rgb.reshape(1, ENC_H, ENC_W, 3)    # [1,384,384,3]

    # ── Encoder ───────────────────────────────────────────────────────────────
    enc_in  = os.path.join(IO_DIR, 'pixel_values.raw')
    enc_out = os.path.join(IO_DIR, 'enc_out')
    _write_raw(enc_in, px_vals)

    try:
        run_snpe(ENCODER, [('pixel_values', enc_in)], enc_out)
    except Exception as e:
        print(f'[trocr] encoder failed: {e}', file=sys.stderr)
        open(out_path, 'w').write('[encoder error]')
        return

    # Read cross-attention KV caches from encoder output
    # Shape: [1, 8, 578, 32] float32 for each of 12 tensors (6 layers × key+val)
    cross_kv = {}
    for layer in range(NUM_LAYERS):
        for kv in ('key', 'val'):
            name = f'kv_cache_{kv}_{layer}'
            fpath = os.path.join(enc_out, f'{name}.raw')
            cross_kv[f'kv_{layer}_cross_attn_{kv}'] = _read_raw(
                fpath, np.float32, (1, NUM_HEADS, KV_SEQ_CROSS, 32))

    # ── Decoder autoregressive loop ───────────────────────────────────────────
    dec_dir = os.path.join(IO_DIR, 'dec')
    os.makedirs(dec_dir, exist_ok=True)

    # Initialize self-attention KV cache (zeros, grows each step)
    self_kv = {}
    for layer in range(NUM_LAYERS):
        for kv in ('key', 'val'):
            # Start with shape [1, 8, 19, 32] (max self-attn seq len before step 0)
            self_kv[f'kv_{layer}_attn_{kv}'] = np.zeros(
                (1, NUM_HEADS, KV_SEQ_SELF, 32), np.float32)

    generated = []
    current_id = BOS_ID

    for step in range(MAX_TOKENS):
        dec_in_dir  = os.path.join(dec_dir, f'step_{step}_in')
        dec_out_dir = os.path.join(dec_dir, f'step_{step}_out')
        os.makedirs(dec_in_dir, exist_ok=True)

        input_items = []

        # input_ids and position index
        ids_path = os.path.join(dec_in_dir, 'input_ids.raw')
        idx_path = os.path.join(dec_in_dir, 'index.raw')
        np.array([[current_id]], dtype=np.int32).tofile(ids_path)
        np.array([step], dtype=np.int32).tofile(idx_path)
        input_items += [('input_ids', ids_path), ('index', idx_path)]

        # Self-attention KV caches
        for name, arr in self_kv.items():
            p = os.path.join(dec_in_dir, f'{name}.raw')
            _write_raw(p, arr)
            input_items.append((name, p))

        # Cross-attention KV caches (from encoder, fixed)
        for name, arr in cross_kv.items():
            p = os.path.join(dec_in_dir, f'{name}.raw')
            _write_raw(p, arr)
            input_items.append((name, p))

        try:
            run_snpe(DECODER, input_items, dec_out_dir)
        except Exception as e:
            print(f'[trocr] decoder step {step} failed: {e}', file=sys.stderr)
            break

        # Read next token
        next_tok_path = os.path.join(dec_out_dir, 'next_token.raw')
        try:
            next_token = int(np.fromfile(next_tok_path, dtype=np.int32)[0])
        except Exception:
            break

        if next_token == EOS_ID:
            break
        generated.append(next_token)
        current_id = next_token

        # Update self-attention KV cache from decoder output
        for layer in range(NUM_LAYERS):
            for kv in ('key', 'val'):
                name_out = f'kv_cache_{kv}_{layer}'
                name_in  = f'kv_{layer}_attn_{kv}'
                fpath = os.path.join(dec_out_dir, f'{name_out}.raw')
                try:
                    self_kv[name_in] = _read_raw(
                        fpath, np.float32, (1, NUM_HEADS, KV_SEQ_SELF + 1, 32))
                except Exception:
                    pass

    # Decode tokens to text
    text = _bytes_to_text(generated, id_to_tok) if id_to_tok else str(generated)
    elapsed = time.time() - t_start
    print(f'[trocr] "{text}" ({elapsed:.1f}s, {len(generated)} tokens)')

    with open(out_path, 'w') as f:
        f.write(text)


if __name__ == '__main__':
    main()
