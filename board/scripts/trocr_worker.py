#!/usr/bin/env python3
"""
TrOCR text-recognition worker — PERSISTENT.

Loads the TrOCR encoder + decoder ONCE (onnxruntime, CPU) and then serves
recognition requests over stdin/stdout. Previously this ran as a fresh
subprocess every ~8s, re-loading ~245 MB of ONNX each time; keeping it resident
removes that repeated load and makes the demo far more responsive.

Run:
    PYTHONPATH=/data/local/tmp/ort181 python3 trocr_worker.py

Binary protocol (little-endian), main <-> worker:
  worker -> main : b'READY\n' once both models are loaded
  main -> worker : 0x01 + (CAM_H*CAM_W*3) uint8 BGR frame  -> recognize
                   0x00                                      -> quit
  worker -> main : 0x01 + int32 text_len + text_len bytes (utf-8)

Model files (ONNX with external data):
    /data/local/tmp/trocr/encoder.onnx + encoder.data  (~92 MB)
    /data/local/tmp/trocr/decoder.onnx + decoder.data  (~153 MB)
"""
import json, os, signal, struct, sys, time
import numpy as np
import cv2

ORT_PATH   = '/data/local/tmp/ort181'
TROCR_DIR  = '/data/local/tmp/trocr'
ENCODER    = f'{TROCR_DIR}/encoder.onnx'
DECODER    = f'{TROCR_DIR}/decoder.onnx'
VOCAB_FILE = f'{TROCR_DIR}/vocab.json'

CAM_W, CAM_H = 1280, 720
ENC_W, ENC_H = 384, 384
START_ID     = 2          # decoder_start_token_id == eos_token_id == 2 (qai-hub)
EOS_ID       = 2
MAX_TOKENS   = 20
NUM_LAYERS   = 6
NUM_HEADS    = 8
KV_SEQ_SELF  = 19
FRAME_BYTES  = CAM_H * CAM_W * 3

# onnxruntime can emit logs on stdout; keep a private result fd and send real
# stdout to /dev/null so the binary channel stays clean.
_pipe_wfd = os.dup(1)
_devnull  = os.open('/dev/null', os.O_WRONLY)
os.dup2(_devnull, 1)
os.close(_devnull)
_out = os.fdopen(_pipe_wfd, 'wb', buffering=0)
_in  = sys.stdin.buffer

signal.signal(signal.SIGTERM, lambda *_: os._exit(0))


def _log(m):
    sys.stderr.write(f'[trocr] {m}\n'); sys.stderr.flush()


def read_exact(fp, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = fp.read(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


# ── PIL-style bicubic resize (transformers uses antialiased BICUBIC) ───────────
def _cubic(x, a=-0.5):
    x = np.abs(x); out = np.zeros_like(x)
    m1 = x < 1.0; m2 = (x >= 1.0) & (x < 2.0)
    out[m1] = ((a + 2.0) * x[m1] - (a + 3.0)) * x[m1] * x[m1] + 1.0
    out[m2] = (((x[m2] - 5.0) * x[m2] + 8.0) * x[m2] - 4.0) * a
    return out


def _axis_weights(in_size, out_size, support=2.0):
    scale = in_size / out_size; fscale = max(1.0, scale); rows = []
    for o in range(out_size):
        center = (o + 0.5) * scale
        lo = int(np.floor(center - support * fscale))
        hi = int(np.ceil(center + support * fscale))
        idx = np.arange(lo, hi)
        w = _cubic((idx - center + 0.5) / fscale)
        idx = np.clip(idx, 0, in_size - 1)
        s = w.sum()
        if s != 0: w = w / s
        rows.append((idx, w))
    return rows


def pil_bicubic_resize(img, out_w, out_h):
    h, w = img.shape[:2]
    x = img.astype(np.float32)
    tmp = np.zeros((h, out_w, x.shape[2]), np.float32)
    for o, (idx, wt) in enumerate(_axis_weights(w, out_w)):
        tmp[:, o, :] = np.tensordot(x[:, idx, :], wt, axes=([1], [0]))
    out = np.zeros((out_h, out_w, x.shape[2]), np.float32)
    for o, (idx, wt) in enumerate(_axis_weights(h, out_h)):
        out[o, :, :] = np.tensordot(tmp[idx, :, :], wt, axes=([0], [0]))
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def _load_vocab(path):
    try:
        with open(path) as f:
            data = json.load(f)
        vocab = data['model']['vocab'] if isinstance(data.get('model'), dict) else data
        return {int(v): k for k, v in vocab.items()}
    except Exception:
        return {}


def _bytes_to_text(tokens, id_to_tok):
    bs = list(range(ord('!'), ord('~')+1)) + list(range(ord('¡'), ord('¬')+1)) + list(range(ord('®'), ord('ÿ')+1))
    cs = bs[:]; n = 0
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


# ── Load models once ───────────────────────────────────────────────────────────
if ORT_PATH not in sys.path:
    sys.path.insert(0, ORT_PATH)
try:
    import onnxruntime as ort
except ImportError as e:
    _log(f'onnxruntime missing (PYTHONPATH={ORT_PATH}): {e}')
    os._exit(1)

id_to_tok = _load_vocab(VOCAB_FILE)
_opts = ort.SessionOptions()
_opts.inter_op_num_threads = 4
_opts.intra_op_num_threads = 4

try:
    _log('loading encoder ...')
    enc_sess = ort.InferenceSession(ENCODER, sess_options=_opts, providers=['CPUExecutionProvider'])
    _log('loading decoder ...')
    dec_sess = ort.InferenceSession(DECODER, sess_options=_opts, providers=['CPUExecutionProvider'])
except Exception as e:
    _log(f'model load failed: {e}')
    os._exit(1)

_enc_out_names = [o.name for o in enc_sess.get_outputs()]
_dec_out_names = [o.name for o in dec_sess.get_outputs()]
_dec_in_names  = [i.name for i in dec_sess.get_inputs()]
_enc_in_names  = [i.name for i in enc_sess.get_inputs()]
_log(f'ENC inputs : {_enc_in_names}')
_log(f'ENC outputs: {_enc_out_names}')
_log(f'DEC inputs : {_dec_in_names}')
_log(f'DEC outputs: {_dec_out_names}')
_log('models ready')
_out.write(b'READY\n')


def recognize(frame_bgr):
    # TrOCR-printed reads a single text LINE. Crop the central horizontal band
    # (where a user holds up printed text) so background doesn't trigger
    # language-model hallucination.
    h, w = frame_bgr.shape[:2]
    y0, y1 = int(h * 0.32), int(h * 0.68)
    band   = frame_bgr[y0:y1, :]
    rgb_full = cv2.cvtColor(band, cv2.COLOR_BGR2RGB)
    rgb      = pil_bicubic_resize(rgb_full, ENC_W, ENC_H).astype(np.float32) / 255.0
    # TrOCR image processor: normalize with mean=0.5, std=0.5 -> range [-1, 1]
    rgb      = (rgb - 0.5) / 0.5
    px_vals  = rgb.transpose(2, 0, 1)[np.newaxis]   # [1,3,384,384]

    enc_outs = enc_sess.run(None, {'pixel_values': px_vals})
    cross_kv = {}
    for i, name in enumerate(_enc_out_names):
        layer = name.split('_')[-1]
        key = 'key' if 'key' in name else 'val'
        cross_kv[f'kv_{layer}_cross_attn_{key}'] = enc_outs[i]

    self_kv = {f'kv_{l}_attn_{kv}': np.zeros((1, NUM_HEADS, KV_SEQ_SELF, 32), np.float32)
               for l in range(NUM_LAYERS) for kv in ('key', 'val')}

    generated, current_id = [], START_ID
    for step in range(MAX_TOKENS):
        feed = {'input_ids': np.array([[current_id]], np.int32),
                'index':     np.array([step], np.int32)}
        feed.update(self_kv); feed.update(cross_kv)
        dec_outs = dec_sess.run(None, feed)
        out_map  = dict(zip(_dec_out_names, dec_outs))
        nxt = int(out_map['next_token'].flat[0])
        if nxt == EOS_ID:
            break
        generated.append(nxt); current_id = nxt
        for l in range(NUM_LAYERS):
            for kv in ('key', 'val'):
                o = f'kv_cache_{kv}_{l}'
                if o in out_map:
                    self_kv[f'kv_{l}_attn_{kv}'] = out_map[o][:, :, 1:, :]

    return _bytes_to_text(generated, id_to_tok) if id_to_tok else str(generated)


# ── Serve requests ───────────────────────────────────────────────────────────
while True:
    cmd = _in.read(1)
    if not cmd or cmd == b'\x00':
        os._exit(0)
    if cmd != b'\x01':
        continue
    payload = read_exact(_in, FRAME_BYTES)
    if payload is None:
        os._exit(0)
    try:
        frame = np.frombuffer(payload, np.uint8).reshape(CAM_H, CAM_W, 3)
        t0 = time.time()
        text = recognize(frame)
        _log(f'"{text}" ({time.time()-t0:.1f}s)')
    except Exception as e:
        _log(f'recognize error: {e}')
        text = ''
    data = text.encode('utf-8')[:512]
    _out.write(b'\x01' + struct.pack('<i', len(data)) + data)
