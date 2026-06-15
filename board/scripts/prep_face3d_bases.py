#!/usr/bin/env python3
"""
PC-side tool: slice the 3DDFA_V2 BFM into the compact 68-keypoint bases the
board needs, so we never ship the full multi-MB pkl to the device.

Inputs  (download from github.com/cleardusk/3DDFA_V2/tree/master/configs):
    bfm_noneck_v3.pkl
    param_mean_std_62d_120x120.pkl
Output:
    face3d_bases.npz   (u_base, w_shp_base, w_exp_base, param_mean, param_std)

Usage:
    python prep_face3d_bases.py <configs_dir> <out_npz>
"""
import pickle
import sys
import numpy as np


def _load(fp):
    with open(fp, 'rb') as f:
        return pickle.load(f, encoding='latin1')


def main():
    cfg_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
    out_fp  = sys.argv[2] if len(sys.argv) > 2 else 'face3d_bases.npz'

    bfm = _load(f'{cfg_dir}/bfm_noneck_v3.pkl')
    ms  = _load(f'{cfg_dir}/param_mean_std_62d_120x120.pkl')

    u      = bfm['u'].astype(np.float32)
    w_shp  = bfm['w_shp'].astype(np.float32)[..., :40]
    w_exp  = bfm['w_exp'].astype(np.float32)[..., :10]
    keypts = bfm['keypoints'].astype(np.int64)

    u_base     = u[keypts].reshape(-1, 1)        # (204, 1)
    w_shp_base = w_shp[keypts]                    # (204, 40)
    w_exp_base = w_exp[keypts]                    # (204, 10)

    np.savez_compressed(
        out_fp,
        u_base=u_base.astype(np.float32),
        w_shp_base=w_shp_base.astype(np.float32),
        w_exp_base=w_exp_base.astype(np.float32),
        param_mean=ms['mean'].astype(np.float32),
        param_std=ms['std'].astype(np.float32),
    )
    print(f'wrote {out_fp}: u_base{u_base.shape} '
          f'w_shp_base{w_shp_base.shape} w_exp_base{w_exp_base.shape}')


if __name__ == '__main__':
    main()
