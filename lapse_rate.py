#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 13:41:46 2025

@author: junsaito
"""
# lapse_rate.py
import numpy as np
from scipy.stats import linregress
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def calc_lapse_rate_block_numba(block_indices, temp2d, elev2d):
    results = []
    for (i, j) in block_indices:
        if i < 2 or j < 2 or i >= temp2d.shape[0]-2 or j >= temp2d.shape[1]-2:
            results.append((i, j, -6.5e-3))
        else:
            t_win = temp2d[i-2:i+3, j-2:j+3].flatten()
            e_win = elev2d[i-2:i+3, j-2:j+3].flatten()
            mean_t = np.mean(t_win)
            mean_e = np.mean(e_win)
            cov = 0.0
            var = 0.0
            for k in range(t_win.shape[0]):
                diff_e = e_win[k] - mean_e
                diff_t = t_win[k] - mean_t
                cov += diff_e * diff_t
                var += diff_e * diff_e
            if var == 0:
                slope = -6.5e-3
            else:
                slope = cov / var
                if slope < -9e-3:
                    slope = -9e-3
                elif slope > -4e-3:
                    slope = -4e-3
            results.append((i, j, slope))
    return results

def calc_lapse_rate_parallel(temp2d, elev2d, max_workers=14):
    H, W = temp2d.shape
    indices = [(i, j) for i in range(H) for j in range(W)]
    block_size = 1000
    lapse = np.full((H, W), -6.5e-3, dtype=np.float32)
    blocks = [indices[k:k+block_size] for k in range(0, len(indices), block_size)]
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(calc_lapse_rate_block_numba, np.array(blk), temp2d, elev2d) for blk in blocks]
        for fut in tqdm(futures, total=len(futures), desc="Calculating Lapse Rate"):
            for i, j, slope in fut.result():
                lapse[i, j] = slope
    return lapse


# 降水-標高 回帰から κ を推定

def calc_precip_kappa_block(block_indices, precip2d, elev2d):
    results = []
    for (i, j) in block_indices:
        if i < 2 or j < 2 or i >= precip2d.shape[0]-2 or j >= precip2d.shape[1]-2:
            results.append((i, j, 0.35))  # デフォルト値
            continue
        p_win = precip2d[i-2:i+3, j-2:j+3].flatten()
        e_win = elev2d[i-2:i+3, j-2:j+3].flatten()
        if np.std(e_win) == 0:
            kappa = 0.35
        else:
            slope, _, _, _, _ = linregress(e_win, p_win)
            # km単位に換算
            kappa = slope * 1000
            kappa = np.clip(kappa, 0.1, 1.0)  # 安定性のための制限
        results.append((i, j, kappa))
    return results

