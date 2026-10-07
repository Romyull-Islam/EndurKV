#!/usr/bin/env python3
"""Host-side KV eviction-policy simulator over .attn.bin attention sidecars.
Compares published baselines (StreamingLLM, H2O, SnapKV, PyramidKV, CAKE, TOVA, ...)
and EndurKV variants on mean KL, avg cache slots used, throughput proxy and DRAM bytes.
Writes policy_results.csv, pareto_summary.csv and plots to <log_dir>_eviction_v2/.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WORKSPACE = Path(os.environ.get("WORKSPACE", r"D:/Research/EndurKV_workspace"))


# attn.bin parser
def load_attn_full(path: Path, return_per_head: bool = False):
    """Read an "ATTN" (head-averaged) or "ATNH" (per-head) sidecar.
    Returns (attn [steps, layers, (heads,) max_kv], n_kv_at [steps])."""
    if not path.exists():
        return None, None
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic == b"ATTN":
            n_heads_per_block = 1
            v2 = False
        elif magic == b"ATNH":
            v2 = True
        else:
            return None, None
        n_steps, n_layers, n_head = struct.unpack("<III", f.read(12))
        if v2:
            n_heads_per_block = max(1, n_head)
        per_step_layer = []
        n_kv_at = []
        for _s in range(n_steps):
            layers = []
            last_nkv = 0
            for _l in range(n_layers):
                (n_kv,) = struct.unpack("<I", f.read(4))
                if n_kv == 0:
                    layers.append(None); continue
                last_nkv = max(last_nkv, n_kv)
                count = n_kv * n_heads_per_block
                vals = np.frombuffer(f.read(4 * count), dtype=np.float32)
                if v2:
                    # reshape per-head: [n_head, n_kv]
                    layers.append(vals.reshape(n_heads_per_block, n_kv))
                else:
                    layers.append(vals)
            per_step_layer.append(layers)
            n_kv_at.append(last_nkv)
    if not per_step_layer:
        return None, None
    max_kv = max(n_kv_at) if n_kv_at else 0

    if return_per_head and v2:
        attn_ph = np.zeros((n_steps, n_layers, n_heads_per_block, max_kv), dtype=np.float32)
        for s, layers in enumerate(per_step_layer):
            for l, arr in enumerate(layers):
                if arr is not None:
                    attn_ph[s, l, :, :arr.shape[1]] = arr
        return attn_ph, np.array(n_kv_at, dtype=np.int32)

    # head-averaged path (works for both v1 and v2)
    attn = np.zeros((n_steps, n_layers, max_kv), dtype=np.float32)
    for s, layers in enumerate(per_step_layer):
        for l, arr in enumerate(layers):
            if arr is None:
                continue
            if arr.ndim == 2:  # v2 per-head: average over heads
                attn[s, l, :arr.shape[1]] = arr.mean(axis=0)
            else:
                attn[s, l, :len(arr)] = arr
    return attn, np.array(n_kv_at, dtype=np.int32)


# KL helper
def kl(p_full, p_evicted, eps=1e-12):
    p_full = np.clip(p_full, eps, 1.0)
    p_evicted = np.clip(p_evicted, eps, 1.0)
    return float(np.sum(p_full * (np.log(p_full) - np.log(p_evicted))))


# policies
# Each returns a boolean mask of shape (n_kv,) plus the actual K used.

def m_full(n_kv, K, **kw):
    m = np.ones(n_kv, dtype=bool); return m, int(m.sum())

def m_local(n_kv, K, **kw):
    m = np.zeros(n_kv, dtype=bool); s = max(0, n_kv - K); m[s:n_kv] = True
    return m, int(m.sum())

def m_streamingllm(n_kv, K, **kw):
    if n_kv <= K:
        m = np.ones(n_kv, dtype=bool); return m, int(m.sum())
    n_sink = min(4, K // 2); K_rec = K - n_sink
    m = np.zeros(n_kv, dtype=bool)
    m[:n_sink] = True; m[max(0, n_kv - K_rec):n_kv] = True
    return m, int(m.sum())

def _hh_norec_mask(n_kv, K, accum, recent_frac=0.5):
    """Top heavy hitters by accum + bottom recent_frac fraction recent."""
    if n_kv <= K:
        m = np.ones(n_kv, dtype=bool); return m, int(m.sum())
    K_rec = max(1, int(round(K * recent_frac)))
    K_hh = K - K_rec
    rec_start = max(0, n_kv - K_rec)
    cand = accum[:rec_start]
    if len(cand) <= K_hh:
        hh_idx = np.arange(rec_start)
    else:
        hh_idx = np.argpartition(-cand, K_hh)[:K_hh]
    m = np.zeros(n_kv, dtype=bool)
    m[hh_idx] = True; m[rec_start:n_kv] = True
    return m, int(m.sum())

def m_h2o(n_kv, K, accum, **kw):
    return _hh_norec_mask(n_kv, K, accum, recent_frac=0.5)

def m_h2o_norec(n_kv, K, accum, **kw):
    return _hh_norec_mask(n_kv, K, accum, recent_frac=1.0/16)

def m_snapkv(n_kv, K, snapkv_score, **kw):
    """SnapKV (Li 2024), selection frozen after prefill. Keeps a recent window
    plus top heavy hitters by the pooled, layer-summed score from simulate()."""
    if n_kv <= K:
        m = np.ones(n_kv, dtype=bool); return m, int(m.sum())
    L_obs = 16
    K_rec = min(L_obs, max(1, K // 4))   # observation window as the "recent"
    K_hh = K - K_rec
    rec_start = max(0, n_kv - K_rec)
    cand = snapkv_score[:rec_start] if snapkv_score is not None else np.zeros(rec_start)
    if len(cand) <= K_hh:
        hh_idx = np.arange(rec_start)
    else:
        hh_idx = np.argpartition(-cand, K_hh)[:K_hh]
    m = np.zeros(n_kv, dtype=bool)
    m[hh_idx] = True; m[rec_start:n_kv] = True
    return m, int(m.sum())

def m_endurkv_tova_spread(n_kv, K, attn_full, **kw):
    """TOVA scoring (top-K_t by current attention) with the attention-spread
    gate K_t = K * (1.3 - 0.6 * max_attn_norm). Avg cache is about K."""
    if n_kv <= K:
        m = np.ones(n_kv, dtype=bool); return m, int(m.sum())
    if attn_full is None or len(attn_full) == 0:
        return m_tova(n_kv, K, attn_full)
    max_a = float(attn_full[:n_kv].max())
    norm = max(0.0, min(1.0, (max_a - 0.4) / 0.4))
    mult = 1.3 - 0.6 * norm
    K_t = max(1, min(n_kv, int(round(K * mult))))
    idx = np.argpartition(-attn_full[:n_kv], K_t)[:K_t]
    m = np.zeros(n_kv, dtype=bool); m[idx] = True
    return m, int(m.sum())


# 2025-26 baselines

def m_lwkd(n_kv, K, attn_full, lwkd_state, layer_idx, **kw):
    """LWKD / SAGE-KV (arXiv 2503.08879): one-shot top-K by the first decode step's
    attention, then frozen. Newly appended decode tokens are always kept."""
    if n_kv <= K:
        m = np.ones(n_kv, dtype=bool); return m, int(m.sum())
    frozen = lwkd_state["frozen"][layer_idx]
    if frozen is None:
        # First call: snapshot top-K by current attention
        idx = np.argpartition(-attn_full[:n_kv], K)[:K]
        m = np.zeros(n_kv, dtype=bool); m[idx] = True
        lwkd_state["frozen"][layer_idx] = m.copy()
        return m, int(m.sum())
    # Subsequent calls: extend frozen mask with all newly appended decode tokens
    L = len(frozen)
    if n_kv <= L:
        return frozen[:n_kv].copy(), int(frozen[:n_kv].sum())
    m = np.zeros(n_kv, dtype=bool)
    m[:L] = frozen
    m[L:n_kv] = True
    return m, int(m.sum())


def m_ahakv(n_kv, K, recent_accum_l, **kw):
    """AhaKV (arXiv 2506.03762), recent-window accumulation part only (no logits or V
    in .attn.bin). Score is attention summed over the last 16 steps, K/8 kept recent."""
    if n_kv <= K:
        m = np.ones(n_kv, dtype=bool); return m, int(m.sum())
    K_rec = max(1, K // 8)
    K_hh = K - K_rec
    rec_start = max(0, n_kv - K_rec)
    cand = recent_accum_l[:rec_start] if recent_accum_l is not None else np.zeros(rec_start)
    if len(cand) <= K_hh:
        hh_idx = np.arange(rec_start)
    else:
        hh_idx = np.argpartition(-cand, K_hh)[:K_hh]
    m = np.zeros(n_kv, dtype=bool)
    m[hh_idx] = True; m[rec_start:n_kv] = True
    return m, int(m.sum())


# EndurKV-Evict v3

def m_endurkv_v3(n_kv, K, attn_full, attn_ewma_l, cross_layer_attn,
                 layer_idx, **kw):
    """v1 max-attention gate K_t = K * (1.3 - 0.6 * max_a_norm) with the v2 score
    0.55*current + 0.25*cross_layer + 0.20*ewma."""
    if n_kv <= K:
        m = np.ones(n_kv, dtype=bool); return m, int(m.sum())
    a = attn_full[:n_kv]
    # v1 gate
    max_a = float(a.max())
    norm = max(0.0, min(1.0, (max_a - 0.4) / 0.4))
    mult = 1.3 - 0.6 * norm
    K_t = max(1, min(n_kv, int(round(K * mult))))
    # v2 composite scoring
    cl = (cross_layer_attn[:n_kv]
          if cross_layer_attn is not None and len(cross_layer_attn) >= n_kv
          else np.zeros(n_kv))
    ewma = (attn_ewma_l[:n_kv]
            if attn_ewma_l is not None and len(attn_ewma_l) >= n_kv
            else a)
    score = 0.55 * a + 0.25 * cl + 0.20 * ewma
    idx = np.argpartition(-score, K_t)[:K_t]
    m = np.zeros(n_kv, dtype=bool); m[idx] = True
    return m, int(m.sum())


def m_lazyeviction(n_kv, K, attn_full, accum, lazy_mri_l, lazy_last_high_l,
                   lazy_current_step, lazy_window, **kw):
    """LazyEviction (arXiv 2506.15969): evict only every W steps, protect tokens whose
    recurrence interval (MRI) predicts reuse, fill the rest by cumulative attention."""
    if n_kv <= K:
        m = np.ones(n_kv, dtype=bool); return m, int(m.sum())

    # Between eviction windows, keep top-(1.25 K) by accum to leave room for re-entry.
    if lazy_current_step % lazy_window != 0 and lazy_current_step > 0:
        K_buf = min(n_kv, int(round(K * 1.25)))
        idx = np.argpartition(-accum, K_buf - 1)[:K_buf]
        m = np.zeros(n_kv, dtype=bool); m[idx] = True
        return m, int(m.sum())

    # At a window boundary, a token is predicted to recur if time since its
    # last high attention is below max(MRI, W).
    time_since = lazy_current_step - lazy_last_high_l
    mri_safe = np.maximum(lazy_mri_l, lazy_window)
    predicted_to_recur = (time_since < mri_safe) & (lazy_last_high_l >= 0)

    n_prot = int(predicted_to_recur.sum())
    if n_prot >= K:
        # Too many protected tokens, keep the most recently high ones.
        scores = -time_since.astype(np.float64)
        scores[~predicted_to_recur] = -1e18
        idx = np.argpartition(-scores, K - 1)[:K]
        m = np.zeros(n_kv, dtype=bool); m[idx] = True
        return m, int(m.sum())
    m = predicted_to_recur.copy()
    # Fill the remaining quota by cumulative attention.
    K_rem = K - n_prot
    if K_rem > 0:
        cand = accum.copy()
        cand[predicted_to_recur] = -1e18
        avail_idx = np.where(~predicted_to_recur)[0]
        if len(avail_idx) <= K_rem:
            m[avail_idx] = True
        else:
            pick = np.argpartition(-cand, K_rem - 1)[:K_rem]
            m[pick] = True
    return m, int(m.sum())


def m_endurkv_v6(n_kv, K, attn_full, accum, lazy_mri_l, lazy_last_high_l,
                 lazy_current_step, lazy_window, **kw):
    """EndurKV-Evict v6: v1 spread gate, score = current_attn * (1 + 0.5 * bonus), where
    the LazyEviction recurrence (MRI) signal gives a soft bonus, not hard protection."""
    if n_kv <= K:
        m = np.ones(n_kv, dtype=bool); return m, int(m.sum())
    a = attn_full[:n_kv]

    # v1 spread gate
    max_a = float(a.max())
    norm = max(0.0, min(1.0, (max_a - 0.4) / 0.4))
    mult = 1.3 - 0.6 * norm
    K_t = max(1, min(n_kv, int(round(K * mult))))

    # Recurrence bonus in [0, 1], near 1 when time_since << MRI, decays to 0 past it.
    time_since = lazy_current_step - lazy_last_high_l
    mri_safe = np.maximum(lazy_mri_l, 1).astype(np.float64)
    bonus = np.where(
        lazy_last_high_l >= 0,
        np.exp(-time_since.astype(np.float64) / mri_safe),
        0.0,
    )

    score = a * (1.0 + 0.5 * bonus)

    idx = np.argpartition(-score, K_t)[:K_t]
    m = np.zeros(n_kv, dtype=bool); m[idx] = True
    return m, int(m.sum())


# Experimental policies: frequency-domain, derivative and optimal-transport scores

def _history_matrix(attn_history_l, n_kv, W):
    """W x n_kv matrix of recent attention, newest in the last row, zero-padded."""
    H = np.zeros((W, n_kv), dtype=np.float64)
    if attn_history_l is None or len(attn_history_l) == 0:
        return H
    recent = attn_history_l[-W:]
    start = W - len(recent)
    for k, h in enumerate(recent):
        if h is None or len(h) == 0:
            continue
        L = min(len(h), n_kv)
        H[start + k, :L] = h[:L]
    return H


def m_endurkv_spectral(n_kv, K, attn_full, attn_history_l, **kw):
    """Score by low-frequency FFT energy (DC + 2 harmonics) of each token's
    attention history, blended 60/40 with current attention."""
    if n_kv <= K:
        m = np.ones(n_kv, dtype=bool); return m, int(m.sum())
    a = attn_full[:n_kv]
    W = 8
    H = _history_matrix(attn_history_l, n_kv, W)
    H[-1, :] = np.maximum(H[-1, :], a)
    fft_H = np.fft.rfft(H, axis=0)
    n_freqs = min(3, fft_H.shape[0])
    energy = (np.abs(fft_H[:n_freqs, :]) ** 2).sum(axis=0)
    energy_max = energy.max()
    if energy_max > 1e-12:
        energy = energy / energy_max
    score = 0.60 * a + 0.40 * energy
    idx = np.argpartition(-score, K)[:K]
    m = np.zeros(n_kv, dtype=bool); m[idx] = True
    return m, int(m.sum())


def m_endurkv_differential(n_kv, K, attn_full, attn_history_l, **kw):
    """Score by Taylor-predicted attention a + 0.30*max(da/dt, 0) + 0.15*max(d2a/dt2, 0),
    so tokens with rising attention are boosted."""
    if n_kv <= K:
        m = np.ones(n_kv, dtype=bool); return m, int(m.sum())
    a = attn_full[:n_kv]
    W = 4
    H = _history_matrix(attn_history_l, n_kv, W)
    H[-1, :] = a  # current step
    # backward finite differences
    d1 = H[-1, :] - H[-2, :] if H.shape[0] >= 2 else np.zeros(n_kv)
    d2 = (H[-1, :] - 2 * H[-2, :] + H[-3, :]) if H.shape[0] >= 3 else np.zeros(n_kv)
    score = a + 0.30 * np.maximum(d1, 0) + 0.15 * np.maximum(d2, 0)
    idx = np.argpartition(-score, K)[:K]
    m = np.zeros(n_kv, dtype=bool); m[idx] = True
    return m, int(m.sum())


def m_endurkv_wasserstein(n_kv, K, attn_full, **kw):
    """Keep K positions at equal-mass quantiles of the attention CDF (1D Wasserstein-1
    optimum), which spreads the budget across peaks instead of crowding one."""
    if n_kv <= K:
        m = np.ones(n_kv, dtype=bool); return m, int(m.sum())
    a = attn_full[:n_kv]
    total = a.sum()
    if total <= 1e-9:
        idx = np.argpartition(-a, K)[:K]
        m = np.zeros(n_kv, dtype=bool); m[idx] = True
        return m, int(m.sum())
    cum = np.cumsum(a) / total
    # K equal-mass quantile positions
    targets = (np.arange(K) + 0.5) / K
    chosen = np.searchsorted(cum, targets, side="left")
    chosen = np.clip(chosen, 0, n_kv - 1)
    chosen = np.unique(chosen)
    # searchsorted can return duplicates, fill the gap by top attention
    if len(chosen) < K:
        kept_mask = np.zeros(n_kv, dtype=bool); kept_mask[chosen] = True
        remaining = np.where(~kept_mask)[0]
        if len(remaining) > 0:
            need = K - len(chosen)
            top_rem = remaining[np.argpartition(-a[remaining], min(need - 1, len(remaining) - 1))[:need]]
            chosen = np.concatenate([chosen, top_rem])
    m = np.zeros(n_kv, dtype=bool); m[chosen] = True
    return m, int(m.sum())


def m_endurkv_v1_spectral(n_kv, K, attn_full, attn_history_l, **kw):
    """v1 spread gate + spectral scoring."""
    if n_kv <= K:
        m = np.ones(n_kv, dtype=bool); return m, int(m.sum())
    a = attn_full[:n_kv]
    max_a = float(a.max())
    norm = max(0.0, min(1.0, (max_a - 0.4) / 0.4))
    mult = 1.3 - 0.6 * norm
    K_t = max(1, min(n_kv, int(round(K * mult))))
    W = 8
    H = _history_matrix(attn_history_l, n_kv, W)
    H[-1, :] = np.maximum(H[-1, :], a)
    fft_H = np.fft.rfft(H, axis=0)
    n_freqs = min(3, fft_H.shape[0])
    energy = (np.abs(fft_H[:n_freqs, :]) ** 2).sum(axis=0)
    energy_max = energy.max()
    if energy_max > 1e-12:
        energy = energy / energy_max
    score = 0.60 * a + 0.40 * energy
    idx = np.argpartition(-score, K_t)[:K_t]
    m = np.zeros(n_kv, dtype=bool); m[idx] = True
    return m, int(m.sum())


def m_endurkv_v1_differential(n_kv, K, attn_full, attn_history_l, **kw):
    """v1 spread gate + differential (Taylor-predicted) scoring."""
    if n_kv <= K:
        m = np.ones(n_kv, dtype=bool); return m, int(m.sum())
    a = attn_full[:n_kv]
    max_a = float(a.max())
    norm = max(0.0, min(1.0, (max_a - 0.4) / 0.4))
    mult = 1.3 - 0.6 * norm
    K_t = max(1, min(n_kv, int(round(K * mult))))
    W = 4
    H = _history_matrix(attn_history_l, n_kv, W)
    H[-1, :] = a
    d1 = H[-1, :] - H[-2, :] if H.shape[0] >= 2 else np.zeros(n_kv)
    d2 = (H[-1, :] - 2 * H[-2, :] + H[-3, :]) if H.shape[0] >= 3 else np.zeros(n_kv)
    score = a + 0.30 * np.maximum(d1, 0) + 0.15 * np.maximum(d2, 0)
    idx = np.argpartition(-score, K_t)[:K_t]
    m = np.zeros(n_kv, dtype=bool); m[idx] = True
    return m, int(m.sum())


def m_endurkv_v1_wasserstein(n_kv, K, attn_full, **kw):
    """v1 spread gate + Wasserstein-quantile selection."""
    if n_kv <= K:
        m = np.ones(n_kv, dtype=bool); return m, int(m.sum())
    a = attn_full[:n_kv]
    max_a = float(a.max())
    norm = max(0.0, min(1.0, (max_a - 0.4) / 0.4))
    mult = 1.3 - 0.6 * norm
    K_t = max(1, min(n_kv, int(round(K * mult))))
    total = a.sum()
    if total <= 1e-9:
        idx = np.argpartition(-a, K_t)[:K_t]
        m = np.zeros(n_kv, dtype=bool); m[idx] = True
        return m, int(m.sum())
    cum = np.cumsum(a) / total
    targets = (np.arange(K_t) + 0.5) / K_t
    chosen = np.searchsorted(cum, targets, side="left")
    chosen = np.clip(chosen, 0, n_kv - 1)
    chosen = np.unique(chosen)
    if len(chosen) < K_t:
        kept_mask = np.zeros(n_kv, dtype=bool); kept_mask[chosen] = True
        remaining = np.where(~kept_mask)[0]
        if len(remaining) > 0:
            need = K_t - len(chosen)
            top_rem = remaining[np.argpartition(-a[remaining], min(need - 1, len(remaining) - 1))[:need]]
            chosen = np.concatenate([chosen, top_rem])
    m = np.zeros(n_kv, dtype=bool); m[chosen] = True
    return m, int(m.sum())


def m_endurkv_v1_spec_diff(n_kv, K, attn_full, attn_history_l, **kw):
    """v1 gate + spectral + differential composite."""
    if n_kv <= K:
        m = np.ones(n_kv, dtype=bool); return m, int(m.sum())
    a = attn_full[:n_kv]
    max_a = float(a.max())
    norm = max(0.0, min(1.0, (max_a - 0.4) / 0.4))
    mult = 1.3 - 0.6 * norm
    K_t = max(1, min(n_kv, int(round(K * mult))))
    W = 8
    H = _history_matrix(attn_history_l, n_kv, W)
    H[-1, :] = np.maximum(H[-1, :], a)
    fft_H = np.fft.rfft(H, axis=0)
    n_freqs = min(3, fft_H.shape[0])
    energy = (np.abs(fft_H[:n_freqs, :]) ** 2).sum(axis=0)
    energy_max = energy.max()
    if energy_max > 1e-12:
        energy = energy / energy_max
    d1 = H[-1, :] - H[-2, :] if H.shape[0] >= 2 else np.zeros(n_kv)
    score = 0.50 * a + 0.30 * energy + 0.20 * np.maximum(d1, 0)
    idx = np.argpartition(-score, K_t)[:K_t]
    m = np.zeros(n_kv, dtype=bool); m[idx] = True
    return m, int(m.sum())


def m_endurkv_v1_svd(n_kv, K, attn_full, attn_history_l, **kw):
    """v1 gate + SVD participation score sum_k sigma_k * |V[k, i]| over the top-r
    right singular vectors of the attention history H, blended 50/50 with current."""
    if n_kv <= K:
        m = np.ones(n_kv, dtype=bool); return m, int(m.sum())
    a = attn_full[:n_kv]
    max_a = float(a.max())
    norm = max(0.0, min(1.0, (max_a - 0.4) / 0.4))
    mult = 1.3 - 0.6 * norm
    K_t = max(1, min(n_kv, int(round(K * mult))))

    W = 8
    H = _history_matrix(attn_history_l, n_kv, W)
    H[-1, :] = np.maximum(H[-1, :], a)

    # compact SVD, r = min(W, n_kv)
    try:
        _, s, vt = np.linalg.svd(H, full_matrices=False)
    except np.linalg.LinAlgError:
        # fall back to TOVA scoring
        idx = np.argpartition(-a, K_t)[:K_t]
        m = np.zeros(n_kv, dtype=bool); m[idx] = True
        return m, int(m.sum())

    r = min(3, len(s))
    svd_score = (s[:r, None] * np.abs(vt[:r, :])).sum(axis=0)
    sv_max = svd_score.max()
    if sv_max > 1e-12:
        svd_score = svd_score / sv_max

    score = 0.50 * a + 0.50 * svd_score
    idx = np.argpartition(-score, K_t)[:K_t]
    m = np.zeros(n_kv, dtype=bool); m[idx] = True
    return m, int(m.sum())


def m_endurkv_v1_lowrank_err(n_kv, K, attn_full, attn_history_l, **kw):
    """v1 gate + low-rank reconstruction score: tokens whose attention history is
    well explained by the rank-r SVD of H (low residual) score higher."""
    if n_kv <= K:
        m = np.ones(n_kv, dtype=bool); return m, int(m.sum())
    a = attn_full[:n_kv]
    max_a = float(a.max())
    norm = max(0.0, min(1.0, (max_a - 0.4) / 0.4))
    mult = 1.3 - 0.6 * norm
    K_t = max(1, min(n_kv, int(round(K * mult))))

    W = 8
    H = _history_matrix(attn_history_l, n_kv, W)
    H[-1, :] = np.maximum(H[-1, :], a)

    try:
        u, s, vt = np.linalg.svd(H, full_matrices=False)
    except np.linalg.LinAlgError:
        idx = np.argpartition(-a, K_t)[:K_t]
        m = np.zeros(n_kv, dtype=bool); m[idx] = True
        return m, int(m.sum())

    r = min(3, len(s))
    H_r = u[:, :r] * s[:r] @ vt[:r, :]
    # per-position squared residual, inverted so low error scores high
    err = ((H - H_r) ** 2).sum(axis=0)
    err_max = err.max()
    if err_max > 1e-12:
        alignment = 1.0 - err / err_max
    else:
        alignment = np.ones(n_kv)

    score = 0.60 * a + 0.40 * alignment * a.max()
    idx = np.argpartition(-score, K_t)[:K_t]
    m = np.zeros(n_kv, dtype=bool); m[idx] = True
    return m, int(m.sum())


def m_endurkv_v1_eigencentrality(n_kv, K, attn_full, attn_history_l, **kw):
    """v1 gate + eigenvector centrality on the cosine-similarity graph of attention
    histories. Restricted to top-2K_t candidates since the full matrix is O(n_kv^2)."""
    if n_kv <= K:
        m = np.ones(n_kv, dtype=bool); return m, int(m.sum())
    a = attn_full[:n_kv]
    max_a = float(a.max())
    norm = max(0.0, min(1.0, (max_a - 0.4) / 0.4))
    mult = 1.3 - 0.6 * norm
    K_t = max(1, min(n_kv, int(round(K * mult))))

    # candidates by current attention, caps the eigenproblem size
    n_cand = min(n_kv, max(2 * K_t, 256))
    if n_cand >= n_kv:
        cand_idx = np.arange(n_kv)
    else:
        cand_idx = np.argpartition(-a, n_cand - 1)[:n_cand]

    W = 8
    H = _history_matrix(attn_history_l, n_kv, W)
    H[-1, :] = a
    H_c = H[:, cand_idx]   # W x n_cand

    # L2-normalised columns, so M holds cosine similarities (n_cand x n_cand)
    norms = np.linalg.norm(H_c, axis=0, keepdims=True)
    norms = np.where(norms > 1e-9, norms, 1.0)
    Hn = H_c / norms

    M = Hn.T @ Hn

    # dominant eigenvector by power iteration
    v = np.ones(n_cand) / np.sqrt(n_cand)
    for _ in range(16):
        v_new = M @ v
        nv = np.linalg.norm(v_new)
        if nv < 1e-12: break
        v = v_new / nv
    centrality = np.abs(v)
    c_max = centrality.max()
    if c_max > 1e-12:
        centrality = centrality / c_max

    # non-candidates get 0
    centr_full = np.zeros(n_kv)
    centr_full[cand_idx] = centrality

    score = 0.55 * a + 0.45 * centr_full * a.max()
    idx = np.argpartition(-score, K_t)[:K_t]
    m = np.zeros(n_kv, dtype=bool); m[idx] = True
    return m, int(m.sum())


def m_endurkv_v1_spec_nodc(n_kv, K, attn_full, attn_history_l, **kw):
    """v1 gate + spectral energy in harmonics 1-3 only. DC is excluded because it
    tracks summed attention, so the score measures oscillation strength."""
    if n_kv <= K:
        m = np.ones(n_kv, dtype=bool); return m, int(m.sum())
    a = attn_full[:n_kv]
    max_a = float(a.max())
    norm = max(0.0, min(1.0, (max_a - 0.4) / 0.4))
    mult = 1.3 - 0.6 * norm
    K_t = max(1, min(n_kv, int(round(K * mult))))
    W = 8
    H = _history_matrix(attn_history_l, n_kv, W)
    H[-1, :] = np.maximum(H[-1, :], a)
    fft_H = np.fft.rfft(H, axis=0)
    # skip DC (bin 0)
    if fft_H.shape[0] > 3:
        ac_energy = (np.abs(fft_H[1:4, :]) ** 2).sum(axis=0)
    elif fft_H.shape[0] > 1:
        ac_energy = (np.abs(fft_H[1:, :]) ** 2).sum(axis=0)
    else:
        ac_energy = np.zeros(n_kv)
    ac_max = ac_energy.max()
    if ac_max > 1e-12:
        ac_energy = ac_energy / ac_max
    score = 0.70 * a + 0.30 * ac_energy
    idx = np.argpartition(-score, K_t)[:K_t]
    m = np.zeros(n_kv, dtype=bool); m[idx] = True
    return m, int(m.sum())


def m_endurkv_v1_phase(n_kv, K, attn_full, attn_history_l, **kw):
    """v1 gate + phase score: |X1| * cos(angle X1) at the first FFT harmonic of
    each token's attention history, blended 70/30 with current attention."""
    if n_kv <= K:
        m = np.ones(n_kv, dtype=bool); return m, int(m.sum())
    a = attn_full[:n_kv]
    max_a = float(a.max())
    norm = max(0.0, min(1.0, (max_a - 0.4) / 0.4))
    mult = 1.3 - 0.6 * norm
    K_t = max(1, min(n_kv, int(round(K * mult))))
    W = 8
    H = _history_matrix(attn_history_l, n_kv, W)
    H[-1, :] = np.maximum(H[-1, :], a)
    fft_H = np.fft.rfft(H, axis=0)
    if fft_H.shape[0] > 1:
        first_harmonic = fft_H[1, :]
        phase_align = (np.abs(first_harmonic) * np.cos(np.angle(first_harmonic))).astype(np.float64)
        # shift to non-negative, then normalize
        phase_align = phase_align - phase_align.min()
        ph_max = phase_align.max()
        if ph_max > 1e-12:
            phase_align = phase_align / ph_max
    else:
        phase_align = np.zeros(n_kv)
    score = 0.70 * a + 0.30 * phase_align
    idx = np.argpartition(-score, K_t)[:K_t]
    m = np.zeros(n_kv, dtype=bool); m[idx] = True
    return m, int(m.sum())


def m_endurkv_v1_logistic(n_kv, K, attn_full, **kw):
    """v1 gate + logistic score sigmoid(3 * (a - mean) / std), which saturates the
    peaks and spreads out mid-range tokens."""
    if n_kv <= K:
        m = np.ones(n_kv, dtype=bool); return m, int(m.sum())
    a = attn_full[:n_kv]
    max_a = float(a.max())
    norm = max(0.0, min(1.0, (max_a - 0.4) / 0.4))
    mult = 1.3 - 0.6 * norm
    K_t = max(1, min(n_kv, int(round(K * mult))))
    mean = a.mean()
    std = max(1e-9, a.std())
    z = (a - mean) / std  # standardized
    score = 1.0 / (1.0 + np.exp(-3.0 * z))   # beta=3 controls sharpness
    idx = np.argpartition(-score, K_t)[:K_t]
    m = np.zeros(n_kv, dtype=bool); m[idx] = True
    return m, int(m.sum())


def m_endurkv_v1_geometric(n_kv, K, attn_full, attn_history_l, **kw):
    """v1 gate + sqrt(a * mean(history)), so a token needs both current and past
    attention to score high."""
    if n_kv <= K:
        m = np.ones(n_kv, dtype=bool); return m, int(m.sum())
    a = attn_full[:n_kv]
    max_a = float(a.max())
    norm = max(0.0, min(1.0, (max_a - 0.4) / 0.4))
    mult = 1.3 - 0.6 * norm
    K_t = max(1, min(n_kv, int(round(K * mult))))
    W = 4
    H = _history_matrix(attn_history_l, n_kv, W)
    H[-1, :] = a
    hist_mean = H.mean(axis=0)
    score = np.sqrt(np.maximum(a, 0) * np.maximum(hist_mean, 0))
    idx = np.argpartition(-score, K_t)[:K_t]
    m = np.zeros(n_kv, dtype=bool); m[idx] = True
    return m, int(m.sum())


def m_endurkv_v1_all(n_kv, K, attn_full, attn_history_l, **kw):
    """v1 gate + spectral + differential score, with 25% of K_t reserved for
    Wasserstein-quantile positions."""
    if n_kv <= K:
        m = np.ones(n_kv, dtype=bool); return m, int(m.sum())
    a = attn_full[:n_kv]
    max_a = float(a.max())
    norm = max(0.0, min(1.0, (max_a - 0.4) / 0.4))
    mult = 1.3 - 0.6 * norm
    K_t = max(1, min(n_kv, int(round(K * mult))))

    W = 8
    H = _history_matrix(attn_history_l, n_kv, W)
    H[-1, :] = np.maximum(H[-1, :], a)
    fft_H = np.fft.rfft(H, axis=0)
    n_freqs = min(3, fft_H.shape[0])
    energy = (np.abs(fft_H[:n_freqs, :]) ** 2).sum(axis=0)
    energy_max = energy.max()
    if energy_max > 1e-12:
        energy = energy / energy_max
    d1 = H[-1, :] - H[-2, :] if H.shape[0] >= 2 else np.zeros(n_kv)
    score = 0.50 * a + 0.25 * energy + 0.15 * np.maximum(d1, 0)

    K_quant = max(1, K_t // 4)
    K_score = K_t - K_quant

    m = np.zeros(n_kv, dtype=bool)

    if K_score > 0:
        top = np.argpartition(-score, min(K_score, n_kv - 1))[:K_score]
        m[top] = True

    # quantile positions not yet kept
    total = a.sum()
    if total > 1e-9 and K_quant > 0:
        cum = np.cumsum(a) / total
        targets = (np.arange(K_quant) + 0.5) / K_quant
        cand = np.searchsorted(cum, targets, side="left")
        cand = np.clip(cand, 0, n_kv - 1)
        for c in cand:
            if not m[c]:
                m[c] = True
                if int(m.sum()) >= K_t: break

    # fill any remaining slots by score
    needed = K_t - int(m.sum())
    if needed > 0:
        remaining = np.where(~m)[0]
        if len(remaining) > 0:
            add = remaining[np.argpartition(-score[remaining], min(needed - 1, len(remaining) - 1))[:needed]]
            m[add] = True
    return m, int(m.sum())


def m_endurkv_v5(n_kv, K, attn_full, attn_history_l, **kw):
    """EndurKV-Evict v5: R-KV-style redundancy penalty using attention-profile cosine
    similarity over the last 8 steps (no K vectors in .attn.bin). Score = a - 0.35*redundancy."""
    if n_kv <= K:
        m = np.ones(n_kv, dtype=bool); return m, int(m.sum())
    a = attn_full[:n_kv]

    # v1 gate
    max_a = float(a.max())
    norm = max(0.0, min(1.0, (max_a - 0.4) / 0.4))
    mult = 1.3 - 0.6 * norm
    K_t = max(1, min(n_kv, int(round(K * mult))))

    # pre-select 3*K_t candidates by current attention, sorted descending
    n_pre = min(n_kv, max(K_t, 3 * K_t))
    if n_pre >= n_kv:
        pre_idx = np.arange(n_kv)
    else:
        pre_idx = np.argpartition(-a, n_pre - 1)[:n_pre]
    pre_idx = pre_idx[np.argsort(-a[pre_idx])]

    # profile matrix: rows are tokens, columns are recent steps
    W = 8
    hist = attn_history_l[-W:] if attn_history_l is not None else [a]
    profiles = np.zeros((len(pre_idx), len(hist)), dtype=np.float64)
    for t, h in enumerate(hist):
        if h is None or len(h) == 0:
            continue
        h_full = np.zeros(n_kv)
        h_full[:min(len(h), n_kv)] = h[:min(len(h), n_kv)]
        profiles[:, t] = h_full[pre_idx]

    norms = np.linalg.norm(profiles, axis=1, keepdims=True)
    safe_norms = np.where(norms > 1e-9, norms, 1.0)
    profiles_n = profiles / safe_norms

    sim = profiles_n @ profiles_n.T
    np.fill_diagonal(sim, 0.0)

    # redundancy[i] = max similarity to any better-scored token j < i
    redundancy = np.zeros(len(pre_idx))
    for i in range(1, len(pre_idx)):
        redundancy[i] = sim[i, :i].max()

    LAMBDA = 0.35
    final = a[pre_idx] - LAMBDA * redundancy

    if len(pre_idx) <= K_t:
        m = np.zeros(n_kv, dtype=bool); m[pre_idx] = True
        return m, int(m.sum())
    top_local = np.argpartition(-final, K_t)[:K_t]
    selected = pre_idx[top_local]
    m = np.zeros(n_kv, dtype=bool); m[selected] = True
    return m, int(m.sum())


def m_endurkv_v4(n_kv, K, attn_full, attn_ewma_l, cross_layer_attn,
                 cross_layer_max, layer_idx, **kw):
    """EndurKV-Evict v4: v1 gate, 4 sinks + 4 recent, then about 1/3 of the rest for
    tokens with high ewma * cross_layer_max, the remainder by the v3 composite score."""
    if n_kv <= K:
        m = np.ones(n_kv, dtype=bool); return m, int(m.sum())
    a = attn_full[:n_kv]

    # v1 budget gate
    max_a = float(a.max())
    norm = max(0.0, min(1.0, (max_a - 0.4) / 0.4))
    mult = 1.3 - 0.6 * norm
    K_t = max(8, min(n_kv, int(round(K * mult))))

    # reserve sinks + recent
    n_sink = min(4, K_t // 4)
    n_recent = min(4, K_t // 4)
    m = np.zeros(n_kv, dtype=bool)
    m[:n_sink] = True
    rec_start = max(n_sink, n_kv - n_recent)
    m[rec_start:n_kv] = True
    K_used = int(m.sum())

    if K_used >= K_t:
        return m, K_used

    # critical-token protection
    ewma = (attn_ewma_l[:n_kv]
            if attn_ewma_l is not None and len(attn_ewma_l) >= n_kv
            else a)
    cl_max = (cross_layer_max[:n_kv]
              if cross_layer_max is not None and len(cross_layer_max) >= n_kv
              else a)
    critical = ewma * cl_max

    # candidates exclude sinks and recent
    candidate_mask = ~m
    cand_idx = np.where(candidate_mask)[0]

    if len(cand_idx) == 0:
        return m, K_used

    K_rem = K_t - K_used

    crit_scores_cand = critical[cand_idx]
    if len(cand_idx) > K_rem and K_rem > 0:
        # critical tokens take about 1/3 of the remaining budget
        n_critical_keep = max(0, min(K_rem // 3, len(cand_idx)))
        if n_critical_keep > 0:
            top_crit = cand_idx[np.argpartition(-crit_scores_cand,
                                                min(n_critical_keep, len(cand_idx) - 1))[:n_critical_keep]]
            m[top_crit] = True

        # remaining budget by the current + cross-layer + ewma composite
        K_used2 = int(m.sum())
        K_left = K_t - K_used2
        if K_left > 0:
            still_candidate = np.where(~m)[0]
            if len(still_candidate) > 0:
                cl = (cross_layer_attn[:n_kv]
                      if cross_layer_attn is not None and len(cross_layer_attn) >= n_kv
                      else np.zeros(n_kv))
                composite = 0.55 * a[still_candidate] + 0.25 * cl[still_candidate] \
                          + 0.20 * ewma[still_candidate]
                if len(still_candidate) <= K_left:
                    m[still_candidate] = True
                else:
                    top = still_candidate[np.argpartition(-composite, K_left)[:K_left]]
                    m[top] = True
    else:
        m[cand_idx] = True

    return m, int(m.sum())


def m_endurkv_v2(n_kv, K, attn_full, attn_ewma_l, cross_layer_attn,
                 layer_idx, **kw):
    """EndurKV-Evict v2: entropy gate mult = 0.7 + 0.6 * H_t / log(n_kv), score
    0.55*current + 0.25*cross-layer mean + 0.20*EWMA (beta 0.3)."""
    if n_kv <= K:
        m = np.ones(n_kv, dtype=bool); return m, int(m.sum())
    a = attn_full[:n_kv]

    # 1) entropy gate
    a_safe = np.maximum(a, 1e-12)
    H = -float(np.sum(a * np.log(a_safe)))
    H_max = max(1e-6, float(np.log(max(2, n_kv))))
    norm_H = max(0.0, min(1.0, H / H_max))
    mult = 0.7 + 0.6 * norm_H
    K_t = max(1, min(n_kv, int(round(K * mult))))

    # 2) cross-layer vote, aggregated in simulate()
    cl = (cross_layer_attn[:n_kv]
          if cross_layer_attn is not None and len(cross_layer_attn) >= n_kv
          else np.zeros(n_kv))

    # 3) EWMA-smoothed history
    ewma = (attn_ewma_l[:n_kv]
            if attn_ewma_l is not None and len(attn_ewma_l) >= n_kv
            else a)

    score = 0.55 * a + 0.25 * cl + 0.20 * ewma

    idx = np.argpartition(-score, K_t)[:K_t]
    m = np.zeros(n_kv, dtype=bool); m[idx] = True
    return m, int(m.sum())


def m_tova(n_kv, K, attn_full, **kw):
    """TOVA (Oren et al., 2024): top-K by the current step's attention, recomputed
    every step, no accumulation or recent window."""
    if n_kv <= K:
        m = np.ones(n_kv, dtype=bool); return m, int(m.sum())
    cand = attn_full[:n_kv]
    idx = np.argpartition(-cand, K)[:K]
    m = np.zeros(n_kv, dtype=bool); m[idx] = True
    return m, int(m.sum())


def m_scissorhands(n_kv, K, scissor_score_l, **kw):
    """Scissorhands (Liu et al., NeurIPS 2023): top heavy hitters by the decayed
    window score from simulate(), with K/8 reserved for the most recent tokens."""
    if n_kv <= K:
        m = np.ones(n_kv, dtype=bool); return m, int(m.sum())
    K_rec = max(1, K // 8)
    K_hh = K - K_rec
    rec_start = max(0, n_kv - K_rec)
    cand = scissor_score_l[:rec_start] if scissor_score_l is not None else np.zeros(rec_start)
    if len(cand) <= K_hh:
        hh_idx = np.arange(rec_start)
    else:
        hh_idx = np.argpartition(-cand, K_hh)[:K_hh]
    m = np.zeros(n_kv, dtype=bool)
    m[hh_idx] = True; m[rec_start:n_kv] = True
    return m, int(m.sum())


def m_endurkv_unfair(n_kv, K, accum, H_tilde=0.5, **kw):
    """Proposal Rule 4, K_t = K * (1 + H_tilde). Not budget-matched (about 1.5K avg)."""
    K_t = int(min(n_kv, K * (1.0 + max(0.0, H_tilde))))
    return _hh_norec_mask(n_kv, K_t, accum, recent_frac=1.0/16)

def m_endurkv_attn_spread(n_kv, K, accum, attn_full=None, **kw):
    """Fair-budget: K_t = K * (1.3 - 0.6 * max_attn_norm)."""
    if attn_full is None or len(attn_full) == 0:
        return _hh_norec_mask(n_kv, K, accum, recent_frac=1.0/16)
    max_a = float(attn_full[:n_kv].max())
    norm = max(0.0, min(1.0, (max_a - 0.4) / 0.4))
    mult = 1.3 - 0.6 * norm
    K_t = max(1, min(n_kv, int(round(K * mult))))
    return _hh_norec_mask(n_kv, K_t, accum, recent_frac=1.0/16)

def m_endurkv_adaptive(n_kv, K, accum, attn_full=None,
                       max_n_ctx=4096, **kw):
    """Budget multiplier by pressure = n_kv / max_n_ctx: 1.0 below 0.5, the
    attention-spread gate up to 0.85, then 0.7."""
    if n_kv <= K:
        m = np.ones(n_kv, dtype=bool); return m, int(m.sum())

    pressure = n_kv / max(1, max_n_ctx)

    if pressure < 0.5:
        mult = 1.0
    elif pressure < 0.85:
        # attention-concentration modulation
        if attn_full is None or len(attn_full) == 0:
            mult = 1.0
        else:
            max_a = float(attn_full[:n_kv].max())
            norm = max(0.0, min(1.0, (max_a - 0.4) / 0.4))
            mult = 1.3 - 0.6 * norm
    else:
        # near context limit: aggressive eviction
        mult = 0.7

    K_t = max(1, min(n_kv, int(round(K * mult))))
    return _hh_norec_mask(n_kv, K_t, accum, recent_frac=1.0/16)


# PyramidKV / CAKE: layer-wise budget allocation
def layer_budget_pyramid(K, layer_idx, n_layers, ratio=0.5):
    """Linear K_max to K_min across depth, K_max = K*(1+ratio), K_min = K*(1-ratio).
    Average across layers is K."""
    if n_layers <= 1: return K
    frac = layer_idx / (n_layers - 1)   # 0 at first layer, 1 at last
    K_max = K * (1 + ratio)
    K_min = K * (1 - ratio)
    return max(1, int(round(K_max - (K_max - K_min) * frac)))

def layer_budget_cake(K, layer_idx, n_layers, ratio=0.5):
    """CAKE-style: deeper layers (more diffuse attention) get more budget,
    linear K_min to K_max across depth."""
    if n_layers <= 1: return K
    frac = layer_idx / (n_layers - 1)
    K_max = K * (1 + ratio)
    K_min = K * (1 - ratio)
    return max(1, int(round(K_min + (K_max - K_min) * frac)))


POLICIES_PER_LAYER = {
    "full":                m_full,
    "local":               m_local,
    "streamingllm":        m_streamingllm,
    "h2o":                 m_h2o,
    "h2o_norec":           m_h2o_norec,
    "snapkv":              m_snapkv,
    "tova":                m_tova,                      # Oren 2024
    "scissorhands":        m_scissorhands,              # Liu NeurIPS 2023
    # 2025-26 baselines
    "lwkd":                m_lwkd,                      # SAGE-KV, 2025
    "ahakv":               m_ahakv,                     # 2025, approximation
    "lazyeviction":        m_lazyeviction,              # 2025
    # endurkv family
    "endurkv":             m_endurkv_unfair,            # not budget-matched
    "endurkv_attn_spread": m_endurkv_attn_spread,       # budget-matched
    "endurkv_adaptive":    m_endurkv_adaptive,          # budget-matched, pressure-aware
    "endurkv_tova_spread": m_endurkv_tova_spread,       # v1: TOVA + attn-spread
    "endurkv_v2":          m_endurkv_v2,                # v2: entropy gate
    "endurkv_v3":          m_endurkv_v3,                # v3: v1-gate + composite
    "endurkv_v4":          m_endurkv_v4,                # v4: critical-token protection
    "endurkv_v5":          m_endurkv_v5,                # v5: matrix redundancy
    "endurkv_v6":          m_endurkv_v6,                # v6: v1-gate + recurrence bonus
    # Fourier, derivative and optimal-transport scores
    "endurkv_spectral":      m_endurkv_spectral,         # FFT low-freq energy alone
    "endurkv_differential":  m_endurkv_differential,     # Taylor-predicted future attn alone
    "endurkv_wasserstein":   m_endurkv_wasserstein,      # 1D-W1 quantile selection alone
    "endurkv_v1_spectral":   m_endurkv_v1_spectral,      # v1 gate + spectral score
    "endurkv_v1_differential": m_endurkv_v1_differential,  # v1 gate + derivative score
    "endurkv_v1_wasserstein":  m_endurkv_v1_wasserstein,   # v1 gate + quantile selection
    "endurkv_v1_spec_diff":    m_endurkv_v1_spec_diff,     # v1 gate + spec + diff composite
    "endurkv_v1_all":          m_endurkv_v1_all,           # v1 + spec + diff + W1
    # signals other than current attention
    "endurkv_v1_spec_nodc":    m_endurkv_v1_spec_nodc,     # FFT magnitude in harmonics 1-3 only (excludes DC)
    "endurkv_v1_phase":        m_endurkv_v1_phase,         # FFT phase at first harmonic
    "endurkv_v1_logistic":     m_endurkv_v1_logistic,      # sigmoid-squashed standardized attention
    "endurkv_v1_geometric":    m_endurkv_v1_geometric,     # geometric mean of current and history
    # linear algebra scores
    "endurkv_v1_svd":              m_endurkv_v1_svd,              # SVD principal-pattern participation
    "endurkv_v1_lowrank_err":      m_endurkv_v1_lowrank_err,      # low-rank reconstruction error
    "endurkv_v1_eigencentrality":  m_endurkv_v1_eigencentrality,  # eigenvector centrality on co-attention graph
}
# Policies with a different K per layer.
LAYERWISE_POLICIES = {
    "pyramidkv":     layer_budget_pyramid,
    "cake":          layer_budget_cake,
    "cake_entropy":  None,   # per-layer budget from attention entropy, set in simulate()
}


# simulator
def simulate(attn, n_kv_at, policy_name, K_nominal, H_norm=None,
             max_n_ctx=4096):
    n_steps, n_layers, max_kv = attn.shape
    accum = np.zeros((n_layers, max_kv), dtype=np.float64)
    # SnapKV (Li 2024) observation-window score, L_obs=32, max-pool kernel 7.
    # attn.bin has no prefill attention, so the first L_obs decode steps stand
    # in for the last L_obs prompt tokens.
    L_obs = 32
    pool_k = 7
    obs_steps = min(L_obs, n_steps)
    if obs_steps > 0 and max_kv > 0:
        snapkv_score = attn[:obs_steps].sum(axis=(0, 1))  # shape (max_kv,)
        pad = pool_k // 2
        padded = np.pad(snapkv_score, pad, mode="edge")
        smoothed = np.maximum.reduce([padded[i:i + max_kv] for i in range(pool_k)])
        snapkv_score = smoothed
    else:
        snapkv_score = np.zeros(max_kv)

    # CAKE (Qin 2025): more cache for more diffuse layers. Step-0 attention
    # entropy per layer is the diffuseness proxy.
    if n_steps > 0:
        per_layer_H = np.zeros(n_layers)
        for l in range(n_layers):
            p = attn[0, l, :n_kv_at[0]]
            p = p[p > 1e-12]
            if len(p) > 0:
                per_layer_H[l] = -np.sum(p * np.log(p))
        # weights sum to n_layers so the total budget is K * n_layers
        if per_layer_H.sum() > 0:
            cake_weights = (per_layer_H / per_layer_H.sum()) * n_layers
            cake_weights = np.clip(cake_weights, 0.5, 1.5)
        else:
            cake_weights = np.ones(n_layers)
    else:
        cake_weights = np.ones(n_layers)
    out_kl = np.zeros((n_steps, n_layers), dtype=np.float64)
    out_K  = np.zeros((n_steps, n_layers), dtype=np.int32)

    is_layerwise = policy_name in LAYERWISE_POLICIES
    if is_layerwise:
        if policy_name == "cake_entropy":
            per_layer_K = [max(1, int(round(K_nominal * cake_weights[l])))
                           for l in range(n_layers)]
        else:
            per_layer_K = [LAYERWISE_POLICIES[policy_name](K_nominal, l, n_layers)
                           for l in range(n_layers)]
    else:
        per_layer_K = [K_nominal] * n_layers
        policy_fn = POLICIES_PER_LAYER[policy_name]

    # Scissorhands (Liu 2023) paper settings
    W_scissor = 64
    decay = 0.95
    scissor_score = np.zeros((n_layers, max_kv), dtype=np.float64)

    # AhaKV recent-window accumulator, 16 decode steps
    W_aha = 16
    aha_recent = np.zeros((n_layers, max_kv), dtype=np.float64)
    aha_history = []   # ring of past (n_layers, max_kv) attention slices

    # EndurKV-v2 EWMA-smoothed attention (per layer)
    ewma_beta = 0.30
    attn_ewma = np.zeros((n_layers, max_kv), dtype=np.float64)

    # per-layer history of the last 8 steps' attention
    W_v5 = 8
    attn_history = [[] for _ in range(n_layers)]   # list per layer of past slices

    # LWKD frozen-mask state (per layer)
    lwkd_state = {"frozen": [None] * n_layers}

    # LazyEviction state per (layer, position): max recurrence interval (MRI) and
    # the last step where attention exceeded tau.
    lazy_state = {
        "mri":               np.zeros((n_layers, max_kv), dtype=np.int32),
        "last_high_step":    -np.ones((n_layers, max_kv), dtype=np.int32),
        "tau":               0.05,    # attention threshold for "high"
        "eviction_window":   16,      # eviction only every W steps
        "current_step":      0,
    }

    for s in range(n_steps):
        n_kv = int(n_kv_at[s])
        if n_kv == 0: continue
        Ht = 0.5 if H_norm is None else float(H_norm[s] if s < len(H_norm) else 0.5)
        for l in range(n_layers):
            accum[l, :n_kv] += attn[s, l, :n_kv]
        # Scissorhands score with exponential decay
        for l in range(n_layers):
            scissor_score[l, :n_kv] = scissor_score[l, :n_kv] * decay + attn[s, l, :n_kv]

        # AhaKV window sum, subtract the step that just expired
        aha_history.append(attn[s].copy())
        if len(aha_history) > W_aha:
            old = aha_history.pop(0)
            for l in range(n_layers):
                aha_recent[l, :old.shape[1]] -= old[l]
        for l in range(n_layers):
            aha_recent[l, :n_kv] += attn[s, l, :n_kv]

        # per-layer attention EWMA (v2, v3, v4)
        for l in range(n_layers):
            attn_ewma[l, :n_kv] = (1 - ewma_beta) * attn_ewma[l, :n_kv] + \
                                  ewma_beta * attn[s, l, :n_kv]

        # cross-layer mean (vote) and max (v4) at this step
        cross_layer_attn = attn[s, :, :n_kv].mean(axis=0)
        cross_layer_max = attn[s, :, :n_kv].max(axis=0)

        # per-layer history of the last W_v5 steps
        for l in range(n_layers):
            attn_history[l].append(attn[s, l, :n_kv].copy())
            if len(attn_history[l]) > W_v5:
                attn_history[l].pop(0)

        # LazyEviction: update last_high_step and MRI where attention exceeds tau
        lazy_state["current_step"] = s
        tau = lazy_state["tau"]
        for l in range(n_layers):
            a = attn[s, l, :n_kv]
            high = a > tau
            for i in np.where(high)[0]:
                prev = lazy_state["last_high_step"][l, i]
                if prev >= 0:
                    interval = s - int(prev)
                    if interval > lazy_state["mri"][l, i]:
                        lazy_state["mri"][l, i] = interval
                lazy_state["last_high_step"][l, i] = s

        for l in range(n_layers):
            K_eff = per_layer_K[l]
            p_full = attn[s, l, :n_kv]
            if is_layerwise:
                # Use h2o_norec mechanism with the per-layer K
                mask, K_used = _hh_norec_mask(n_kv, K_eff,
                                              accum[l, :n_kv],
                                              recent_frac=1.0/16)
            else:
                mask, K_used = policy_fn(
                    n_kv=n_kv, K=K_eff, accum=accum[l, :n_kv],
                    attn_full=p_full, H_tilde=Ht,
                    snapkv_score=snapkv_score[:n_kv],
                    scissor_score_l=scissor_score[l, :n_kv],
                    recent_accum_l=aha_recent[l, :n_kv],
                    attn_ewma_l=attn_ewma[l, :n_kv],
                    attn_history_l=attn_history[l],
                    lazy_mri_l=lazy_state["mri"][l, :n_kv],
                    lazy_last_high_l=lazy_state["last_high_step"][l, :n_kv],
                    lazy_current_step=lazy_state["current_step"],
                    lazy_window=lazy_state["eviction_window"],
                    cross_layer_attn=cross_layer_attn,
                    cross_layer_max=cross_layer_max,
                    lwkd_state=lwkd_state,
                    layer_idx=l,
                    max_n_ctx=max_n_ctx,
                )
            kept = p_full * mask
            tot = kept.sum()
            if tot <= 0:
                out_kl[s, l] = 20.0
            else:
                out_kl[s, l] = kl(p_full, kept / tot)
            out_K[s, l] = K_used
    return out_kl, out_K


def normalize_entropy(H_nats, window=64):
    H = np.asarray(H_nats, dtype=np.float64)
    if len(H) == 0: return H
    out = np.zeros_like(H)
    for i in range(len(H)):
        a = max(0, i - window + 1)
        seg = H[a:i+1]
        lo, hi = seg.min(), seg.max()
        out[i] = 0.5 if hi - lo < 1e-9 else (H[i] - lo) / (hi - lo)
    return out


# main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--budgets", default="64,128,256")
    ap.add_argument("--max-prompts", type=int, default=0)
    ap.add_argument("--model-ctx", type=int, default=4096,
                    help="max context size for pressure_ratio in adaptive")
    ap.add_argument("--bytes-per-token", type=int, default=128*1024,
                    help="for 8B Q4_K_M FP16 KV: 128 KB/token; "
                         "for 1B Q4_K_M FP16 KV: 32 KB/token")
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir) if args.out_dir else log_dir.parent / f"{log_dir.name}_eviction_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    budgets = [int(b) for b in args.budgets.split(",")]

    attn_files = sorted(log_dir.glob("*.attn.bin"))
    if args.max_prompts > 0:
        attn_files = attn_files[:args.max_prompts]

    all_policies = list(POLICIES_PER_LAYER.keys()) + list(LAYERWISE_POLICIES.keys())
    print(f"[sim] {len(attn_files)} prompts, K={budgets}, policies={all_policies}")
    print(f"[sim] bytes_per_token={args.bytes_per_token}  model_ctx={args.model_ctx}")

    rows = []
    for ap_path in attn_files:
        pid = ap_path.name[:-len(".attn.bin")]
        ent_path = log_dir / f"{pid}.entropy.csv"
        if not ent_path.exists(): continue
        attn, n_kv_at = load_attn_full(ap_path)
        if attn is None or attn.shape[0] == 0: continue
        try:
            ent = pd.read_csv(ent_path)
        except Exception: continue
        H_norm = normalize_entropy(ent["H_nats"].values[:attn.shape[0]])
        task = pid.rsplit("_", 1)[0]
        if task.endswith("_lc"): task = task[:-3]
        n_steps, n_layers, max_kv = attn.shape
        print(f"  {pid:<28} steps={n_steps} layers={n_layers} max_kv={max_kv}")
        for K in budgets:
            for pname in all_policies:
                kls, Ks = simulate(attn, n_kv_at, pname, K, H_norm=H_norm,
                                   max_n_ctx=args.model_ctx)
                mean_kl = float(np.mean(kls))
                avg_K   = float(np.mean(Ks))
                # throughput proxy, inversely proportional to avg cache, full = 1.0
                full_avg_K = float(np.mean(n_kv_at))  # full-cache uses n_kv per step
                tput_rel = full_avg_K / max(1.0, avg_K)  # >=1 means faster than full
                dram_bytes = avg_K * args.bytes_per_token * n_layers
                rows.append({
                    "prompt_id": pid, "task": task,
                    "policy": pname, "K_nominal": K,
                    "mean_kl": mean_kl,
                    "avg_actual_K": avg_K,
                    "throughput_rel": tput_rel,
                    "dram_bytes_avg": dram_bytes,
                    "n_steps": n_steps, "n_layers": n_layers,
                    "max_kv": max_kv,
                })

    if not rows:
        print("ERROR: no prompts processed", file=sys.stderr); return 1
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "policy_results.csv", index=False)
    print(f"\n[sim] wrote policy_results.csv ({len(df)} rows)")

    # Pareto summary (per policy, per nominal K)
    summary = df.groupby(["policy", "K_nominal"]).agg(
        mean_kl=("mean_kl", "mean"),
        avg_actual_K=("avg_actual_K", "mean"),
        throughput_rel=("throughput_rel", "mean"),
        dram_bytes_avg=("dram_bytes_avg", "mean"),
        n_prompts=("prompt_id", "nunique"),
    ).reset_index()

    # budget_matched: avg_actual_K within 10% of K_nominal. Informational only,
    # the main comparison is the Pareto frontier over (avg_actual_K, mean_kl).
    summary["budget_matched"] = (summary["avg_actual_K"] <= summary["K_nominal"] * 1.10).astype(bool)
    summary.to_csv(out_dir / "pareto_summary.csv", index=False)

    # Pareto frontier: non-dominated points in (avg_actual_K, mean_kl)
    pts = summary[["policy", "K_nominal", "avg_actual_K", "mean_kl"]].copy()
    pts = pts.sort_values("avg_actual_K").reset_index(drop=True)
    pareto = []
    best_kl = float("inf")
    for _, row in pts.iterrows():
        if row.mean_kl < best_kl:
            pareto.append(row.name)
            best_kl = row.mean_kl
    pts["on_pareto"] = pts.index.isin(pareto)
    pts.to_csv(out_dir / "pareto_frontier.csv", index=False)
    print("Pareto-frontier operating points (best quality at each cache size)")
    print(pts[pts.on_pareto][["policy", "K_nominal", "avg_actual_K", "mean_kl"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # print headline tables
    print("Per-policy at each K_nominal")
    print("(fair = avg_actual_K within 10% of K_nominal)\n")
    pivot = summary.pivot_table(
        index="policy", columns="K_nominal",
        values=["mean_kl", "avg_actual_K"], aggfunc="mean"
    )
    print(pivot.to_string(float_format=lambda x: f"{x:7.3f}"))

    # plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 130, "font.size": 9})

    # 01: each policy traces (avg_actual_K, mean_kl) as K varies, lower-left is better.
    # Give every new policy a color here, otherwise lines fall back to the same gray.
    colors = {
        "full":                "#000000", "local":             "#bdc3c7",
        "streamingllm":        "#e67e22",
        "h2o":                 "#2ecc71", "h2o_norec":         "#196f3d",
        "scissorhands":        "#16a085",
        "snapkv":              "#5dade2", "tova":              "#1f618d",
        "pyramidkv":           "#bb8fce", "cake":              "#7d3c98",
        "cake_entropy":        "#4a235a",
        "endurkv":             "#ffb3ba", "endurkv_adaptive":  "#ff7f7f",
        "endurkv_attn_spread": "#c0392b",
        "endurkv_tova_spread": "#d6006e",   # EndurKV-Evict (ours)
    }
    display = {
        "endurkv_tova_spread": "EndurKV-Evict (ours)",
        "endurkv_attn_spread": "EndurKV-Evict (H2O variant)",
        "endurkv_adaptive":    "EndurKV-adaptive (ablation)",
    }
    OURS = "endurkv_tova_spread"
    fig, ax = plt.subplots(figsize=(11, 7))
    for pname in summary.policy.unique():
        rows = summary[summary.policy == pname].sort_values("avg_actual_K")
        if rows.empty: continue
        c = colors.get(pname, "tab:gray")
        is_ours = (pname == OURS)
        ax.plot(rows.avg_actual_K, rows.mean_kl,
                marker="D" if is_ours else "o",
                linestyle="-", color=c,
                lw=3.0 if is_ours else 1.4,
                markersize=11 if is_ours else 9,
                label=display.get(pname, pname),
                alpha=0.95 if is_ours else 0.85,
                zorder=10 if is_ours else 5)
        # annotate K_nominal at each point
        for _, r in rows.iterrows():
            ax.annotate(f"K={int(r.K_nominal)}",
                        (r.avg_actual_K, r.mean_kl),
                        xytext=(4, -10), textcoords="offset points",
                        fontsize=7, color=c)
    # Pareto frontier line (best mean_kl at each cache size)
    pareto_rows = pts[pts.on_pareto].sort_values("avg_actual_K")
    if not pareto_rows.empty:
        ax.plot(pareto_rows.avg_actual_K, pareto_rows.mean_kl,
                "k--", lw=2.0, alpha=0.4, label="Pareto frontier")
    ax.set_xlabel("avg actual cache size used (per layer per step)")
    ax.set_ylabel("mean KL (attention preservation)")
    ax.set_yscale("log")
    ax.set_title(
        "Pareto frontier — quality vs cache used (each policy at multiple budgets)\n"
        "Lower-left = better. Different policies sit at different operating points; the "
        "Pareto frontier is the envelope of best achievable quality."
    )
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_dir / "01_pareto_kl_vs_cache.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote 01_pareto_kl_vs_cache.png")

    # 01b: same Pareto plot on a DRAM bytes axis
    fig, ax = plt.subplots(figsize=(11, 7))
    for pname in summary.policy.unique():
        rows = summary[summary.policy == pname].sort_values("dram_bytes_avg")
        if rows.empty: continue
        c = colors.get(pname, "tab:gray")
        is_ours = (pname == OURS)
        ax.plot(rows.dram_bytes_avg/(1024*1024), rows.mean_kl,
                marker="D" if is_ours else "o",
                linestyle="-", color=c,
                lw=3.0 if is_ours else 1.4,
                markersize=11 if is_ours else 9,
                label=display.get(pname, pname),
                alpha=0.95 if is_ours else 0.85,
                zorder=10 if is_ours else 5)
        for _, r in rows.iterrows():
            ax.annotate(f"K={int(r.K_nominal)}",
                        (r.dram_bytes_avg/(1024*1024), r.mean_kl),
                        xytext=(4, -10), textcoords="offset points",
                        fontsize=7, color=c)
    ax.set_xlabel("avg KV-cache DRAM footprint per layer (MB)")
    ax.set_ylabel("mean KL")
    ax.set_yscale("log")
    ax.set_title("Pareto frontier on DRAM bytes — the operationally-relevant view")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_dir / "01b_pareto_kl_vs_dram_mb.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote 01b_pareto_kl_vs_dram_mb.png")

    # 02: bar chart at fixed K_nominal (informational, the Pareto plot is primary)
    median_K = budgets[len(budgets)//2]
    sub = summary[summary.K_nominal == median_K]
    fair_only = sub[sub.budget_matched].sort_values("mean_kl")
    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(fair_only.policy, fair_only.mean_kl,
                  color=[colors.get(p, "gray") for p in fair_only.policy])
    ax.set_yscale("log")
    ax.set_ylabel("mean KL (log)")
    ax.set_title(f"Budget-matched comparison at K_nominal={median_K} (avg_K ≤ 1.1×K)")
    ax.tick_params(axis="x", rotation=25)
    for b, v in zip(bars, fair_only.mean_kl):
        ax.text(b.get_x() + b.get_width()/2, v * 1.05, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8)
    ax.grid(alpha=0.3, axis="y", which="both")
    fig.tight_layout()
    fig.savefig(out_dir / "02_fair_budget_bar.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote 02_fair_budget_bar.png")

    # 03: DRAM bytes vs KL
    fig, ax = plt.subplots(figsize=(10, 6))
    for _, row in sub.iterrows():
        c = colors.get(row.policy, "tab:gray")
        mark = "o" if row.budget_matched else "X"
        ax.scatter(row.dram_bytes_avg/(1024*1024), row.mean_kl,
                   s=110, color=c, marker=mark, edgecolors="black", linewidths=0.8)
        ax.annotate(row.policy, (row.dram_bytes_avg/(1024*1024), row.mean_kl),
                    xytext=(6, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("avg DRAM footprint per layer per step (MB)")
    ax.set_ylabel("mean KL")
    ax.set_yscale("log")
    ax.set_title(f"DRAM bytes vs attention preservation  (K_nominal={median_K})\n"
                 f"O = fair,  X = unfair.  Lower-left is better.")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_dir / "03_dram_bytes_vs_kl.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote 03_dram_bytes_vs_kl.png")

    print(f"\nall outputs in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
