"""Direct K-vector + K+V baseline reimplementations for our bench.

Adds faithful reimplementations of:
  * KeyDiff   (Park et al. arXiv:2504.15364, Apr 2025) — K-vector cosine similarity
  * LaProx    (Mai & Kim arXiv:2605.07234, May 2026) — attention × W_o × V global ranks

Both consume the .kv.bin sidecar produced by the patched attention_probe
(magic "KVCP", written when ATTNPROBE_CAPTURE_KV=1) alongside the existing
ATNH .attn.bin per-step attention.

For the simulator: at decode step s with n_kv_at_step positions, we treat
K[:, :, :n_kv_at_step] and V[:, :, :n_kv_at_step] as the per-step cache state.
This is valid because K and V for already-cached positions don't change as
new tokens are appended.

Output is the same `pareto_summary.csv` format as host_simulate_eviction_perhead.py
so the global ranker picks it up without modification.

Usage:
    python host_simulate_kv_baselines.py \
        --log-dir   logs/study_phone_mistral_longbench   (contains .attn.bin AND .kv.bin)
        --out-dir   logs/eval_mistral_longbench_kv
        --w-o-dir   models_extra/Mistral-7B-W_o          (only needed for LaProx)
        --budgets   64,128,256
        --policies  perhead_v1,perhead_tova,keydiff,laprox
"""
from __future__ import annotations
import argparse
import os
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_attn_perhead(path: Path):
    """Returns (attn_ph[steps, layers, heads, max_kv] fp16, n_kv_at[steps] int32)
    or (None, None) if file isn't ATNH format."""
    if not path.exists():
        return None, None
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"ATNH":
            return None, None
        n_steps, n_layers, n_head = struct.unpack("<III", f.read(12))
        if n_head <= 0:
            return None, None
        per_step = []
        n_kv_at = []
        for _s in range(n_steps):
            layers = []
            last_nkv = 0
            for _l in range(n_layers):
                (n_kv,) = struct.unpack("<I", f.read(4))
                if n_kv == 0:
                    layers.append(None); continue
                last_nkv = max(last_nkv, n_kv)
                count = n_kv * n_head
                vals = np.frombuffer(f.read(4 * count), dtype=np.float32)
                layers.append(vals.reshape(n_head, n_kv))
            per_step.append(layers)
            n_kv_at.append(last_nkv)
    if not per_step:
        return None, None
    max_kv = max(n_kv_at) if n_kv_at else 0
    attn_ph = np.zeros((n_steps, n_layers, n_head, max_kv), dtype=np.float16)
    for s, layers in enumerate(per_step):
        for l, arr in enumerate(layers):
            if arr is not None:
                attn_ph[s, l, :, :arr.shape[1]] = arr.astype(np.float16)
    return attn_ph, np.array(n_kv_at, dtype=np.int32)


def load_kv_sidecar(path: Path):
    """Returns dict {layer_idx: {"K": np.array, "V": np.array, "K_ne": tuple, "V_ne": tuple}}
    or None if file missing / malformed.

    Format ("KVCP" v2, 2026-05-24):
      magic(4) + n_layers(u32)
      per-layer:
        has_K(u32)
        if has_K: K_ne[0..3](4*u32), data_count(u32), data_count*float32
        has_V(u32)
        if has_V: V_ne[0..3](4*u32), data_count(u32), data_count*float32

    For non-contiguous tensors, data_count may be larger than prod(ne) since the
    full underlying buffer was dumped. K and V arrays returned are 1D; reshape
    in the simulator using the appropriate ne+stride logic if needed.
    """
    if not path.exists():
        return None
    with open(path, "rb") as f:
        if f.read(4) != b"KVCP":
            return None
        (n_layers,) = struct.unpack("<I", f.read(4))
        result = {}
        for l in range(n_layers):
            (has_K,) = struct.unpack("<I", f.read(4))
            K = K_ne = None
            if has_K:
                K_ne = struct.unpack("<IIII", f.read(16))
                (data_count,) = struct.unpack("<I", f.read(4))
                K = np.frombuffer(f.read(4 * data_count), dtype=np.float32).copy()
                # Try logical reshape if data fits; otherwise leave as 1D.
                prod_ne = int(np.prod(K_ne))
                if prod_ne > 0 and prod_ne == data_count:
                    # Contiguous case — reshape to logical [ne3, ne2, ne1, ne0]
                    K = K.reshape(K_ne[::-1])
                # else: keep K as 1D; caller decides how to slice
            (has_V,) = struct.unpack("<I", f.read(4))
            V = V_ne = None
            if has_V:
                V_ne = struct.unpack("<IIII", f.read(16))
                (data_count,) = struct.unpack("<I", f.read(4))
                V = np.frombuffer(f.read(4 * data_count), dtype=np.float32).copy()
                prod_ne = int(np.prod(V_ne))
                if prod_ne > 0 and prod_ne == data_count:
                    V = V.reshape(V_ne[::-1])
            result[l] = {"K": K, "V": V, "K_ne": K_ne, "V_ne": V_ne}
        return result


# ---------------------------------------------------------------------------
# KL divergence helper (fp16-safe; matches host_simulate_eviction_perhead.py)
# ---------------------------------------------------------------------------

def kl(p_full: np.ndarray, p_evicted: np.ndarray, eps: float = 1e-12) -> float:
    pf = np.clip(p_full.astype(np.float32, copy=False), eps, 1.0)
    pe = np.clip(p_evicted.astype(np.float32, copy=False), eps, 1.0)
    return float(np.sum(pf * (np.log(pf) - np.log(pe))))


# ---------------------------------------------------------------------------
# Baselines we reimplement
# ---------------------------------------------------------------------------

def policy_perhead_v1(attn_ph_layer_step, K, **kw):
    """EndurKV-Evict per-head form, for direct comparison (uses attention only)."""
    nh, nk = attn_ph_layer_step.shape
    if nk <= K:
        return np.ones((nh, nk), dtype=bool)
    m = np.zeros((nh, nk), dtype=bool)
    for h in range(nh):
        a = attn_ph_layer_step[h]
        max_a = float(a.max())
        norm = max(0.0, min(1.0, (max_a - 0.4) / 0.4))
        mult = 1.3 - 0.6 * norm
        K_t = max(1, min(nk, int(round(K * mult))))
        idx = np.argpartition(-a, K_t)[:K_t]
        m[h, idx] = True
    return m


def policy_perhead_tova(attn_ph_layer_step, K, **kw):
    """TOVA per-head (Oren ACL'24, App.A): top-K by current per-head attention."""
    nh, nk = attn_ph_layer_step.shape
    if nk <= K:
        return np.ones((nh, nk), dtype=bool)
    m = np.zeros((nh, nk), dtype=bool)
    for h in range(nh):
        idx = np.argpartition(-attn_ph_layer_step[h], K)[:K]
        m[h, idx] = True
    return m


def policy_keydiff(attn_ph_layer_step, K, K_cache=None, **kw):
    """KeyDiff (Park et al., arXiv:2504.15364, Apr 2025) reimplementation.

    Mechanism: rank positions by *uniqueness* of their K vector — positions whose
    K vector is most redundant with other Ks are evicted first. Per-head: each
    head's K subspace is treated independently.

    Faithful interpretation: for each (layer, head), compute pairwise cosine
    similarity matrix S of K vectors at this layer & head. Position importance =
    1 / (1 + mean of off-diagonal similarities). Keep top-K by importance.

    K_cache shape (passed in): [n_kv, n_head_kv, head_dim] (we slice it for this layer).
    For GQA models the per-head K is shared across grouped query heads; we replicate
    so each query head sees the same K subspace (which matches the paper's intent).
    """
    nh, nk = attn_ph_layer_step.shape
    if nk <= K or K_cache is None:
        # Fallback to TOVA if no K available
        return policy_perhead_tova(attn_ph_layer_step, K)
    # K_cache here is sliced to current n_kv: shape [n_kv, n_head_kv, dim]
    if K_cache.shape[0] < nk:
        return policy_perhead_tova(attn_ph_layer_step, K)
    K_now = K_cache[:nk]                       # [nk, n_head_kv, dim]
    n_head_kv = K_now.shape[1]
    # GQA-aware: map query-head h to kv-head h % n_head_kv
    m = np.zeros((nh, nk), dtype=bool)
    for h in range(nh):
        kh = h % n_head_kv
        K_h = K_now[:, kh, :]                  # [nk, dim]
        norms = np.linalg.norm(K_h, axis=1, keepdims=True) + 1e-9
        K_n = K_h / norms                      # unit-norm
        # Cosine sim: [nk, nk]. For large nk this is heavy — cap to fast path.
        if nk > 4096:
            # Approximation: sample 1024 random rows for similarity comparison
            rng = np.random.default_rng(42)
            idx = rng.choice(nk, 1024, replace=False)
            sims = K_n @ K_n[idx].T            # [nk, 1024]
            redundancy = sims.mean(axis=1)
        else:
            sims = K_n @ K_n.T                 # [nk, nk]
            np.fill_diagonal(sims, 0.0)
            redundancy = sims.mean(axis=1)
        importance = 1.0 / (1.0 + redundancy)
        idx_top = np.argpartition(-importance, K)[:K]
        m[h, idx_top] = True
    return m


def policy_laprox(attn_ph_layer_step, K, V_cache=None, W_o=None, **kw):
    """LaProx (Mai & Kim, arXiv:2605.07234, May 2026) reimplementation.

    Mechanism: rank positions by their contribution to the output projection.
    Per-position importance ≈ attention_h[i] · ||W_o^{(h)} · V_h[i]||_2
    averaged or summed across heads (paper says "globally comparable importance
    scores enabling model-wide token selection").

    Faithful approximation: importance[i] = Σ_h attention[h,i] · ||V_h[i]||_2
    if W_o is not provided. If W_o is provided, importance[i] = Σ_h attention[h,i] ·
    ||W_o^{(h)} · V_h[i]||_2 (head-wise output projection).

    Then keep top-K globally — i.e. each head uses the same set S = top-K of
    importance, matching the paper's "global token selection" design.
    """
    nh, nk = attn_ph_layer_step.shape
    if nk <= K or V_cache is None:
        return policy_perhead_tova(attn_ph_layer_step, K)
    if V_cache.shape[0] < nk:
        return policy_perhead_tova(attn_ph_layer_step, K)
    V_now = V_cache[:nk]                       # [nk, n_head_kv, dim]
    n_head_kv = V_now.shape[1]
    head_dim = V_now.shape[2]

    if W_o is not None:
        # W_o shape: [hidden_size, hidden_size] or [n_head * head_dim, hidden_size]
        # Per-head slice: W_o_h = W_o[h*head_dim:(h+1)*head_dim, :]
        # Output contribution magnitude per (h, i):
        #   contrib_norm[h, i] = ||W_o_h @ V_now[i, h, :]||_2
        # = ||V_now[i, h, :] @ W_o_h||_2     (if W_o is hidden×hidden)
        # We avoid materializing full output and just use per-head V norm projected.
        contrib = np.zeros((nh, nk), dtype=np.float32)
        for h in range(nh):
            kh = h % n_head_kv
            V_h = V_now[:, kh, :]              # [nk, dim]
            try:
                W_o_h = W_o[h*head_dim:(h+1)*head_dim, :]   # [dim, hidden]
                proj = V_h @ W_o_h             # [nk, hidden]
                contrib[h] = np.linalg.norm(proj, axis=1)
            except Exception:
                # Fall back to V norm if W_o slicing fails
                contrib[h] = np.linalg.norm(V_h, axis=1)
    else:
        # No W_o available — use ||V_h||_2 as a proxy for output contribution.
        contrib = np.zeros((nh, nk), dtype=np.float32)
        for h in range(nh):
            kh = h % n_head_kv
            contrib[h] = np.linalg.norm(V_now[:, kh, :], axis=1)

    # Per-position global importance = sum over heads of (attention × contribution)
    importance = (attn_ph_layer_step.astype(np.float32) * contrib).sum(axis=0)  # [nk]
    idx_top = np.argpartition(-importance, K)[:K]
    # ALL heads keep the same top-K (matches LaProx "global" framing)
    m = np.zeros((nh, nk), dtype=bool)
    m[:, idx_top] = True
    return m


def policy_rkv(attn_ph_layer_step, K, K_cache=None, lambda_rkv=0.1, **kw):
    """R-KV (Cai et al., NeurIPS 2025) reimplementation.

    Score per (head, position) = λ · Â − (1 − λ) · R̂, where:
      Â = current-step attention received by this position (L1-normalized over positions),
      R̂ = softmax of mean cosine similarity to other K vectors at this layer & head.
    Paper default λ = 0.1 (heavy redundancy weighting). Per-head ranking — each query
    head keeps its own top-K under the joint score.

    Comparison point: faithful published baseline for reasoning-model decoding-time
    eviction. Combines KeyDiff-style K-redundancy with TOVA-style current attention.
    """
    nh, nk = attn_ph_layer_step.shape
    if nk <= K or K_cache is None:
        return policy_perhead_tova(attn_ph_layer_step, K)
    if K_cache.shape[0] < nk:
        return policy_perhead_tova(attn_ph_layer_step, K)
    K_now = K_cache[:nk]
    n_head_kv = K_now.shape[1]
    m = np.zeros((nh, nk), dtype=bool)
    for h in range(nh):
        kh = h % n_head_kv
        K_h = K_now[:, kh, :]
        norms = np.linalg.norm(K_h, axis=1, keepdims=True) + 1e-9
        K_n = K_h / norms
        if nk > 4096:
            rng = np.random.default_rng(42)
            idx = rng.choice(nk, 1024, replace=False)
            sims = K_n @ K_n[idx].T
            redundancy = sims.mean(axis=1)
        else:
            sims = K_n @ K_n.T
            np.fill_diagonal(sims, 0.0)
            redundancy = sims.mean(axis=1)
        a = attn_ph_layer_step[h].astype(np.float32)
        a_norm = a / (a.sum() + 1e-9)
        r_shift = redundancy.astype(np.float32) - float(redundancy.max())
        r_exp = np.exp(r_shift)
        r_norm = r_exp / (r_exp.sum() + 1e-9)
        score = lambda_rkv * a_norm - (1.0 - lambda_rkv) * r_norm
        idx_top = np.argpartition(-score, K)[:K]
        m[h, idx_top] = True
    return m


def policy_kvzip_approx(attn_ph_layer_step, K, attn_cum_max=None, **kw):
    """KVzip (Kim et al., NeurIPS 2025) decoding-time approximation.

    The published KVzip computes importance via attention received during a
    *reconstruction* prefill (forward pass with prompt "Repeat the previous context:"
    + the context itself). Our probe does not run that extra pass.

    Closest no-prefill approximation: importance per (head, position) =
        max over all decoding steps so far of attention this position received.
    This captures the "any query that ever cared about this position" intuition,
    parallel to KVzip's cross-attention max. Strictly weaker than full KVzip — flagged
    in the comparison table as "KVzip-decode-approx" not "KVzip".

    `attn_cum_max` shape: [n_head, n_kv], updated in the simulator each step.
    """
    nh, nk = attn_ph_layer_step.shape
    if nk <= K:
        return np.ones((nh, nk), dtype=bool)
    score = attn_cum_max if attn_cum_max is not None else attn_ph_layer_step
    if score.shape[1] != nk:
        score = score[:, :nk]
    m = np.zeros((nh, nk), dtype=bool)
    for h in range(nh):
        idx = np.argpartition(-score[h], K)[:K]
        m[h, idx] = True
    return m


def policy_perhead_v2(attn_ph_layer_step, K,
                      alpha_low=0.7, alpha_high=1.3,
                      gamma=8.0, c0=0.30, lam=0.5, **kw):
    """EndurKV-Evict v2 — Participation Resonance Gate + edge-aware ranking.

    Two principled ideas:
      (a) Pythagoras / Boltzmann: use the participation ratio PR_h = 1/Σa[h,k]²
          (effective number of attended positions, quantum-mechanics analog)
          as the sharpness signal — strictly more informative than max_a.
      (b) Surface tension: weight each position by its discrete Laplacian
          |a[k+1] − 2a[k] + a[k−1]|, so positions at attention edges (where the
          attention pattern changes rapidly) are prioritized within a head.

    The gate is a logistic sigmoid (smooth) instead of a clipped linear.
    """
    nh, nk = attn_ph_layer_step.shape
    if nk <= K:
        return np.ones((nh, nk), dtype=bool)
    m = np.zeros((nh, nk), dtype=bool)
    eps = 1e-12
    for h in range(nh):
        a = attn_ph_layer_step[h].astype(np.float32)
        s = float(a.sum())
        if s <= eps:
            m[h, :K] = True
            continue
        a_norm = a / s
        # (a) Participation ratio → sharpness score in [0, 1]
        sum_sq = float((a_norm * a_norm).sum()) + eps
        pr = 1.0 / sum_sq
        pr_norm = (pr - 1.0) / max(nk - 1.0, 1.0)         # ∈ [0, 1]
        sharpness = 1.0 - pr_norm                          # 0=diffuse, 1=sharp
        # (b) Smooth logistic gate
        x = gamma * (sharpness - c0)
        sig = 1.0 / (1.0 + np.exp(-x))
        mult = alpha_high - (alpha_high - alpha_low) * sig
        K_t = max(1, min(nk, int(round(K * mult))))
        # (c) Edge-aware importance: a[k] · (1 + λ · |Laplacian| / mean)
        if lam > 0 and nk >= 3:
            lap = np.zeros(nk, dtype=np.float32)
            lap[1:-1] = np.abs(a_norm[2:] - 2.0 * a_norm[1:-1] + a_norm[:-2])
            lap[0] = lap[1]
            lap[-1] = lap[-2]
            lap_mean = float(lap.mean()) + eps
            imp = a_norm * (1.0 + lam * lap / lap_mean)
        else:
            imp = a_norm
        idx = np.argpartition(-imp, K_t - 1)[:K_t]
        m[h, idx] = True
    return m


def policy_perhead_v3(attn_ph_layer_step, K,
                      alpha_low=0.75, alpha_high=1.25,
                      low_freq_fraction=0.125, **kw):
    """EndurKV-Evict v3 — Spectral Resonance Gate (Tesla / Fourier).

    A smooth attention pattern (diffuse head) has most of its energy
    concentrated in the LOW-frequency band of its DFT — like the spectrum
    of a low-pass filter. A sharp single-spike attention (delta) has flat
    DFT magnitude across all frequencies. So:
        diffuse  →  high low-frequency energy fraction  →  give MORE cache
        sharp    →  flat spectrum, low low-freq fraction →  give LESS cache

    This is the spectral analogue of participation ratio — captures the
    full distribution shape via its Fourier signature, not just the peak.
    """
    nh, nk = attn_ph_layer_step.shape
    if nk <= K:
        return np.ones((nh, nk), dtype=bool)
    m = np.zeros((nh, nk), dtype=bool)
    n_low = max(2, int(low_freq_fraction * (nk // 2 + 1)))
    for h in range(nh):
        a = attn_ph_layer_step[h].astype(np.float32)
        s = float(a.sum())
        if s <= 1e-12:
            m[h, :K] = True
            continue
        a_norm = a / s
        # Real-to-real DFT (rfft) of attention sequence
        spec = np.abs(np.fft.rfft(a_norm))
        spec_sq = spec * spec
        total_energy = float(spec_sq.sum()) + 1e-12
        low_energy = float(spec_sq[:n_low].sum())
        low_frac = low_energy / total_energy            # ∈ [0, 1]
        # Linear gate: diffuse (high low_frac) gets alpha_high
        mult = alpha_low + (alpha_high - alpha_low) * low_frac
        K_t = max(1, min(nk, int(round(K * mult))))
        idx = np.argpartition(-a_norm, K_t - 1)[:K_t]
        m[h, idx] = True
    return m


def policy_perhead_v4(attn_ph_layer_step, K, iters=3, **kw):
    """EndurKV-Evict v4 — Newton-Raphson budget redistribution.

    Iteratively allocate the total budget K·n_heads across heads in proportion
    to the *marginal attention value* of the next position each head would
    keep. Heads where the K-th-best position carries a lot of attention "want"
    more cache; heads where it's near zero need less. Total budget stays
    constant (it's pure redistribution, like a water-level adjustment).
    """
    nh, nk = attn_ph_layer_step.shape
    if nk <= K:
        return np.ones((nh, nk), dtype=bool)
    m = np.zeros((nh, nk), dtype=bool)
    total = int(K) * nh
    # Sort attention per head descending (so position K_h is the marginal one)
    sorted_idx = np.argsort(-attn_ph_layer_step, axis=1)
    sorted_a = np.take_along_axis(attn_ph_layer_step, sorted_idx, axis=1)
    # Initial budget = uniform K per head
    K_h = np.full(nh, K, dtype=np.int64)
    for _ in range(iters):
        # Marginal value of next position each head would keep
        ks = np.clip(K_h, 1, nk) - 1
        marg = sorted_a[np.arange(nh), ks].astype(np.float64) + 1e-12
        # New K_h proportional to marginal (water-filling allocation)
        share = marg / marg.sum()
        K_h = (share * total).round().astype(np.int64)
        K_h = np.clip(K_h, 1, nk)
        # Re-balance to keep total constant
        diff = total - int(K_h.sum())
        if diff != 0:
            # Adjust the heads with the largest fractional rounding error
            order = np.argsort(-marg if diff > 0 else marg)
            for i in range(abs(diff)):
                K_h[order[i % nh]] += 1 if diff > 0 else -1
            K_h = np.clip(K_h, 1, nk)
    for h in range(nh):
        m[h, sorted_idx[h, :K_h[h]]] = True
    return m


def policy_perhead_v5(attn_ph_layer_step, K,
                      alpha_low=0.75, alpha_high=1.25, **kw):
    """EndurKV-Evict v5 — Multi-scale Rényi concentration (L4/L2 ratio).

    The ratio  ||a||_4 / ||a||_2  is a higher-order concentration measure:
      uniform a (all 1/n_kv):  L4/L2 = n_kv^(-1/4)         (small)
      single-spike a:          L4/L2 = 1                    (max)
    It captures distribution *shape* beyond what max_a or PR alone see —
    distinguishes "one giant peak + tail" from "two medium peaks + tail",
    which the participation ratio collapses.
    """
    nh, nk = attn_ph_layer_step.shape
    if nk <= K:
        return np.ones((nh, nk), dtype=bool)
    m = np.zeros((nh, nk), dtype=bool)
    floor = nk ** (-0.25)
    for h in range(nh):
        a = attn_ph_layer_step[h].astype(np.float32)
        s = float(a.sum())
        if s <= 1e-12:
            m[h, :K] = True
            continue
        a_norm = a / s
        l2 = float((a_norm ** 2).sum()) ** 0.5
        l4 = float((a_norm ** 4).sum()) ** 0.25
        ratio = l4 / (l2 + 1e-12)                          # ∈ [floor, 1]
        sharpness = (ratio - floor) / (1.0 - floor + 1e-12)
        sharpness = max(0.0, min(1.0, sharpness))
        mult = alpha_high - (alpha_high - alpha_low) * sharpness
        K_t = max(1, min(nk, int(round(K * mult))))
        idx = np.argpartition(-a_norm, K_t - 1)[:K_t]
        m[h, idx] = True
    return m


def policy_perhead_v6(attn_ph_layer_step, K,
                      alpha_low=0.7, alpha_high=1.3,
                      gamma=10.0, c0=0.6, **kw):
    """EndurKV-Evict v6 — Logistic gate on max_a (smooth ablation of v1).

    Same SIGNAL as v1 (max_a), but a smooth logistic transition replaces
    v1's clipped-linear gate. Tests whether the clipping was hurting or
    helping. Pure ablation; identical-cost compute as v1.
    """
    nh, nk = attn_ph_layer_step.shape
    if nk <= K:
        return np.ones((nh, nk), dtype=bool)
    m = np.zeros((nh, nk), dtype=bool)
    for h in range(nh):
        a = attn_ph_layer_step[h]
        max_a = float(a.max())
        x = gamma * (max_a - c0)
        sig = 1.0 / (1.0 + np.exp(-x))
        mult = alpha_high - (alpha_high - alpha_low) * sig
        K_t = max(1, min(nk, int(round(K * mult))))
        idx = np.argpartition(-a, K_t - 1)[:K_t]
        m[h, idx] = True
    return m


def policy_perhead_v7(attn_ph_layer_step, K,
                      alpha_low=0.70, alpha_high=1.30,
                      mu=0.40, sigma=0.25, **kw):
    """EndurKV-Evict v7 — Leidenfrost Gate (bell-curve allocation).

    Physics analogy: like water on a frying pan, the "interaction rate"
    between cache budget and attention information is NON-MONOTONIC in
    head sharpness. Both extremes (very sharp / very diffuse) need LESS
    cache:
      * very sharp head (max_a ≈ 0.95) — one big spike captures it all,
        tail is throw-away noise
      * very diffuse head (max_a ≈ 0.05) — almost-uniform attention, likely
        a "noise-floor" head doing background smoothing, evicting positions
        barely changes its output
      * medium head (max_a ≈ 0.3-0.5) — structured multi-peak attention
        carrying real retrieval signal across several positions — these
        need the most cache (the "Leidenfrost peak")

    Multiplier: Gaussian bump centered at max_a = μ, falling toward α_low
    at both extremes.
    """
    nh, nk = attn_ph_layer_step.shape
    if nk <= K:
        return np.ones((nh, nk), dtype=bool)
    m = np.zeros((nh, nk), dtype=bool)
    inv2s2 = 1.0 / (2.0 * sigma * sigma)
    for h in range(nh):
        a = attn_ph_layer_step[h]
        max_a = float(a.max())
        bell = float(np.exp(-((max_a - mu) ** 2) * inv2s2))   # ∈ (0, 1]
        mult = alpha_low + (alpha_high - alpha_low) * bell
        K_t = max(1, min(nk, int(round(K * mult))))
        idx = np.argpartition(-a, K_t - 1)[:K_t]
        m[h, idx] = True
    return m


POLICIES = {
    "perhead_v1":   policy_perhead_v1,
    "perhead_v2":   policy_perhead_v2,
    "perhead_v3":   policy_perhead_v3,
    "perhead_v4":   policy_perhead_v4,
    "perhead_v5":   policy_perhead_v5,
    "perhead_v6":   policy_perhead_v6,
    "perhead_v7":   policy_perhead_v7,
    "perhead_tova": policy_perhead_tova,
    "keydiff":      policy_keydiff,
    "laprox":       policy_laprox,
    "rkv":          policy_rkv,
    "kvzip_approx": policy_kvzip_approx,
}


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

def simulate(attn_ph, n_kv_at, kv_data, w_o_per_layer, policy_name, K_nominal,
             needle_positions=None):
    n_steps, n_layers, n_head, max_kv = attn_ph.shape
    fn = POLICIES[policy_name]
    out_kl = np.zeros((n_steps, n_layers), dtype=np.float64)
    out_K = np.zeros((n_steps, n_layers), dtype=np.int32)
    # Attention mass retained per (step, layer): Σ_kept attention / Σ_all attention,
    # averaged across heads. 1.0 = lossless, 0.0 = all important mass evicted.
    # Reader-friendlier than KL — "we retain 96% of attention mass at 25% cache".
    out_mass = np.zeros((n_steps, n_layers), dtype=np.float64)
    # Per-(layer, head, step) needle retention for NIAH: did the policy keep the
    # ground-truth answer position? Populated only if needle_positions is given
    # via the simulate() caller. Stored as float (1.0 kept, 0.0 evicted) so
    # averages give straight hit-rate. nan = no needle for this step.
    out_needle_hit = np.full((n_steps, n_layers), np.nan, dtype=np.float64)
    # Running max attention per (layer, head, position). Used by policies that
    # need "max attention received across history" (KVzip-decode-approx).
    attn_cum_max = np.zeros((n_layers, n_head, max_kv), dtype=np.float32)
    for s in range(n_steps):
        n_kv = int(n_kv_at[s])
        if n_kv == 0: continue
        for l in range(n_layers):
            ph_slice = attn_ph[s, l, :, :n_kv]
            np.maximum(attn_cum_max[l, :, :n_kv], ph_slice,
                       out=attn_cum_max[l, :, :n_kv])
            kw = {"attn_cum_max": attn_cum_max[l, :, :n_kv]}
            # Provide K/V if available for this layer
            if kv_data is not None and l in kv_data:
                lc = kv_data[l]
                K_ne = lc.get("K_ne"); V_ne = lc.get("V_ne")
                # K_ne layout from ggml: (head_dim, n_kv, n_head_kv, 1). After
                # reshape(ne[::-1]) the array is (1, n_head_kv, n_kv, head_dim).
                # Drop batch and transpose to (n_kv, n_head_kv, head_dim).
                if lc.get("K") is not None and lc["K"].ndim == 4:
                    kw["K_cache"] = lc["K"][0].transpose(1, 0, 2)
                # V is captured in the v_trans layout used by the FA-off path:
                # V_ne = (n_kv, head_dim, n_head_kv, 1). After reshape(ne[::-1])
                # the array is (1, n_head_kv, head_dim, n_kv). Transpose to
                # (n_kv, n_head_kv, head_dim) to match K's axis order.
                if lc.get("V") is not None and lc["V"].ndim == 4:
                    V4 = lc["V"][0]  # (n_head_kv, head_dim, n_kv)
                    # Detect v_trans by comparing V_ne[0] vs K_ne[0]: if they
                    # disagree, V is transposed.
                    if K_ne is not None and V_ne is not None and V_ne[0] != K_ne[0]:
                        kw["V_cache"] = V4.transpose(2, 0, 1)  # (n_kv, n_head_kv, head_dim)
                    else:
                        kw["V_cache"] = V4.transpose(1, 0, 2)  # same as K
                if w_o_per_layer is not None and l in w_o_per_layer:
                    kw["W_o"] = w_o_per_layer[l]
            mask = fn(ph_slice, K_nominal, **kw)
            kls = []
            masses = []
            kept_sum = 0
            for h in range(n_head):
                p_full = ph_slice[h]
                kept = p_full * mask[h]
                tot_kept = kept.sum()
                tot_full = p_full.sum()
                if tot_full > 0:
                    masses.append(float(tot_kept / tot_full))
                else:
                    masses.append(1.0)  # nothing to retain, vacuously perfect
                if tot_kept <= 0:
                    kls.append(20.0)
                else:
                    kls.append(kl(p_full, kept / tot_kept))
                kept_sum += int(mask[h].sum())
            out_kl[s, l] = float(np.mean(kls))
            out_mass[s, l] = float(np.mean(masses))
            out_K[s, l]  = kept_sum / n_head
            # Needle retention: did the policy keep the ground-truth answer
            # position for this prompt? mask[h, p*] across all heads.
            if needle_positions is not None and s < len(needle_positions):
                p_star = needle_positions[s]
                if p_star is not None and 0 <= p_star < n_kv:
                    out_needle_hit[s, l] = float(mask[:, p_star].mean())
    return out_kl, out_K, out_mass, out_needle_hit


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", required=True,
                    help="Dir with *.attn.bin (ATNH) and *.kv.bin (KVCP) per prompt")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--budgets", default="64,128,256")
    ap.add_argument("--policies", default="perhead_v1,perhead_tova,keydiff,laprox")
    ap.add_argument("--max-prompts", type=int, default=0)
    ap.add_argument("--w-o-dir", default="",
                    help="Dir with per-layer W_o weights (one .npy per layer named layer_<idx>.npy). "
                         "If unset, LaProx falls back to ||V||_2 proxy.")
    ap.add_argument("--needles", default="",
                    help="JSONL with one record per prompt_id giving 'needle_position' "
                         "(absolute KV index where the ground-truth answer lives, valid for "
                         "NIAH and RULER tasks). Enables per-policy needle-hit-rate output.")
    ap.add_argument("--model", default="",
                    help="Model key for KV-memory conversion (e.g. llama3.1-8b, phi3-mini-4k, "
                         "qwen2-7b). If unset, inferred from log-dir name. See model_arch_specs.py.")
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir) if args.out_dir else log_dir.parent / f"{log_dir.name}_kvsim"
    out_dir.mkdir(parents=True, exist_ok=True)
    budgets  = [int(b) for b in args.budgets.split(",")]
    policies = [p.strip() for p in args.policies.split(",")]

    # Model-arch lookup for memory column.
    try:
        from model_arch_specs import MODEL_SPECS, kv_memory_per_head_mb, infer_model_from_dir
        model_key = args.model.strip() or infer_model_from_dir(log_dir.name)
        if model_key and model_key in MODEL_SPECS:
            spec = MODEL_SPECS[model_key]
            print(f"[sim-kv] model={model_key} ({spec['attn_kind']}, "
                  f"{spec['n_layers']}L × {spec['n_head_kv']} KV heads × {spec['head_dim']}d)")
        else:
            spec = None
            print(f"[sim-kv] no model spec resolved (--model unset, dir-infer failed); "
                  f"kv_memory_mb will be 0.0")
    except ImportError:
        spec = None
        model_key = ""

    w_o_per_layer = None
    if args.w_o_dir:
        w_o_dir = Path(args.w_o_dir)
        if w_o_dir.is_dir():
            w_o_per_layer = {}
            for f in sorted(w_o_dir.glob("layer_*.npy")):
                idx = int(f.stem.split("_")[1])
                w_o_per_layer[idx] = np.load(f)
            print(f"[sim-kv] loaded W_o for {len(w_o_per_layer)} layers from {w_o_dir}")

    # Optional needle metadata for NIAH/RULER answer-hit-rate. Accepts either
    #   {"prompt_id":..., "needle_position": int}  (absolute KV index — exact)
    # or
    #   {"prompt_id":..., "needle_fraction": float in [0,1]}  (char-position based;
    #     converted to absolute index at run time using per-capture n_kv)
    needle_by_pid = {}     # absolute position
    fraction_by_pid = {}   # fractional position
    if args.needles:
        import json
        for line in Path(args.needles).read_text().splitlines():
            line = line.strip()
            if not line: continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = r.get("prompt_id")
            if pid is None: continue
            if "needle_position" in r and r["needle_position"] is not None:
                needle_by_pid[pid] = int(r["needle_position"])
            elif "needle_fraction" in r and r["needle_fraction"] is not None:
                fraction_by_pid[pid] = float(r["needle_fraction"])
        print(f"[sim-kv] loaded needle metadata for {len(needle_by_pid) + len(fraction_by_pid)} "
              f"prompts ({len(needle_by_pid)} absolute, {len(fraction_by_pid)} fractional)")

    attn_files = sorted(log_dir.glob("*.attn.bin"))
    attn_files = [f for f in attn_files if not f.name.endswith(".v1.attn.bin")]
    if args.max_prompts > 0: attn_files = attn_files[:args.max_prompts]
    if not attn_files:
        print(f"ERROR: no *.attn.bin in {log_dir}", file=sys.stderr); return 1
    print(f"[sim-kv] {len(attn_files)} attn files, budgets={budgets}, policies={policies}")

    rows = []
    for af in attn_files:
        pid = af.name[:-len(".attn.bin")]
        attn_ph, n_kv_at = load_attn_perhead(af)
        if attn_ph is None:
            print(f"  [skip] {pid}: not ATNH format"); continue
        kv_path = af.parent / f"{pid}.kv.bin"
        kv_data = load_kv_sidecar(kv_path)
        if kv_data is None:
            print(f"  [warn] {pid}: no .kv.bin (KeyDiff + LaProx will fall back to TOVA)")
        n_steps, n_layers, n_head, max_kv = attn_ph.shape
        task = pid.rsplit("_", 1)[0]
        if task.endswith("_lc") or task.endswith("_pub"):
            task = task.rsplit("_", 1)[0]
        print(f"  {pid:<32} steps={n_steps} layers={n_layers} heads={n_head} max_kv={max_kv}")
        # Build per-step needle position vector for this prompt, if known.
        # Needle is in the prompt (fixed position in the prefix), so the same
        # absolute index is valid for every decode step.
        needle_pos_per_step = None
        if pid in needle_by_pid:
            needle_pos_per_step = [needle_by_pid[pid]] * n_steps
        elif pid in fraction_by_pid:
            # Convert char-fraction to absolute KV index using n_kv at the
            # final step (= prompt_len). Approximation accurate to ~1-2% of
            # prompt length, fine for eviction-granularity hit rate.
            n_kv_prompt = int(n_kv_at[-1]) if len(n_kv_at) else max_kv
            pos = int(round(fraction_by_pid[pid] * n_kv_prompt))
            pos = max(0, min(n_kv_prompt - 1, pos))
            needle_pos_per_step = [pos] * n_steps
        for K in budgets:
            for pname in policies:
                kls, Ks, mass, needle_hit = simulate(
                    attn_ph, n_kv_at, kv_data, w_o_per_layer, pname, K,
                    needle_positions=needle_pos_per_step,
                )
                # needle_hit is nan where no needle was provided.
                hit_vals = needle_hit[~np.isnan(needle_hit)]
                needle_hit_rate = float(hit_vals.mean()) if hit_vals.size else float("nan")
                avg_actual_K = float(np.mean(Ks))
                # Memory occupancy after eviction: per-head kept positions ×
                # bytes-per-token-per-head-per-layer × n_layers. Uses fp16.
                if spec is not None:
                    kv_memory_mb = kv_memory_per_head_mb(model_key, avg_actual_K)
                    # Full-cache memory at this n_kv (no eviction) for ratio reporting
                    full_kv_mb = kv_memory_per_head_mb(
                        model_key, float(np.mean(n_kv_at)))
                else:
                    kv_memory_mb = 0.0
                    full_kv_mb = 0.0
                rows.append({"prompt_id": pid, "task": task, "policy": pname,
                             "model": model_key,
                             "K_nominal": K,
                             "mean_kl": float(np.mean(kls)),
                             "attn_mass_retained": float(np.mean(mass)),
                             "needle_hit_rate": needle_hit_rate,
                             "avg_actual_K": avg_actual_K,
                             "avg_n_kv": float(np.mean(n_kv_at)),
                             "kv_memory_mb": kv_memory_mb,
                             "full_kv_memory_mb": full_kv_mb,
                             "memory_saving_pct": (
                                 100.0 * (1.0 - kv_memory_mb / full_kv_mb)
                                 if full_kv_mb > 0 else 0.0),
                             "n_steps": n_steps, "n_layers": n_layers,
                             "n_prompts": len(attn_files)})
    df = pd.DataFrame(rows)
    if df.empty:
        print("[sim-kv] no rows produced"); return 1
    # Aggregate to pareto_summary.csv. Add attn_mass_retained (headline-friendly)
    # and needle_hit_rate (NIAH-style, NaN-aware mean).
    def _nanmean(s):
        return float(np.nanmean(s.values)) if s.notna().any() else float("nan")
    agg = (df.groupby(["policy", "K_nominal"])
             .agg(avg_actual_K=("avg_actual_K", "mean"),
                  avg_n_kv=("avg_n_kv", "mean"),
                  mean_kl=("mean_kl", "mean"),
                  attn_mass_retained=("attn_mass_retained", "mean"),
                  needle_hit_rate=("needle_hit_rate", _nanmean),
                  kv_memory_mb=("kv_memory_mb", "mean"),
                  full_kv_memory_mb=("full_kv_memory_mb", "mean"),
                  memory_saving_pct=("memory_saving_pct", "mean"),
                  n_prompts=("n_prompts", "first"))
             .reset_index())
    agg.to_csv(out_dir / "pareto_summary.csv", index=False)
    df.to_csv(out_dir / "policy_results.csv", index=False)
    print(f"[sim-kv] wrote {len(agg)} aggregate rows -> {out_dir / 'pareto_summary.csv'}")
    print(agg.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
