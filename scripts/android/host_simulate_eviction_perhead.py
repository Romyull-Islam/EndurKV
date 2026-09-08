"""Per-head eviction-policy Pareto simulator.

Consumes the v2 ATNH .attn.bin format from the upgraded attention_probe
(attention_probe_v2_perhead), which dumps per-head softmax attention.

This script is intentionally a sibling of host_simulate_eviction_policies_v2.py
(the layer-averaged sim), not a merge of the two — separation keeps both code
paths legible.

Policies implemented here (none of which can be expressed in the layer-averaged
data; each one uses per-head signals):

  perhead_tova         — TOVA scoring applied independently per head
  perhead_v1           — v1 spread gate + TOVA, applied per head
  adakv                — Ada-KV (Feng 2024): per-head budget proportional to
                         per-head attention sharpness; total budget = K * n_head
  headkv               — HeadKV (Fu 2024): per-head importance from accumulated
                         attention, sorted heads by importance then per-head TOVA
  duoattention_lite    — DuoAttention (Xiao 2025) shadow: profile heads into
                         "retrieval" (keep full) vs "streaming" (sliding window)
                         based on cumulative attention concentration
  endurkv_perhead      — *** our candidate *** v1 spread gate per head + a
                         "head-disagreement protection" rule: any KV position
                         that some head finds important (top-K in that head)
                         AND most heads agree on (cross-head consensus) is
                         hard-protected. This is the unclaimed signal that no
                         published paper uses — heads disagree where layers
                         agree, and we exploit the disagreement directly.

KL metric: per-head KL between full and evicted distribution, averaged over
heads. Each head's softmax is its own distribution, so this is the faithful
multi-head extension of the layer-averaged metric.
"""
from __future__ import annotations
import argparse
import os
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WORKSPACE = Path(os.environ.get("WORKSPACE", r"D:/Research/EndurKV_workspace"))


# ============================================================================
# binary loader (ATNH = v2)
# ============================================================================

def load_attn_perhead(path: Path):
    """Returns (attn_ph[steps,layers,heads,max_kv], n_kv_at[steps]) or (None, None)."""
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
    # fp16: per-head attention sums to 1.0 per head, so values are in [0,1] —
    # fp16's 11 mantissa bits give precision ~5e-4, fine for our top-K selections.
    # Halves memory: 32 layers x 32 heads x 12K kv x 64 steps fits in ~1.5 GB instead of 3 GB.
    attn_ph = np.zeros((n_steps, n_layers, n_head, max_kv), dtype=np.float16)
    for s, layers in enumerate(per_step):
        for l, arr in enumerate(layers):
            if arr is not None:
                attn_ph[s, l, :, :arr.shape[1]] = arr.astype(np.float16)
    return attn_ph, np.array(n_kv_at, dtype=np.int32)


def kl(p_full: np.ndarray, p_evicted: np.ndarray, eps: float = 1e-12) -> float:
    # Promote to float32 before clipping — fp16 cannot represent 1e-12 (gets
    # rounded to 0, then log(0) = -inf and the result becomes NaN).
    pf = np.clip(p_full.astype(np.float32, copy=False), eps, 1.0)
    pe = np.clip(p_evicted.astype(np.float32, copy=False), eps, 1.0)
    return float(np.sum(pf * (np.log(pf) - np.log(pe))))


# ============================================================================
# per-head policies. Each takes per-head attention [n_head, n_kv] and a budget K
# (interpreted as PER-HEAD budget unless the policy reallocates across heads).
# Returns boolean mask [n_head, n_kv] indicating positions kept per head.
# ============================================================================

def mph_full(attn_ph: np.ndarray, K: int, **kw) -> np.ndarray:
    nh, nk = attn_ph.shape
    return np.ones((nh, nk), dtype=bool)


def mph_local(attn_ph: np.ndarray, K: int, **kw) -> np.ndarray:
    nh, nk = attn_ph.shape
    m = np.zeros((nh, nk), dtype=bool)
    start = max(0, nk - K)
    m[:, start:] = True
    return m


def mph_tova(attn_ph: np.ndarray, K: int, **kw) -> np.ndarray:
    """TOVA per head: each head independently keeps top-K by current attention."""
    nh, nk = attn_ph.shape
    if nk <= K:
        return np.ones((nh, nk), dtype=bool)
    m = np.zeros((nh, nk), dtype=bool)
    for h in range(nh):
        idx = np.argpartition(-attn_ph[h], K)[:K]
        m[h, idx] = True
    return m


def mph_v1(attn_ph: np.ndarray, K: int, **kw) -> np.ndarray:
    """v1 spread gate + TOVA, per head."""
    nh, nk = attn_ph.shape
    if nk <= K:
        return np.ones((nh, nk), dtype=bool)
    m = np.zeros((nh, nk), dtype=bool)
    for h in range(nh):
        a = attn_ph[h]
        max_a = float(a.max())
        norm = max(0.0, min(1.0, (max_a - 0.4) / 0.4))
        mult = 1.3 - 0.6 * norm
        K_t = max(1, min(nk, int(round(K * mult))))
        idx = np.argpartition(-a, K_t)[:K_t]
        m[h, idx] = True
    return m


def mph_adakv(attn_ph: np.ndarray, K: int, accum_ph: np.ndarray | None = None,
              **kw) -> np.ndarray:
    """Ada-KV (Feng 2024): per-head budget proportional to head's attention sharpness.

    Sharpness proxy: 1 - normalized_entropy. Sharp heads get MORE budget (because
    they're more discriminating), diffuse heads less. Total cache = K * n_head
    (matching what plain per-head TOVA spends).
    """
    nh, nk = attn_ph.shape
    if nk <= K:
        return np.ones((nh, nk), dtype=bool)
    # Per-head sharpness (1 - normalized entropy)
    sharp = np.zeros(nh)
    log_nk = max(1e-9, np.log(max(2, nk)))
    for h in range(nh):
        p = np.clip(attn_ph[h].astype(np.float32, copy=False), 1e-12, 1.0)
        H = -float(np.sum(p * np.log(p)))
        sharp[h] = 1.0 - H / log_nk
    # Allocate total = K*nh in proportion to sharp; floor at 0.5 K, ceil at 1.5 K
    total = K * nh
    if sharp.sum() > 1e-9:
        budgets = sharp / sharp.sum() * total
    else:
        budgets = np.full(nh, K, dtype=np.float64)
    budgets = np.clip(budgets, 0.5 * K, 1.5 * K)
    # Renormalize to total
    budgets = budgets * (total / max(1.0, budgets.sum()))
    budgets = np.maximum(1, budgets.astype(np.int32))

    m = np.zeros((nh, nk), dtype=bool)
    for h in range(nh):
        K_h = min(int(budgets[h]), nk)
        idx = np.argpartition(-attn_ph[h], K_h - 1)[:K_h] if K_h < nk else np.arange(nk)
        m[h, idx] = True
    return m


def mph_headkv(attn_ph: np.ndarray, K: int, accum_ph: np.ndarray | None = None,
               **kw) -> np.ndarray:
    """HeadKV (Fu 2024) shadow: rank heads by cumulative-attention concentration
    (Gini-style sum of squared accum vector), give top heads MORE budget. Per-head
    TOVA within budget.
    """
    nh, nk = attn_ph.shape
    if nk <= K:
        return np.ones((nh, nk), dtype=bool)
    importance = np.zeros(nh)
    src = accum_ph if accum_ph is not None else attn_ph
    for h in range(nh):
        v = src[h]
        s = v.sum()
        if s > 1e-9:
            p = v / s
            # head importance = inverse entropy (more concentrated = more important)
            importance[h] = float(np.sum(p ** 2))   # higher = sharper concentration
    # Allocate total = K*nh; top heads get 1.5K, bottom get 0.5K, linear interp
    order = np.argsort(-importance)
    rank = np.empty(nh, dtype=np.int32)
    rank[order] = np.arange(nh)
    # interp: rank 0 -> 1.5, rank nh-1 -> 0.5
    frac = rank / max(1, nh - 1)
    mult = 1.5 - frac
    budgets = np.maximum(1, (K * mult).astype(np.int32))
    m = np.zeros((nh, nk), dtype=bool)
    for h in range(nh):
        K_h = min(int(budgets[h]), nk)
        if K_h >= nk:
            m[h, :] = True; continue
        idx = np.argpartition(-attn_ph[h], K_h - 1)[:K_h]
        m[h, idx] = True
    return m


def mph_duoattention_lite(attn_ph: np.ndarray, K: int,
                          retrieval_mask: np.ndarray | None = None,
                          **kw) -> np.ndarray:
    """DuoAttention (Xiao ICLR'25) shadow: partition heads into
    "retrieval" (keep full cache) and "streaming" (sliding window only).

    retrieval_mask is precomputed offline per layer-head from cumulative
    attention concentration over the first few decode steps. If a head's
    cumulative attention top-K covers >70% of mass, it's a retrieval head.
    """
    nh, nk = attn_ph.shape
    if retrieval_mask is None:
        # Fallback: top-half by current sharpness are retrieval
        sharp = np.zeros(nh)
        log_nk = max(1e-9, np.log(max(2, nk)))
        for h in range(nh):
            p = np.clip(attn_ph[h].astype(np.float32, copy=False), 1e-12, 1.0)
            sharp[h] = 1.0 - (-float(np.sum(p * np.log(p)))) / log_nk
        retrieval_mask = sharp > np.median(sharp)
    m = np.zeros((nh, nk), dtype=bool)
    for h in range(nh):
        if retrieval_mask[h]:
            # Retrieval head: keep top-K (more than streaming budget)
            K_h = min(int(round(K * 1.5)), nk)
            idx = np.argpartition(-attn_ph[h], K_h - 1)[:K_h] if K_h < nk else np.arange(nk)
            m[h, idx] = True
        else:
            # Streaming head: sliding window of K/2
            K_h = max(1, K // 2)
            start = max(0, nk - K_h)
            m[h, start:] = True
            # plus a few attention sinks
            m[h, :min(4, nk)] = True
    return m


def mph_endurkv(attn_ph: np.ndarray, K: int,
                attn_ewma_ph: np.ndarray | None = None,
                **kw) -> np.ndarray:
    """*** EndurKV-Evict per-head v1 *** — our candidate.

    Two ideas, both unclaimed in the literature:

      (1) PER-HEAD spread gate: each head's budget K_t = K * (1.3 - 0.6 * max_a_norm)
          (proven in our layer-averaged data).

      (2) CROSS-HEAD CONSENSUS PROTECTION: a position i is "load-bearing"
          if MOST heads (more than 50%) place it in their top-K. These
          positions get HARD PROTECTION across ALL heads — even the heads
          that don't currently attend to i keep i, because the consensus
          says it matters globally to this layer.

    This is the inverse of what made v4 fail at the layer level. Layers
    agree too much; heads disagree enough to provide signal. Cross-head
    consensus filters for positions that MULTIPLE heads independently
    deem important — those are the load-bearing tokens.
    """
    nh, nk = attn_ph.shape
    if nk <= K:
        return np.ones((nh, nk), dtype=bool)

    # Step 1: per-head spread-gate budget
    K_t_per_head = np.zeros(nh, dtype=np.int32)
    raw_top_idx = np.zeros((nh, K * 2), dtype=np.int32)
    raw_top_count = np.zeros(nh, dtype=np.int32)
    for h in range(nh):
        a = attn_ph[h]
        max_a = float(a.max())
        norm = max(0.0, min(1.0, (max_a - 0.4) / 0.4))
        mult = 1.3 - 0.6 * norm
        K_t_per_head[h] = max(1, min(nk, int(round(K * mult))))
        # also compute a broader top-2K for consensus detection
        n_broad = min(nk, 2 * K)
        if n_broad < nk:
            idx = np.argpartition(-a, n_broad - 1)[:n_broad]
        else:
            idx = np.arange(nk)
        raw_top_idx[h, :n_broad] = idx
        raw_top_count[h] = n_broad

    # Step 2: cross-head consensus — count how many heads have position i in their top-2K
    consensus = np.zeros(nk, dtype=np.int32)
    for h in range(nh):
        consensus[raw_top_idx[h, :raw_top_count[h]]] += 1
    # Positions where >50% of heads agree → load-bearing
    consensus_threshold = nh // 2
    load_bearing = consensus > consensus_threshold  # bool [nk]
    n_load = int(load_bearing.sum())

    # Step 3: build the mask per head
    m = np.zeros((nh, nk), dtype=bool)
    for h in range(nh):
        K_t = int(K_t_per_head[h])
        # First, hard-protect load-bearing positions (up to K_t cap)
        protected = load_bearing.copy()
        if protected.sum() > K_t:
            # Too many load-bearing — keep the ones with highest cross-head consensus
            top_lb = np.argpartition(-consensus * protected.astype(np.int32),
                                     K_t - 1)[:K_t]
            new_p = np.zeros(nk, dtype=bool); new_p[top_lb] = True
            protected = new_p
        m[h] = protected
        # Fill remainder by THIS head's current attention
        remaining = K_t - int(m[h].sum())
        if remaining > 0:
            mask_avail = ~m[h]
            avail_idx = np.where(mask_avail)[0]
            if len(avail_idx) <= remaining:
                m[h, avail_idx] = True
            else:
                scores = attn_ph[h, avail_idx]
                pick = avail_idx[np.argpartition(-scores, remaining - 1)[:remaining]]
                m[h, pick] = True
    return m


def mph_endurkv_disagree(attn_ph: np.ndarray, K: int,
                         attn_ewma_ph: np.ndarray | None = None,
                         **kw) -> np.ndarray:
    """*** EndurKV-Evict per-head v2 *** — DISAGREEMENT protection.

    The opposite move from cross-head consensus: positions where heads
    DISAGREE the most are the discriminating tokens. A position where every
    head gives roughly the same attention conveys no head-specific signal —
    it's a sink-like generic position. A position where some heads give 0.05
    and others give 0.001 is being used by SOME heads for SOMETHING specific.

    Mechanism:
      1) For each head: v1 spread-gate sized budget K_t.
      2) Compute per-position cross-head std (s_pos) and mean (m_pos).
      3) Score each position per head =
            0.55 * current_attn[h, i]
          + 0.25 * s_pos[i]       <-- head disagreement bonus (NOVEL)
          + 0.20 * ewma[h, i]
      4) Top-K_t by score per head.
    """
    nh, nk = attn_ph.shape
    if nk <= K:
        return np.ones((nh, nk), dtype=bool)

    # Per-position cross-head statistics
    pos_std = attn_ph.std(axis=0)      # [nk]
    pos_mean = attn_ph.mean(axis=0)    # [nk]
    # Normalize disagreement to [0, 1]-ish scale (relative to the position's mean)
    disagreement = pos_std / np.maximum(pos_mean + 1e-12, 1e-9)
    # Clip to avoid outliers dominating
    disagreement = np.clip(disagreement, 0, 5)

    ewma = attn_ewma_ph if attn_ewma_ph is not None else attn_ph

    m = np.zeros((nh, nk), dtype=bool)
    for h in range(nh):
        a = attn_ph[h]
        max_a = float(a.max())
        norm = max(0.0, min(1.0, (max_a - 0.4) / 0.4))
        mult = 1.3 - 0.6 * norm
        K_t = max(1, min(nk, int(round(K * mult))))
        # Scale disagreement to be commensurate with attention values
        disag_scale = disagreement * float(a.mean()) if a.mean() > 0 else disagreement
        score = 0.55 * a + 0.25 * disag_scale + 0.20 * ewma[h]
        idx = np.argpartition(-score, K_t)[:K_t]
        m[h, idx] = True
    return m


def mph_volatility(attn_ph: np.ndarray, K: int,
                   attn_ewma_ph: np.ndarray | None = None,
                   **kw) -> np.ndarray:
    """Per-head budget reallocated by temporal volatility.

    Volatile heads (current attention deviates a lot from EWMA) capture
    *transient* peaks — they need MORE budget because eviction decisions
    based on stale state are riskier. Stable heads (current ~ EWMA) are
    safe with less budget. Total budget preserved at K * n_head.
    """
    nh, nk = attn_ph.shape
    if nk <= K:
        return np.ones((nh, nk), dtype=bool)
    ewma = attn_ewma_ph if attn_ewma_ph is not None else attn_ph
    dev = np.abs(attn_ph.astype(np.float32) - ewma.astype(np.float32))
    vol = dev.mean(axis=1)
    vol_norm = vol / vol.max() if vol.max() > 1e-12 else np.zeros(nh)
    mult = np.clip(0.8 + 0.5 * vol_norm, 0.5, 1.5)
    mult = mult * (nh / mult.sum())
    budgets = np.maximum(1, (K * mult).astype(np.int32))
    m = np.zeros((nh, nk), dtype=bool)
    for h in range(nh):
        K_h = min(int(budgets[h]), nk)
        if K_h >= nk:
            m[h, :] = True; continue
        score = 0.7 * attn_ph[h].astype(np.float32) + 0.3 * dev[h]
        idx = np.argpartition(-score, K_h - 1)[:K_h]
        m[h, idx] = True
    return m


def mph_svd(attn_ph: np.ndarray, K: int, **kw) -> np.ndarray:
    """SVD on the [n_heads x n_kv] attention matrix.

    Top-r right-singular vectors of the layer's [head x position] attention
    matrix capture dominant ATTENTION PATTERNS that heads collectively express.
    Positions with high energy in those vectors are load-bearing; protect them
    globally, fill remainder with per-head TOVA. Unclaimed for KV eviction
    at the per-head level (LoRC / Eigen-Attention low-rank K/V vectors, not
    attention weights).
    """
    nh, nk = attn_ph.shape
    if nk <= K:
        return np.ones((nh, nk), dtype=bool)
    try:
        A = attn_ph.astype(np.float32)
        r = min(4, nh, nk)
        _U, S, Vt = np.linalg.svd(A, full_matrices=False)
        S2 = S[:r] ** 2
        pos_energy = (Vt[:r] ** 2 * S2[:, None]).sum(axis=0)
    except np.linalg.LinAlgError:
        return mph_tova(attn_ph, K)
    n_protect = min(K, nk)
    protect_idx = np.argpartition(-pos_energy, n_protect - 1)[:n_protect]
    global_protected = np.zeros(nk, dtype=bool)
    global_protected[protect_idx] = True
    m = np.zeros((nh, nk), dtype=bool)
    for h in range(nh):
        m[h] = global_protected.copy()
        remaining = K - int(m[h].sum())
        if remaining > 0:
            avail = np.where(~m[h])[0]
            if len(avail) <= remaining:
                m[h, avail] = True
            else:
                scores = attn_ph[h, avail].astype(np.float32)
                pick = avail[np.argpartition(-scores, remaining - 1)[:remaining]]
                m[h, pick] = True
    return m


def mph_endurkv_v2(attn_ph: np.ndarray, K: int,
                   attn_ewma_ph: np.ndarray | None = None,
                   **kw) -> np.ndarray:
    """*** EndurKV-Evict per-head v2 *** — combined: spread + consensus + volatility.

    Three signals stacked:
      (1) Per-head spread gate K_h = K * (1.3 - 0.6 * max_a_norm).
      (2) Cross-head consensus protection (>50% heads agree on a position).
      (3) Volatility-aware K_h modulation (more budget to swinging heads).
    """
    nh, nk = attn_ph.shape
    if nk <= K:
        return np.ones((nh, nk), dtype=bool)
    ewma = attn_ewma_ph if attn_ewma_ph is not None else attn_ph
    dev = np.abs(attn_ph.astype(np.float32) - ewma.astype(np.float32))
    vol = dev.mean(axis=1)
    vol_norm = vol / vol.max() if vol.max() > 1e-12 else np.zeros(nh)
    vol_mult = 0.85 + 0.3 * vol_norm

    K_t_per_head = np.zeros(nh, dtype=np.int32)
    raw_top_idx = np.zeros((nh, K * 2), dtype=np.int32)
    raw_top_count = np.zeros(nh, dtype=np.int32)
    for h in range(nh):
        a = attn_ph[h]
        max_a = float(a.max())
        norm = max(0.0, min(1.0, (max_a - 0.4) / 0.4))
        mult = (1.3 - 0.6 * norm) * vol_mult[h]
        K_t_per_head[h] = max(1, min(nk, int(round(K * mult))))
        n_broad = min(nk, 2 * K)
        if n_broad < nk:
            idx = np.argpartition(-a, n_broad - 1)[:n_broad]
        else:
            idx = np.arange(nk)
        raw_top_idx[h, :n_broad] = idx
        raw_top_count[h] = n_broad

    consensus = np.zeros(nk, dtype=np.int32)
    for h in range(nh):
        consensus[raw_top_idx[h, :raw_top_count[h]]] += 1
    load_bearing = consensus > (nh // 2)

    m = np.zeros((nh, nk), dtype=bool)
    for h in range(nh):
        K_t = int(K_t_per_head[h])
        protected = load_bearing.copy()
        if protected.sum() > K_t:
            top_lb = np.argpartition(-consensus * protected.astype(np.int32),
                                     K_t - 1)[:K_t]
            new_p = np.zeros(nk, dtype=bool); new_p[top_lb] = True
            protected = new_p
        m[h] = protected
        remaining = K_t - int(m[h].sum())
        if remaining > 0:
            avail = np.where(~m[h])[0]
            if len(avail) <= remaining:
                m[h, avail] = True
            else:
                scores = (0.85 * attn_ph[h, avail].astype(np.float32) +
                          0.15 * dev[h, avail])
                pick = avail[np.argpartition(-scores, remaining - 1)[:remaining]]
                m[h, pick] = True
    return m


def mph_v1_tir(attn_ph: np.ndarray, K: int,
               tir_last_high: np.ndarray | None = None,
               tir_mri: np.ndarray | None = None,
               tir_step: int = 0,
               **kw) -> np.ndarray:
    """*** EndurKV-Evict per-head v1 + TIR *** (P0 design, May 2026).

    Per-head LazyEviction-style Token Importance Recurrence on top of v1's
    spread gate. The layer-averaged endurkv_v6 tied v1 because layer-avg TIR
    correlates with current attention. Per-head TIR may not — heads disagree
    about which positions recur (DuoAttention's premise).

    Per-(head, position) state:
      tir_last_high[h, i]   step when head h last placed position i in its top-K
      tir_mri[h, i]         max recurrence interval observed for (h, i)

    Score per head per position:
      score[h, i] = a[h, i] * (1.0 + 0.5 * tir_bonus[h, i])
      tir_bonus[h, i] = exp(-time_since_last_high / mri_safe)  if seen else 0
    """
    nh, nk = attn_ph.shape
    if nk <= K:
        return np.ones((nh, nk), dtype=bool)
    m = np.zeros((nh, nk), dtype=bool)
    for h in range(nh):
        a = attn_ph[h]
        max_a = float(a.max())
        norm = max(0.0, min(1.0, (max_a - 0.4) / 0.4))
        mult = 1.3 - 0.6 * norm
        K_t = max(1, min(nk, int(round(K * mult))))
        if tir_last_high is not None and tir_mri is not None:
            time_since = tir_step - tir_last_high[h, :nk]
            mri_safe = np.maximum(tir_mri[h, :nk].astype(np.float32), 1.0)
            bonus = np.where(tir_last_high[h, :nk] >= 0,
                             np.exp(-time_since.astype(np.float32) / mri_safe),
                             0.0)
            score = a.astype(np.float32) * (1.0 + 0.5 * bonus)
        else:
            score = a.astype(np.float32)
        idx = np.argpartition(-score, K_t)[:K_t]
        m[h, idx] = True
    return m


def mph_v1_tiered(attn_ph: np.ndarray, K: int,
                  tir_last_high: np.ndarray | None = None,
                  tir_mri: np.ndarray | None = None,
                  tir_step: int = 0,
                  **kw) -> np.ndarray:
    """*** EndurKV-Evict (tiered, v2) ***  — 2026-05-24.

    Parametric family of per-head attention-only eviction policies indexed by
    context length. CORE = perhead_v1 (per-head TOVA + max-attention spread gate).
    Conditional add-ons activate at longer contexts where their failure modes
    (documented for perhead_v1_tir and endurkv_perhead in shorter contexts) no
    longer apply.

      ALL ctx — CORE:  K_h = K * (1.3 - 0.6 * norm(max_a_h))    (perhead_v1)
                       score_base[h, i] = attn_ph[h, i]

      ctx_len >= 16384 — TIER 3:  add TIR bonus
                       bonus_TIR[h, i] = exp(-(s - last_high[h, i]) / mri[h, i])
                       score *= (1 + 0.25 * bonus_TIR)

      ctx_len >= 32768 — TIER 4:  add cross-head consensus bonus
                       (positions in top-K of many heads are load-bearing)
                       score += 0.10 * consensus_count[i] / n_head

    For our currently-measured regime (ctx 64-13K), TIER 3+ do NOT activate, so
    this policy is byte-for-byte equivalent to perhead_v1. Tier-3+ are forward-
    looking for the planned 32K-context controller deployment captures.
    """
    nh, nk = attn_ph.shape
    if nk <= K:
        return np.ones((nh, nk), dtype=bool)

    use_tir       = nk >= 16384 and tir_last_high is not None
    use_consensus = nk >= 32768

    # Consensus precompute (only if needed)
    consensus_norm = None
    if use_consensus:
        # rough consensus: how many heads have position i in their top-(2K)
        top_n = min(nk, 2 * K)
        cnt = np.zeros(nk, dtype=np.int32)
        for h in range(nh):
            if top_n < nk:
                idx = np.argpartition(-attn_ph[h], top_n - 1)[:top_n]
            else:
                idx = np.arange(nk)
            cnt[idx] += 1
        consensus_norm = cnt.astype(np.float32) / nh   # in [0, 1]

    m = np.zeros((nh, nk), dtype=bool)
    for h in range(nh):
        a = attn_ph[h]
        max_a = float(a.max())
        norm = max(0.0, min(1.0, (max_a - 0.4) / 0.4))
        mult = 1.3 - 0.6 * norm
        K_t = max(1, min(nk, int(round(K * mult))))

        score = a.astype(np.float32, copy=False)
        if use_tir:
            time_since = tir_step - tir_last_high[h, :nk]
            mri_safe = np.maximum(tir_mri[h, :nk].astype(np.float32), 1.0)
            bonus = np.where(tir_last_high[h, :nk] >= 0,
                             np.exp(-time_since.astype(np.float32) / mri_safe),
                             0.0)
            score = score * (1.0 + 0.25 * bonus)
        if use_consensus:
            score = score + 0.10 * consensus_norm

        idx = np.argpartition(-score, K_t)[:K_t]
        m[h, idx] = True
    return m


POLICIES = {
    "full":                  mph_full,
    "local":                 mph_local,
    "perhead_tova":          mph_tova,
    "perhead_v1":            mph_v1,                 # *** THE policy: EndurKV-Evict ***
    "perhead_v1_tir":        mph_v1_tir,             # null-result variant (TIR adds nothing)
    # "perhead_v1_tiered":   mph_v1_tiered,          # un-registered 2026-05-24: tiered
    #                                                  architecture dropped from paper claims.
    #                                                  perhead_v1 wins everywhere measured;
    #                                                  conditional tiers (TIR @ ≥16K,
    #                                                  consensus @ ≥32K) are speculation
    #                                                  without 32K+ data to validate.
    #                                                  Re-register if future captures support.
    "adakv":                 mph_adakv,              # baseline (Feng 2024)
    "headkv":                mph_headkv,             # baseline (Fu 2024)
    "duoattention":          mph_duoattention_lite,  # baseline (Xiao ICLR'25)
    "endurkv_perhead":       mph_endurkv,            # negative result (cross-head consensus)
    "endurkv_disagree":      mph_endurkv_disagree,   # negative result (cross-head disagree)
    "perhead_volatility":    mph_volatility,         # negative result (temporal volatility)
    "perhead_svd":           mph_svd,                # negative result (SVD principal)
    "endurkv_perhead_v2":    mph_endurkv_v2,         # negative result (spread+cons+vol)
}


# ============================================================================
# simulator
# ============================================================================

def simulate(attn_ph: np.ndarray, n_kv_at: np.ndarray, policy_name: str,
             K_nominal: int):
    """Run a per-head policy over all (step, layer) pairs.

    Per-head KL: for each head, KL between full head distribution and
    masked-renormalized head distribution. Aggregate = mean over heads.
    Per-step per-layer mean_kl + actual cache used.
    """
    n_steps, n_layers, n_head, max_kv = attn_ph.shape
    fn = POLICIES[policy_name]

    out_kl = np.zeros((n_steps, n_layers), dtype=np.float64)
    out_K  = np.zeros((n_steps, n_layers), dtype=np.int32)

    # Per-head cumulative attention (for HeadKV / DuoAttention)
    accum_ph = np.zeros((n_layers, n_head, max_kv), dtype=np.float64)
    ewma_ph  = np.zeros((n_layers, n_head, max_kv), dtype=np.float64)
    ewma_b = 0.3

    # Per-(layer, head, position) TIR state for perhead_v1_tir / perhead_v1_tiered.
    # Only allocate when the policy needs it (saves ~100 MB on 8B long).
    tir_last_high = tir_mri = None
    if policy_name in ("perhead_v1_tir", "perhead_v1_tiered"):
        tir_last_high = -np.ones((n_layers, n_head, max_kv), dtype=np.int32)
        tir_mri       =  np.zeros((n_layers, n_head, max_kv), dtype=np.int32)
        tir_tau = 0.05  # mirrors layer-avg LazyEviction threshold

    # Pre-pass: retrieval/streaming classification for DuoAttention (use first
    # ~8 decode steps' cumulative attention concentration to classify heads)
    retrieval_masks = None
    if policy_name == "duoattention":
        warm = min(8, n_steps)
        if warm > 0:
            cum_warm = attn_ph[:warm].sum(axis=0)  # [layers, heads, max_kv]
            retrieval_masks = np.zeros((n_layers, n_head), dtype=bool)
            for l in range(n_layers):
                for h in range(n_head):
                    v = cum_warm[l, h]
                    s = v.sum()
                    if s < 1e-9:
                        continue
                    sorted_v = np.sort(v)[::-1]
                    # if top 16 positions cover >70% of total → retrieval
                    top_share = sorted_v[:16].sum() / s
                    retrieval_masks[l, h] = top_share > 0.7

    for s in range(n_steps):
        n_kv = int(n_kv_at[s])
        if n_kv == 0:
            continue
        # Update per-head accumulators
        for l in range(n_layers):
            accum_ph[l, :, :n_kv] += attn_ph[s, l, :, :n_kv]
            ewma_ph[l, :, :n_kv]  = (1 - ewma_b) * ewma_ph[l, :, :n_kv] + \
                                    ewma_b * attn_ph[s, l, :, :n_kv]
        # TIR state update (per layer, head, position) BEFORE policy decision.
        # Positions whose current head-attn exceeds tau update their last_high_step
        # and grow their MRI (max recurrence interval). Vectorized — no Python loops
        # over positions.
        if tir_last_high is not None:
            cur_step = s
            for l in range(n_layers):
                a_lh = attn_ph[s, l, :, :n_kv].astype(np.float32, copy=False)
                high = a_lh > tir_tau
                prev = tir_last_high[l, :, :n_kv]
                seen = prev >= 0
                interval = np.where(high & seen, cur_step - prev, 0)
                cur_mri = tir_mri[l, :, :n_kv]
                np.maximum(cur_mri, interval, out=cur_mri)
                tir_last_high[l, :, :n_kv] = np.where(high, cur_step, prev)

        for l in range(n_layers):
            ph_slice = attn_ph[s, l, :, :n_kv]
            kw = dict(accum_ph=accum_ph[l, :, :n_kv],
                      attn_ewma_ph=ewma_ph[l, :, :n_kv])
            if policy_name == "duoattention" and retrieval_masks is not None:
                kw["retrieval_mask"] = retrieval_masks[l]
            if policy_name in ("perhead_v1_tir", "perhead_v1_tiered") and tir_last_high is not None:
                kw["tir_last_high"] = tir_last_high[l, :, :n_kv]
                kw["tir_mri"]       = tir_mri[l, :, :n_kv]
                kw["tir_step"]      = s
            mask = fn(ph_slice, K_nominal, **kw)
            # Per-head KL: average over heads
            kls = []
            kept_sum = 0
            for h in range(n_head):
                p_full = ph_slice[h]
                kept = p_full * mask[h]
                tot = kept.sum()
                if tot <= 0:
                    kls.append(20.0)
                else:
                    kls.append(kl(p_full, kept / tot))
                kept_sum += int(mask[h].sum())
            out_kl[s, l] = float(np.mean(kls))
            out_K[s, l]  = kept_sum / n_head   # avg per-head cache spend
    return out_kl, out_K


# ============================================================================
# main
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--budgets", default="64,74,128,149,256,298")
    ap.add_argument("--max-prompts", type=int, default=0)
    ap.add_argument("--bytes-per-token", type=int, default=128 * 1024)
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir) if args.out_dir else log_dir.parent / f"{log_dir.name}_perhead"
    out_dir.mkdir(parents=True, exist_ok=True)

    budgets = [int(b) for b in args.budgets.split(",")]

    rows = []
    attn_files = sorted(log_dir.glob("*.attn.bin"))
    if args.max_prompts > 0:
        attn_files = attn_files[: args.max_prompts]
    if not attn_files:
        print(f"ERROR: no *.attn.bin in {log_dir}", file=sys.stderr); return 1
    print(f"[sim-perhead] {len(attn_files)} attn files, budgets={budgets}, "
          f"policies={list(POLICIES.keys())}")

    for af in attn_files:
        pid = af.name[:-len(".attn.bin")]
        attn_ph, n_kv_at = load_attn_perhead(af)
        if attn_ph is None:
            print(f"  [skip] {pid}: not ATNH format")
            continue
        n_steps, n_layers, n_head, max_kv = attn_ph.shape
        task = pid.rsplit("_", 1)[0]
        if task.endswith("_lc"):
            task = task[:-3]
        print(f"  {pid:<28} steps={n_steps} layers={n_layers} "
              f"heads={n_head} max_kv={max_kv}")
        for K in budgets:
            for pname in POLICIES:
                kls, Ks = simulate(attn_ph, n_kv_at, pname, K)
                rows.append({
                    "prompt_id": pid, "task": task, "policy": pname,
                    "K_nominal": K,
                    "mean_kl": float(np.mean(kls)),
                    "avg_actual_K": float(np.mean(Ks)),
                    "n_steps": n_steps, "n_layers": n_layers,
                    "n_head": n_head, "max_kv": max_kv,
                })

    if not rows:
        return 1
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "policy_results.csv", index=False)
    summary = df.groupby(["policy", "K_nominal"]).agg(
        mean_kl=("mean_kl", "mean"),
        avg_actual_K=("avg_actual_K", "mean"),
        n_prompts=("prompt_id", "nunique"),
    ).reset_index()
    summary.to_csv(out_dir / "pareto_summary.csv", index=False)

    print("\n=== Per-policy at each K_nominal ===")
    print(summary.pivot_table(index="policy", columns="K_nominal",
                              values=["mean_kl", "avg_actual_K"], aggfunc="mean")
          .to_string(float_format=lambda x: f"{x:7.3f}"))
    print(f"\nall outputs in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
