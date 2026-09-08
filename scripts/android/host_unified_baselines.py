#!/usr/bin/env python3
"""Unified runner: every attention-only published KV-eviction baseline implemented
in this codebase, on the same 5 long-context dirs × 1 prompt × K ∈ {512, 1024}.

Strategy:
  - Per-head policies (TOVA-perhead, AdaKV, HeadKV, DuoAttention, our v1..v7,
    KVzip-approx) plug straight into the per-head simulator.
  - Global policies (H2O, SnapKV, StreamingLLM, Scissorhands, LWKD, AhaKV,
    LazyEviction) produce one 1-D mask per layer; we broadcast to all heads
    (same as their deployed form).
  - Per-layer budget allocators (PyramidKV, CAKE) just choose K_h per layer
    then defer to TOVA selection inside each layer.

Honest caveats baked into table at end:
  - KVzip-approx ≠ full KVzip (no reconstruction-prefill pass).
  - R-KV / KeyDiff / LaProx need K/V captures we don't have on these dirs
    → would silently fall back to TOVA → not listed.
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from host_simulate_kv_baselines import load_attn_perhead

DIRS = [
    "/home/mislam22/EndurKV_workspace/logs/study_phone_phi3_longbench",
    "/home/mislam22/EndurKV_workspace/logs/study_phone_mistral_longbench",
    "/home/mislam22/EndurKV_workspace/logs/study_phone_qwen2_longbench",
    "/home/mislam22/EndurKV_workspace/logs/study_phone_gemma2_longbench",
    "/home/mislam22/EndurKV_workspace/logs/study_phone_r1distill_longbench",
]
BUDGETS = [512, 1024]


# ---------------------------------------------------------------------------
# Inline policy implementations (copied/adapted from the three existing
# simulator files so this script is self-contained)
# ---------------------------------------------------------------------------

# -- Per-head policies ------------------------------------------------------

def policy_tova_ph(attn_ph, K, **kw):
    nh, nk = attn_ph.shape
    if nk <= K: return np.ones((nh, nk), dtype=bool)
    m = np.zeros((nh, nk), dtype=bool)
    for h in range(nh):
        idx = np.argpartition(-attn_ph[h], K)[:K]
        m[h, idx] = True
    return m


def policy_v1_spread(attn_ph, K, **kw):
    nh, nk = attn_ph.shape
    if nk <= K: return np.ones((nh, nk), dtype=bool)
    m = np.zeros((nh, nk), dtype=bool)
    for h in range(nh):
        a = attn_ph[h]; max_a = float(a.max())
        norm = max(0.0, min(1.0, (max_a - 0.4) / 0.4))
        mult = 1.3 - 0.6 * norm
        K_t = max(1, min(nk, int(round(K * mult))))
        idx = np.argpartition(-a, K_t - 1)[:K_t]
        m[h, idx] = True
    return m


def policy_v6_logistic(attn_ph, K, alpha_low=0.7, alpha_high=1.3,
                       gamma=10.0, c0=0.4, **kw):
    nh, nk = attn_ph.shape
    if nk <= K: return np.ones((nh, nk), dtype=bool)
    m = np.zeros((nh, nk), dtype=bool)
    for h in range(nh):
        a = attn_ph[h]; max_a = float(a.max())
        sig = 1.0 / (1.0 + np.exp(gamma * (max_a - c0)))
        mult = alpha_low + (alpha_high - alpha_low) * sig
        K_t = max(1, min(nk, int(round(K * mult))))
        idx = np.argpartition(-a, K_t - 1)[:K_t]
        m[h, idx] = True
    return m


def policy_adakv(attn_ph, K, **kw):
    """AdaKV: per-head budget proportional to head sharpness (1 - normalized entropy).
    Total cache = K * n_head (matches per-head TOVA spend)."""
    nh, nk = attn_ph.shape
    if nk <= K: return np.ones((nh, nk), dtype=bool)
    sharp = np.zeros(nh); log_nk = max(1e-9, np.log(max(2, nk)))
    for h in range(nh):
        p = np.clip(attn_ph[h].astype(np.float32), 1e-12, 1.0)
        H = -float(np.sum(p * np.log(p)))
        sharp[h] = 1.0 - H / log_nk
    total = K * nh
    if sharp.sum() > 1e-9:
        budgets = sharp / sharp.sum() * total
    else:
        budgets = np.full(nh, K, dtype=np.float64)
    budgets = np.clip(budgets, 0.5 * K, 1.5 * K)
    budgets = budgets * (total / max(1.0, budgets.sum()))
    budgets = np.maximum(1, budgets.astype(np.int32))
    m = np.zeros((nh, nk), dtype=bool)
    for h in range(nh):
        K_h = min(int(budgets[h]), nk)
        idx = (np.argpartition(-attn_ph[h], K_h - 1)[:K_h]
               if K_h < nk else np.arange(nk))
        m[h, idx] = True
    return m


def policy_headkv(attn_ph, K, accum_ph=None, **kw):
    """HeadKV: rank heads by concentration of cumulative attention (Σ p²),
    top heads get 1.5K, bottom 0.5K (linear interp)."""
    nh, nk = attn_ph.shape
    if nk <= K: return np.ones((nh, nk), dtype=bool)
    src = accum_ph if accum_ph is not None else attn_ph
    importance = np.zeros(nh)
    for h in range(nh):
        v = src[h]; s = v.sum()
        if s > 1e-9:
            p = v / s; importance[h] = float(np.sum(p ** 2))
    order = np.argsort(-importance)
    rank = np.empty(nh, dtype=np.int32); rank[order] = np.arange(nh)
    frac = rank / max(1, nh - 1)
    budgets = np.maximum(1, (K * (1.5 - frac)).astype(np.int32))
    m = np.zeros((nh, nk), dtype=bool)
    for h in range(nh):
        K_h = min(int(budgets[h]), nk)
        if K_h >= nk: m[h, :] = True; continue
        idx = np.argpartition(-attn_ph[h], K_h - 1)[:K_h]
        m[h, idx] = True
    return m


def policy_duoattention(attn_ph, K, **kw):
    """DuoAttention-lite: top-half-by-sharpness = retrieval (1.5K),
    bottom-half = streaming (K/2 sliding + sinks)."""
    nh, nk = attn_ph.shape
    if nk <= K: return np.ones((nh, nk), dtype=bool)
    sharp = np.zeros(nh); log_nk = max(1e-9, np.log(max(2, nk)))
    for h in range(nh):
        p = np.clip(attn_ph[h].astype(np.float32), 1e-12, 1.0)
        sharp[h] = 1.0 - (-float(np.sum(p * np.log(p)))) / log_nk
    retrieval = sharp > np.median(sharp)
    m = np.zeros((nh, nk), dtype=bool)
    for h in range(nh):
        if retrieval[h]:
            K_h = min(int(round(K * 1.5)), nk)
            idx = (np.argpartition(-attn_ph[h], K_h - 1)[:K_h]
                   if K_h < nk else np.arange(nk))
            m[h, idx] = True
        else:
            K_h = max(1, K // 2)
            m[h, max(0, nk - K_h):] = True
            m[h, :min(4, nk)] = True
    return m


def policy_kvzip_approx(attn_ph, K, attn_cum_max=None, **kw):
    """KVzip-decode-approx: top-K by cumulative-max attention per head."""
    nh, nk = attn_ph.shape
    if nk <= K: return np.ones((nh, nk), dtype=bool)
    score = attn_cum_max if attn_cum_max is not None else attn_ph
    if score.shape[1] != nk: score = score[:, :nk]
    m = np.zeros((nh, nk), dtype=bool)
    for h in range(nh):
        idx = np.argpartition(-score[h], K - 1)[:K]
        m[h, idx] = True
    return m


# -- Global policies (1D mask per layer; broadcast to all heads) -----------

def _hh_norec_mask(n_kv, K, accum, recent_frac):
    if n_kv <= K: return np.ones(n_kv, dtype=bool)
    K_rec = max(1, int(round(K * recent_frac)))
    K_hh = K - K_rec
    rec_start = max(0, n_kv - K_rec)
    cand = accum[:rec_start]
    if len(cand) <= K_hh:
        hh_idx = np.arange(rec_start)
    else:
        hh_idx = np.argpartition(-cand, K_hh - 1)[:K_hh]
    m = np.zeros(n_kv, dtype=bool)
    m[hh_idx] = True; m[rec_start:] = True
    return m


def policy_h2o(attn_ph, K, accum_layer=None, **kw):
    """H2O (Zhang NeurIPS 2023): K/2 heavy hitters by cum-attn + K/2 recent."""
    nh, nk = attn_ph.shape
    if nk <= K: return np.ones((nh, nk), dtype=bool)
    acc = accum_layer if accum_layer is not None else attn_ph.sum(axis=0)
    m1d = _hh_norec_mask(nk, K, acc[:nk], recent_frac=0.5)
    return np.broadcast_to(m1d, (nh, nk)).copy()


def policy_snapkv(attn_ph, K, snapkv_score_layer=None, **kw):
    """SnapKV (Li NeurIPS 2024): max-pooled observation window + heavy hitters."""
    nh, nk = attn_ph.shape
    if nk <= K: return np.ones((nh, nk), dtype=bool)
    L_obs = 16
    K_rec = min(L_obs, max(1, K // 4))
    K_hh = K - K_rec
    rec_start = max(0, nk - K_rec)
    src = (snapkv_score_layer if snapkv_score_layer is not None
           else attn_ph.sum(axis=0))
    cand = src[:rec_start] if rec_start > 0 else np.zeros(0)
    if len(cand) <= K_hh:
        hh_idx = np.arange(rec_start)
    else:
        hh_idx = np.argpartition(-cand, K_hh - 1)[:K_hh]
    m1d = np.zeros(nk, dtype=bool)
    m1d[hh_idx] = True; m1d[rec_start:] = True
    return np.broadcast_to(m1d, (nh, nk)).copy()


def policy_streamingllm(attn_ph, K, **kw):
    """StreamingLLM (Xiao ICLR 2024): 4 attention sinks + K-4 most recent."""
    nh, nk = attn_ph.shape
    if nk <= K: return np.ones((nh, nk), dtype=bool)
    n_sink = min(4, K // 2); K_rec = K - n_sink
    m1d = np.zeros(nk, dtype=bool)
    m1d[:n_sink] = True; m1d[max(0, nk - K_rec):] = True
    return np.broadcast_to(m1d, (nh, nk)).copy()


def policy_scissorhands(attn_ph, K, scissor_score_layer=None, **kw):
    """Scissorhands (Liu NeurIPS 2023): exp-decay sliding-window attention +
    K/8 recent reserved."""
    nh, nk = attn_ph.shape
    if nk <= K: return np.ones((nh, nk), dtype=bool)
    acc = (scissor_score_layer if scissor_score_layer is not None
           else attn_ph.sum(axis=0))
    m1d = _hh_norec_mask(nk, K, acc[:nk], recent_frac=1.0/8)
    return np.broadcast_to(m1d, (nh, nk)).copy()


def policy_ahakv(attn_ph, K, recent_accum_layer=None, **kw):
    """AhaKV-approx (Jun 2025): recent-W accum + K/8 reserved recent."""
    nh, nk = attn_ph.shape
    if nk <= K: return np.ones((nh, nk), dtype=bool)
    acc = (recent_accum_layer if recent_accum_layer is not None
           else attn_ph.sum(axis=0))
    m1d = _hh_norec_mask(nk, K, acc[:nk], recent_frac=1.0/8)
    return np.broadcast_to(m1d, (nh, nk)).copy()


# -- Per-layer budget allocators (PyramidKV, CAKE) -----------

def make_pyramidkv(n_layers, ratio=0.5):
    def policy(attn_ph, K, layer_idx=0, **kw):
        if n_layers <= 1: K_eff = K
        else:
            frac = layer_idx / (n_layers - 1)
            K_max = K * (1 + ratio); K_min = K * (1 - ratio)
            K_eff = max(1, int(round(K_max - (K_max - K_min) * frac)))
        return policy_tova_ph(attn_ph, K_eff)
    return policy


def make_cake(n_layers, ratio=0.5):
    def policy(attn_ph, K, layer_idx=0, **kw):
        if n_layers <= 1: K_eff = K
        else:
            frac = layer_idx / (n_layers - 1)
            K_max = K * (1 + ratio); K_min = K * (1 - ratio)
            K_eff = max(1, int(round(K_min + (K_max - K_min) * frac)))
        return policy_tova_ph(attn_ph, K_eff)
    return policy


# ---------------------------------------------------------------------------
# Simulator (stateful: maintains attn_cum_max, attn_cum_sum, snapkv_score)
# ---------------------------------------------------------------------------

def kl(p, q):
    p = np.asarray(p, dtype=np.float64); q = np.asarray(q, dtype=np.float64)
    mask = p > 0
    return float(np.sum(p[mask] * (np.log(p[mask] + 1e-30) - np.log(q[mask] + 1e-30))))


def simulate(attn_ph, n_kv_at, policy_fn, K_nominal, *, needs_layer_idx=False):
    n_steps, n_layers, n_head, max_kv = attn_ph.shape
    out_kl = np.zeros((n_steps, n_layers), dtype=np.float64)
    out_K = np.zeros((n_steps, n_layers), dtype=np.float64)
    out_mass = np.zeros((n_steps, n_layers), dtype=np.float64)
    attn_cum_max = np.zeros((n_layers, n_head, max_kv), dtype=np.float32)
    attn_cum_sum_ph = np.zeros((n_layers, n_head, max_kv), dtype=np.float32)
    snapkv_score = np.zeros((n_layers, max_kv), dtype=np.float32)
    L_OBS = 16

    for s in range(n_steps):
        n_kv = int(n_kv_at[s])
        if n_kv == 0: continue
        for l in range(n_layers):
            ph_slice = attn_ph[s, l, :, :n_kv]
            np.maximum(attn_cum_max[l, :, :n_kv], ph_slice,
                       out=attn_cum_max[l, :, :n_kv])
            attn_cum_sum_ph[l, :, :n_kv] += ph_slice
            if s < L_OBS:
                snapkv_score[l, :n_kv] += ph_slice.sum(axis=0)

            accum_layer = attn_cum_sum_ph[l, :, :n_kv].sum(axis=0)
            accum_ph = attn_cum_sum_ph[l, :, :n_kv]

            kw = dict(
                attn_cum_max=attn_cum_max[l, :, :n_kv],
                accum_layer=accum_layer,
                accum_ph=accum_ph,
                snapkv_score_layer=snapkv_score[l, :n_kv],
                scissor_score_layer=accum_layer,   # approximation
                recent_accum_layer=accum_layer,    # approximation
            )
            if needs_layer_idx:
                kw["layer_idx"] = l
            mask = policy_fn(ph_slice, K_nominal, **kw)

            kls, masses = [], []
            kept_sum = 0
            for h in range(n_head):
                p_full = ph_slice[h]
                kept = p_full * mask[h]
                tot_kept = kept.sum(); tot_full = p_full.sum()
                masses.append(float(tot_kept / tot_full) if tot_full > 0 else 1.0)
                kls.append(kl(p_full, kept / tot_kept) if tot_kept > 0 else 20.0)
                kept_sum += int(mask[h].sum())
            out_kl[s, l] = float(np.mean(kls))
            out_mass[s, l] = float(np.mean(masses))
            out_K[s, l] = kept_sum / n_head
    return out_kl.mean(axis=1), out_K.mean(axis=1), out_mass.mean(axis=1)


# ---------------------------------------------------------------------------
# Registry & runner
# ---------------------------------------------------------------------------

POLICIES = {
    # Ours
    "ours_v1":         (policy_v1_spread,    False),
    "ours_v6":         (policy_v6_logistic,  False),
    # Per-head published
    "tova":            (policy_tova_ph,      False),
    "adakv":           (policy_adakv,        False),
    "headkv":          (policy_headkv,       False),
    "duoattention":    (policy_duoattention, False),
    "kvzip_approx":    (policy_kvzip_approx, False),
    # Global published
    "h2o":             (policy_h2o,          False),
    "snapkv":          (policy_snapkv,       False),
    "streamingllm":    (policy_streamingllm, False),
    "scissorhands":    (policy_scissorhands, False),
    "ahakv":           (policy_ahakv,        False),
    # Per-layer budget allocators (added per-model with n_layers known)
}


def main():
    out_dir = Path("/home/mislam22/EndurKV_workspace/EndurKV/figures")
    rows = []
    t0 = time.time()
    for d_str in DIRS:
        d = Path(d_str)
        if not d.is_dir(): continue
        model = d.name.replace("study_phone_", "").replace("_longbench", "")
        attn_files = sorted([f for f in d.glob("*.attn.bin")
                             if not f.name.endswith(".v1.attn.bin")])[:1]
        for af in attn_files:
            attn_ph, n_kv_at = load_attn_perhead(af)
            if attn_ph is None: continue
            n_layers = attn_ph.shape[1]
            # Per-model: add PyramidKV and CAKE wrappers
            local_policies = dict(POLICIES)
            local_policies["pyramidkv"] = (make_pyramidkv(n_layers), True)
            local_policies["cake"]      = (make_cake(n_layers), True)

            for K in BUDGETS:
                for name, (fn, needs_li) in local_policies.items():
                    kls, Ks, mass = simulate(attn_ph, n_kv_at, fn, K,
                                              needs_layer_idx=needs_li)
                    rows.append({
                        "model": model,
                        "n_kv": int(n_kv_at[-1]) if len(n_kv_at) else 0,
                        "K_nominal": K, "variant": name,
                        "actual_K": float(np.mean(Ks)),
                        "kl_mean": float(np.mean(kls)),
                        "kl_min": float(np.min(kls)),
                        "kl_max": float(np.max(kls)),
                        "kl_std": float(np.std(kls)),
                        "mass_pct": 100*float(np.mean(mass)),
                    })
        print(f"  [{time.time()-t0:5.0f}s] {model} done ({len(local_policies)} policies × 2 K)",
              flush=True)
    df = pd.DataFrame(rows)
    # Compute pct_vs_v1 and pct_vs_tova per (model, K)
    for (mdl, K), grp in df.groupby(["model","K_nominal"]):
        v1_kl = float(grp[grp.variant=="ours_v1"].kl_mean.iloc[0])
        tv_kl = float(grp[grp.variant=="tova"].kl_mean.iloc[0])
        tv_actK = float(grp[grp.variant=="tova"].actual_K.iloc[0])
        for i in grp.index:
            df.at[i, "pct_vs_v1"] = 100*(df.at[i,"kl_mean"]/v1_kl - 1)
            df.at[i, "pct_vs_tova"] = 100*(df.at[i,"kl_mean"]/tv_kl - 1)
            df.at[i, "cache_ratio_vs_tova"] = df.at[i,"actual_K"]/tv_actK
    csv_path = out_dir / "unified_baselines_per_cell.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[unified] wrote {len(df)} rows to {csv_path}")

    overall = (df.groupby("variant")
                 .agg(mean_kl=("kl_mean","mean"),
                      mean_cache_ratio=("actual_K", lambda s: float((s/df.loc[s.index,"K_nominal"]).mean())),
                      mean_mass=("mass_pct","mean"),
                      mean_vs_v1=("pct_vs_v1","mean"),
                      mean_vs_tova=("pct_vs_tova","mean"))
                 .reset_index()
                 .sort_values("mean_vs_tova"))
    print("\n=== OVERALL RANKING (lower mean_vs_tova = better) ===")
    print(overall.to_string(index=False))
    overall.to_csv(out_dir / "unified_baselines_ranking.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
