#!/usr/bin/env python3
"""Sweep over the SIGNAL and SELECTION axes of the eviction problem.

The BUDGET axis is fixed at the winner of the gate-shape sweep:
    sigmoid(α=1.5, β=0.5, c=0.4, γ=8.0)  →  K_h per head

What varies:
  SIGNAL    — the score that ranks each position for top-K selection
  SELECTION — the rule that picks the top-K from those scores

Designed to find which physics/math-inspired ranking beats TOVA's plain
argmax-of-current-attention.

Outputs:
  EndurKV/figures/signal_selection_results.csv
  EndurKV/figures/signal_selection_ranking.csv
"""
import sys, time
from multiprocessing import Pool, set_start_method
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from host_simulate_kv_baselines import load_attn_perhead


# ---------------------------------------------------------------------------
# Best gate (winner from gate_search_mp): sigmoid α=1.5, β=0.5, c=0.4, γ=8.0
# ---------------------------------------------------------------------------
def best_gate_K_h(max_a):
    """Returns multiplier on K_nominal for a head with this max_a."""
    sig = 1.0 / (1.0 + np.exp(8.0 * (max_a - 0.4)))
    return 0.5 + (1.5 - 0.5) * sig


# ---------------------------------------------------------------------------
# Signal computers — each returns score[h, i] for the current step+layer
# All take (a_now, cum_max, cum_sum) and return [n_head, n_kv] scores
# ---------------------------------------------------------------------------
def signal_current(a_now, cum_max, cum_sum):
    return a_now

def signal_cum_max(a_now, cum_max, cum_sum):
    return cum_max

def signal_cum_sum(a_now, cum_max, cum_sum):
    return cum_sum

def signal_hybrid(a_now, cum_max, cum_sum):
    return 0.7 * a_now + 0.3 * cum_max

def _diffuse1d(x, w_self=0.6, w_nbr=0.2):
    """1-step Laplacian smoothing along the position axis."""
    nh, nk = x.shape
    out = np.empty_like(x)
    out[:, 0]    = w_self * x[:, 0]    + w_nbr * 2.0 * x[:, 1]
    out[:, -1]   = w_self * x[:, -1]   + w_nbr * 2.0 * x[:, -2]
    out[:, 1:-1] = w_self * x[:, 1:-1] + w_nbr * (x[:, :-2] + x[:, 2:])
    return out

def signal_diffuse_current(a_now, cum_max, cum_sum):
    return _diffuse1d(a_now)

def signal_diffuse_cum_max(a_now, cum_max, cum_sum):
    return _diffuse1d(cum_max)

def signal_diffuse3_current(a_now, cum_max, cum_sum):
    x = a_now
    for _ in range(3):
        x = _diffuse1d(x, w_self=0.6, w_nbr=0.2)
    return x

SIGNALS = {
    "current":        signal_current,
    "cum_max":        signal_cum_max,
    "cum_sum":        signal_cum_sum,
    "hybrid":         signal_hybrid,
    "diffuse_curr":   signal_diffuse_current,
    "diffuse_cmax":   signal_diffuse_cum_max,
    "diffuse3_curr":  signal_diffuse3_current,
}

# ---------------------------------------------------------------------------
# Selection rules — each takes (score[h], K_h) and returns boolean mask
# ---------------------------------------------------------------------------
def sel_argmax(score_h, K_h, nk):
    if K_h >= nk:
        return np.ones(nk, dtype=bool)
    idx = np.argpartition(-score_h, K_h - 1)[:K_h]
    m = np.zeros(nk, dtype=bool); m[idx] = True
    return m

def sel_wave_argmax(score_h, K_h, nk):
    """Smooth score one extra step before argmax (selection-level diffusion)."""
    if K_h >= nk:
        return np.ones(nk, dtype=bool)
    if nk >= 3:
        s = np.empty_like(score_h)
        s[0]    = 0.6*score_h[0]   + 0.4*score_h[1]
        s[-1]   = 0.6*score_h[-1]  + 0.4*score_h[-2]
        s[1:-1] = 0.6*score_h[1:-1] + 0.2*(score_h[:-2] + score_h[2:])
    else:
        s = score_h
    idx = np.argpartition(-s, K_h - 1)[:K_h]
    m = np.zeros(nk, dtype=bool); m[idx] = True
    return m

def sel_threshold_then_topk(score_h, K_h, nk):
    """Keep all positions with score > median + 0.5*MAD; cap at K_h.
    Below cap, fewer than K_h positions may be kept (variable selection)."""
    if K_h >= nk: return np.ones(nk, dtype=bool)
    med = np.median(score_h)
    mad = np.median(np.abs(score_h - med)) + 1e-9
    thr = med + 0.5 * mad
    candidates = np.where(score_h > thr)[0]
    if len(candidates) > K_h:
        # Pick top-K_h among those above threshold
        sub_scores = score_h[candidates]
        sub_top = np.argpartition(-sub_scores, K_h - 1)[:K_h]
        idx = candidates[sub_top]
    else:
        idx = candidates
    m = np.zeros(nk, dtype=bool); m[idx] = True
    return m

def sel_gumbel_topk(score_h, K_h, nk):
    """Add Gumbel noise then top-K — stochastic relaxation of argmax."""
    if K_h >= nk: return np.ones(nk, dtype=bool)
    eps = 1e-9
    logits = np.log(np.maximum(score_h, eps))
    rng = np.random.default_rng(42)
    g = -np.log(-np.log(rng.uniform(eps, 1.0, size=nk)))
    perturbed = logits + 0.5 * g  # temperature 0.5
    idx = np.argpartition(-perturbed, K_h - 1)[:K_h]
    m = np.zeros(nk, dtype=bool); m[idx] = True
    return m

SELECTIONS = {
    "argmax":    sel_argmax,
    "wave":      sel_wave_argmax,
    "threshold": sel_threshold_then_topk,
    "gumbel":    sel_gumbel_topk,
}


# ---------------------------------------------------------------------------
# Custom simulator — like host_simulate_kv_baselines.simulate but maintains
# cum_max AND cum_sum per (layer, head, position), and applies a configurable
# (signal, selection) pair.
# ---------------------------------------------------------------------------
def kl(p, q, eps=1e-12):
    pf = np.clip(p.astype(np.float32, copy=False), eps, 1.0)
    pe = np.clip(q.astype(np.float32, copy=False), eps, 1.0)
    return float(np.sum(pf * (np.log(pf) - np.log(pe))))


def simulate_signal_select(attn_ph, n_kv_at, K_nominal, signal_name, selection_name):
    n_steps, n_layers, n_head, max_kv = attn_ph.shape
    sig_fn = SIGNALS[signal_name]
    sel_fn = SELECTIONS[selection_name]
    out_kl = np.zeros((n_steps, n_layers), dtype=np.float64)
    out_K  = np.zeros((n_steps, n_layers), dtype=np.float64)
    out_mass = np.zeros((n_steps, n_layers), dtype=np.float64)
    cum_max = np.zeros((n_layers, n_head, max_kv), dtype=np.float32)
    cum_sum = np.zeros((n_layers, n_head, max_kv), dtype=np.float32)

    for s in range(n_steps):
        n_kv = int(n_kv_at[s])
        if n_kv == 0: continue
        for l in range(n_layers):
            ph = attn_ph[s, l, :, :n_kv]
            np.maximum(cum_max[l, :, :n_kv], ph, out=cum_max[l, :, :n_kv])
            cum_sum[l, :, :n_kv] += ph
            # Compute score per (head, position)
            score = sig_fn(ph, cum_max[l, :, :n_kv], cum_sum[l, :, :n_kv])

            kls, masses = [], []
            kept_sum = 0
            for h in range(n_head):
                max_a = float(ph[h].max())
                mult = best_gate_K_h(max_a)
                K_h = max(1, min(n_kv, int(round(K_nominal * mult))))
                mask_h = sel_fn(score[h], K_h, n_kv)
                p_full = ph[h]
                kept = p_full * mask_h
                tk = kept.sum(); tf = p_full.sum()
                masses.append(float(tk / tf) if tf > 0 else 1.0)
                kls.append(kl(p_full, kept / tk) if tk > 0 else 20.0)
                kept_sum += int(mask_h.sum())
            out_kl[s, l] = float(np.mean(kls))
            out_mass[s, l] = float(np.mean(masses))
            out_K[s, l] = kept_sum / n_head
    return out_kl.mean(axis=1), out_K.mean(axis=1), out_mass.mean(axis=1)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
DIRS = [
    "/home/mislam22/EndurKV_workspace/logs/study_phone_phi3_longbench",
    "/home/mislam22/EndurKV_workspace/logs/study_phone_mistral_longbench",
    "/home/mislam22/EndurKV_workspace/logs/study_phone_qwen2_longbench",
    "/home/mislam22/EndurKV_workspace/logs/study_phone_gemma2_longbench",
    "/home/mislam22/EndurKV_workspace/logs/study_phone_r1distill_longbench",
]
BUDGETS = [512, 1024]

CONFIGS = []
for sig in SIGNALS:
    for sel in SELECTIONS:
        CONFIGS.append((sig, sel))
print(f"[sig-sel] {len(CONFIGS)} (signal, selection) pairs to test")

ATTN_PER_MODEL = {}
TOVA_REF = {}


def init_data():
    global ATTN_PER_MODEL, TOVA_REF
    from host_simulate_kv_baselines import simulate as sim_baseline
    for d_str in DIRS:
        d = Path(d_str)
        if not d.is_dir(): continue
        model = d.name.replace("study_phone_", "").replace("_longbench", "")
        attn_files = sorted([f for f in d.glob("*.attn.bin")
                             if not f.name.endswith(".v1.attn.bin")])[:1]
        for af in attn_files:
            attn_ph, n_kv_at = load_attn_perhead(af)
            if attn_ph is None: continue
            ATTN_PER_MODEL[model] = (attn_ph, n_kv_at)
            print(f"[sig-sel] preloaded {model}", flush=True)
    print(f"[sig-sel] computing TOVA refs ...", flush=True)
    for model, (attn_ph, n_kv_at) in ATTN_PER_MODEL.items():
        for K in BUDGETS:
            kls, Ks, _, _ = sim_baseline(attn_ph, n_kv_at, None, None, "perhead_tova", K)
            TOVA_REF[(model, K)] = (float(np.mean(kls)), float(np.mean(Ks)))
    print(f"[sig-sel] TOVA ref done", flush=True)


def worker(args):
    cfg_idx, model, K = args
    sig, sel = CONFIGS[cfg_idx]
    attn_ph, n_kv_at = ATTN_PER_MODEL[model]
    tova_kl, tova_K = TOVA_REF[(model, K)]
    kls, Ks, mass = simulate_signal_select(attn_ph, n_kv_at, K, sig, sel)
    return {
        "signal": sig, "selection": sel,
        "model": model, "K_nominal": K,
        "actual_K": float(np.mean(Ks)),
        "cache_ratio_vs_tova": float(np.mean(Ks)) / tova_K,
        "kl_mean": float(np.mean(kls)),
        "mass_pct": 100*float(np.mean(mass)),
        "delta_tova_pct": 100*(float(np.mean(kls)) / tova_kl - 1),
    }


def main():
    out_dir = Path("/home/mislam22/EndurKV_workspace/EndurKV/figures")
    init_data()
    tasks = [(i, m, K) for i in range(len(CONFIGS))
                       for m in ATTN_PER_MODEL.keys()
                       for K in BUDGETS]
    print(f"[sig-sel] {len(tasks)} tasks "
          f"({len(CONFIGS)} configs × {len(ATTN_PER_MODEL)} models × {len(BUDGETS)} K)")
    t0 = time.time()
    rows = []
    with Pool(processes=16) as pool:
        for i, row in enumerate(pool.imap_unordered(worker, tasks, chunksize=2)):
            rows.append(row)
            if (i+1) % 20 == 0 or i+1 == len(tasks):
                print(f"  [{time.time()-t0:5.0f}s] {i+1}/{len(tasks)} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "signal_selection_results.csv", index=False)
    print(f"\n[sig-sel] wrote {len(df)} rows in {time.time()-t0:.0f}s")

    agg = (df.groupby(["signal","selection"])
             .agg(mean_delta_tova=("delta_tova_pct","mean"),
                  mean_cache=("cache_ratio_vs_tova","mean"),
                  mean_mass=("mass_pct","mean"),
                  n=("delta_tova_pct","count"))
             .reset_index())
    agg["cache_pp_over"] = (agg["mean_cache"]-1.0)*100
    agg["cache_adj"] = agg["mean_delta_tova"] + agg["cache_pp_over"].clip(lower=0)
    agg = agg.sort_values("cache_adj")
    agg.to_csv(out_dir / "signal_selection_ranking.csv", index=False)
    print("\n=== TOP-15 by cache-adjusted Δ TOVA ===")
    print(agg.head(15).to_string(index=False))
    print("\n=== Cache-neutral (cache ≤ 1.05) — sorted by raw Δ TOVA ===")
    print(agg[agg.mean_cache <= 1.05].sort_values("mean_delta_tova").to_string(index=False))
    print("\n=== Best signal per selection rule ===")
    for sel in SELECTIONS:
        s = agg[agg.selection==sel].sort_values("cache_adj").head(1)
        if not s.empty:
            r = s.iloc[0]
            print(f"{sel:12s}: signal={r['signal']:14s}  Δ={r['mean_delta_tova']:+.2f}%  cache={r['mean_cache']:.3f}  adj={r['cache_adj']:+.2f}%")
    return 0


if __name__ == "__main__":
    set_start_method("fork")
    sys.exit(main())
