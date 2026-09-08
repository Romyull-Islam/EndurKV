#!/usr/bin/env python3
"""
host_simulate_eviction_policies.py — offline simulation of KV-cache eviction
policies on attention sidecars captured by attention_probe.

For each prompt we already have .attn.bin (per-step, per-layer, n_kv float32
attention probabilities) and .entropy.csv (per-step H_nats). We use these to
simulate, without re-running on phone, what would have happened at decode
step t if we had been operating with a budget-K cache and a particular
eviction rule.

The metric is per-step, per-layer KL divergence between the full attention
distribution and the renormalized "kept-only" distribution:

    p_evicted_i = (p_full_i * mask_i) / sum_j (p_full_j * mask_j)
    KL = sum_i p_full_i * log(p_full_i / p_evicted_i)

Lower KL = closer to no-eviction-at-all = better preserved generation.

Policies implemented:
    full        — keep everything (reference, KL = 0)
    local       — keep K most recent positions
    h2o         — accumulated attention score; K/2 heavy hitters + K/2 recent
    h2o_norec   — same scoring, K=15/16 heavy + 1/16 recent  (slide-24 finding)
    endurkv     — h2o-style scoring, but at each step the prune-fraction is
                  modulated by entropy:  evicted_now = base * (1 - H_tilde)
                  We approximate this by giving uncertain steps more cache:
                  K_t = K_base + (K_max - K_base) * H_tilde_t
    streamingllm — first 4 (sink) + K-4 most recent
    oracle      — Belady-style upper bound: keep the K positions whose
                  removal would minimize KL at the NEXT step (cheats by
                  looking ahead one step). Useful as a topline only.

Output:
    <log_dir>_eviction/policy_results.csv     — long-form (prompt, layer, step,
                                                policy, budget, kl)
    <log_dir>_eviction/per_policy_summary.csv — aggregated by policy
    <log_dir>_eviction/per_task_summary.csv   — aggregated by (policy, task)
    <log_dir>_eviction/01_kl_by_policy.png    — boxplot of step-mean KL
    <log_dir>_eviction/02_kl_vs_budget.png    — KL as a function of budget K
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


# ---------- attn.bin parser → padded 3D array ------------------------------
def load_attn_full(path: Path):
    """Return (attn[steps, layers, max_n_kv], n_kv_at_step[steps]).

    Cells beyond per-step n_kv are filled with 0.0. n_kv_at_step[t] is the
    actual length for step t.
    """
    if not path.exists():
        return None, None
    with open(path, "rb") as f:
        if f.read(4) != b"ATTN":
            return None, None
        n_steps, n_layers, _n_head = struct.unpack("<III", f.read(12))
        per_step_layer = []  # list of list-of-arrays
        n_kv_at = []
        for _s in range(n_steps):
            layers = []
            last_nkv = 0
            for _l in range(n_layers):
                (n_kv,) = struct.unpack("<I", f.read(4))
                if n_kv == 0:
                    layers.append(None)
                    continue
                last_nkv = max(last_nkv, n_kv)
                vals = np.frombuffer(f.read(4 * n_kv), dtype=np.float32)
                layers.append(vals)
            per_step_layer.append(layers)
            n_kv_at.append(last_nkv)

    if not per_step_layer:
        return None, None
    max_kv = max(n_kv_at) if n_kv_at else 0
    attn = np.zeros((n_steps, n_layers, max_kv), dtype=np.float32)
    for s, layers in enumerate(per_step_layer):
        for l, arr in enumerate(layers):
            if arr is not None:
                attn[s, l, :len(arr)] = arr
    return attn, np.array(n_kv_at, dtype=np.int32)


# ---------- policies --------------------------------------------------------
# All policies return a boolean mask of shape (n_kv,) where True = keep.
# Inputs:
#   step              — current decode step (0-indexed)
#   n_kv              — current size of populated KV cache
#   K                 — target budget (max positions to keep)
#   accum_attn[layer] — per-layer accumulated attention score per source position
#                       (shape (n_layers, max_kv) updated as we go)
#   H_tilde_t         — current normalized entropy (for endurkv policy)
#   layer_idx         — for layer-aware policies

def mask_full(step, n_kv, K, **kw):
    m = np.zeros(n_kv, dtype=bool)
    m[:n_kv] = True
    return m


def mask_local(step, n_kv, K, **kw):
    """Keep the K most recent positions."""
    m = np.zeros(n_kv, dtype=bool)
    start = max(0, n_kv - K)
    m[start:n_kv] = True
    return m


def mask_h2o(step, n_kv, K, accum_attn_l, **kw):
    """H2O: top-K/2 by accumulated attention + K/2 most recent."""
    if n_kv <= K:
        m = np.zeros(n_kv, dtype=bool); m[:n_kv] = True; return m
    K_hh = K // 2
    K_rec = K - K_hh
    # heavy hitters: top K_hh by accumulated score, EXCLUDING recent window
    rec_start = max(0, n_kv - K_rec)
    cand = accum_attn_l[:rec_start]
    if len(cand) <= K_hh:
        hh_idx = np.arange(rec_start)
    else:
        hh_idx = np.argpartition(-cand, K_hh)[:K_hh]
    m = np.zeros(n_kv, dtype=bool)
    m[hh_idx] = True
    m[rec_start:n_kv] = True
    return m


def mask_h2o_norec(step, n_kv, K, accum_attn_l, **kw):
    """Same scoring as H2O but only 1/16 recent (paper slide 24 finding)."""
    if n_kv <= K:
        m = np.zeros(n_kv, dtype=bool); m[:n_kv] = True; return m
    K_rec = max(1, K // 16)
    K_hh = K - K_rec
    rec_start = max(0, n_kv - K_rec)
    cand = accum_attn_l[:rec_start]
    if len(cand) <= K_hh:
        hh_idx = np.arange(rec_start)
    else:
        hh_idx = np.argpartition(-cand, K_hh)[:K_hh]
    m = np.zeros(n_kv, dtype=bool)
    m[hh_idx] = True
    m[rec_start:n_kv] = True
    return m


def mask_streamingllm(step, n_kv, K, **kw):
    """First 4 attention sinks + K-4 most recent."""
    if n_kv <= K:
        m = np.zeros(n_kv, dtype=bool); m[:n_kv] = True; return m
    n_sink = min(4, K // 2)
    K_rec = K - n_sink
    m = np.zeros(n_kv, dtype=bool)
    m[:n_sink] = True
    m[max(0, n_kv - K_rec):n_kv] = True
    return m


def mask_endurkv(step, n_kv, K, accum_attn_l, H_tilde_t=0.5, **kw):
    """Entropy-gated H2O (UNCAPPED variant).

    Idea: at uncertain steps (H_tilde -> 1) we prune LESS — i.e. budget grows.
    At committed steps (H_tilde -> 0) we prune at full rate. Uses K_t =
    K * (1 + H_tilde_t), capped at n_kv. AVERAGE cache used exceeds K
    because the bonus is one-sided. Useful as an upper bound. The fair
    head-to-head comparison is endurkv_matched below.
    """
    K_t = int(min(n_kv, K * (1.0 + max(0.0, H_tilde_t))))
    return mask_h2o_norec(step, n_kv, K_t, accum_attn_l=accum_attn_l)


def _mask_endurkv_matched_factory(half_range):
    """Build a budget-matched policy with mult in [1-half_range, 1+half_range]."""
    def _fn(step, n_kv, K, accum_attn_l, H_tilde_t=0.5, **kw):
        mult = (1.0 - half_range) + 2 * half_range * max(0.0, min(1.0, H_tilde_t))
        K_t = int(round(K * mult))
        K_t = max(1, min(n_kv, K_t))
        return mask_h2o_norec(step, n_kv, K_t, accum_attn_l=accum_attn_l)
    return _fn


def mask_endurkv_attn_spread(step, n_kv, K, accum_attn_l, attn_full_layer=None, **kw):
    """Budget-MATCHED policy modulated by attention spread instead of entropy.

    Use the current layer's max attention probability as the concentration
    signal. High max_attn (peaked) -> prune more. Low max_attn (diffuse) ->
    prune less. Multiplier range [0.7, 1.3] keeps avg-K = K.
    """
    if attn_full_layer is None or len(attn_full_layer) == 0:
        return mask_h2o_norec(step, n_kv, K, accum_attn_l=accum_attn_l)
    max_a = float(attn_full_layer[:n_kv].max())
    # Map max_a in [0.4, 0.8] (typical range observed) to mult in [1.3, 0.7]
    norm = max(0.0, min(1.0, (max_a - 0.4) / 0.4))   # 0..1
    mult = 1.3 - 0.6 * norm                          # 1.3 .. 0.7
    K_t = int(round(K * mult))
    K_t = max(1, min(n_kv, K_t))
    return mask_h2o_norec(step, n_kv, K_t, accum_attn_l=accum_attn_l)


POLICIES = {
    "full":                   mask_full,
    "local":                  mask_local,
    "h2o":                    mask_h2o,
    "h2o_norec":              mask_h2o_norec,
    "streamingllm":           mask_streamingllm,
    "endurkv":                mask_endurkv,                              # 0..1   -> 1..2x  (uncapped)
    "endurkv_matched":        _mask_endurkv_matched_factory(0.5),         # 0.5..1.5
    "endurkv_matched_25":     _mask_endurkv_matched_factory(0.25),        # 0.75..1.25
    "endurkv_matched_10":     _mask_endurkv_matched_factory(0.10),        # 0.90..1.10
    "endurkv_attn_spread":    mask_endurkv_attn_spread,
}


# ---------- KL helper -------------------------------------------------------
def kl_divergence(p_full, p_evicted, eps=1e-12):
    """KL(p_full || p_evicted) safely."""
    p_full = np.clip(p_full, eps, 1.0)
    p_evicted = np.clip(p_evicted, eps, 1.0)
    return float(np.sum(p_full * (np.log(p_full) - np.log(p_evicted))))


def simulate(attn, n_kv_at, policy_name, K, H_norm=None):
    """Return per-(step, layer) KL array of shape (n_steps, n_layers).

    H_norm: optional per-step normalized entropy (in [0,1]) for endurkv.
    """
    n_steps, n_layers, max_kv = attn.shape
    accum = np.zeros((n_layers, max_kv), dtype=np.float64)  # per-layer
    out = np.zeros((n_steps, n_layers), dtype=np.float64)
    policy_fn = POLICIES[policy_name]
    for s in range(n_steps):
        n_kv = int(n_kv_at[s])
        if n_kv == 0:
            continue
        # update accumulated attention with the CURRENT step's attention
        # (so the score at step s reflects steps 0..s, mimicking H2O paper)
        for l in range(n_layers):
            accum[l, :n_kv] += attn[s, l, :n_kv]

        Ht = 0.5 if H_norm is None else float(H_norm[s] if s < len(H_norm) else 0.5)
        for l in range(n_layers):
            p_full = attn[s, l, :n_kv]
            mask = policy_fn(step=s, n_kv=n_kv, K=K,
                             accum_attn_l=accum[l, :n_kv],
                             attn_full_layer=p_full,
                             H_tilde_t=Ht, layer_idx=l)
            # renormalize the kept portion
            kept = p_full * mask
            tot = kept.sum()
            if tot <= 0:
                # everything evicted — KL is infinite; cap at 20 as
                # a numerical proxy so the boxplot doesn't explode
                out[s, l] = 20.0
                continue
            p_evicted = kept / tot
            out[s, l] = kl_divergence(p_full, p_evicted)
    return out


# ---------- entropy normalization (rolling) ---------------------------------
def normalize_entropy(H_nats, n_vocab_log=None, window=64):
    """Map raw H_nats to [0,1] via rolling-window min-max, matching the
    proposal's H_tilde definition.
    """
    H = np.asarray(H_nats, dtype=np.float64)
    if len(H) == 0:
        return H
    # rolling window
    out = np.zeros_like(H)
    for i in range(len(H)):
        a = max(0, i - window + 1)
        seg = H[a : i + 1]
        lo, hi = seg.min(), seg.max()
        if hi - lo < 1e-9:
            out[i] = 0.5
        else:
            out[i] = (H[i] - lo) / (hi - lo)
    return out


# ---------- main ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--budgets", default="32,64,128,256",
                    help="comma-separated K values to simulate")
    ap.add_argument("--max-prompts", type=int, default=0,
                    help="cap prompts processed (0 = all)")
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir) if args.out_dir else log_dir.parent / f"{log_dir.name}_eviction"
    out_dir.mkdir(parents=True, exist_ok=True)
    budgets = [int(b) for b in args.budgets.split(",") if b.strip()]

    attn_files = sorted(log_dir.glob("*.attn.bin"))
    if args.max_prompts > 0:
        attn_files = attn_files[: args.max_prompts]
    print(f"[sim] {len(attn_files)} prompts, budgets={budgets}, policies={list(POLICIES.keys())}")

    long_rows = []
    for ap_path in attn_files:
        pid = ap_path.name[: -len(".attn.bin")]
        ent_path = log_dir / f"{pid}.entropy.csv"
        if not ent_path.exists():
            continue
        attn, n_kv_at = load_attn_full(ap_path)
        if attn is None or attn.shape[0] == 0:
            continue
        try:
            ent = pd.read_csv(ent_path)
        except Exception:
            continue
        if ent.empty:
            continue
        H = ent["H_nats"].values[:attn.shape[0]]
        H_norm = normalize_entropy(H)
        task = pid.rsplit("_", 1)[0]
        if task.endswith("_lc"):
            task = task[:-3]

        n_steps, n_layers, max_kv = attn.shape
        print(f"  {pid:<28} steps={n_steps} layers={n_layers} max_kv={max_kv}")
        for K in budgets:
            for pname in POLICIES.keys():
                kls = simulate(attn, n_kv_at, pname, K, H_norm=H_norm)
                # per-prompt aggregate: mean across (step, layer)
                mean_kl = float(np.mean(kls))
                p95_kl  = float(np.percentile(kls, 95))
                long_rows.append({
                    "prompt_id": pid, "task": task,
                    "policy": pname, "budget_K": K,
                    "mean_kl": mean_kl, "p95_kl": p95_kl,
                    "n_steps": n_steps, "n_layers": n_layers,
                    "max_kv": max_kv,
                })

    if not long_rows:
        print("ERROR: no prompts processed", file=sys.stderr)
        return 1

    df = pd.DataFrame(long_rows)
    df.to_csv(out_dir / "policy_results.csv", index=False)
    print(f"\n[sim] wrote policy_results.csv  ({len(df)} rows)")

    # per-policy summary
    per_pol = df.groupby(["policy", "budget_K"]).agg(
        mean_kl_avg=("mean_kl", "mean"),
        mean_kl_median=("mean_kl", "median"),
        p95_kl_avg=("p95_kl", "mean"),
        n_prompts=("prompt_id", "nunique"),
    ).reset_index()
    per_pol.to_csv(out_dir / "per_policy_summary.csv", index=False)

    # per-task summary
    per_task = df.groupby(["policy", "budget_K", "task"]).agg(
        mean_kl_avg=("mean_kl", "mean"),
        n_prompts=("prompt_id", "nunique"),
    ).reset_index()
    per_task.to_csv(out_dir / "per_task_summary.csv", index=False)

    # ---------- plots ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 130, "font.size": 9})

    # 01 — boxplot of mean_kl by policy at the median budget
    median_K = budgets[len(budgets)//2]
    sub = df[df.budget_K == median_K]
    polys = ["full", "local", "streamingllm", "h2o", "h2o_norec", "endurkv"]
    polys = [p for p in polys if p in sub.policy.unique()]
    fig, ax = plt.subplots(figsize=(10, 5))
    data = [sub[sub.policy == p]["mean_kl"].values for p in polys]
    ax.boxplot(data, tick_labels=polys, showmeans=True)
    ax.set_yscale("symlog", linthresh=0.01)
    ax.set_ylabel("per-prompt mean KL (full || evicted), log-symlog")
    ax.set_title(f"Eviction-policy comparison at budget K={median_K} (n={sub.prompt_id.nunique()} prompts)")
    ax.grid(alpha=0.3, axis="y", which="both")
    fig.tight_layout()
    fig.savefig(out_dir / "01_kl_by_policy.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote 01_kl_by_policy.png")

    # 02 — KL as function of budget K
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for p in polys:
        if p == "full":
            continue
        ys = []
        for K in budgets:
            ys.append(df[(df.policy == p) & (df.budget_K == K)]["mean_kl"].mean())
        ax.plot(budgets, ys, "o-", label=p)
    ax.set_xlabel("budget K (cache slots)")
    ax.set_ylabel("mean KL over all (prompt, step, layer)")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_title("KL vs budget K — lower is better, log-log")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_dir / "02_kl_vs_budget.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote 02_kl_vs_budget.png")

    # 03 — per-task per-policy heatmap-like bar chart at median budget
    fig, ax = plt.subplots(figsize=(12, 6))
    tasks = sorted(sub.task.unique())
    width = 0.8 / len(polys)
    x = np.arange(len(tasks))
    for i, p in enumerate(polys):
        if p == "full":
            continue
        ys = [sub[(sub.task == t) & (sub.policy == p)]["mean_kl"].mean() for t in tasks]
        ax.bar(x + i * width, ys, width, label=p)
    ax.set_xticks(x + width * (len(polys) - 1) / 2)
    ax.set_xticklabels(tasks, rotation=35, ha="right")
    ax.set_ylabel("mean KL per prompt")
    ax.set_yscale("log")
    ax.set_title(f"Per-task KL by policy at K={median_K}")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3, axis="y", which="both")
    fig.tight_layout()
    fig.savefig(out_dir / "03_per_task_kl.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote 03_per_task_kl.png")

    # summary print
    print("\n=== per-policy mean KL at each budget ===")
    pivot = df.groupby(["policy", "budget_K"])["mean_kl"].mean().unstack()
    print(pivot.to_string(float_format=lambda x: f"{x:.4f}"))

    print(f"\nall outputs in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
