#!/usr/bin/env python3
# ============================================================================
# make_gpu_16k_table.py -- phone-GPU full-16K-context results table. (2026-08-02)
#
# WHY THIS EXISTS. The 256-token phone-GPU matrix reported a 2.12x Phi-3 decode
# speedup that does not survive a longer run: with only 256 tokens, a fixed
# post-prefill cost (~2.9s on vanilla, ~0.2s on muKV -- vanilla pays a larger
# one-time FA graph/alloc cost) is amortised over far too few tokens and inflates
# the ratio. Fitting decode_ms = n*t + C across the 256- and 4096-token runs
# recovers a consistent steady-state t for both policies. This script reports the
# 4096-token (full-ctx) numbers, which are the ones with no such contamination,
# and prints BOTH decode-only and wall speedups because eviction cannot shrink
# prefill and wall is prefill-dominated on this device.
#
# It also surfaces, per cell: whether compaction actually ran (compaction_applied
# -- a silent fallback when the second FA-on context fails to allocate would
# otherwise look like a policy result), retained cache, and how many times the
# muKV-only GPU watchdog v5 actually changed a clock tier.
# ============================================================================
import json, re, os, sys, glob

BASE = sys.argv[1] if len(sys.argv) > 1 else "/tmp/phone_gpu_16k"


def meta(tag):
    f = os.path.join(BASE, tag, "meta.json")
    if not os.path.exists(f):
        return None
    s = open(f).read()
    s = re.sub(r':\s*-?nan\b', ': NaN', s)
    s = re.sub(r':\s*-?inf\b', ': Infinity', s)
    return json.loads(s)


def wd_tiers(tag):
    f = os.path.join(BASE, tag, "gpu_wd_v5.log")
    if not os.path.exists(f):
        return None
    return sum(1 for L in open(f) if "tier=" in L)


ARMS = [("vanilla", "vanilla"), ("mukv_dfg", "muKV (compaction)"),
        ("mukv_nodfg", "muKV (no compaction)"), ("snapkv", "SnapKV")]

print("Phone GPU (Adreno 840) -- WikiText, 12K-token prompt + 4096 generated = full 16K ctx")
print("q8_0 KV, batch 1, cool gate (DDR<=35C, batt<=33C) before every cell.")
print("Watchdog: GPU v5, muKV cells only.  'dec x'/'wall x' are vs that model's vanilla.\n")
hdr = "%-8s %-22s %8s %8s %8s %8s %7s %7s %8s %6s %4s" % (
    "model", "policy", "prefill", "decode", "wall", "dec tps", "dec x", "wall x", "ret MiB", "cmpct", "wd")
print(hdr); print("-" * len(hdr))

for mt in ["llama1b", "phi3"]:
    v = meta(mt + "_vanilla")
    for suf, label in ARMS:
        j = meta("%s_%s" % (mt, suf))
        if j is None:
            d = os.path.join(BASE, "%s_%s" % (mt, suf))
            err = os.path.join(d, "err")
            msg = "not run"
            if os.path.exists(err):
                tail = [L for L in open(err, errors="replace").read().strip().split("\n") if L.strip()]
                if tail:
                    msg = tail[-1][:46]
            print("%-8s %-22s %s" % (mt, label, msg))
            continue
        pf, dc, w = j["prefill_ms"] / 1000, j["decode_ms"] / 1000, j["total_ms"] / 1000
        tps = j.get("decode_tps") or 0
        ret = (j.get("retained_kv_bytes") or 0) / 1048576.0
        dx = (v["decode_ms"] / j["decode_ms"]) if v else 0
        wx = (v["total_ms"] / j["total_ms"]) if v else 0
        t = wd_tiers("%s_%s" % (mt, suf))
        print("%-8s %-22s %8.1f %8.1f %8.1f %8.2f %6.2fx %6.2fx %8.1f %6s %4s" % (
            mt, label, pf, dc, w, tps, dx, wx, ret,
            j.get("compaction_applied"), "-" if t is None else t))
    print()

# the honest framing: wall is prefill-dominated on Adreno
for mt in ["llama1b", "phi3"]:
    v = meta(mt + "_vanilla")
    if v:
        share = 100.0 * v["prefill_ms"] / v["total_ms"]
        print("%-8s vanilla prefill is %.0f%% of wall -- eviction cannot shrink prefill, so a "
              "decode win of X shows up as far less than X end-to-end at this generation length."
              % (mt, share))
