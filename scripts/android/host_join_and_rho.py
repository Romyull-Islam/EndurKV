#!/usr/bin/env python3
"""
host_join_and_rho.py — post-run analysis for the on-phone study.

For each prompt_id in --log-dir:
  * read <id>.entropy.csv  (per-step entropy + top-k)
  * read <id>.sensors.csv  (per-sample thermal + memory + endurance signals)
  * read <id>.run.json     (start_wall_s for time-join)
  * IF .attn.bin present: parse and compute per-step layer-averaged top-1 attention
                          (the metric used in the paper for ρ = -0.37)

Outputs:
  <log_dir>.joined.csv     all per-step rows joined to nearest sensor sample
                           + an `attn_top1_layer_avg` column when .attn.bin was found

Console summary:
  * Pooled Spearman ρ between H_nats and attn_top1_layer_avg (paper metric)
  * Pooled Spearman ρ between H_nats and top1_prob (entropy_probe fallback)
  * Gate decision against the proposal's fallback threshold ρ <= -0.20
  * Thermal trajectory: max ACTIVE zone (filtering trip-point zones with constant
    high values), and the dT over the run
  * pswpout delta in pages and KB (4 KB pages) — primary non-root endurance signal

Run:
  python scripts/android/host_join_and_rho.py [--log-dir logs/study_phone_1b]
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
WORKSPACE = Path(os.environ.get("WORKSPACE", HERE.parents[3]))


# ---------- attn.bin parser --------------------------------------------------
def parse_attn_bin(path: Path) -> list[float]:
    """Return a list of length n_steps with per-step layer-averaged top-1 attention.

    Returns [] if the file is missing or malformed.
    """
    if not path.exists():
        return []
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != b"ATTN":
                return []
            n_steps, n_layers, _n_head = struct.unpack("<III", f.read(12))
            out: list[float] = []
            for _s in range(n_steps):
                top1_per_layer = []
                for _l in range(n_layers):
                    (n_kv,) = struct.unpack("<I", f.read(4))
                    if n_kv == 0:
                        # layer not captured this step
                        continue
                    vals = struct.unpack(f"<{n_kv}f", f.read(4 * n_kv))
                    if vals:
                        top1_per_layer.append(max(vals))
                if top1_per_layer:
                    out.append(sum(top1_per_layer) / len(top1_per_layer))
                else:
                    out.append(float("nan"))
            return out
    except Exception as e:
        print(f"  WARN: parse_attn_bin({path}) failed: {e}")
        return []


# ---------- thermal zone classifier ------------------------------------------
# Thermal zones that report STATIC values are configured trip thresholds, not
# real readings; they pollute "max temp" stats.  Filter them out.
TRIP_PATTERNS = ("cpu-hw-trip", "_trip_", "bcl-lvl", "ibat-lvl", "vbat", "pmh", "pmr", "pmih010")


def is_trip_zone(col: str) -> bool:
    low = col.lower()
    return any(pat in low for pat in TRIP_PATTERNS)


# ---------- main ------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default=str(WORKSPACE / "logs" / "study_phone_1b"))
    ap.add_argument("--out",     default="")
    args = ap.parse_args()
    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"ERROR: {log_dir} does not exist", file=sys.stderr)
        return 1

    try:
        import pandas as pd
    except ImportError:
        print("ERROR: pandas required. pip install pandas scipy", file=sys.stderr)
        return 2
    try:
        from scipy.stats import spearmanr
    except ImportError:
        print("ERROR: scipy required. pip install scipy", file=sys.stderr)
        return 2

    ent_files = sorted(log_dir.glob("*.entropy.csv"))
    if not ent_files:
        print(f"ERROR: no *.entropy.csv files in {log_dir}", file=sys.stderr)
        return 1

    joined_frames: list = []
    summary_rows: list[dict] = []
    have_attn = False

    for ent_path in ent_files:
        pid = ent_path.name[: -len(".entropy.csv")]
        sen_path  = log_dir / f"{pid}.sensors.csv"
        run_path  = log_dir / f"{pid}.run.json"
        attn_path = log_dir / f"{pid}.attn.bin"

        if not sen_path.exists() or not run_path.exists():
            print(f"  skip {pid}: missing sensors.csv or run.json")
            continue

        try:
            ent = pd.read_csv(ent_path)
            sen = pd.read_csv(sen_path)
            run = json.loads(run_path.read_text())
        except Exception as e:
            print(f"  skip {pid}: parse error: {e}")
            continue
        if ent.empty or sen.empty:
            print(f"  skip {pid}: empty probe or sensor data")
            continue

        # Wall-clock join.
        ent["step_wall_s"] = run["start_wall_s"] + ent["wall_clock_us"] / 1e6

        # Optional attention sidecar — produces a new column.
        attn_top1 = parse_attn_bin(attn_path)
        if attn_top1:
            have_attn = True
            # Truncate either side if lengths disagree (shouldn't, but safe).
            n = min(len(attn_top1), len(ent))
            ent = ent.iloc[:n].copy()
            ent["attn_top1_layer_avg"] = attn_top1[:n]

        sen = sen.sort_values("wall_clock_s").reset_index(drop=True)
        ent = ent.sort_values("step_wall_s").reset_index(drop=True)
        merged = pd.merge_asof(
            ent, sen,
            left_on="step_wall_s",
            right_on="wall_clock_s",
            direction="nearest",
            tolerance=1.0,
        )
        merged["prompt_id"] = pid
        # run.json doesn't carry the task, but every prompt_id is "<task>_<NNN>".
        # Derive task from the prefix so per-task rollups work without a join
        # back to the original prompts.jsonl.
        if "_" in pid and pid.rsplit("_", 1)[-1].isdigit():
            merged["task"] = pid.rsplit("_", 1)[0]
        else:
            merged["task"] = run.get("task", "smoke")
        joined_frames.append(merged)

        # Per-prompt headline metrics
        n = len(merged)
        if n >= 10:
            rho_t1, p_t1 = spearmanr(merged["H_nats"], merged["top1_prob"])
            if "attn_top1_layer_avg" in merged.columns:
                clean = merged.dropna(subset=["attn_top1_layer_avg"])
                if len(clean) >= 10:
                    rho_at, p_at = spearmanr(clean["H_nats"], clean["attn_top1_layer_avg"])
                else:
                    rho_at, p_at = float("nan"), float("nan")
            else:
                rho_at, p_at = float("nan"), float("nan")
        else:
            rho_t1 = p_t1 = rho_at = p_at = float("nan")

        # Thermal: ACTIVE zones only (filter out trip-points)
        zone_cols_all = [c for c in sen.columns if c.endswith("_temp_mc")]
        zone_cols_active = [c for c in zone_cols_all if not is_trip_zone(c)]
        max_active_C = 0.0
        dT_C = 0.0
        if zone_cols_active:
            active = sen[zone_cols_active].apply(pd.to_numeric, errors="coerce")
            max_active_C = float(active.max().max()) / 1000.0
            # warming: max(end-of-run) - max(start-of-run) across zones
            try:
                start_max = float(active.iloc[0].max()) / 1000.0
                end_max   = float(active.iloc[-1].max()) / 1000.0
                dT_C = end_max - start_max
            except Exception:
                dT_C = 0.0

        # Endurance proxy: pswpout delta from /proc/vmstat would be in the sampler
        # output. The smoke-version sampler doesn't yet emit it; defensively read
        # whatever endurance columns are present.
        pswpout_delta = 0
        for col in ("vmstat_pswpout", "vmstat_pgmajfault"):
            if col in sen.columns:
                try:
                    s = pd.to_numeric(sen[col], errors="coerce").dropna()
                    if len(s) >= 2:
                        pswpout_delta = int(s.iloc[-1] - s.iloc[0])
                        break
                except Exception:
                    pass

        summary_rows.append({
            "prompt_id": pid,
            "task": merged["task"].iloc[0],
            "n_steps": n,
            "rho_H_vs_top1": rho_t1,
            "rho_H_vs_attn": rho_at,
            "max_active_temp_C": max_active_C,
            "dT_C": dT_C,
            "pswpout_delta_pages": pswpout_delta,
        })
        attn_str = f"  rho_attn={rho_at:+.3f}" if not (rho_at != rho_at) else "  (no attn.bin)"
        print(f"  {pid:<28} n={n:>4} rho_top1={rho_t1:+.3f}{attn_str}  "
              f"maxT={max_active_C:.1f}C  dT={dT_C:+.1f}C  pswpout_d={pswpout_delta}")

    if not joined_frames:
        print("ERROR: nothing joined.", file=sys.stderr)
        return 1

    full = pd.concat(joined_frames, ignore_index=True)
    out_path = Path(args.out) if args.out else log_dir.parent / f"{log_dir.name}_joined.csv"
    full.to_csv(out_path, index=False)
    print(f"\n[join] full joined CSV: {out_path}  ({len(full)} rows)")

    # Pooled rho
    rho_top1 = rho_attn = p_top1 = p_attn = float("nan")
    if len(full) >= 20:
        rho_top1, p_top1 = spearmanr(full["H_nats"], full["top1_prob"])
        if "attn_top1_layer_avg" in full.columns:
            clean = full.dropna(subset=["attn_top1_layer_avg"])
            if len(clean) >= 20:
                rho_attn, p_attn = spearmanr(clean["H_nats"], clean["attn_top1_layer_avg"])

    print(f"[join] pooled rho(H, top1_prob)            = {rho_top1:+.3f}  "
          f"(n={len(full)}, p={p_top1:.2e})   <-- math-coupled proxy")
    if have_attn:
        print(f"[join] pooled rho(H, attn_top1_layer_avg)  = {rho_attn:+.3f}  "
              f"(n={len(full)}, p={p_attn:.2e})   <-- PAPER metric")

    print("\n[join] gate decision (paper metric is the authoritative one):")
    print(f"  server-side rho on long-form (paper):   -0.37")
    print(f"  fallback threshold (proposal):          rho <= -0.20")
    if have_attn:
        if rho_attn <= -0.20:
            print(f"  on-phone (attention_probe):             rho = {rho_attn:+.3f}  PASS")
        elif rho_attn != rho_attn:  # NaN
            print(f"  on-phone (attention_probe):             rho = NaN   (need more data)")
        else:
            print(f"  on-phone (attention_probe):             rho = {rho_attn:+.3f}  BELOW fallback")
            print(f"  -> fall back to thermal-and-endurance-only controller (proposal Plan B).")
    else:
        print(f"  on-phone (attention_probe):             not run yet.")
        print(f"  on-phone (entropy_probe top1 proxy):    rho = {rho_top1:+.3f}  (not a valid gate; H and top1_prob are math-coupled)")

    print("\n[join] per-task rho (paper metric where available):")
    for task, grp in full.groupby("task"):
        if len(grp) < 10:
            print(f"  {task:<20} n={len(grp):>4}  (too few rows)")
            continue
        r1, _ = spearmanr(grp["H_nats"], grp["top1_prob"])
        if "attn_top1_layer_avg" in grp.columns:
            cl = grp.dropna(subset=["attn_top1_layer_avg"])
            if len(cl) >= 10:
                ra, _ = spearmanr(cl["H_nats"], cl["attn_top1_layer_avg"])
                print(f"  {task:<20} n={len(grp):>4}  rho_top1={r1:+.3f}  rho_attn={ra:+.3f}")
                continue
        print(f"  {task:<20} n={len(grp):>4}  rho_top1={r1:+.3f}  (no attn.bin)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
