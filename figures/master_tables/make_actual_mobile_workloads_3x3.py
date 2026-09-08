#!/usr/bin/env python3
"""Actual mobile LLM workload signatures from existing benchmark traces.

Rows:
  1. Performance over time: prefill effective throughput and decode throughput.
  2. USB-rail power over time.
  3. KV-cache resident memory over time.
  4. SoC/DDR thermal response over time.

The selected runs are existing local traces with meta.json, sensors.csv, and
steps.csv. This script does not touch the phone or run inference.
"""

import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = "/home/mislam22/EndurKV_workspace/EndurKV/figures/master_tables/figures"

COLS = [
    {
        "key": "A",
        "color": "#2a78d6",
        "title": "(A) Long-document QA",
        "sub": "HotpotQA: 9.6K prompt + 128-token answer",
        "path": "/home/mislam22/EndurKV_workspace/phone-logs/sweep3M_1780245255/Phi-3-128k/vanilla/K1024/hotpotqa_pub_001/rep2/rep2",
    },
    {
        "key": "B",
        "color": "#1baf7a",
        "title": "(B) Medium-document QA",
        "sub": "Qasper: 3.4K prompt + 128-token answer",
        "path": "/home/mislam22/EndurKV_workspace/phone-logs/sweep3M_1780245255/Phi-3-128k/vanilla/K1024/qasper_pub_001/rep1/rep1",
    },
    {
        "key": "C",
        "color": "#e34948",
        "title": "(C) Sustained generation",
        "sub": "7.5K prompt + 2048-token output",
        "path": "/tmp/phone_3x3_v2/phone_3x3_v2_20260625_142715/phi3_vanilla",
    },
]

GRAY = "#8a8987"


def _array(rows, name, scale=1.0):
    vals = []
    for row in rows:
        try:
            vals.append(float(row.get(name, "")) / scale)
        except Exception:
            vals.append(np.nan)
    return np.array(vals, dtype=float)


def _smooth(y, w):
    y = np.array(y, dtype=float)
    if len(y) < w or w < 2:
        return y
    return np.convolve(y, np.ones(w) / w, mode="same")


def load_run(path):
    with open(os.path.join(path, "meta.json")) as f:
        meta = json.load(f)

    with open(os.path.join(path, "sensors.csv")) as f:
        sensor_rows = list(csv.DictReader(f))

    t = _array(sensor_rows, "monotonic_s")
    t = t - t[0]
    ddr = _array(sensor_rows, "ddr_temp_mc", 1000.0)
    usb_i = _array(sensor_rows, "usb_current_ua", 1e6)
    usb_v = _array(sensor_rows, "usb_voltage_uv", 1e6)
    usb_power = np.abs(usb_i) * usb_v
    if np.isfinite(usb_power).sum() > 0:
        power = usb_power
        power_label = "USB rail"
    else:
        # Older LongBench traces did not log USB rail power. Their battery
        # current column is too coarse here, often quantized to 0 mA, so do
        # not plot it as power.
        power = np.full_like(t, np.nan)
        power_label = "not logged"

    temp_cols = list(sensor_rows[0].keys())
    cpu_cols = [
        c
        for c in temp_cols
        if c.endswith("_temp_mc")
        and "trip" not in c
        and (c.startswith("cpu-") or c.startswith("cpullc"))
    ]
    cpu = np.nanmax(np.vstack([_array(sensor_rows, c, 1000.0) for c in cpu_cols]), axis=0)

    steps = []
    with open(os.path.join(path, "steps.csv")) as f:
        for row in csv.DictReader(f):
            try:
                steps.append(
                    {
                        "step": int(row["step"]),
                        "wall_s": float(row["wall_us"]) / 1e6,
                        "kv": int(row["n_kv_cells"]),
                    }
                )
            except Exception:
                pass

    mb_per_cell = meta["peak_kv_mb"] / meta["peak_kv_cells"]
    prefill_s = meta["prefill_ms"] / 1000.0
    decode_s = meta["decode_ms"] / 1000.0
    total_s = meta["total_ms"] / 1000.0

    # Some traces log step wall time relative to request start. Normalize decode
    # steps so the first generated token begins at the measured prefill boundary.
    step_t = np.array([s["wall_s"] for s in steps], dtype=float)
    if len(step_t):
        step_t = prefill_s + (step_t - step_t[0])
    step_kv = np.array([s["kv"] for s in steps], dtype=float)

    decode_rate_t = np.array([])
    decode_rate = np.array([])
    if len(step_t) > 2:
        dt = np.diff(step_t)
        good = dt > 0
        decode_rate_t = step_t[1:][good]
        decode_rate = 1.0 / dt[good]
        decode_rate = _smooth(decode_rate, min(25, max(3, len(decode_rate) // 8)))

    return {
        "meta": meta,
        "t": t,
        "ddr": ddr,
        "cpu": cpu,
        "power": power,
        "power_label": power_label,
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "total_s": total_s,
        "step_t": step_t,
        "step_kv": step_kv,
        "decode_rate_t": decode_rate_t,
        "decode_rate": decode_rate,
        "mb_per_cell": mb_per_cell,
    }


def kv_trace(run):
    meta = run["meta"]
    prefill_s = run["prefill_s"]
    mb_per_cell = run["mb_per_cell"]
    prompt = meta["n_prompt_tokens"]

    tp = np.linspace(0, prefill_s, 80)
    kvp = np.linspace(0, prompt * mb_per_cell, 80)

    if len(run["step_t"]):
        td = run["step_t"]
        kvd = run["step_kv"] * mb_per_cell
    else:
        td = np.array([prefill_s, run["total_s"]])
        kvd = np.array([prompt, prompt]) * mb_per_cell

    return np.concatenate([tp, td]), np.concatenate([kvp, kvd])


def main():
    data = [load_run(c["path"]) for c in COLS]

    fig, axes = plt.subplots(4, 3, figsize=(13.2, 11.2))
    plt.subplots_adjust(left=0.075, right=0.985, top=0.925, bottom=0.065, hspace=0.34, wspace=0.23)

    perf_max = 1.15 * max(
        max(d["meta"]["n_prompt_tokens"] / d["prefill_s"], np.nanmax(d["decode_rate"]) if len(d["decode_rate"]) else 0)
        for d in data
    )
    kv_max = 1.10 * max(kv_trace(d)[1].max() for d in data)
    power_peaks = []
    for d in data:
        if np.isfinite(d["power"]).sum() > 0:
            power_peaks.append(np.nanmax(_smooth(d["power"], min(151, max(3, len(d["power"]) // 25)))))
    power_max = 1.08 * max(power_peaks) if power_peaks else 1.0
    temp_min = 30
    temp_max = 82

    for j, (col, run) in enumerate(zip(COLS, data)):
        color = col["color"]
        meta = run["meta"]
        prefill_rate = meta["n_prompt_tokens"] / run["prefill_s"]
        total_min = run["total_s"] / 60.0

        # Row 1: performance.
        ax = axes[0][j]
        ax.hlines(prefill_rate, 0, run["prefill_s"], color=color, lw=2.0)
        ax.fill_between([0, run["prefill_s"]], [0, 0], [prefill_rate, prefill_rate], color=color, alpha=0.10)
        if len(run["decode_rate_t"]):
            ax.plot(run["decode_rate_t"], run["decode_rate"], color=color, lw=1.5)
            ax.fill_between(run["decode_rate_t"], 0, run["decode_rate"], color=color, alpha=0.10)
        ax.axvline(run["prefill_s"], color=GRAY, ls=(0, (4, 3)), lw=1.1)
        ax.set_xlim(0, run["total_s"] * 1.02)
        ax.set_ylim(0, perf_max)
        ax.set_title(col["title"], fontsize=11.2, color=color, weight="bold", pad=16)
        ax.text(0.5, 1.02, col["sub"], transform=ax.transAxes, ha="center", va="bottom", fontsize=8.1, color="#444")
        if j == 0:
            ax.set_ylabel("Throughput\n(tokens/s)", fontsize=10.5)
        ax.text(
            0.96,
            0.83,
            f"prefill {prefill_rate:.1f} tok/s\n"
            f"decode {meta['decode_tps']:.1f} tok/s\n"
            f"total {total_min:.1f} min",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.0,
            color=color,
            weight="bold",
        )
        ax.text(run["prefill_s"], perf_max * 0.94, " decode", fontsize=7.5, color=GRAY, va="top")
        ax.text(run["prefill_s"], perf_max * 0.94, "prefill ", fontsize=7.5, color=GRAY, va="top", ha="right")

        # Row 2: USB power.
        ax = axes[1][j]
        ps = _smooth(run["power"], min(151, max(3, len(run["power"]) // 25)))
        ax.axvline(run["prefill_s"], color=GRAY, ls=(0, (4, 3)), lw=1.1)
        ax.set_xlim(0, run["total_s"] * 1.02)
        ax.set_ylim(0, power_max)
        if j == 0:
            ax.set_ylabel("Power\n(W)", fontsize=10.5)
        if np.isfinite(ps).sum() > 0:
            ax.plot(run["t"], ps, color=color, lw=1.5)
            ax.fill_between(run["t"], 0, ps, color=color, alpha=0.10)
            power_text = f"{run['power_label']}\nmedian {np.nanmedian(run['power']):.1f} W\npeak {np.nanmax(ps):.1f} W"
        else:
            ax.text(0.5, 0.52, "power not logged", transform=ax.transAxes, ha="center", va="center", fontsize=9.0, color=GRAY, weight="bold")
            power_text = "not logged"
        ax.text(
            0.96,
            0.10,
            power_text,
            transform=ax.transAxes,
            ha="right",
            fontsize=8.0,
            color=color if np.isfinite(ps).sum() > 0 else GRAY,
            weight="bold",
        )

        # Row 3: KV memory.
        ax = axes[2][j]
        tk, kv = kv_trace(run)
        ax.plot(tk, kv, color=color, lw=1.8)
        ax.fill_between(tk, 0, kv, color=color, alpha=0.12)
        ax.axvline(run["prefill_s"], color=GRAY, ls=(0, (4, 3)), lw=1.1)
        ax.set_xlim(0, run["total_s"] * 1.02)
        ax.set_ylim(0, kv_max)
        if j == 0:
            ax.set_ylabel("KV-cache resident\n(MiB, f16)", fontsize=10.5)
        ax.text(0.96, 0.10, f"peak {kv.max():.0f} MiB", transform=ax.transAxes, ha="right", fontsize=8.2, color=color, weight="bold")

        # Row 4: thermals.
        ax = axes[3][j]
        ax.plot(run["t"], run["cpu"], color=color, lw=1.25, label="CPU live max")
        ax.plot(run["t"], run["ddr"], color="#444444", lw=1.25, ls=(0, (3, 2)), label="DDR")
        ax.axvline(run["prefill_s"], color=GRAY, ls=(0, (4, 3)), lw=1.1)
        ax.set_xlim(0, run["total_s"] * 1.02)
        ax.set_ylim(temp_min, temp_max)
        if j == 0:
            ax.set_ylabel("Temperature\n(deg C)", fontsize=10.5)
        ax.set_xlabel("time (s)", fontsize=9.5)
        ax.text(
            0.96,
            0.10,
            f"CPU peak {np.nanmax(run['cpu']):.0f} C\nDDR peak {np.nanmax(run['ddr']):.0f} C",
            transform=ax.transAxes,
            ha="right",
            fontsize=8.0,
            color=color,
            weight="bold",
        )

    for row in axes:
        for ax in row:
            ax.grid(alpha=0.18, lw=0.6)
            ax.tick_params(labelsize=8.5)

    axes[3][2].legend(loc="lower right", fontsize=7.8, framealpha=0.92)

    fig.suptitle(
        "Actual mobile LLM workload signatures on OnePlus 15 (Phi-3-mini, full-cache)",
        fontsize=13,
        weight="bold",
        y=0.975,
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    out_png = os.path.join(OUT_DIR, "fig_actual_mobile_workloads_4x3_same_workloads.png")
    out_pdf = out_png.replace(".png", ".pdf")
    plt.savefig(out_png, dpi=145, bbox_inches="tight", facecolor="white")
    plt.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    print(out_png)
    for col, run in zip(COLS, data):
        meta = run["meta"]
        print(
            f"{col['key']}: prompt={meta['n_prompt_tokens']} decode={meta['n_decode_steps']} "
            f"total={run['total_s']:.1f}s prefill_tps={meta['n_prompt_tokens']/run['prefill_s']:.2f} "
            f"decode_tps={meta['decode_tps']:.2f} peak_kv={meta['peak_kv_mb']:.1f} "
            f"power={run['power_label']} med={np.nanmedian(run['power']) if np.isfinite(run['power']).sum() else float('nan'):.2f} "
            f"cpu_peak={np.nanmax(run['cpu']):.1f} ddr_peak={np.nanmax(run['ddr']):.1f}"
        )


if __name__ == "__main__":
    main()
