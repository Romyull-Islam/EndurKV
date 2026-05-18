#!/usr/bin/env python3
"""
host_make_summary_slide.py — produce a single 1920x1080 PNG summarizing
today's data-collection session on the OnePlus 15.

Output: logs/today_summary_slide.png (or wherever WORKSPACE points).

Layout (slide-style):
   +---------------------------------------------+
   |  Title bar                                  |
   +-----------------------+---------------------+
   |  STEPS WE DID         |  DATA WE GOT        |
   |  (left half, numbered)|  (right half, table)|
   +-----------------------+---------------------+
   |  RESULT bar at bottom (headline rho + gate) |
   +---------------------------------------------+
"""
import os
import sys
from pathlib import Path

WORKSPACE = Path(os.environ.get("WORKSPACE", r"D:/Research/EndurKV_workspace"))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Rectangle
    plt.rcParams.update({"font.family": "DejaVu Sans"})

    fig = plt.figure(figsize=(16, 9), dpi=120)
    fig.patch.set_facecolor("#FAFAFA")

    # ---- Title bar ------------------------------------------------------------
    ax_title = fig.add_axes([0, 0.91, 1, 0.09])
    ax_title.set_xlim(0, 1); ax_title.set_ylim(0, 1); ax_title.axis("off")
    ax_title.add_patch(Rectangle((0, 0), 1, 1, color="#0d2a52"))
    ax_title.text(0.02, 0.62,
        "EndurKV — On-Phone Measurement Gate on OnePlus 15",
        fontsize=22, color="white", weight="bold", va="center")
    ax_title.text(0.02, 0.22,
        "Snapdragon 8 Elite Gen 5  ·  Android 16  ·  attention_probe on Llama-3.2-1B Q4_K_M",
        fontsize=12, color="#cfd8e6", va="center")
    ax_title.text(0.98, 0.62, "Session: 2026-05-18",
        fontsize=11, color="#cfd8e6", va="center", ha="right")

    # ---- LEFT: numbered steps -------------------------------------------------
    ax_steps = fig.add_axes([0.02, 0.20, 0.46, 0.69])
    ax_steps.set_xlim(0, 1); ax_steps.set_ylim(0, 1); ax_steps.axis("off")
    ax_steps.add_patch(FancyBboxPatch((0, 0), 1, 1,
        boxstyle="round,pad=0.005,rounding_size=0.015",
        fc="white", ec="#c0c0c0", lw=1.2))
    ax_steps.text(0.04, 0.96, "STEPS WE DID TODAY",
        fontsize=13, color="#0d2a52", weight="bold", va="top")

    steps = [
        ("1.", "Cloned EndurKV from GitHub; consolidated workspace at d:/Research/EndurKV_workspace/."),
        ("2.", "Installed Android NDK r27c + CMake + Ninja + adb via winget."),
        ("3.", "Patched entropy_probe CMakeLists for Android (armv8.7-a, $ORIGIN rpath, sysroot escape)."),
        ("4.", "Cross-compiled libllama.so + libggml*.so + 3 probes for aarch64-Android."),
        ("5.", "Wrote on-phone scripts: discover_sensors, sample_sensors (10/5 Hz), run_one_prompt."),
        ("6.", "Wrote host scripts: phone_check, host_run_study, host_join_and_rho, host_make_plots."),
        ("7.", "Connected OnePlus 15 (CPH2749) via adb. Verified Android 16, arm64-v8a, no root."),
        ("8.", "Mapped 98 thermal zones; non-root I/O via /proc/vmstat (pswpout)."),
        ("9.", "Downloaded 1B Q4_K_M (771 MB, sha256 verified) and 8B Q4_K_M (4.6 GB)."),
        ("10.", "Generated 80 prompts via script 09 (LongBench 8 + HELM 2 + lm-eval 2)."),
        ("11.", "Smoke-tested attention_probe; on-phone smoke rho = -0.42."),
        ("12.", "Ran full 1B study: 80/80 prompts OK, 4025 decode steps, 16 min wall-clock."),
        ("13.", "Ran joiner + plotter: 7 figures + per-task rho breakdown."),
        ("14.", "Upgraded sampler to v3 (GPU busy, CPU freq, cooling state, battery)."),
        ("15.", "Launched 8B study (80 prompts, 5 Hz sampler) — in flight, ETA ~10:30."),
    ]
    y = 0.91
    for tag, body in steps:
        ax_steps.text(0.06, y, tag, fontsize=10, weight="bold", color="#0d2a52", va="top")
        ax_steps.text(0.10, y, body, fontsize=10, color="#222", va="top", wrap=True)
        y -= 0.060

    # ---- RIGHT: data table ---------------------------------------------------
    ax_data = fig.add_axes([0.52, 0.20, 0.46, 0.69])
    ax_data.set_xlim(0, 1); ax_data.set_ylim(0, 1); ax_data.axis("off")
    ax_data.add_patch(FancyBboxPatch((0, 0), 1, 1,
        boxstyle="round,pad=0.005,rounding_size=0.015",
        fc="white", ec="#c0c0c0", lw=1.2))
    ax_data.text(0.04, 0.96, "DATA WE COLLECTED  (1B run, 80 prompts)",
        fontsize=13, color="#0d2a52", weight="bold", va="top")

    rows = [
        ("Per-step probe metrics", "H_nats, top1_prob, top5_cumprob, wall_clock_us"),
        ("  steps logged", "4 025 across 12 tasks"),
        ("Per-step attention sidecar (.attn.bin)", "16 layers x n_kv x f32; 2.1 MB/prompt"),
        ("Sensor sampler frequency", "10 Hz (1B run) / 5 Hz (8B run)"),
        ("", ""),
        ("THERMAL", ""),
        ("  thermal zones logged", "98 per sample"),
        ("  CPU peak temperature", "70.9 deg C  (triviaqa_002)"),
        ("  DDR / RAM controller", "min 34.3  mean 47.8  max 56.3  deg C"),
        ("  Flash / UFS controller", "min 32.5  mean 39.6  max 41.9  deg C"),
        ("  Skin (phone case)", "~33-35 deg C"),
        ("  prompts hitting 70 deg C trip", "lcc, trec, triviaqa"),
        ("", ""),
        ("ENDURANCE  (non-root proxies)", ""),
        ("  pswpout (KB swapped to UFS)", "median 0,  max 34 MB (hotpotqa_003)"),
        ("  pgmajfault (UFS-reads count)", "captured per-sample"),
        ("  probe's own write_bytes", "captured per-sample"),
        ("  raw /sys/block/*/stat", "PERM-DENIED  (needs root)"),
        ("", ""),
        ("MEMORY", ""),
        ("  MemTotal / MemAvailable", "15.5 GB / 8.9 GB available"),
        ("  decode working-set drop", "~1.0 GB during 1B inference"),
        ("", ""),
        ("LATENCY", ""),
        ("  median decode latency", "27-28 ms/token"),
        ("  thermal drift over run", "27 -> 35-50 ms/token by step 3500+"),
        ("", ""),
        ("TOTAL DATA on disk", "132 MB  (80 entropy.csv + 80 sensors.csv + 80 attn.bin + 80 run.json)"),
    ]
    y = 0.90
    for k, v in rows:
        if not k and not v:
            y -= 0.012; continue
        if not v:
            ax_data.text(0.04, y, k, fontsize=10, color="#0d2a52", weight="bold", va="top")
        else:
            ax_data.text(0.05, y, k, fontsize=9.5, color="#222", va="top")
            ax_data.text(0.46, y, v, fontsize=9.5, color="#222", va="top", family="monospace")
        y -= 0.027

    # ---- BOTTOM: headline result --------------------------------------------
    ax_res = fig.add_axes([0.02, 0.03, 0.96, 0.15])
    ax_res.set_xlim(0, 1); ax_res.set_ylim(0, 1); ax_res.axis("off")
    ax_res.add_patch(FancyBboxPatch((0, 0), 1, 1,
        boxstyle="round,pad=0.005,rounding_size=0.015",
        fc="#e7f4ea", ec="#1b6e1b", lw=1.5))
    ax_res.text(0.02, 0.78, "HEADLINE  -  Month 1 measurement gate: PASS",
        fontsize=15, color="#1b6e1b", weight="bold", va="top")
    ax_res.text(0.02, 0.50,
        "Long-form generation (paper's intended gate regime):    on-phone rho = -0.404,  n = 2161,  p ~ 10^-85",
        fontsize=11.5, color="#0d2a52", va="top", family="monospace")
    ax_res.text(0.02, 0.30,
        "Paper server-side baseline:                              rho = -0.37  (8B long-ctx)  /  -0.43  (1B gov_report)",
        fontsize=11.5, color="#0d2a52", va="top", family="monospace")
    ax_res.text(0.02, 0.10,
        "Proposal fallback threshold:  rho <= -0.20    -->    on-phone passes by 2x margin.  Per-task pattern matches slide 23.",
        fontsize=11.5, color="#0d2a52", va="top", family="monospace")

    out = WORKSPACE / "logs" / "today_summary_slide.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main() or 0)
