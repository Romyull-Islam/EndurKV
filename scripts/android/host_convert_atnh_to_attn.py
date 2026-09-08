"""Convert v2 ATNH (per-head) .attn.bin files into v1 ATTN (head-averaged) format.

Mathematically lossless reduction:
    ATTN[step, layer, pos] = mean over heads of ATNH[step, layer, head, pos]

Used to derive layer-avg simulation cells from per-head captures when the
corresponding v1 capture doesn't exist (e.g., 8B long-ctx was only captured
with the v2 probe). Output files are byte-compatible with the original v1
attention_probe so the layer-avg simulator consumes them unchanged.

Sidecar files (.run.json, .sensors.csv, .entropy.csv) are hard-linked into
the output dir so the layer-avg simulator sees a complete capture dir.

Usage:
    python host_convert_atnh_to_attn.py --src logs/_combined_8b_longctx_perhead_52 \
                                        --dst logs/_derived_8b_longctx_layeravg
"""
from __future__ import annotations
import argparse
import os
import struct
import sys
from pathlib import Path

import numpy as np


def convert_one(src_path: Path, dst_path: Path) -> tuple[int, int, int]:
    """Read one ATNH file, write the equivalent ATTN file. Returns
    (n_steps, n_layers, n_head) for sanity."""
    with open(src_path, "rb") as f:
        magic = f.read(4)
        if magic != b"ATNH":
            raise ValueError(f"not ATNH: {src_path} (magic={magic!r})")
        n_steps, n_layers, n_head = struct.unpack("<III", f.read(12))

        # We can't know n_kv per (step, layer) until we read the per-block
        # n_kv prefix. Stream: read each block, write the averaged block.
        with open(dst_path, "wb") as g:
            g.write(b"ATTN")
            # v1 layout (matches attention_probe.cpp + load_attn_full):
            # magic(4) + n_steps(u32) + n_layers(u32) + n_head(u32) + per-step
            # per-layer (n_kv(u32) + n_kv floats). v1 stores n_head=1.
            g.write(struct.pack("<III", n_steps, n_layers, 1))
            for _s in range(n_steps):
                for _l in range(n_layers):
                    (n_kv,) = struct.unpack("<I", f.read(4))
                    if n_kv == 0:
                        g.write(struct.pack("<I", 0))
                        continue
                    count = n_kv * n_head
                    vals = np.frombuffer(f.read(4 * count), dtype=np.float32)
                    per_head = vals.reshape(n_head, n_kv)
                    avg = per_head.mean(axis=0).astype(np.float32)
                    g.write(struct.pack("<I", n_kv))
                    g.write(avg.tobytes())
    return n_steps, n_layers, n_head


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dir with ATNH .attn.bin files")
    ap.add_argument("--dst", required=True, help="output dir for ATTN .attn.bin")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    attn_files = sorted(src.glob("*.attn.bin"))
    if not attn_files:
        print(f"ERROR: no .attn.bin in {src}", file=sys.stderr)
        return 1

    print(f"[convert] {len(attn_files)} files: {src} -> {dst}")
    for i, src_f in enumerate(attn_files, 1):
        dst_f = dst / src_f.name
        try:
            ns, nl, nh = convert_one(src_f, dst_f)
            print(f"  [{i:3d}/{len(attn_files)}] {src_f.name:<40} steps={ns} "
                  f"layers={nl} heads={nh}  -> {dst_f.stat().st_size/1e6:.1f} MB")
        except Exception as e:
            print(f"  [{i:3d}/{len(attn_files)}] {src_f.name}: FAILED ({e})",
                  file=sys.stderr)
            continue
        # Hard-link sidecars so the layer-avg sim has full context.
        stem = src_f.name[:-len(".attn.bin")]
        for ext in (".run.json", ".sensors.csv", ".entropy.csv", ".probe.stderr"):
            sc_src = src / f"{stem}{ext}"
            sc_dst = dst / f"{stem}{ext}"
            if sc_src.exists() and not sc_dst.exists():
                try:
                    os.link(sc_src, sc_dst)
                except OSError:
                    # Cross-device or perm issue — copy instead
                    sc_dst.write_bytes(sc_src.read_bytes())
    print(f"[convert] done. Output: {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
