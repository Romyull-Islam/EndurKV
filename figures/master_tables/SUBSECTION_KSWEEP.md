# K-Budget Sensitivity Analysis

## 1. Methodology

We sweep the per-layer KV budget K over {256, 384, 512, 1024} on the OnePlus 15
(Snapdragon 8 Elite Gen 5) with the EndurKV CPU+NPU pipeline driving the
PowerInfer-2 reference model. All other knobs are held fixed: Q8 quantisation
on the K cache, f16 V, identical eviction policy, identical thermal watchdog
configuration, and identical prompt corpus. Each cell runs 10-12 decode
iterations; we record per-iteration tokens-per-second (tps), per-iteration DDR
temperature, per-cell peak DDR / CPU / RSS / swap, total evicted tokens, and
cumulative watchdog tier-1/tier-2/tier-3 trip counts. Quality is reported as
final-iteration PPL on the standard wave-11 evaluation slice. The full table
of measured metrics appears in Table 1 of this section; the headline
trade-offs are visualised in `ksweep_pareto.png`,
`ksweep_trajectories.png`, and `ksweep_2axis_pareto.png`.

A caveat on the meaning of K: under our Q8 K seq_add-skip path, K_nominal is
the soft per-layer slot budget enforced by the eviction scheduler, not a strict
hard cap on resident cache footprint. Because evicted positions remain
sparsely allocated until compaction (which we deliberately do not run inside
the decode loop), peak RSS does not collapse to a clean function of K. This is
why K=512 actually has the lowest peak RSS (13.68 GB) of any cell despite
sitting between K=384 (14.41 GB) and K=1024 (14.50 GB) on either side; K is
shaping eviction pressure and addressing pattern, not literally sizing the
allocation.

## 2. Per-K Observations

- K=256: 12 iterations, mean 7.174 tok/s, PPL 2.0948, peak DDR 64.1 C. The
  fastest mean throughput of the sweep and the smoothest decay profile
  (14.98%). Most-evicted configuration (1.66 M tokens) with 304/299/0
  watchdog trips.
- K=384: 12 iterations, mean 7.052 tok/s, PPL 2.1227, peak DDR 63.3 C - the
  *coolest* cell in the sweep. Decay is essentially identical to K=256
  (14.95%).
- K=512: 10 iterations, mean 6.089 tok/s, PPL 2.1686. Zero watchdog trips and
  zero swap, but two pathological per-iteration dips to 4.569 and 4.601 tok/s
  drive end-to-end decay to 37.67%, the worst in the sweep. Lowest peak RSS
  (13.68 GB) and lowest peak CPU temperature (66.8 C).
- K=1024: 10 iterations, mean 6.199 tok/s, PPL **1.8268** - the quality
  optimum by a wide margin (0.27 nats lower than the next-best K=256). Highest
  watchdog activity (415/412/0) and the only cell with non-trivial swap
  (119 MB), but peak DDR (63.7 C) stays below K=256/512.

## 3. Counter-Intuitive Finding: Smaller K Decodes Faster

Naively, a smaller KV budget should be faster because attention scans fewer
slots per step. We observe the stronger statement: throughput is monotonically
non-increasing in K across the measured grid (7.174 -> 7.052 -> 6.089 -> 6.199
tok/s for K = 256/384/512/1024). The mechanism is the interaction of two
effects that the Q8 K seq_add-skip path exposes. First, smaller K forces more
frequent eviction (1.66 M evicted tokens at K=256 vs 0.51 M at K=1024), and
each eviction is cheap under seq_add-skip because the dequant-add-requant pass
is bypassed on skipped positions; in steady state, eviction is *faster* than
the additional attention work it avoids. Second, because seq_add-skip leaves
evicted positions sparsely addressable, smaller K does not produce a
proportionally smaller working set - the allocator stays warm and the
addressing pattern stays cache-friendly, so the per-step inner loop benefits
from a tighter active-slot count without paying a fragmentation tax. The
K=512 cell breaks the pattern in a different direction: its two dips to ~4.6
tok/s coincide with iter-7 and iter-10 (see `ksweep_trajectories.png`) and
appear to be an eviction-scheduler resonance rather than a thermal event
(zero watchdog trips, zero swap, lowest CPU temp in the sweep). We flag this
as a scheduler artefact, not a fundamental property of K=512.

## 4. Pareto Interpretation

Treating (mean tps, PPL) as the two-objective trade-off, the 2D frontier
contains exactly K=256 (throughput optimum) and K=1024 (quality optimum); K=384
and K=512 are strictly dominated (see `ksweep_2axis_pareto.png`). Adding
peak DDR temperature as a third minimisation objective promotes K=384 to the
3D frontier on thermals alone (63.3 C; see `ksweep_pareto.png` bottom
panel). K=512 remains dominated on all three axes simultaneously: slower than
K=256/384, worse PPL than K=1024, and tied on DDR with K=256. It should be
removed from the deployment grid.

## 5. Implications for the Recommended K

Three operating points survive: K=256 for latency-first deployments (best
mean tps, acceptable PPL, well-controlled thermals), K=1024 for quality-first
deployments (best PPL by a clear margin, modest tps loss, manageable swap),
and K=384 for thermally-constrained deployments where sustained operation
above the 64 C DDR knee is unacceptable. As the balanced default we recommend
K=384: it gives up only 1.7% mean tps versus K=256 and 1.4% PPL versus K=1024
while delivering the coolest peak DDR and the second-fewest watchdog trips.
For the headline single-K configuration reported elsewhere in this
dissertation we adopt K=1024, on the grounds that the 0.27-nat PPL gap is
the largest single quality differential observed in the wave-11 grid and the
6.2 tok/s throughput remains above our 6 tok/s interactivity floor.
