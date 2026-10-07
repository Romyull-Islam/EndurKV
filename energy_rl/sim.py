"""Simulator of (cache budget, clock cap, thermal state) to (tok/s, J/token, DDR temp).

Fit from a GPU cache-budget run (/tmp/ea_n3), a CPU clock sweep and a CPU soak. The
joint (K, clock) surface was not measured, so the two effects are assumed separable.
"""
import math, random

CACHE = [   # cells, tok/s, W   (GPU, native DVFS, clock not pinned)
            # W is decode-phase USB-rail power, not the whole-run rail+pack energy
            # per token used in the report tables (about 2.5x larger).
    (1379, 32.62, 4.804),
    ( 688, 39.35, 4.751),
    ( 342, 35.07, 4.557),
]
CLOCK = [   # MHz, W, tok/s     (CPU prime core)
    ( 883, 2.289,  7.36),
    (1018, 2.478,  8.31),
    (1267, 2.983, 10.27),
    (1498, 3.528, 11.82),
    (1632, 3.970, 13.44),
]
TAU_S    = 221.0      # thermal time constant, measured
T_AMB    = 30.0       # ambient baseline
T_SS     = 63.6       # measured steady state under muKV load
P_SS     = 5.65       # muKV CPU power at that steady state (595 mWh / 379 s)
R_THERM  = (T_SS - T_AMB) / P_SS          # degC per watt
THROTTLE_T = 62.0     # DDR at which the vendor limiter drops the clock
THROTTLE_MHZ = 883

CLK_REF = 1632.0
_clk = {m: (w, t) for m, w, t in CLOCK}

def _interp(xs, ys, x):
    if x <= xs[0]:  return ys[0]
    if x >= xs[-1]: return ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            f = (x - xs[i]) / (xs[i + 1] - xs[i])
            return ys[i] + f * (ys[i + 1] - ys[i])
    return ys[-1]

def clock_factors(mhz):
    """Normalised (throughput, power) multipliers relative to the 1632 MHz reference."""
    ms = [m for m, _, _ in CLOCK]
    ts = [t for _, _, t in CLOCK]
    ws = [w for _, w, _ in CLOCK]
    t_ref, w_ref = _clk[CLK_REF][1], _clk[CLK_REF][0]
    return _interp(ms, ts, mhz) / t_ref, _interp(ms, ws, mhz) / w_ref

def cache_point(cells):
    """Absolute (tok/s, W) on the measured cache axis, interpolated."""
    cs = [c for c, _, _ in CACHE][::-1]
    ts = [t for _, t, _ in CACHE][::-1]
    ws = [w for _, _, w in CACHE][::-1]
    return _interp(cs, ts, cells), _interp(cs, ws, cells)

class Phone:
    """One decoding session. A step is one request of `tokens` tokens."""
    def __init__(self, battery=1.0, temp=39.0, capacity_wh=18.0, seed=0):
        self.b0 = battery
        self.battery = battery
        self.temp = temp
        self.capacity_j = capacity_wh * 3600.0
        self.rng = random.Random(seed)
        self.throttled = False
        self.charging = False
        self.throttle_steps = 0

    def step(self, cells, mhz, tokens=256):
        # the vendor limiter overrides the requested clock once DDR crosses the cliff
        eff_mhz = THROTTLE_MHZ if self.temp >= THROTTLE_T else mhz
        self.throttled = eff_mhz != mhz
        if self.throttled:
            self.throttle_steps += 1
        tps_c, w_c = cache_point(cells)
        ft, fw = clock_factors(eff_mhz)
        tps = tps_c * ft
        watt = w_c * fw
        dur = tokens / max(tps, 1e-6)
        energy_j = watt * dur
        # first-order thermal response toward the power-driven steady state
        t_target = T_AMB + R_THERM * watt
        self.temp += (t_target - self.temp) * (1.0 - math.exp(-dur / TAU_S))
        self.battery = max(0.0, self.battery - energy_j / self.capacity_j)
        return {"tps": tps, "watt": watt, "dur_s": dur, "energy_j": energy_j,
                "quality": quality(cells),
                "j_per_tok": energy_j / tokens, "temp": self.temp,
                "throttled": self.throttled, "eff_mhz": eff_mhz}


# Quality vs retention r (%): q = 1 - 0.884 * exp(-r / 3.03), fitted on Bonsai
# hotpotqa F1 at 6.8%, 13.6% and 26.7% retention. Nearly flat past ~14%.
Q_A, Q_B = 0.884, 3.03
PEAK_CELLS = 12500.0     # ea_n3 peak KV cells at the 9737-token prompt

def quality(cells):
    r = cells / PEAK_CELLS * 100.0
    return min(1.0, 1.0 - Q_A * math.exp(-r / Q_B))
