"""
Simulator of (cache budget, clock cap, thermal state) -> (tok/s, J/token, DDR).

FITTED FROM MEASURED DATA:
  cache axis : /tmp/ea_n3        muKV at 3 budgets, GPU, clock held 902 MHz
  clock axis : /tmp/phone_freq_sweep   CPU prime core, 5 frequencies, 2 repeats
  thermal    : /tmp/cpu_soak/mukv_wdoff  44 min soak, tau = 221 s, rise 39.0 -> 63.6 C

KNOWN LIMITATION, stated because it bounds every conclusion drawn from this file:
every campaign varied ONE actuator and pinned the other. The cache axis is GPU at a
held clock; the clock axis is CPU at one policy. The JOINT (K, clock) surface has
never been measured, so this model assumes the two are SEPARABLE and multiplies
their normalised effects. The interesting operating point -- energy-optimal cache
with the clock absorbing its heat -- sits exactly in that unmeasured region.
A 3x3 grid on the phone would replace this assumption with data.
"""
import math, random

CACHE = [   # cells, tok/s, W   (GPU, native DVFS: /tmp/ea_n3 did NOT pin the clock; corrected 2026-09-02)
            # W here is decode-phase USB-rail power; the report's tables use whole-run rail+pack
            # energy per decoded token (about 2.5x larger). Do not compare the two directly.
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


# ---- quality axis, fitted on the measured budget curve -------------------------
# q(retention) = 1 - 0.884 * exp(-r / 3.03), fitted on Bonsai hotpotqa:
#   6.8% kept -> 42.71 F1 (q=0.906), 13.6% -> 46.69 (0.990), 26.7% -> 47.17 (1.000)
# Quality saturates fast: past ~14% retention there is almost nothing left to buy.
Q_A, Q_B = 0.884, 3.03
PEAK_CELLS = 12500.0     # ea_n3 peak KV cells at the 9737-token prompt

def quality(cells):
    r = cells / PEAK_CELLS * 100.0
    return min(1.0, 1.0 - Q_A * math.exp(-r / Q_B))
