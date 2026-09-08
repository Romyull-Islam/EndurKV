"""Contextual bandit vs tabular Q-learning on the fitted phone simulator.

THE OBJECTIVE BALANCES ENERGY AND PERFORMANCE. Minimising energy alone picks a
degenerate arm (crawl slowly forever), so utility carries both terms and the
weight moves with remaining charge:

    u = a(b) * perf_norm + (1 - a(b)) * energy_norm - throttle_penalty
    a(b) = battery fraction

At a full battery the agent is paid for throughput; near empty it is paid for
frugality. The balance point is therefore not a constant -- it is what the agent
has to discover per battery bucket.
"""
import random, math
import sim

CELLS = [342, 688, 1379]   # measured range only; no data above 1379
MHZ   = [883, 1267, 1632]
ACTIONS = [(c, m) for c in CELLS for m in MHZ]

TPS_MAX  = 39.35 * 1.0          # best observed throughput
JTOK_MAX = 0.311 * (4.804/3.970)  # worst-case J/token scale
THROTTLE_PENALTY = 0.25

def weights(battery, charging):
    """How the phone's condition sets the trade. Plugged in, energy is free and the
    agent should buy accuracy and speed. Near empty, energy dominates."""
    if charging:                 return 0.55, 0.45, 0.00   # quality, perf, energy
    if battery >= 0.70:          return 0.50, 0.30, 0.20
    if battery >= 0.40:          return 0.40, 0.25, 0.35
    if battery >= 0.15:          return 0.30, 0.15, 0.55
    return                              0.20, 0.10, 0.70

def utility(res, battery, charging=False):
    wq, wp, we = weights(battery, charging)
    perf = min(1.0, res["tps"] / TPS_MAX)
    ener = max(0.0, 1.0 - res["j_per_tok"] / JTOK_MAX)
    u = wq * res["quality"] + wp * perf + we * ener
    if res["throttled"]:
        u -= THROTTLE_PENALTY
    return u

def b_bucket(b):  return 0 if b < .15 else 1 if b < .40 else 2 if b < .70 else 3
def c_bucket(c):  return 1 if c else 0
def t_bucket(t):  return 0 if t < 45 else 1 if t < 52 else 2 if t < 58 else 3

class Bandit:
    """Memoryless in thermal state: context is battery only."""
    name = "bandit (battery only)"
    def __init__(self, eps=.15, seed=0):
        self.q = {}; self.n = {}; self.eps = eps; self.rng = random.Random(seed)
    def state(self, ph): return (b_bucket(ph.battery), c_bucket(ph.charging))
    def act(self, ph):
        s = self.state(ph)
        if self.rng.random() < self.eps: return self.rng.randrange(len(ACTIONS))
        vals = [self.q.get((s, a), 0.0) for a in range(len(ACTIONS))]
        return max(range(len(ACTIONS)), key=lambda a: vals[a])
    def update(self, s, a, r, s2, done):
        k = (s, a); self.n[k] = self.n.get(k, 0) + 1
        self.q[k] = self.q.get(k, 0.0) + (r - self.q.get(k, 0.0)) / self.n[k]

class QLearn:
    """Carries thermal state, so it can act before the cliff rather than after."""
    name = "Q-learning (battery+temp+throttle)"
    def __init__(self, eps=.15, alpha=.15, gamma=.95, seed=0):
        self.q = {}; self.eps = eps; self.alpha = alpha; self.gamma = gamma
        self.rng = random.Random(seed)
    def state(self, ph):
        return (b_bucket(ph.battery), c_bucket(ph.charging), t_bucket(ph.temp), 1 if ph.throttled else 0)
    def act(self, ph):
        s = self.state(ph)
        if self.rng.random() < self.eps: return self.rng.randrange(len(ACTIONS))
        vals = [self.q.get((s, a), 0.0) for a in range(len(ACTIONS))]
        return max(range(len(ACTIONS)), key=lambda a: vals[a])
    def update(self, s, a, r, s2, done):
        best = 0.0 if done else max(self.q.get((s2, x), 0.0) for x in range(len(ACTIONS)))
        k = (s, a)
        self.q[k] = self.q.get(k, 0.0) + self.alpha * (r + self.gamma * best - self.q.get(k, 0.0))

class Fixed:
    def __init__(self, cells, mhz, label):
        self.a = ACTIONS.index((cells, mhz)); self.name = label
    def state(self, ph): return (0,)
    def act(self, ph): return self.a
    def update(self, *args): pass

def episode(agent, seed, start_batt=.15, max_steps=4000, learn=True, charging=False):
    ph = sim.Phone(battery=start_batt, temp=39.0, seed=seed)
    ph.charging = charging
    tot_u = tot_tok = 0.0; steps = 0; thr = 0
    while ph.battery > 0 and steps < max_steps:
        s = agent.state(ph); a = agent.act(ph)
        cells, mhz = ACTIONS[a]
        res = ph.step(cells, mhz, tokens=256)
        r = utility(res, ph.battery, ph.charging)
        s2 = agent.state(ph); done = ph.battery <= 0
        if learn: agent.update(s, a, r, s2, done)
        tot_u += r; tot_tok += 256; steps += 1; thr += int(res["throttled"])
    return {"utility": tot_u / max(steps, 1), "tokens": tot_tok,
            "steps": steps, "throttle_frac": thr / max(steps, 1),
            "end_temp": ph.temp}
