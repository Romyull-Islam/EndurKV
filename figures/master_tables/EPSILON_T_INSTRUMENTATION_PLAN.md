# Per-Step ε_t Instrumentation Plan — `eviction_bench.cpp`

**Goal**: Measure the per-step retained-mass deficit
$$\varepsilon_t := 1 - \sum_{i \in S_t} A_{t,i}$$
where $A_t$ is the full (oracle) softmax attention row at decode step $t$ and $S_t$ is the policy's retained index set.

Useful for: (a) MobiSys 2028 empirical evidence section, (b) instantiating Assumption A3 / A4 in any future theory paper, (c) head-by-head allocation gap vs Ada-KV's optimum.

---

## 1. The measurement subtlety

In a WITH-EVICTION decode run, the model sees only the retained cache, so the visible softmax always sums to 1 over $S_t$ by construction. We cannot measure $\varepsilon_t$ from a single with-eviction run.

**Trick**: run the model with the FULL cache (vanilla), capture $A_t^{\text{oracle}}$ at every step, and **simulate** what each candidate eviction policy WOULD HAVE retained at step $t$. Then
$$\varepsilon_t^{(\mathcal A)} = 1 - \sum_{i \in S_t^{(\mathcal A)}} A_{t,i}^{\text{oracle}}$$
This requires **one** vanilla run per (model, prompt, $T$) tuple, and any number of policies can be evaluated post hoc against it.

---

## 2. Algorithm (offline ε_t evaluation)

```
INPUT:  prompt P, model M, max_tokens T, policy A, budget B
OUTPUT: per-step CSV: step, layer, head, eps_t, |S_t|, sum_kept_mass

1. Run vanilla decode of M on P for T tokens.
   For each (decode step t, layer L, head h):
       Capture A_{t,L,h} ∈ R^{prompt_len + t}  via eval_callback (already exists).
       Append to per-step trace file.

2. For each (decode step t, layer L, head h):
       Apply policy A to the captured A_{t,L,h} to choose retained set S_{t,L,h}^{(A)} of size B.
       Compute kept_mass = sum_{i ∈ S} A_{t,L,h,i}
       eps_t = 1 - kept_mass
       Emit row: t, L, h, eps_t, |S|, kept_mass

3. Aggregate:
       Per-layer: mean_{h,t} eps_t
       Per-step: mean_{L,h} eps_t  (this is the curve we plot)
       Cumulative: sum_t mean_{L,h} eps_{t}  (the regret estimate)
```

---

## 3. C++ implementation sketch (incremental patch)

### 3a. Extend `AttnCapture` to record per-step traces

```cpp
// entropy_probe/eviction_bench.cpp (around line 355)
struct AttnCapture {
    std::map<int, std::vector<float>> per_layer;
    int n_head = 0;
    int n_kv = 0;
    bool active = false;
    
    // NEW: per-step trace buffer (one entry per decode step)
    struct StepTrace {
        int step;
        int n_kv;
        // per_layer[L] = flat [n_head * n_kv] softmax row for last query
        std::map<int, std::vector<float>> per_layer_attn;
    };
    std::vector<StepTrace> trace;
    bool trace_enabled = false;
    int current_step = 0;
    
    void reset() { per_layer.clear(); n_kv = 0; trace.clear(); current_step = 0; }
};
```

### 3b. Hook in `eval_callback`

```cpp
bool eval_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * cap = static_cast<AttnCapture *>(user_data);
    if (!cap || !cap->active) return true;
    if (!t || !t->name) return true;
    if (std::strncmp(t->name, "kq_soft_max", 11) != 0) return true;
    if (ask) return true;
    
    const int layer = parse_layer_index(t->name);
    if (layer < 0) return true;
    
    const int64_t ne0 = t->ne[0];   // n_kv
    const int64_t ne1 = t->ne[1];   // n_head
    const int64_t ne2 = t->ne[2];   // n_seq_tokens
    if (ne2 < 1) return true;
    
    const int n_head = (int)ne1;
    const int n_kv   = (int)ne0;
    
    std::vector<float> buf((size_t)n_head * (size_t)n_kv);
    const size_t off = (size_t)(ne2 - 1) * (size_t)n_head * (size_t)n_kv * sizeof(float);
    ggml_backend_tensor_get(t, buf.data(), off, buf.size() * sizeof(float));
    
    // existing prefill capture
    cap->per_layer[layer] = buf;
    cap->n_head = n_head;
    cap->n_kv   = n_kv;
    
    // NEW: per-step trace
    if (cap->trace_enabled) {
        if (cap->trace.empty() || cap->trace.back().step != cap->current_step) {
            cap->trace.push_back({cap->current_step, n_kv, {}});
        }
        cap->trace.back().per_layer_attn[layer] = std::move(buf);
    }
    
    return true;
}
```

### 3c. Bump `current_step` after each `llama_decode` call in the decode loop

```cpp
// In the decode loop (around the existing token-emission point)
// after llama_decode succeeds:
cap.current_step += 1;
```

### 3d. Post-decode simulation routine

```cpp
// New file: entropy_probe/epsilon_logger.cpp
//
// Reads cap.trace AFTER decode finishes; for each step+layer+head,
// simulates the chosen policy, computes eps_t, writes CSV.

#include "policies.h"  // existing policy functions

void log_epsilon_trace(const AttnCapture & cap,
                       const std::string & policy_name,
                       int K_nominal,
                       const std::string & out_csv_path) {
    std::ofstream f(out_csv_path);
    f << "step,layer,head,n_kv,B,kept_mass,eps_t\n";
    
    for (const auto & trace : cap.trace) {
        for (const auto & [layer, attn_flat] : trace.per_layer_attn) {
            int n_head = cap.n_head;
            int n_kv = trace.n_kv;
            
            for (int h = 0; h < n_head; ++h) {
                const float * A_th = attn_flat.data() + h * n_kv;
                
                // Apply policy: pick top-K_nominal positions for this head
                // (or per-head budget for v1 / adakv)
                std::vector<int> S = pick_topk(A_th, n_kv, K_nominal);
                
                float kept = 0.0f;
                for (int p : S) kept += A_th[p];
                
                float eps = 1.0f - kept;
                f << trace.step << "," << layer << "," << h
                  << "," << n_kv << "," << S.size()
                  << "," << kept << "," << eps << "\n";
            }
        }
    }
}
```

---

## 4. Command-line surface

```
eviction_bench \
    --policy vanilla \
    --model PHI3.gguf --prompt PROMPT.txt \
    --max-tokens 2048 \
    --log-epsilon \
    --epsilon-policy adakv \
    --epsilon-budget 512 \
    --epsilon-out epsilon_phi3_adakv_K512.csv
```

A single vanilla decode run produces ε_t traces for ANY simulated policy and ANY budget. To compare AdaKV vs ours at K=512:

```
# One run, two simulations:
eviction_bench --policy vanilla ... --log-epsilon \
    --epsilon-policy adakv --epsilon-budget 512 --epsilon-out adakv_K512.csv

# Same run, different simulation:
eviction_bench --policy vanilla ... --log-epsilon \
    --epsilon-policy v1 --epsilon-budget 512 --epsilon-out v1_K512.csv
```

Or batch both in one call with `--epsilon-policy adakv,v1,h2o,streamingllm`.

---

## 5. What the data buys us

### For the MobiSys 2028 systems paper (§6)
- A new figure: cumulative $\sum_t \varepsilon_t$ vs $t$, comparing policies
- A new figure: per-layer mean $\varepsilon_t$ heatmap, showing which layers are most affected
- A new claim: "Our heuristic μ(max_a) achieves $\bar\varepsilon$ within X% of Ada-KV's allocator's $\bar\varepsilon$ on Phi-3-mini"

### For a future theory paper
- Empirical fit of the power-law exponent $\gamma$ (Assumption A4 in failed Theorem 1)
- Sink-head identification: which heads violate the monotone-decay assumption (A4-POS)
- Per-head allocation-gap measurement: $\sum_t (\varepsilon_t^{\text{ours}} - \varepsilon_t^{\text{Ada-KV}})$, head-by-head

### Cross-cutting
- Reviewer-defensible answer to "does your heuristic actually approximate Ada-KV?" — quantitatively, per head, per step, per layer

---

## 6. Time estimate

| Step | Hours |
|---|---|
| Add `StepTrace` to `AttnCapture` | 1 |
| Hook `eval_callback` for per-step capture | 1 |
| Add `log_epsilon_trace()` routine | 2 |
| Add CLI flags `--log-epsilon`, `--epsilon-policy`, `--epsilon-budget`, `--epsilon-out` | 1 |
| Implement post-decode simulators for vanilla / v1 / adakv / streamingllm / h2o | 4 |
| Push binary + run one Phi-3 trace on OnePlus 15 | 2 |
| Aggregate + plot | 2 |
| **Total** | **~13 hours** focused work |

This is the foundation for both the systems paper's empirical strengthening and any future theory work. No theory claim depends on running this first — it strengthens whatever we end up claiming.

---

## 7. What this does NOT do

- Does not validate Theorem 1's bound (only the empirical $\gamma$)
- Does not produce a verified proof
- Does not change the MobiSys paper's primary claims (still systems + measurement)

It is the cleanest empirical foundation we can build on the available hardware in <2 days of focused work, and it survives any future theoretical pivot.
