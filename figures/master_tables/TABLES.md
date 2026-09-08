# Master tables — everything in two views

Sources:
- **CPU sweep** (cpu_sweep_1780268970): 27 cells, ngl=0, valid quality + CPU/DDR thermals
- **GPU sweep** (sweep3M_1780245255):  52 cells, GPU offload, thermal data only (output text degenerate due to upstream Vulkan bug)
- **Wave-2 PPL** (ppl_1780287362):     9 cells, intrinsic LM perplexity on WT2

## Table 1 — Quality, latency, memory, KV behaviour

| Model | Policy | n | **F1** | **WT2 PPL** | Decode t/s | Prefill ms | Peak KV (MB) | Peak RSS (MB) | Mass retained | Retention ratio | Efficiency | Evicted total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Llama-1B | vanilla | 3 | 0.115 | 8.7802 | 11.31 | 176970.0 | 131.4 | 1147.0 | 1.0 | 1.0 | 1.0 | 0 |
| Llama-1B | v1 | 3 | 0.114 | 8.7862 | 6.97 | 215157.0 | 130.0 | 1208.0 | 0.9914 | 0.4741 | 3.205 | 25765.0 |
| Llama-1B | tova | 3 | 0.0238 | 8.7862 | 6.93 | 217492.0 | 129.8 | 1208.0 | 0.9892 | 0.3751 | 4.092 | 36984.0 |
| Gemma-2-2B | vanilla | 3 | 0.1407 | 8.8546 | 5.2 | 437877.0 | 517.7 | 2539.0 | 1.0 | 1.0 | 1.0 | 0 |
| Gemma-2-2B | v1 | 3 | 0.1064 | 8.8177 | 3.21 | 495500.0 | 511.3 | 2582.0 | 0.9931 | 0.4349 | 3.46 | 66825 |
| Gemma-2-2B | tova | 3 | 0.0312 | 8.8177 | 3.2 | 498524.0 | 510.6 | 2582.0 | 0.99 | 0.3398 | 4.434 | 98563.0 |
| Phi-3-128k | vanilla | 3 | 0.2111 | 5.0299 | 2.98 | 848538.0 | 1872.8 | 6146.0 | 1.0 | 1.0 | 1.0 | 0 |
| Phi-3-128k | v1 | 3 | 0.2054 | 5.0534 | 2.32 | 972544.0 | 1860.1 | 6265.0 | 0.9799 | 0.3174 | 4.409 | 27037.0 |
| Phi-3-128k | tova | 3 | 0.2059 | 5.0534 | 2.26 | 1055507.0 | 1860.1 | 6286.0 | 0.9747 | 0.2972 | 4.824 | 32360.0 |

## Table 2 — Thermal peaks and rises, per zone × backend

CPU-mode columns are from the CPU sweep (our valid workload). GPU-mode columns are from the GPU sweep (thermals are valid even though outputs were degenerate).

| Model | Policy | CPU peak (CPU mode) | CPU rise | DDR peak (CPU mode) | DDR rise | GPU peak (GPU mode) | GPU rise | DDR peak (GPU mode) | DDR rise | NPU peak (GPU mode) |
|---|---|---|---|---|---|---|---|---|---|---|
| Llama-1B | vanilla | 56.6 °C | 9.8 °C | 48.9 °C | 6.6 °C | 59.9 °C | 16.9 °C | 56.8 °C | 13.7 °C | 48.8 °C |
| Llama-1B | v1 | 67.2 °C | 21.0 °C | 51.5 °C | 9.8 °C | 55.8 °C | 13.4 °C | 54.5 °C | 12.0 °C | 48.4 °C |
| Llama-1B | tova | 54.8 °C | 8.1 °C | 47.6 °C | 5.6 °C | 56.2 °C | 13.2 °C | 54.8 °C | 11.8 °C | 48.9 °C |
| Gemma-2-2B | vanilla | 70.1 °C | 21.7 °C | 53.0 °C | 11.0 °C | 58.9 °C | 15.3 °C | 58.5 °C | 14.4 °C | 50.2 °C |
| Gemma-2-2B | v1 | 64.5 °C | 16.6 °C | 49.8 °C | 7.4 °C | 52.8 °C | 10.1 °C | 52.3 °C | 9.4 °C | 47.3 °C |
| Gemma-2-2B | tova | 75.7 °C | 27.1 °C | 53.5 °C | 11.1 °C | 49.1 °C | 6.8 °C | 48.6 °C | 6.5 °C | 44.4 °C |
| Phi-3-128k | vanilla | 68.7 °C | 21.8 °C | 51.8 °C | 9.4 °C | 57.0 °C | 13.9 °C | 55.2 °C | 12.3 °C | 47.7 °C |
| Phi-3-128k | v1 | 65.7 °C | 17.7 °C | 54.2 °C | 11.1 °C | 54.8 °C | 11.5 °C | 53.5 °C | 10.1 °C | 47.3 °C |
| Phi-3-128k | tova | 57.2 °C | 11.9 °C | 50.3 °C | 8.1 °C | 54.6 °C | 11.4 °C | 52.9 °C | 9.6 °C | 47.0 °C |
