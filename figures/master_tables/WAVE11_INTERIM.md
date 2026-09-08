# Wave-11 INTERIM status

Snapshot at: **2026-06-07 18:22:36 EDT**  
Run dir: `/home/mislam22/EndurKV_workspace/phone-logs/wave11_eval_1780862534`  
Progress: **0 / 10 cells complete**, 1 partial, 9 pending.  
ETA to finish (flat per-cell estimate): **7.8h**

Rule-based NIAH judge: case-insensitive substring `"sandwich at dolores park"`.

| model | policy | bench | cells_complete | cells_pending | chunks_done/expected | mean_ppl | niah_acc | status |
|---|---|---|---:|---:|---:|---:|---:|---|
| Phi-3-mini-128k | h2o | niah | 0 | 1 | 0/8 | - | - | pending |
| Phi-3-mini-128k | h2o | ppl | 0 | 1 | 0/8 | - | - | pending |
| Phi-3-mini-128k | tova | niah | 0 | 1 | 0/8 | - | - | pending |
| Phi-3-mini-128k | tova | ppl | 0 | 1 | 0/8 | - | - | pending |
| Phi-3-mini-128k | v1 | niah | 0 | 1 | 0/8 | - | - | pending |
| Phi-3-mini-128k | v1 | ppl | 0 | 1 | 0/8 | - | - | pending |
| Phi-3-mini-128k | v1_fa2_stack | niah | 0 | 1 | 0/8 | - | - | pending |
| Phi-3-mini-128k | v1_fa2_stack | ppl | 0 | 1 | 0/8 | - | - | pending |
| Phi-3-mini-128k | vanilla | niah | 0 | 1 | 0/8 | - | - | pending |
| Phi-3-mini-128k | vanilla | ppl | 0 | 1 | 7/8 | 5.615 | - | partial |

Outputs:
- `figures/eval_plots/wave11_interim_ppl.png`
- `figures/eval_plots/wave11_interim_niah.png`
