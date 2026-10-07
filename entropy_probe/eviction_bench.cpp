// eviction_bench.cpp - on-phone benchmark for KV-cache eviction policies.
//
// Loads a gguf model, prefills a prompt, applies the chosen eviction policy and
// logs per-step latency and KV cells used, plus summary stats (prefill ms,
// decode tok/s, peak KV cells).
//
// Policies:
//   vanilla   no eviction (cache grows up to the context size)
//   tova      per-head top-K by current attention (fixed K)
//   pyramid   per-layer K (linear K_max to K_min across depth), TOVA selection
//   v1        EndurKV-Evict per-head spread gate (α=1.3 β=0.6 on max_a)
//
// GQA: a position is kept for a kv-head if any query head sharing it wants it.
// Eviction is sequence-level (llama_kv_self_seq_rm), so a position is removed
// only if no layer or head wants it. That is what llama.cpp's public API allows.
//
// Output:
//   <out_csv>         per-step CSV (step, token, latency_us, n_kv, peak_rss_kb)
//   <out_meta>        JSON with summary (prefill_ms, decode_tps, peak_kv, etc.)
//
// Run on phone:
//   LD_LIBRARY_PATH=bin bin/eviction_bench \
//       --model models/Llama-3.2-1B.Q4_K_M.gguf \
//       --prompt prompts/narrativeqa_lc_01.txt \
//       --prompt-id narrativeqa_lc_01 \
//       --policy v1 --k-nominal 512 \
//       --max-tokens 64 --ctx-size 4096 \
//       --out-csv logs/run/x.steps.csv \
//       --out-meta logs/run/x.meta.json

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
// Optional multimodal prefill via libmtmd. Image tokens enter the same KV cache.
// Built only when CMake defines EVB_HAS_MTMD.
#ifdef EVB_HAS_MTMD
#include "mtmd.h"
#include "mtmd-helper.h"
#endif

#include <unistd.h>     // getpid()
#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

// Args

// Cap the GPU clock. max_gpuclk sets kgsl's thermal_pwrlevel, which the OEM thermal engine
// also writes and can clear mid-request. max_pwrlevel is a separate user cap combined with
// the thermal one, so write both: the index of mhz in gpu_available_frequencies, and the clock.
static int gpu_cap_write(int mhz) {
    const std::string hz = std::to_string((long long) mhz * 1000000LL);
    const std::string cmd = "su -c 'cd /sys/class/kgsl/kgsl-3d0; i=0; for x in $(cat gpu_available_frequencies); do "
                            "[ \"$x\" = " + hz + " ] && echo $i > max_pwrlevel; i=$((i+1)); done; echo " + hz + " > max_gpuclk' >/dev/null 2>&1";
    return std::system(cmd.c_str());
}

struct Args {
    std::string model;
    std::string prompt_file;
    std::string prompt_id;
    std::string policy = "vanilla";   // vanilla|tova|pyramid|v1
    int         k_nominal = 512;
    int         max_tokens = 64;
    int         ctx_size = 4096;
    int         seed = 42;
    int         n_threads = 4;
    int         n_gpu_layers = 0;
    int         n_batch = 512;        // physical batch (prompt chunks fed at once)
    int         n_ubatch = 64;        // micro-batch, kept small for Adreno Vulkan (TDR)
    int         n_sink = 4;           // sink-token protection (StreamingLLM-style)
    int         repeat_last_n = 64;   // repetition window
    float       repeat_penalty = 1.1; // llama.cpp default
    bool        fa_vanilla = true;    // enable FA for vanilla (gives realistic latency baseline)
    // Diagnostic: install the attention-capture callback even where the policy would not
    // (e.g. vanilla), to test whether the callback itself corrupts Adreno/Vulkan output.
    // Not for measurement runs.
    bool        force_cb_eval = false;
    // Diagnostic: the callback still returns true in the ask phase, so the graph still splits
    // at that node, but reads no tensor. Separates split-induced from readback-induced corruption.
    bool        cb_eval_noread = false;
    bool        no_evict_decode = false;  // if true: evict only after prefill, frozen thereafter
    bool        no_perhead_gate = false;  // bypass the per-head budget gate (policy_v1 ramp), select by pooled cross-head score over all positions
    bool        ignore_eos = false;   // if true: ignore EOS and generate all max_tokens (sustained-throughput runs)
    int         keydiff_decode_block = 128; // KeyDiff only: re-evict every N decode steps to hold the cache at its budget
                                            // (0 = select once at the end of prefill)
    bool        decode_bound = false; // decode: when n_kv > 1.5*k_nominal, keep [0, n_sink) and the last k_nominal - n_sink cells (recency only)
    bool        decode_tiered = false;// v1_FA² mode: keep all prefill anchors plus a recent window, drop the oldest decode-time cells
    int         recent_budget = 256;  // tiered decode recent window (tokens), evicts when n_kv > 1.25*(n_anchored + recent_budget)
    int         anchor_top_k = 0;     // if > 0, keep only the top anchor_top_k prefill survivors by score as anchors (0 = all)
    std::string anchor_score_mode = "mean"; // mean | entropy (broadly attended) | neg_entropy (sharply attended) | hybrid: mean*(1 + hybrid_alpha*(1 - normalized entropy))
    float       hybrid_alpha = 0.5f;        // weight of the sharpness term in hybrid mode (0 = pure mean)
    std::string gate_mode = "v1";           // "v1": m_h = max attention per head, "v2": s_h = top-1 + 0.5*top-2 (double-peaked heads count as sharp)
    float       anchor_coverage = 0.0f;     // if > 0, keep anchors until their cumulative attention mass reaches this fraction (overrides anchor_top_k)
    bool        thermal_driven_k = false;   // TDAK: every thermal_poll_steps decode steps, map DDR temperature to a K tier and evict to it
    int         thermal_poll_steps = 20;    // decode steps between thermal polls
    int         tdak_k_cool = 1024;         // K_target when DDR < tdak_t_warm
    int         tdak_k_mid  = 768;          // K_target when tdak_t_warm <= DDR < tdak_t_hot
    int         tdak_k_warm = 512;          // K_target when tdak_t_hot <= DDR < tdak_t_crit
    int         tdak_k_crit = 256;          // K_target when DDR >= tdak_t_crit (emergency, avoid kernel cliff)
    int         tdak_t_warm = 60000;        // DDR threshold (mC) for stepping cool -> mid
    int         tdak_t_hot  = 63000;        // DDR threshold for stepping mid -> warm
    int         tdak_t_crit = 65000;        // DDR threshold for stepping warm -> crit (matches kernel BCL trigger)
    bool        tdak_ratchet = false;       // tiers rise at once, step back only after DDR cools tdak_ratchet_hyst below the entry threshold
    int         tdak_ratchet_hyst = 1000;   // hysteresis (mC) for the ratchet step-down
    std::string thermal_zone_path = "/sys/class/thermal/thermal_zone47/temp"; // DDR sensor
    int         obs_window = 1;  // number of last prefill queries whose attention is averaged for scoring (1 = last query only, 32 = AdaKV-style)
    int         snapkv_pool = 0; // SnapKV 1D max-pool kernel over per-position scores (0/1 = off), so top-K picks token clusters, not isolated spikes
    bool        adaptive_anchor = false;  // per-prompt α_a splits K - n_sink into anchors and recent (overrides --anchor-top-k, --recent-budget)
    int         adaptive_rmin = 32;       // a head's peak is "distant" if argmax_p < N - adaptive_rmin
    bool        gate_count = false;       // α_a from the fraction of heads with a distant peak (count) instead of the distant attention mass
    // Budget as a percentage of the prompt instead of an absolute cell count, so K scales
    // with context. alpha_a only splits K between anchor and recent, so a fixed K can keep
    // nearly all of a long prompt. 0 = use k_nominal as given.
    float       k_pct = 0.0f;
    // Energy-aware budget: K from the battery state of charge. The healthy tier is
    // --k-nominal and lower tiers use 1/2 and 1/4 of it. Off by default because it
    // lowers output quality.
    bool        energy_aware   = false;
    int         ea_soc_hi      = 50;    // above this SoC: no degradation
    int         ea_soc_lo      = 20;    // below this SoC: most aggressive tier
    int         ea_k_hi        = 0;     // absolute K at high charge; 0 = k_nominal
    int         ea_k_mid       = 0;     // 0 = k_nominal / 2
    int         ea_k_lo        = 0;     // 0 = k_nominal / 4
    // The energy lever differs by backend. On GPU, decode is bandwidth-bound and KV is small
    // next to the weights, so the ladder caps the GPU clock and holds K. On CPU, decode is
    // attention-bound over live cells, so the ladder shrinks K and leaves the clock alone.
    // The clock write needs root (su). If it fails, the controller reports it and holds.
    int         ea_gpu_mhz_hi  = 1200;  // GPU max clock at high charge / mains (device max)
    int         ea_gpu_mhz_mid = 902;
    int         ea_gpu_mhz_lo  = 726;
    bool        ea_gpu_k_ladder = false; // measurement only: also shrink K on GPU
    int         ea_gpu_mhz_set = 0;     // observed, for meta.json (0 = no write attempted)
    int         ea_gpu_write_rc = -1;   // return code of the clock write (0 = ok)
    // Decode-only clock cap. The cap costs time mainly in prefill (compute-bound) but saves
    // energy in both phases. --gpu-mhz-decode N writes N MHz after prefill and the exit path
    // restores the device max. 0 = off (one clock for the whole request).
    int         gpu_mhz_decode = 0;
    int         gpu_mhz_decode_rc = -1;
    float       ea_pct_hi      = 0.0f;  // legacy percent-of-prompt ladder; --ea-pct hi mid lo enables it
    float       ea_pct_mid     = 0.0f;
    float       ea_pct_lo      = 0.0f;
    std::string ea_state_file  = "/data/local/tmp/ukv_ea_level";  // hysteresis across requests
    // observed, for meta.json
    int         ea_soc_seen    = -1;
    std::string ea_status_seen = "";
    int         ea_level_seen  = -1;
    // --no-fa-positional: run the positional policies (StreamingLLM, KeyDiff) with FA off.
    bool        no_fa_positional = false;
    float       gate_alpha_floor = 0.0f;  // if >0, lower bound on α_a (protects the anchor budget on retrieval prompts)
    float       gate_alpha_max = 0.0f;     // if >0, upper bound on α_a (keeps a recent window for fluency)
    bool        refresh_at_decode_0 = false;  // RefreshKV-style: run one FA-off decode step, redo anchor selection from its mean attention, then state-swap to FA-on
    bool        refresh_mukv_gate = false;  // like --refresh-at-decode-0 but reapplies the full μKV gate (per-head K_h, α_a, cross-head anchors) at refresh
    std::string user_policy;          // policy name as given on the CLI, before alias rewriting (reported in meta.json)
    std::string cache_type_k = "f16"; // KV cache K data type (f16, q8_0, q4_0)
    std::string cache_type_v = "f16"; // KV cache V data type.
    // sampling (llama.cpp defaults, same for all policies)
    bool        greedy = false;       // if true: argmax instead of sampling
    float       temperature = 0.8f;   // llama.cpp default
    float       top_p = 0.95f;        // nucleus
    int         top_k = 40;           // top-k filter (0 = disabled)
    // evaluation mode
    // "gen": sample tokens (default, writes gen.txt)
    // "ppl": after prefill and post-prefill eviction, teacher-force eval_text and compute
    //        mean -log P(token|prefix) from raw logits, as llama-perplexity does.
    std::string eval_mode = "gen";
    std::string eval_text_file;
    // multimodal (needs an EVB_HAS_MTMD build)
    std::string mmproj;                  // --mmproj PATH: vision projector gguf, empty = text-only (default)
    std::vector<std::string> images;     // --image PATH (repeatable): images injected before the prompt text
    // Vision encoder on GPU by default, as in mtmd_context_params_default, for every policy.
    // On CPU the encoder takes far longer than the LLM prefill being measured.
    bool        mmproj_gpu = true;       // --mmproj-cpu to opt out (e.g. if VRAM-constrained)
    // --ppl-per-step: in ppl mode, run the per-step eviction during teacher forcing even when
    // the mask would be frozen, so progressive baselines (TOVA, H2O) are scored with their
    // own one-eviction-per-step schedule.
    bool        ppl_per_step = false;
    // SnapKV-style two-context decode (--snapkv-decode):
    //   1. Prefill in an FA-off context (attention captured, eviction applied).
    //   2. Save the sequence with llama_state_seq_get_data.
    //   3. Destroy the FA-off context and create an FA-on one.
    //   4. Restore it with llama_state_seq_set_data.
    //   5. Decode on the FA-on context with no further eviction.
    // Implies --no-evict-decode, since FA-on does not expose kq_soft_max.
    bool        snapkv_decode = false;
    int         snapkv_kernel = 5;      // SnapKV avg-pool kernel. The paper uses window 16 / kernel 5 (NiaH),
                                       // window 32 / kernel 7 (LongBench), window 64 / kernel 13 (Command-R).
    int         defrag_mode   = -1;  // -1 = auto (per GPU backend), 0 = off, 1 = on. See detect_gpu_profile_wants_defrag().
    // --compact-inplace slides the survivors down inside the tensors prefill already
    // allocated, so peak memory does not rise. defrag_mode==1 instead round-trips the cache
    // through a second context and peaks at 2x the cache. compact_chunk is the staging size
    // in cells (512 KiB of host buffer for Llama-3.2-1B: one 512-cell chunk of one layer, f16).
    int         compact_inplace = 0;
    // After compaction, madvise(MADV_DONTNEED) the dead tail so its pages return to the OS.
    // The cache is one buffer sized and memset for the full context, so compaction alone
    // does not lower residency. Host (CPU) caches only.
    int         reclaim_tail = 0;
    int         compact_chunk   = 512;
                                       // Without compaction the evicted cache stays physically sparse: seq_rm
                                       // frees cells but does not move survivors, and attention still scans to
                                       // the highest occupied index. Defrag is on by default for Vulkan/Adreno
                                       // and CUDA, --no-defrag turns it off.
    bool        no_extra_bufts = false;  // disable llama.cpp weight-repack buffer types
    bool        no_state_swap = false;   // keep the gate and evict-once but skip the FA-off to FA-on state swap (slow on Vulkan)
    bool        fa_on_evict = false;     // FA-on prefill and decode, gate scores from the in-graph kq_evict node (softmax(K.Q_lastW)), no state swap
    std::string out_csv;
    std::string out_meta;
    std::string out_gen;
    std::string out_prefill_csv;  // per-chunk prefill trace (optional)
};

void usage(const char * argv0) {
    std::fprintf(stderr,
        "Usage: %s --model PATH --prompt PATH --prompt-id ID \\\n"
        "       --policy {vanilla|tova|pyramid|v1} --k-nominal N \\\n"
        "       --max-tokens N [--ctx-size N] [--seed N] [--threads N] \\\n"
        "       --out-csv PATH --out-meta PATH [--out-gen PATH]\n",
        argv0);
}

bool parse_args(int argc, char ** argv, Args & a) {
    // Track whether the caller set the KV cache types, so policy presets do not override them.
    bool k_explicit = false, v_explicit = false;
    std::string k_req, v_req;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto need = [&](int j) {
            if (i + j >= argc) { std::fprintf(stderr, "missing value for %s\n", k.c_str()); std::exit(1); }
            return std::string(argv[i + j]);
        };
        if      (k == "--model")        { a.model       = need(1); i++; }
        else if (k == "--prompt")       { a.prompt_file = need(1); i++; }
        else if (k == "--prompt-id")    { a.prompt_id   = need(1); i++; }
        else if (k == "--policy")       { a.policy      = need(1); i++; }
        else if (k == "--k-nominal")    { a.k_nominal   = std::atoi(need(1).c_str()); i++; }
        else if (k == "--k-pct")        { a.k_pct       = std::atof(need(1).c_str()); i++; }
        else if (k == "--energy-aware") { a.energy_aware = true; }
        else if (k == "--ea-soc-hi")    { a.ea_soc_hi    = std::atoi(need(1).c_str()); i++; }
        else if (k == "--ea-soc-lo")    { a.ea_soc_lo    = std::atoi(need(1).c_str()); i++; }
        else if (k == "--ea-pct")       { a.ea_pct_hi = std::atof(need(1).c_str()); i++;
                                          a.ea_pct_mid = std::atof(need(1).c_str()); i++;
                                          a.ea_pct_lo  = std::atof(need(1).c_str()); i++; }
        else if (k == "--ea-k")         { a.ea_k_hi  = std::atoi(need(1).c_str()); i++;
                                          a.ea_k_mid = std::atoi(need(1).c_str()); i++;
                                          a.ea_k_lo  = std::atoi(need(1).c_str()); i++; }
        else if (k == "--ea-gpu-mhz")   { a.ea_gpu_mhz_hi  = std::atoi(need(1).c_str()); i++;
                                          a.ea_gpu_mhz_mid = std::atoi(need(1).c_str()); i++;
                                          a.ea_gpu_mhz_lo  = std::atoi(need(1).c_str()); i++; }
        else if (k == "--ea-gpu-k-ladder") { a.ea_gpu_k_ladder = true; }
        else if (k == "--gpu-mhz-decode") { a.gpu_mhz_decode = std::atoi(need(1).c_str()); i++; }
        else if (k == "--ea-state")     { a.ea_state_file = need(1); i++; }
        else if (k == "--no-fa-positional") { a.no_fa_positional = true; }
        else if (k == "--max-tokens")   { a.max_tokens  = std::atoi(need(1).c_str()); i++; }
        else if (k == "--ctx-size")     { a.ctx_size    = std::atoi(need(1).c_str()); i++; }
        else if (k == "--seed")         { a.seed        = std::atoi(need(1).c_str()); i++; }
        else if (k == "--threads")      { a.n_threads   = std::atoi(need(1).c_str()); i++; }
        else if (k == "--n-gpu-layers") { a.n_gpu_layers= std::atoi(need(1).c_str()); i++; }
        else if (k == "--n-batch")      { a.n_batch     = std::atoi(need(1).c_str()); i++; }
        else if (k == "--ubatch-size" || k == "--n-ubatch")
                                        { a.n_ubatch    = std::atoi(need(1).c_str()); i++; }
        else if (k == "--n-sink")       { a.n_sink      = std::atoi(need(1).c_str()); i++; }
        else if (k == "--repeat-last-n"){ a.repeat_last_n= std::atoi(need(1).c_str()); i++; }
        else if (k == "--repeat-penalty"){a.repeat_penalty= std::atof(need(1).c_str()); i++; }
        else if (k == "--no-fa-vanilla"){ a.fa_vanilla   = false; }
        else if (k == "--force-cb-eval"){ a.force_cb_eval = true; }   // diagnostic, see Args
        else if (k == "--cb-eval-noread"){ a.cb_eval_noread = true; } // diagnostic, see Args
        else if (k == "--no-evict-decode"){ a.no_evict_decode= true; }
        else if (k == "--ignore-eos")   { a.ignore_eos    = true; }
        else if (k == "--decode-bound") { a.decode_bound  = true; }
        else if (k == "--decode-tiered"){ a.decode_tiered = true; }
        else if (k == "--keydiff-decode-block"){ a.keydiff_decode_block = std::stoi(argv[++i]); }
        else if (k == "--recent-budget"){ a.recent_budget = std::atoi(need(1).c_str()); i++; }
        else if (k == "--anchor-top-k") { a.anchor_top_k  = std::atoi(need(1).c_str()); i++; }
        else if (k == "--anchor-score-mode") { a.anchor_score_mode = need(1); i++; }
        else if (k == "--gate-mode") { a.gate_mode = need(1); i++; }
        else if (k == "--anchor-coverage") { a.anchor_coverage = std::atof(need(1).c_str()); i++; }
        else if (k == "--thermal-driven-k") { a.thermal_driven_k = true; }
        else if (k == "--thermal-poll-steps") { a.thermal_poll_steps = std::atoi(need(1).c_str()); i++; }
        else if (k == "--tdak-k-cool") { a.tdak_k_cool = std::atoi(need(1).c_str()); i++; }
        else if (k == "--tdak-k-mid") { a.tdak_k_mid = std::atoi(need(1).c_str()); i++; }
        else if (k == "--tdak-k-warm") { a.tdak_k_warm = std::atoi(need(1).c_str()); i++; }
        else if (k == "--tdak-k-crit") { a.tdak_k_crit = std::atoi(need(1).c_str()); i++; }
        else if (k == "--tdak-t-warm") { a.tdak_t_warm = std::atoi(need(1).c_str()); i++; }
        else if (k == "--tdak-t-hot")  { a.tdak_t_hot  = std::atoi(need(1).c_str()); i++; }
        else if (k == "--tdak-t-crit") { a.tdak_t_crit = std::atoi(need(1).c_str()); i++; }
        else if (k == "--tdak-ratchet") { a.tdak_ratchet = true; }
        else if (k == "--tdak-ratchet-hyst") { a.tdak_ratchet_hyst = std::atoi(need(1).c_str()); i++; }
        else if (k == "--thermal-zone-path") { a.thermal_zone_path = need(1); i++; }
        else if (k == "--obs-window") { a.obs_window = std::atoi(need(1).c_str()); i++; }
        else if (k == "--snapkv-pool"){ a.snapkv_pool = std::atoi(need(1).c_str()); i++; }
        else if (k == "--adaptive-anchor") { a.adaptive_anchor = true; }
        else if (k == "--gate-count") { a.gate_count = true; }
        else if (k == "--gate-alpha-floor") { a.gate_alpha_floor = std::atof(need(1).c_str()); i++; }
        else if (k == "--gate-alpha-max") { a.gate_alpha_max = std::atof(need(1).c_str()); i++; }
        else if (k == "--adaptive-rmin") { a.adaptive_rmin = std::atoi(need(1).c_str()); i++; }
        else if (k == "--refresh-at-decode-0") { a.refresh_at_decode_0 = true; }
        else if (k == "--refresh-mukv-gate") { a.refresh_mukv_gate = true; a.refresh_at_decode_0 = true; }
        else if (k == "--hybrid-alpha") { a.hybrid_alpha = std::atof(need(1).c_str()); i++; }
        // Explicit cache types win over the per-policy presets below.
        else if (k == "--cache-type-k") { a.cache_type_k  = need(1); k_req = a.cache_type_k; k_explicit = true; i++; }
        else if (k == "--cache-type-v") { a.cache_type_v  = need(1); v_req = a.cache_type_v; v_explicit = true; i++; }
        else if (k == "--greedy")       { a.greedy      = true; }
        else if (k == "--temperature")  { a.temperature = std::atof(need(1).c_str()); i++; }
        else if (k == "--top-p")        { a.top_p       = std::atof(need(1).c_str()); i++; }
        else if (k == "--top-k")        { a.top_k       = std::atoi(need(1).c_str()); i++; }
        else if (k == "--eval-mode")    { a.eval_mode   = need(1); i++; }
        else if (k == "--eval-text")    { a.eval_text_file = need(1); i++; }
        else if (k == "--mmproj")       { a.mmproj      = need(1); i++; }
        else if (k == "--image")        { a.images.push_back(need(1)); i++; }
        else if (k == "--mmproj-gpu")   { a.mmproj_gpu  = true; }   // default (llama.cpp parity)
        else if (k == "--mmproj-cpu")   { a.mmproj_gpu  = false; }  // opt out of the default
        else if (k == "--ppl-per-step") { a.ppl_per_step = true; }
        else if (k == "--snapkv-decode"){ a.snapkv_decode = true; a.no_evict_decode = true; }
        else if (k == "--no-state-swap"){ a.no_state_swap = true; }
        else if (k == "--compact-inplace") { a.defrag_mode = 1; a.compact_inplace = 1; }
        else if (k == "--reclaim-tail")    { a.reclaim_tail = 1; }
        else if (k == "--compact-chunk")   { a.compact_chunk  = std::atoi(argv[++i]); }
        else if (k == "--force-defrag") { a.defrag_mode   = 1; }
        else if (k == "--no-defrag")    { a.defrag_mode   = 0; }
        else if (k == "--no-extra-bufts") { a.no_extra_bufts = true; }
        else if (k == "--snapkv-kernel"){ a.snapkv_kernel = std::atoi(need(1).c_str()); i++; }
        else if (k == "--fa-on-evict"){ a.fa_on_evict = true; a.no_evict_decode = true; }
        else if (k == "--no-perhead-gate"){ a.no_perhead_gate = true; }
        else if (k == "--out-csv")      { a.out_csv     = need(1); i++; }
        else if (k == "--out-meta")     { a.out_meta    = need(1); i++; }
        else if (k == "--out-gen")      { a.out_gen     = need(1); i++; }
        else if (k == "--out-prefill-csv") { a.out_prefill_csv = need(1); i++; }
        else if (k == "-h" || k == "--help") { usage(argv[0]); std::exit(0); }
        else { std::fprintf(stderr, "unknown arg: %s\n", k.c_str()); usage(argv[0]); return false; }
    }
    if (a.model.empty() || a.prompt_file.empty() || a.prompt_id.empty() ||
        a.out_csv.empty() || a.out_meta.empty()) {
        usage(argv[0]); return false;
    }
    if (a.policy != "vanilla" && a.policy != "tova" && a.policy != "tova_canonical" && a.policy != "pyramid" &&
        a.policy != "v1"      && a.policy != "h2o"  && a.policy != "v1_fa" &&
        a.policy != "v1_fa2"  && a.policy != "streamingllm" &&
        a.policy != "v1_entropy" && a.policy != "v1_entropy_stack" &&
        a.policy != "v1_predictive" && a.policy != "v1_predictive_stack" &&
        a.policy != "v1_fa2_hybrid" && a.policy != "v1_fa2_f16" &&
        a.policy != "endurkv_optimal" && a.policy != "endurkv_adaptive" &&
        a.policy != "adakv" && a.policy != "snapkv" &&
        a.policy != "keydiff") {   // KeyDiff baseline
        std::fprintf(stderr, "bad policy: %s\n", a.policy.c_str()); return false;
    }
    // v1_fa: v1 eviction with FA-off prefill and a state swap to FA-on decode. FA-on does
    // not expose kq_soft_max, so the mask freezes after prefill. On long generations the
    // cache then grows again, so a recency bound (sinks plus recent window) is enabled too.
    if (a.policy == "v1_fa") {
        a.snapkv_decode    = true;
        a.no_evict_decode  = true;
        a.decode_bound     = true;   // bound cache during FA-on decode via recency
    }
    // v1_fa2 (v1_FA²): like v1_fa, but tiered decode eviction keeps the prefill anchors and
    // drops only the oldest decode-time cells. The launcher adapts K from DDR temperature.
    if (a.policy == "v1_fa2") {
        a.user_policy      = "v1_fa2"; // preserve original label for meta.json
        a.policy           = "v1_fa";  // reuse v1_fa code path
        a.snapkv_decode    = true;
        a.no_evict_decode  = true;
        a.decode_tiered    = true;   // override: tiered eviction instead of recency
        a.decode_bound     = false;
        // Selective anchoring: top-32 scored prompt positions plus a large recent window.
        // Anchoring the whole prompt starved the recent context and raised PPL.
        if (a.anchor_top_k == 0) a.anchor_top_k = 32;
        if (a.recent_budget == 256) a.recent_budget = 476;  // K=512 - n_sink=4 - anchor=32
    }
    // v1_fa2_hybrid: isolation cell. Canonical H2O eviction (recent K/2 + heavy K/2) with
    // the v1_FA² watchdog and memory gate but no state swap and no Q8 K (cache f16/f16,
    // set by the launcher). Isolates what the watchdog alone does to H2O.
    if (a.policy == "v1_fa2_hybrid") {
        a.user_policy      = "v1_fa2_hybrid"; // preserve original label for meta.json
        a.policy           = "h2o";           // reuse canonical h2o eviction path
        a.snapkv_decode    = false;           // no state-swap
        a.no_evict_decode  = false;           // h2o keeps evicting during decode
        a.decode_tiered    = false;
        a.decode_bound     = false;
        // anchor_top_k and recent_budget are not used by h2o.
    }
    // endurkv_optimal: canonical H2O eviction (recent K/2 + heavy K/2) with the v1_FA²
    // thermal stack: state swap to FA-on decode, plus Q8 K + f16 V, the watchdog and the
    // memory gate from the launcher. The mask is frozen after prefill.
    // a.policy is not rewritten to "h2o", so a dedicated prefill dispatch branch calls
    // policy_h2o() once at prefill instead of h2o's per-step decode eviction.
    if (a.policy == "endurkv_optimal") {
        a.user_policy      = "endurkv_optimal"; // preserve original label for meta.json
        // a.policy left as "endurkv_optimal" - handled by dedicated dispatch branch.
        a.snapkv_decode    = true;               // state swap to FA-on decode
        a.no_evict_decode  = true;               // frozen mask during decode
        a.decode_tiered    = false;              // h2o has no anchor/tiered structure
        a.decode_bound     = false;
        // cache_type_k = q8_0 and cache_type_v = f16 are set by the launcher.
        // anchor_top_k and recent_budget are not used by h2o.
    }
    // endurkv_adaptive: v1 spread-gate eviction with a runtime K_effective(t) controller
    // driven by DDR/CPU/skin/battery temperatures (AdaptiveKController). No state swap,
    // since the FA-off to FA-on swap raised PPL on Llama-1B, so per-step eviction stays on.
    // f16 K + f16 V, because Q8 K cost PPL with no speedup. Top-32 anchoring as in v1_fa2.
    if (a.policy == "endurkv_adaptive") {
        a.user_policy      = "endurkv_adaptive"; // preserve original label for meta.json
        a.policy           = "v1_adaptive";      // routes to policy_v1_adaptive dispatch
        a.snapkv_decode    = false;              // no state-swap
        a.no_evict_decode  = false;              // allow decode-time re-eviction
        a.decode_tiered    = false;
        a.decode_bound     = false;
        if (a.anchor_top_k == 0) a.anchor_top_k = 32;
    }
    // v1_fa2_f16: isolation cell. Same as v1_fa2_stack (spread gate, top-32 anchors, state
    // swap to FA-on decode) but with f16 K + f16 V, to isolate the PPL cost of Q8 K.
    if (a.policy == "v1_fa2_f16") {
        a.user_policy      = "v1_fa2_f16"; // preserve original label for meta.json
        a.policy           = "v1_fa";      // reuse v1_fa code path (same as v1_fa2)
        a.snapkv_decode    = true;
        a.no_evict_decode  = true;
        a.decode_tiered    = true;
        a.decode_bound     = false;
        if (a.anchor_top_k == 0) a.anchor_top_k = 32;
        if (a.recent_budget == 256) a.recent_budget = 476;  // K=512 - n_sink=4 - anchor=32
    }
    // v1_entropy_stack: entropy-weighted scoring with the recent+heavy split and the v1_fa2
    // thermal stack. The launcher sets --cache-type-k q8_0 and the watchdog.
    if (a.policy == "v1_entropy_stack" || a.policy == "v1_entropy") {
        const bool is_stack = (a.policy == "v1_entropy_stack");
        a.policy           = "v1_entropy";
        if (is_stack) {
            a.user_policy     = "v1_entropy_stack";
            a.snapkv_decode   = true;
            a.no_evict_decode = true;
            a.decode_tiered   = true;
            a.decode_bound    = false;
            if (a.anchor_top_k  == 0)   a.anchor_top_k  = 32;
            if (a.recent_budget == 256) a.recent_budget = 476;
        }
    }
    // v1_predictive_stack: scores by attention trajectory (slope), so it needs kq_soft_max
    // at every decode step and runs FA-off throughout. Memory gate and Q8 K come from the launcher.
    if (a.policy == "v1_predictive_stack" || a.policy == "v1_predictive") {
        const bool is_stack = (a.policy == "v1_predictive_stack");
        a.policy           = "v1_predictive";
        if (is_stack) {
            a.user_policy     = "v1_predictive_stack";
            // Per-step eviction needs attention at every step, so no FA-on swap.
            a.snapkv_decode   = false;
            a.no_evict_decode = false;
            a.decode_bound    = false;
            a.decode_tiered   = false;
            if (a.recent_budget == 256) a.recent_budget = a.k_nominal / 2;
        }
    }
    // Ada-KV (Feng et al. NeurIPS 2025): adaptive per-head budgets. Algorithm 1 takes the
    // global top-(B·H) attention weights across heads and counts picks per head to get
    // {B_i*}. Algorithm 2 applies it to SnapKV-style window attention. Wired as Ada-SnapKV:
    // FA-off prefill, no state swap, mask frozen after prefill, f16 K + f16 V.
    if (a.policy == "adakv") {
        a.user_policy      = "adakv";
        a.snapkv_decode    = false;   // Ada-KV does not state-swap
        a.no_evict_decode  = true;    // frozen mask after end-of-prefill selection
        a.cache_type_k     = "f16";
        a.cache_type_v     = "f16";
    }
    if (a.policy == "snapkv") {       // canonical SnapKV baseline (per-head, avgpool, uniform K)
        a.user_policy      = "snapkv";
        a.snapkv_decode    = false;   // SnapKV freezes the prompt keep-set, no state swap
        a.no_evict_decode  = true;    // frozen mask after end-of-prefill selection
        a.cache_type_k     = "f16";   // SnapKV defaults (q8_0 is ours)
        a.cache_type_v     = "f16";
    }
    // Presets are defaults, an explicit --cache-type-k/--cache-type-v wins. V is still forced
    // to f16 downstream for FA-off policies, because llama.cpp needs flash attention for a
    // quantized V.
    if (k_explicit) a.cache_type_k = k_req;
    if (v_explicit) a.cache_type_v = v_req;
    return true;
}


// Post-eviction compaction is a state get/set round-trip. Without it the cache is logically
// evicted but physically sparse, so decode still walks the full span. The default is chosen
// per backend below, --force-defrag and --no-defrag override it.
//
// g_compaction_applied records whether compaction actually happened. The swap builds a
// second (FA-on) context that can fail to allocate, and the run then falls back to FA-off
// over the uncompacted cache. retained_kv_bytes reads the same either way.
static size_t g_reclaimed_bytes = 0;   // bytes madvise()d back to the OS
static long   g_rss_pre_reclaim_kb  = -1;  // RSS just before the madvise
static long   g_rss_post_reclaim_kb = -1;  // RSS just after
static bool g_compaction_applied = false;
// Which compaction ran: "none", "roundtrip" (second context, 2x peak) or "inplace"
// (chunked slide, no extra peak). Same keep-set, different memory behaviour.
static const char * g_compaction_mode = "none";
static const char * g_fa_off_gate     = "n/a";

static bool detect_gpu_profile_wants_defrag() {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        const char * n = ggml_backend_dev_name(ggml_backend_dev_get(i));
        if (!n) continue;
        std::string d(n);
        for (auto & c : d) c = (char) std::tolower((unsigned char) c);
        if (d.find("cuda") != std::string::npos || d.find("rocm") != std::string::npos ||
            d.find("hip")  != std::string::npos || d.find("sycl") != std::string::npos) return true;
        // On Adreno the round-trip costs a few seconds once and decode is faster for the
        // rest of the run, so it pays off over a long generation. Metal is unmeasured.
        if (d.find("vulkan") != std::string::npos || d.find("adreno") != std::string::npos) return true;
        if (d.find("metal")  != std::string::npos) return false;   // unmeasured
    }
    return false;   // unknown backend: no compaction
}

bool read_file(const std::string & p, std::string & out) {
    std::ifstream f(p, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", p.c_str()); return false; }
    std::stringstream ss; ss << f.rdbuf();
    out = ss.str();
    return true;
}

// Attention capture (a simplified attention_probe.cpp).
// Per layer, kq_soft_max has shape [n_kv, n_head, n_seq_tokens].
struct AttnCapture {
    // per_layer[layer] = flat [n_head * n_kv], layout [h0_p0..h0_pN, h1_p0..], averaged over
    // the last n_window query positions (1 = last query only, 32 = AdaKV-style window).
    std::map<int, std::vector<float>> per_layer;

    // High-water mark of host_bytes(), the host cost of the scoring state (one
    // [n_head x n_kv] vector per layer, growing with context). per_layer is freed once the
    // keep-set is computed, but peak RSS still counts it, so only the maximum is comparable.
    size_t peak_host = 0;

    size_t host_bytes() const {
        size_t b = 0;
        for (const auto & kv : per_layer) b += kv.second.capacity() * sizeof(float);
        return b;
    }
    int n_head = 0;
    int n_kv = 0;
    bool active = false;
    int  obs_window = 1;  // number of trailing query positions to average over

    // Set when eval_callback meets a kq_soft_max it cannot read (blocked or quantized
    // type), so the run is flagged in meta.json instead of scoring from a bad read.
    bool capture_failed = false;

    void reset() { per_layer.clear(); n_kv = 0; }
};

static int parse_layer_index(const char * name) {
    if (!name) return -1;
    size_t L = std::strlen(name);
    size_t end = L;
    while (end > 0 && !std::isdigit((unsigned char)name[end-1])) --end;
    if (end == 0) return -1;
    size_t s = end;
    while (s > 0 && std::isdigit((unsigned char)name[s-1])) --s;
    if (s == end) return -1;
    return std::atoi(name + s);
}

// Set from args.cb_eval_noread. The callback has no access to Args.
static bool g_cb_eval_noread = false;

bool eval_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * cap = static_cast<AttnCapture *>(user_data);
    // Return false for nodes we do not read. A true from the ask phase makes
    // ggml_backend_sched add a compute+sync boundary at that node, which breaks node
    // batching and slows decode badly. Only kq_soft_max and kq_evict return true.
    if (!cap || !cap->active) return false;
    if (!t || !t->name) return false;
    const bool is_evict = std::strncmp(t->name, "kq_evict", 8) == 0;
    const bool is_soft  = std::strncmp(t->name, "kq_soft_max", 11) == 0;
    if (!is_evict && !is_soft) return false;
    if (ask) return true;  // tell scheduler we want to read it after compute
    // Diagnostic: keep the split, skip the readback. See Args::cb_eval_noread.
    if (g_cb_eval_noread) return true;

    const int layer = parse_layer_index(t->name);
    if (layer < 0) return true;

    if (is_evict) {
        // kq_evict side node, already softmaxed. Averaging its W window rows gives the same
        // [n_head*n_kv] signal as the FA-off kq_soft_max path, with flash attention on.
        // Shapes: [1, n_kv, n_head] (mean already taken on device) or [n_kv, W, n_head]
        // (mean taken here). ne[0]==1 tells them apart.
        if ((int)t->ne[0] == 1) {
            const int n_kv   = (int)t->ne[1];
            const int n_head = (int)t->ne[2];
            if (n_kv < 1 || n_head < 1) return true;
            std::vector<float> buf((size_t)n_head * (size_t)n_kv);
            ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * sizeof(float)); // already the mean
            cap->per_layer[layer] = std::move(buf);
        cap->peak_host = std::max(cap->peak_host, cap->host_bytes());
            cap->peak_host = std::max(cap->peak_host, cap->host_bytes());
            cap->n_head = n_head;
            cap->n_kv   = n_kv;
            return true;
        }
        const int n_kv   = (int)t->ne[0];
        const int W      = (int)t->ne[1];
        const int n_head = (int)t->ne[2];
        if (n_kv < 1 || W < 1 || n_head < 1) return true;
        std::vector<float> all((size_t)n_kv * (size_t)W * (size_t)n_head);
        ggml_backend_tensor_get(t, all.data(), 0, all.size() * sizeof(float));
        std::vector<float> buf((size_t)n_head * (size_t)n_kv, 0.0f);
        for (int h = 0; h < n_head; ++h) {
            for (int w = 0; w < W; ++w) {
                const size_t base = (size_t)w*n_kv + (size_t)h*n_kv*W;
                for (int p = 0; p < n_kv; ++p) buf[(size_t)h*n_kv + p] += all[base + p];
            }
        }
        const float inv = 1.0f / (float)W;
        for (auto & v : buf) v *= inv;
        cap->per_layer[layer] = std::move(buf);
        cap->peak_host = std::max(cap->peak_host, cap->host_bytes());
        cap->n_head = n_head;
        cap->n_kv   = n_kv;
        return true;
    }

    // t shape: [n_kv, n_head, n_seq_tokens] (after softmax)
    const int64_t ne0 = t->ne[0];   // n_kv
    const int64_t ne1 = t->ne[1];   // n_head
    const int64_t ne2 = t->ne[2];   // n_seq_tokens (1 during decode)

    if (ne2 < 1) return true;
    const int n_head = (int)ne1;
    const int n_kv   = (int)ne0;

    // Read kq_soft_max with its own element size and strides (nb[1] row, nb[2] query
    // slice), converting f16 to f32 when needed. A type it cannot read (blocked or
    // quantized) sets capture_failed instead of scoring from a mis-sized read.
    //
    // On Adreno/Vulkan, returning true from the ask phase splits the graph at kq_soft_max,
    // and that split alone corrupts later attention output, the readback does not (see
    // --force-cb-eval and --cb-eval-noread). kq_evict is a separate side node and is not
    // affected. Phone-GPU output tokens of FA-off policies using this path are not valid.
    const size_t ts = ggml_type_size(t->type);
    if (ggml_blck_size(t->type) != 1 || (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16)) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            fprintf(stderr, "[endurkv] WARNING: %s has unreadable type %s "
                            "(ne=[%lld,%lld,%lld]); attention capture DISABLED for this run\n",
                    t->name, ggml_type_name(t->type),
                    (long long)ne0, (long long)ne1, (long long)ne2);
        }
        cap->capture_failed = true;
        return true;
    }

    static bool logged_layout = false;
    if (!logged_layout) {
        logged_layout = true;
        fprintf(stderr, "[endurkv] kq_soft_max layout: type=%s ne=[%lld,%lld,%lld] "
                        "nb=[%zu,%zu,%zu] packed_row=%s\n",
                ggml_type_name(t->type), (long long)ne0, (long long)ne1, (long long)ne2,
                t->nb[0], t->nb[1], t->nb[2],
                (t->nb[1] == (size_t)ne0 * ts) ? "yes" : "NO (padded)");
    }

    // Average attn[h, p] over the last cap->obs_window query positions (1 = last query
    // only, 32 = AdaKV-style observation window).
    const int win = std::max(1, std::min(cap->obs_window, (int)ne2));
    std::vector<float> buf((size_t)n_head * (size_t)n_kv, 0.0f);
    std::vector<float> tmp((size_t)n_head * (size_t)n_kv);
    std::vector<uint8_t> raw((size_t)n_kv * ts);   // one row, in the tensor's own type
    for (int w = 0; w < win; ++w) {
        // Query row index, counting back from the last query position
        const int64_t q_idx = ne2 - 1 - w;
        if (q_idx < 0) break;
        // One row at a time with the tensor's own strides, so a padded nb[1] works.
        for (int h = 0; h < n_head; ++h) {
            const size_t off = (size_t)q_idx * t->nb[2] + (size_t)h * t->nb[1];
            ggml_backend_tensor_get(t, raw.data(), off, (size_t)n_kv * ts);
            float * dst = tmp.data() + (size_t)h * n_kv;
            if (t->type == GGML_TYPE_F32) {
                std::memcpy(dst, raw.data(), (size_t)n_kv * sizeof(float));
            } else {
                ggml_fp16_to_fp32_row((const ggml_fp16_t *)raw.data(), dst, n_kv);
            }
        }
        for (size_t i = 0; i < buf.size(); ++i) buf[i] += tmp[i];
    }
    // Average over the window
    const float inv_win = 1.0f / (float)win;
    for (auto & v : buf) v *= inv_win;
    cap->per_layer[layer] = std::move(buf);
    cap->n_head = n_head;
    cap->n_kv   = n_kv;
    return true;
}

// Policies - each returns per-(layer, kv_head, position) boolean keep mask
// flattened as keep[layer * n_kv_heads * n_kv + kv_head * n_kv + pos]
struct PolicyResult {
    // For each layer, the positions to keep (per kv-head, OR'd over query heads).
    // Sequence-level eviction then ORs these across layers into one keep set.
    std::vector<std::unordered_set<int>> per_layer_keep;
    int n_layers = 0;
};

// Persistent state for H2O (Zhang et al., NeurIPS '23): cumulative attention per
// (layer, head, position) over all forward passes. Heavy hitters are the positions
// whose cumulative attention ranks in the top K_nominal.
struct H2OState {
    // sum_attn[layer] = [n_head][n_kv] flat accumulator
    std::map<int, std::vector<double>> sum_attn;     // per-layer flat [h*n_kv + p]
    std::map<int, int> n_head_per_layer;
    std::map<int, int> n_kv_per_layer;

    // Update cumulative sums with this step's attention.
    void update(const AttnCapture & cap) {
        if (cap.per_layer.empty()) return;
        for (const auto & [layer, attn] : cap.per_layer) {
            const int H = cap.n_head, K = cap.n_kv;
            const size_t need = (size_t)H * (size_t)K;
            auto & sum = sum_attn[layer];
            if (sum.size() < need) sum.resize(need, 0.0);
            n_head_per_layer[layer] = H;
            n_kv_per_layer[layer]   = K;
            for (size_t i = 0; i < need; ++i) sum[i] += (double)attn[i];
        }
    }
};

// Sink-token protection (StreamingLLM): always keep the first n_sink positions in every
// layer. They hold BOS and instructions, and dropping them causes degenerate repetition.
static void protect_sink(PolicyResult & r, int n_sink, int n_kv) {
    if (n_sink <= 0) return;
    for (auto & kept : r.per_layer_keep) {
        for (int p = 0; p < std::min(n_sink, n_kv); ++p) kept.insert(p);
    }
}

// Compute "effective KV" metrics: attention mass retained + retention ratio.
// Returns {mass_retained, retention_ratio, eviction_efficiency}.
// mass_retained = Σ_kept attn / Σ_total attn  (closer to 1 = better)
// retention_ratio = n_kept_cells / n_total_cells (smaller = more aggressive)
// efficiency = mass_retained / retention_ratio  (higher = better)
struct EffectiveKV {
    double mass_retained = 0.0;
    double retention_ratio = 0.0;
    double efficiency = 0.0;
};
static EffectiveKV compute_effective_kv(const AttnCapture & cap,
                                        const PolicyResult & pol,
                                        int n_query_heads) {
    EffectiveKV out{};
    if (cap.per_layer.empty() || pol.per_layer_keep.empty()) return out;
    const int n_kv = cap.n_kv;
    double total_mass_kept = 0.0, total_mass = 0.0;
    int total_kept_cells = 0, total_cells = 0;
    for (const auto & [layer, attn] : cap.per_layer) {
        if (layer >= (int)pol.per_layer_keep.size()) continue;
        const auto & kept = pol.per_layer_keep[layer];
        for (int h = 0; h < n_query_heads; ++h) {
            const float * a = attn.data() + (size_t)h * n_kv;
            double sum_all = 0.0, sum_kept = 0.0;
            for (int p = 0; p < n_kv; ++p) {
                sum_all += a[p];
                if (kept.count(p)) sum_kept += a[p];
            }
            total_mass += sum_all;
            total_mass_kept += sum_kept;
            total_cells += n_kv;
            total_kept_cells += (int)kept.size();
        }
    }
    if (total_mass > 0) out.mass_retained = total_mass_kept / total_mass;
    if (total_cells > 0) out.retention_ratio = (double)total_kept_cells / (double)total_cells;
    if (out.retention_ratio > 1e-9) out.efficiency = out.mass_retained / out.retention_ratio;
    return out;
}


// Canonical SnapKV (Li et al. NeurIPS 2024), without any of µKV's additions. The reference
// init_snapkv() (benchmarks/snapkv_utils_reference.py) uses window_size=32,
// max_capacity_prompt=2048, kernel_size=5, avgpool. K_nominal maps to max_capacity_prompt
// and is kept equal to muKV's budget for a matched memory comparison.
//   1. Per head, the last window_size keys (the observation window) are always kept.
//   2. From the prefix [0, n_kv - window_size), keep the top (max_capacity_prompt -
//      window_size) keys by avg-pooled window attention.
//   3. apply_eviction() unions the per-head sets across heads.
// Heads keep different prefix positions, so the union covers nearly the whole prefix and a
// sequence-level engine (llama.cpp seq_rm) cannot physically reclaim the cache.
static PolicyResult policy_snapkv(const AttnCapture & cap, int K_nominal,
                                  int window_size = 64, int kernel = 5) {
    PolicyResult r;
    if (cap.per_layer.empty()) return r;
    const int n_query_heads = cap.n_head;
    const int n_kv          = cap.n_kv;
    for (const auto & [layer, attn] : cap.per_layer) r.n_layers = std::max(r.n_layers, layer + 1);
    r.per_layer_keep.resize(r.n_layers);
    const int radius     = std::max(1, kernel / 2);              // avgpool kernel 5 -> radius 2
    const int W          = std::min(window_size, n_kv);          // observation window (always kept)
    const int prefix_len = n_kv - W;                             // prefix region [0, prefix_len)
    const int K_prefix   = std::max(0, std::min(K_nominal - W, prefix_len)); // top-K from prefix
    std::vector<float> pooled;
    std::vector<int>   idx;
    for (const auto & [layer, attn] : cap.per_layer) {
        auto & keep = r.per_layer_keep[layer];
        for (int p = prefix_len; p < n_kv; ++p) keep.insert(p);  // always keep the window (all heads)
        if (K_prefix <= 0 || prefix_len <= 0) continue;
        pooled.assign((size_t)prefix_len, 0.0f);
        idx.resize(prefix_len);
        for (int h = 0; h < n_query_heads; ++h) {
            const float * a = attn.data() + (size_t)h * n_kv;    // obs-window attention over all keys
            for (int p = 0; p < prefix_len; ++p) {               // 1-D avgpool over the prefix only
                int lo = std::max(0, p - radius), hi = std::min(prefix_len - 1, p + radius);
                float s = 0.0f; for (int q = lo; q <= hi; ++q) s += a[q];
                pooled[p] = s / (float)(hi - lo + 1);
            }
            for (int p = 0; p < prefix_len; ++p) idx[p] = p;
            const int K = std::min(K_prefix, prefix_len);
            if (K < prefix_len)
                std::nth_element(idx.begin(), idx.begin() + K, idx.end(),
                                 [&](int x, int y) { return pooled[x] > pooled[y]; });
            for (int k = 0; k < K; ++k) keep.insert(idx[k]);     // per-head top-K over prefix
        }
    }
    return r;
}

// Ada-KV (Feng et al. NeurIPS 2025): Algorithm 1 inside Ada-SnapKV (Algorithm 2).
// A total budget B = K_nominal·H is split across heads by how many of each head's weights
// land in the global top-B (B_i = f_i). Max-pool kernel 7 before ranking, safeguard
// B_i* = α·B_i + (1-α)·(B/H) with α = 0.2, then per-head top-B_i*.
// Differences from the paper: no reserved observation window (winsize = 0), Algorithm 1
// runs per layer instead of over all (layer, head) pairs, and the union over heads is
// capped at K_nominal per layer. cap.n_head is the query-head count, n_kv_heads is unused.
static PolicyResult policy_adakv(const AttnCapture & cap, int K_nominal,
                                  int n_kv_heads,
                                  float alpha_safeguard = 0.2f) {
    PolicyResult r;
    if (cap.per_layer.empty()) return r;
    const int n_query_heads = cap.n_head;
    const int n_kv          = cap.n_kv;
    (void)n_kv_heads;
    for (const auto & [layer, attn] : cap.per_layer) {
        r.n_layers = std::max(r.n_layers, layer + 1);
    }
    r.per_layer_keep.resize(r.n_layers);

    // Max-pool kernel size 7 (SnapKV convention). Radius = 3 (symmetric).
    const int pool_radius = 3;

    std::vector<float> pooled((size_t)n_query_heads * (size_t)n_kv);
    std::vector<std::pair<float, int>> all_weights;  // (weight, h*n_kv+p)
    std::vector<int> B_per_head(n_query_heads, 0);

    for (const auto & [layer, attn] : cap.per_layer) {
        // Max-pool each head's attention vector with kernel size 7
        for (int h = 0; h < n_query_heads; ++h) {
            const float * a = attn.data() + (size_t)h * n_kv;
            float * out = pooled.data() + (size_t)h * n_kv;
            for (int p = 0; p < n_kv; ++p) {
                int lo = std::max(0, p - pool_radius);
                int hi = std::min(n_kv - 1, p + pool_radius);
                float m = a[lo];
                for (int q = lo + 1; q <= hi; ++q) if (a[q] > m) m = a[q];
                out[p] = m;
            }
        }

        // Algorithm 1 step 1+2: concatenate and select top-(K_nominal·H)
        all_weights.clear();
        all_weights.reserve((size_t)n_query_heads * (size_t)n_kv);
        for (int h = 0; h < n_query_heads; ++h) {
            const float * a = pooled.data() + (size_t)h * n_kv;
            for (int p = 0; p < n_kv; ++p) {
                all_weights.emplace_back(a[p], h * n_kv + p);
            }
        }
        const int total_budget = std::min<int>((long long)K_nominal * n_query_heads,
                                                (long long)all_weights.size());
        if (total_budget > 0 && total_budget < (int)all_weights.size()) {
            std::nth_element(all_weights.begin(),
                             all_weights.begin() + total_budget,
                             all_weights.end(),
                             [](const std::pair<float,int>& x,
                                const std::pair<float,int>& y) {
                                 return x.first > y.first;
                             });
        }

        // Algorithm 1 step 3+4: count selections per head, f_i = B_i
        std::fill(B_per_head.begin(), B_per_head.end(), 0);
        for (int k = 0; k < total_budget; ++k) {
            int head_idx = all_weights[k].second / n_kv;
            if (head_idx >= 0 && head_idx < n_query_heads) B_per_head[head_idx]++;
        }

        // Safeguard: B_i* = α · B_i + (1-α) · K_nominal
        for (int h = 0; h < n_query_heads; ++h) {
            int B_i = (int)std::lround(alpha_safeguard * (float)B_per_head[h]
                                       + (1.0f - alpha_safeguard) * (float)K_nominal);
            if (B_i < 1)    B_i = 1;
            if (B_i > n_kv) B_i = n_kv;
            B_per_head[h] = B_i;
        }

        // Per-head top-B_i* on the raw (un-pooled) attention, the same scoring as v1/tova,
        // so the comparison isolates the budget allocation.
        std::unordered_set<int> kept;
        for (int h = 0; h < n_query_heads; ++h) {
            const float * a = attn.data() + (size_t)h * n_kv;
            const int K_h = B_per_head[h];
            std::vector<int> idx(n_kv);
            std::iota(idx.begin(), idx.end(), 0);
            std::nth_element(idx.begin(), idx.begin() + K_h, idx.end(),
                             [&](int x, int y) { return a[x] > a[y]; });
            for (int i = 0; i < K_h; ++i) kept.insert(idx[i]);
        }

        // Cap the union at K_nominal positions, keeping the highest mean attention over heads.
        if ((int)kept.size() > K_nominal) {
            std::vector<float> mean_a(n_kv, 0.0f);
            const float inv_H = 1.0f / (float)n_query_heads;
            for (int h = 0; h < n_query_heads; ++h) {
                const float * a = attn.data() + (size_t)h * n_kv;
                for (int p = 0; p < n_kv; ++p) mean_a[p] += a[p] * inv_H;
            }
            std::vector<int> cand(kept.begin(), kept.end());
            std::nth_element(cand.begin(), cand.begin() + K_nominal, cand.end(),
                             [&](int x, int y) { return mean_a[x] > mean_a[y]; });
            kept.clear();
            for (int i = 0; i < K_nominal; ++i) kept.insert(cand[i]);
        }

        r.per_layer_keep[layer] = std::move(kept);
    }
    return r;
}

// Gate mode for the per-head budget loop: "v1" uses m_h, "v2" uses s_h = m_h + 0.5*m'_h.
static std::string g_gate_mode = "v1";

static PolicyResult policy_v1(const AttnCapture & cap, int K_nominal, int n_kv_heads) {
    PolicyResult r;
    if (cap.per_layer.empty()) return r;
    const int n_query_heads = cap.n_head;
    const int n_kv = cap.n_kv;
    (void)n_kv_heads;  // kept as parameter for future per-kv-head expansion
    r.n_layers = 0;
    for (const auto & [layer, attn] : cap.per_layer) {
        r.n_layers = std::max(r.n_layers, layer + 1);
    }
    r.per_layer_keep.resize(r.n_layers);

    for (const auto & [layer, attn] : cap.per_layer) {
        std::unordered_set<int> kept;
        for (int h = 0; h < n_query_heads; ++h) {
            const float * a = attn.data() + (size_t)h * n_kv;
            // Top-1 and (for v2 gate) top-2 attention values for this head.
            float max_a = 0.0f, max2_a = 0.0f;
            for (int p = 0; p < n_kv; ++p) {
                float v = a[p];
                if (v > max_a)       { max2_a = max_a; max_a = v; }
                else if (v > max2_a) { max2_a = v; }
            }
            // Gate: m_h (v1) or m_h + 0.5*m'_h (v2).
            // v1 breakpoints (0.4, 0.8) calibrated to single-peak distribution.
            // v2 breakpoints (0.5, 1.0) calibrated to top-1+0.5*top-2 ranging ~[0, 1.5].
            float gate_in, brk_lo, brk_hi;
            if (g_gate_mode == "v2") {
                gate_in = max_a + 0.5f * max2_a;
                brk_lo = 0.5f;
                brk_hi = 1.0f;
            } else {
                gate_in = max_a;
                brk_lo = 0.4f;
                brk_hi = 0.8f;
            }
            float norm = (gate_in - brk_lo) / (brk_hi - brk_lo);
            if (norm < 0.0f) norm = 0.0f;
            if (norm > 1.0f) norm = 1.0f;
            float mult = 1.3f - 0.6f * norm;
            int K_h = (int)std::lround(K_nominal * mult);
            if (K_h < 1) K_h = 1;
            if (K_h > n_kv) K_h = n_kv;

            // top-K_h positions by a[p]
            std::vector<int> idx(n_kv);
            std::iota(idx.begin(), idx.end(), 0);
            std::nth_element(idx.begin(), idx.begin() + K_h, idx.end(),
                             [&](int x, int y) { return a[x] > a[y]; });
            for (int i = 0; i < K_h; ++i) kept.insert(idx[i]);
        }
        r.per_layer_keep[layer] = std::move(kept);
    }
    return r;
}

// endurkv_adaptive: policy_v1's per-head budget K_h = round(K_eff·μ_head(max_a[h])),
// μ_head(x) = 1.3 - 0.6·clip((x-0.4)/0.4, 0, 1), with K_eff set at run time by a thermal
// controller. The worst sensor threat (linear from warn to crit, clipped to [0, 1], limits
// from THROTTLE_TRIGGER_EMPIRICAL.md) gives μ_thermal = 1.3 - 0.6·threat and
// K_eff = max(32, round(K_nominal·μ_thermal)). Watchdog and memory gate come from the
// launcher. Zones are polled at most every poll_interval_s, on each policy call.
// BCL current is not read because its sysfs paths are not always readable from this UID.
struct ThermalState {
    float ddr_c     = 0.0f;   // current DDR temp °C
    float skin_c    = 0.0f;   // shell_front temp °C
    float battery_c = 0.0f;   // battery temp °C
    float cpu_big_c = 0.0f;   // cpu-1-0-0 temp °C
    int   bcl_current_ma = 0; // battery current (BCL), 0 when unreachable
    int64_t poll_us = 0;      // last poll wall_us
};

class AdaptiveKController {
public:
    ThermalState last;
    float poll_interval_s = 2.0f;

    // Linear threat in [0, 1] between warn and crit (limits are set in mu_thermal).
    float threat_normalized(float t, float warn, float crit) const {
        if (crit <= warn) return 0.0f;
        float x = (t - warn) / (crit - warn);
        return x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x);
    }

    float mu_thermal() const {
        // Worst-of-N threat.
        float threat = 0.0f;
        threat = std::max(threat, threat_normalized(last.ddr_c,     51.7f, 56.7f));
        threat = std::max(threat, threat_normalized(last.cpu_big_c, 56.2f, 61.2f));
        threat = std::max(threat, threat_normalized(last.skin_c,    37.7f, 42.7f));
        threat = std::max(threat, threat_normalized(last.battery_c, 36.8f, 39.8f));
        // threat 0 gives mu 1.3, threat 1 gives mu 0.7
        return 1.3f - 0.6f * threat;
    }

    int K_effective(int K_nominal) const {
        int K = (int)std::lround((double)K_nominal * (double)mu_thermal());
        if (K < 32) K = 32; // floor
        return K;
    }

    // Read /sys/class/thermal/thermal_zoneN/temp in °C (sysfs reports millidegrees).
    // Returns NaN on failure so the previous reading in `last` is kept.
    static float read_zone_c(int zone) {
        char path[128];
        std::snprintf(path, sizeof(path), "/sys/class/thermal/thermal_zone%d/temp", zone);
        std::ifstream f(path);
        if (!f) return std::nanf("");
        long mC = 0;
        if (!(f >> mC)) return std::nanf("");
        return (float)mC / 1000.0f;
    }

    // Refresh the thermal state. Rate-limited by `poll_interval_s`.
    void update() {
        const int64_t now = ggml_time_us();
        if (last.poll_us != 0 &&
            (now - last.poll_us) < (int64_t)(poll_interval_s * 1e6f)) {
            return; // recent reading is fine
        }
        float ddr  = read_zone_c(47);
        float cpu  = read_zone_c(24);
        float skin = read_zone_c(60);
        float batt = read_zone_c(94);
        if (std::isfinite(ddr))  last.ddr_c     = ddr;
        if (std::isfinite(cpu))  last.cpu_big_c = cpu;
        if (std::isfinite(skin)) last.skin_c    = skin;
        if (std::isfinite(batt)) last.battery_c = batt;
        // BCL current is not reliably readable from this binary, leave 0.
        last.bcl_current_ma = 0;
        last.poll_us = now;
    }
};

// Process-wide adaptive-K controller. Shared across the prefill and decode
// dispatch sites so the poll-interval rate-limiter accumulates state.
static AdaptiveKController g_adaptive_K;

// policy_v1_adaptive: policy_v1 with K_nominal scaled by μ_thermal, so the cache grows
// when the phone is cool and shrinks under thermal pressure.
static PolicyResult policy_v1_adaptive(const AttnCapture & cap, int K_nominal, int n_kv_heads) {
    g_adaptive_K.update();
    const int K_eff = g_adaptive_K.K_effective(K_nominal);
    return policy_v1(cap, K_eff, n_kv_heads);
}

// Canonical TOVA (Oren et al. ACL 2024, arXiv 2401.06104, Algorithm 1), per-layer variant:
// keep the top-K positions by the last query's attention averaged over heads,
// mean_a[p] = (1/H) · Σ_h softmax(Q_h^{last} · K_p). Use it in generation-mode evals
// (NIAH, qasper, hotpotqa), where the last query is the real last token, not in chunked PPL.
static PolicyResult policy_tova_canonical(const AttnCapture & cap, int K_nominal, int /*n_kv_heads*/) {
    PolicyResult r;
    if (cap.per_layer.empty()) return r;
    const int n_query_heads = cap.n_head;
    const int n_kv = cap.n_kv;
    for (const auto & [layer, attn] : cap.per_layer) {
        r.n_layers = std::max(r.n_layers, layer + 1);
    }
    r.per_layer_keep.resize(r.n_layers);
    std::vector<float> mean_a(n_kv);
    for (const auto & [layer, attn] : cap.per_layer) {
        std::fill(mean_a.begin(), mean_a.end(), 0.0f);
        const float inv_H = 1.0f / (float)n_query_heads;
        for (int h = 0; h < n_query_heads; ++h) {
            const float * a = attn.data() + (size_t)h * n_kv;
            for (int p = 0; p < n_kv; ++p) mean_a[p] += a[p] * inv_H;
        }
        std::vector<int> idx(n_kv); std::iota(idx.begin(), idx.end(), 0);
        int K = std::min(K_nominal, n_kv);
        if (K > 0 && K < n_kv) {
            std::nth_element(idx.begin(), idx.begin() + K, idx.end(),
                             [&](int x, int y) { return mean_a[x] > mean_a[y]; });
        }
        std::unordered_set<int> kept;
        for (int i = 0; i < K; ++i) kept.insert(idx[i]);
        r.per_layer_keep[layer] = std::move(kept);
    }
    return r;
}

// TOVA with a recent window: same per-layer mean-attention score, plus a 50/50 recent+heavy
// split as in policy_h2o. In chunked prefill the last query ends a prefill chunk, so pure
// top-K drops local context the next eval chunk needs. Sinks are added by protect_sink().
// The paper drops one token per step, this keeps the top-K_nominal interface of v1/H2O.
static PolicyResult policy_tova(const AttnCapture & cap, int K_nominal, int /*n_kv_heads*/) {
    PolicyResult r;
    if (cap.per_layer.empty()) return r;
    const int n_query_heads = cap.n_head;
    const int n_kv = cap.n_kv;
    for (const auto & [layer, attn] : cap.per_layer) {
        r.n_layers = std::max(r.n_layers, layer + 1);
    }
    r.per_layer_keep.resize(r.n_layers);

    // 50/50 recent+heavy split of the K_nominal budget, as in policy_h2o.
    int n_recent = K_nominal / 2;
    int n_heavy  = K_nominal - n_recent;
    if (n_recent > n_kv) n_recent = n_kv;
    if (n_heavy  > n_kv) n_heavy  = n_kv;
    const int recent_start = std::max(0, n_kv - n_recent);

    std::vector<float> mean_a(n_kv);
    for (const auto & [layer, attn] : cap.per_layer) {
        // 1. Average attention across heads for this layer (paper §4, App A).
        std::fill(mean_a.begin(), mean_a.end(), 0.0f);
        const float inv_H = 1.0f / (float)n_query_heads;
        for (int h = 0; h < n_query_heads; ++h) {
            const float * a = attn.data() + (size_t)h * n_kv;
            for (int p = 0; p < n_kv; ++p) mean_a[p] += a[p] * inv_H;
        }

        std::unordered_set<int> kept;

        // (a) Always keep the last n_recent positions for this layer.
        for (int p = recent_start; p < n_kv; ++p) kept.insert(p);

        // (b) Top-n_heavy by mean_a among positions outside the recent window.
        if (recent_start > 0 && n_heavy > 0) {
            std::vector<int> idx;
            idx.reserve(recent_start);
            for (int p = 0; p < recent_start; ++p) idx.push_back(p);
            int K_h = std::min(n_heavy, (int)idx.size());
            if (K_h > 0) {
                std::nth_element(idx.begin(), idx.begin() + K_h, idx.end(),
                                 [&](int x, int y) { return mean_a[x] > mean_a[y]; });
                for (int i = 0; i < K_h; ++i) kept.insert(idx[i]);
            }
        }

        r.per_layer_keep[layer] = std::move(kept);
    }
    return r;
}

// H2O (Heavy-Hitter Oracle, Zhang et al. NeurIPS 2023, Algorithm 1, §4.2). Per (layer, head)
// keep n_heavy positions by accumulated attention plus the n_recent most recent ones. The
// reference impl (FMInference/H2O) uses heavy_ratio = recent_ratio = 0.1, a 50/50 split:
// n_recent = K_nominal/2, n_heavy = K_nominal - n_recent. Without the recent window, new
// decode tokens have near-zero sums and are evicted at once.
// `state` must be update(cap)-ed once per forward pass. Heads are OR'd within a layer and
// sinks are added by protect_sink().
static PolicyResult policy_h2o(const AttnCapture & cap,
                                const H2OState & state,
                                int K_nominal, int /*n_kv_heads*/) {
    PolicyResult r;
    if (cap.per_layer.empty()) return r;
    const int n_query_heads = cap.n_head;
    const int n_kv = cap.n_kv;
    for (const auto & [layer, attn] : cap.per_layer) {
        r.n_layers = std::max(r.n_layers, layer + 1);
    }
    r.per_layer_keep.resize(r.n_layers);

    // Canonical 50/50 recent+heavy split of the K_nominal budget (paper default).
    int n_recent = K_nominal / 2;
    int n_heavy  = K_nominal - n_recent;
    if (n_recent > n_kv) n_recent = n_kv;
    if (n_heavy  > n_kv) n_heavy  = n_kv;

    // First position that is part of the "recent window".
    const int recent_start = std::max(0, n_kv - n_recent);

    for (const auto & [layer, attn] : cap.per_layer) {
        std::unordered_set<int> kept;

        // (b) Always keep the last n_recent positions for this layer.
        for (int p = recent_start; p < n_kv; ++p) kept.insert(p);

        auto it = state.sum_attn.find(layer);
        if (it == state.sum_attn.end()) {
            // Cold start, no accumulator yet: pick heavy positions from this step's attention.
            for (int h = 0; h < n_query_heads; ++h) {
                const float * a = attn.data() + (size_t)h * n_kv;
                // Heavy candidates are positions outside the recent window.
                const int n_cand = recent_start;
                if (n_cand <= 0 || n_heavy <= 0) continue;
                std::vector<int> idx; idx.reserve(n_cand);
                for (int p = 0; p < recent_start; ++p) idx.push_back(p);
                const int K_h = std::min(n_heavy, (int)idx.size());
                if (K_h <= 0) continue;
                std::nth_element(idx.begin(), idx.begin() + K_h, idx.end(),
                                 [&](int x, int y) { return a[x] > a[y]; });
                for (int i = 0; i < K_h; ++i) kept.insert(idx[i]);
            }
        } else {
            const auto & sum = it->second;
            // Per-head top-n_heavy by cumulative attention sum among positions
            // outside the recent window (Aggregate-OR across heads).
            for (int h = 0; h < n_query_heads; ++h) {
                const double * s = sum.data() + (size_t)h * n_kv;
                const int n_cand = recent_start;
                if (n_cand <= 0 || n_heavy <= 0) continue;
                std::vector<int> idx; idx.reserve(n_cand);
                for (int p = 0; p < recent_start; ++p) idx.push_back(p);
                const int K_h = std::min(n_heavy, (int)idx.size());
                if (K_h <= 0) continue;
                std::nth_element(idx.begin(), idx.begin() + K_h, idx.end(),
                                 [&](int x, int y) { return s[x] > s[y]; });
                for (int i = 0; i < K_h; ++i) kept.insert(idx[i]);
            }
        }
        r.per_layer_keep[layer] = std::move(kept);
    }
    return r;
}

static PolicyResult policy_pyramid(const AttnCapture & cap, int K_nominal, int n_layers_total) {
    PolicyResult r;
    if (cap.per_layer.empty()) return r;
    const int n_query_heads = cap.n_head;
    const int n_kv = cap.n_kv;
    for (const auto & [layer, attn] : cap.per_layer) {
        r.n_layers = std::max(r.n_layers, layer + 1);
    }
    r.per_layer_keep.resize(r.n_layers);

    const float ratio = 0.5f;
    for (const auto & [layer, attn] : cap.per_layer) {
        // Linear K_max -> K_min by depth.
        float K_max = K_nominal * (1.0f + ratio);
        float K_min = K_nominal * (1.0f - ratio);
        float frac = n_layers_total > 1 ? (float)layer / (float)(n_layers_total - 1) : 0.0f;
        int K_layer = (int)std::lround(K_max - (K_max - K_min) * frac);
        if (K_layer < 1) K_layer = 1;
        if (K_layer > n_kv) K_layer = n_kv;

        std::unordered_set<int> kept;
        for (int h = 0; h < n_query_heads; ++h) {
            const float * a = attn.data() + (size_t)h * n_kv;
            std::vector<int> idx(n_kv);
            std::iota(idx.begin(), idx.end(), 0);
            std::nth_element(idx.begin(), idx.begin() + K_layer, idx.end(),
                             [&](int x, int y) { return a[x] > a[y]; });
            for (int i = 0; i < K_layer; ++i) kept.insert(idx[i]);
        }
        r.per_layer_keep[layer] = std::move(kept);
    }
    return r;
}

// StreamingLLM (Xiao et al., ICLR 2024, arXiv:2309.17453): keep the first n_sink tokens and
// the last K_nominal - n_sink, drop the middle. Ignores attention, so it works with FA on.
static PolicyResult policy_streamingllm(const AttnCapture & cap, int K_nominal,
                                         int n_sink) {
    PolicyResult r;
    if (cap.per_layer.empty()) return r;
    const int n_kv = cap.n_kv;
    int n_recent = K_nominal - n_sink;
    if (n_recent < 1) n_recent = std::max(1, K_nominal - 1);
    if (n_sink < 0) n_sink = 0;

    for (const auto & [layer, attn] : cap.per_layer) {
        r.n_layers = std::max(r.n_layers, layer + 1);
    }
    r.per_layer_keep.resize(r.n_layers);

    std::unordered_set<int> keep_set;
    // Sink positions [0, n_sink).
    for (int p = 0; p < std::min(n_sink, n_kv); ++p) keep_set.insert(p);
    // Recency window [n_kv - n_recent, n_kv).
    int recent_start = std::max(n_sink, n_kv - n_recent);
    for (int p = recent_start; p < n_kv; ++p) keep_set.insert(p);

    for (const auto & [layer, _attn] : cap.per_layer) {
        r.per_layer_keep[layer] = keep_set;
    }
    return r;
}

// v1_entropy: confidence-weighted scoring. Per layer and head, w_h = exp(-H_h), where H_h
// is the entropy of the last query's attention (near 1 for sharp heads, near 0 for diffuse).
// score(p) = Σ_h w_h * a_h(p), then the recent+heavy split used by H2O and TOVA.
// eps guards log(0), and H is clamped to log(n_kv)+1 so w_h does not underflow to 0.
static PolicyResult policy_v1_entropy(const AttnCapture & cap,
                                       int K_nominal, int /*n_kv_heads*/) {
    PolicyResult r;
    if (cap.per_layer.empty()) return r;
    const int n_query_heads = cap.n_head;
    const int n_kv = cap.n_kv;
    for (const auto & [layer, attn] : cap.per_layer) {
        r.n_layers = std::max(r.n_layers, layer + 1);
    }
    r.per_layer_keep.resize(r.n_layers);

    int n_recent = K_nominal / 2;
    int n_heavy  = K_nominal - n_recent;
    if (n_recent > n_kv) n_recent = n_kv;
    if (n_heavy  > n_kv) n_heavy  = n_kv;
    const int recent_start = std::max(0, n_kv - n_recent);

    const float eps = 1e-12f;
    const float H_cap = std::log((float)std::max(1, n_kv)) + 1.0f;

    std::vector<float> w(n_query_heads);
    std::vector<float> score(n_kv);
    for (const auto & [layer, attn] : cap.per_layer) {
        // 1. Per-head entropy and confidence weight w_h = exp(-H_h).
        for (int h = 0; h < n_query_heads; ++h) {
            const float * a = attn.data() + (size_t)h * n_kv;
            float H_h = 0.0f;
            for (int p = 0; p < n_kv; ++p) {
                float ap = a[p];
                if (ap > eps) H_h -= ap * std::log(ap);
            }
            if (H_h > H_cap) H_h = H_cap;  // prevent w_h underflow on diffuse heads
            w[h] = std::exp(-H_h);
        }
        // 2. Confidence-weighted per-position score.
        std::fill(score.begin(), score.end(), 0.0f);
        for (int h = 0; h < n_query_heads; ++h) {
            const float * a = attn.data() + (size_t)h * n_kv;
            const float wh = w[h];
            for (int p = 0; p < n_kv; ++p) score[p] += wh * a[p];
        }

        std::unordered_set<int> kept;
        // 3. Always keep the recent window (see the chunked-prefill note on policy_tova).
        for (int p = recent_start; p < n_kv; ++p) kept.insert(p);
        // 4. Top n_heavy positions outside the recent window.
        if (recent_start > 0 && n_heavy > 0) {
            std::vector<int> idx; idx.reserve(recent_start);
            for (int p = 0; p < recent_start; ++p) idx.push_back(p);
            int K_h = std::min(n_heavy, (int)idx.size());
            if (K_h > 0) {
                std::nth_element(idx.begin(), idx.begin() + K_h, idx.end(),
                                 [&](int x, int y) { return score[x] > score[y]; });
                for (int i = 0; i < K_h; ++i) kept.insert(idx[i]);
            }
        }
        r.per_layer_keep[layer] = std::move(kept);
    }
    return r;
}

// AttnHistory: ring buffer of mean-across-heads attention vectors per layer.
// Used by v1_predictive to compute the temporal trajectory (slope) of each
// position's attention over the last W decode steps.
struct AttnHistory {
    int W = 8;
    int n_kv_per_layer = 0;
    // ring[layer]: W * n_kv flat. ring[layer][slot*n_kv + p] is the layer's
    // mean-across-heads attention at position p, captured at write slot `slot`.
    std::map<int, std::vector<float>> ring;
    std::map<int, int> count;  // samples seen per layer (capped at W)
    std::map<int, int> head;   // next write slot (per layer, mod W)
    std::map<int, int> n_kv_layer; // last known n_kv per layer

    void clear() { ring.clear(); count.clear(); head.clear(); n_kv_layer.clear(); }

    // Push one capture: append mean-across-heads attention to each layer's ring.
    void push(const AttnCapture & cap) {
        if (cap.per_layer.empty()) return;
        const int H = cap.n_head;
        const int n_kv = cap.n_kv;
        for (const auto & [layer, attn] : cap.per_layer) {
            auto & buf = ring[layer];
            int prev_n_kv = n_kv_layer.count(layer) ? n_kv_layer[layer] : 0;
            // Reset the ring on cold start or when n_kv shrinks (eviction moved positions,
            // old slopes are meaningless). When n_kv grows by a decode token, pad each slot
            // and keep count/head so history accumulates across steps.
            if (prev_n_kv == 0 || n_kv < prev_n_kv) {
                buf.assign((size_t)W * (size_t)n_kv, 0.0f);
                count[layer] = 0;
                head[layer]  = 0;
                n_kv_layer[layer] = n_kv;
            } else if (n_kv > prev_n_kv) {
                // Grow each of the W slot rows from prev_n_kv to n_kv floats, zero-padding.
                std::vector<float> newbuf((size_t)W * (size_t)n_kv, 0.0f);
                for (int s = 0; s < W; ++s) {
                    const float * src = buf.data() + (size_t)s * (size_t)prev_n_kv;
                    float * dst_row = newbuf.data() + (size_t)s * (size_t)n_kv;
                    std::copy(src, src + prev_n_kv, dst_row);
                    // tail [prev_n_kv, n_kv) already zero-initialized
                }
                buf = std::move(newbuf);
                n_kv_layer[layer] = n_kv;
                // count[layer] and head[layer] preserved.
            }
            // If n_kv == prev_n_kv, nothing to resize.

            int slot = head[layer];
            float * dst = buf.data() + (size_t)slot * (size_t)n_kv;
            const float inv_H = 1.0f / (float)H;
            std::fill(dst, dst + n_kv, 0.0f);
            for (int h = 0; h < H; ++h) {
                const float * a = attn.data() + (size_t)h * (size_t)n_kv;
                for (int p = 0; p < n_kv; ++p) dst[p] += a[p] * inv_H;
            }
            head[layer]  = (slot + 1) % W;
            count[layer] = std::min(count[layer] + 1, W);
        }
    }

    // Per-position slope numerator (the denominator is constant per window).
    // Walks the ring oldest to newest. Returns -INFINITY when N<3 (too few samples).
    float slope(int layer, int p) const {
        auto itc = count.find(layer);
        if (itc == count.end()) return -INFINITY;
        int N = itc->second;
        if (N < 3) return -INFINITY;
        auto itr = ring.find(layer);
        if (itr == ring.end()) return -INFINITY;
        auto itn = n_kv_layer.find(layer);
        if (itn == n_kv_layer.end()) return -INFINITY;
        const int n_kv = itn->second;
        if (p < 0 || p >= n_kv) return -INFINITY;
        const float * buf = itr->second.data();
        int hd = head.at(layer);
        // Oldest sample slot = (hd - N + W) % W (equals hd when N == W).
        int oldest = (hd - N + W) % W;
        double a_bar = 0.0;
        double t_bar = (N - 1) / 2.0;
        for (int i = 0; i < N; ++i) {
            int slot = (oldest + i) % W;
            a_bar += buf[(size_t)slot * (size_t)n_kv + p];
        }
        a_bar /= (double)N;
        double num = 0.0;
        for (int i = 0; i < N; ++i) {
            int slot = (oldest + i) % W;
            double a_t = buf[(size_t)slot * (size_t)n_kv + p];
            num += ((double)i - t_bar) * (a_t - a_bar);
        }
        return (float)num;
    }
};

// v1_predictive: per position, the regression slope of the last W mean-over-heads attention
// samples. Keep set: sinks (protect_sink), the recent window [n_kv - recent, n_kv) since
// newcomers have too few samples for a slope, and the top slopes outside it, tie-broken by
// last-step attention. With too few samples every slope is -INF and this is recency only.
static PolicyResult policy_v1_predictive(const AttnCapture & cap,
                                          const AttnHistory & hist,
                                          int K_nominal, int n_sink) {
    PolicyResult r;
    if (cap.per_layer.empty()) return r;
    const int n_query_heads = cap.n_head;
    const int n_kv = cap.n_kv;
    for (const auto & [layer, attn] : cap.per_layer) {
        r.n_layers = std::max(r.n_layers, layer + 1);
    }
    r.per_layer_keep.resize(r.n_layers);

    if (n_sink < 0) n_sink = 0;
    int n_recent = K_nominal / 2;
    if (n_recent > n_kv) n_recent = n_kv;
    int n_trend = K_nominal - n_sink - n_recent;
    if (n_trend < 0) n_trend = 0;
    const int recent_start = std::max(n_sink, n_kv - n_recent);

    std::vector<float> last_a(n_kv);
    for (const auto & [layer, attn] : cap.per_layer) {
        std::unordered_set<int> kept;
        // Recency window - always kept.
        for (int p = recent_start; p < n_kv; ++p) kept.insert(p);

        // Candidates = positions in [n_sink, recent_start). Sink is added by
        // protect_sink() externally.
        const int cand_lo = std::min(n_sink, recent_start);
        const int cand_hi = recent_start;
        if (cand_hi > cand_lo && n_trend > 0) {
            // Build last-step mean attention (for tie-break).
            std::fill(last_a.begin(), last_a.end(), 0.0f);
            const float inv_H = 1.0f / (float)n_query_heads;
            for (int h = 0; h < n_query_heads; ++h) {
                const float * a = attn.data() + (size_t)h * n_kv;
                for (int p = 0; p < n_kv; ++p) last_a[p] += a[p] * inv_H;
            }
            std::vector<int> idx; idx.reserve(cand_hi - cand_lo);
            for (int p = cand_lo; p < cand_hi; ++p) idx.push_back(p);
            std::vector<float> sl(idx.size());
            for (size_t i = 0; i < idx.size(); ++i) sl[i] = hist.slope(layer, idx[i]);
            int K_h = std::min(n_trend, (int)idx.size());
            if (K_h > 0) {
                // Rank by slope desc, tie-break by last_a desc.
                std::partial_sort(idx.begin(), idx.begin() + K_h, idx.end(),
                                  [&](int x, int y) {
                                      // Map original position back to its sl index via cand_lo offset.
                                      float sx = sl[x - cand_lo], sy = sl[y - cand_lo];
                                      if (sx != sy) return sx > sy;
                                      return last_a[x] > last_a[y];
                                  });
                for (int i = 0; i < K_h; ++i) kept.insert(idx[i]);
            }
        }
        r.per_layer_keep[layer] = std::move(kept);
    }
    return r;
}

// True when the K cache is quantized. llama.cpp's K-shift graph (RoPE re-rotation after
// seq_add) does not support quantized K and crashes, so seq_add is skipped and positions
// stay sparse. FA-on attention is still correct since each cell carries its own RoPE'd K.
static bool g_k_is_quantized = false;

// True for a multimodal (M-RoPE) cache. Eviction must then address cells, not positions,
// because all tokens of an image share one dim-0 position. llama_memory_seq_add() also
// aborts on M-RoPE caches (it asserts n_pos_per_embd()==1), so position compaction is
// skipped, which is safe for the same reason as with g_k_is_quantized.
static bool g_evict_by_cell_index = false;

// Tiered decode-time eviction for v1_FA² (v1_fa2). After the state swap the cache is:
//   [0, n_anchored)         prefill anchors chosen by v1's spread gate
//   [n_anchored, n_kv - R)  older decode tokens (eviction candidates)
//   [n_kv - R, n_kv)        recent decode tokens (recent budget R)
// The middle band is dropped once n_kv exceeds the budget by 25%. Anchors are never
// dropped. Returns positions evicted.
static int apply_tiered_decode_eviction(llama_context * ctx, int n_kv,
                                        int n_anchored, int recent_budget) {
    // M-RoPE (VL): position ranges cannot address image cells and seq_add aborts.
    if (g_evict_by_cell_index) return 0;
    int total_budget = n_anchored + recent_budget;
    // Hysteresis: trigger when cache has accumulated 25% beyond the budget.
    if (n_kv <= total_budget + total_budget / 4) return 0;
    int drop_start = n_anchored;
    int drop_end   = n_kv - recent_budget;   // exclusive
    if (drop_end <= drop_start) return 0;
    llama_memory_t mem = llama_get_memory(ctx);
    llama_memory_seq_rm(mem, 0, drop_start, drop_end);
    int shift = drop_end - drop_start;
    if (!g_k_is_quantized) {
        // Compact positions only when K is f16 - seq_add triggers a K-shift
        // graph that doesn't support quantized K.
        llama_memory_seq_add(mem, 0, drop_end, n_kv, -shift);
    }
    return shift;
}

// Recency-only decode eviction for v1_fa during FA-on decode, where attention scores are
// not available. Keeps [0, n_sink) and [n_kv - n_recent, n_kv). Runs only when
// n_kv > 1.5*(n_sink + n_recent), to amortize seq_rm and compaction. Returns positions evicted.
static int apply_recency_decode_eviction(llama_context * ctx, int n_kv,
                                          int n_sink, int n_recent) {
    // M-RoPE (VL): position ranges cannot address image cells and seq_add aborts.
    if (g_evict_by_cell_index) return 0;
    int budget = n_sink + n_recent;
    if (n_kv <= (budget * 3) / 2) return 0;
    int drop_start = n_sink;
    int drop_end   = n_kv - n_recent;   // exclusive
    if (drop_end <= drop_start) return 0;
    llama_memory_t mem = llama_get_memory(ctx);
    llama_memory_seq_rm(mem, 0, drop_start, drop_end);
    // Shift the surviving recent positions down so positions stay dense. Skipped for
    // quantized K, since the K-shift RoPE rotation does not support it.
    int shift = drop_end - drop_start;
    if (!g_k_is_quantized) {
        llama_memory_seq_add(mem, 0, drop_end, n_kv, -shift);
    }
    return shift;
}

// Apply the policy: seq_rm every position that no layer keeps (sequence-level).
// Returns how many positions were evicted this step.
static int apply_eviction(llama_context * ctx, const PolicyResult & pol, int n_kv) {
    if (pol.per_layer_keep.empty()) return 0;
    int n_keep = (int)pol.per_layer_keep.size();
    // A position is evicted iff no layer keeps it.
    std::vector<uint8_t> keep(n_kv, 0);
    for (int l = 0; l < n_keep; ++l) {
        for (int p : pol.per_layer_keep[l]) if (p >= 0 && p < n_kv) keep[p] = 1;
    }
    llama_memory_t mem = llama_get_memory(ctx);
    int evicted = 0;
    // On M-RoPE (vision) caches all tokens of an image share one dim-0 position, so
    // position-based seq_rm cannot express a per-token mask. Use the fork's cell-index
    // API there. Text caches (cell index == position) use contiguous-run seq_rm.
    if (g_evict_by_cell_index) {
        std::vector<int8_t> keep_i8(keep.begin(), keep.end());
        evicted = (int) llama_endurkv_seq_rm_cells(mem, 0, keep_i8.data(), (uint32_t) n_kv);
    } else {
        // One seq_rm call per contiguous run of evicted positions.
        int p = 0;
        while (p < n_kv) {
            if (keep[p]) { ++p; continue; }
            int start = p;
            while (p < n_kv && !keep[p]) ++p;
            int end = p;  // exclusive
            llama_memory_seq_rm(mem, 0, start, end);
            evicted += (end - start);
        }
    }
    // Diagnostic: shows whether keep[] marks nearly everything or seq_rm fails to shrink the cache.
    {
        int kept = 0; for (int q = 0; q < n_kv; ++q) kept += keep[q];
        int pmin = (int)llama_memory_seq_pos_min(mem, 0);
        int pmax = (int)llama_memory_seq_pos_max(mem, 0);
        std::fprintf(stderr, "[evict-dbg] n_kv=%d kept=%d evicted=%d  post-rm pos=[%d..%d]\n",
                     n_kv, kept, evicted, pmin, pmax);
    }
    return evicted;
}

// Helpers
static int argmax_logits(const float * logits, int n_vocab) {
    int best = 0;
    float bv = logits[0];
    for (int i = 1; i < n_vocab; ++i) if (logits[i] > bv) { bv = logits[i]; best = i; }
    return best;
}

// Repetition penalty (llama.cpp-standard): divide logits of tokens
// that appear in the recent window by `penalty`. Positive logits divide,
// negative logits multiply (standard convention).
static int argmax_with_repetition_penalty(const float * logits, int n_vocab,
                                          const std::vector<int> & recent_tokens,
                                          float penalty) {
    if (penalty == 1.0f || recent_tokens.empty()) {
        return argmax_logits(logits, n_vocab);
    }
    // Make a local copy so we don't mutate caller's logits
    std::vector<float> adj(logits, logits + n_vocab);
    for (int tok : recent_tokens) {
        if (tok < 0 || tok >= n_vocab) continue;
        if (adj[tok] > 0) adj[tok] /= penalty;
        else              adj[tok] *= penalty;
    }
    return argmax_logits(adj.data(), n_vocab);
}

// Standard llama.cpp sampling pipeline:
//   1. repeat_penalty over last N tokens
//   2. logits /= temperature
//   3. top-K filter
//   4. top-P (nucleus) filter
//   5. weighted random sample from the surviving nucleus
// The same for every eviction policy.
static int sample_token(const float * logits_in, int n_vocab,
                        const std::vector<int> & recent_tokens,
                        float repeat_penalty, float temperature,
                        float top_p, int top_k,
                        std::mt19937 & rng) {
    // Copy + repetition penalty
    std::vector<float> logits(logits_in, logits_in + n_vocab);
    if (repeat_penalty != 1.0f) {
        for (int tok : recent_tokens) {
            if (tok < 0 || tok >= n_vocab) continue;
            if (logits[tok] > 0) logits[tok] /= repeat_penalty;
            else                 logits[tok] *= repeat_penalty;
        }
    }
    // Temperature
    if (temperature > 0.0f && temperature != 1.0f) {
        for (int i = 0; i < n_vocab; ++i) logits[i] /= temperature;
    }
    // (id, logit) pairs, sort by logit descending
    std::vector<std::pair<int, float>> pool(n_vocab);
    for (int i = 0; i < n_vocab; ++i) pool[i] = {i, logits[i]};
    // Top-K (partial sort): if top_k <= 0 or > n_vocab, no truncation
    int kk = (top_k > 0 && top_k < n_vocab) ? top_k : n_vocab;
    std::partial_sort(pool.begin(), pool.begin() + kk, pool.end(),
                      [](const auto & a, const auto & b){ return a.second > b.second; });
    pool.resize(kk);
    // Softmax over the top-K pool (numerically stable)
    float lmax = pool[0].second;
    double Z = 0.0;
    std::vector<double> probs(kk);
    for (int i = 0; i < kk; ++i) {
        probs[i] = std::exp((double)pool[i].second - (double)lmax);
        Z += probs[i];
    }
    for (int i = 0; i < kk; ++i) probs[i] /= Z;
    // Top-P: keep tokens whose cumulative prob reaches top_p
    if (top_p > 0.0f && top_p < 1.0f) {
        double cum = 0.0; int cutoff = kk;
        for (int i = 0; i < kk; ++i) { cum += probs[i]; if (cum >= top_p) { cutoff = i + 1; break; } }
        pool.resize(cutoff); probs.resize(cutoff);
        // Renormalize
        double Z2 = 0.0;
        for (auto & p : probs) Z2 += p;
        for (auto & p : probs) p /= Z2;
    }
    // Weighted random sample
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    double r = uni(rng);
    double acc = 0.0;
    for (size_t i = 0; i < probs.size(); ++i) {
        acc += probs[i];
        if (r <= acc) return pool[i].first;
    }
    return pool.back().first;
}

static long read_rss_kb(int pid) {
    char path[64];
    std::snprintf(path, sizeof(path), "/proc/%d/status", pid);
    std::FILE * f = std::fopen(path, "r");
    if (!f) return 0;
    char buf[256]; long rss = 0;
    while (std::fgets(buf, sizeof(buf), f)) {
        if (std::strncmp(buf, "VmRSS:", 6) == 0) {
            std::sscanf(buf + 6, "%ld", &rss);
            break;
        }
    }
    std::fclose(f);
    return rss;
}

// Main
int main(int argc, char ** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) return 1;
    // policy_v1 reads the gate mode from a global.
    g_gate_mode = args.gate_mode;

    std::string prompt;
    if (!read_file(args.prompt_file, prompt)) return 1;

    ggml_backend_load_all();
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = args.n_gpu_layers;
    // --no-extra-bufts disables llama.cpp's weight-repack buffer types (use_extra_bufts,
    // default true). Bonsai-8B (Q1_0) weights are otherwise repacked to the ARM CPU format
    // q1_0_4x8 at load for every tensor, GPU ones included, which crashes the Vulkan queue
    // (vk::DeviceLostError). The model stays Q1_0 either way. Keep the repack for CPU runs,
    // where it is the fast i8mm/dotprod path.
    if (args.no_extra_bufts) {
        mparams.use_extra_bufts = false;
        std::fprintf(stderr, "[eviction_bench] extra buffer types DISABLED (no weight repack)\n");
    }
    llama_model * model = llama_model_load_from_file(args.model.c_str(), mparams);
    if (!model) { std::fprintf(stderr, "model load failed\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int n_layers_total = llama_model_n_layer(model);
    const int n_kv_heads = llama_model_n_head_kv(model);

    int n_neg = llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, true, true);
    int n_prompt = -n_neg;
    if (n_prompt <= 0) { std::fprintf(stderr, "tokenize fail %d\n", n_neg); return 1; }
    std::vector<llama_token> ptoks(n_prompt);
    llama_tokenize(vocab, prompt.c_str(), prompt.size(), ptoks.data(), n_prompt, true, true);

    // Multimodal tokenization, before context creation so ctx_size counts image tokens.
    // The prompt is wrapped in the qwen ChatML template with one media marker per --image.
    // Only eval_mode=gen with vanilla or --fa-on-evict is supported.
#ifdef EVB_HAS_MTMD
    mtmd_context     * vl_ctx    = nullptr;
    mtmd_input_chunks * vl_chunks = nullptr;
    std::vector<mtmd_bitmap *> vl_bitmaps;
    if (!args.mmproj.empty()) {
        if (args.images.empty()) { std::fprintf(stderr, "[vl] --mmproj given but no --image\n"); return 1; }
        const bool pol_ok = (args.policy == "vanilla") || args.fa_on_evict;
        if (args.eval_mode != "gen" || !pol_ok) {
            std::fprintf(stderr, "[vl] multimodal supports eval_mode=gen with vanilla or fa-on-evict only\n");
            return 1;
        }
        mtmd_context_params mp = mtmd_context_params_default();
        mp.use_gpu       = args.mmproj_gpu;
        mp.n_threads     = args.n_threads;
        mp.print_timings = true;
        vl_ctx = mtmd_init_from_file(args.mmproj.c_str(), model, mp);
        if (!vl_ctx) { std::fprintf(stderr, "[vl] mtmd_init_from_file failed: %s\n", args.mmproj.c_str()); return 1; }
        for (const auto & im : args.images) {
            mtmd_bitmap * bm = mtmd_helper_bitmap_init_from_file(vl_ctx, im.c_str());
            if (!bm) { std::fprintf(stderr, "[vl] failed to load image: %s\n", im.c_str()); return 1; }
            vl_bitmaps.push_back(bm);
        }
        std::string marked = "<|im_start|>user\n";
        for (size_t i = 0; i < vl_bitmaps.size(); ++i) marked += mtmd_default_marker();
        marked += "\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
        mtmd_input_text itext { marked.c_str(), /*add_special*/ true, /*parse_special*/ true };
        vl_chunks = mtmd_input_chunks_init();
        int32_t tok_rc = mtmd_tokenize(vl_ctx, vl_chunks, &itext,
                                       (const mtmd_bitmap **) vl_bitmaps.data(), vl_bitmaps.size());
        if (tok_rc != 0) { std::fprintf(stderr, "[vl] mtmd_tokenize rc=%d\n", tok_rc); return 1; }
        // n_prompt now = total KV cells the prefill will occupy (text + image
        // tokens) so ctx sizing, the α-gate's N, and logging stay correct.
        n_prompt = (int) mtmd_helper_get_n_tokens(vl_chunks);
        // M-RoPE cache: eviction must address cells, not positions.
        g_evict_by_cell_index = true;
        std::fprintf(stderr, "[vl] images=%zu chunks=%zu prefill_cells=%d n_pos=%d (cell-index eviction ON)\n",
                     vl_bitmaps.size(), mtmd_input_chunks_size(vl_chunks), n_prompt,
                     (int) mtmd_helper_get_n_pos(vl_chunks));
    }
#else
    if (!args.mmproj.empty()) {
        std::fprintf(stderr, "[vl] this binary was built without libmtmd (EVB_HAS_MTMD off)\n");
        return 1;
    }
#endif

    AttnCapture cap;
    // Parse cache-type strings into ggml types. Supports f16, q8_0, q4_0
    // (most useful subset for KV cache quantization).
    auto parse_cache_type = [](const std::string & s) -> ggml_type {
        if (s == "f16")  return GGML_TYPE_F16;
        if (s == "f32")  return GGML_TYPE_F32;
        if (s == "q8_0") return GGML_TYPE_Q8_0;
        if (s == "q4_0") return GGML_TYPE_Q4_0;
        if (s == "q4_1") return GGML_TYPE_Q4_1;
        if (s == "bf16") return GGML_TYPE_BF16;
        std::fprintf(stderr, "[eviction_bench] WARN unknown cache type %s, falling back to f16\n", s.c_str());
        return GGML_TYPE_F16;
    };

    // An FA-off prefill context needs f16 V (llama.cpp quantizes V only with FA on). A
    // post-swap FA-on context (snapkv_decode) can use the requested V type, so keep it.
    const ggml_type user_type_v = parse_cache_type(args.cache_type_v);
    const ggml_type user_type_k = parse_cache_type(args.cache_type_k);
    llama_context_params cparams = llama_context_default_params();
    cparams.type_k = user_type_k;
    cparams.type_v = user_type_v;
    bool prefill_needs_f16_v = (!args.fa_vanilla || args.policy != "vanilla")
                               && cparams.type_v != GGML_TYPE_F16;
    // Force f16 V only when prefill runs FA-off. Vanilla and --fa-on-evict run FA-on
    // throughout, where quantized V works. Forcing it always made a q8_0 request for
    // Phi-3 at 16K exceed Tegra's ~4 GiB single-allocation limit.
    if (args.policy != "vanilla" && !args.fa_on_evict) {
        cparams.type_v = GGML_TYPE_F16;
    }
    std::fprintf(stderr, "[eviction_bench] KV cache types (prefill): K=%s V=%s (user requested V=%s)\n",
                 args.cache_type_k.c_str(),
                 cparams.type_v == GGML_TYPE_F16 ? "f16" : args.cache_type_v.c_str(),
                 args.cache_type_v.c_str());
    // Track whether K is quantized so eviction helpers skip seq_add (which
    // would trigger an unsupported RoPE shift on quantized K).
    g_k_is_quantized = (cparams.type_k != GGML_TYPE_F16 && cparams.type_k != GGML_TYPE_F32 && cparams.type_k != GGML_TYPE_BF16);
    (void)prefill_needs_f16_v;
    int needed = n_prompt + args.max_tokens + 32;
    cparams.n_ctx = (uint32_t)std::max(args.ctx_size, needed);
    std::fprintf(stderr, "[eviction_bench] ctx_size=%u (prompt=%d, max_tok=%d, policy=%s)\n",
                 cparams.n_ctx, n_prompt, args.max_tokens, args.policy.c_str());
    // n_batch is the logical batch per llama_decode call. Prefill is chunked by hand so
    // n_batch can stay at a GPU-safe ~512 even for 4K+ token prompts.
    // n_ubatch controls per-kernel chunking inside one llama_decode call.
    cparams.n_batch  = (uint32_t)args.n_batch;
    cparams.n_ubatch = (uint32_t)args.n_ubatch;
    cparams.no_perf  = false;
    std::fprintf(stderr, "[eviction_bench] n_batch=%u n_ubatch=%u n_gpu_layers=%d\n",
                 cparams.n_batch, cparams.n_ubatch, args.n_gpu_layers);
    // FA is on for vanilla. Policies that read kq_soft_max need FA off. StreamingLLM evicts
    // by position and KeyDiff (arXiv:2504.15364) scores by key geometry, -cos(mu(K), k_i),
    // so neither reads attention and both run with flash attention and no cb_eval.
    const bool positional_policy = (args.policy == "streamingllm" || args.policy == "keydiff");

    // FA-off capture gate. On Adreno/Vulkan the FA-off kq_soft_max capture emits corrupt
    // tokens when head_dim is neither 64 nor 128 (Phi-3-mini, head_dim 96). The same
    // policies are clean on Llama-1B and on CPU, so the capture path is at fault.
    // SnapKV and Ada-KV score once at the end of prefill, so they move to the in-graph side
    // node with the same selection. H2O and TOVA re-score every decode step and are refused.
    {
        const int  hd_probe      = llama_model_n_head(model) > 0
                                 ? (int)(llama_model_n_embd(model) / llama_model_n_head(model)) : 0;
        const bool fa_off_unsafe = args.n_gpu_layers > 0 && hd_probe != 64 && hd_probe != 128;
        const bool needs_capture = !(args.policy == "vanilla" || positional_policy || args.fa_on_evict);
        if (fa_off_unsafe && needs_capture) {
            if (args.policy == "snapkv" || args.policy == "adakv") {
                args.fa_on_evict     = true;
                args.no_evict_decode = true;
                // Also force in-place compaction. fa_on_evict alone uses the state
                // round-trip, whose second full cache gets the process OS-killed for
                // Phi-3 at 16K on the phone.
                args.compact_inplace = 1;
                args.defrag_mode     = 1;
                g_fa_off_gate        = "auto-promoted-to-side-node+inplace";
                std::fprintf(stderr,
                    "[gate] FA-off capture is unsafe on this backend (head_dim=%d, GPU): "
                    "auto-promoting policy=%s to the in-graph side node (--fa-on-evict). "
                    "Selection is unchanged; decode-time re-eviction is disabled.\n",
                    hd_probe, args.policy.c_str());
            } else {
                g_fa_off_gate = "refused";
                std::fprintf(stderr,
                    "[gate] REFUSING TO RUN: policy=%s requires the FA-off kq_soft_max capture, "
                    "which is numerically broken on this backend (head_dim=%d, GPU) and emits "
                    "corrupt tokens. This policy re-scores every decode step, so the in-graph "
                    "side node cannot serve it. Use --n-gpu-layers 0 (CPU), or a policy that "
                    "scores at prefill.\n", args.policy.c_str(), hd_probe);
                return 2;
            }
        }
    }

    bool use_fa = (args.policy == "vanilla" && args.fa_vanilla) || args.fa_on_evict
                  || (positional_policy && !args.no_fa_positional);
    cparams.flash_attn_type = use_fa ? LLAMA_FLASH_ATTN_TYPE_AUTO
                                     : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    // A positional policy running FA-on has nothing to capture: skipping cb_eval
    // also removes the per-step kq_soft_max host readback it was paying for free.
    cparams.cb_eval = (!args.force_cb_eval && (args.policy == "vanilla" || (positional_policy && use_fa)))
                      ? nullptr : eval_callback;
    g_cb_eval_noread = args.cb_eval_noread;
    if (args.force_cb_eval) {
        std::fprintf(stderr, "[endurkv] DIAGNOSTIC: cb_eval force-attached to policy=%s "
                             "(FA=%s). This run is for isolating the Vulkan corruption, "
                             "not for measurement.\n", args.policy.c_str(), use_fa ? "ON" : "OFF");
    }
    cparams.cb_eval_user_data = &cap;
    cparams.n_threads = args.n_threads;
    cparams.n_threads_batch = args.n_threads;
    std::fprintf(stderr, "[eviction_bench] policy=%s K=%d n_sink=%d FA=%s repeat_penalty=%.2f\n",
                 args.policy.c_str(), args.k_nominal, args.n_sink,
                 use_fa ? "ON" : "OFF", args.repeat_penalty);

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { std::fprintf(stderr, "ctx init fail\n"); return 1; }

    std::FILE * csv = std::fopen(args.out_csv.c_str(), "w");
    if (!csv) { std::fprintf(stderr, "open out_csv fail\n"); return 1; }
    std::fprintf(csv, "step,token_id,token_text,wall_us,n_kv_cells,rss_kb,log_prob,nll,evicted_this_step\n");

    std::FILE * gen = nullptr;
    if (!args.out_gen.empty()) gen = std::fopen(args.out_gen.c_str(), "w");

    const int pid = getpid();
    const int64_t t_start = ggml_time_us();

    // Energy-aware controller (--energy-aware). Runs before prefill so the GPU clock cap is
    // in force during prefill. Reads the state of charge once per request (one process per
    // request), so the budget never changes mid-answer. On mains (Charging/Full) the top tier
    // is always used. The last level is persisted: moving down a tier needs a 3-point SoC
    // band past the threshold, moving up is immediate.
    // current_now and power_now are not read, they report 0 while USB is attached.
    if (args.energy_aware) {
        auto rd = [](const char * path, std::string & out) -> bool {
            std::FILE * f = std::fopen(path, "r");
            if (!f) return false;
            char buf[128] = {0};
            bool ok = std::fgets(buf, sizeof(buf), f) != nullptr;
            std::fclose(f);
            if (!ok) return false;
            out = buf;
            while (!out.empty() && (out.back() == '\n' || out.back() == '\r' || out.back() == ' ')) out.pop_back();
            return true;
        };
        std::string soc_s, status_s;
        bool have_soc = rd("/sys/class/power_supply/battery/capacity", soc_s);
        rd("/sys/class/power_supply/battery/status", status_s);
        int  soc      = have_soc ? std::atoi(soc_s.c_str()) : -1;
        bool on_mains = status_s.find("Charging") != std::string::npos ||
                        status_s.find("Full")     != std::string::npos;

        // Fallback: on stock Android the sysfs battery nodes are root-only and this runs as
        // `shell`, but `dumpsys battery` is readable. BatteryManager status codes:
        // 2 CHARGING and 5 FULL count as mains, 3 DISCHARGING and 4 NOT_CHARGING do not.
        if (!have_soc) {
            std::FILE * pp = popen("dumpsys battery 2>/dev/null", "r");
            if (pp) {
                char line[256];
                int lvl = -1, st = -1;
                // Match keys only at line start. `dumpsys battery` also prints
                // "Capacity level: 3", which a substring search would take as the SoC.
                auto key_at_line_start = [](const char * ln, const char * key) -> const char * {
                    while (*ln == ' ' || *ln == '\t') ++ln;
                    const size_t n = std::strlen(key);
                    return std::strncmp(ln, key, n) == 0 ? ln + n : nullptr;
                };
                while (std::fgets(line, sizeof(line), pp)) {
                    if (const char * p1 = key_at_line_start(line, "level:"))  lvl = std::atoi(p1);
                    if (const char * p2 = key_at_line_start(line, "status:")) st  = std::atoi(p2);
                }
                pclose(pp);
                if (lvl >= 0) {
                    soc = lvl; have_soc = true;
                    on_mains = (st == 2 || st == 5);
                    status_s = (st == 2) ? "Charging" : (st == 5) ? "Full"
                             : (st == 3) ? "Discharging" : (st == 4) ? "Not charging" : "Unknown";
                    std::fprintf(stderr, "[energy-aware] sysfs denied; using dumpsys battery (level=%d status=%d)\n", lvl, st);
                }
            }
        }

        int prev = -1;
        { std::string p; if (rd(args.ea_state_file.c_str(), p)) prev = std::atoi(p.c_str()); }

        int level;                                   // 0 = mildest, 2 = most aggressive
        if (!have_soc || soc < 0) {
            level = 0;                               // no sensor -> never degrade
            std::fprintf(stderr, "[energy-aware] battery capacity unreadable; staying at tier 0\n");
        } else if (on_mains) {
            level = 0;
        } else {
            const int hyst = 3;                      // must fall this far past a threshold to descend
            if      (soc >  args.ea_soc_hi) level = 0;
            else if (soc >  args.ea_soc_lo) level = 1;
            else                            level = 2;
            if (prev >= 0 && level > prev) {         // descending: require the band
                const int th = (level == 1) ? args.ea_soc_hi : args.ea_soc_lo;
                if (soc > th - hyst) level = prev;   // inside the band -> hold
            }
        }
        // Tiers in absolute K, anchored on --k-nominal, the same on both backends:
        //   level 0 (healthy / mains): K = k_nominal      (1024 by default)
        //   level 1 (mid):             K = k_nominal / 2  (512)
        //   level 2 (low):             K = k_nominal / 4  (256)
        // --ea-k hi mid lo sets the three values, --ea-pct hi mid lo selects the legacy
        // percent-of-prompt ladder instead.
        const bool on_gpu  = args.n_gpu_layers > 0;
        const int  k_floor = args.n_sink + args.adaptive_rmin + 1;   // sinks + recent window must fit
        const int  k_hi    = args.ea_k_hi  > 0 ? args.ea_k_hi  : args.k_nominal;
        const int  k_mid   = args.ea_k_mid > 0 ? args.ea_k_mid : args.k_nominal / 2;
        const int  k_lo    = args.ea_k_lo  > 0 ? args.ea_k_lo  : args.k_nominal / 4;
        int k_sel = (level == 0) ? k_hi : (level == 1) ? k_mid : k_lo;
        k_sel = std::max(k_floor, k_sel);
        if (args.ea_pct_hi > 0.0f) {
            args.k_pct = (level == 0) ? args.ea_pct_hi : (level == 1) ? args.ea_pct_mid : args.ea_pct_lo;
            std::fprintf(stderr, "[energy-aware] soc=%d%% status=%s mains=%d backend=%s prev_level=%d -> level=%d k-pct=%.1f%% (legacy ladder)\n",
                         soc, status_s.c_str(), (int) on_mains, on_gpu ? "GPU" : "CPU", prev, level, args.k_pct);
        } else if (on_gpu) {
            // GPU: cap the GPU clock and hold K (see the Args comments).
            const int mhz = (level == 0) ? args.ea_gpu_mhz_hi : (level == 1) ? args.ea_gpu_mhz_mid : args.ea_gpu_mhz_lo;
            args.k_pct = 0.0f;
            if (args.ea_gpu_k_ladder) args.k_nominal = k_sel;      // measurement mode only
            args.ea_gpu_write_rc = gpu_cap_write(mhz);
            args.ea_gpu_mhz_set  = (args.ea_gpu_write_rc == 0) ? mhz : 0;
            std::fprintf(stderr, "[energy-aware] soc=%d%% status=%s mains=%d backend=GPU prev_level=%d -> level=%d gpu_max=%d MHz (ladder %d/%d/%d) write_rc=%d K=%d%s\n",
                         soc, status_s.c_str(), (int) on_mains, prev, level, mhz,
                         args.ea_gpu_mhz_hi, args.ea_gpu_mhz_mid, args.ea_gpu_mhz_lo, args.ea_gpu_write_rc,
                         args.k_nominal, args.ea_gpu_k_ladder ? " (K ladder forced)" : " (K held)");
            if (args.ea_gpu_write_rc != 0)
                std::fprintf(stderr, "[energy-aware] GPU clock write failed (no root?); running at the clock in force\n");
        } else {
            // CPU: shrink the cache, leave the clock alone.
            args.k_pct     = 0.0f;      // absolute ladder wins over any --k-pct on the command line
            args.k_nominal = k_sel;
            std::fprintf(stderr, "[energy-aware] soc=%d%% status=%s mains=%d backend=CPU prev_level=%d -> level=%d K=%d (ladder %d/%d/%d)\n",
                         soc, status_s.c_str(), (int) on_mains, prev, level, k_sel, k_hi, k_mid, k_lo);
        }
        args.ea_soc_seen = soc; args.ea_status_seen = status_s; args.ea_level_seen = level;
        { std::FILE * f = std::fopen(args.ea_state_file.c_str(), "w");
          if (f) { std::fprintf(f, "%d\n", level); std::fclose(f); } }
    }

    // Prefill
    // --force-cb-eval must also switch the capture ON, or the callback returns early at
    // its !cap->active guard and the diagnostic run is identical to a normal vanilla run.
    cap.reset(); cap.active = (args.policy != "vanilla") || args.force_cb_eval;
    cap.obs_window = std::max(1, args.obs_window);  // propagate observation-window setting
    // --fa-on-evict: enable the kq_evict side node for prefill so the gate is captured with
    // FA on. It is disabled again after prefill and eviction.
    if (args.fa_on_evict) llama_endurkv_set_evict_obs_window(cap.obs_window);
    int64_t t_prefill_0 = ggml_time_us();
    // H2O accumulates attention over every forward pass, prefill chunks included. Each
    // chunk is folded in after its llama_decode, since eval_callback overwrites per_layer.
    H2OState h2o_state;
    // Persistent state for v1_predictive - ring buffer of mean-across-heads
    // attention vectors per layer. Updated every decode forward pass.
    AttnHistory attn_hist;
    // Optional per-chunk prefill trace (--out-prefill-csv), one line per chunk.
    FILE * prefill_csv = nullptr;
    if (!args.out_prefill_csv.empty()) {
        prefill_csv = std::fopen(args.out_prefill_csv.c_str(), "w");
        if (prefill_csv) std::fprintf(prefill_csv,
            "chunk_idx,wall_us,n_tokens_processed,n_kv_cells\n");
    }
#ifdef EVB_HAS_MTMD
    if (vl_ctx) {
        // Multimodal prefill: evaluate the mtmd chunks (text, image embeddings, text) in
        // order. The question is in the final text chunk, so the kq_evict side node is on
        // only for that chunk, as in text prefill. Its scores cover all cells, image tokens
        // included. mtmd_helper_eval_chunk_single batches at n_batch and advances M-RoPE.
        llama_pos vl_n_past = 0;
        const size_t n_ck = mtmd_input_chunks_size(vl_chunks);
        for (size_t ci = 0; ci < n_ck; ++ci) {
            const mtmd_input_chunk * ck = mtmd_input_chunks_get(vl_chunks, ci);
            const bool is_last = (ci + 1 == n_ck);
            if (args.fa_on_evict) llama_endurkv_set_evict_obs_window(is_last ? cap.obs_window : 0);
            if (mtmd_helper_eval_chunk_single(vl_ctx, ctx, ck, vl_n_past, 0,
                                              args.n_batch, /*logits_last*/ is_last, &vl_n_past) != 0) {
                std::fprintf(stderr, "[vl] chunk %zu eval failed\n", ci);
                return 1;
            }
        }
        std::fprintf(stderr, "[vl] prefill done: n_past=%d cells=%d\n",
                     (int) vl_n_past,
                     (int)(llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1));
    } else
#endif
    {
        // Prefill in n_batch-sized chunks. On Adreno Vulkan one 2840-token llama_decode
        // trips the GPU watchdog even with a small n_ubatch.
        const int chunk = std::max(1, args.n_batch);
        int chunk_idx = 0;
        for (int off = 0; off < n_prompt; off += chunk) {
            int n = std::min(chunk, n_prompt - off);
            // kq_evict scores are overwritten every chunk and eviction uses only the last
            // chunk's, so emit the side node only on the last prefill chunk. On earlier
            // chunks it would only add compute/sync boundaries. Selection is unchanged.
            if (args.fa_on_evict) {
                const bool is_last_chunk = (off + n >= n_prompt);
                llama_endurkv_set_evict_obs_window(is_last_chunk ? cap.obs_window : 0);
            }
            llama_batch batch = llama_batch_get_one(ptoks.data() + off, n);
            if (llama_decode(ctx, batch) != 0) {
                std::fprintf(stderr, "prefill fail at offset %d (n=%d)\n", off, n);
                return 1;
            }
            // Fold this chunk's attention into the H2O sums before the next chunk
            // overwrites cap.per_layer. endurkv_optimal also uses policy_h2o() at prefill.
            if (args.policy == "h2o" || args.policy == "endurkv_optimal") h2o_state.update(cap);
            // v1_predictive: push prefill samples too, so slopes exist at decode step 0.
            if (args.policy == "v1_predictive") attn_hist.push(cap);
            // Per-chunk prefill trace, after llama_decode so n_kv is the post-chunk state.
            if (prefill_csv) {
                int64_t t_now = ggml_time_us();
                int n_kv_now = (int)(llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1);
                std::fprintf(prefill_csv, "%d,%lld,%d,%d\n",
                             chunk_idx, (long long)t_now, off + n, n_kv_now);
            }
            ++chunk_idx;
        }
    }
    if (prefill_csv) { std::fclose(prefill_csv); prefill_csv = nullptr; }
    int64_t t_prefill_1 = ggml_time_us();
    long peak_rss = read_rss_kb(pid);
    int  peak_kv  = (int)(llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1);

    // Eviction after prefill (sets the working set going into decode)
    int evicted_prefill = 0;
    if (args.policy == "v1" || args.policy == "v1_fa" || args.policy == "v1_adaptive") {
        PolicyResult r = (args.policy == "v1_adaptive")
            ? policy_v1_adaptive(cap, args.k_nominal, n_kv_heads)
            : policy_v1(cap, args.k_nominal, n_kv_heads);
        protect_sink(r, args.n_sink, cap.n_kv);
        evicted_prefill = apply_eviction(ctx, r, cap.n_kv);
        // Decode-only clock cap: prefill is done, lower the GPU cap if asked.
        if (args.gpu_mhz_decode > 0 && args.n_gpu_layers > 0) {
            args.gpu_mhz_decode_rc = gpu_cap_write(args.gpu_mhz_decode);
            std::fprintf(stderr, "[phase-clock] prefill done; GPU cap for decode = %d MHz (write_rc=%d)\n",
                         args.gpu_mhz_decode, args.gpu_mhz_decode_rc);
        }
        // Resolve a percentage budget now that n_prompt is known, before the
        // adaptive anchor, so the anchor/recent split is computed against the real K.
        if (args.k_pct > 0.0f) {
            const int k_from_pct = (int)std::lround(args.k_pct / 100.0f * (float)n_prompt);
            const int k_floor    = args.n_sink + args.adaptive_rmin + 1;   // sinks + recent window must fit
            args.k_nominal = std::max(k_floor, k_from_pct);
            std::fprintf(stderr, "[mukv] K from --k-pct %.1f%% of %d prompt tokens -> K=%d\n",
                         args.k_pct, n_prompt, args.k_nominal);
        }
        if (args.adaptive_anchor) {
            // Adaptive anchor split. alpha_a is the share of prefill attention mass that falls
            // outside the last R_min prompt tokens, from the column mean used for selection:
            //   col_mean[p]   = mean over (layer, head, query) of attention[q,l,h,p]
            //   N_eff         = min(n_prompt, cap.n_kv)
            //   recent_cutoff = max(0, N_eff - R_min)
            //   alpha_a       = sum col_mean[0, recent_cutoff) / sum col_mean[0, N_eff)
            // K - n_sink is then split into round(alpha_a * (K - n_sink)) anchors and the rest recent.
            const int N_eff = std::min(n_prompt, cap.n_kv);
            const int recent_cutoff = std::max(0, N_eff - args.adaptive_rmin);
            std::vector<double> col_mean(cap.n_kv, 0.0);
            long long n_samples = 0;
            for (const auto & [layer, attn] : cap.per_layer) {
                for (int h = 0; h < cap.n_head; ++h) {
                    const float * a = attn.data() + (size_t)h * cap.n_kv;
                    for (int p = 0; p < cap.n_kv; ++p) col_mean[p] += a[p];
                    n_samples++;
                }
            }
            if (n_samples > 0) for (int p = 0; p < cap.n_kv; ++p) col_mean[p] /= (double)n_samples;

            double distant_mass = 0.0, recent_mass = 0.0;
            for (int p = 0; p < recent_cutoff;          ++p) distant_mass += col_mean[p];
            for (int p = recent_cutoff; p < N_eff;      ++p) recent_mass  += col_mean[p];
            double total_mass = distant_mass + recent_mass;

            float alpha_a = (total_mass > 1e-12)
                ? (float)(distant_mass / total_mass) : 0.5f;
            if (alpha_a < 0.05f) alpha_a = 0.05f;
            if (alpha_a > 0.95f) alpha_a = 0.95f;

            // Count-based metric: logged, and used for alpha_a only with --gate-count
            int distant_count = 0, total_count = 0;
            for (const auto & [layer, attn] : cap.per_layer) {
                for (int h = 0; h < cap.n_head; ++h) {
                    const float * a = attn.data() + (size_t)h * cap.n_kv;
                    int argmax_p = 0; float max_v = a[0];
                    for (int p = 1; p < N_eff; ++p) {
                        if (a[p] > max_v) { max_v = a[p]; argmax_p = p; }
                    }
                    if (argmax_p < recent_cutoff) distant_count++;
                    total_count++;
                }
            }

            // --gate-count: count-based α_a instead of mass.
            if (args.gate_count && total_count > 0) {
                alpha_a = (float)distant_count / (float)total_count;
                if (alpha_a < 0.05f) alpha_a = 0.05f;
                if (alpha_a > 0.95f) alpha_a = 0.95f;
            }
            // --gate-alpha-floor: guard the anchor budget (clamp α_a up).
            if (args.gate_alpha_floor > 0.0f && alpha_a < args.gate_alpha_floor) {
                alpha_a = args.gate_alpha_floor > 0.95f ? 0.95f : args.gate_alpha_floor;
            }
            // --gate-alpha-max: guard the recent window (clamp α_a down).
            if (args.gate_alpha_max > 0.0f && alpha_a > args.gate_alpha_max) {
                alpha_a = args.gate_alpha_max;
            }
            int non_sink = args.k_nominal - args.n_sink;
            int new_anchor = (int)std::lround(alpha_a * (float)non_sink);
            int new_recent = non_sink - new_anchor;
            if (new_anchor < 1) new_anchor = 1;
            if (new_recent < 1) new_recent = 1;
            std::fprintf(stderr,
                "[adaptive-anchor] gate=%s alpha_a=%.3f (mass: distant=%.4f recent=%.4f) "
                "[count-legacy: %d/%d] -> anchor=%d recent=%d "
                "(K=%d, n_sink=%d, R_min=%d, N=%d, N_eff=%d, cap.n_kv=%d)\n",
                (args.gate_count?"count":(args.gate_alpha_floor>0?"mass+floor":"mass")), alpha_a, (double)distant_mass, (double)recent_mass,
                distant_count, total_count, new_anchor, new_recent,
                args.k_nominal, args.n_sink, args.adaptive_rmin, n_prompt, N_eff, cap.n_kv);
            args.anchor_top_k = new_anchor;
            args.recent_budget = new_recent;
            args.snapkv_decode = !(args.no_state_swap || args.fa_on_evict);   // no swap with --no-state-swap or --fa-on-evict (already FA-on)
            args.no_evict_decode = true;
        }
        if (args.anchor_top_k > 0 && (args.snapkv_decode || args.policy == "v1_adaptive" || args.fa_on_evict)) {
            // Selective anchoring: keep only the top anchor_top_k spread-gate survivors.
            // Per-position score from attention received across (layer, head):
            //   mean         mean attention
            //   entropy      entropy across (layer, head), high = many heads attend
            //   neg_entropy  low entropy preferred (positions few heads attend sharply)
            std::vector<float> pos_score(cap.n_kv, 0.0f);
            int n_score_samples = 0;
            const bool use_entropy   = (args.anchor_score_mode == "entropy");
            const bool use_neg_entropy = (args.anchor_score_mode == "neg_entropy");
            const bool use_hybrid    = (args.anchor_score_mode == "hybrid");
            if (use_hybrid) {
                // hybrid: mean * (1 + alpha * sharpness), sharpness = 1 - normalized entropy
                std::vector<float> pos_mass(cap.n_kv, 0.0f);
                int H_total = 0;
                for (const auto & [layer, attn] : cap.per_layer) {
                    for (int h = 0; h < cap.n_head; ++h) {
                        const float * a = attn.data() + (size_t)h * cap.n_kv;
                        for (int p = 0; p < cap.n_kv; ++p) pos_mass[p] += a[p];
                    }
                    H_total += cap.n_head;
                }
                const float eps = 1e-12f;
                const float H_max = std::log((float)std::max(1, H_total));  // max possible entropy
                const float alpha = args.hybrid_alpha;
                for (int p = 0; p < cap.n_kv; ++p) {
                    if (pos_mass[p] < eps) { pos_score[p] = 0.0f; continue; }
                    float mean = pos_mass[p] / (float)H_total;
                    float H = 0.0f;
                    const float inv_mass = 1.0f / pos_mass[p];
                    for (const auto & [layer, attn] : cap.per_layer) {
                        for (int h = 0; h < cap.n_head; ++h) {
                            const float a = attn[(size_t)h * cap.n_kv + p];
                            if (a > eps) {
                                const float w = a * inv_mass;
                                H -= w * std::log(w);
                            }
                        }
                    }
                    float H_norm = (H_max > 0.0f) ? std::min(1.0f, std::max(0.0f, H / H_max)) : 0.0f;
                    float sharpness = 1.0f - H_norm;  // [0, 1], 1 = perfectly sharp
                    pos_score[p] = mean * (1.0f + alpha * sharpness);
                }
                n_score_samples = H_total;
            } else if (use_entropy || use_neg_entropy) {
                // First pass: total mass per position across (layer, head).
                std::vector<float> pos_mass(cap.n_kv, 0.0f);
                int H_total = 0;
                for (const auto & [layer, attn] : cap.per_layer) {
                    for (int h = 0; h < cap.n_head; ++h) {
                        const float * a = attn.data() + (size_t)h * cap.n_kv;
                        for (int p = 0; p < cap.n_kv; ++p) pos_mass[p] += a[p];
                    }
                    H_total += cap.n_head;
                }
                // Second pass: per-position entropy across the H_total head-samples.
                const float eps = 1e-12f;
                for (int p = 0; p < cap.n_kv; ++p) {
                    if (pos_mass[p] < eps) { pos_score[p] = 0.0f; continue; }
                    float H = 0.0f;
                    const float inv_mass = 1.0f / pos_mass[p];
                    for (const auto & [layer, attn] : cap.per_layer) {
                        for (int h = 0; h < cap.n_head; ++h) {
                            const float a = attn[(size_t)h * cap.n_kv + p];
                            if (a > eps) {
                                const float w = a * inv_mass;
                                H -= w * std::log(w);
                            }
                        }
                    }
                    pos_score[p] = use_neg_entropy ? -H : H;
                }
                // Break ties by mass so zero-mass positions do not beat real ones.
                const float scale = 1e-6f;
                for (int p = 0; p < cap.n_kv; ++p) pos_score[p] += scale * pos_mass[p];
                n_score_samples = H_total;
            } else if (args.anchor_score_mode == "gate_weighted") {
                // gate_weighted: weight each (layer, head) by its peak m_h = max_p Ā_h[p],
                // so sharp heads count more than diffuse ones:
                //   pos_score[p] = Σ_{l,h} m_{l,h} · Ā_{l,h}[p]  /  Σ_{l,h} m_{l,h}
                std::vector<float> m_per_lh;
                m_per_lh.reserve(cap.per_layer.size() * cap.n_head);
                float total_weight = 0.0f;
                for (const auto & [layer, attn] : cap.per_layer) {
                    for (int h = 0; h < cap.n_head; ++h) {
                        const float * a = attn.data() + (size_t)h * cap.n_kv;
                        float m = 0.0f;
                        for (int p = 0; p < cap.n_kv; ++p) if (a[p] > m) m = a[p];
                        m_per_lh.push_back(m);
                        total_weight += m;
                    }
                }
                size_t idx = 0;
                for (const auto & [layer, attn] : cap.per_layer) {
                    for (int h = 0; h < cap.n_head; ++h) {
                        const float w = m_per_lh[idx++];
                        const float * a = attn.data() + (size_t)h * cap.n_kv;
                        for (int p = 0; p < cap.n_kv; ++p) pos_score[p] += w * a[p];
                        n_score_samples += 1;
                    }
                }
                if (total_weight > 0.0f) {
                    const float inv_w = 1.0f / total_weight;
                    for (auto & s : pos_score) s *= inv_w;
                }
                std::fprintf(stderr, "[gate_weighted] total m_h weight = %.3f over %d (layer,head) pairs\n",
                             total_weight, (int)m_per_lh.size());
            } else {
                // Default "mean" mode
                for (const auto & [layer, attn] : cap.per_layer) {
                    for (int h = 0; h < cap.n_head; ++h) {
                        const float * a = attn.data() + (size_t)h * cap.n_kv;
                        for (int p = 0; p < cap.n_kv; ++p) pos_score[p] += a[p];
                    }
                    n_score_samples += cap.n_head;
                }
                if (n_score_samples > 0) {
                    const float inv_n = 1.0f / (float)n_score_samples;
                    for (auto & s : pos_score) s *= inv_n;
                }
            }
            // SnapKV 1D max-pool: replace each score by the max in a ±(kernel/2) window so
            // top-K picks token clusters. Only when --snapkv-pool > 1.
            if (args.snapkv_pool > 1) {
                const int r = args.snapkv_pool / 2;
                std::vector<float> pooled(cap.n_kv);
                for (int p = 0; p < cap.n_kv; ++p) {
                    float mx = pos_score[p];
                    for (int d = -r; d <= r; ++d) {
                        const int q = p + d;
                        if (q >= 0 && q < cap.n_kv && pos_score[q] > mx) mx = pos_score[q];
                    }
                    pooled[p] = mx;
                }
                pos_score.swap(pooled);
                std::fprintf(stderr, "[snapkv-pool] 1D max-pool kernel=%d applied to position scores\n", args.snapkv_pool);
            }
            // Identify "kept after spread-gate" positions (anywhere any layer kept).
            std::vector<uint8_t> spread_kept(cap.n_kv, 0);
            if (args.no_perhead_gate) {
                // gate bypassed: the pool is every prompt position (cells past n_prompt are padding)
                const int n_pool = std::min(n_prompt, cap.n_kv);
                std::fill(spread_kept.begin(), spread_kept.begin() + n_pool, 1);
            } else {
                for (const auto & layer_kept : r.per_layer_keep) {
                    for (int p : layer_kept) if (p >= 0 && p < cap.n_kv) spread_kept[p] = 1;
                }
            }
            // Among spread-gate survivors, pick top-anchor_top_k by score
            // (always include sink positions in the top-K set).
            std::vector<std::pair<float, int>> ranked;
            ranked.reserve(cap.n_kv);
            for (int p = 0; p < cap.n_kv; ++p) {
                if (!spread_kept[p]) continue;
                if (p < args.n_sink) {
                    ranked.emplace_back(1e30f, p);  // sink always wins
                } else {
                    ranked.emplace_back(pos_score[p], p);
                }
            }
            // Diagnostic: number of non-sink positions holding 50% and 90% of the non-sink
            // attention mass, and the normalized entropy of that distribution.
            {
                std::vector<double> ns;
                for (const auto & rk : ranked) if (rk.first < 1e29f && rk.first > 0.0f) ns.push_back((double)rk.first);
                std::sort(ns.begin(), ns.end(), std::greater<double>());
                double tot = 0.0; for (double v : ns) tot += v;
                int s50=0,s90=0; double acc=0.0;
                for (size_t i=0;i<ns.size();++i){ acc+=ns[i]; if(s50==0&&tot>0&&acc>=0.50*tot)s50=(int)(i+1); if(tot>0&&acc>=0.90*tot){s90=(int)(i+1);break;} }
                double H=0.0; if(tot>0){ for(size_t i=0;i<ns.size();++i){ double p=ns[i]/tot; if(p>0) H-=p*std::log(p);} }
                double Hn = ns.size()>1 ? H/std::log((double)ns.size()) : 0.0;
                std::fprintf(stderr, "[DIAG] sink_excl_spread50=%d spread90=%d nonsink_pos=%d entropy_norm=%.3f n_sink=%d N=%d\n",
                             s50, s90, (int)ns.size(), Hn, args.n_sink, cap.n_kv);
            }
            // If anchor_coverage > 0, take the fewest top-scored positions whose cumulative
            // mass reaches that fraction, instead of a fixed anchor_top_k.
            int top;
            if (args.anchor_coverage > 0.0f) {
                std::sort(ranked.begin(), ranked.end(),
                          [](const auto & a, const auto & b) { return a.first > b.first; });
                double total_mass = 0.0;
                for (const auto & rk : ranked) total_mass += std::max(0.0f, rk.first);
                if (total_mass <= 0.0) {
                    top = (int)ranked.size();
                } else {
                    double target = (double)args.anchor_coverage * total_mass;
                    double accum = 0.0;
                    top = 0;
                    for (const auto & rk : ranked) {
                        accum += std::max(0.0f, rk.first);
                        top++;
                        if (accum >= target) break;
                    }
                    if (top < args.n_sink) top = args.n_sink;
                }
                std::fprintf(stderr, "[v1_fa2] adaptive anchor: coverage=%.2f -> top=%d/%d positions\n",
                             args.anchor_coverage, top, (int)ranked.size());
            } else {
                top = std::min<int>(args.anchor_top_k, (int)ranked.size());
                if (top < (int)ranked.size()) {
                    std::nth_element(ranked.begin(), ranked.begin() + top, ranked.end(),
                                     [](const auto & a, const auto & b) { return a.first > b.first; });
                }
            }
            if (top < (int)ranked.size()) {
                // Build the keep mask, everything else is dropped.
                std::vector<uint8_t> keep(cap.n_kv, 0);
                for (int i = 0; i < top; ++i) keep[ranked[i].second] = 1;
                if (const char * dump = std::getenv("EVICT_DUMP_KEEP")) {   // keep-set dump for gate comparisons
                    if (FILE * kf = std::fopen(dump, "w")) {
                        for (int p = 0; p < cap.n_kv; ++p) if (keep[p]) std::fprintf(kf, "%d\n", p);
                        std::fclose(kf);
                    }
                }
                // Drop everything not in the top set.
                llama_memory_t mem = llama_get_memory(ctx);
                int dropped_extra = 0;
                // M-RoPE caches are evicted by cell index, since all tokens of an image
                // share one dim-0 position.
                if (g_evict_by_cell_index) {
                    std::vector<int8_t> keep_i8(keep.begin(), keep.end());
                    dropped_extra = (int) llama_endurkv_seq_rm_cells(mem, 0, keep_i8.data(),
                                                                    (uint32_t) cap.n_kv);
                } else {
                    int pp = 0;
                    while (pp < cap.n_kv) {
                        if (keep[pp]) { ++pp; continue; }
                        int s = pp;
                        while (pp < cap.n_kv && !keep[pp]) ++pp;
                        int e = pp;
                        llama_memory_seq_rm(mem, 0, s, e);
                        dropped_extra += (e - s);
                    }
                }
                evicted_prefill += dropped_extra;
                std::fprintf(stderr, "[v1_fa2] selective anchoring: spread-gate kept %d, top-%d by score retained (additional %d dropped)\n",
                             (int)ranked.size(), top, dropped_extra);
            }
        }
    } else if (args.policy == "tova") {
        PolicyResult r = policy_tova(cap, args.k_nominal, n_kv_heads);
        protect_sink(r, args.n_sink, cap.n_kv);
        evicted_prefill = apply_eviction(ctx, r, cap.n_kv);
    } else if (args.policy == "tova_canonical") {
        PolicyResult r = policy_tova_canonical(cap, args.k_nominal, n_kv_heads);
        protect_sink(r, args.n_sink, cap.n_kv);
        evicted_prefill = apply_eviction(ctx, r, cap.n_kv);
    } else if (args.policy == "pyramid") {
        PolicyResult r = policy_pyramid(cap, args.k_nominal, n_layers_total);
        protect_sink(r, args.n_sink, cap.n_kv);
        evicted_prefill = apply_eviction(ctx, r, cap.n_kv);
    } else if (args.policy == "h2o") {
        PolicyResult r = policy_h2o(cap, h2o_state, args.k_nominal, n_kv_heads);
        protect_sink(r, args.n_sink, cap.n_kv);
        evicted_prefill = apply_eviction(ctx, r, cap.n_kv);
    } else if (args.policy == "endurkv_optimal") {
        // endurkv_optimal: policy_h2o gives per-layer keep sets (recent K/2 + heavy K/2),
        // but OR-ing them across layers and heads for sequence-level seq_rm keeps nearly
        // every position. So rank that union by mean cumulative attention (h2o_state, filled
        // during prefill) and keep the top K_nominal, sinks first. The state swap then hands
        // the cache to an FA-on decode context.
        PolicyResult r = policy_h2o(cap, h2o_state, args.k_nominal, n_kv_heads);
        protect_sink(r, args.n_sink, cap.n_kv);

        // Collapse per-layer keep sets to a global top-K_nominal by mean
        // cumulative attention (H2O canonical heavy-hitter ordering).
        const int K_target = std::min<int>(args.k_nominal, cap.n_kv);
        if (K_target > 0 && cap.n_kv > 0) {
            // Score each position by mean h2o_state.sum_attn across layers + heads.
            std::vector<double> pos_score(cap.n_kv, 0.0);
            int n_score_samples = 0;
            for (const auto & [layer, sum] : h2o_state.sum_attn) {
                auto it_h = h2o_state.n_head_per_layer.find(layer);
                auto it_k = h2o_state.n_kv_per_layer.find(layer);
                if (it_h == h2o_state.n_head_per_layer.end() ||
                    it_k == h2o_state.n_kv_per_layer.end()) continue;
                const int H = it_h->second;
                const int K_lay = it_k->second;
                for (int h = 0; h < H; ++h) {
                    const double * s = sum.data() + (size_t)h * K_lay;
                    const int P = std::min(cap.n_kv, K_lay);
                    for (int p = 0; p < P; ++p) pos_score[p] += s[p];
                }
                n_score_samples += H;
            }
            if (n_score_samples > 0) {
                const double inv_n = 1.0 / (double)n_score_samples;
                for (auto & s : pos_score) s *= inv_n;
            }
            // Union of per-layer keep sets (candidates for global top-K).
            std::vector<uint8_t> any_keep(cap.n_kv, 0);
            for (const auto & layer_kept : r.per_layer_keep) {
                for (int p : layer_kept) if (p >= 0 && p < cap.n_kv) any_keep[p] = 1;
            }
            // Rank candidates by score, sink positions always win.
            std::vector<std::pair<double, int>> ranked;
            ranked.reserve(cap.n_kv);
            for (int p = 0; p < cap.n_kv; ++p) {
                if (!any_keep[p]) continue;
                if (p < args.n_sink) ranked.emplace_back(1e300, p);
                else                 ranked.emplace_back(pos_score[p], p);
            }
            const int K_keep = std::min<int>(K_target, (int)ranked.size());
            if (K_keep < (int)ranked.size()) {
                std::nth_element(ranked.begin(), ranked.begin() + K_keep, ranked.end(),
                                 [](const auto & a, const auto & b) { return a.first > b.first; });
                // Rebuild a single keep set (broadcast to all layers so the
                // apply_eviction OR-union collapses to this exact set).
                std::unordered_set<int> global_keep;
                global_keep.reserve(K_keep);
                for (int i = 0; i < K_keep; ++i) global_keep.insert(ranked[i].second);
                for (auto & layer_kept : r.per_layer_keep) layer_kept = global_keep;
                std::fprintf(stderr,
                             "[endurkv_optimal] global top-K collapse: candidates=%zu, kept=%d (K_target=%d)\n",
                             ranked.size(), K_keep, K_target);
            }
        }
        evicted_prefill = apply_eviction(ctx, r, cap.n_kv);
        std::fprintf(stderr,
                     "[endurkv_optimal] evicted_prefill=%d (cap.n_kv=%d)\n",
                     evicted_prefill, cap.n_kv);
    } else if (args.policy == "streamingllm") {
        // With FA on there is no capture, but this policy needs only the cache extent and the
        // layer count. Synthesize empty per-layer entries so the keep-set loop runs.
        if (use_fa && cap.per_layer.empty()) {
            cap.n_kv   = n_prompt;
            cap.n_head = 1;
            for (int il = 0; il < llama_model_n_layer(model); ++il) {
                cap.per_layer[il] = std::vector<float>();
            }
        }
        PolicyResult r = policy_streamingllm(cap, args.k_nominal, args.n_sink);
        evicted_prefill = apply_eviction(ctx, r, cap.n_kv);
    } else if (args.policy == "keydiff") {
        // KeyDiff (Park et al., NeurIPS 2025, arXiv:2504.15364). libllama scores each key by
        // -cos to the per-layer mean key, averaged over layers (llama_endurkv_keydiff_scores),
        // and the K_nominal most distinctive positions are kept. Runs FA-on with no capture.
        // Differences from the paper: prefill selection happens once at the end, not
        // block-wise (B=128), so peak memory is not comparable. The mean over layers is needed
        // for a sequence-level cache. The paper uses no sinks, so run with --n-sink 0.
        std::vector<float> kd_scores((size_t) n_prompt, 0.0f);
        const uint32_t kd_n = llama_endurkv_keydiff_scores(
                llama_get_memory(ctx), 0, kd_scores.data(), (uint32_t) kd_scores.size());
        if (kd_n == 0) {
            std::fprintf(stderr, "[keydiff] scores unavailable (declined) -- NO EVICTION run\n");
        } else {
            PolicyResult r;
            r.n_layers = llama_model_n_layer(model);
            r.per_layer_keep.resize(r.n_layers);
            std::vector<int> idx(kd_n);
            std::iota(idx.begin(), idx.end(), 0);
            const int K_keep = std::min<int>(args.k_nominal, (int) kd_n);
            std::nth_element(idx.begin(), idx.begin() + K_keep, idx.end(),
                             [&](int a, int b) { return kd_scores[a] > kd_scores[b]; });
            std::unordered_set<int> keep;
            keep.reserve(K_keep);
            for (int i = 0; i < K_keep; ++i) keep.insert(idx[i]);
            for (auto & lk : r.per_layer_keep) lk = keep;
            cap.n_kv = (int) kd_n;   // apply_eviction needs the cache extent
            protect_sink(r, args.n_sink, cap.n_kv);
            evicted_prefill = apply_eviction(ctx, r, cap.n_kv);
            std::fprintf(stderr, "[keydiff] scored=%u kept=%d evicted_prefill=%d\n",
                         kd_n, K_keep, evicted_prefill);
        }
    } else if (args.policy == "adakv") {
        // Ada-KV (Feng NeurIPS '25, Alg 2 / Ada-SnapKV). End-of-prefill global
        // budget allocation with α=0.2 safeguard, mask frozen during decode.
        PolicyResult r = policy_adakv(cap, args.k_nominal, n_kv_heads);
        protect_sink(r, args.n_sink, cap.n_kv);
        evicted_prefill = apply_eviction(ctx, r, cap.n_kv);
    } else if (args.policy == "snapkv") {
        // Canonical SnapKV (Li et al. 2024): per-head top-K over avgpool-5 pooled
        // observation-window attention, uniform budget, mask frozen during decode.
        PolicyResult r = policy_snapkv(cap, args.k_nominal, args.obs_window, args.snapkv_kernel);
        protect_sink(r, args.n_sink, cap.n_kv);
        evicted_prefill = apply_eviction(ctx, r, cap.n_kv);
    } else if (args.policy == "v1_entropy") {
        PolicyResult r = policy_v1_entropy(cap, args.k_nominal, n_kv_heads);
        protect_sink(r, args.n_sink, cap.n_kv);
        evicted_prefill = apply_eviction(ctx, r, cap.n_kv);
        // Selective anchoring as in the v1_fa2 path: with snapkv_decode, keep only the top
        // anchor_top_k survivors by mean attention.
        if (args.anchor_top_k > 0 && args.snapkv_decode) {
            std::vector<float> pos_score(cap.n_kv, 0.0f);
            int n_score_samples = 0;
            for (const auto & [layer, attn] : cap.per_layer) {
                for (int h = 0; h < cap.n_head; ++h) {
                    const float * a = attn.data() + (size_t)h * cap.n_kv;
                    for (int p = 0; p < cap.n_kv; ++p) pos_score[p] += a[p];
                }
                n_score_samples += cap.n_head;
            }
            if (n_score_samples > 0) {
                const float inv_n = 1.0f / (float)n_score_samples;
                for (auto & s : pos_score) s *= inv_n;
            }
            std::vector<uint8_t> spread_kept(cap.n_kv, 0);
            for (const auto & layer_kept : r.per_layer_keep) {
                for (int p : layer_kept) if (p >= 0 && p < cap.n_kv) spread_kept[p] = 1;
            }
            std::vector<std::pair<float, int>> ranked;
            ranked.reserve(cap.n_kv);
            for (int p = 0; p < cap.n_kv; ++p) {
                if (!spread_kept[p]) continue;
                if (p < args.n_sink) ranked.emplace_back(1e30f, p);
                else                  ranked.emplace_back(pos_score[p], p);
            }
            int top = std::min<int>(args.anchor_top_k, (int)ranked.size());
            if (top < (int)ranked.size()) {
                std::nth_element(ranked.begin(), ranked.begin() + top, ranked.end(),
                                 [](const auto & a, const auto & b) { return a.first > b.first; });
                std::vector<uint8_t> keep(cap.n_kv, 0);
                for (int i = 0; i < top; ++i) keep[ranked[i].second] = 1;
                llama_memory_t mem = llama_get_memory(ctx);
                int dropped_extra = 0;
                int pp = 0;
                while (pp < cap.n_kv) {
                    if (keep[pp]) { ++pp; continue; }
                    int s = pp;
                    while (pp < cap.n_kv && !keep[pp]) ++pp;
                    int e = pp;
                    llama_memory_seq_rm(mem, 0, s, e);
                    dropped_extra += (e - s);
                }
                evicted_prefill += dropped_extra;
                std::fprintf(stderr, "[v1_entropy_stack] selective anchoring: kept %d, top-%d retained (additional %d dropped)\n",
                             (int)ranked.size(), top, dropped_extra);
            }
        }
    } else if (args.policy == "v1_predictive") {
        // First-pass eviction: history may be empty (or have only prefill chunks).
        // policy_v1_predictive falls back to recency-only when slope=-INF.
        PolicyResult r = policy_v1_predictive(cap, attn_hist, args.k_nominal, args.n_sink);
        protect_sink(r, args.n_sink, cap.n_kv);
        evicted_prefill = apply_eviction(ctx, r, cap.n_kv);
    }

    // Refresh at decode step 0 (RefreshKV-style): after prefill and anchor selection, run one
    // FA-off decode step, redo anchor selection from its attention, then state-swap as usual.
    if (args.refresh_at_decode_0 && args.snapkv_decode && cap.active &&
        (args.policy == "v1_fa" || args.policy == "v1_fa2")) {

        std::fprintf(stderr, "[refresh-0] running 1 FA-off step to capture decode-time attention\n");

        // Pick argmax token from the most recent prefill logits
        const float * cur_logits = llama_get_logits_ith(ctx, -1);
        int t1 = 0;
        const int n_vocab = llama_vocab_n_tokens(vocab);
        if (cur_logits) {
            float max_v = cur_logits[0];
            for (int v = 1; v < n_vocab; ++v) {
                if (cur_logits[v] > max_v) { max_v = cur_logits[v]; t1 = v; }
            }
        }

        // Clear cap.per_layer so cb_eval captures only the refresh step's attention
        cap.per_layer.clear();

        // Decode 1 step in FA-off (cb_eval still active) - cap.n_kv will update
        llama_batch refresh_batch = llama_batch_get_one(&t1, 1);
        const int rc = llama_decode(ctx, refresh_batch);
        if (rc != 0) {
            std::fprintf(stderr, "[refresh-0] decode failed (rc=%d); skipping refresh\n", rc);
        } else {
            // Update n_kv to include the new position
            cap.n_kv = (int)(llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1);
            std::fprintf(stderr, "[refresh-0] captured layers=%zu, n_kv=%d, t1=%d\n",
                         cap.per_layer.size(), cap.n_kv, t1);

            // Compute effective anchor_top_k and recent_budget for this refresh
            int eff_anchor = args.anchor_top_k;
            int eff_recent = args.recent_budget;

            if (args.refresh_mukv_gate) {
                // L7-style adaptive α_a from decode-step attention
                int distant_count = 0;
                int total_count = 0;
                const int recent_cutoff = std::max(0, cap.n_kv - args.adaptive_rmin);
                for (const auto & [layer, attn] : cap.per_layer) {
                    for (int h = 0; h < cap.n_head; ++h) {
                        const float * a = attn.data() + (size_t)h * cap.n_kv;
                        int argmax_p = 0; float max_v = a[0];
                        for (int p = 1; p < cap.n_kv; ++p) {
                            if (a[p] > max_v) { max_v = a[p]; argmax_p = p; }
                        }
                        if (argmax_p < recent_cutoff) distant_count++;
                        total_count++;
                    }
                }
                float alpha_a = (total_count > 0)
                    ? (float)distant_count / (float)total_count : 0.5f;
                if (alpha_a < 0.05f) alpha_a = 0.05f;
                if (alpha_a > 0.95f) alpha_a = 0.95f;
                const int non_sink = args.k_nominal - args.n_sink;
                eff_anchor = (int)std::lround(alpha_a * (float)non_sink);
                eff_recent = non_sink - eff_anchor;
                if (eff_anchor < 1) eff_anchor = 1;
                if (eff_recent < 1) eff_recent = 1;
                std::fprintf(stderr, "[refresh-0/μKV-gate] α_a=%.3f distant=%d/%d → new anchor=%d recent=%d\n",
                             alpha_a, distant_count, total_count, eff_anchor, eff_recent);
            }

            // L1-style per-head K_h candidate pool
            std::vector<uint8_t> spread_kept(cap.n_kv, 0);
            if (args.refresh_mukv_gate) {
                for (const auto & [layer, attn] : cap.per_layer) {
                    for (int h = 0; h < cap.n_head; ++h) {
                        const float * a = attn.data() + (size_t)h * cap.n_kv;
                        float m_h = 0.0f;
                        for (int p = 0; p < cap.n_kv; ++p) if (a[p] > m_h) m_h = a[p];
                        // closed-form ramp μ(x) = 1.3 - 0.6·clip((x-0.4)/0.4, 0, 1)
                        float norm = (m_h - 0.4f) / 0.4f;
                        if (norm < 0.0f) norm = 0.0f;
                        if (norm > 1.0f) norm = 1.0f;
                        float mult = 1.3f - 0.6f * norm;
                        int K_h = (int)std::lround((float)args.k_nominal * mult);
                        if (K_h < 1) K_h = 1;
                        if (K_h > cap.n_kv) K_h = cap.n_kv;
                        // Top-K_h positions per head into the union
                        std::vector<int> idx(cap.n_kv);
                        std::iota(idx.begin(), idx.end(), 0);
                        std::nth_element(idx.begin(), idx.begin() + K_h, idx.end(),
                                         [&](int x, int y) { return a[x] > a[y]; });
                        for (int i = 0; i < K_h; ++i) spread_kept[idx[i]] = 1;
                    }
                }
            } else {
                // Simple-mean variant: all positions are candidates
                for (int p = 0; p < cap.n_kv; ++p) spread_kept[p] = 1;
            }

            // L2-style cross-head mean over the candidate pool
            std::vector<float> pos_score(cap.n_kv, 0.0f);
            int n_score_samples = 0;
            for (const auto & [layer, attn] : cap.per_layer) {
                for (int h = 0; h < cap.n_head; ++h) {
                    const float * a = attn.data() + (size_t)h * cap.n_kv;
                    for (int p = 0; p < cap.n_kv; ++p) pos_score[p] += a[p];
                }
                n_score_samples += cap.n_head;
            }
            if (n_score_samples > 0) {
                const float inv_n = 1.0f / (float)n_score_samples;
                for (auto & s : pos_score) s *= inv_n;
            }

            // Build keep mask: sink + last recent + top-eff_anchor by score in middle
            std::vector<uint8_t> keep(cap.n_kv, 0);
            for (int p = 0; p < args.n_sink && p < cap.n_kv; ++p) keep[p] = 1;
            const int recent_start = std::max(0, cap.n_kv - eff_recent);
            for (int p = recent_start; p < cap.n_kv; ++p) keep[p] = 1;

            std::vector<std::pair<float, int>> ranked;
            for (int p = args.n_sink; p < recent_start; ++p) {
                if (spread_kept[p]) ranked.emplace_back(pos_score[p], p);
            }
            int top = std::min<int>(eff_anchor, (int)ranked.size());
            if (top > 0 && top < (int)ranked.size()) {
                std::nth_element(ranked.begin(), ranked.begin() + top, ranked.end(),
                                 [](const auto & a, const auto & b) { return a.first > b.first; });
            }
            for (int i = 0; i < std::min(top, (int)ranked.size()); ++i) keep[ranked[i].second] = 1;

            // Drop unkept positions
            llama_memory_t mem = llama_get_memory(ctx);
            int dropped = 0, pp = 0;
            while (pp < cap.n_kv) {
                if (keep[pp]) { ++pp; continue; }
                int s = pp;
                while (pp < cap.n_kv && !keep[pp]) ++pp;
                llama_memory_seq_rm(mem, 0, s, pp);
                dropped += (pp - s);
            }
            evicted_prefill += dropped;
            std::fprintf(stderr, "[refresh-0] anchor refreshed: kept %d/%d, dropped %d more\n",
                         args.n_sink + std::min(top, (int)ranked.size()) + (cap.n_kv - recent_start),
                         cap.n_kv, dropped);
        }
    }

    // SnapKV-style two-context swap (--snapkv-decode): after prefill and eviction, move the
    // cache to an FA-on context so decode runs at vanilla speed. The same state get/set
    // round-trip also compacts the cache, since seq_rm leaves holes that decode still scans.
    // A failed restore falls back to the current context, so it is compaction or a no-op.
    const bool want_defrag = args.defrag_mode >= 0 ? (args.defrag_mode == 1)
                           : (args.n_gpu_layers == 0 || detect_gpu_profile_wants_defrag());
    // Compaction is a mechanism, not part of muKV's selection rule. An explicit
    // --compact-inplace or --force-defrag applies it to any non-vanilla policy, so
    // baselines can be compared with the same mechanism.
    if ((args.snapkv_decode || (args.fa_on_evict && want_defrag) || args.compact_inplace ||
         args.defrag_mode == 1) &&
        !args.no_state_swap && args.policy != "vanilla") {
        std::fprintf(stderr, "[mukv] scoring host state: %.1f MiB (%zu layers x n_head*n_kv floats)\n",
                     cap.peak_host / (1024.0*1024.0), cap.per_layer.size());
        int64_t t_swap_0 = ggml_time_us();
        // In-place chunked compaction. Like the round-trip below it makes the survivors
        // contiguous, but it reuses the tensors prefill allocated, so peak memory does not
        // rise. The round-trip's 2x peak does not fit for Phi-3 at 16K on the phone or at 64K.
        bool inplace_ok = false;
        if (args.compact_inplace) {
            const uint32_t live = llama_endurkv_compact_seq(llama_get_memory(ctx), 0,
                                                            (uint32_t) args.compact_chunk);
            if (live > 0) {
                g_compaction_applied = true;
                g_compaction_mode    = "inplace";
                inplace_ok           = true;
                std::fprintf(stderr, "[mukv] cache COMPACTED IN PLACE (%u cells, chunk=%d) in %.1f ms\n",
                             live, args.compact_chunk, (ggml_time_us() - t_swap_0) / 1000.0);
                if (args.reclaim_tail) {
                    g_rss_pre_reclaim_kb = read_rss_kb(pid);
                    const size_t got = llama_endurkv_reclaim_tail(llama_get_memory(ctx), 0);
                    g_rss_post_reclaim_kb = read_rss_kb(pid);
                    g_reclaimed_bytes += got;
                    std::fprintf(stderr, "[mukv] tail RECLAIMED %.1f MiB back to the OS "
                                 "(RSS %.1f -> %.1f MiB)\n", got / 1048576.0,
                                 g_rss_pre_reclaim_kb / 1024.0, g_rss_post_reclaim_kb / 1024.0);
                }
            } else {
                // declined (cache holds other sequences), fall through to the round-trip
                std::fprintf(stderr, "[mukv] in-place compaction declined; using state round-trip\n");
            }
        }
        if (!inplace_ok) {
        // Heap-buffered state transfer. A file-backed transfer (state_seq_save_file) caused
        // more UFS writes, because the large state file pushed other processes' pages to swap.
        size_t state_size = llama_state_seq_get_size(ctx, /*seq_id=*/0);
        std::fprintf(stderr, "[snapkv] state_size=%zu bytes (%.1f MiB)\n",
                     state_size, state_size / (1024.0 * 1024.0));
        std::vector<uint8_t> state_buf(state_size);
        size_t got = llama_state_seq_get_data(ctx, state_buf.data(), state_size, /*seq_id=*/0);
        if (got == 0) {
            std::fprintf(stderr, "[snapkv] state_get failed; falling back to FA-off decode\n");
        } else {
            // Build the FA-on context before freeing the FA-off one, so a failed restore
            // (e.g. a KV layout that differs across FA modes on Vulkan) falls back to
            // decoding over the evicted cache. Eviction stays in effect, only compaction is lost.
            // Two paths use this block. snapkv_decode switches FA-off prefill to FA-on
            // decode. fa_on_evict is FA-on in both, so the second context only makes the
            // survivors contiguous.
            cap.active = false;

            // Build FA-on context. No cb_eval (we don't need attention anymore).
            llama_context_params on_params = llama_context_default_params();
            // The destination must span the full n_ctx. llama_state_seq_set_data restores
            // cells with their original positions (needed for RoPE), and compaction makes
            // cells contiguous without renumbering them, so a smaller destination silently
            // corrupts the cache. The round-trip therefore peaks at two full caches.
            on_params.n_ctx           = cparams.n_ctx;
            on_params.n_threads       = cparams.n_threads;
            on_params.n_threads_batch = cparams.n_threads_batch;
            on_params.type_k          = cparams.type_k;
            on_params.type_v          = user_type_v;
            on_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
            on_params.cb_eval         = nullptr;
            on_params.cb_eval_user_data = nullptr;
            // The destination must place all surviving cells in one find_slot() call, since
            // llama_state_seq_set_data builds a single ubatch of cell_count tokens. The default
            // n_batch (2048) would cap compaction at about 2048 live cells, so size it here.
            {
                const int    hd_  = llama_model_n_head(model) > 0
                                    ? llama_model_n_embd(model) / llama_model_n_head(model) : 0;
                const double bk_  = (double)ggml_type_size(cparams.type_k) / ggml_blck_size(cparams.type_k);
                const double bv_  = (double)ggml_type_size(cparams.type_v) / ggml_blck_size(cparams.type_v);
                const double bpc_ = (double)llama_model_n_layer(model) * llama_model_n_head_kv(model) * hd_ * (bk_ + bv_);
                const uint32_t live = bpc_ > 0 ? (uint32_t)((double)state_size / bpc_) + 1 : cparams.n_batch;
                const uint32_t need = std::max<uint32_t>(cparams.n_batch, live + 64);
                on_params.n_batch  = need;
                on_params.n_ubatch = std::max<uint32_t>(cparams.n_ubatch, need);
                std::fprintf(stderr, "[mukv] compaction dest: n_batch=%u (live cells ~%u)\n", need, live);
            }
            llama_context * ctx_on = llama_init_from_model(model, on_params);
            if (!ctx_on) {
                std::fprintf(stderr, "[%s] second context alloc failed; %s\n",
                             args.fa_on_evict ? "mukv" : "snapkv",
                             args.fa_on_evict ? "compaction skipped, decode stays FA-on over the SPARSE cache"
                                              : "continuing FA-off decode");
            } else {
                size_t loaded = llama_state_seq_set_data(ctx_on, state_buf.data(), state_size, /*dest_seq_id=*/0);
                if (loaded == 0) {
                    // The backend (e.g. Vulkan/Adreno) cannot restore the FA-off KV into an
                    // FA-on context. Keep the current context and decode over the evicted
                    // cache. Eviction still applies, only the speedup is lost.
                    std::fprintf(stderr, "[%s] state_set failed; %s\n",
                                 args.fa_on_evict ? "mukv" : "snapkv",
                                 args.fa_on_evict ? "compaction skipped, decode stays FA-on over the SPARSE cache"
                                                  : "continuing FA-off decode (KV layout differs across FA modes)");
                    llama_free(ctx_on);
                } else {
                    // Swap succeeded: retire the FA-off context, adopt FA-on.
                    llama_free(ctx);
                    ctx = ctx_on;
                    int64_t t_swap_1 = ggml_time_us();
                    g_compaction_applied = true;
                    g_compaction_mode    = "roundtrip";
                    std::fprintf(stderr, "[%s] %s in %.1fms\n",
                                 args.fa_on_evict ? "mukv" : "snapkv",
                                 args.fa_on_evict ? "cache COMPACTED (FA-on throughout)"
                                                  : "swap done, decode now FA-on",
                                 (t_swap_1 - t_swap_0)/1000.0);
                    // Warm-up: re-decode the last prompt token so the new context's output
                    // buffer is filled for get_logits_ith (n_outputs is 0 after a state load).
                    // Remove that position from the KV first so it is not duplicated.
                    const llama_pos last_pos = (llama_pos)(n_prompt - 1);
                    llama_memory_seq_rm(llama_get_memory(ctx), 0, last_pos, last_pos + 1);
                    llama_batch warmup = llama_batch_get_one(ptoks.data() + (n_prompt - 1), 1);
                    if (llama_decode(ctx, warmup) != 0) {
                        std::fprintf(stderr, "[snapkv] warm-up decode failed; logits may be invalid\n");
                    } else {
                        std::fprintf(stderr, "[snapkv] warm-up decode ok; n_outputs populated\n");
                    }
                }
            }
        }
        }   // end if (!inplace_ok), the state round-trip path
    }

    // --no-state-swap and --fa-on-evict: eviction is already done, so freeze the capture.
    // Otherwise every decode token pays a host readback of kq_soft_max.
    if ((args.no_state_swap || args.fa_on_evict) && args.policy != "vanilla") {
        // --ppl-per-step keeps the FA-off capture alive so per-step re-scoring sees fresh
        // attention. fa_on_evict has no kq_soft_max, so there per-step eviction is a no-op.
        const bool keep_cap_for_ppl =
            args.eval_mode == "ppl" && args.ppl_per_step && !args.fa_on_evict;
        if (!keep_cap_for_ppl) cap.active = false;
        if (args.fa_on_evict) llama_endurkv_set_evict_obs_window(0);  // side node off for decode
        std::fprintf(stderr, "[%s] capture %s; %s decode over compacted cache\n",
                     args.fa_on_evict ? "fa-on-evict" : "no-state-swap",
                     keep_cap_for_ppl ? "kept ALIVE for --ppl-per-step" : "frozen",
                     args.fa_on_evict ? "FA-ON" : "FA-off");
    }

    int n_steps = 0; int eos_step = -1;
    double sum_nll = 0.0;
    int    n_total_evicted = 0;
    // v1_FA² tiered eviction: the anchored count is n_kv after all prefill evictions and
    // any state swap, taken before decode starts.
    int    n_anchored = (int)(llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1);
    // Retained cache after prefill eviction (and swap/defrag). llama_state_seq_get_size
    // serializes only live cells, so this is the KV size for every policy, unlike the
    // position-range peak_kv.
    size_t retained_kv_bytes = llama_state_seq_get_size(ctx, 0);
    std::fprintf(stderr, "[cache] retained_kv=%.2f MiB (TRUE compacted, post-prefill)\n",
                 retained_kv_bytes / (1024.0 * 1024.0));
    if (args.decode_tiered) {
        std::fprintf(stderr, "[v1_fa2] decode-tiered: n_anchored=%d, recent_budget=%d, total=%d\n",
                     n_anchored, args.recent_budget, n_anchored + args.recent_budget);
    }
    // Effective KV running accumulators
    double sum_mass_retained = 0.0;
    double sum_retention_ratio = 0.0;
    double sum_efficiency = 0.0;
    int    n_kv_samples = 0;

    // PPL mode: after prefill and eviction, teacher-force a reference text disjoint from
    // the prefill (KIVI/H2O-style eviction PPL, not llama-perplexity's sliding window). The
    // launcher passes chunk i as --prompt and chunk i+1 as --eval-text. Scores are raw
    // logit log-probs, with no sampling, temperature or repetition penalty.
    if (args.eval_mode == "ppl") {
        if (args.eval_text_file.empty()) {
            std::fprintf(stderr, "--eval-mode ppl requires --eval-text PATH\n");
            return 1;
        }
        std::string ref;
        if (!read_file(args.eval_text_file, ref)) return 1;
        int n_neg2 = llama_tokenize(vocab, ref.c_str(), ref.size(), nullptr, 0, false, true);
        int n_ref = -n_neg2;
        if (n_ref <= 1) { std::fprintf(stderr, "eval-text too short or tokenize fail\n"); return 1; }
        std::vector<llama_token> rtoks(n_ref);
        llama_tokenize(vocab, ref.c_str(), ref.size(), rtoks.data(), n_ref, false, true);

        int64_t t_eval_0 = ggml_time_us();
        double sum_ref_nll = 0.0;
        int    n_scored = 0;
        // --ppl-per-step runs the eviction block below after every teacher-forced token,
        // even with a frozen-mask preset, to match the per-step schedule of TOVA and H2O.
        // It is slow (FA-off) and must not be batched, or the schedule changes.
        const bool ppl_step_evict = args.ppl_per_step || !args.no_evict_decode;
        if (args.ppl_per_step) {
            std::fprintf(stderr,
                "[eviction_bench/ppl] --ppl-per-step: progressive per-step policy "
                "eviction ACTIVE during teacher-forcing (policy=%s, cap_active=%d)\n",
                args.policy.c_str(), cap.active ? 1 : 0);
        }
        // Feed one token at a time so logits_ith(-1) is the next-token distribution after
        // rtoks[0..i-1]. rtoks[0] is scored against the prefill's last position.
        for (int i = 0; i < n_ref; ++i) {
            const float * logits = llama_get_logits_ith(ctx, -1);
            if (!logits) break;
            // Raw softmax log-prob of rtoks[i], no penalties or temperature.
            float lmax = logits[0];
            for (int v = 1; v < n_vocab; ++v) if (logits[v] > lmax) lmax = logits[v];
            double Z = 0.0;
            for (int v = 0; v < n_vocab; ++v) Z += std::exp((double)logits[v] - (double)lmax);
            double log_prob = (double)logits[rtoks[i]] - (double)lmax - std::log(Z);
            sum_ref_nll += -log_prob;
            ++n_scored;

            // Per-step CSV for debugging the PPL trajectory (which tokens fail).
            int n_kv_now = (int)(llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1);
            char tok_text[128] = {0};
            int nb = llama_token_to_piece(vocab, rtoks[i], tok_text, sizeof(tok_text)-1, 0, true);
            if (nb > 0) tok_text[nb] = 0;
            // Replace commas and quotes in the token text with '_'
            for (int j = 0; tok_text[j]; ++j) if (tok_text[j] == '"' || tok_text[j] == ',') tok_text[j] = '_';
            std::fprintf(csv, "%d,%d,\"%s\",0,%d,0,%.6f,%.6f,0\n",
                         i, (int)rtoks[i], tok_text, n_kv_now, log_prob, -log_prob);

            // Feed this token to set up the next distribution.
            llama_token tok = rtoks[i];
            cap.reset();
            llama_batch batch = llama_batch_get_one(&tok, 1);
            if (llama_decode(ctx, batch) != 0) {
                std::fprintf(stderr, "eval-decode fail at i=%d\n", i);
                break;
            }
            // Keep evicting during the eval pass, as in generation. --no-evict-decode scores
            // under a frozen mask, --ppl-per-step forces per-step eviction. The NLL of
            // rtoks[i] came from the pre-feed logits, so eviction here only affects
            // rtoks[i+1], as in the gen loop. This mirrors the gen-mode per-step block.
            // Policies without a dispatch entry (or with capture frozen) do nothing.
            if (ppl_step_evict) {
                if (args.policy == "h2o") h2o_state.update(cap);
                if (args.policy == "v1_predictive") attn_hist.push(cap);
                PolicyResult r;
                if (args.policy == "v1")          r = policy_v1(cap, args.k_nominal, n_kv_heads);
                else if (args.policy == "v1_adaptive") r = policy_v1_adaptive(cap, args.k_nominal, n_kv_heads);
                else if (args.policy == "tova")   r = policy_tova(cap, args.k_nominal, n_kv_heads);
                else if (args.policy == "tova_canonical") r = policy_tova_canonical(cap, args.k_nominal, n_kv_heads);
                else if (args.policy == "pyramid") r = policy_pyramid(cap, args.k_nominal, n_layers_total);
                else if (args.policy == "h2o")     r = policy_h2o(cap, h2o_state, args.k_nominal, n_kv_heads);
                else if (args.policy == "streamingllm") r = policy_streamingllm(cap, args.k_nominal, args.n_sink);
                else if (args.policy == "adakv")   r = policy_adakv(cap, args.k_nominal, n_kv_heads);
                else if (args.policy == "v1_entropy") r = policy_v1_entropy(cap, args.k_nominal, n_kv_heads);
                else if (args.policy == "v1_predictive") r = policy_v1_predictive(cap, attn_hist, args.k_nominal, args.n_sink);
                if (!r.per_layer_keep.empty()) {
                    if (args.policy != "streamingllm") protect_sink(r, args.n_sink, cap.n_kv);
                    EffectiveKV ekv = compute_effective_kv(cap, r, cap.n_head);
                    sum_mass_retained   += ekv.mass_retained;
                    sum_retention_ratio += ekv.retention_ratio;
                    sum_efficiency      += ekv.efficiency;
                    n_kv_samples++;
                    n_total_evicted += apply_eviction(ctx, r, cap.n_kv);
                }
            }
        }
        int64_t t_eval_1 = ggml_time_us();
        double mean_ref_nll = (n_scored > 0) ? (sum_ref_nll / n_scored) : std::nan("");
        double ref_ppl      = (n_scored > 0) ? std::exp(mean_ref_nll)   : std::nan("");
        // Hijack the existing accumulators so the JSON output uses these.
        sum_nll = sum_ref_nll;
        n_steps = n_scored;
        std::fprintf(stderr, "[eviction_bench/ppl] scored=%d  mean_nll=%.4f  ppl=%.4f  eval_ms=%.1f\n",
                     n_scored, mean_ref_nll, ref_ppl, (t_eval_1 - t_eval_0)/1000.0);
    }

    // Repetition penalty window
    std::vector<int> recent_tokens;
    recent_tokens.reserve(args.repeat_last_n);
    // RNG seeded from args.seed for reproducible sampling
    std::mt19937 rng((uint32_t)args.seed);
    int64_t t_decode_0 = ggml_time_us();
    // PPL mode already scored the reference text above, so skip the sampling loop.
    int max_steps_loop = (args.eval_mode == "ppl") ? 0 : args.max_tokens;
    for (int step = 0; step < max_steps_loop; ++step) {
        const float * logits = llama_get_logits_ith(ctx, -1);
        if (!logits) break;
        // Standard llama.cpp protocol: top-p/top-k/temp + repeat-penalty sampling.
        // --greedy forces argmax for diagnostic / deterministic mode.
        llama_token tok;
        if (args.greedy) {
            tok = (llama_token)argmax_with_repetition_penalty(
                logits, n_vocab, recent_tokens, args.repeat_penalty);
        } else {
            tok = (llama_token)sample_token(
                logits, n_vocab, recent_tokens,
                args.repeat_penalty, args.temperature,
                args.top_p, args.top_k, rng);
        }
        // Update the recent-tokens ring buffer
        recent_tokens.push_back((int)tok);
        if ((int)recent_tokens.size() > args.repeat_last_n)
            recent_tokens.erase(recent_tokens.begin());
        // Compute log-prob of chosen token via softmax (numerically stable)
        float lmax = logits[0];
        for (int i = 1; i < n_vocab; ++i) if (logits[i] > lmax) lmax = logits[i];
        double Z = 0.0;
        for (int i = 0; i < n_vocab; ++i) Z += std::exp((double)logits[i] - (double)lmax);
        double log_prob = (double)logits[tok] - (double)lmax - std::log(Z);
        double nll = -log_prob;
        sum_nll += nll;

        char text[128] = {0};
        int nb = llama_token_to_piece(vocab, tok, text, sizeof(text)-1, 0, true);
        if (nb > 0) text[nb] = 0;
        if (gen) std::fprintf(gen, "%s", text);

        int64_t now = ggml_time_us();
        int n_kv_now = (int)(llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1);
        long rss_now = read_rss_kb(pid);
        if (n_kv_now > peak_kv) peak_kv = n_kv_now;
        if (rss_now > peak_rss) peak_rss = rss_now;

        ++n_steps;
        bool is_eog = llama_vocab_is_eog(vocab, tok);

        int evicted_this = 0;
        if (!is_eog) {
            cap.reset();
            llama_batch batch = llama_batch_get_one(&tok, 1);
            if (llama_decode(ctx, batch) != 0) {
                std::fprintf(csv, "%d,%d,\"%s\",%lld,%d,%ld,%.6f,%.6f,0\n",
                             step, (int)tok, text, (long long)(now - t_start),
                             n_kv_now, rss_now, log_prob, nll);
                break;
            }
            // Apply eviction policy after this decode step
            // --no-evict-decode: freeze the mask after prefill (SnapKV-style)
            if (!args.no_evict_decode) {
                if (args.policy == "h2o") h2o_state.update(cap);
                if (args.policy == "v1_predictive") attn_hist.push(cap);
                PolicyResult r;
                if (args.policy == "v1")          r = policy_v1(cap, args.k_nominal, n_kv_heads);
                else if (args.policy == "v1_adaptive") r = policy_v1_adaptive(cap, args.k_nominal, n_kv_heads);
                else if (args.policy == "tova")   r = policy_tova(cap, args.k_nominal, n_kv_heads);
                else if (args.policy == "tova_canonical") r = policy_tova_canonical(cap, args.k_nominal, n_kv_heads);
                else if (args.policy == "pyramid") r = policy_pyramid(cap, args.k_nominal, n_layers_total);
                else if (args.policy == "h2o")     r = policy_h2o(cap, h2o_state, args.k_nominal, n_kv_heads);
                else if (args.policy == "streamingllm") r = policy_streamingllm(cap, args.k_nominal, args.n_sink);
                else if (args.policy == "adakv")   r = policy_adakv(cap, args.k_nominal, n_kv_heads);
                else if (args.policy == "v1_entropy") r = policy_v1_entropy(cap, args.k_nominal, n_kv_heads);
                else if (args.policy == "v1_predictive") r = policy_v1_predictive(cap, attn_hist, args.k_nominal, args.n_sink);
                if (!r.per_layer_keep.empty()) {
                    if (args.policy != "streamingllm") protect_sink(r, args.n_sink, cap.n_kv);
                    // Compute effective-KV before applying eviction (full attention)
                    EffectiveKV ekv = compute_effective_kv(cap, r, cap.n_head);
                    sum_mass_retained   += ekv.mass_retained;
                    sum_retention_ratio += ekv.retention_ratio;
                    sum_efficiency      += ekv.efficiency;
                    n_kv_samples++;
                    evicted_this = apply_eviction(ctx, r, cap.n_kv);
                }
                n_total_evicted += evicted_this;
            }
            // TDAK (thermal-driven adaptive K): every thermal_poll_steps decode steps, map
            // DDR temperature through a 4-tier ladder to K_target. The tiered recent_budget
            // becomes K_target - n_sink - n_anchored.
            static int tdak_current_k = args.k_nominal;  // start at nominal
            static int tdak_step_last_poll = -1;
            static int tdak_ddr_mc_last = 0;
            static int tdak_tier_last = 0;  // 0=cool, 1=mid, 2=warm, 3=crit
            if (args.thermal_driven_k && (step - tdak_step_last_poll) >= args.thermal_poll_steps) {
                FILE * tfp = std::fopen(args.thermal_zone_path.c_str(), "r");
                int ddr_mc = 0;
                if (tfp) { std::fscanf(tfp, "%d", &ddr_mc); std::fclose(tfp); }
                tdak_ddr_mc_last = ddr_mc;
                // raw_tier = tier implied by the current DDR temperature (symmetric mapping)
                int raw_tier;
                if      (ddr_mc < args.tdak_t_warm) raw_tier = 0;
                else if (ddr_mc < args.tdak_t_hot ) raw_tier = 1;
                else if (ddr_mc < args.tdak_t_crit) raw_tier = 2;
                else                                 raw_tier = 3;
                int new_tier;
                if (args.tdak_ratchet) {
                    // Ratchet: escalate at once, but step down one tier only once DDR is hyst
                    // below that tier's entry threshold, so K does not grow while heat still rises.
                    const int entry_thr[4] = { 0, args.tdak_t_warm, args.tdak_t_hot, args.tdak_t_crit };
                    if (raw_tier > tdak_tier_last) {
                        new_tier = raw_tier;                       // escalate immediately
                    } else if (raw_tier < tdak_tier_last &&
                               ddr_mc < entry_thr[tdak_tier_last] - args.tdak_ratchet_hyst) {
                        new_tier = tdak_tier_last - 1;             // de-escalate one tier, with hysteresis
                    } else {
                        new_tier = tdak_tier_last;                 // hold
                    }
                } else {
                    new_tier = raw_tier;                           // symmetric ladder
                }
                int new_k;
                switch (new_tier) {
                    case 0:  new_k = args.tdak_k_cool;  break;
                    case 1:  new_k = args.tdak_k_mid;   break;
                    case 2:  new_k = args.tdak_k_warm;  break;
                    default: new_k = args.tdak_k_crit;  break;
                }
                if (new_tier != tdak_tier_last) {
                    std::fprintf(stderr, "[tdak] step=%d DDR=%.1fC tier=%d->%d K=%d->%d\n",
                                 step, ddr_mc/1000.0, tdak_tier_last, new_tier,
                                 tdak_current_k, new_k);
                    tdak_tier_last = new_tier;
                }
                tdak_current_k = new_k;
                tdak_step_last_poll = step;
            }
            // K and recent_budget for this step's decode eviction. With TDAK on, both
            // follow tdak_current_k.
            int eff_recent_budget = args.recent_budget;
            int eff_k_nominal     = args.k_nominal;
            if (args.thermal_driven_k) {
                eff_k_nominal = tdak_current_k;
                int rb = tdak_current_k - args.n_sink - (n_anchored > 0 ? n_anchored : 0);
                eff_recent_budget = std::max(32, rb);
            }
            // KeyDiff decode-time re-eviction. KeyDiff bounds the cache during generation too
            // (B = 1, Park et al., NeurIPS 2025, Sec. 2.4), so one end-of-prefill selection
            // would let the cache grow by every decode token. A re-score is O(n*d) per layer,
            // so this re-evicts every keydiff_decode_block steps (default 128, their prefill
            // block size) and the cache stays in [N, N + B] instead of exactly N. Scores come
            // from the current cache, so decode tokens compete with prompt tokens.
            // The cadence uses the step counter, not seq_pos_max(), because in-place
            // compaction does not renumber positions.
            static int kd_last_evict_step = -1;
            if (args.policy == "keydiff" && args.keydiff_decode_block > 0 &&
                (step - kd_last_evict_step) >= args.keydiff_decode_block) {
                const int n_kv_after = (int)(llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1);
                {
                    kd_last_evict_step = step;
                    std::vector<float> kds((size_t) n_kv_after, 0.0f);
                    const uint32_t kn = llama_endurkv_keydiff_scores(
                            llama_get_memory(ctx), 0, kds.data(), (uint32_t) kds.size());
                    if (kn > (uint32_t) args.k_nominal) {   // kn is the live dense-prefix length
                        PolicyResult r;
                        r.n_layers = llama_model_n_layer(model);
                        r.per_layer_keep.resize(r.n_layers);
                        std::vector<int> idx(kn);
                        std::iota(idx.begin(), idx.end(), 0);
                        const int K_keep = std::min<int>(args.k_nominal, (int) kn);
                        std::nth_element(idx.begin(), idx.begin() + K_keep, idx.end(),
                                         [&](int a, int b) { return kds[a] > kds[b]; });
                        std::unordered_set<int> keep;
                        keep.reserve(K_keep);
                        for (int i = 0; i < K_keep; ++i) keep.insert(idx[i]);
                        for (auto & lk : r.per_layer_keep) lk = keep;
                        const int dropped = apply_eviction(ctx, r, (int) kn);
                        // Re-compact so the next scoring call sees the dense prefix it
                        // requires. It declines otherwise, which disables decode eviction.
                        if (dropped > 0 && args.compact_inplace) {
                            llama_endurkv_compact_seq(llama_get_memory(ctx), 0,
                                                      (uint32_t) args.compact_chunk);
                            if (args.reclaim_tail) {
                                g_reclaimed_bytes += llama_endurkv_reclaim_tail(llama_get_memory(ctx), 0);
                            }
                        }
                        evicted_this    += dropped;
                        n_total_evicted += dropped;
                    }
                }
            }
            // v1_FA² tiered decode eviction (takes precedence over decode_bound). Keeps the
            // prefill anchors and a recent window, drops the oldest decode-time cells.
            else if (args.decode_tiered && args.no_evict_decode && n_anchored > 0) {
                int n_kv_after = (int)(llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1);
                int dropped = apply_tiered_decode_eviction(ctx, n_kv_after,
                                                           n_anchored, eff_recent_budget);
                evicted_this    += dropped;
                n_total_evicted += dropped;
            }
            // Recency-only decode bound (v1_fa enables it), used when the paths above are not.
            else if (args.decode_bound && args.no_evict_decode && eff_k_nominal > 0) {
                int n_kv_after = (int)(llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1);
                int n_recent   = eff_k_nominal - args.n_sink;
                if (n_recent < 1) n_recent = eff_k_nominal;
                int dropped = apply_recency_decode_eviction(ctx, n_kv_after,
                                                            args.n_sink, n_recent);
                evicted_this   += dropped;
                n_total_evicted += dropped;
            }
        }
        // Now write the per-step row with the correct evicted_this count
        std::fprintf(csv, "%d,%d,\"%s\",%lld,%d,%ld,%.6f,%.6f,%d\n",
                     step, (int)tok, text, (long long)(now - t_start),
                     n_kv_now, rss_now, log_prob, nll, evicted_this);
        // Stop at EOS unless --ignore-eos, which records it and keeps generating.
        if (is_eog) {
            if (eos_step < 0) eos_step = step;
            if (!args.ignore_eos) break;
        }
    }
    int64_t t_decode_1 = ggml_time_us();
    cap.active = false;
    if (gen) std::fclose(gen);
    std::fclose(csv);

    // summary meta
    double prefill_ms = (t_prefill_1 - t_prefill_0) / 1000.0;
    double decode_ms  = (t_decode_1 - t_decode_0) / 1000.0;
    double decode_tps = n_steps > 0 ? (double)n_steps * 1000.0 / decode_ms : 0.0;
    double total_ms   = (t_decode_1 - t_start) / 1000.0;
    // Perplexity over the generated tokens (geometric-mean per-token NLL)
    double mean_nll = n_steps > 0 ? sum_nll / (double)n_steps : 0.0;
    double perplexity = std::exp(mean_nll);
    double mean_mass_retained   = n_kv_samples > 0 ? sum_mass_retained / n_kv_samples : 1.0;
    double mean_retention_ratio = n_kv_samples > 0 ? sum_retention_ratio / n_kv_samples : 1.0;
    double mean_efficiency      = n_kv_samples > 0 ? sum_efficiency / n_kv_samples : 1.0;
    // KV-cache MB (fp16): n_layers × n_kv_heads × head_dim × n_cells × 2 (K+V) × 2 bytes
    int head_dim = llama_model_n_embd(model) / llama_model_n_head(model);
    double kv_bytes_per_cell = 2.0 /* K+V */ * (double)n_layers_total * (double)n_kv_heads * (double)head_dim * 2.0 /* fp16 */;
    double peak_kv_mb = peak_kv * kv_bytes_per_cell / (1024.0*1024.0);

    // Render doubles as valid JSON: NaN/Inf become null (bare "nan" is not valid JSON).
    auto jd = [](double v) -> std::string {
        if (std::isnan(v) || std::isinf(v)) return std::string("null");
        char buf[32]; std::snprintf(buf, sizeof(buf), "%.6f", v);
        return std::string(buf);
    };

    std::FILE * mf = std::fopen(args.out_meta.c_str(), "w");
    if (mf) {
        std::fprintf(mf,
            "{\n"
            "  \"prompt_id\": \"%s\",\n"
            "  \"model\": \"%s\",\n"
            "  \"policy\": \"%s\",\n"
            "  \"k_nominal\": %d,\n"
            // obs_window and snapkv_kernel are recorded so a baseline's configuration can be
            // checked from its own output.
            "  \"obs_window\": %d,\n"
            "  \"snapkv_kernel\": %d,\n"
            "  \"ctx_size\": %d,\n"
            "  \"n_prompt_tokens\": %d,\n"
            "  \"n_decode_steps\": %d,\n"
            "  \"eos_step\": %d,\n"
            "  \"n_layers\": %d,\n"
            "  \"n_kv_heads\": %d,\n"
            "  \"prefill_ms\": %.3f,\n"
            "  \"decode_ms\": %.3f,\n"
            "  \"decode_tps\": %.3f,\n"
            "  \"total_ms\": %.3f,\n"
            "  \"peak_rss_kb\": %ld,\n"
            "  \"reclaimed_bytes\": %zu,\n"
            "  \"rss_pre_reclaim_kb\": %ld,\n"
            "  \"rss_post_reclaim_kb\": %ld,\n"
            "  \"rss_end_kb\": %ld,\n"
            "  \"peak_kv_cells\": %d,\n"
            "  \"peak_kv_mb\": %.3f,\n"
            "  \"head_dim\": %d,\n"
            "  \"evicted_prefill\": %d,\n"
            "  \"evicted_total_decode\": %d,\n"
            "  \"retained_kv_bytes\": %zu,\n"
            "  \"compaction_applied\": %s,\n"
            "  \"compaction_mode\": \"%s\",\n"
            "  \"fa_off_gate\": \"%s\",\n"
            "  \"energy_aware\": %s,\n"
            "  \"ea_soc\": %d,\n"
            "  \"ea_level\": %d,\n"
            "  \"ea_status\": \"%s\",\n"
            "  \"ea_gpu_mhz\": %d,\n"
            "  \"ea_gpu_write_rc\": %d,\n"
            "  \"gpu_mhz_decode\": %d,\n"
            "  \"score_host_mb\": %.1f,\n"
            "  \"mean_nll\": %s,\n"
            "  \"perplexity\": %s,\n"
            "  \"mean_mass_retained\": %s,\n"
            "  \"mean_retention_ratio\": %s,\n"
            "  \"mean_eviction_efficiency\": %s,\n"
            "  \"n_sink\": %d,\n"
            "  \"repeat_penalty\": %.3f,\n"
            "  \"fa_vanilla_enabled\": %s,\n"
            "  \"no_evict_decode\": %s,\n"
            "  \"ppl_per_step\": %s,\n"
            "  \"sampling\": \"%s\",\n"
            "  \"temperature\": %.3f,\n"
            "  \"top_p\": %.3f,\n"
            "  \"top_k\": %d\n"
            "}\n",
            args.prompt_id.c_str(), args.model.c_str(),
            args.user_policy.empty() ? args.policy.c_str() : args.user_policy.c_str(),
            args.k_nominal, args.obs_window, args.snapkv_kernel, args.ctx_size,
            n_prompt, n_steps, eos_step, n_layers_total, n_kv_heads,
            prefill_ms, decode_ms, decode_tps, total_ms, peak_rss,
            g_reclaimed_bytes, g_rss_pre_reclaim_kb, g_rss_post_reclaim_kb, read_rss_kb(pid),
            peak_kv,
            peak_kv_mb, head_dim, evicted_prefill, n_total_evicted, retained_kv_bytes,
            g_compaction_applied ? "true" : "false",
            g_compaction_mode,
            g_fa_off_gate,
            args.energy_aware ? "true" : "false",
            args.ea_soc_seen, args.ea_level_seen, args.ea_status_seen.c_str(),
            args.ea_gpu_mhz_set, args.ea_gpu_write_rc, args.gpu_mhz_decode,
            cap.peak_host / (1024.0*1024.0),
            jd(mean_nll).c_str(), jd(perplexity).c_str(),
            jd(mean_mass_retained).c_str(), jd(mean_retention_ratio).c_str(), jd(mean_efficiency).c_str(),
            args.n_sink, args.repeat_penalty,
            (args.fa_vanilla ? "true" : "false"),
            (args.no_evict_decode ? "true" : "false"),
            (args.ppl_per_step ? "true" : "false"),
            (args.greedy ? "greedy" : "top-p"),
            args.temperature, args.top_p, args.top_k);
        std::fclose(mf);
    }

    std::fprintf(stderr,
        "[eviction_bench] policy=%s K=%d prompt=%s "
        "n_prompt=%d steps=%d eos=%d prefill=%.1fms decode_tps=%.1f peak_kv=%d peak_rss=%ldkb\n",
        args.policy.c_str(), args.k_nominal, args.prompt_id.c_str(),
        n_prompt, n_steps, eos_step, prefill_ms, decode_tps, peak_kv, peak_rss);

    // Energy-aware GPU action: hand the clock back to the device max when we lowered it.
    if ((args.ea_gpu_mhz_set > 0 && args.ea_gpu_mhz_set != args.ea_gpu_mhz_hi) || args.gpu_mhz_decode_rc == 0) {
        const int rc = gpu_cap_write(args.ea_gpu_mhz_hi);   // level 0 and the top clock
        std::fprintf(stderr, "[energy-aware] GPU clock restored to %d MHz (rc=%d)\n", args.ea_gpu_mhz_hi, rc);
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
