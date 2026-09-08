// eviction_bench.cpp — on-phone benchmark tool for KV-cache eviction policies.
//
// Loads a gguf model, decodes a prompt, applies a chosen eviction policy after
// each decode step, logs per-step latency, KV-cache-used cells, and final
// summary stats (prefill ms, decode tok/s, peak KV cells).
//
// Policies:
//   vanilla   no eviction (cache grows; relies on context size)
//   tova      per-head top-K by current attention (fixed K)
//   pyramid   per-layer K (linear K_max -> K_min across depth), TOVA selection
//   v1        EndurKV-Evict per-head spread gate (α=1.3 β=0.6 on max_a)
//
// GQA handling: aggregate-OR at kv-head level (positions kept if ANY query
// head sharing that kv-head wants them).
//
// Eviction granularity in this version: sequence-level (llama_kv_self_seq_rm).
// A position is removed from the sequence only if NO layer/head wants it.
// This is conservative but matches the only public API in llama.cpp.
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
// 2026-07-25 μKV×VL: optional multimodal prefill via libmtmd (image tokens enter
// the same KV cache; the frozen μKV mechanism applies unchanged). Compiled only
// when the build links libmtmd (EVB_HAS_MTMD from CMakeLists); phone/text-only
// builds are bit-identical without it.
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

// ---------------------------------------------------------------------------
// Args
// ---------------------------------------------------------------------------

// GPU cap write that survives the vendor thermal engine. max_gpuclk lands in kgsl's
// thermal_pwrlevel, the node the OEM thermal engine also owns; when that engine changes
// state mid-request (seen 2026-09-05: it cleared a decode cap at 140 s on a cool phone) our
// cap is gone. max_pwrlevel is the user cap, composed with the thermal cap by max(), so we
// write both: the level index of mhz in gpu_available_frequencies, and the clock.
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
    int         n_ubatch = 64;        // micro-batch — keep small for Adreno Vulkan (TDR)
    int         n_sink = 4;           // sink-token protection (StreamingLLM-style)
    int         repeat_last_n = 64;   // repetition window
    float       repeat_penalty = 1.1; // llama.cpp default
    bool        fa_vanilla = true;    // enable FA for vanilla (gives realistic latency baseline)
    // DIAGNOSTIC FLAG, added 2026-08-13. Attaches the attention-capture callback to a run
    // that would not otherwise install it (notably vanilla). Its only purpose is to
    // isolate the Adreno/Vulkan corruption: five FA-off policies emit degenerate tokens
    // there while vanilla FA-off stays clean, and vanilla is also the ONE configuration
    // that never installs cb_eval -- so "the callback corrupts the graph" and "those
    // policies are broken" predict the same observation. Running vanilla WITH the callback
    // separates them: if it corrupts, the callback is the cause. Not for measurement runs.
    bool        force_cb_eval = false;
    // Second diagnostic. Returning true from the ask phase makes ggml_backend_sched END A
    // SPLIT at that node and materialize it; only afterwards do we read it. Those are two
    // separate suspects for the Vulkan corruption. With this flag the callback still
    // returns true (so the split still happens) but performs NO tensor read, which tells
    // them apart: still corrupt => the graph split is the cause and the capture must move
    // off the mid-graph intercept; clean => the readback itself is what perturbs it.
    bool        cb_eval_noread = false;
    bool        no_evict_decode = false;  // if true: evict only after prefill, frozen thereafter
    bool        no_perhead_gate = false;  // 2026-09-07: bypass the per-head budget gate (policy_v1 ramp); the candidate pool is every position, selection is the pooled cross-head score alone
    bool        ignore_eos = false;   // if true: don't break on EOS, generate full max_tokens (for sustained-throughput benches where chat templates would otherwise terminate after a few tokens)
    int         keydiff_decode_block = 128; // KeyDiff only: re-evict every N decode steps to hold the cache at its budget
                                            // (their B=1 generation cadence, blocked for cost -- see the decode-loop comment).
                                            // 0 disables, which reverts to single-shot end-of-prefill selection.
    bool        decode_bound = false; // if true: during decode, bound the KV cache via cheap recency eviction (no attention needed). Designed for v1_fa-style FA-on decode in long-generation workloads. When n_kv > 1.5 * k_nominal, drop middle positions, keep [0, n_sink) and [n_kv - (k_nominal - n_sink), n_kv).
    bool        decode_tiered = false;// v1_FA² mode: preserve ALL prefill-anchored positions (the attention-aware selection from prefill) AND maintain a recent window. During decode, only drop OLDEST new-during-decode positions. Better PPL than --decode-bound at same K.
    int         recent_budget = 256;  // for tiered decode: size of recent window (in tokens). Total cache = n_anchored + recent_budget. Triggers eviction when n_kv > 1.25*(n_anchored + recent_budget).
    int         anchor_top_k = 0;     // selective anchoring: if > 0, after prefill eviction, keep only the top anchor_top_k positions by mean attention score. Reduces anchored block size so recent_budget can be larger. 0 = keep all prefill survivors as anchored.
    std::string anchor_score_mode = "mean"; // "mean": top-K by sum/mean attention received (current default). "entropy": top-K by cross-(layer,head) entropy of attention received at each position (broadly-attended positions kept). "neg_entropy": invert (sharply-attended positions kept). "hybrid": mean attention * (1 + hybrid_alpha * sharpness) where sharpness = 1 - normalized_entropy. Preserves magnitude as primary signal and boosts sharply-attended positions (specialist-head signal).
    float       hybrid_alpha = 0.5f;        // weight of the sharpness term in hybrid scoring mode. 0 = pure mean, 1 = doubles sharply-attended position scores.
    std::string gate_mode = "v1";           // "v1": current single-statistic gate using m_h = max attention per head. "v2": two-statistic gate using s_h = m_h + 0.5*m'_h (top-1 + 0.5*top-2) to capture double-peaked heads as sharp.
    float       anchor_coverage = 0.0f;     // ADAPTIVE ANCHOR: if > 0, the anchor selection step keeps positions until cumulative attention mass >= anchor_coverage (e.g., 0.95 = keep enough positions to explain 95% of mean cross-head attention mass). Variable count per prompt, principled. Overrides anchor_top_k when > 0.
    bool        thermal_driven_k = false;   // THERMAL-DRIVEN ADAPTIVE K (TDAK): close the loop between cache size and hardware thermal state. Every thermal_poll_steps decode steps, read DDR temperature from /sys/class/thermal/thermal_zone47/temp, map to K_target via a 4-tier ladder, and evict the cache to K_target. Closes L1 (cache) <-> L5 (thermal) loop with cache size as a new actuator.
    int         thermal_poll_steps = 20;    // Decode steps between thermal polls. 20 steps at ~5 tps = 4s thermal cadence. Cheap (~50us per poll).
    int         tdak_k_cool = 1024;         // K_target when DDR < tdak_t_warm
    int         tdak_k_mid  = 768;          // K_target when tdak_t_warm <= DDR < tdak_t_hot
    int         tdak_k_warm = 512;          // K_target when tdak_t_hot <= DDR < tdak_t_crit
    int         tdak_k_crit = 256;          // K_target when DDR >= tdak_t_crit (emergency, avoid kernel cliff)
    int         tdak_t_warm = 60000;        // DDR threshold (mC) for stepping cool -> mid
    int         tdak_t_hot  = 63000;        // DDR threshold for stepping mid -> warm
    int         tdak_t_crit = 65000;        // DDR threshold for stepping warm -> crit (matches kernel BCL trigger)
    bool        tdak_ratchet = false;       // TDAK RATCHET-DOWN: if set, the tier is monotone non-decreasing (K only shrinks) except it may step back down one tier once DDR has cooled at least tdak_ratchet_hyst mC BELOW the current tier's entry threshold. Preempts the symmetric-ladder thermal-mass lag (K never ramps back up into rising heat). Default off = original symmetric behavior.
    int         tdak_ratchet_hyst = 1000;   // hysteresis band (mC) for ratchet step-down; 1000 mC = 1 C.
    std::string thermal_zone_path = "/sys/class/thermal/thermal_zone47/temp"; // DDR sensor
    int         obs_window = 1;  // OBSERVATION WINDOW for attention scoring. 1 = original (last query only). 32 = AdaKV-style. Averages attention across the last N prefill queries → richer signal for retrieval-heavy workloads.
    int         snapkv_pool = 0; // SnapKV 1D max-pool kernel over the per-position scores (0/1 = off). The SnapKV paper max-pools the observation-window attention (kernel ~7-13) so top-K selects coherent CLUSTERS of tokens, not isolated spikes. Set to make the SnapKV baseline faithful.
    bool        adaptive_anchor = false;  // ADAPTIVE ANCHOR DISPATCHER (Signal 1, distant sharpness). When true, ignore --anchor-top-k / --recent-budget and compute α_a per prompt from per-head attention peaks: count fraction of (layer,head) pairs whose argmax sits outside the recent-default window of width adaptive_rmin. n_anchor = round(α_a · (K - n_sink)); n_recent = K - n_sink - n_anchor. Auto-dispatches between throughput corner (chat) and quality corner (NIAH/QA) per prompt.
    int         adaptive_rmin = 32;       // "distant" threshold (tokens) for the adaptive-anchor dispatcher. A head's peak is counted as "distant" (anchor-worthy) if argmax_p < N - adaptive_rmin. Defaults to 32 (last-32 tokens treated as the local recent window).
    bool        gate_count = false;       // If true, set adaptive-anchor α_a from the COUNT metric (fraction of heads whose argmax is distant) instead of the MASS metric (fraction of attention mass that is distant). Count is anchor-heavy (~0.79-0.95); mass is recent-heavier (~0.54-0.78). Lets us A/B the gate at equal K.
    // ADDED 2026-08-07: express the budget as a PERCENTAGE of the prompt instead of an
    // absolute cell count. k_nominal is an absolute input and the adaptive machinery
    // never changes it -- alpha_a only splits K between anchor and recent, so a K set
    // for a 12K prompt silently becomes a no-op at 57K (K=65536 retains 92.5% of cells
    // and is worth 1.03x). A percentage makes the budget scale with context by
    // construction and states the deployment constraint directly ("never hold more than
    // 20% of the cache"). 0 = disabled, use k_nominal as given.
    float       k_pct = 0.0f;
    // ADDED 2026-08-11, REWRITTEN 2026-09-02: ENERGY-AWARE BUDGET. Sets the cache budget
    // K from the phone's state of charge. The healthy tier IS the shipped --k-nominal
    // (the quality-validated operating point, 1024 by default), so mains power or a full
    // battery gets the best answers the policy is validated for. Lower tiers descend from
    // it by fixed fractions (1/2, 1/4). The ladder is IDENTICAL on both backends.
    //
    // Why the rewrite: the first ladder was percent-of-prompt (GPU 20/10/5, CPU 5/5/5),
    // chosen on throughput and joules only. It silently discarded --k-nominal, never
    // ran muKV at its own headline K, and on CPU gave a full battery the same smallest
    // cache as an empty one, so the battery reading changed nothing. Its throughput
    // figures also came from an unidentified campaign that disagrees with both ea_n3 and
    // ea_proof on ordering and on energy scale. See run_energy_aware_proof.sh.
    //
    // Energy context (OnePlus 15, Adreno 840, rail+coulomb corrected): vanilla 6.38 W,
    // muKV+compaction 5.58 W, muKV no-compaction 4.75 W. The battery supplements the
    // USB rail, so rail-only readings understate the cache's effect on power.
    // OFF BY DEFAULT: a policy that degrades output quality must be opt-in.
    bool        energy_aware   = false;
    int         ea_soc_hi      = 50;    // above this SoC: no degradation
    int         ea_soc_lo      = 20;    // below this SoC: most aggressive tier
    int         ea_k_hi        = 0;     // absolute K at high charge; 0 = k_nominal
    int         ea_k_mid       = 0;     // 0 = k_nominal / 2
    int         ea_k_lo        = 0;     // 0 = k_nominal / 4
    // PER-BACKEND ACTION (2026-09-02). The lever that saves energy differs by backend, and
    // the measurements say which:
    //   GPU: decode is bandwidth-bound, KV is small next to the weights, and the cache tier
    //        moves energy per token by <7% in three campaigns while costing 6.6 F1 on qasper.
    //        The clock cap trades time-to-first-token for prefill energy: 1200->902 MHz cut
    //        335->281 mJ/token at K=1024 with decode throughput unchanged (30.0 vs 30.3
    //        tok/s) and decode energy nearly so (790 vs 759 J), while prefill took 24% longer
    //        (137->171 s on 9737 tokens) at 2.3 W instead of 4.2 W, i.e. -32% prefill energy
    //        (576->391 J). 726 MHz: prefill +53%, decode -6%, only 2% more saving. So on GPU
    //        the ladder caps the GPU clock and HOLDS K.
    //   CPU: decode is attention-bound over live cells, so the cache tier is the lever
    //        (9741->721 cells was -61% energy) and the clock is not (power ~ f^1.1, J/token
    //        flat). So on CPU the ladder shrinks K and leaves the clock alone.
    // The GPU clock write needs root (su); when it fails the controller says so and holds.
    int         ea_gpu_mhz_hi  = 1200;  // GPU max clock at high charge / mains (device max)
    int         ea_gpu_mhz_mid = 902;
    int         ea_gpu_mhz_lo  = 726;
    bool        ea_gpu_k_ladder = false; // measurement only: also shrink K on GPU
    int         ea_gpu_mhz_set = 0;     // observed, for meta.json (0 = no write attempted)
    int         ea_gpu_write_rc = -1;   // return code of the clock write (0 = ok)
    // PHASE-AWARE CLOCK (2026-09-03). The clock cap costs time only in prefill (compute-bound)
    // but saves energy in both phases, so a lower cap during decode alone is nearly free in
    // time. --gpu-mhz-decode N writes N MHz right after prefill, before the first decode step;
    // the exit path restores the device max. 0 = off (one clock for the whole request).
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
    // Escape hatch to reproduce the old FA-off StreamingLLM numbers.
    bool        no_fa_positional = false;
    float       gate_alpha_floor = 0.0f;  // If >0, clamp α_a up to at least this floor (guards the anchor budget on retrieval prompts where mass α_a collapses to ~0.54).
    float       gate_alpha_max = 0.0f;     // If >0, clamp α_a down to at most this ceiling (guards the recent window from being starved when α_a runs high — protects fluency/long-decode coherence). Deployed gate = clamp(mass, floor, max).
    bool        refresh_at_decode_0 = false;  // RefreshKV-inspired (simple-mean variant): after prefill + initial anchor selection, run 1 FA-off decode step to capture attention from the model's "first generation" query, then re-apply anchor selection using mean attention from this fresh attention. Then state-swap to FA-on for the rest of decode. Tests whether decode-time queries carry additional signal beyond prefill scoring.
    bool        refresh_mukv_gate = false;  // μKV-style refresh: same as --refresh-at-decode-0 BUT re-applies the FULL μKV gate logic at refresh time (L1 per-head K_h ramp + L7 adaptive α_a dispatcher + L2 union/cross-head anchor selection) instead of simple mean attention. Lets L7 decide the new anchor:recent split from the decode-time attention pattern, potentially switching from quality corner to throughput corner mid-decode if the model's focus has shifted from retrieval to generation.
    std::string user_policy;          // original policy name as supplied on the CLI (before v1_fa2→v1_fa alias rewrite); reported in meta.json so downstream analysis can distinguish variants.
    std::string cache_type_k = "f16"; // KV cache K data type (f16, q8_0, q4_0). q8_0 cuts DRAM bandwidth ~2× → cooler DDR.
    std::string cache_type_v = "f16"; // KV cache V data type.
    // --- sampling (standard llama.cpp protocol; applies IDENTICALLY to all policies) ---
    bool        greedy = false;       // if true: argmax instead of sampling
    float       temperature = 0.8f;   // llama.cpp default
    float       top_p = 0.95f;        // nucleus
    int         top_k = 40;           // top-k filter (0 = disabled)
    // --- evaluation mode ---
    // "gen" : sample tokens with the standard pipeline (default; produces gen.txt)
    // "ppl" : after prefill+post-prefill-eviction, teacher-force eval_text and
    //         compute mean -log P(token|prefix) from RAW logits (no sampling,
    //         no rep-penalty, no temperature). This is the llama-perplexity
    //         methodology and the only PPL number you should publish.
    std::string eval_mode = "gen";
    std::string eval_text_file;
    // --- multimodal (μKV×VL, 2026-07-25; requires an EVB_HAS_MTMD build) ---
    std::string mmproj;                  // --mmproj PATH: vision projector gguf; empty = text-only (default)
    std::vector<std::string> images;     // --image PATH (repeatable): images injected before the prompt text
    // Vision encoder placement. MUST match llama.cpp's default (mtmd_context_params_default
    // sets use_gpu = true) so the baseline is a true default-config baseline. This is a
    // PLATFORM/harness setting, identical for every policy — never an μKV mechanism, and
    // never to be reported as an μKV benefit. (It was briefly defaulted to CPU on
    // 2026-07-25 after an NvMap OOM that was actually page-cache pressure from model
    // downloads; on CPU the encoder cost 184–242 s with ~30% run-to-run variance and
    // swamped the ~5.4 s LLM prefill being measured. Corrected 2026-07-26.)
    bool        mmproj_gpu = true;       // --mmproj-cpu to opt out (e.g. if VRAM-constrained)
    // --ppl-per-step: progressive teacher-forced PPL. In "ppl" mode, FORCE the
    // per-step policy-eviction machinery (the same block the gen-mode decode
    // loop runs after every sampled token) during the teacher-forced eval pass,
    // even when a policy preset or an explicit --no-evict-decode froze the mask.
    // This scores progressive-eviction baselines (TOVA, TOVA-canonical, H2O)
    // under their papers' own one-eviction-per-generation-step schedule instead
    // of a one-shot end-of-prefill eviction with a frozen mask. Default false:
    // without the flag the PPL path behaves exactly as before.
    bool        ppl_per_step = false;
    // Strategy A — SnapKV-style two-context decode:
    //   --snapkv-decode = true
    //     1. Prefill in an FA-off context (attention captured, eviction applied).
    //     2. llama_state_seq_get_data → blob.
    //     3. Destroy FA-off ctx; create FA-on ctx.
    //     4. llama_state_seq_set_data ← blob.
    //     5. Decode on the FA-on ctx (no further eviction).
    // Implies --no-evict-decode (frozen mask) since FA-on doesn't expose kq_soft_max.
    bool        snapkv_decode = false;
    int         snapkv_kernel = 5;      // 2026-08-01: SnapKV's avgpool kernel. The paper retunes per
                                       // experiment: NiaH uses window 16 / kernel 5, LongBench window 32 /
                                       // kernel 7, Command-R window 64 / kernel 13. Our baseline had these
                                       // HARDCODED at the FasterDecoding repo default (64/5) and --obs-window
                                       // was never passed through to policy_snapkv, so the flag was inert.
                                       // Now both are configurable so the baseline can be run at the paper's
                                       // own NiaH setting rather than only the repo default.
    // -1 = auto (decide from the active GPU backend), 0 = off, 1 = on.
    int         defrag_mode   = -1;  // -1 = auto (per GPU backend), 0 = off, 1 = on. See detect_gpu_profile_wants_defrag().
    // ADDED 2026-08-07: a SECOND compaction mode. defrag_mode==1 round-trips the cache
    // through llama_state_seq_get/set_data into a SECOND context, so its peak is 2x the
    // cache -- that is what stops Phi-3 compacting at ctx 16384 on the phone (15.28 GB
    // vs 15.1 GB RAM) and bounds 64K on the RTX. --compact-inplace slides the survivors
    // down inside the tensors prefill ALREADY allocated, so peak memory does not rise.
    // compact_chunk is the staging granularity in cells (~256 KB of host buffer at 512).
    int         compact_inplace = 0;
    // EndurKV (2026-08-31): after compaction, madvise(MADV_DONTNEED) the dead tail
    // so the pages go back to the OS. Compaction alone does NOT lower footprint --
    // the cache is one buffer sized for the full context and memset at init, so
    // residency is fixed at allocation. This is what converts the bandwidth win
    // into a footprint win. Host (CPU) caches only.
    int         reclaim_tail = 0;
    int         compact_chunk   = 512;
                                       // Without compaction the evicted cache stays physically SPARSE: seq_rm
                                       // frees cells but never moves survivors, and attention still scans to the
                                       // highest occupied index, so a 10x smaller cache buys nothing. That was
                                       // the sole reason the GPU realized only 10-21% of its memory roofline.
                                       // MEASURED 2026-08-02 on Adreno 840: the round-trip costs 3.9s (Llama-1B)
                                       // / 6.1s (Phi-3) once and returns 2.28x decode thereafter, so the auto
                                       // default is now ON for Vulkan/Adreno as well as CUDA. Use --no-defrag to
                                       // reproduce the old sparse-cache behaviour.
    bool        no_extra_bufts = false;  // disable llama.cpp weight-repack buffer types
    bool        no_state_swap = false;   // EndurKV: keep adaptive-anchor gate + evict-once, but SKIP the FA-off->FA-on state-swap (decode stays FA-off over the compacted cache). For mobile GPU (Vulkan) where the swap's KV transfer is prohibitively slow and FA-on decode is ~moot (bandwidth-bound).
    bool        fa_on_evict = false;     // EndurKV Solution 2: FA-ON prefill AND decode. Gate scores come from the in-graph kq_evict side node (softmax(K.Q_lastW)) instead of the FA-off kq_soft_max capture, so NO FA-off prefill and NO state-swap. Works on every backend incl. Vulkan/Adreno; FA-on decode is ~8x faster than FA-off on Adreno.
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
    // Did the caller explicitly pick a KV cache type? Policy presets must not
    // override an explicit choice (2026-08-02 fairness fix; see end of function).
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
        // k_explicit/v_explicit: remember that the CALLER chose these, so the
        // per-policy presets below cannot silently overwrite them (see the
        // 2026-08-02 fairness fix at the end of this function).
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
        a.policy != "keydiff") {   // added 2026-08-16 with the KeyDiff baseline
        std::fprintf(stderr, "bad policy: %s\n", a.policy.c_str()); return false;
    }
    // v1_fa: same eviction algorithm as v1, but use Strategy A (FA-off prefill
    // + state-swap to FA-on for decode) to recover Flash Attention's thermal
    // benefit. Eviction freezes after prefill (no per-step eviction during
    // decode since FA-on doesn't expose kq_soft_max).
    //
    // However: with --ignore-eos or long max_tokens, the frozen-mask design
    // lets the cache grow during decode and re-triggers the thermal throttle
    // (measured in Wave-4 — see TABLE_PHI3_WAVE4_LONGDECODE.md).
    // Auto-enable cheap recency-based decode-time bounding to keep the cache
    // capped without needing attention scores. Sink + most-recent window.
    if (a.policy == "v1_fa") {
        a.snapkv_decode    = true;
        a.no_evict_decode  = true;
        a.decode_bound     = true;   // bound cache during FA-on decode via recency
    }
    // v1_fa2 (v1_FA²): same as v1_fa but with tiered decode-time eviction that
    // PRESERVES the prefill-attention-anchored positions and only drops the
    // OLDEST decode-generated positions. Adaptive K is handled by the launcher
    // script reading DDR temp between iters.
    if (a.policy == "v1_fa2") {
        a.user_policy      = "v1_fa2"; // preserve original label for meta.json
        a.policy           = "v1_fa";  // reuse v1_fa code path
        a.snapkv_decode    = true;
        a.no_evict_decode  = true;
        a.decode_tiered    = true;   // override: tiered eviction instead of recency
        a.decode_bound     = false;
        // SELECTIVE ANCHORING: top-32 attention-scored prompt positions + large
        // recent window. Diagnosis from Wave-7 showed all-prompt anchoring
        // starved recent context → high PPL. v1_fa2 defaults below match the
        // workflow's recommended Wave-8 config.
        if (a.anchor_top_k == 0) a.anchor_top_k = 32;
        if (a.recent_budget == 256) a.recent_budget = 476;  // K=512 - n_sink=4 - anchor=32
    }
    // v1_fa2_hybrid: ISOLATION CELL — H2O canonical 50/50 eviction algorithm
    // (recent K/2 + heavy K/2) wrapped with the v1_FA² thermal stack's external
    // watchdog + memory gate, but NO state-swap and NO Q8 K. Cache stays f16/f16
    // (set by the launcher). Purpose: tease apart "what does just adding the
    // preempt-throttle watchdog do to canonical H2O?" — PPL should match h2o.
    if (a.policy == "v1_fa2_hybrid") {
        a.user_policy      = "v1_fa2_hybrid"; // preserve original label for meta.json
        a.policy           = "h2o";           // reuse canonical h2o eviction path
        a.snapkv_decode    = false;           // no state-swap
        a.no_evict_decode  = false;           // h2o keeps evicting during decode
        a.decode_tiered    = false;
        a.decode_bound     = false;
        // anchor_top_k / recent_budget are not used by h2o; leave defaults.
    }
    // endurkv_optimal: canonical H2O eviction (recent K/2 + heavy K/2) wrapped
    // with the v1_FA² thermal stack — state-swap to FA-on decode (throughput
    // win like v1_fa2_stack), Q8 K + f16 V (memory + bandwidth savings, set by
    // the launcher), multi-sensor watchdog v2 (started by the launcher), and
    // the >=4 GB memory gate (launcher-side). Decode mask is frozen after
    // prefill (no_evict_decode) since FA-on decode doesn't expose
    // kq_soft_max; h2o has no anchor/tiered structure so neither
    // decode_tiered nor decode_bound are enabled.
    //
    // BUG FIX (v6): we keep a.policy = "endurkv_optimal" (NOT rewritten to
    // "h2o") and route a dedicated branch in the prefill-time eviction
    // dispatcher to call policy_h2o(). This avoids inheriting h2o's
    // expectation of per-step decode-time eviction (which is disabled here
    // via no_evict_decode for state-swap → FA-on decode) and forces a
    // PREFILL-TIME eviction that the v1_fa state-swap machinery can consume.
    if (a.policy == "endurkv_optimal") {
        a.user_policy      = "endurkv_optimal"; // preserve original label for meta.json
        // a.policy left as "endurkv_optimal" — handled by dedicated dispatch branch.
        a.snapkv_decode    = true;               // state-swap → FA-on decode
        a.no_evict_decode  = true;               // frozen mask during decode
        a.decode_tiered    = false;              // h2o has no anchor/tiered structure
        a.decode_bound     = false;
        // cache_type_k = q8_0 and cache_type_v = f16 are set by the launcher.
        // anchor_top_k / recent_budget are not used by h2o; leave defaults.
    }
    // endurkv_adaptive: v1 spread-gate eviction wrapped with a runtime
    // K_effective(t) controller driven by DDR/CPU/skin/battery temperatures
    // (see AdaptiveKController above). NO state-swap (Wave-11 smoke showed
    // ~2 PPL regression on Llama-1B from FA-off→FA-on swap), so prefill and
    // decode both run FA-off and per-step eviction stays alive. Cache is
    // f16 K + f16 V (Q8 K seq_add-skip artifact costs ~1 nat PPL with no
    // measured speedup once the watchdog enforces the thermal envelope).
    // Selective anchoring (top-32 mean attention) is enabled to keep the
    // anchored block tight, mirroring v1_fa2's defaults but without the
    // state-swap.
    if (a.policy == "endurkv_adaptive") {
        a.user_policy      = "endurkv_adaptive"; // preserve original label for meta.json
        a.policy           = "v1_adaptive";      // routes to policy_v1_adaptive dispatch
        a.snapkv_decode    = false;              // no state-swap
        a.no_evict_decode  = false;              // allow decode-time re-eviction
        a.decode_tiered    = false;
        a.decode_bound     = false;
        if (a.anchor_top_k == 0) a.anchor_top_k = 32;
    }
    // v1_fa2_f16: ISOLATION CELL — same v1 spread-gate + selective anchoring
    // (top-32) and state-swap → FA-on decode as v1_fa2_stack, but drops Q8 K.
    // Cache is f16 K + f16 V (set by the launcher). Purpose: isolate "what does
    // Q8 K seq_add-skip cost in PPL?" — should fall between v1_fa2_stack and
    // vanilla if Q8 K is the bottleneck.
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
    // v1_entropy_stack: entropy-weighted attention scoring + canonical recent+heavy
    // split. Same thermal stack as v1_fa2 (FA-off prefill, state-swap to FA-on
    // for decode, tiered decode-time eviction). External --cache-type-k q8_0
    // and watchdog are wired by the launcher script.
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
    // v1_predictive_stack: temporal-trajectory (slope) based scoring.
    // Requires kq_soft_max readback every decode step → FA-off for the entire
    // decode (snapkv_decode keeps the prefill-mask anchored but per-step
    // eviction stays alive). Inherits memory gate + Q8 K via the launcher.
    if (a.policy == "v1_predictive_stack" || a.policy == "v1_predictive") {
        const bool is_stack = (a.policy == "v1_predictive_stack");
        a.policy           = "v1_predictive";
        if (is_stack) {
            a.user_policy     = "v1_predictive_stack";
            // Keep per-step eviction alive (need attention every step for slope).
            // snapkv_decode=false here: FA-off prefill AND FA-off decode (no swap).
            a.snapkv_decode   = false;
            a.no_evict_decode = false;
            a.decode_bound    = false;
            a.decode_tiered   = false;
            if (a.recent_budget == 256) a.recent_budget = a.k_nominal / 2;
        }
    }
    // Ada-KV (Feng et al. NeurIPS 2025): adaptive per-head budget allocation.
    // Algorithm 1: concatenate attention weights across all heads, take top-(B·H)
    // globally, count selections per head → derive {B_i*}. Algorithm 2 applies
    // this to SnapKV-style observation-window attention.
    //
    // Our wiring matches the Ada-SnapKV recipe: FA-off prefill (so kq_soft_max
    // is exposed to eval_callback), no state-swap (Ada-KV doesn't perform one),
    // and a frozen mask during decode (Ada-KV's adaptive selection is done once
    // at end-of-prefill). f16 K + f16 V (no Q8 K interactions to confound the
    // PPL comparison vs. the paper's reference numbers).
    if (a.policy == "adakv") {
        a.user_policy      = "adakv";
        a.snapkv_decode    = false;   // Ada-KV does not state-swap
        a.no_evict_decode  = true;    // frozen mask after end-of-prefill selection
        a.cache_type_k     = "f16";
        a.cache_type_v     = "f16";
    }
    if (a.policy == "snapkv") {       // canonical SnapKV baseline (per-head, avgpool, uniform K)
        a.user_policy      = "snapkv";
        a.snapkv_decode    = false;   // SnapKV freezes the prompt keep-set; no state-swap
        a.no_evict_decode  = true;    // frozen mask after end-of-prefill selection
        a.cache_type_k     = "f16";   // SnapKV defaults (no q8_0 — that's ours)
        a.cache_type_v     = "f16";
    }
    // FIXED 2026-08-02: these presets used to clobber --cache-type-k/--cache-type-v
    // even when the caller passed them EXPLICITLY, regardless of flag order. That
    // silently made the phone-GPU memory table unfair: the script requested q8_0 for
    // every arm, vanilla/muKV honoured it, and SnapKV/Ada-KV were forced back to f16 --
    // so SnapKV appeared to use MORE bytes than the full cache (3550 vs 2222 MiB on
    // Phi-3), when 1.88x of that ratio is just f16-vs-q8_0 bytes per cell and not the
    // realizability gap we are actually claiming. Presets are DEFAULTS now; an explicit
    // flag wins. Note V=f16 is still forced downstream for FA-off policies because
    // llama.cpp requires flash-attention for a quantized V -- that one is an engine
    // constraint, not our choice, and is exactly why a per-head evictor cannot reach
    // muKV's memory footprint on this stack.
    if (k_explicit) a.cache_type_k = k_req;
    if (v_explicit) a.cache_type_v = v_req;
    return true;
}


// ---------------------------------------------------------------------------
// GPU PROFILE (2026-08-01). Post-eviction compaction is a state get/set round-trip.
// Whether it pays depends ENTIRELY on the backend, and the two we ship differ:
//
//   CUDA (RTX 4500 Ada, measured):  muKV 51.5 -> 121.5 tps with compaction (2.42x).
//                                   Without it the cache is logically evicted but
//                                   physically sparse, so decode still walks the full
//                                   span and the 12.7x smaller cache buys nothing.
//   Vulkan / Adreno:                the round-trip crosses a driver-managed buffer and
//                                   was measured prohibitive; compaction stays OFF until
//                                   re-measured on-device.
//
// The old gate was `n_gpu_layers == 0`, i.e. CPU-only, which silently disabled the
// 2.42x on every discrete GPU. Default is now chosen per backend; --force-defrag and
// --no-defrag override it explicitly.
// ---------------------------------------------------------------------------
// 2026-08-01: records whether post-eviction COMPACTION actually happened. The swap
// builds a second (FA-on) context and can fail to allocate it -- at ctx 32768 with
// Phi-3 f16 KV it does, and the run silently falls back to FA-off over the UNcompacted
// cache. retained_kv_bytes does NOT reveal this: it measures the logical keep-set and
// reads the same either way, so a fallback run looks identical in meta.json while
// performing like vanilla. Measured: ctx 16384 -> 121.5 tps, ctx 32768 -> 51.5 tps.
static size_t g_reclaimed_bytes = 0;   // EndurKV: bytes madvise()d back to the OS
static long   g_rss_pre_reclaim_kb  = -1;  // RSS immediately BEFORE the madvise
static long   g_rss_post_reclaim_kb = -1;  // ... and immediately AFTER
static bool g_compaction_applied = false;
// WHICH compaction path ran: "none", "roundtrip" (second context, 2x peak) or
// "inplace" (chunked slide, no extra peak). Recorded in meta.json so the two can
// never be confused in a table -- they have identical keep-sets but very different
// memory behaviour, and that difference is the reason the second mode exists.
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
        // CHANGED 2026-08-02: Vulkan/Adreno flipped false -> true.
        // The old `false` came from a source comment asserting the state round-trip
        // "crosses a driver-managed buffer" and was therefore prohibitive on Adreno.
        // It was never measured. When measured on the Adreno 840 it costs 3.9s
        // (Llama-1B) to 6.1s (Phi-3) ONCE, against a decode that is 2.28x faster for
        // the rest of the run -- it pays for itself within a few hundred tokens and is
        // a large net win over a 4096-token generation. Without it the evicted cache
        // stays physically sparse and decode still walks the full prompt span, which is
        // exactly why the GPU was realizing only 10-21% of its memory roofline.
        // Metal is left at the old default: we have no Metal device to measure on, and
        // guessing in the other direction is how this bug happened the first time.
        if (d.find("vulkan") != std::string::npos || d.find("adreno") != std::string::npos) return true;
        if (d.find("metal")  != std::string::npos) return false;   // unmeasured; see above
    }
    return false;   // unknown backend: keep the conservative pre-2026-08 behaviour
}

bool read_file(const std::string & p, std::string & out) {
    std::ifstream f(p, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", p.c_str()); return false; }
    std::stringstream ss; ss << f.rdbuf();
    out = ss.str();
    return true;
}

// ---------------------------------------------------------------------------
// Attention capture (same idea as attention_probe.cpp, simplified)
// Per layer: kq_soft_max tensor of shape [n_kv, n_head, n_seq_tokens]
// We extract the LAST seq token's row → [n_head, n_kv] flat
// ---------------------------------------------------------------------------
struct AttnCapture {
    // per_layer[layer] = flat [n_head * n_kv], layout [h0_p0..h0_pN, h1_p0..]
    // Stores the attention from a SCORING WINDOW (last N query positions),
    // averaged across the window. When n_window=1, this is just the last-query
    // attention (the original behavior). When n_window=32, this is AdaKV-style
    // observation-window scoring — captures the attention the questions would
    // have paid, not just the very last token.
    std::map<int, std::vector<float>> per_layer;

    // ADDED 2026-08-07: exact host cost of muKV's scoring state.
    // muKV's peak RSS at ctx 65536 is ~208 MiB ABOVE vanilla's, which looks wrong for
    // a policy whose whole point is to hold less cache. It is not the cache: the KV
    // buffer is allocated at full n_ctx capacity up front and eviction frees CELLS,
    // not the allocation, so on a discrete GPU the saving is in VRAM traffic and is
    // invisible to RSS. The host cost is THIS: one [n_head x n_kv] score vector per
    // layer, all layers resident at once, which grows with context. Reporting it in
    // meta.json means the memory accounting can net it out instead of guessing --
    // and on the phone, where host and device memory are the same pool, this term is
    // charged against muKV's savings directly.
    // High-water mark of the above. Sampling host_bytes() after eviction reports 0:
    // per_layer is released as soon as the keep-set is computed, but peak RSS is a
    // high-water mark and still charges those pages. Only the running maximum is
    // comparable to the RSS delta.
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

    // ADDED 2026-08-13. Set when eval_callback meets a kq_soft_max it cannot read
    // (blocked/quantized type). Before this existed, an unreadable node was scored from
    // whatever the mis-sized read returned, so the run LOOKED successful and only the
    // output tokens were degenerate -- which is how five phone-GPU baselines were
    // published as valid. Surfacing it in meta.json makes such a cell self-identifying
    // instead of something a human has to notice in the generated text.
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

// Set from args.cb_eval_noread. The callback is a free function with no access to
// Args, and this is a diagnostic-only switch, so a file-static is the honest way.
static bool g_cb_eval_noread = false;

bool eval_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * cap = static_cast<AttnCapture *>(user_data);
    // Return FALSE for nodes we don't need: ggml_backend_sched treats a `true`
    // from the ask-phase as "I need this node", which forces a compute+sync
    // boundary there and DEFEATS node batching (per-node sync = huge decode
    // slowdown). Only the kq_soft_max / kq_evict nodes we actually read should
    // return true. (Fixes ~6x slower FA-on decode when the callback stays
    // attached with capture frozen, e.g. --fa-on-evict.)
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
        // EndurKV Solution 2 side node: t shape [n_kv, W, n_head, ns], already
        // softmax'd. Average over the W observation-window query rows to get the
        // SAME [n_head*n_kv] per-position signal the FA-off kq_soft_max path
        // produces — but captured while the model runs FlashAttention-ON.
        // Two shapes supported: the on-device-reduced [1, n_kv, n_head] (readback
        // optimization — already mean-over-W) and the legacy [n_kv, W, n_head]
        // (mean computed on host). Detect by ne[0]==1.
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

    // ---------------------------------------------------------------------
    // HARDENED 2026-08-13. This block used to assume kq_soft_max is a densely packed
    // f32 tensor and consulted neither t->type nor the strides nb[]:
    //     off = q_idx * n_head * n_kv * sizeof(float)
    //     ggml_backend_tensor_get(t, tmp.data(), off, tmp.size()*sizeof(float))
    // It now derives the element width from ggml_type_size(), the query-slice stride
    // from nb[2], the row stride from nb[1], and converts f16->f32 when a backend hands
    // back half precision. A layout it cannot read (blocked/quantized type) sets
    // capture_failed rather than scoring from whatever the mis-sized read returned.
    //
    // THIS IS NOT THE FIX FOR THE ADRENO/VULKAN CORRUPTION, and the reader should not
    // assume it is. That was the hypothesis this change was written to test -- five
    // FA-off baselines emit 1-13% degenerate tokens on the phone GPU while the same
    // policies are clean on CUDA and phone CPU, and a packed-f32 assumption that only
    // held on some backends would have explained it exactly. The diagnostic below was
    // added to confirm it and REFUTED it instead. Measured on the device, 2026-08-13:
    //     Adreno 840 / Vulkan : type=f32 ne=[256,64,32] nb=[4,1024,65536] packed_row=yes
    //     RTX 4500 Ada / CUDA : type=f32 ne=[256,64,32] nb=[4,1024,65536] packed_row=yes
    // Byte-for-byte the same layout, so the old read was correct on both.
    //
    // ROOT CAUSE, MEASURED 2026-08-13 (--force-cb-eval / --cb-eval-noread exist for this).
    // Three vanilla FA-off runs on the Adreno, same seed, same prompt, 200 tokens, the
    // ONLY difference being what the callback does:
    //     A  no callback at all                    0 / 978  degenerate '!'   0.0%
    //     B  callback: split + readback           76 / 609                  12.5%
    //     C  callback: split, NO readback         68 / 679                  10.0%
    // C is the answer. Reading the tensor is innocent -- merely returning true from the
    // ask phase is enough. That return makes ggml_backend_sched END A SPLIT at
    // kq_soft_max and materialize it, and on this Vulkan backend a split there produces
    // wrong results for the rest of the attention. Vanilla looked clean for months only
    // because it is the one policy that never installs the callback.
    //
    // CONSEQUENCE, and it is a design argument rather than a mere bug report: the
    // mid-graph attention intercept -- the standard way every score-reading evictor is
    // implemented -- is not numerically safe on this mobile GPU, independent of the
    // policy layered on top. muKV is unaffected because it never intercepts: kq_evict is
    // a real side NODE the graph produces and the scheduler materializes normally. The
    // mechanism muKV needed in order to keep flash-attention ON turns out to be the only
    // way to read attention safely here at all.
    //
    // FIX FOR THE FA-OFF BASELINES: give them the same treatment -- emit a side node that
    // copies the soft_max output, instead of splitting the live graph to intercept it.
    // Until that lands, phone-GPU numbers for FA-off policies are TIMING-ONLY; their
    // output tokens are invalid and their perplexity must not be reported.
    // The hardening below is kept because it is correct and self-documenting, not
    // because it resolved the defect.
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

    // Observation-window scoring: average attention across the last
    // cap->obs_window query positions (default 1 = original behavior;
    // 32 = AdaKV-style observation window). For window=W, average attn[h, p]
    // over the last W query slots. This captures the attention the imminent
    // decode query is most likely to need, not just the very-last-token's.
    const int win = std::max(1, std::min(cap->obs_window, (int)ne2));
    std::vector<float> buf((size_t)n_head * (size_t)n_kv, 0.0f);
    std::vector<float> tmp((size_t)n_head * (size_t)n_kv);
    std::vector<uint8_t> raw((size_t)n_kv * ts);   // one row, in the tensor's own type
    for (int w = 0; w < win; ++w) {
        // Query row index = (ne2 - 1 - w) — the last W query positions
        const int64_t q_idx = ne2 - 1 - w;
        if (q_idx < 0) break;
        // Row-at-a-time using the tensor's OWN strides, so a padded nb[1] (which the
        // packed-f32 assumption above silently mis-indexed) is handled correctly.
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

// ---------------------------------------------------------------------------
// Policies — each returns per-(layer, kv_head, position) boolean keep mask
// flattened as keep[layer * n_kv_heads * n_kv + kv_head * n_kv + pos]
// ---------------------------------------------------------------------------
struct PolicyResult {
    // For each layer, set of positions to KEEP (per kv-head, OR'd over query heads)
    // For sequence-level eviction we then OR across layers → keep set.
    std::vector<std::unordered_set<int>> per_layer_keep;
    int n_layers = 0;
};

// Persistent state for H2O (Zhang NeurIPS '23): tracks cumulative attention
// per (layer, head, position) across all forward passes. The "heavy hitters"
// are positions whose cumulative attention rank in the top-K_nominal.
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

// Sink-token protection: always keep the first n_sink positions in every layer.
// StreamingLLM-style — these positions carry global structural info (BOS, instructions)
// and dropping them causes degenerate-repetition failure modes.
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


// ---------------------------------------------------------------------------
// Ada-KV (Feng et al. NeurIPS 2025) — Algorithm 1 + Ada-SnapKV (Algorithm 2)
//
// Insight: instead of giving every head the same K_nominal budget, distribute
// a TOTAL budget B = K_nominal · H across heads in proportion to how much of
// each head's attention mass falls in the top of a GLOBAL ranking.
//
//   Algorithm 1 (paper):
//     1. Concatenate all heads' attention vectors: A = Cat({A_i}).
//     2. Take the top-B weights of A globally.
//     3. f_i = count of selected weights belonging to head i.
//     4. B_i = f_i.
//
//   Algorithm 2 (Ada-SnapKV — what we run here):
//     - Use the last query's post-softmax attention as A_bar_i (we do this for
//       every layer that exposes kq_soft_max via eval_callback).
//     - Apply max-pool kernel of size 7 (SnapKV convention) over each head's
//       attention before ranking — this preserves local "spike" structure.
//     - Reserve observation window: B ← B − winsize · H. We use winsize = 0
//       to keep the comparison with v1/h2o/tova apples-to-apples.
//     - Safeguard: B_i* = α · B_i + (1−α) · (B/H) with α = 0.2. Prevents any
//       head from getting starved when the global ranking is dominated by a
//       few heads.
//     - Per-head TopK using B_i*.
//
// Implementation notes vs. the paper:
//   * We apply Algorithm 1 PER LAYER. The paper runs it once across all
//     (layer, head) pairs; here we keep per-layer locality to match how
//     llama.cpp's per-layer kq_soft_max is delivered to us. Empirically the
//     difference is small because attention concentrates within layers.
//   * The PolicyResult interface returns per-layer keep sets that are unioned
//     across query heads by apply_eviction(); we cap the union at K_nominal
//     positions per layer to keep the sequence-level keep-set bounded.
//   * cap.n_head is the QUERY-head count; n_kv_heads is unused (kept in the
//     signature for symmetry with the other policies' signatures).
// ---------------------------------------------------------------------------
// Canonical SnapKV (Li et al. NeurIPS 2024). CORRECTED 2026-08-04: this comment
// previously claimed the FasterDecoding defaults were window_size=64 and
// max_capacity_prompt=320 "verified from snapkv_utils.py". The actual file
// (init_snapkv(), archived at benchmarks/snapkv_utils_reference.py) sets
// window_size=32, max_capacity_prompt=2048, kernel_size=5, pooling=avgpool.
// We keep K=1024 deliberately -- budget-matching muKV is the right control for a
// memory comparison -- but the window/kernel must come from the real source.
// Faithful reproduction — NONE of µKV's optimizations (no gate, no adaptive budget, no state-swap):
//   1. PER-HEAD. For each head, the observation window = the LAST window_size keys is ALWAYS kept
//      (k_cur = key_states[:, :, -window_size:]).
//   2. From the PREFIX ONLY (positions [0, n_kv-window_size)), select the top (max_capacity_prompt
//      - window_size) keys by avg-pooled (kernel 5) observation-window attention.
//   3. keep = window ∪ per-head top-K-prefix, unioned across heads by apply_eviction().
// The per-head union is exactly why canonical SnapKV cannot physically reclaim jointly-stored KV on
// a sequence-level engine (llama.cpp seq_rm): different heads keep different prefix positions, so the
// union covers nearly the whole prefix → lands in the full-cache class (no real on-device compaction).
// K_nominal maps to max_capacity_prompt; window_size/kernel are SnapKV's canonical defaults.
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
            for (int p = 0; p < prefix_len; ++p) {               // 1-D avgpool over the PREFIX only
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
        // --- Max-pool each head's attention vector with kernel size 7 ---
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

        // --- Algorithm 1 step 1+2: concatenate and select top-(K_nominal·H) ---
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

        // --- Algorithm 1 step 3+4: count selections per head → f_i = B_i ---
        std::fill(B_per_head.begin(), B_per_head.end(), 0);
        for (int k = 0; k < total_budget; ++k) {
            int head_idx = all_weights[k].second / n_kv;
            if (head_idx >= 0 && head_idx < n_query_heads) B_per_head[head_idx]++;
        }

        // --- Safeguard: B_i* = α · B_i + (1−α) · K_nominal ---
        for (int h = 0; h < n_query_heads; ++h) {
            int B_i = (int)std::lround(alpha_safeguard * (float)B_per_head[h]
                                       + (1.0f - alpha_safeguard) * (float)K_nominal);
            if (B_i < 1)    B_i = 1;
            if (B_i > n_kv) B_i = n_kv;
            B_per_head[h] = B_i;
        }

        // --- Per-head TopK selection using B_i* on the RAW (un-pooled)
        //     attention (selection should follow the same scoring as v1/tova
        //     so the comparison isolates the BUDGET allocation, not the
        //     scoring function).
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

        // --- Cap the union at K_nominal positions (clip) so the
        //     sequence-level keep-set stays bounded. Prefer positions with
        //     the highest mean attention across heads when clipping.
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

// Gate-mode global: read by the per-head budget loop. Default "v1" preserves
// the historical single-statistic gate. "v2" uses s_h = m_h + 0.5*m'_h.
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
            // Gate: single-statistic m_h (v1, original) or two-statistic m_h + 0.5*m'_h (v2).
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

// ---------------------------------------------------------------------------
// EndurKV-Adaptive — DESIGN
// ---------------------------------------------------------------------------
// The endurkv_adaptive policy combines five building blocks observed to
// individually contribute PPL/throughput wins on Llama-1B during the
// Wave-11 smoke pass on the OnePlus 15:
//
//   1. v1 per-head spread-gate budget formula (already in policy_v1):
//        K_h = round(K_eff · μ_head(max_a[h]))
//        μ_head(x) = 1.3 − 0.6 · clip((x−0.4)/0.4, 0, 1)
//      → heads whose attention is sharply concentrated keep fewer cells.
//
//   2. NEW — runtime K_effective(t) modulation via a thermal feedback
//      controller. Empirical thresholds from THROTTLE_TRIGGER_EMPIRICAL.md:
//        DDR    : warn 51.7 °C / crit 56.7 °C
//        CPU-big: warn 56.2 °C / crit 61.2 °C
//        Skin   : warn 37.7 °C / crit 42.7 °C
//        Battery: warn 36.8 °C / crit 39.8 °C
//      The worst-of-N threat (linear ramp warn→crit, clipped to [0,1])
//      drives μ_thermal in the range [0.7, 1.3]:
//        threat = 0 (cool)  → μ_thermal = 1.3 (grow K → better PPL)
//        threat = 1 (crit)  → μ_thermal = 0.7 (shrink K → cool DDR)
//      K_eff = max(32, round(K_nominal · μ_thermal)).
//
//   3. Multi-sensor watchdog v2 (external; started by the launcher).
//
//   4. Memory gate (launcher-side; ≥4 GB free RAM required).
//
//   5. f16 K cache (NOT Q8). Wave-11 smoke showed Q8 K hurts PPL ~1 nat
//      via the seq_add-skip artifact and offers no end-to-end speedup once
//      the DDR thermal envelope is enforced by (2) + (3).
//
// We DELIBERATELY skip:
//   - state-swap (FA-off→FA-on) — Wave-11 smoke saw ~2 PPL regression on
//     Llama-1B; we run FA-off prefill + FA-off decode and let per-step
//     eviction stay alive.
//
// Implementation: AdaptiveKController polls the thermal sysfs zones each
// time policy_v1_adaptive() is invoked (i.e. once per prefill chunk and
// once per decode step). Poll is cheap (~4 small file reads) and is rate-
// limited by `poll_interval_s` to avoid hammering the kernel.
//
// Read locations on OnePlus 15:
//   /sys/class/thermal/thermal_zone47/temp  → DDR
//   /sys/class/thermal/thermal_zone24/temp  → CPU big cluster
//   /sys/class/thermal/thermal_zone60/temp  → skin (shell_front)
//   /sys/class/thermal/thermal_zone94/temp  → battery
// BCL current (battery current limit signature) requires sysfs paths that
// are not always reachable from this binary's UID; we skip it here and
// rely on temperatures alone — the same signal mix the multi-sensor
// watchdog v2 uses.
// ---------------------------------------------------------------------------
struct ThermalState {
    float ddr_c     = 0.0f;   // current DDR temp °C
    float skin_c    = 0.0f;   // shell_front temp °C
    float battery_c = 0.0f;   // battery temp °C
    float cpu_big_c = 0.0f;   // cpu-1-0-0 temp °C
    int   bcl_current_ma = 0; // battery current (BCL signature) — 0 when unreachable
    int64_t poll_us = 0;      // last poll wall_us
};

class AdaptiveKController {
public:
    ThermalState last;
    float poll_interval_s = 2.0f;

    // Empirical thresholds from THROTTLE_TRIGGER_EMPIRICAL.md:
    //   DDR     warn 51.7 / crit 56.7
    //   CPU big warn 56.2 / crit 61.2
    //   Skin    warn 37.7 / crit 42.7
    //   Battery warn 36.8 / crit 39.8
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
        // Linear interpolation: threat=0 → mu=1.3, threat=1 → mu=0.7.
        return 1.3f - 0.6f * threat;
    }

    int K_effective(int K_nominal) const {
        int K = (int)std::lround((double)K_nominal * (double)mu_thermal());
        if (K < 32) K = 32; // floor
        return K;
    }

    // Read one /sys/class/thermal/thermal_zoneN/temp value. The sysfs nodes
    // report millidegrees Celsius; we divide by 1000. Returns NaN on failure
    // so we can preserve the previous reading in `last`.
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
        // BCL current is not reliably reachable from this binary; leave 0.
        last.bcl_current_ma = 0;
        last.poll_us = now;
    }
};

// Process-wide adaptive-K controller. Shared across the prefill and decode
// dispatch sites so the poll-interval rate-limiter accumulates state.
static AdaptiveKController g_adaptive_K;

// policy_v1_adaptive: identical per-head budget loop to policy_v1, but
// scales K_nominal by μ_thermal before invoking the budget formula. This
// keeps the eviction structure unchanged (so existing analysis tools that
// expect a v1-shaped per_layer_keep stay valid) while allowing the cache
// to breathe at low temperatures and contract under thermal pressure.
static PolicyResult policy_v1_adaptive(const AttnCapture & cap, int K_nominal, int n_kv_heads) {
    g_adaptive_K.update();
    const int K_eff = g_adaptive_K.K_effective(K_nominal);
    return policy_v1(cap, K_eff, n_kv_heads);
}

// TOVA-layer (Oren et al. ACL 2024, arXiv 2401.06104) — the paper's preferred
// per-layer variant. Per their Algorithm 1 (App B) and ablation in App A
// Table 3: averaging attention across the heads of a layer is empirically
// superior to per-head selection (PPL 7.71 vs 8.69 at multi-state=256).
//
// Algorithm (paper-faithful):
//   1. For each layer, compute mean attention across all query heads from
//      the LAST query position (cb_eval captures the last seq token's row):
//        mean_a[p] = (1/H) · Σ_h softmax(Q_h^{last} · K_p)
//   2. Keep top positions by mean_a (one keep-set per layer).
//
// Bug history (catastrophic PPL pre-fix):
//   Pure top-K by mean attention was catastrophic in our chunked-eval setup
//   (Phi-3 PPL=190.7 on chunk 0). Root cause: in canonical autoregressive
//   TOVA the last query is always THE most recent token, so positions
//   immediately preceding it naturally score high (local attention). In our
//   chunked-prefill setup the "last query" is the END of a prefill chunk —
//   the EVAL chunk that follows then queries positions that may have been
//   evicted because they had low attention from the prefill's last query
//   (semantic vs. local mismatch). Symptom matches the H2O pre-fix bug
//   (PPL 158→17 after adding the recent window).
//
// Fix (same canonical pattern as H2O in this file, paper-aligned):
//   - Recent component: always keep the last (K_nominal/2) positions.
//   - Heavy component:  fill the remaining budget from top-mean-attention
//                       among positions OUTSIDE the recent window.
//   - Sink (n_sink=4):  applied externally by protect_sink() at call site,
//                       just like the other policies.
//
// Note: the original paper drops exactly one token per step when the cache
// exceeds capacity; we keep the same top-K_nominal interface used by v1/H2O
// so policies are comparable at matched budgets. The scoring function and
// per-layer aggregation match the paper.
// Canonical TOVA (no recent-window safety net) — pure top-K by mean head attention
// from the last query, as in Oren ACL 2024 §3 Algorithm 1. Use this in
// generation-mode evals (NIAH, qasper, hotpotqa) where the last query IS the
// real autoregressive last token. NOT recommended for chunked PPL eval.
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

static PolicyResult policy_tova(const AttnCapture & cap, int K_nominal, int /*n_kv_heads*/) {
    PolicyResult r;
    if (cap.per_layer.empty()) return r;
    const int n_query_heads = cap.n_head;
    const int n_kv = cap.n_kv;
    for (const auto & [layer, attn] : cap.per_layer) {
        r.n_layers = std::max(r.n_layers, layer + 1);
    }
    r.per_layer_keep.resize(r.n_layers);

    // Canonical 50/50 recent+heavy split of the K_nominal budget
    // (matches the H2O fix in this same file; addresses the same failure mode).
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

        // (a) ALWAYS keep the last n_recent positions for this layer.
        for (int p = recent_start; p < n_kv; ++p) kept.insert(p);

        // (b) Top-n_heavy by mean_a among positions OUTSIDE the recent window.
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

// H2O (Heavy-Hitter Oracle, Zhang NeurIPS 2023, Algorithm 1, §4.2):
// Canonical H2O maintains a BALANCE of two distinct sets per (layer, head):
//   (a) heavy_budget positions chosen by accumulated attention (the "H2"), AND
//   (b) recent_budget most-recent positions (the local window).
// Reference impl (FMInference/H2O) uses heavy_ratio=recent_ratio=0.1 of context,
// i.e. a 50/50 split of the total budget. We default to that 50/50 split:
//   n_recent = K_nominal / 2
//   n_heavy  = K_nominal - n_recent
// Sink protection (n_sink positions at the front) is applied externally via
// protect_sink() — that handles the BOS / instruction sink separately.
//
// Bug history: prior implementation was pure "top-K by cumulative attention"
// with NO recent component. Newly arrived decode tokens had near-zero
// accumulated sums and were immediately evicted, which is catastrophic for
// next-token prediction (local n-gram statistics live in the recent window).
// That bug explained PPL=158 in smoke; the recent+heavy split (this code)
// is the canonical paper algorithm and should drop PPL into the TOVA range.
//
// Accumulators are maintained in H2OState — `state` must have been
// .update(cap)-ed once per forward pass before invoking this policy.
// Aggregate-OR across query heads within a layer (same convention as TOVA/v1).
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

        // (b) ALWAYS keep the last n_recent positions for this layer.
        for (int p = recent_start; p < n_kv; ++p) kept.insert(p);

        auto it = state.sum_attn.find(layer);
        if (it == state.sum_attn.end()) {
            // Cold start — no accumulator yet; choose heavy from current-step
            // attention as a best-effort fallback. Still apply the recent window
            // (already inserted above), so this is graceful degradation.
            for (int h = 0; h < n_query_heads; ++h) {
                const float * a = attn.data() + (size_t)h * n_kv;
                // Candidates for "heavy" are positions OUTSIDE the recent window.
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

// StreamingLLM (Xiao et al., ICLR 2024, arXiv:2309.17453).
// Pure positional eviction: keep the FIRST n_sink "attention sink" tokens plus
// the LAST (K_nominal - n_sink) tokens (sliding window). Drops the middle.
// Ignores attention scores entirely — the canonical FA-on-compatible KV
// eviction policy. K_nominal here is the total budget (sink + recent).
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

// ---------------------------------------------------------------------------
// v1_entropy — confidence-weighted attention scoring.
//
// For each layer L and head h, compute Shannon entropy H_{L,h} from the
// post-softmax attention vector at the LAST query position. A confidence
// weight w_h = exp(-H_h) is near 1 for sharply-peaked heads (the head has
// "decided") and near 0 for diffuse heads. The per-position keep score is
// the confidence-weighted attention mass:
//     score(p) = Σ_h w_h * a_h(p)
// Then we apply the canonical recent+heavy split (same as H2O/TOVA in this
// file): K_nominal/2 positions at the tail unconditionally kept; the rest
// of the budget filled by the top-score positions outside the recent window.
// Sink protection (--n-sink) is applied externally by protect_sink().
//
// Differs from TOVA (raw mean attention) by weighting heads by their
// confidence rather than treating all heads equally.
//
// Numerical safety:
//   - Entropy uses ap > eps guard (eps=1e-12) to avoid -inf log on
//     denormalized softmax outputs.
//   - H is clamped to log(n_kv)+1 before exp() to prevent w_h underflow
//     to exactly 0 on huge contexts.
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
        // 1. Per-head entropy → confidence weight w_h = exp(-H_h).
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
        // 3. Recency window always kept (defends against TOVA-style chunked-prefill
        //    semantic-vs-local mismatch).
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

// ---------------------------------------------------------------------------
// AttnHistory: ring buffer of mean-across-heads attention vectors per layer.
// Used by v1_predictive to compute the temporal trajectory (slope) of each
// position's attention over the last W decode steps.
// ---------------------------------------------------------------------------
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
            // FIX A: only RESET the ring when n_kv DECREASES (real eviction —
            // positions get compacted, stale slopes meaningless). When n_kv
            // INCREASES (decode appending one new token), GROW per-slot arrays
            // by padding with 0.0f and PRESERVE count/head so we accumulate
            // history across decode steps. Cold start (prev_n_kv == 0) also
            // initializes.
            if (prev_n_kv == 0 || n_kv < prev_n_kv) {
                buf.assign((size_t)W * (size_t)n_kv, 0.0f);
                count[layer] = 0;
                head[layer]  = 0;
                n_kv_layer[layer] = n_kv;
            } else if (n_kv > prev_n_kv) {
                // Grow each of the W slot rows from prev_n_kv to n_kv by padding
                // with 0.0f, preserving the existing samples at positions
                // [0, prev_n_kv). Layout is W rows of n_kv floats each; the
                // existing buffer is W rows of prev_n_kv. Rebuild in-place.
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
            // If n_kv == prev_n_kv, nothing to resize; just write.

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

    // Per-position slope (numerator-only; denominator is constant per window).
    // Walks ring oldest→newest. Returns -INFINITY when N<3 (insufficient samples).
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
        // Oldest sample slot = (hd - N + W) % W when N<=W; when N==W, hd is oldest.
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

// ---------------------------------------------------------------------------
// v1_predictive — temporal-trajectory (slope-based) scoring.
//
// For each position p in each layer, fit a linear regression slope over the
// last W attention samples (mean-across-heads). Positions with positive
// slope are "rising" (the model is returning to them) and protected; positions
// with negative slope are eviction candidates. The keep set is:
//   (1) [0, n_sink) sink positions (added externally by protect_sink).
//   (2) [n_kv - recent, n_kv) recency floor (newcomers with <3 samples can't
//       be ranked by slope, so they must be protected by recency).
//   (3) Remaining (K_nominal - n_sink - n_recent) slots filled by the top-slope
//       positions OUTSIDE the recent window. Tie-break by last-step attention.
// On cold start (history empty / too few samples), slope=-INF for every
// candidate → policy degrades to recency-only (StreamingLLM-equivalent),
// which is safe.
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
        // Recency window — always kept.
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
                // Rank by slope desc; tie-break by last_a desc.
                // nth_element is faster than full sort; partial_sort gives a
                // tie-broken stable rank but cost is similar at small K.
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

// ---------------------------------------------------------------------------
// Whether the K cache uses a non-f16 (quantized) type. When quantized, the
// llama.cpp K-shift graph (RoPE re-rotation after seq_add) is not supported
// and crashes. In that case we MUST NOT call seq_add — leave positions sparse,
// the FA-on attention still computes correctly because each cell carries its
// own RoPE-encoded K.
static bool g_k_is_quantized = false;

// μKV×VL (2026-07-25): true when the run uses a multimodal (M-RoPE) cache.
// Two consequences, both hard requirements rather than optimizations:
//   1. Eviction must address CELLS, not positions — every token of an image
//      shares one dim-0 position, so a position range cannot express a
//      per-token keep mask (see apply_eviction / selective anchoring).
//   2. llama_memory_seq_add() is ILLEGAL on M-RoPE caches — llama.cpp asserts
//      n_pos_per_embd()==1 and aborts. Position compaction must be skipped;
//      FA-on attention is correct over a sparse cache because each cell
//      carries its own RoPE-encoded K (same rationale as g_k_is_quantized).
static bool g_evict_by_cell_index = false;

// Tiered decode-time eviction for v1_FA² (v1_fa2). The cache layout after
// state-swap is:
//   [0, n_anchored)         — attention-anchored positions chosen by v1's
//                              spread-gate eviction at end-of-prefill (smart
//                              selection; these are the IMPORTANT prefill
//                              tokens we want to keep across decode).
//   [n_anchored, n_kv - R)  — older decode-generated positions (these are
//                              candidates for eviction).
//   [n_kv - R, n_kv)        — recent decode tokens (within recent budget R).
//
// We drop the middle band when n_kv exceeds n_anchored + R + hysteresis.
// Unlike apply_recency_decode_eviction, this NEVER drops anchored positions,
// preserving the attention-aware selection across the entire decode phase.
// Returns positions evicted.
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
        // Compact positions only when K is f16 — seq_add triggers a K-shift
        // graph that doesn't support quantized K.
        llama_memory_seq_add(mem, 0, drop_end, n_kv, -shift);
    }
    return shift;
}

// Cheap recency-based decode-time eviction. Used by v1_fa during FA-on decode,
// where attention scores aren't available (FA fuses softmax). Keeps positions
// [0, n_sink) and [n_kv - n_recent, n_kv); drops the middle.
//
// Trigger: only when n_kv > 1.5 * (n_sink + n_recent) to amortize the cost
// of seq_rm + position compaction. Returns positions evicted.
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
    // Compact: shift the surviving recent positions down so positions stay
    // dense (required by llama.cpp KV layout). Skip when K is quantized
    // since K-shift RoPE rotation doesn't support quantized cache.
    int shift = drop_end - drop_start;
    if (!g_k_is_quantized) {
        llama_memory_seq_add(mem, 0, drop_end, n_kv, -shift);
    }
    return shift;
}

// Apply policy → call seq_rm for positions evicted from ALL layers
// (sequence-level: conservative — keeps a position if any layer wants it).
// Returns how many positions were evicted this step.
// ---------------------------------------------------------------------------
static int apply_eviction(llama_context * ctx, const PolicyResult & pol, int n_kv) {
    if (pol.per_layer_keep.empty()) return 0;
    int n_keep = (int)pol.per_layer_keep.size();
    // Position is evicted ⟺ NO layer keeps it.
    std::vector<uint8_t> keep(n_kv, 0);
    for (int l = 0; l < n_keep; ++l) {
        for (int p : pol.per_layer_keep[l]) if (p >= 0 && p < n_kv) keep[p] = 1;
    }
    llama_memory_t mem = llama_get_memory(ctx);
    int evicted = 0;
    // 2026-07-25 (μKV×VL): on M-RoPE / vision caches every token of an image
    // shares ONE dim-0 position (the value the cache stores), so position-based
    // seq_rm cannot express a per-token keep mask — it is all-or-nothing per
    // image, and feeding it cell indices corrupts the cache. Use the fork's
    // cell-index API there. Text caches (cell index == position) keep the
    // original contiguous-run seq_rm path, so text behavior is bit-identical.
    if (g_evict_by_cell_index) {
        std::vector<int8_t> keep_i8(keep.begin(), keep.end());
        evicted = (int) llama_endurkv_seq_rm_cells(mem, 0, keep_i8.data(), (uint32_t) n_kv);
    } else {
        // Find contiguous runs of evictions; call seq_rm per run for efficiency.
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
    // 2026-07-25 VL-compaction diagnostic: is keep[] marking ~everything (union
    // problem) or is seq_rm not reducing the cache (coordinate/API problem)?
    {
        int kept = 0; for (int q = 0; q < n_kv; ++q) kept += keep[q];
        int pmin = (int)llama_memory_seq_pos_min(mem, 0);
        int pmax = (int)llama_memory_seq_pos_max(mem, 0);
        std::fprintf(stderr, "[evict-dbg] n_kv=%d kept=%d evicted=%d  post-rm pos=[%d..%d]\n",
                     n_kv, kept, evicted, pmin, pmax);
    }
    return evicted;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
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
// Applied IDENTICALLY across all eviction policies for fair comparison.
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

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char ** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) return 1;
    // Propagate the gate mode to the policy_v1 global so the per-head budget
    // loop picks up the correct ramp without threading through every call site.
    g_gate_mode = args.gate_mode;

    std::string prompt;
    if (!read_file(args.prompt_file, prompt)) return 1;

    ggml_backend_load_all();
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = args.n_gpu_layers;
    // ADDED 2026-08-05: --no-extra-bufts disables llama.cpp's weight-repack buffer
    // types (llama_model_params::use_extra_bufts, which defaults to TRUE).
    // WHY IT MATTERS: Bonsai-8B (Q1_0) cannot run on the Adreno GPU because its
    // weights are repacked into the ARM CPU format q1_0_4x8 at load time -- for
    // EVERY tensor, regardless of whether it will live on the GPU -- and the
    // repacked data then kills the Vulkan queue
    // ("vk::DeviceLostError: vk::Queue::submit"). It fails identically at
    // --n-gpu-layers 99, 24 and 12, and the GGML_CPU_REPACK / GGML_NO_REPACK env
    // vars do not suppress it (252 repack lines either way) because the repack is
    // compiled in. Turning the extra buffer types off keeps the model at Q1_0 --
    // so the 1-bit composition claim is unaffected -- and simply skips the repack.
    // Leave it ON for CPU runs: that repack is exactly the i8mm/dotprod path that
    // makes Q1_0 fast there.
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

    // --- μKV×VL (2026-07-25): multimodal tokenization. Runs BEFORE context
    // creation so ctx_size accounts for image tokens. The prompt text is wrapped
    // in the qwen ChatML template with one media marker per --image; image
    // embeddings enter the same KV cache, so eviction/selection downstream is
    // untouched. Supported only for eval_mode=gen with policy vanilla or
    // v1_fa2+fa-on-evict (the frozen GPU μKV) — anything else exits loudly.
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
        // M-RoPE cache → eviction must address cells, not positions.
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

    // Cache type policy:
    //   - Prefill context (may be FA-off): K can be quantized, V MUST be f16
    //     (llama.cpp requires FA-on for V quantization).
    //   - Post state-swap FA-on context (snapkv_decode): V CAN be quantized
    //     since FA is enabled there. Save the user's requested V type for
    //     later; downgrade for prefill if needed.
    const ggml_type user_type_v = parse_cache_type(args.cache_type_v);
    const ggml_type user_type_k = parse_cache_type(args.cache_type_k);
    llama_context_params cparams = llama_context_default_params();
    cparams.type_k = user_type_k;
    cparams.type_v = user_type_v;
    bool prefill_needs_f16_v = (!args.fa_vanilla || args.policy != "vanilla")
                               && cparams.type_v != GGML_TYPE_F16;
    // Only force F16 V at prefill when the prefill ctx is going to be FA-off.
    // v1_fa* WITHOUT fa-on-evict uses FA-off prefill (kq_soft_max capture);
    // vanilla and fa-on-evict run FA-on end-to-end (see use_fa below), where
    // llama.cpp supports quantized V. 2026-07-24: gate added because the
    // unconditional force made a q8_0-KV request allocate K=q8+V=f16 =
    // 4704 MiB for Phi-3@16K, exceeding Tegra's ~4 GiB single-alloc limit
    // (f16/f16 requests are unaffected — identical behavior to before).
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
    // n_batch is the *per-decode-call* logical batch size; we chunk the prefill
    // manually so n_batch can stay at a GPU-safe ~512 even for 4K+ token prompts.
    // n_ubatch controls per-kernel chunking inside one llama_decode call.
    cparams.n_batch  = (uint32_t)args.n_batch;
    cparams.n_ubatch = (uint32_t)args.n_ubatch;
    cparams.no_perf  = false;
    std::fprintf(stderr, "[eviction_bench] n_batch=%u n_ubatch=%u n_gpu_layers=%d\n",
                 cparams.n_batch, cparams.n_ubatch, args.n_gpu_layers);
    // Latency optimization: enable FA-auto for vanilla (no attention capture needed).
    // For eviction policies we need kq_soft_max as a discrete tensor → must keep FA off.
    // FAIRNESS FIX 2026-08-07: StreamingLLM evicts by POSITION (n_sink sinks + a recent
    // window) and never reads an attention value -- policy_streamingllm() touches the
    // capture only to learn n_kv and the layer count. Forcing it FA-off, as this line
    // used to, made it forfeit flash-attention for data it does not use, and its 64K
    // decode was reported at 0.23x vanilla partly as a result. That is a harness
    // artifact, not a property of StreamingLLM, and it is the same class of mistake as
    // running the baselines at our K instead of theirs. Positional policies now get FA.
    // EXTENDED 2026-08-16: keydiff joins the no-capture set. KeyDiff
    // (arXiv:2504.15364) scores by key geometry alone -- -cos(mu(K), k_i), keep the
    // most distinctive -- and never reads an attention value, so like StreamingLLM
    // it gets flash-attention and no cb_eval. It is the only score-BASED baseline
    // with that property, which is exactly why it is the mobile-positioned
    // competitor the comparison needs: it cannot be handicapped by the FA-off path
    // even in principle.
    const bool positional_policy = (args.policy == "streamingllm" || args.policy == "keydiff");

    // ---- FA-off capture viability gate (2026-08-24) -------------------------
    // The FA-off kq_soft_max capture is NUMERICALLY BROKEN on Adreno/Vulkan when
    // head_dim is neither 64 nor 128. Measured on Phi-3-mini (head_dim=96,
    // --n-gpu-layers 99, 4K prompt, 64 greedy tokens), <unk> per 100 chars of
    // generated text: SnapKV 8.55, Ada-KV 9.82, H2O 9.05, TOVA 6.01 -- while
    // vanilla / StreamingLLM / KeyDiff, which all run FA-ON, emit 0.00. The same
    // four policies are clean on Llama-1B (head_dim=64) and clean on CPU, so the
    // fault is the capture path at this head geometry on this driver, NOT the
    // policies. Left ungated it silently produces publishable-looking tok/s
    // numbers for a computation that is emitting garbage.
    //
    // SnapKV and Ada-KV are rescuable: both score ONCE at the end of prefill, so
    // the in-graph side node supplies the same scores without any FA-off pass.
    // That is plumbing, not a policy change -- and it is also faster (measured
    // 3.02->14.46 and 5.61->11.74 tok/s). H2O and TOVA re-score at every decode
    // step, so no prefill-time side node can serve them; they are refused rather
    // than reported.
    {
        const int  hd_probe      = llama_model_n_head(model) > 0
                                 ? (int)(llama_model_n_embd(model) / llama_model_n_head(model)) : 0;
        const bool fa_off_unsafe = args.n_gpu_layers > 0 && hd_probe != 64 && hd_probe != 128;
        const bool needs_capture = !(args.policy == "vanilla" || positional_policy || args.fa_on_evict);
        if (fa_off_unsafe && needs_capture) {
            if (args.policy == "snapkv" || args.policy == "adakv") {
                args.fa_on_evict     = true;
                args.no_evict_decode = true;
                // MUST also force IN-PLACE compaction. fa_on_evict alone routes to
                // the state-API round-trip, which needs a second full cache: measured
                // 4073.4 MiB for Phi-3@16K, and the process is OS-killed before it
                // emits a token. The gate only ever fires on a big-head_dim model on
                // GPU -- precisely where the round-trip does not fit -- so promoting
                // without this just trades a corrupt cell for a dead one.
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

    // MOVED BEFORE PREFILL (2026-09-03): the GPU clock action must be in force during
    // prefill, where the clock lever pays (the first proof cell ran prefill at 1191 MHz
    // because this block used to sit after the prefill loop). Needs only n_prompt.
    // ENERGY-AWARE CONTROLLER (opt-in via --energy-aware).
    //
    // Reads the phone's state of charge ONCE, here, and picks a k-pct tier. Once per
    // request is not a limitation to work around -- eviction_bench is one process per
    // request, so mid-generation adaptation is structurally impossible and the budget
    // is guaranteed stable for the whole answer. That is the property you want: a
    // request never changes fidelity halfway through.
    //
    // NEVER DEGRADES ON MAINS. If status is Charging/Full the top tier is used
    // regardless of SoC: trading answer quality for energy while plugged in buys
    // nothing.
    //
    // HYSTERESIS. SoC is an integer percent from a smoothed fuel gauge, so a request
    // arriving at exactly the threshold could flip tiers between consecutive requests.
    // The last level is persisted and a 3-point band must be crossed to move DOWN a
    // tier, while moving UP (less aggressive) is immediate -- asymmetric on purpose,
    // because recovering quality should not be delayed.
    //
    // current_now / power_now are deliberately NOT read: they report 0 whenever USB is
    // attached on this device, which is also why system energy is integrated from the
    // USB rail plus the coulomb counter rather than from the battery's own power node.
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

        // FALLBACK: on stock Android the sysfs battery nodes are root-only, and this
        // process runs as `shell`. `dumpsys battery` exposes the same state to an
        // unprivileged caller, which matters: a controller that needs root is not a
        // deployable design. Status codes are Android BatteryManager constants --
        // 2 = CHARGING, 5 = FULL are "on mains"; 3 = DISCHARGING, 4 = NOT_CHARGING are not.
        if (!have_soc) {
            std::FILE * pp = popen("dumpsys battery 2>/dev/null", "r");
            if (pp) {
                char line[256];
                int lvl = -1, st = -1;
                // The key must be the FIRST token on the line. `dumpsys battery`
                // also prints "Capacity level: 3" (a discrete BatteryManager bucket),
                // and a plain substring search takes the LAST match -- which silently
                // reported soc=3% on a phone at 98% and pinned the controller to its
                // most aggressive tier. A mis-read here degrades output quality with
                // no symptom, so the match is anchored.
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
        // TIERS IN ABSOLUTE K, anchored on the shipped --k-nominal, identical on both
        // backends. The healthy tier is the quality-validated point; energy that a full
        // battery or mains makes free is spent on answer quality, and only a low battery
        // trades quality for joules.
        //   level 0 (healthy / mains): K = k_nominal      (1024 by default)
        //   level 1 (mid):             K = k_nominal / 2  (512)
        //   level 2 (low):             K = k_nominal / 4  (256)
        // --ea-k hi mid lo sets the three absolute values; --ea-pct hi mid lo selects the
        // legacy percent-of-prompt ladder instead.
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
            // GPU ACTION: cap the GPU clock, hold K (see the field comments for the data).
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
            // CPU ACTION: shrink the cache, leave the clock alone.
            args.k_pct     = 0.0f;      // absolute ladder wins over any --k-pct on the command line
            args.k_nominal = k_sel;
            std::fprintf(stderr, "[energy-aware] soc=%d%% status=%s mains=%d backend=CPU prev_level=%d -> level=%d K=%d (ladder %d/%d/%d)\n",
                         soc, status_s.c_str(), (int) on_mains, prev, level, k_sel, k_hi, k_mid, k_lo);
        }
        args.ea_soc_seen = soc; args.ea_status_seen = status_s; args.ea_level_seen = level;
        { std::FILE * f = std::fopen(args.ea_state_file.c_str(), "w");
          if (f) { std::fprintf(f, "%d\n", level); std::fclose(f); } }
    }

    // --- Prefill ---
    // --force-cb-eval must also switch the capture ON, or the callback returns early at
    // its !cap->active guard and the diagnostic run is identical to a normal vanilla run.
    cap.reset(); cap.active = (args.policy != "vanilla") || args.force_cb_eval;
    cap.obs_window = std::max(1, args.obs_window);  // propagate observation-window setting
    // EndurKV Solution 2: enable the FA-on kq_evict side node for the prefill so
    // the gate is captured while the model runs FlashAttention-ON (no FA-off, no
    // state-swap). Disabled again once prefill+eviction is done.
    if (args.fa_on_evict) llama_endurkv_set_evict_obs_window(cap.obs_window);
    int64_t t_prefill_0 = ggml_time_us();
    // Persistent state for H2O — accumulates attention across every forward pass.
    // Must be declared BEFORE the prefill loop so that we can roll the chunk's
    // attention into the accumulator after each llama_decode, otherwise the
    // per-callback `cap.per_layer[layer] = std::move(buf)` (eval_callback)
    // overwrites earlier chunks and only the LAST chunk contributes to the
    // heavy-hitter accumulator (the canonical H2O paper requires cumulative
    // attention over ALL forward passes including every prefill chunk).
    H2OState h2o_state;
    // Persistent state for v1_predictive — ring buffer of mean-across-heads
    // attention vectors per layer. Updated every decode forward pass.
    AttnHistory attn_hist;
    // Optional per-chunk prefill trace (--out-prefill-csv). One fprintf per
    // chunk → ~14 writes for a 7K-token prefill = negligible overhead (~tens
    // of μs each, against ~95 s/chunk processing time).
    FILE * prefill_csv = nullptr;
    if (!args.out_prefill_csv.empty()) {
        prefill_csv = std::fopen(args.out_prefill_csv.c_str(), "w");
        if (prefill_csv) std::fprintf(prefill_csv,
            "chunk_idx,wall_us,n_tokens_processed,n_kv_cells\n");
    }
#ifdef EVB_HAS_MTMD
    if (vl_ctx) {
        // μKV×VL prefill (2026-07-25): evaluate mtmd chunks (text → image
        // embeddings → text) sequentially. The chat template puts the question
        // in the FINAL text chunk, so enabling the kq_evict side node only for
        // that chunk preserves the last-chunk prefill-tax fix semantics exactly
        // (the obs window = the question tail; scores cover ALL cells including
        // image tokens). mtmd_helper_eval_chunk_single batches internally at
        // n_batch and advances M-RoPE positions correctly.
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
        // Chunk prefill into n_batch-sized pieces. On Adreno Vulkan, a single
        // 2840-token llama_decode crashes the GPU watchdog even with small
        // n_ubatch — llama-bench works because it batches at n_batch=512.
        const int chunk = std::max(1, args.n_batch);
        int chunk_idx = 0;
        for (int off = 0; off < n_prompt; off += chunk) {
            int n = std::min(chunk, n_prompt - off);
            // EndurKV Solution 2 prefill-tax fix: the kq_evict side node's scores
            // are OVERWRITTEN each chunk (eval_callback: cap.per_layer[layer] =
            // std::move(buf)), and apply_eviction() below runs ONCE after this loop
            // using ONLY the last chunk's scores. Emitting + reading (and forcing a
            // per-node compute/sync boundary at) the side node on the earlier ~18
            // chunks is pure waste — ~300 GPU pipeline stalls collapse to ~16. So
            // emit the side node ONLY on the last prefill chunk: the surviving
            // scores are bit-identical, so selection / PPL / the α-gate are
            // unchanged, and decode is untouched. Only fa_on_evict emits the side
            // node (H2O / predictive fold every chunk via the FA-off path above and
            // are not affected by this global).
            if (args.fa_on_evict) {
                const bool is_last_chunk = (off + n >= n_prompt);
                llama_endurkv_set_evict_obs_window(is_last_chunk ? cap.obs_window : 0);
            }
            llama_batch batch = llama_batch_get_one(ptoks.data() + off, n);
            if (llama_decode(ctx, batch) != 0) {
                std::fprintf(stderr, "prefill fail at offset %d (n=%d)\n", off, n);
                return 1;
            }
            // H2O accumulator fix: fold THIS chunk's attention into the running
            // sum BEFORE the next chunk overwrites cap.per_layer in eval_callback.
            // Without this, only the last prefill chunk's attention seeds H2O.
            // endurkv_optimal also uses policy_h2o() for its prefill-time eviction
            // (see dedicated branch below), so the same accumulator must be fed.
            if (args.policy == "h2o" || args.policy == "endurkv_optimal") h2o_state.update(cap);
            // v1_predictive: also fold chunk attention into the history ring buffer
            // so the slope estimate has prefill samples to work with at decode
            // step 0. (Without this, slope=-INF for the first W-3 decode steps
            // and the policy degrades to pure recency.)
            if (args.policy == "v1_predictive") attn_hist.push(cap);
            // Per-chunk prefill trace (after llama_decode so n_kv reflects the
            // post-chunk cache state). One fprintf, no fflush → buffered.
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

    // --- Eviction after prefill (sets the working set going into decode) ---
    int evicted_prefill = 0;
    if (args.policy == "v1" || args.policy == "v1_fa" || args.policy == "v1_adaptive") {
        PolicyResult r = (args.policy == "v1_adaptive")
            ? policy_v1_adaptive(cap, args.k_nominal, n_kv_heads)
            : policy_v1(cap, args.k_nominal, n_kv_heads);
        protect_sink(r, args.n_sink, cap.n_kv);
        evicted_prefill = apply_eviction(ctx, r, cap.n_kv);
        // SELECTIVE ANCHORING (v1_FA² Wave-8 fix): after v1's spread-gate
        // eviction, further filter the survivors to the top-N by mean attention
        // score. Reduces anchored block size so recent decode window can be
        // larger → better PPL on long-decode workloads.
        // Triggered by snapkv_decode (v1_fa2 stack) OR by v1_adaptive, which
        // wants a tight anchored block but does NOT use state-swap.
        // ADAPTIVE ANCHOR DISPATCHER (Signal 1: distant sharpness).
        // Computes alpha_a per prompt from per-(layer,head) attention peaks.
        // A head whose argmax sits outside the trailing recent-default window
        // [N - adaptive_rmin, N) is treated as "anchor-worthy".
        // alpha_a = distant_count / total_count, clipped to [0.05, 0.95].
        // Overrides --anchor-top-k and --recent-budget with new_anchor =
        // round(alpha_a * (K - n_sink)), new_recent = (K - n_sink) - new_anchor.
        // Enables snapkv_decode flow so the override actually runs through L3.
        // ------------------------------------------------------------------
        // PHASE-AWARE CLOCK: prefill is done; lower the GPU cap for decode if asked.
        if (args.gpu_mhz_decode > 0 && args.n_gpu_layers > 0) {
            args.gpu_mhz_decode_rc = gpu_cap_write(args.gpu_mhz_decode);
            std::fprintf(stderr, "[phase-clock] prefill done; GPU cap for decode = %d MHz (write_rc=%d)\n",
                         args.gpu_mhz_decode, args.gpu_mhz_decode_rc);
        }
        // Resolve a percentage budget now that n_prompt is known. Done BEFORE
        // adaptive-anchor so the anchor/recent split is computed against the real K.
        if (args.k_pct > 0.0f) {
            const int k_from_pct = (int)std::lround(args.k_pct / 100.0f * (float)n_prompt);
            const int k_floor    = args.n_sink + args.adaptive_rmin + 1;   // sinks + recent window must fit
            args.k_nominal = std::max(k_floor, k_from_pct);
            std::fprintf(stderr, "[mukv] K from --k-pct %.1f%% of %d prompt tokens -> K=%d\n",
                         args.k_pct, n_prompt, args.k_nominal);
        }
        if (args.adaptive_anchor) {
            // MASS-BASED ADAPTIVE ANCHOR DISPATCHER (v2).
            // Partition prefill attention mass between "distant" (anchor-worthy) and
            // "recent" (last R_min prompt tokens) regions, using the same column-mean
            // signal that drives anchor selection.
            //   col_mean[p]   = mean over (layer, head, query) of attention[q,l,h,p]
            //   N_eff         = min(n_prompt, cap.n_kv)   -- actual prompt extent in cache
            //   recent_cutoff = max(0, N_eff - R_min)
            //   distant_mass  = sum col_mean[p] for p in [0, recent_cutoff)
            //   recent_mass   = sum col_mean[p] for p in [recent_cutoff, N_eff)
            //   alpha_a       = distant_mass / (distant_mass + recent_mass)
            // Captures IMPORTANCE distribution (varies per prompt), not just argmax position.
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

            // Legacy count-based metric (kept for log cross-check; not used to set alpha_a)
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

            // --gate-count: use the COUNT-based α_a (anchor-heavy) instead of MASS.
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
            args.snapkv_decode = !(args.no_state_swap || args.fa_on_evict);   // no swap for --no-state-swap OR --fa-on-evict (the latter is already FA-on end-to-end)
            args.no_evict_decode = true;
        }
        if (args.anchor_top_k > 0 && (args.snapkv_decode || args.policy == "v1_adaptive" || args.fa_on_evict)) {
            // Compute per-position score from attention received across (layer, head).
            // mode "mean":     sum/mean of attention values  (high = broadly+strongly attended)
            // mode "entropy":  cross-(layer,head) Shannon entropy of the per-position
            //                  attention distribution (high = MANY heads attend = broadly
            //                  relevant; low = only a few heads attend = head-specific)
            // mode "neg_entropy": invert (low entropy preferred = "sharp consensus" positions)
            std::vector<float> pos_score(cap.n_kv, 0.0f);
            int n_score_samples = 0;
            const bool use_entropy   = (args.anchor_score_mode == "entropy");
            const bool use_neg_entropy = (args.anchor_score_mode == "neg_entropy");
            const bool use_hybrid    = (args.anchor_score_mode == "hybrid");
            if (use_hybrid) {
                // HYBRID: mean attention * (1 + alpha * sharpness)
                // where sharpness = 1 - normalized_entropy in [0, 1]
                // Preserves magnitude (mean) and multiplicatively boosts sharply-attended positions.
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
                // Tie-break ties by mass so degenerate (zero-mass) positions don't beat real ones.
                const float scale = 1e-6f;
                for (int p = 0; p < cap.n_kv; ++p) pos_score[p] += scale * pos_mass[p];
                n_score_samples = H_total;
            } else if (args.anchor_score_mode == "gate_weighted") {
                // GATE-WEIGHTED MEAN: weight each (layer, head)'s contribution to pos_score
                // by its own peak m_h = max_p Ā_h[p]. Sharp heads get more vote weight;
                // diffuse heads less. Preserves the per-head peakedness signal that L1's
                // closed-form ramp captures but that the default cross-head mean discards.
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
                // Default "mean" mode (original behavior)
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
            // SnapKV 1D max-pool over the position scores: replace each score by the max
            // in a ±(kernel/2) window so top-K selects coherent CLUSTERS of tokens (the
            // SnapKV paper's pooling step). Active only when --snapkv-pool > 1.
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
                // gate bypassed: the pool is every PROMPT position (cells past n_prompt are chunk padding,
                // which the per-head gate excluded incidentally)
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
            // [DIAG] sink-EXCLUDED content-spread signals: does the non-sink attention
            // spread track content demand? Count non-sink positions needed for 50/90%
            // of the non-sink attention mass, plus normalized entropy of that distribution.
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
            // ADAPTIVE ANCHOR: if anchor_coverage > 0, derive `top` from cumulative
            // mass instead of using a fixed anchor_top_k. Sort scores descending,
            // accumulate until cumulative_mass / total_mass >= coverage, take that
            // many positions as the anchor set.
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
                // Build a "keep these positions" set; everything else gets dropped.
                std::vector<uint8_t> keep(cap.n_kv, 0);
                for (int i = 0; i < top; ++i) keep[ranked[i].second] = 1;
                if (const char * dump = std::getenv("EVICT_DUMP_KEEP")) {   // 2026-09-07: keep-set dump for the gate A/B
                    if (FILE * kf = std::fopen(dump, "w")) {
                        for (int p = 0; p < cap.n_kv; ++p) if (keep[p]) std::fprintf(kf, "%d\n", p);
                        std::fclose(kf);
                    }
                }
                // Drop everything NOT in the top set.
                llama_memory_t mem = llama_get_memory(ctx);
                int dropped_extra = 0;
                // 2026-07-25 (μKV×VL): M-RoPE caches must be evicted by CELL
                // INDEX — every token of an image shares one dim-0 position, so
                // the position-based path below cannot express this mask (it is
                // all-or-nothing per image). Text path unchanged (bit-identical).
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
        // endurkv_optimal: use canonical H2O eviction (50/50 recent + heavy
        // split) to COMPUTE candidate keep positions PER LAYER, then collapse
        // them to a sequence-level keep set via top-K cumulative-attention
        // ranking. The collapse step is required because llama.cpp's KV
        // eviction is sequence-level (seq_rm operates on (seq, pos), not
        // (seq, layer, pos)), so naively OR-ing per-layer keep sets across
        // 16 layers × 32 query heads × 256 heavy picks → ~all positions
        // kept → nothing evicted (same root cause as h2o's pre-fix bug
        // noted in LLAMA1B_COMPREHENSIVE_TABLE.md). The collapse uses the
        // H2O cumulative-attention accumulator that was already populated
        // during chunked prefill, so the keep set IS the H2O top-K by
        // cumulative attention — the canonical heavy-hitter signal.
        //
        // Algorithm:
        //   1. policy_h2o → per-layer keep sets (recent K/2 + heavy K/2).
        //   2. protect_sink (sink positions always kept).
        //   3. Score every position by mean cumulative attention across
        //      layers + heads (using the populated h2o_state).
        //   4. Among positions kept by ANY layer in step 1, keep the
        //      top K_nominal by score (sink always wins).
        //   5. apply_eviction drops everything else.
        // This is "H2O 50/50 selection + sequence-level top-K collapse"
        // — the v1_fa2 state-swap machinery (snapkv_decode + no_evict_decode
        // set by the alias) then hands the compressed cache off to an FA-on
        // decode context.
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
            // Rank candidates by score; sink positions always win.
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
        // With FA on there is no cb_eval and therefore no capture, but this policy only
        // needs the cache extent and the layer count -- no attention values. Synthesize
        // exactly that: empty per-layer entries (zero bytes) so the keep-set loop runs.
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
        // KeyDiff (Park et al., NeurIPS 2025, arXiv:2504.15364). ADDED 2026-08-16 --
        // the mobile-positioned eviction baseline this harness previously lacked.
        // Scores come from libllama (llama_endurkv_keydiff_scores: -cos of each key
        // to the per-layer mean key, averaged over layers); keep the K_nominal MOST
        // distinctive positions. Runs FA-on with no capture: the policy needs key
        // geometry only.
        //
        // FIDELITY NOTES, to be repeated wherever these numbers are reported:
        //  * their eviction is BLOCK-WISE during prefill (B=128), which bounds peak
        //    cache; this realization selects once at end of prefill, so the final
        //    keep-set rule is theirs but the PEAK-memory column is not comparable.
        //  * their paper treats K as one matrix and specifies no cross-layer rule;
        //    the mean over layers here is the sequence-level realization this
        //    engine's shared cell array requires -- the same adaptation every
        //    baseline gets, documented rather than silent.
        //  * faithful invocation has NO sinks (--n-sink 0): vanilla KeyDiff keeps
        //    nothing positional. The flag is honored if passed, but a sink is a
        //    deviation from their published policy.
        //  * mask is frozen during decode (their decode-time cadence is likewise
        //    block-wise; the frozen realization matches how SnapKV/Ada-KV are run
        //    here, keeping the cross-policy comparison like-for-like).
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
        // budget allocation with α=0.2 safeguard; mask is frozen during decode.
        PolicyResult r = policy_adakv(cap, args.k_nominal, n_kv_heads);
        protect_sink(r, args.n_sink, cap.n_kv);
        evicted_prefill = apply_eviction(ctx, r, cap.n_kv);
    } else if (args.policy == "snapkv") {
        // Canonical SnapKV (Li et al. 2024): per-head top-K over avgpool-5 pooled
        // observation-window attention, uniform budget; mask frozen during decode.
        PolicyResult r = policy_snapkv(cap, args.k_nominal, args.obs_window, args.snapkv_kernel);
        protect_sink(r, args.n_sink, cap.n_kv);
        evicted_prefill = apply_eviction(ctx, r, cap.n_kv);
    } else if (args.policy == "v1_entropy") {
        PolicyResult r = policy_v1_entropy(cap, args.k_nominal, n_kv_heads);
        protect_sink(r, args.n_sink, cap.n_kv);
        evicted_prefill = apply_eviction(ctx, r, cap.n_kv);
        // Selective anchoring (same as v1_fa2 path): when running the stack
        // alias with snapkv_decode, further filter survivors to top anchor_top_k
        // by mean attention score so the recent_budget can host more decode-time
        // positions.
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

    // -----------------------------------------------------------------------
    // REFRESH AT DECODE STEP 0 (RefreshKV-inspired, mobile-deployable variant)
    // After prefill + initial anchor selection, run ONE FA-off decode step to
    // capture attention from the model's "first response" query. Then re-rank
    // anchor selection using this fresh attention. Then continue to state-swap
    // as normal. Tests whether decode-time queries carry additional information
    // beyond prefill scoring (obs_window).
    // -----------------------------------------------------------------------
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

        // Clear cap.per_layer so cb_eval captures ONLY the refresh step's attention
        cap.per_layer.clear();

        // Decode 1 step in FA-off (cb_eval still active) — cap.n_kv will update
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
                // ---------- L7-style adaptive α_a from decode-step attention ----------
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

            // ---------- L1-style per-head K_h candidate pool ----------
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

            // ---------- L2-style cross-head mean over the candidate pool ----------
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

    // -----------------------------------------------------------------------
    // Strategy A — SnapKV-style two-context swap.
    // Active for: any non-vanilla policy when --snapkv-decode is set.
    // After prefill+eviction the KV is in the "compressed" state. Swap to an
    // FA-on context so decode runs at vanilla speed.
    // -----------------------------------------------------------------------
    // fa-on-evict CPU defrag: fa-on prefill leaves the KV physically SPARSE (seq_rm
    // marks holes but never compacts), so FA-on decode traverses the full prompt span
    // (~30% slower, +52 MB RSS vs state-swap). The state get/set round-trip below drops
    // the holes -> contiguous cache. Enable it for fa_on_evict on CPU (n_gpu_layers==0);
    // on GPU/Adreno the round-trip transfer is prohibitively slow, so fa_on_evict skips
    // it there. Source and dest are BOTH FA-on here (no cross-FA-mode layout risk), and a
    // failed restore falls back to the current context, so this is compaction-or-noop.
    const bool want_defrag = args.defrag_mode >= 0 ? (args.defrag_mode == 1)
                           : (args.n_gpu_layers == 0 || detect_gpu_profile_wants_defrag());
    // GENERALIZED 2026-08-07: compaction is a MECHANISM, not a property of muKV's
    // selection rule. It used to be reachable only via snapkv_decode or fa_on_evict,
    // which meant a baseline could never be measured with it -- and since compaction is
    // worth ~2x while eviction alone is worth ~1.03x, that silently attributed our
    // mechanism's speedup to our policy. An explicit --compact-inplace now applies to
    // ANY non-vanilla policy, so baselines can be given the same mechanism and the
    // selection rules can be compared on equal footing.
    // FIXED 2026-08-08: this was ASYMMETRIC. --compact-inplace was generalized to all
    // policies but --force-defrag was not, so for a baseline (fa_on_evict=false,
    // snapkv_decode=false) --force-defrag silently did NOTHING. A cross-policy check
    // that thought it was comparing round-trip against in-place was really comparing
    // NO COMPACTION against in-place for StreamingLLM, TOVA and SnapKV, which made
    // TOVA look like a 2.6% quality mismatch between the two modes. An explicit
    // --force-defrag (defrag_mode==1) now enables the round-trip for any non-vanilla
    // policy, exactly as --compact-inplace does.
    if ((args.snapkv_decode || (args.fa_on_evict && want_defrag) || args.compact_inplace ||
         args.defrag_mode == 1) &&
        !args.no_state_swap && args.policy != "vanilla") {
        std::fprintf(stderr, "[mukv] scoring host state: %.1f MiB (%zu layers x n_head*n_kv floats)\n",
                     cap.peak_host / (1024.0*1024.0), cap.per_layer.size());
        int64_t t_swap_0 = ggml_time_us();
        // ---- in-place, chunked compaction (memory-neutral alternative) ----------
        // Same goal as the round-trip below -- land the survivors CONTIGUOUSLY so that
        // attention stops scanning to the highest occupied index -- but it reuses the
        // tensors prefill already allocated instead of restoring into a second context.
        // Peak memory therefore does not rise at all, which is the entire point: the
        // round-trip's 2x peak makes compaction infeasible exactly where it is worth
        // most (Phi-3 at 16K on the phone, and 64K on any of our devices).
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
                // declined (cache holds other sequences) -- fall through to the round-trip
                std::fprintf(stderr, "[mukv] in-place compaction declined; using state round-trip\n");
            }
        }
        if (!inplace_ok) {
        // Heap-buffered state transfer. The file-backed alternative
        // (state_seq_save_file/load_file) was tested in Wave-5 and INCREASED
        // total UFS writes: writing a 3.7 GB state file caused the kernel to
        // evict other processes' anon pages to swap to make room for the
        // file page cache. Net swap-out went 515 MB → 739 MB (+43%). Reverted.
        //
        // For a true memory fix, per-layer streaming state transfer would
        // avoid both the heap buffer AND the UFS file. Left as future work.
        size_t state_size = llama_state_seq_get_size(ctx, /*seq_id=*/0);
        std::fprintf(stderr, "[snapkv] state_size=%zu bytes (%.1f MiB)\n",
                     state_size, state_size / (1024.0 * 1024.0));
        std::vector<uint8_t> state_buf(state_size);
        size_t got = llama_state_seq_get_data(ctx, state_buf.data(), state_size, /*seq_id=*/0);
        if (got == 0) {
            std::fprintf(stderr, "[snapkv] state_get failed; falling back to FA-off decode\n");
        } else {
            // Build the FA-on decode context WITHOUT tearing down the FA-off
            // context yet, so that if the state restore fails (e.g. the Vulkan
            // backend serializes the KV cache with a layout that differs across
            // FA modes) we can gracefully fall back to FA-off decode over the
            // already-compressed KV instead of aborting the whole run. muKV
            // eviction stays in effect either way; only COMPACTION is forgone.
            //
            // NOTE (2026-08-01): this block serves two different paths and they
            // must not be conflated.
            //   snapkv_decode : prefill FA-OFF -> decode FA-ON. A real FA-mode switch.
            //   fa_on_evict   : prefill FA-ON  -> decode FA-ON. use_fa is already true
            //                   (see `bool use_fa = ... || args.fa_on_evict`), so the
            //                   second context changes NOTHING about FlashAttention.
            //                   Its only effect is to land the surviving cells
            //                   CONTIGUOUSLY, because seq_rm marks holes without
            //                   moving survivors and attention still scans to the
            //                   highest occupied index.
            // The log strings used to say "continuing FA-off decode" on both paths,
            // which is false for fa_on_evict and caused a measured 2.42x speedup to be
            // mis-attributed to an FA-mode change rather than to compaction.
            cap.active = false;

            // Build FA-on context. No cb_eval (we don't need attention anymore).
            llama_context_params on_params = llama_context_default_params();
            // REVERTED 2026-08-03. An attempt to size this destination context to the
            // SURVIVING cells (live + remaining generation) instead of the full n_ctx
            // made compaction fit at 64K and appeared to give 13.09x decode -- but it
            // CORRUPTS THE CACHE and the result is worthless: Phi-3 PPL went 7.88 ->
            // 78690 at 16K, where the full-size destination is known good.
            // Reason: llama_state_seq_set_data restores cells carrying their ORIGINAL
            // positions (required for RoPE), so the destination must span the original
            // position range, not merely hold the surviving population. Compaction makes
            // the cells CONTIGUOUS; it does not renumber them. A destination smaller than
            // the position range therefore silently produces a fast, wrong model.
            // Sizing it down needs position renumbering in the restore path, which is a
            // llama.cpp change, not a caller-side one. Until then the memory ceiling at
            // 64K (2 x full cache + weights > VRAM) is a REAL limitation of this design.
            on_params.n_ctx           = cparams.n_ctx;
            on_params.n_threads       = cparams.n_threads;
            on_params.n_threads_batch = cparams.n_threads_batch;
            on_params.type_k          = cparams.type_k;
            on_params.type_v          = user_type_v;
            on_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
            on_params.cb_eval         = nullptr;
            on_params.cb_eval_user_data = nullptr;
            // FIXED 2026-08-06: the destination must place ALL surviving cells in ONE
            // find_slot() call -- llama_state_seq_set_data builds a single ubatch of
            // cell_count tokens. The default n_batch (2048) silently capped compaction
            // at ~2048 live cells: at ctx 65536, K=1024 (823 cells) restored fine while
            // K=4096/8192/16384 (3301/6606/13215 cells) hit "failed to find available
            // cells in kv cache", fell back to the SPARSE path, and reported
            // compaction_applied=false with no speedup -- which looked like compaction
            // being useless at larger budgets when it was never running.
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
                    // Backend (e.g. Vulkan/Adreno) cannot restore the FA-off KV
                    // blob into an FA-on context. Keep the FA-off context and
                    // decode FA-off over the compressed KV — μKV eviction is
                    // still fully applied (memory + correctness); only the FA-on
                    // decode speedup is forgone on this backend.
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
                    // Warm-up: re-decode the LAST prompt token so the FA-on
                    // context's output buffer is populated for get_logits_ith.
                    // State load restores KV cells but n_outputs starts at 0 in
                    // the new context — without this the first sampling call
                    // fails with "corrupt output buffer (n_outputs=0)". Remove
                    // the last position from KV first to avoid duplicating it.
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
        }   // end if (!inplace_ok) -- the state round-trip path
    }

    // --no-state-swap: decode stays on the FA-off prefill context (no swap), but
    // the attention capture must be frozen or every decode token pays a host
    // readback of kq_soft_max (the H2O/AdaKV per-step slowdown). Eviction is
    // already done (no_evict_decode), so no capture is needed during decode.
    if ((args.no_state_swap || args.fa_on_evict) && args.policy != "vanilla") {
        // --ppl-per-step needs the FA-off kq_soft_max capture alive during the
        // teacher-forced eval pass so per-step policy re-scoring sees fresh
        // attention. Only possible on the FA-off (no-state-swap) path:
        // fa_on_evict runs FA-ON and has no kq_soft_max to capture — its evict
        // side node stays off for decode, so per-step eviction degrades to a
        // no-op there (the PPL run still completes, just without re-eviction).
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
    // For v1_FA² tiered eviction: anchor = current n_kv after all prefill-time
    // evictions (and after any state-swap). Captured BEFORE decode starts so
    // it represents the attention-aware selection's footprint.
    int    n_anchored = (int)(llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1);
    // 100%-CORRECT retained cache after prefill eviction (+swap/defrag for muKV):
    // llama_state_seq_get_size serializes ONLY live cells, so this is the TRUE compacted
    // KV footprint for EVERY policy -- NOT the position-range peak_kv (which is muddy).
    // Per-head baselines (SnapKV/AdaKV) that can't compact report a large value; muKV small.
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

    // -----------------------------------------------------------------------
    // PPL evaluation mode: teacher-force a reference text (DISJOINT from prefill)
    // after prefill+evict. This is the KIVI/H2O-style KV-eviction PPL protocol,
    // NOT llama-perplexity's sliding-window scheme. Prefill text and eval text
    // MUST be different chunks; the launcher script (phone_wave11_eval.sh)
    // enforces this by passing chunk i as --prompt and chunk i+1 as --eval-text.
    // RAW logit log-probs of the actual next token; no sampling/temp/rep-penalty.
    // -----------------------------------------------------------------------
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
        // Progressive (per-step) teacher-forced PPL: with --ppl-per-step the
        // policy-eviction block below runs after EVERY teacher-forced token,
        // overriding any frozen-mask preset (no_evict_decode). This reproduces
        // the canonical per-generation-step eviction schedule of TOVA /
        // TOVA-canonical / H2O during PPL scoring. ~4-6 tps FA-off — slow by
        // design; do NOT batch, the schedule fidelity is the point.
        const bool ppl_step_evict = args.ppl_per_step || !args.no_evict_decode;
        if (args.ppl_per_step) {
            std::fprintf(stderr,
                "[eviction_bench/ppl] --ppl-per-step: progressive per-step policy "
                "eviction ACTIVE during teacher-forcing (policy=%s, cap_active=%d)\n",
                args.policy.c_str(), cap.active ? 1 : 0);
        }
        // Feed one token at a time so logits_ith(-1) corresponds to the *next*-token
        // distribution given everything fed so far. Score rtoks[i] against the
        // distribution produced after feeding rtoks[0..i-1].
        // First token rtoks[0] is scored against the distribution from prefill's
        // last position, which is already in the context.
        for (int i = 0; i < n_ref; ++i) {
            const float * logits = llama_get_logits_ith(ctx, -1);
            if (!logits) break;
            // RAW softmax log-prob of rtoks[i] — no penalties, no temperature.
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
            // Quote+escape any commas/quotes in the token text
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
            // Optional: keep evicting during eval pass so the policy is exercised
            // exactly as it would be during real long-form generation. Use
            // --no-evict-decode to score under a frozen mask (SnapKV-style),
            // or --ppl-per-step to FORCE per-step eviction (progressive PPL)
            // regardless of frozen-mask presets. NLL of rtoks[i] was computed
            // ABOVE from the pre-feed logits, so the eviction applied here can
            // only influence the distribution used for rtoks[i+1] — the same
            // accounting as the gen loop (evict after decode, affect next step).
            // Mirrors the gen-mode per-step block verbatim: h2o_state.update →
            // policy dispatch → protect_sink (n_sink, except streamingllm) →
            // effective-KV accumulation → apply_eviction. Policies without a
            // dispatch entry (or with capture frozen after a state-swap) yield
            // an empty keep-set and degrade to a no-op.
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
        // overwrite the latency/decode metrics with eval-mode equivalents
        // (caller will see decode_tps = scored_tokens / eval_seconds — useful)
        std::fprintf(stderr, "[eviction_bench/ppl] scored=%d  mean_nll=%.4f  ppl=%.4f  eval_ms=%.1f\n",
                     n_scored, mean_ref_nll, ref_ppl, (t_eval_1 - t_eval_0)/1000.0);
    }

    // Repetition penalty window
    std::vector<int> recent_tokens;
    recent_tokens.reserve(args.repeat_last_n);
    // RNG seeded from args.seed for reproducible sampling
    std::mt19937 rng((uint32_t)args.seed);
    int64_t t_decode_0 = ggml_time_us();
    // In PPL mode we already scored the reference text above; skip the sampling loop.
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
                    // Compute effective-KV BEFORE applying eviction (use full attention)
                    EffectiveKV ekv = compute_effective_kv(cap, r, cap.n_head);
                    sum_mass_retained   += ekv.mass_retained;
                    sum_retention_ratio += ekv.retention_ratio;
                    sum_efficiency      += ekv.efficiency;
                    n_kv_samples++;
                    evicted_this = apply_eviction(ctx, r, cap.n_kv);
                }
                n_total_evicted += evicted_this;
            }
            // ====================================================================
            // TDAK: THERMAL-DRIVEN ADAPTIVE K — close the loop between cache size
            // and chip thermal state. Every thermal_poll_steps decode steps, read
            // DDR temperature and map it through a 4-tier ladder to a K_target.
            // The recent_budget for tiered eviction is updated to (K_target - n_sink
            // - n_anchored) so the cache shrinks/grows in response to thermals.
            // This is novel: no published KV policy reads hardware sensors to size
            // its cache. Prior thermal-aware mobile works (zTT, FUSE, BCL) act on
            // CPU frequency only.
            // ====================================================================
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
                    // Ratchet-down: K is monotone (tier non-decreasing) while heat rises;
                    // only step DOWN a tier once DDR cools >= hyst below that tier's ENTRY
                    // threshold, so K never ramps back up into a still-rising thermal front.
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
                    new_tier = raw_tier;                           // original symmetric ladder
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
            // Effective K and recent_budget used for this step's decode-time eviction:
            // - If TDAK is off, use the static args values (original behavior).
            // - If TDAK is on, derive recent_budget from tdak_current_k so the cache
            //   shrinks/grows in response to thermal feedback.
            int eff_recent_budget = args.recent_budget;
            int eff_k_nominal     = args.k_nominal;
            if (args.thermal_driven_k) {
                eff_k_nominal = tdak_current_k;
                int rb = tdak_current_k - args.n_sink - (n_anchored > 0 ? n_anchored : 0);
                eff_recent_budget = std::max(32, rb);
            }
            // ---- KeyDiff decode-time re-eviction (ADDED 2026-08-17) -------------
            // WHY THIS EXISTS. KeyDiff bounds the cache during GENERATION too, not
            // just prefill: "choosing B = 1 corresponds to the token generation
            // phase of LLM evaluation" (Park et al., NeurIPS 2025, Sec. 2.4), so
            // their cache never exceeds the budget N at any point. Without this,
            // a single end-of-prefill selection lets the cache grow N + n_decode --
            // on the 4K-decode WikiText workload that is 2048 -> 6144 cells, a 3x
            // inflation that would show up as "KeyDiff is slow" when it is really
            // our harness omitting their mechanism. That is the same class of
            // defect as the handicapped StreamingLLM row, and it is corrected here
            // BEFORE the number is measured rather than after a reviewer finds it.
            //
            // CADENCE. Their generation cadence is B=1 (re-evict every step). We
            // re-evict every keydiff_decode_block steps (default 128, their own
            // prefill block size) because a full re-score is O(n*d) per layer and
            // at B=1 the scoring would dominate decode on a phone CPU. The cost of
            // the approximation is bounded and must be reported: the cache sits in
            // [N, N + B] instead of exactly N -- at N=2048, B=128 that is <=6%
            // above their bound, and ActKV is measured, not assumed.
            //
            // The keep-set rule is theirs, unmodified: re-score every live cell by
            // -cos to the per-layer mean key, keep the K_nominal most distinctive.
            // Scores are recomputed from the CURRENT cache, so decode-generated
            // tokens compete with prompt tokens on equal footing -- which is what
            // their block-wise formulation does.
            // FIXED 2026-08-18. The first version gated on
            //   llama_memory_seq_pos_max() > k_nominal + block
            // which is WRONG: seq_pos_max returns the highest POSITION, and
            // in-place compaction does not renumber positions, so the condition
            // never cleared once true. The branch then fired on EVERY decode step
            // and re-scored the whole cache each time -- 7.4M evictions over 4096
            // steps instead of 32, and KeyDiff measured 0.46-0.90x vanilla when its
            // own paper reports a 30% latency REDUCTION. Those numbers were an
            // artifact of this bug and are discarded, not reported.
            // The cadence is now driven by the step counter, which is what "block
            // size B" means anyway, and is free to evaluate.
            static int kd_last_evict_step = -1;
            if (args.policy == "keydiff" && args.keydiff_decode_block > 0 &&
                (step - kd_last_evict_step) >= args.keydiff_decode_block) {
                const int n_kv_after = (int)(llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1);
                {
                    kd_last_evict_step = step;
                    std::vector<float> kds((size_t) n_kv_after, 0.0f);
                    const uint32_t kn = llama_endurkv_keydiff_scores(
                            llama_get_memory(ctx), 0, kds.data(), (uint32_t) kds.size());
                    if (kn > (uint32_t) args.k_nominal) {   // kn is the LIVE dense-prefix length
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
                        // Re-compact so the next scoring call still sees the dense
                        // prefix its precondition requires (it declines otherwise,
                        // which would silently disable decode-time eviction).
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
            // v1_FA² tiered decode-time eviction (PREFERRED over decode_bound when
            // both are set). Preserves the prefill-attention-anchored block and
            // only drops the OLDEST decode-generated positions. Maintains a
            // recency window for new tokens.
            else if (args.decode_tiered && args.no_evict_decode && n_anchored > 0) {
                int n_kv_after = (int)(llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1);
                int dropped = apply_tiered_decode_eviction(ctx, n_kv_after,
                                                           n_anchored, eff_recent_budget);
                evicted_this    += dropped;
                n_total_evicted += dropped;
            }
            // Cheap recency-based decode-time bounding (v1_fa auto-enables this).
            // Runs AFTER the attention-aware path so it only fires when the
            // attention-aware path is skipped (no_evict_decode) and cache grew.
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
        // Honor EOS by default; with --ignore-eos, record it but keep generating.
        if (is_eog) {
            if (eos_step < 0) eos_step = step;
            if (!args.ignore_eos) break;
        }
    }
    int64_t t_decode_1 = ggml_time_us();
    cap.active = false;
    if (gen) std::fclose(gen);
    std::fclose(csv);

    // --- summary meta ---
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

    // Render doubles as valid JSON: NaN/Inf → null (not bare "nan", which is illegal JSON).
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
            // ADDED 2026-08-02: obs_window/snapkv_kernel were not recorded, so the
            // SnapKV configuration of a finished run could only be inferred from the
            // launching script -- and that inference was wrong at least once (the
            // NIAH sweep passed --obs-window 64 to a binary where the flag was inert).
            // A baseline's exact configuration must be auditable from its own artifact.
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
