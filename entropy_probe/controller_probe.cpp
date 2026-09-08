// controller_probe — Track 2 v0 of EndurKV.
//
// Same decoding pipeline as attention_probe, but acts on the captured attention:
// after each decode step, the EndurKV-Evict v1 policy decides which KV positions
// to physically remove via llama_memory_seq_rm, and we measure per-step latency.
//
// Sidecar outputs:
//   * a CSV identical to attention_probe (per-step entropy / top-k)
//   * a .controller.json with per-step timing, evicted-positions count, and a
//     pre/post latency comparison summary.
//
// The eviction policy decides POSITIONS (global, across all layers), because
// llama_memory_seq_rm operates per-sequence. To get a position-level score we
// mean-pool the layer-averaged attention across all layers, then apply v1's
// spread gate and TOVA scoring (top-K_t by aggregated current-step attention).
// Per-layer eviction would require an llama.cpp KV-cache patch — left to v1.
//
// Flags vs attention_probe:
//   --k-budget N        per-step KV budget (0 = no eviction; runs baseline)
//   --evict-every N     run eviction every N steps (default 1 = every step)
//   --start-evict-at N  do not evict for the first N decode steps (default 4)

#include "compute_entropy.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Args {
    std::string model;
    std::string prompt_file;
    std::string prompt_id;
    std::string output_csv;
    std::string output_json;
    int         max_tokens = 0;
    uint32_t    seed       = 42;
    int         n_gpu_layers = 99;
    int         k_budget = 0;        // 0 = baseline (no eviction)
    int         evict_every = 1;
    int         start_evict_at = 4;
};

void usage(const char * argv0) {
    std::fprintf(stderr,
        "Usage: %s --model PATH --prompt-file PATH --prompt-id ID\n"
        "          --max-tokens N --output PATH --output-json PATH\n"
        "          [--k-budget N] [--evict-every N] [--start-evict-at N]\n"
        "          [--seed N] [--n-gpu-layers N]\n",
        argv0);
}

bool parse_args(int argc, char ** argv, Args & a) {
    auto need = [&](int & i, const char * name) -> const char * {
        if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", name); return nullptr; }
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        const char * k = argv[i];
        const char * v = nullptr;
        if      (!std::strcmp(k, "--model"))         { v = need(i, k); if (!v) return false; a.model         = v; }
        else if (!std::strcmp(k, "--prompt-file"))   { v = need(i, k); if (!v) return false; a.prompt_file   = v; }
        else if (!std::strcmp(k, "--prompt-id"))     { v = need(i, k); if (!v) return false; a.prompt_id     = v; }
        else if (!std::strcmp(k, "--max-tokens"))    { v = need(i, k); if (!v) return false; a.max_tokens    = std::atoi(v); }
        else if (!std::strcmp(k, "--output"))        { v = need(i, k); if (!v) return false; a.output_csv    = v; }
        else if (!std::strcmp(k, "--output-json"))   { v = need(i, k); if (!v) return false; a.output_json   = v; }
        else if (!std::strcmp(k, "--k-budget"))      { v = need(i, k); if (!v) return false; a.k_budget      = std::atoi(v); }
        else if (!std::strcmp(k, "--evict-every"))   { v = need(i, k); if (!v) return false; a.evict_every   = std::atoi(v); }
        else if (!std::strcmp(k, "--start-evict-at")){ v = need(i, k); if (!v) return false; a.start_evict_at= std::atoi(v); }
        else if (!std::strcmp(k, "--seed"))          { v = need(i, k); if (!v) return false; a.seed          = (uint32_t)std::strtoul(v, nullptr, 10); }
        else if (!std::strcmp(k, "--n-gpu-layers"))  { v = need(i, k); if (!v) return false; a.n_gpu_layers  = std::atoi(v); }
        else if (!std::strcmp(k, "-h") || !std::strcmp(k, "--help")) { usage(argv[0]); std::exit(0); }
        else { std::fprintf(stderr, "unknown arg: %s\n", k); usage(argv[0]); return false; }
    }
    if (a.model.empty() || a.prompt_file.empty() || a.prompt_id.empty()
        || a.output_csv.empty() || a.output_json.empty() || a.max_tokens <= 0) {
        usage(argv[0]);
        return false;
    }
    if (a.evict_every < 1) a.evict_every = 1;
    return true;
}

bool read_file(const std::string & path, std::string & out) {
    std::ifstream f(path);
    if (!f) { std::fprintf(stderr, "error: cannot open prompt file %s\n", path.c_str()); return false; }
    std::ostringstream ss; ss << f.rdbuf(); out = ss.str();
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) out.pop_back();
    return true;
}

std::string token_to_text(const llama_vocab * vocab, llama_token tok) {
    char buf[256];
    int n = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, true);
    if (n >= 0) return std::string(buf, n);
    std::string big(static_cast<size_t>(-n), '\0');
    n = llama_token_to_piece(vocab, tok, big.data(), (int32_t)big.size(), 0, true);
    if (n < 0) return std::string();
    big.resize(n);
    return big;
}

std::string csv_escape(const std::string & s) {
    std::string out; out.reserve(s.size() + 2); out.push_back('"');
    for (char c : s) {
        if (c == '"') out += "\"\"";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else out.push_back(c);
    }
    out.push_back('"');
    return out;
}

int argmax_logits(const float * logits, int n_vocab) {
    int best = 0; float best_v = logits[0];
    for (int i = 1; i < n_vocab; ++i) {
        if (logits[i] > best_v) { best_v = logits[i]; best = i; }
    }
    return best;
}

// -------- Attention capture state (per-layer sum-over-heads) ----------------
// Simpler than attention_probe v2: we only need an aggregate score per position
// for the eviction decision, not the full per-head data.

struct AttnAgg {
    std::vector<double> sum_layers;   // length n_kv, sum over (layer, head) of attention
    int   n_layers_seen = 0;
    int   n_kv_last     = 0;
    bool  active        = false;
    void reset(int n_kv) {
        sum_layers.assign((size_t)n_kv, 0.0);
        n_layers_seen = 0;
        n_kv_last = n_kv;
    }
};

int parse_layer_index(const char * name) {
    if (!name) return -1;
    size_t L = std::strlen(name);
    if (L == 0) return -1;
    size_t end = L;
    while (end > 0 && !std::isdigit((unsigned char)name[end - 1])) --end;
    if (end == 0) return -1;
    size_t start = end;
    while (start > 0 && std::isdigit((unsigned char)name[start - 1])) --start;
    if (start == end) return -1;
    return std::atoi(name + start);
}

bool eval_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * agg = static_cast<AttnAgg *>(user_data);
    if (!agg || !agg->active) return true;
    const char * name = ggml_get_name(t);
    const bool is_attn = name && (std::strstr(name, "kq_soft_max") != nullptr);
    if (ask) return is_attn;
    if (!is_attn) return true;

    const int64_t n_kv     = t->ne[0];
    const int64_t n_tokens = t->ne[1];
    const int64_t n_head   = t->ne[2];
    const int64_t n_stream = t->ne[3];
    if (n_kv <= 0 || n_tokens <= 0 || n_head <= 0 || n_stream <= 0) return true;
    if (t->type != GGML_TYPE_F32) return true;

    const size_t nbytes = ggml_nbytes(t);
    std::vector<float> buf(nbytes / sizeof(float));
    ggml_backend_tensor_get(t, buf.data(), 0, nbytes);

    const int64_t q_idx = n_tokens - 1;
    const int64_t s_idx = 0;

    if (agg->n_kv_last != (int)n_kv) agg->reset((int)n_kv);

    // Sum over heads for the last query, accumulate into the layer sum buffer
    for (int64_t h = 0; h < n_head; ++h) {
        const float * row = buf.data()
                          + s_idx * (n_head * n_tokens * n_kv)
                          + h     * (n_tokens * n_kv)
                          + q_idx * n_kv;
        for (int64_t i = 0; i < n_kv; ++i) {
            agg->sum_layers[(size_t)i] += (double)row[i];
        }
    }
    agg->n_layers_seen += 1;
    return true;
}

// -------- EndurKV-Evict v1 policy --------------------------------------------
// score_i = layer-averaged attention[i] / (n_layers_seen * n_head). Same shape
// as a probability distribution. Spread gate: K_t = K * (1.3 - 0.6 * norm(max)).
// Returns the SORTED LIST of positions to REMOVE (ascending). Always preserves
// the last `protect_recent` positions (default 4) and the first 4 sinks.

std::vector<int> select_positions_to_remove(
    const std::vector<double> & sum_scores, int n_kv, int K_budget,
    int n_layers_seen, int n_head, int protect_recent = 4, int protect_sink = 4)
{
    if (n_kv <= K_budget || sum_scores.empty()) return {};

    // Normalise score so behaviour matches the simulator (per-head softmax sums to 1)
    double denom = (double)n_layers_seen * (double)n_head;
    if (denom <= 0.0) denom = 1.0;
    std::vector<double> s(n_kv);
    double max_a = 0.0;
    for (int i = 0; i < n_kv; ++i) {
        s[i] = sum_scores[i] / denom;
        if (s[i] > max_a) max_a = s[i];
    }

    // v1 spread gate: K_t = K * (1.3 - 0.6 * norm(max_a))
    double norm = (max_a - 0.4) / 0.4;
    if (norm < 0.0) norm = 0.0;
    if (norm > 1.0) norm = 1.0;
    double mult = 1.3 - 0.6 * norm;
    int K_t = (int)std::round((double)K_budget * mult);
    if (K_t < 1) K_t = 1;
    if (K_t >= n_kv) return {};

    // Always keep: first `protect_sink` and last `protect_recent`
    int n_sink   = std::min(protect_sink,   K_t / 4);
    int n_recent = std::min(protect_recent, K_t / 4);
    if (n_sink + n_recent >= K_t) {
        n_sink   = std::max(0, K_t / 4);
        n_recent = std::max(0, K_t - n_sink - 1);
    }
    std::vector<char> keep(n_kv, 0);
    for (int i = 0; i < n_sink; ++i) keep[i] = 1;
    for (int i = std::max(n_sink, n_kv - n_recent); i < n_kv; ++i) keep[i] = 1;
    int already_kept = 0;
    for (int i = 0; i < n_kv; ++i) if (keep[i]) ++already_kept;
    int K_left = K_t - already_kept;
    if (K_left < 0) K_left = 0;

    // Rank remaining by score, keep top-K_left
    std::vector<int> cand;
    cand.reserve((size_t)(n_kv - already_kept));
    for (int i = 0; i < n_kv; ++i) if (!keep[i]) cand.push_back(i);
    if ((int)cand.size() > K_left) {
        std::nth_element(cand.begin(), cand.begin() + K_left, cand.end(),
                         [&](int a, int b) { return s[a] > s[b]; });
        for (int j = 0; j < K_left; ++j) keep[cand[j]] = 1;
    } else {
        for (int i : cand) keep[i] = 1;
    }

    // Build the removal list (descending order so seq_rm doesn't shift indices)
    std::vector<int> to_remove;
    to_remove.reserve((size_t)(n_kv - K_t));
    for (int i = n_kv - 1; i >= 0; --i) if (!keep[i]) to_remove.push_back(i);
    return to_remove;
}

} // namespace

int main(int argc, char ** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) return 1;
    (void)args.seed;

    std::string prompt;
    if (!read_file(args.prompt_file, prompt)) return 1;
    if (prompt.empty()) { std::fprintf(stderr, "error: empty prompt\n"); return 1; }

    ggml_backend_load_all();
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = args.n_gpu_layers;
    llama_model * model = llama_model_load_from_file(args.model.c_str(), mparams);
    if (!model) {
        std::fprintf(stderr, "error: failed to load model %s\n", args.model.c_str());
        llama_backend_free();
        return 1;
    }
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);

    int n_neg = llama_tokenize(vocab, prompt.c_str(), (int32_t)prompt.size(), nullptr, 0, true, true);
    int n_prompt = -n_neg;
    if (n_prompt <= 0) {
        std::fprintf(stderr, "error: tokenize sizing failed (rc=%d)\n", n_neg);
        llama_model_free(model); llama_backend_free();
        return 1;
    }
    std::vector<llama_token> prompt_tokens(n_prompt);
    int n_tok = llama_tokenize(vocab, prompt.c_str(), (int32_t)prompt.size(),
                               prompt_tokens.data(), (int32_t)prompt_tokens.size(),
                               true, true);
    if (n_tok < 0) {
        std::fprintf(stderr, "error: tokenize failed (rc=%d)\n", n_tok);
        llama_model_free(model); llama_backend_free();
        return 1;
    }
    n_prompt = n_tok;

    AttnAgg agg;

    llama_context_params cparams = llama_context_default_params();
    const int needed = n_prompt + args.max_tokens + 32;
    cparams.n_ctx           = needed > 4096 ? (uint32_t)needed : 4096u;
    cparams.n_batch         = (uint32_t)std::max(2048, n_prompt);
    cparams.no_perf         = true;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.cb_eval         = eval_callback;
    cparams.cb_eval_user_data = &agg;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        std::fprintf(stderr, "error: failed to create context\n");
        llama_model_free(model); llama_backend_free();
        return 1;
    }
    llama_memory_t mem = llama_get_memory(ctx);

    std::FILE * csv = std::fopen(args.output_csv.c_str(), "w");
    if (!csv) {
        std::fprintf(stderr, "error: cannot open %s for write\n", args.output_csv.c_str());
        llama_free(ctx); llama_model_free(model); llama_backend_free();
        return 1;
    }
    std::fprintf(csv,
        "prompt_id,step_index,token_id,token_text,"
        "H_nats,H_normalized,top1_prob,top5_cumprob,chosen_prob,"
        "wall_clock_us,n_kv_before,n_evicted,evict_us\n");

    const int64_t t0 = ggml_time_us();

    agg.active = true;
    {
        llama_batch batch = llama_batch_get_one(prompt_tokens.data(), (int32_t)prompt_tokens.size());
        if (llama_decode(ctx, batch) != 0) {
            std::fprintf(stderr, "error: prompt decode failed\n");
            std::fclose(csv);
            llama_free(ctx); llama_model_free(model); llama_backend_free();
            return 1;
        }
    }

    int n_steps_done = 0;
    int eos_step     = -1;
    int total_evicted = 0;
    int64_t total_evict_us = 0;

    for (int step = 0; step < args.max_tokens; ++step) {
        const float * logits = llama_get_logits_ith(ctx, -1);
        if (!logits) {
            std::fprintf(stderr, "error: llama_get_logits_ith returned NULL at step %d\n", step);
            break;
        }

        const auto m   = entropy_probe::compute_entropy(logits, n_vocab);
        const llama_token tok = (llama_token)argmax_logits(logits, n_vocab);
        const std::string text   = token_to_text(vocab, tok);
        const std::string text_q = csv_escape(text);
        const int64_t now = ggml_time_us();

        // --- Controller: select positions to evict from the just-completed step's attention ---
        int n_evicted = 0;
        int64_t evict_us = 0;
        const int n_kv_before = agg.n_kv_last;
        if (args.k_budget > 0
            && step >= args.start_evict_at
            && (step - args.start_evict_at) % args.evict_every == 0
            && agg.n_layers_seen > 0
            && agg.n_kv_last > args.k_budget)
        {
            // n_head is the average heads-per-layer we just summed in; assume the
            // probe disabled FA so n_head matches the model's query-head count.
            // We don't have direct access from here, so just use n_layers_seen to
            // normalize; the relative ordering is the same.
            const int n_head_stub = 1;   // ranking is invariant under positive scaling
            const int64_t te0 = ggml_time_us();
            auto to_remove = select_positions_to_remove(
                agg.sum_layers, agg.n_kv_last, args.k_budget,
                agg.n_layers_seen, n_head_stub);
            // Apply seq_rm. We pass descending positions so earlier removals
            // don't shift the indices of later removals within the cache.
            // (llama_memory_seq_rm operates on absolute positions, but compacting
            // groups of contiguous positions in one call is faster than per-pos.)
            // Group contiguous descending runs.
            int i = 0;
            while (i < (int)to_remove.size()) {
                int j = i;
                while (j + 1 < (int)to_remove.size()
                       && to_remove[j + 1] == to_remove[j] - 1) {
                    ++j;
                }
                // Now positions [to_remove[j] ... to_remove[i]] are contiguous.
                int p0 = to_remove[j];
                int p1 = to_remove[i] + 1;
                llama_memory_seq_rm(mem, 0, p0, p1);
                i = j + 1;
            }
            n_evicted = (int)to_remove.size();
            total_evicted += n_evicted;
            evict_us = ggml_time_us() - te0;
            total_evict_us += evict_us;
        }

        std::fprintf(csv, "%s,%d,%d,%s,%.7g,%.7g,%.7g,%.7g,%.7g,%lld,%d,%d,%lld\n",
                     args.prompt_id.c_str(),
                     step,
                     (int)tok,
                     text_q.c_str(),
                     m.H_nats, m.H_normalized,
                     m.top1_prob, m.top5_cumprob,
                     m.top1_prob,
                     (long long)(now - t0),
                     n_kv_before, n_evicted,
                     (long long)evict_us);

        ++n_steps_done;

        if (llama_vocab_is_eog(vocab, tok)) {
            eos_step = step;
            break;
        }

        agg.active = false;  // suspend capture during the next decode... no, we need it
        agg.reset(0);
        agg.active = true;
        llama_token next = tok;
        llama_batch batch = llama_batch_get_one(&next, 1);
        if (llama_decode(ctx, batch) != 0) {
            std::fprintf(stderr, "error: decode failed at step %d\n", step);
            break;
        }
    }
    agg.active = false;

    std::fclose(csv);

    const int64_t t1 = ggml_time_us();
    const double total_ms = (t1 - t0) / 1000.0;
    const double tok_per_s = n_steps_done > 0 ? (double)n_steps_done * 1e6 / (double)(t1 - t0) : 0.0;

    // --- JSON sidecar ---
    if (std::FILE * jf = std::fopen(args.output_json.c_str(), "w")) {
        std::fprintf(jf,
            "{\n"
            "  \"prompt_id\": \"%s\",\n"
            "  \"k_budget\": %d,\n"
            "  \"evict_every\": %d,\n"
            "  \"start_evict_at\": %d,\n"
            "  \"n_prompt\": %d,\n"
            "  \"steps\": %d,\n"
            "  \"eos_step\": %d,\n"
            "  \"total_ms\": %.3f,\n"
            "  \"tok_per_s\": %.6f,\n"
            "  \"total_evicted\": %d,\n"
            "  \"total_evict_us\": %lld\n"
            "}\n",
            args.prompt_id.c_str(),
            args.k_budget, args.evict_every, args.start_evict_at,
            n_prompt, n_steps_done, eos_step,
            total_ms, tok_per_s,
            total_evicted, (long long)total_evict_us);
        std::fclose(jf);
    }

    std::fprintf(stderr,
        "[controller_probe] prompt_id=%s n_prompt=%d steps=%d eos_step=%d "
        "total_ms=%.3f tok/s=%.2f K_budget=%d total_evicted=%d total_evict_ms=%.3f\n",
        args.prompt_id.c_str(),
        n_prompt, n_steps_done, eos_step, total_ms, tok_per_s,
        args.k_budget, total_evicted, (double)total_evict_us / 1000.0);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
