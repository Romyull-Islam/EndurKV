// prune_probe — Phase E.2 of the EndurKV measurement study.
//
// Direct test of the safety-gate hypothesis:
//   "Low output entropy at step t  =>  pruning K oldest KV entries at step t
//    causes a small KL divergence in the next-token distribution."
//
// At each decode step we:
//   1. Read the unpruned next-token distribution P_full and its entropy H.
//   2. Save the pre-decode KV state.
//   3. For each K in {16, 64, 256}:
//        - restore the pre-decode state
//        - evict K oldest KV entries via llama_memory_seq_rm
//        - re-decode the same token to recompute logits
//        - read P_pruned, compute KL(P_full || P_pruned)
//   4. Restore to the post-decode state and continue normally.
//
// Output CSV: one row per (step, K) pair —
//   prompt_id, step_index, K, n_kv_before, kept_kv_after,
//   H_nats, top1_full, top1_prob_pruned,
//   kl_nats, kl_top1_lost_prob, ms_step
//
// We DO NOT disable flash attention here (unlike attention_probe). prune_probe
// only needs the next-token distribution, not internal attention tensors, so
// FA stays on for speed.

#include "compute_entropy.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Args {
    std::string model;
    std::string prompt_file;
    std::string prompt_id;
    std::string output_csv;
    int         max_tokens = 0;
    uint32_t    seed       = 42;
    int         n_gpu_layers = 99;
    // Comma-separated list of K values, e.g. "16,64,256".
    std::string ks_csv = "16,64,256";
};

void usage(const char * argv0) {
    std::fprintf(stderr,
        "Usage: %s --model PATH --prompt-file PATH --prompt-id ID\n"
        "          --max-tokens N --output PATH [--seed N] [--n-gpu-layers N]\n"
        "          [--ks 16,64,256]\n",
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
        if      (!std::strcmp(k, "--model"))         { v = need(i, k); if (!v) return false; a.model        = v; }
        else if (!std::strcmp(k, "--prompt-file"))   { v = need(i, k); if (!v) return false; a.prompt_file  = v; }
        else if (!std::strcmp(k, "--prompt-id"))     { v = need(i, k); if (!v) return false; a.prompt_id    = v; }
        else if (!std::strcmp(k, "--max-tokens"))    { v = need(i, k); if (!v) return false; a.max_tokens   = std::atoi(v); }
        else if (!std::strcmp(k, "--output"))        { v = need(i, k); if (!v) return false; a.output_csv   = v; }
        else if (!std::strcmp(k, "--seed"))          { v = need(i, k); if (!v) return false; a.seed         = (uint32_t)std::strtoul(v, nullptr, 10); }
        else if (!std::strcmp(k, "--n-gpu-layers"))  { v = need(i, k); if (!v) return false; a.n_gpu_layers = std::atoi(v); }
        else if (!std::strcmp(k, "--ks"))            { v = need(i, k); if (!v) return false; a.ks_csv       = v; }
        else if (!std::strcmp(k, "-h") || !std::strcmp(k, "--help")) { usage(argv[0]); std::exit(0); }
        else { std::fprintf(stderr, "unknown arg: %s\n", k); usage(argv[0]); return false; }
    }
    if (a.model.empty() || a.prompt_file.empty() || a.prompt_id.empty()
        || a.output_csv.empty() || a.max_tokens <= 0) {
        usage(argv[0]);
        return false;
    }
    return true;
}

bool read_file(const std::string & path, std::string & out) {
    std::ifstream f(path);
    if (!f) { std::fprintf(stderr, "error: cannot open prompt file %s\n", path.c_str()); return false; }
    std::ostringstream ss; ss << f.rdbuf(); out = ss.str();
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) out.pop_back();
    return true;
}

std::vector<int> parse_ks(const std::string & csv) {
    std::vector<int> ks;
    std::string cur;
    for (char c : csv) {
        if (c == ',') { if (!cur.empty()) { ks.push_back(std::atoi(cur.c_str())); cur.clear(); } }
        else if (!std::isspace((unsigned char)c)) { cur.push_back(c); }
    }
    if (!cur.empty()) ks.push_back(std::atoi(cur.c_str()));
    return ks;
}

int argmax_logits(const float * logits, int n_vocab) {
    int best = 0; float best_v = logits[0];
    for (int i = 1; i < n_vocab; ++i) {
        if (logits[i] > best_v) { best_v = logits[i]; best = i; }
    }
    return best;
}

// Compute the full softmax probability of `chosen` from raw logits.
float chosen_prob_from_logits(const float * logits, int n_vocab, int chosen) {
    float m = logits[0];
    for (int i = 1; i < n_vocab; ++i) if (logits[i] > m) m = logits[i];
    double sum_exp = 0.0;
    for (int i = 0; i < n_vocab; ++i) sum_exp += std::exp((double)(logits[i] - m));
    const double log_Z = (double)m + std::log(sum_exp);
    return (float)std::exp((double)logits[chosen] - log_Z);
}

// KL(P_full || P_pruned) in nats.
//   P_full[i]    = exp(logits_full[i] - log_Z_full)
//   P_pruned[i]  = exp(logits_pruned[i] - log_Z_pruned)
//   log_p_full   = logits_full[i] - log_Z_full
//   log_p_prune  = logits_pruned[i] - log_Z_pruned
//   KL = sum P_full[i] * (log_p_full[i] - log_p_prune[i])
double kl_full_vs_pruned(const float * lf, const float * lp, int n_vocab) {
    // log-Z for both, in stable form
    float mf = lf[0], mp = lp[0];
    for (int i = 1; i < n_vocab; ++i) {
        if (lf[i] > mf) mf = lf[i];
        if (lp[i] > mp) mp = lp[i];
    }
    double Zf = 0.0, Zp = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        Zf += std::exp((double)(lf[i] - mf));
        Zp += std::exp((double)(lp[i] - mp));
    }
    const double log_Zf = (double)mf + std::log(Zf);
    const double log_Zp = (double)mp + std::log(Zp);

    double kl = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        const double lpf = (double)lf[i] - log_Zf; // log P_full
        const double pf  = std::exp(lpf);
        if (pf <= 0.0) continue;
        const double lpp = (double)lp[i] - log_Zp; // log P_pruned
        kl += pf * (lpf - lpp);
    }
    if (kl < 0.0) kl = 0.0;  // numerical floor
    return kl;
}

} // namespace

int main(int argc, char ** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) return 1;
    (void)args.seed;

    const std::vector<int> Ks = parse_ks(args.ks_csv);
    if (Ks.empty()) {
        std::fprintf(stderr, "error: --ks must list at least one positive K\n");
        return 1;
    }

    std::string prompt;
    if (!read_file(args.prompt_file, prompt)) return 1;
    if (prompt.empty()) { std::fprintf(stderr, "error: empty prompt\n"); return 1; }

    ggml_backend_load_all();
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = args.n_gpu_layers;
    llama_model * model = llama_model_load_from_file(args.model.c_str(), mparams);
    if (!model) { std::fprintf(stderr, "error: load model %s\n", args.model.c_str()); llama_backend_free(); return 1; }

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

    llama_context_params cparams = llama_context_default_params();
    const int needed = n_prompt + args.max_tokens + 32;
    cparams.n_ctx           = needed > 4096 ? (uint32_t)needed : 4096u;
    cparams.n_batch         = (uint32_t)std::max(2048, n_prompt);
    cparams.no_perf         = true;
    // Flash attention is FINE for prune_probe — we only read final-position logits, never
    // intercept intermediate attention tensors.
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { std::fprintf(stderr, "error: ctx\n"); llama_model_free(model); llama_backend_free(); return 1; }

    std::FILE * csv = std::fopen(args.output_csv.c_str(), "w");
    if (!csv) { std::fprintf(stderr, "error: open %s\n", args.output_csv.c_str()); llama_free(ctx); llama_model_free(model); llama_backend_free(); return 1; }
    std::fprintf(csv,
        "prompt_id,step_index,K,n_kv_before,kept_kv_after,"
        "H_nats,top1_prob_full,top1_prob_pruned,"
        "kl_nats,kl_top1_lost_prob,ms_step\n");

    const int64_t t0 = ggml_time_us();
    llama_memory_t mem = llama_get_memory(ctx);

    // Decode the prompt as one batch.
    {
        llama_batch batch = llama_batch_get_one(prompt_tokens.data(), (int32_t)prompt_tokens.size());
        if (llama_decode(ctx, batch) != 0) {
            std::fprintf(stderr, "error: prompt decode failed\n");
            std::fclose(csv); llama_free(ctx); llama_model_free(model); llama_backend_free();
            return 1;
        }
    }

    // We keep one rolling state snapshot — the state BEFORE the next per-step decode.
    // After the prompt has been decoded, that snapshot is "KV = prompt".
    auto save_state = [&]() -> std::vector<uint8_t> {
        const size_t sz = llama_state_seq_get_size(ctx, 0);
        std::vector<uint8_t> buf(sz);
        const size_t got = llama_state_seq_get_data(ctx, buf.data(), sz, 0);
        buf.resize(got);
        return buf;
    };
    auto restore_state = [&](const std::vector<uint8_t> & buf) {
        llama_state_seq_set_data(ctx, buf.data(), buf.size(), 0);
    };

    std::vector<uint8_t> S_pre = save_state();   // state with KV = [prompt]
    std::vector<float>   logits_pruned_buf;       // scratch

    int n_steps_done = 0;
    int eos_step     = -1;
    llama_token chosen_prev = (llama_token)0;
    bool        have_prev   = false;

    for (int step = 0; step < args.max_tokens; ++step) {
        const int64_t t_step_start = ggml_time_us();

        const float * logits_full = llama_get_logits_ith(ctx, -1);
        if (!logits_full) {
            std::fprintf(stderr, "error: get_logits_ith NULL at step %d\n", step);
            break;
        }

        // Snapshot of the current logits — they will be overwritten by the
        // re-decodes below, so copy them out.
        std::vector<float> logits_full_copy(logits_full, logits_full + n_vocab);

        const auto m = entropy_probe::compute_entropy(logits_full_copy.data(), n_vocab);
        const llama_token tok = (llama_token)argmax_logits(logits_full_copy.data(), n_vocab);
        const float top1_full = m.top1_prob;

        // Current KV size BEFORE any pruning: we need this for the "n_kv_before" column.
        const llama_pos pos_max = llama_memory_seq_pos_max(mem, 0);
        const int n_kv_before = (pos_max < 0) ? 0 : (int)pos_max + 1;

        // === Prune-and-measure block (only for step >= 1; for step 0 the prior decode was the prompt batch) ===
        if (step >= 1 && have_prev) {
            for (int K : Ks) {
                if (K <= 0) continue;
                if (K >= n_kv_before - 1) {
                    // Would empty the cache — skip; no meaningful pruning to measure.
                    std::fprintf(csv,
                        "%s,%d,%d,%d,%d,%.7g,%.7g,%.7g,%.7g,%.7g,%.3f\n",
                        args.prompt_id.c_str(), step, K,
                        n_kv_before, n_kv_before,  // unchanged — skipped prune
                        m.H_nats, top1_full, top1_full,
                        0.0, 0.0,
                        (ggml_time_us() - t_step_start) / 1000.0);
                    continue;
                }
                // Restore to "before the previous decode" state.
                restore_state(S_pre);
                // Evict K oldest tokens.
                llama_memory_seq_rm(mem, 0, 0, K);
                // Re-decode the previously chosen token. This recomputes logits
                // for the same query position but with K oldest gone.
                {
                    llama_token t_prev = chosen_prev;
                    llama_batch batch  = llama_batch_get_one(&t_prev, 1);
                    if (llama_decode(ctx, batch) != 0) {
                        std::fprintf(stderr, "warn: pruned re-decode failed at step=%d K=%d\n", step, K);
                        continue;
                    }
                }
                const float * logits_p = llama_get_logits_ith(ctx, -1);
                if (!logits_p) continue;
                logits_pruned_buf.assign(logits_p, logits_p + n_vocab);

                const double kl  = kl_full_vs_pruned(logits_full_copy.data(),
                                                     logits_pruned_buf.data(), n_vocab);
                const float top1_pr = chosen_prob_from_logits(logits_pruned_buf.data(), n_vocab, tok);
                const float lost    = top1_full - top1_pr;

                const llama_pos pos_max_after = llama_memory_seq_pos_max(mem, 0);
                const int kept_kv_after = (pos_max_after < 0) ? 0 : (int)pos_max_after + 1;

                std::fprintf(csv,
                    "%s,%d,%d,%d,%d,%.7g,%.7g,%.7g,%.7g,%.7g,%.3f\n",
                    args.prompt_id.c_str(), step, K,
                    n_kv_before, kept_kv_after,
                    m.H_nats, top1_full, top1_pr,
                    kl, (lost < 0.0f ? 0.0f : lost),
                    (ggml_time_us() - t_step_start) / 1000.0);
            }
            // After all K measurements, restore S_pre and re-do the unpruned decode,
            // bringing the context back to where it would be without the pruning experiment.
            restore_state(S_pre);
            {
                llama_token t_prev = chosen_prev;
                llama_batch batch  = llama_batch_get_one(&t_prev, 1);
                if (llama_decode(ctx, batch) != 0) {
                    std::fprintf(stderr, "error: clean re-decode failed at step=%d\n", step);
                    break;
                }
            }
        } else if (step == 0) {
            // Step 0 — emit one row per K with kl=0 (placeholder) so the analysis script
            // sees consistent shape.
            for (int K : Ks) {
                std::fprintf(csv,
                    "%s,%d,%d,%d,%d,%.7g,%.7g,%.7g,%.7g,%.7g,%.3f\n",
                    args.prompt_id.c_str(), step, K,
                    n_kv_before, n_kv_before,
                    m.H_nats, top1_full, top1_full,
                    0.0, 0.0,
                    (ggml_time_us() - t_step_start) / 1000.0);
            }
        }

        ++n_steps_done;

        if (llama_vocab_is_eog(vocab, tok)) {
            eos_step = step;
            break;
        }

        // Save the CURRENT state as S_pre for next iteration's prune measurement.
        S_pre = save_state();
        // Decode the chosen token to advance the context.
        {
            llama_token next = tok;
            llama_batch batch = llama_batch_get_one(&next, 1);
            if (llama_decode(ctx, batch) != 0) {
                std::fprintf(stderr, "error: chosen-token decode failed at step %d\n", step);
                break;
            }
        }
        chosen_prev = tok;
        have_prev   = true;
    }

    std::fclose(csv);
    const int64_t t1 = ggml_time_us();

    std::fprintf(stderr,
        "[prune_probe] prompt_id=%s n_prompt=%d steps=%d eos_step=%d "
        "Ks=%s total_ms=%.3f csv=%s\n",
        args.prompt_id.c_str(),
        n_prompt,
        n_steps_done,
        eos_step,
        args.ks_csv.c_str(),
        (t1 - t0) / 1000.0,
        args.output_csv.c_str());

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
