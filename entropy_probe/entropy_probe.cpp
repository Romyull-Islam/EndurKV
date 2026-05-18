// entropy_probe — standalone program that runs greedy decoding through
// llama.cpp's public C API and (when ENABLE_PROBE is defined) emits one CSV
// row per decode step with entropy / top-k statistics.

#include "compute_entropy.h"
#include "llama.h"

#include <algorithm>
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
    std::string output;
    int         max_tokens = 0;
    uint32_t    seed       = 42;   // accepted for harness compatibility; greedy is deterministic
};

void usage(const char * argv0) {
    std::fprintf(stderr,
        "Usage: %s --model PATH --prompt-file PATH --prompt-id ID\n"
        "          --max-tokens N --output PATH [--seed N]\n",
        argv0);
}

bool parse_args(int argc, char ** argv, Args & a) {
    auto need = [&](int & i, const char * name) -> const char * {
        if (i + 1 >= argc) {
            std::fprintf(stderr, "missing value for %s\n", name);
            return nullptr;
        }
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        const char * k = argv[i];
        const char * v = nullptr;
        if      (!std::strcmp(k, "--model"))       { v = need(i, k); if (!v) return false; a.model       = v; }
        else if (!std::strcmp(k, "--prompt-file")) { v = need(i, k); if (!v) return false; a.prompt_file = v; }
        else if (!std::strcmp(k, "--prompt-id"))   { v = need(i, k); if (!v) return false; a.prompt_id   = v; }
        else if (!std::strcmp(k, "--max-tokens"))  { v = need(i, k); if (!v) return false; a.max_tokens  = std::atoi(v); }
        else if (!std::strcmp(k, "--output"))      { v = need(i, k); if (!v) return false; a.output      = v; }
        else if (!std::strcmp(k, "--seed"))        { v = need(i, k); if (!v) return false; a.seed        = (uint32_t)std::strtoul(v, nullptr, 10); }
        else if (!std::strcmp(k, "-h") || !std::strcmp(k, "--help")) { usage(argv[0]); std::exit(0); }
        else { std::fprintf(stderr, "unknown arg: %s\n", k); usage(argv[0]); return false; }
    }
    if (a.model.empty() || a.prompt_file.empty() || a.prompt_id.empty()
        || a.output.empty() || a.max_tokens <= 0) {
        usage(argv[0]);
        return false;
    }
    return true;
}

bool read_file(const std::string & path, std::string & out) {
    std::ifstream f(path);
    if (!f) {
        std::fprintf(stderr, "error: cannot open prompt file %s\n", path.c_str());
        return false;
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) {
        out.pop_back();
    }
    return true;
}

#ifdef ENABLE_PROBE
std::string token_to_text(const llama_vocab * vocab, llama_token tok) {
    char buf[256];
    int n = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, /*special=*/true);
    if (n >= 0) return std::string(buf, n);
    // Buffer too small: -n is the size we actually need.
    std::string big(static_cast<size_t>(-n), '\0');
    n = llama_token_to_piece(vocab, tok, big.data(), (int32_t)big.size(), 0, true);
    if (n < 0) return std::string();
    big.resize(n);
    return big;
}

// CSV-quote in RFC 4180 style:
//   - wrap in double quotes
//   - escape an inner quote by doubling it (the "" form)
//   - replace newline / CR with the literal two-char sequences "\n" / "\r"
//     so each token stays on a single CSV line (purely cosmetic — the
//     pandas reader and the std csv module would also accept embedded
//     newlines inside quotes).
std::string csv_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 2);
    out.push_back('"');
    for (char c : s) {
        if (c == '"') {
            out += "\"\"";
        } else if (c == '\n') {
            out += "\\n";
        } else if (c == '\r') {
            out += "\\r";
        } else {
            out.push_back(c);
        }
    }
    out.push_back('"');
    return out;
}
#endif

int argmax_logits(const float * logits, int n_vocab) {
    int best = 0;
    float best_v = logits[0];
    for (int i = 1; i < n_vocab; ++i) {
        if (logits[i] > best_v) { best_v = logits[i]; best = i; }
    }
    return best;
}

} // namespace

int main(int argc, char ** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) return 1;
    (void)args.seed;  // greedy decode is deterministic; flag preserved for harness use

    std::string prompt;
    if (!read_file(args.prompt_file, prompt)) return 1;
    if (prompt.empty()) {
        std::fprintf(stderr, "error: prompt file %s is empty\n", args.prompt_file.c_str());
        return 1;
    }

    ggml_backend_load_all();
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 99;  // offload everything if a GPU backend is loaded
    llama_model * model = llama_model_load_from_file(args.model.c_str(), mparams);
    if (!model) {
        std::fprintf(stderr, "error: failed to load model %s\n", args.model.c_str());
        llama_backend_free();
        return 1;
    }
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);

    // Tokenize: first call with NULL/0 returns -required_size.
    int n_neg = llama_tokenize(vocab, prompt.c_str(), (int32_t)prompt.size(), nullptr, 0,
                               /*add_special=*/true, /*parse_special=*/true);
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
    cparams.n_ctx   = needed > 4096 ? (uint32_t)needed : 4096u;
    cparams.n_batch = (uint32_t)std::max(2048, n_prompt);
    cparams.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        std::fprintf(stderr, "error: failed to create context\n");
        llama_model_free(model); llama_backend_free();
        return 1;
    }

#ifdef ENABLE_PROBE
    std::FILE * csv = std::fopen(args.output.c_str(), "w");
    if (!csv) {
        std::fprintf(stderr, "error: cannot open %s for write\n", args.output.c_str());
        llama_free(ctx); llama_model_free(model); llama_backend_free();
        return 1;
    }
    std::fprintf(csv,
        "prompt_id,step_index,token_id,token_text,"
        "H_nats,H_normalized,top1_prob,top5_cumprob,chosen_prob,wall_clock_us\n");
#else
    (void)args.output;
#endif

    const int64_t t0 = ggml_time_us();

    {
        llama_batch batch = llama_batch_get_one(prompt_tokens.data(), (int32_t)prompt_tokens.size());
        if (llama_decode(ctx, batch) != 0) {
            std::fprintf(stderr, "error: prompt decode failed\n");
#ifdef ENABLE_PROBE
            std::fclose(csv);
#endif
            llama_free(ctx); llama_model_free(model); llama_backend_free();
            return 1;
        }
    }

    int n_steps_done = 0;
    int eos_step     = -1;

    for (int step = 0; step < args.max_tokens; ++step) {
        const float * logits = llama_get_logits_ith(ctx, -1);
        if (!logits) {
            std::fprintf(stderr, "error: llama_get_logits_ith returned NULL at step %d\n", step);
            break;
        }

#ifdef ENABLE_PROBE
        const auto m = entropy_probe::compute_entropy(logits, n_vocab);
#endif

        const llama_token tok = (llama_token)argmax_logits(logits, n_vocab);

#ifdef ENABLE_PROBE
        const std::string text   = token_to_text(vocab, tok);
        const std::string text_q = csv_escape(text);
        const int64_t     now    = ggml_time_us();
        std::fprintf(csv, "%s,%d,%d,%s,%.7g,%.7g,%.7g,%.7g,%.7g,%lld\n",
                     args.prompt_id.c_str(),
                     step,
                     (int)tok,
                     text_q.c_str(),
                     m.H_nats, m.H_normalized,
                     m.top1_prob, m.top5_cumprob,
                     m.top1_prob,                       // chosen_prob == top1_prob in greedy mode
                     (long long)(now - t0));
#endif

        ++n_steps_done;

        if (llama_vocab_is_eog(vocab, tok)) {
            eos_step = step;
            break;
        }

        llama_token next = tok;
        llama_batch batch = llama_batch_get_one(&next, 1);
        if (llama_decode(ctx, batch) != 0) {
            std::fprintf(stderr, "error: decode failed at step %d\n", step);
            break;
        }
    }

    const int64_t t1       = ggml_time_us();
    const double  total_ms = (t1 - t0) / 1000.0;

    std::fprintf(stderr,
        "[entropy_probe] prompt_id=%s n_prompt=%d steps=%d eos_step=%d "
        "total_ms=%.3f decode_ms_per_step=%.3f\n",
        args.prompt_id.c_str(),
        n_prompt,
        n_steps_done,
        eos_step,
        total_ms,
        n_steps_done > 0 ? total_ms / n_steps_done : 0.0);

#ifdef ENABLE_PROBE
    std::fclose(csv);
#endif
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
