// attention_probe — runs greedy decoding through llama.cpp's public C API,
// captures the per-decode-step attention softmax tensors via the cb_eval hook,
// and writes:
//   * a CSV identical in schema to entropy_probe (per-step entropy / top-k)
//   * a binary sidecar with per-step, per-layer attention summed over heads,
//     intended for slide heatmaps in the style of the H2O paper
//     (Zhang et al., NeurIPS 2023, "Heavy-Hitter Oracle for Efficient
//      Generative Inference of LLMs").
//
// Sidecar binary format (little-endian, host order):
//   bytes [0..3]   : magic "ATTN"
//   bytes [4..7]   : uint32 n_steps_written
//   bytes [8..11]  : uint32 n_layers
//   bytes [12..15] : uint32 n_head
//   then for each (step, layer) pair, in step-major order:
//     uint32 n_kv
//     n_kv * float32 (attention from the query at this step to source token i,
//                     averaged over heads of this layer)
// We write n_steps_written and n_layers as zero up front and patch them
// after the run completes.

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
    std::string output_attn;   // sidecar binary
    int         max_tokens = 0;
    uint32_t    seed       = 42;
    int         n_gpu_layers = 99;
};

void usage(const char * argv0) {
    std::fprintf(stderr,
        "Usage: %s --model PATH --prompt-file PATH --prompt-id ID\n"
        "          --max-tokens N --output PATH --output-attn PATH\n"
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
        else if (!std::strcmp(k, "--output-attn"))   { v = need(i, k); if (!v) return false; a.output_attn   = v; }
        else if (!std::strcmp(k, "--seed"))          { v = need(i, k); if (!v) return false; a.seed          = (uint32_t)std::strtoul(v, nullptr, 10); }
        else if (!std::strcmp(k, "--n-gpu-layers"))  { v = need(i, k); if (!v) return false; a.n_gpu_layers  = std::atoi(v); }
        else if (!std::strcmp(k, "-h") || !std::strcmp(k, "--help")) { usage(argv[0]); std::exit(0); }
        else { std::fprintf(stderr, "unknown arg: %s\n", k); usage(argv[0]); return false; }
    }
    if (a.model.empty() || a.prompt_file.empty() || a.prompt_id.empty()
        || a.output_csv.empty() || a.output_attn.empty() || a.max_tokens <= 0) {
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

std::string token_to_text(const llama_vocab * vocab, llama_token tok) {
    char buf[256];
    int n = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, /*special=*/true);
    if (n >= 0) return std::string(buf, n);
    std::string big(static_cast<size_t>(-n), '\0');
    n = llama_token_to_piece(vocab, tok, big.data(), (int32_t)big.size(), 0, true);
    if (n < 0) return std::string();
    big.resize(n);
    return big;
}

// RFC-4180-style CSV quote: double the inner quote, replace newline/CR with
// literal two-char escapes (cosmetic — keeps each token on one CSV line).
std::string csv_escape(const std::string & s) {
    std::string out; out.reserve(s.size() + 2); out.push_back('"');
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

int argmax_logits(const float * logits, int n_vocab) {
    int best = 0; float best_v = logits[0];
    for (int i = 1; i < n_vocab; ++i) {
        if (logits[i] > best_v) { best_v = logits[i]; best = i; }
    }
    return best;
}

// -------- Attention capture state ---------------------------------------------
//
// One instance lives in main(); a pointer is passed to llama via cparams.cb_eval
// and re-cast inside the callback. The callback fires once per ggml_tensor,
// twice (ask=true, then ask=false=after-compute) only for the tensors we accept.

struct AttnCapture {
    // For the just-completed llama_decode, accumulate per-layer attention to
    // each source position. Layout: per_layer[layer_index] = vector of length n_kv.
    // We use a std::map keyed by layer index so out-of-order evaluation is fine.
    std::map<int, std::vector<float>> per_layer;

    int      n_head = 0;
    int      n_kv_last = 0;
    bool     active = false;
    int      bytes_warning = 0; // ggml_backend_tensor_get failure counter

    void reset() { per_layer.clear(); n_kv_last = 0; }
};

// Parse trailing layer index from a name like "kq_soft_max-3" or "kq_soft_max-(view)".
// Returns -1 if no digit suffix found.
int parse_layer_index(const char * name) {
    if (!name) return -1;
    size_t L = std::strlen(name);
    // walk back to last run of digits
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
    auto * cap = static_cast<AttnCapture *>(user_data);
    if (!cap || !cap->active) return true;

    const char * name = ggml_get_name(t);
    const bool   is_attn = name && (std::strstr(name, "kq_soft_max") != nullptr);

    if (ask) {
        // Tell the scheduler: copy this node back to the host so we can read it.
        return is_attn;
    }

    if (!is_attn) return true;

    // Tensor shape from build_attn_mha: [n_kv, n_tokens, n_head, n_stream].
    const int64_t n_kv     = t->ne[0];
    const int64_t n_tokens = t->ne[1];
    const int64_t n_head   = t->ne[2];
    const int64_t n_stream = t->ne[3];

    if (n_kv <= 0 || n_tokens <= 0 || n_head <= 0 || n_stream <= 0) return true;
    if (t->type != GGML_TYPE_F32) return true;  // attention softmax outputs F32 with FA off

    const size_t nbytes = ggml_nbytes(t);
    std::vector<float> buf(nbytes / sizeof(float));
    ggml_backend_tensor_get(t, buf.data(), 0, nbytes);

    // We want attention from the LAST query in the batch (single-token decode → only query;
    // prompt prefill → the last prompt token, which is the one whose logits we'll read next).
    const int64_t q_idx = n_tokens - 1;
    const int64_t s_idx = 0; // single-stream

    // Layout: index(i, q, h, s) =
    //   s * (n_head * n_tokens * n_kv) + h * (n_tokens * n_kv) + q * n_kv + i
    std::vector<float> per_src(n_kv, 0.0f);
    for (int64_t h = 0; h < n_head; ++h) {
        const float * row = buf.data()
                          + s_idx * (n_head * n_tokens * n_kv)
                          + h     * (n_tokens * n_kv)
                          + q_idx * n_kv;
        for (int64_t i = 0; i < n_kv; ++i) {
            per_src[i] += row[i];
        }
    }
    // Average over heads so the result is itself a probability distribution over sources
    // (each head's softmax sums to 1, so the sum-over-heads divided by n_head sums to 1).
    const float inv_h = 1.0f / (float)n_head;
    for (int64_t i = 0; i < n_kv; ++i) per_src[i] *= inv_h;

    int layer = parse_layer_index(name);
    if (layer < 0) layer = (int)cap->per_layer.size();  // fallback: order-of-arrival index

    cap->per_layer[layer] = std::move(per_src);
    cap->n_head     = (int)n_head;
    cap->n_kv_last  = (int)n_kv;
    return true;
}

// Sidecar writer: we patch the header at the end with the real n_steps / n_layers.
struct AttnSink {
    std::FILE * fp = nullptr;
    uint32_t    n_steps   = 0;
    uint32_t    n_layers  = 0;
    uint32_t    n_head    = 0;
    bool        layer_count_locked = false;

    bool open(const std::string & path) {
        fp = std::fopen(path.c_str(), "wb");
        if (!fp) { std::fprintf(stderr, "error: cannot open %s for write\n", path.c_str()); return false; }
        const char magic[4] = {'A','T','T','N'};
        std::fwrite(magic, 1, 4, fp);
        uint32_t zero = 0;
        std::fwrite(&zero, sizeof(uint32_t), 1, fp); // n_steps placeholder
        std::fwrite(&zero, sizeof(uint32_t), 1, fp); // n_layers placeholder
        std::fwrite(&zero, sizeof(uint32_t), 1, fp); // n_head placeholder
        return true;
    }

    void write_step(const AttnCapture & cap) {
        if (!fp) return;
        if (cap.per_layer.empty()) return;

        if (!layer_count_locked) {
            n_layers = (uint32_t)cap.per_layer.size();
            n_head   = (uint32_t)cap.n_head;
            layer_count_locked = true;
        }
        // For each layer index in sorted order, write n_kv + values.
        for (uint32_t l = 0; l < n_layers; ++l) {
            auto it = cap.per_layer.find((int)l);
            uint32_t n_kv = 0;
            if (it == cap.per_layer.end()) {
                std::fwrite(&n_kv, sizeof(uint32_t), 1, fp);
                continue;
            }
            n_kv = (uint32_t)it->second.size();
            std::fwrite(&n_kv, sizeof(uint32_t), 1, fp);
            std::fwrite(it->second.data(), sizeof(float), n_kv, fp);
        }
        ++n_steps;
    }

    void close() {
        if (!fp) return;
        // Patch header
        std::fseek(fp, 4, SEEK_SET);
        std::fwrite(&n_steps,  sizeof(uint32_t), 1, fp);
        std::fwrite(&n_layers, sizeof(uint32_t), 1, fp);
        std::fwrite(&n_head,   sizeof(uint32_t), 1, fp);
        std::fclose(fp);
        fp = nullptr;
    }
};

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

    AttnCapture cap;

    llama_context_params cparams = llama_context_default_params();
    const int needed = n_prompt + args.max_tokens + 32;
    cparams.n_ctx           = needed > 4096 ? (uint32_t)needed : 4096u;
    cparams.n_batch         = (uint32_t)std::max(2048, n_prompt);
    cparams.no_perf         = true;
    // Disable flash attention so kq_soft_max is a discrete tensor we can intercept.
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.cb_eval         = eval_callback;
    cparams.cb_eval_user_data = &cap;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        std::fprintf(stderr, "error: failed to create context\n");
        llama_model_free(model); llama_backend_free();
        return 1;
    }

    std::FILE * csv = std::fopen(args.output_csv.c_str(), "w");
    if (!csv) {
        std::fprintf(stderr, "error: cannot open %s for write\n", args.output_csv.c_str());
        llama_free(ctx); llama_model_free(model); llama_backend_free();
        return 1;
    }
    std::fprintf(csv,
        "prompt_id,step_index,token_id,token_text,"
        "H_nats,H_normalized,top1_prob,top5_cumprob,chosen_prob,wall_clock_us\n");

    AttnSink sink;
    if (!sink.open(args.output_attn)) {
        std::fclose(csv);
        llama_free(ctx); llama_model_free(model); llama_backend_free();
        return 1;
    }

    const int64_t t0 = ggml_time_us();

    // --- Decode the prompt as one batch (cap captures attention from last prompt query) ---
    cap.reset();
    cap.active = true;
    {
        llama_batch batch = llama_batch_get_one(prompt_tokens.data(), (int32_t)prompt_tokens.size());
        if (llama_decode(ctx, batch) != 0) {
            std::fprintf(stderr, "error: prompt decode failed\n");
            sink.close(); std::fclose(csv);
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

        const auto m   = entropy_probe::compute_entropy(logits, n_vocab);
        const llama_token tok = (llama_token)argmax_logits(logits, n_vocab);
        const std::string text   = token_to_text(vocab, tok);
        const std::string text_q = csv_escape(text);
        const int64_t now = ggml_time_us();

        std::fprintf(csv, "%s,%d,%d,%s,%.7g,%.7g,%.7g,%.7g,%.7g,%lld\n",
                     args.prompt_id.c_str(),
                     step,
                     (int)tok,
                     text_q.c_str(),
                     m.H_nats, m.H_normalized,
                     m.top1_prob, m.top5_cumprob,
                     m.top1_prob,
                     (long long)(now - t0));

        // Persist the attention snapshot that was captured during the most recent decode
        // (= the decode that produced the logits we just consumed).
        sink.write_step(cap);

        ++n_steps_done;

        if (llama_vocab_is_eog(vocab, tok)) {
            eos_step = step;
            break;
        }

        // Decode the chosen token. cap is reset so we capture only this one decode's attention.
        cap.reset();
        llama_token next = tok;
        llama_batch batch = llama_batch_get_one(&next, 1);
        if (llama_decode(ctx, batch) != 0) {
            std::fprintf(stderr, "error: decode failed at step %d\n", step);
            break;
        }
    }

    cap.active = false;
    sink.close();
    std::fclose(csv);

    const int64_t t1 = ggml_time_us();
    std::fprintf(stderr,
        "[attention_probe] prompt_id=%s n_prompt=%d steps=%d eos_step=%d "
        "total_ms=%.3f csv=%s attn=%s\n",
        args.prompt_id.c_str(),
        n_prompt,
        n_steps_done,
        eos_step,
        (t1 - t0) / 1000.0,
        args.output_csv.c_str(),
        args.output_attn.c_str());

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
