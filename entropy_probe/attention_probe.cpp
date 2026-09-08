// attention_probe — runs greedy decoding through llama.cpp's public C API,
// captures the per-decode-step attention softmax tensors via the cb_eval hook,
// and writes:
//   * a CSV identical in schema to entropy_probe (per-step entropy / top-k)
//   * a binary sidecar with per-step, per-layer, per-head attention
//     (post-softmax probabilities, per head), enabling per-head eviction
//     policy studies (AdaKV / HeadKV / DuoAttention / AhaKV).
//
// Sidecar binary format v2 (little-endian, host order):
//   bytes [0..3]   : magic "ATNH"   (was "ATTN" in v1; v1 was head-averaged)
//   bytes [4..7]   : uint32 n_steps_written
//   bytes [8..11]  : uint32 n_layers
//   bytes [12..15] : uint32 n_head        (number of QUERY heads per layer)
//   then for each (step, layer) pair, in step-major order:
//     uint32 n_kv
//     n_head * n_kv * float32   (attention from query at this step to source
//                                position i, for each head; row-major over
//                                heads then positions; each head's row sums to 1)
// We write n_steps_written and n_layers as zero up front and patch them
// after the run completes.
//
// Backward compat: if env var ATTNPROBE_AVERAGE_HEADS=1 is set, falls back to
// v1 head-averaged format (magic "ATTN"). Default is v2.

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
    // For the just-completed llama_decode, store attention as
    // per_layer_heads[layer_index] = flat float vector of length n_head * n_kv,
    // laid out as [head0_pos0, head0_pos1, ..., head0_posN-1, head1_pos0, ...].
    // Each head's slice already sums to 1 (post-softmax).
    std::map<int, std::vector<float>> per_layer_heads;

    // 2026-05-24: K/V capture for KeyDiff + LaProx baseline reimplementation.
    // Captures the FULL K and V tensors at the LAST decode step (positions never
    // change once added; final-step capture lets the offline sim slice K[:,:,:s+1]
    // for any earlier step s). per_layer_K[il] is a flat vector in the tensor's
    // natural ne[0..3] layout, stored together with the shape header.
    struct KVCap {
        std::vector<float> data;     // raw fp32 dump
        int64_t ne[4] = {0,0,0,0};   // tensor shape
    };
    std::map<int, KVCap> per_layer_K;
    std::map<int, KVCap> per_layer_V;
    bool     capture_kv = false;     // toggled by env ATTNPROBE_CAPTURE_KV=1

    int      n_head = 0;
    int      n_kv_last = 0;
    bool     active = false;
    int      bytes_warning = 0;       // ggml_backend_tensor_get failure counter

    void reset() {
        per_layer_heads.clear();
        // NOTE: don't clear K/V here — we only want the FINAL step's K/V.
        // Reset is called between decode steps for attention; K/V we keep
        // overwriting with the latest, so the last write wins.
        n_kv_last = 0;
    }
    void reset_full() {
        per_layer_heads.clear();
        per_layer_K.clear();
        per_layer_V.clear();
        n_kv_last = 0;
    }
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
    const bool   is_K    = cap->capture_kv && name && (std::strstr(name, "K_capture") != nullptr);
    const bool   is_V    = cap->capture_kv && name && (std::strstr(name, "V_capture") != nullptr);

    if (ask) {
        // Tell the scheduler: copy this node back to the host so we can read it.
        return is_attn || is_K || is_V;
    }

    // K/V capture path: dump the full tensor into per-layer K or V slot.
    // We OVERWRITE on each call (per step), so after all steps the final state
    // is the one written to disk — exactly what KeyDiff/LaProx need (K vectors
    // for past positions don't change once added).
    if (is_K || is_V) {
        int layer = parse_layer_index(name);
        if (layer < 0) return true;
        // BUG FIX (2026-05-24): old code computed nfloats = nbytes / sizeof(float),
        // which is half the true logical element count for f16 tensors. With
        // ggml_cont() added in llama-graph.cpp (the capture hook now copies the
        // view to a contiguous tensor), ggml_nelements(t) is the correct logical
        // element count and ggml_nbytes(t) is the correct byte count.
        const size_t n_elem = ggml_nelements(t);
        AttnCapture::KVCap kv;
        kv.data.resize(n_elem);
        kv.ne[0] = t->ne[0]; kv.ne[1] = t->ne[1];
        kv.ne[2] = t->ne[2]; kv.ne[3] = t->ne[3];
        if (t->type == GGML_TYPE_F32) {
            ggml_backend_tensor_get(t, kv.data.data(), 0, n_elem * sizeof(float));
        } else if (t->type == GGML_TYPE_F16) {
            // Convert fp16 → fp32 by reading into a temp buffer and casting
            std::vector<uint16_t> tmp(n_elem);
            ggml_backend_tensor_get(t, tmp.data(), 0, n_elem * sizeof(uint16_t));
            for (size_t i = 0; i < n_elem; ++i) {
                kv.data[i] = ggml_fp16_to_fp32(tmp[i]);
            }
        } else {
            return true;  // skip quantized types; KeyDiff/LaProx need full precision
        }
        if (is_K) cap->per_layer_K[layer] = std::move(kv);
        else      cap->per_layer_V[layer] = std::move(kv);
        return true;
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

    int layer = parse_layer_index(name);
    if (layer < 0) layer = (int)cap->per_layer_heads.size();

    // Since 2026-05-24 dual-format probe revision: ALWAYS store per-head in cap.
    // Dual-format AttnSink instances write both ATNH (verbatim) and ATTN (averaged
    // on the fly) outputs, so we no longer need cap.average_heads. Per-head storage
    // is a strict superset of head-averaged.
    std::vector<float> per_layer_buf(static_cast<size_t>(n_head) * n_kv, 0.0f);
    for (int64_t h = 0; h < n_head; ++h) {
        const float * row = buf.data()
                          + s_idx * (n_head * n_tokens * n_kv)
                          + h     * (n_tokens * n_kv)
                          + q_idx * n_kv;
        float * dst = per_layer_buf.data() + h * n_kv;
        std::memcpy(dst, row, sizeof(float) * (size_t)n_kv);
    }
    cap->per_layer_heads[layer] = std::move(per_layer_buf);
    cap->n_head     = (int)n_head;
    cap->n_kv_last  = (int)n_kv;
    return true;
}

// Sidecar writer: we patch the header at the end with the real n_steps / n_layers / n_head.
// v2 ATNH (primary):       per-step per-layer block holds n_head * n_kv floats (per-head).
// v1 ATTN (secondary):     per-step per-layer block holds n_kv floats (head-averaged).
// When v1_format=true and cap stores per-head data, write_step averages on the fly.
// This lets main() open BOTH sinks so every capture produces both formats natively
// for publication-quality provenance (see logs/STRATEGY.md, 2026-05-24 decision).
struct AttnSink {
    std::FILE * fp = nullptr;
    uint32_t    n_steps   = 0;
    uint32_t    n_layers  = 0;
    uint32_t    n_head    = 0;     // for v1: stored as 1; for v2: real head count
    bool        layer_count_locked = false;
    bool        v1_format = false;     // true → magic "ATTN" + head-averaged on write

    bool open(const std::string & path, bool v1) {
        fp = std::fopen(path.c_str(), "wb");
        if (!fp) { std::fprintf(stderr, "error: cannot open %s for write\n", path.c_str()); return false; }
        v1_format = v1;
        const char magic_v1[4] = {'A','T','T','N'};
        const char magic_v2[4] = {'A','T','N','H'};
        std::fwrite(v1 ? magic_v1 : magic_v2, 1, 4, fp);
        uint32_t zero = 0;
        std::fwrite(&zero, sizeof(uint32_t), 1, fp); // n_steps placeholder
        std::fwrite(&zero, sizeof(uint32_t), 1, fp); // n_layers placeholder
        std::fwrite(&zero, sizeof(uint32_t), 1, fp); // n_head placeholder
        return true;
    }

    void write_step(const AttnCapture & cap) {
        if (!fp) return;
        if (cap.per_layer_heads.empty()) return;

        if (!layer_count_locked) {
            n_layers = (uint32_t)cap.per_layer_heads.size();
            n_head   = v1_format ? 1u : (uint32_t)cap.n_head;
            layer_count_locked = true;
        }
        // cap always stores per-head data (n_head * n_kv floats per layer)
        // since 2026-05-24 dual-format probe revision. v1 sink averages at write time.
        const uint32_t real_n_head = (uint32_t)cap.n_head;
        for (uint32_t l = 0; l < n_layers; ++l) {
            auto it = cap.per_layer_heads.find((int)l);
            uint32_t n_kv = 0;
            if (it == cap.per_layer_heads.end()) {
                std::fwrite(&n_kv, sizeof(uint32_t), 1, fp);
                continue;
            }
            const auto & vec = it->second;
            n_kv = real_n_head > 0 ? (uint32_t)(vec.size() / real_n_head) : 0u;
            std::fwrite(&n_kv, sizeof(uint32_t), 1, fp);
            if (v1_format) {
                // Head-average on the fly: mean_h(vec[h*n_kv + i]) for each i.
                std::vector<float> avg(n_kv, 0.0f);
                for (uint32_t h = 0; h < real_n_head; ++h) {
                    const float * src = vec.data() + (size_t)h * n_kv;
                    for (uint32_t i = 0; i < n_kv; ++i) avg[i] += src[i];
                }
                const float inv_h = 1.0f / (float)real_n_head;
                for (uint32_t i = 0; i < n_kv; ++i) avg[i] *= inv_h;
                std::fwrite(avg.data(), sizeof(float), n_kv, fp);
            } else {
                // ATNH: write per-head verbatim
                std::fwrite(vec.data(), sizeof(float), vec.size(), fp);
            }
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

// Derive the v1 (head-averaged) sidecar path from the primary ATNH path.
// "foo/bar/baz.attn.bin"  →  "foo/bar/baz.v1.attn.bin"
// Anything else → append ".v1.attn.bin" verbatim.
std::string derive_v1_path(const std::string & primary) {
    const std::string suffix = ".attn.bin";
    if (primary.size() > suffix.size()
        && primary.compare(primary.size() - suffix.size(), suffix.size(), suffix) == 0) {
        return primary.substr(0, primary.size() - suffix.size()) + ".v1.attn.bin";
    }
    return primary + ".v1.attn.bin";
}

// Derive the K/V sidecar path.
// "foo/bar/baz.attn.bin"  →  "foo/bar/baz.kv.bin"
std::string derive_kv_path(const std::string & primary) {
    const std::string suffix = ".attn.bin";
    if (primary.size() > suffix.size()
        && primary.compare(primary.size() - suffix.size(), suffix.size(), suffix) == 0) {
        return primary.substr(0, primary.size() - suffix.size()) + ".kv.bin";
    }
    return primary + ".kv.bin";
}

// Write the final-step K and V tensors per layer to a single sidecar.
//
// File format ("KVCP" v2, 2026-05-24):
//   bytes [0..3]   : magic "KVCP"
//   bytes [4..7]   : uint32 n_layers (max layer index + 1)
//   then for each layer L in [0..n_layers-1]:
//     uint32 has_K        (0 or 1)
//     if has_K:
//       uint32 K_ne[0..3]  (logical shape; may not match data_count due to
//                          non-contiguous tensor views — use data_count for
//                          read sizing)
//       uint32 data_count  (actual float count written = ggml_nbytes/4)
//       data_count × float32
//     uint32 has_V        (0 or 1)
//     if has_V:
//       uint32 V_ne[0..3]
//       uint32 data_count
//       data_count × float32
bool write_kv_sidecar(const std::string & path, const AttnCapture & cap) {
    FILE * fp = std::fopen(path.c_str(), "wb");
    if (!fp) {
        std::fprintf(stderr, "error: cannot open %s for write\n", path.c_str());
        return false;
    }
    std::fwrite("KVCP", 1, 4, fp);
    // n_layers = (max layer index in either map) + 1
    int max_l = -1;
    for (const auto & kv : cap.per_layer_K) if (kv.first > max_l) max_l = kv.first;
    for (const auto & kv : cap.per_layer_V) if (kv.first > max_l) max_l = kv.first;
    uint32_t n_layers = (uint32_t)(max_l + 1);
    std::fwrite(&n_layers, sizeof(uint32_t), 1, fp);
    for (uint32_t l = 0; l < n_layers; ++l) {
        auto itK = cap.per_layer_K.find((int)l);
        uint32_t has_K = (itK != cap.per_layer_K.end()) ? 1u : 0u;
        std::fwrite(&has_K, sizeof(uint32_t), 1, fp);
        if (has_K) {
            uint32_t ne[4];
            for (int i = 0; i < 4; ++i) ne[i] = (uint32_t)itK->second.ne[i];
            std::fwrite(ne, sizeof(uint32_t), 4, fp);
            uint32_t data_count = (uint32_t)itK->second.data.size();
            std::fwrite(&data_count, sizeof(uint32_t), 1, fp);
            std::fwrite(itK->second.data.data(), sizeof(float),
                        itK->second.data.size(), fp);
        }
        auto itV = cap.per_layer_V.find((int)l);
        uint32_t has_V = (itV != cap.per_layer_V.end()) ? 1u : 0u;
        std::fwrite(&has_V, sizeof(uint32_t), 1, fp);
        if (has_V) {
            uint32_t ne[4];
            for (int i = 0; i < 4; ++i) ne[i] = (uint32_t)itV->second.ne[i];
            std::fwrite(ne, sizeof(uint32_t), 4, fp);
            uint32_t data_count = (uint32_t)itV->second.data.size();
            std::fwrite(&data_count, sizeof(uint32_t), 1, fp);
            std::fwrite(itV->second.data.data(), sizeof(float),
                        itV->second.data.size(), fp);
        }
    }
    std::fclose(fp);
    return true;
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

    AttnCapture cap;
    // 2026-05-24 dual-format revision: cap ALWAYS stores per-head. Output mode
    // controlled by ATTNPROBE_OUTPUT env var:
    //   "dual"  (default) → write BOTH X.attn.bin (ATNH) and X.v1.attn.bin (ATTN)
    //   "atnh"           → write only X.attn.bin (ATNH per-head)
    //   "attn"           → write only X.attn.bin renamed to ATTN (head-avg only)
    // For backward compat: ATTNPROBE_AVERAGE_HEADS=1 → behave as "attn" mode.
    std::string output_mode = "dual";
    {
        const char * om = std::getenv("ATTNPROBE_OUTPUT");
        if (om && *om) output_mode = std::string(om);
        const char * av = std::getenv("ATTNPROBE_AVERAGE_HEADS");
        if (av && av[0] == '1') output_mode = "attn";
    }
    // 2026-05-24: K/V capture for KeyDiff + LaProx reimplementation.
    // Env ATTNPROBE_CAPTURE_KV=1 enables; default off (preserves existing
    // captures' file layout exactly). Adds one .kv.bin sidecar per prompt.
    {
        const char * kv = std::getenv("ATTNPROBE_CAPTURE_KV");
        cap.capture_kv = (kv && kv[0] == '1');
    }

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

    // Open primary (ATNH) and optional secondary (v1 ATTN) sinks.
    AttnSink sink_atnh;     // X.attn.bin    (per-head)
    AttnSink sink_attn;     // X.v1.attn.bin (head-averaged)
    bool want_atnh = (output_mode == "dual" || output_mode == "atnh");
    bool want_attn = (output_mode == "dual" || output_mode == "attn");
    std::string atnh_path = args.output_attn;
    std::string attn_path = (output_mode == "attn") ? args.output_attn
                                                    : derive_v1_path(args.output_attn);
    if (want_atnh && !sink_atnh.open(atnh_path, /*v1=*/false)) {
        std::fclose(csv);
        llama_free(ctx); llama_model_free(model); llama_backend_free();
        return 1;
    }
    if (want_attn && !sink_attn.open(attn_path, /*v1=*/true)) {
        if (want_atnh) sink_atnh.close();
        std::fclose(csv);
        llama_free(ctx); llama_model_free(model); llama_backend_free();
        return 1;
    }
    std::fprintf(stderr, "[attention_probe] output_mode=%s atnh=%s attn=%s\n",
                 output_mode.c_str(),
                 want_atnh ? atnh_path.c_str() : "(off)",
                 want_attn ? attn_path.c_str() : "(off)");

    const int64_t t0 = ggml_time_us();

    // --- Decode the prompt as one batch (cap captures attention from last prompt query) ---
    cap.reset();
    cap.active = true;
    {
        llama_batch batch = llama_batch_get_one(prompt_tokens.data(), (int32_t)prompt_tokens.size());
        if (llama_decode(ctx, batch) != 0) {
            std::fprintf(stderr, "error: prompt decode failed\n");
            if (want_atnh) sink_atnh.close();
            if (want_attn) sink_attn.close();
            std::fclose(csv);
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
        // (= the decode that produced the logits we just consumed). Dual sinks: each
        // writes the format it's configured for; per-head data in cap is the superset.
        if (want_atnh) sink_atnh.write_step(cap);
        if (want_attn) sink_attn.write_step(cap);

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
    if (want_atnh) sink_atnh.close();
    if (want_attn) sink_attn.close();
    if (cap.capture_kv && !cap.per_layer_K.empty()) {
        std::string kv_path = derive_kv_path(args.output_attn);
        if (write_kv_sidecar(kv_path, cap)) {
            std::fprintf(stderr, "[attention_probe] wrote KVCP sidecar to %s (n_layers_K=%zu, n_layers_V=%zu)\n",
                         kv_path.c_str(), cap.per_layer_K.size(), cap.per_layer_V.size());
        }
    }
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
