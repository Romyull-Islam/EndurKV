#pragma once

// Header-only per-step entropy and top-k stats from raw logits.
// Uses log-sum-exp with the max logit subtracted, and no heap allocations.

#include <array>
#include <cmath>
#include <limits>

namespace entropy_probe {

struct EntropyMetrics {
    float H_nats;        // Shannon entropy in nats: H = -sum p_i log p_i
    float H_normalized;  // H_nats / log(n_vocab); in [0, 1]
    float top1_prob;     // max softmax probability
    float top5_cumprob;  // sum of top 5 softmax probabilities
};

// Entropy and top-k stats from n_vocab logits. Pass 1 finds the max and top-5 logits.
// Pass 2 sums exp(l - m) and exp(l - m) * l in float so expf vectorises (libmvec).
// H = log_Z - E[logits], with log_Z = m + log(sum_exp) and E[logits] = sum_qx / sum_exp.
inline EntropyMetrics compute_entropy(const float * __restrict__ logits, int n_vocab) {
    constexpr float NEG_INF = -std::numeric_limits<float>::infinity();

    float m = NEG_INF;
    std::array<float, 5> top5{};
    top5.fill(NEG_INF);

    for (int i = 0; i < n_vocab; ++i) {
        const float l = logits[i];
        if (l > m) m = l;
        if (l > top5[4]) {
            int j = 4;
            while (j > 0 && top5[j - 1] < l) {
                top5[j] = top5[j - 1];
                --j;
            }
            top5[j] = l;
        }
    }

    float fsum_exp = 0.0f;
    float fsum_qx  = 0.0f;
#pragma omp simd reduction(+:fsum_exp, fsum_qx)
    for (int i = 0; i < n_vocab; ++i) {
        const float l = logits[i];
        const float q = std::exp(l - m);  // expf; vectorised by libmvec under -fopenmp-simd
        fsum_exp += q;
        fsum_qx  += q * l;
    }

    const double sum_exp = static_cast<double>(fsum_exp);
    const double sum_qx  = static_cast<double>(fsum_qx);
    const double log_Z   = static_cast<double>(m) + std::log(sum_exp);
    const double H       = log_Z - sum_qx / sum_exp;

    EntropyMetrics out{};
    out.H_nats       = static_cast<float>(H);
    out.H_normalized = static_cast<float>(H / std::log(static_cast<double>(n_vocab)));
    out.top1_prob    = static_cast<float>(std::exp(static_cast<double>(top5[0]) - log_Z));

    double t5 = 0.0;
    for (int i = 0; i < 5; ++i) {
        if (std::isfinite(top5[i])) {
            t5 += std::exp(static_cast<double>(top5[i]) - log_Z);
        }
    }
    out.top5_cumprob = static_cast<float>(t5);

    return out;
}

} // namespace entropy_probe
