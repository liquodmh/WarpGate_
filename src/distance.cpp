#include "warpgate/distance.hpp"
#include <cmath>
#include <cstddef>
#include <stdexcept>
#if defined(WARPGATE_AVX2) && defined(__AVX2__)
#include <immintrin.h>
#endif
namespace warpgate {
float l2_squared_scalar(std::span<const float> a, std::span<const float> b) {
    if (a.size() != b.size()) throw std::invalid_argument("distance dimension mismatch");
    float sum = 0.0F;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const float d = a[i] - b[i];
        sum = std::fma(d, d, sum);
    }
    return sum;
}
float l2_squared(std::span<const float> a, std::span<const float> b) {
    if (a.size() != b.size()) throw std::invalid_argument("distance dimension mismatch");
#if defined(WARPGATE_AVX2) && defined(__AVX2__)
    std::size_t i = 0;
    __m256 acc = _mm256_setzero_ps();
    for (; i + 8 <= a.size(); i += 8) {
        const __m256 va = _mm256_loadu_ps(a.data() + i);
        const __m256 vb = _mm256_loadu_ps(b.data() + i);
        const __m256 d = _mm256_sub_ps(va, vb);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(d, d));
    }
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, acc);
    float sum = 0.0F;
    for (float v : lanes) sum += v;
    for (; i < a.size(); ++i) {
        const float d = a[i] - b[i];
        sum = std::fma(d, d, sum);
    }
    return sum;
#else
    return l2_squared_scalar(a, b);
#endif
}
std::string_view distance_kernel_name() noexcept {
#if defined(WARPGATE_AVX2) && defined(__AVX2__)
    return "avx2";
#else
    return "scalar";
#endif
}
}
