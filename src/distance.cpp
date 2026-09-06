#include "warpgate/distance.hpp"
#include <cmath>
#include <cstddef>
#include <stdexcept>
#if defined(WARPGATE_AVX2) && defined(__AVX2__)
#include <immintrin.h>
#endif
namespace warpgate {
namespace {
void require_same(std::span<const float> a, std::span<const float> b) {
    if (a.size() != b.size()) throw std::invalid_argument("distance dimension mismatch");
}
}
float l2_squared_scalar(std::span<const float> a, std::span<const float> b) {
    require_same(a,b); float sum=0.0F;
    for(std::size_t i=0;i<a.size();++i){const float d=a[i]-b[i];sum=std::fma(d,d,sum);} return sum;
}
float cosine_distance_scalar(std::span<const float> a, std::span<const float> b) {
    require_same(a,b); float dot=0.0F,aa=0.0F,bb=0.0F;
    for(std::size_t i=0;i<a.size();++i){dot=std::fma(a[i],b[i],dot);aa=std::fma(a[i],a[i],aa);bb=std::fma(b[i],b[i],bb);} 
    if(aa<=0.0F || bb<=0.0F) return 1.0F;
    return 1.0F - dot/(std::sqrt(aa)*std::sqrt(bb));
}
float l2_squared(std::span<const float> a, std::span<const float> b) {
    require_same(a,b);
#if defined(WARPGATE_AVX2) && defined(__AVX2__)
    std::size_t i=0; __m256 acc=_mm256_setzero_ps();
    for(;i+8<=a.size();i+=8){auto va=_mm256_loadu_ps(a.data()+i);auto vb=_mm256_loadu_ps(b.data()+i);auto d=_mm256_sub_ps(va,vb);acc=_mm256_add_ps(acc,_mm256_mul_ps(d,d));}
    alignas(32) float lanes[8]; _mm256_store_ps(lanes,acc); float sum=0.0F; for(float v:lanes)sum+=v;
    for(;i<a.size();++i){const float d=a[i]-b[i];sum=std::fma(d,d,sum);} return sum;
#else
    return l2_squared_scalar(a,b);
#endif
}
float cosine_distance(std::span<const float> a, std::span<const float> b) {
    require_same(a,b);
#if defined(WARPGATE_AVX2) && defined(__AVX2__)
    std::size_t i=0; __m256 dotv=_mm256_setzero_ps(), aav=_mm256_setzero_ps(), bbv=_mm256_setzero_ps();
    for(;i+8<=a.size();i+=8){auto va=_mm256_loadu_ps(a.data()+i);auto vb=_mm256_loadu_ps(b.data()+i);dotv=_mm256_add_ps(dotv,_mm256_mul_ps(va,vb));aav=_mm256_add_ps(aav,_mm256_mul_ps(va,va));bbv=_mm256_add_ps(bbv,_mm256_mul_ps(vb,vb));}
    alignas(32) float d[8],aa8[8],bb8[8]; _mm256_store_ps(d,dotv);_mm256_store_ps(aa8,aav);_mm256_store_ps(bb8,bbv);
    float dot=0,aa=0,bb=0;for(int j=0;j<8;++j){dot+=d[j];aa+=aa8[j];bb+=bb8[j];}
    for(;i<a.size();++i){dot=std::fma(a[i],b[i],dot);aa=std::fma(a[i],a[i],aa);bb=std::fma(b[i],b[i],bb);}
    if (aa <= 0.0F || bb <= 0.0F) return 1.0F;
    return 1.0F - dot / (std::sqrt(aa) * std::sqrt(bb));
#else
    return cosine_distance_scalar(a,b);
#endif
}
float vector_distance(std::span<const float> a, std::span<const float> b, DistanceMetric metric) {
    return metric==DistanceMetric::Cosine ? cosine_distance(a,b) : l2_squared(a,b);
}
std::string_view distance_kernel_name() noexcept {
#if defined(WARPGATE_AVX2) && defined(__AVX2__)
    return "avx2";
#else
    return "scalar";
#endif
}
std::string_view distance_metric_name(DistanceMetric metric) noexcept { return metric==DistanceMetric::Cosine?"cosine":"l2"; }
}
