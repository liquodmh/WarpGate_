#include "warpgate/cpu_exact_backend.hpp"
#include "warpgate/scheduler.hpp"
#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <vector>
namespace { std::uint64_t now_us() { return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count()); } }
int main(int argc, char** argv) {
    std::size_t rows = 50000, dim = 128, queries = 200;
    if (argc > 1) rows = std::stoull(argv[1]); if (argc > 2) dim = std::stoull(argv[2]); if (argc > 3) queries = std::stoull(argv[3]);
    std::mt19937 rng(42); std::uniform_real_distribution<float> dist(-1.0F,1.0F); std::vector<float> data(rows*dim); for(auto&x:data)x=dist(rng);
    auto cpu=std::make_shared<warpgate::CpuExactBackend>(std::move(data),rows,dim); warpgate::AdaptiveScheduler s({16,50,100}); s.add_backend(cpu,5000.0);
    for(std::size_t i=0;i<queries;++i){std::vector<float> q(dim);for(auto&x:q)x=dist(rng);auto t=now_us();s.submit({i,std::move(q),10,t,t+50000});}
    auto start=std::chrono::steady_clock::now();std::size_t done=0;while(s.pending()){done+=s.dispatch(now_us()).size();}
    double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();std::cout<<"scheduler_qps="<<done/sec<<" queries="<<done<<"\n";
}
