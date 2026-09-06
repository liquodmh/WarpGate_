#include "warpgate/cpu_exact_backend.hpp"
#include "warpgate/distance.hpp"
#include "warpgate/hnsw_backend.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <unordered_set>
#include <vector>
namespace {
template<class B> double run(B& b,const std::vector<warpgate::SearchQuery>& q,std::vector<warpgate::SearchResult>& out){auto s=std::chrono::steady_clock::now();out=b.search(q);return q.size()/std::chrono::duration<double>(std::chrono::steady_clock::now()-s).count();}
double recall(const std::vector<warpgate::SearchResult>& x,const std::vector<warpgate::SearchResult>& h){std::size_t hit=0,total=0;for(std::size_t i=0;i<x.size();++i){std::unordered_set<std::size_t> t;for(const auto&n:x[i])t.insert(n.index);total+=x[i].size();for(const auto&n:h[i])if(t.contains(n.index))++hit;}return total?static_cast<double>(hit)/static_cast<double>(total):1.0;}
}
int main(int argc,char**argv){std::size_t rows=20000,dim=128,nq=200,k=10;if(argc>1)rows=std::stoull(argv[1]);if(argc>2)dim=std::stoull(argv[2]);if(argc>3)nq=std::stoull(argv[3]);std::mt19937 rng(42);std::normal_distribution<float>d(0,1);std::vector<float>data(rows*dim);for(auto&v:data)v=d(rng);std::vector<warpgate::SearchQuery>qs;for(std::size_t i=0;i<nq;++i){std::vector<float>q(dim);for(auto&v:q)v=d(rng);qs.push_back({i,std::move(q),k,1,1000000});}auto bs=std::chrono::steady_clock::now();warpgate::HnswBackend h(data,rows,dim,{16,100,64,42});double build=std::chrono::duration<double>(std::chrono::steady_clock::now()-bs).count();warpgate::CpuExactBackend x(std::move(data),rows,dim);std::vector<warpgate::SearchResult>xr,hr;double xq=run(x,qs,xr),hq=run(h,qs,hr);std::cout<<"rows="<<rows<<" dim="<<dim<<" queries="<<nq<<" kernel="<<warpgate::distance_kernel_name()<<" build_s="<<std::fixed<<std::setprecision(3)<<build<<" exact_qps="<<std::setprecision(2)<<xq<<" hnsw_qps="<<hq<<" recall@"<<k<<"="<<std::setprecision(4)<<recall(xr,hr)<<"\n";}
