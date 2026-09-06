#include "warpgate/distance.hpp"
#include "warpgate/hnsw_backend.hpp"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <span>
#include <string>
#include <vector>
int main(){
    std::vector<float>a{1,2,3,4,5,6,7,8,9},b{1,1,2,3,5,8,13,21,34};
    assert(std::fabs(warpgate::l2_squared_scalar(a,b)-warpgate::l2_squared(a,b))<1e-3F);
    assert(std::fabs(warpgate::cosine_distance_scalar(a,b)-warpgate::cosine_distance(a,b))<1e-4F);
    std::vector<float>data{0,0,1,0,2,0,3,0,4,0,0,5,1,5,2,5,3,5,4,5};
    warpgate::HnswBackend h(data,10,2,{4,32,32,7,warpgate::DistanceMetric::L2,2});
    warpgate::SearchQuery q{1,{2.1F,0.1F},3,1,1000};auto r=h.search(std::span<const warpgate::SearchQuery>(&q,1));assert(r.size()==1&&r[0].size()==3&&r[0][0].index==2);
    const std::string path="warpgate_test.hnsw";h.save(path);auto loaded=warpgate::HnswBackend::load(path,2);auto r2=loaded.search(std::span<const warpgate::SearchQuery>(&q,1));assert(r2.size()==r.size()&&r2[0].size()==r[0].size());for(std::size_t i=0;i<r[0].size();++i){assert(r2[0][i].index==r[0][i].index);assert(std::fabs(r2[0][i].distance-r[0][i].distance)<1e-6F);}std::remove(path.c_str());
    std::vector<float>cosdata{1,0,0,1,-1,0,0,-1};warpgate::HnswBackend hc(cosdata,4,2,{2,16,16,9,warpgate::DistanceMetric::Cosine,2});warpgate::SearchQuery cq{2,{0.9F,0.1F},1,1,1000};auto cr=hc.search(std::span<const warpgate::SearchQuery>(&cq,1));assert(cr[0][0].index==0);
    std::cout<<"hnsw/cosine/persistence tests passed\n";
}
