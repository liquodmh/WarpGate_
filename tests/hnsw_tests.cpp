#include "warpgate/distance.hpp"
#include "warpgate/hnsw_backend.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
int main(){std::vector<float>a{1,2,3,4,5,6,7,8,9},b{1,1,2,3,5,8,13,21,34};assert(std::fabs(warpgate::l2_squared_scalar(a,b)-warpgate::l2_squared(a,b))<1e-4F);std::vector<float>data{0,0,1,0,2,0,3,0,4,0,0,5,1,5,2,5,3,5,4,5};warpgate::HnswBackend h(data,10,2,{4,32,32,7});warpgate::SearchQuery q{1,{2.1F,0.1F},3,1,1000};auto r=h.search(std::span<const warpgate::SearchQuery>(&q,1));assert(r.size()==1&&r[0].size()==3&&r[0][0].index==2);std::cout<<"hnsw tests passed\n";}
