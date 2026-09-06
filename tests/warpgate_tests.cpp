#include "warpgate/backend.hpp"
#include "warpgate/latency_model.hpp"
#include "warpgate/scheduler.hpp"
#include <cassert>
#include <iostream>
#include <memory>
#include <string_view>
class Fake final:public warpgate::SearchBackend{public:Fake(std::string_view n,std::size_t b):n_(n),b_(b){}std::string_view name()const noexcept override{return n_;}std::size_t preferred_max_batch()const noexcept override{return b_;}std::vector<warpgate::SearchResult> search(std::span<const warpgate::SearchQuery>q)override{return std::vector<warpgate::SearchResult>(q.size());}private:std::string_view n_;std::size_t b_;};
int main(){warpgate::LatencyModel m(1000,0.5);m.observe(1,200);m.observe(1,100);assert(m.predict(1)==150);warpgate::AdaptiveScheduler s({8,100,20});auto cpu=std::make_shared<Fake>("cpu",8),gpu=std::make_shared<Fake>("gpu",64);s.add_backend(cpu,300);s.add_backend(gpu,700);s.observe("cpu",8,600);s.observe("gpu",8,250);for(std::size_t i=0;i<8;++i)s.submit({i,{0},1,1000,10000});auto p=s.plan(1100);assert(p.batch_size==8&&std::string_view(p.backend_name)=="gpu");warpgate::AdaptiveScheduler edf({8,100,20});edf.add_backend(cpu,100);edf.submit({1,{0},1,1000,5000});edf.submit({2,{0},1,1000,2000});assert(edf.plan(1100).batch_size==2);std::cout<<"scheduler tests passed\n";}
