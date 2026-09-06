#include "warpgate/cpu_exact_backend.hpp"
#include "warpgate/parallel.hpp"
#include <algorithm>
#include <span>
#include <stdexcept>
namespace warpgate {
CpuExactBackend::CpuExactBackend(std::vector<float> dataset,std::size_t rows,std::size_t dim,DistanceMetric metric,std::size_t workers)
    :dataset_(std::move(dataset)),rows_(rows),dim_(dim),metric_(metric),workers_(workers){if(rows_==0||dim_==0||dataset_.size()!=rows_*dim_)throw std::invalid_argument("dataset shape does not match rows * dim");}
std::string_view CpuExactBackend::name() const noexcept{return "cpu-exact";}
std::size_t CpuExactBackend::preferred_max_batch() const noexcept{return 64;}
SearchResult CpuExactBackend::search_one(const SearchQuery&q) const{if(q.vector.size()!=dim_)throw std::invalid_argument("query dimension mismatch");const auto k=std::min(q.top_k,rows_);SearchResult out;out.reserve(rows_);auto query=std::span<const float>(q.vector.data(),q.vector.size());for(std::size_t row=0;row<rows_;++row){auto candidate=std::span<const float>(dataset_.data()+row*dim_,dim_);out.push_back({row,vector_distance(query,candidate,metric_)});}if(k<out.size()){std::nth_element(out.begin(),out.begin()+static_cast<std::ptrdiff_t>(k),out.end(),[](const Neighbor&a,const Neighbor&b){return a.distance<b.distance;});out.resize(k);}std::sort(out.begin(),out.end(),[](const Neighbor&a,const Neighbor&b){return a.distance<b.distance;});return out;}
std::vector<SearchResult> CpuExactBackend::search(std::span<const SearchQuery>q){return parallel_query_map(q,workers_,[this](const SearchQuery&x){return search_one(x);});}
}
