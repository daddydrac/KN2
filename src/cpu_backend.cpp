#include "engine.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

// Test-only reference backend. This source is explicitly excluded from CUDA Docker builds.
class CpuBackend final : public Backend {
    int dims_;
public:
    explicit CpuBackend(int dims) : dims_(dims) {}
    std::string name() const override { return "CPU_REFERENCE_TEST_ONLY"; }
    void initialize(const float*, const std::uint8_t*, int) override {}
    void update(const float*, const std::uint8_t*, const std::vector<int>&, int) override {}
    std::vector<std::vector<Neighbor>> search(const float* docs,
        const std::uint8_t* active, int n, const std::vector<std::vector<float>>& queries, int k) override {
        std::vector<std::vector<Neighbor>> all;
        for (const auto& q : queries) {
            std::vector<Neighbor> hits;
            for (int i = 0; i < n; ++i) {
                if (!active[i]) continue;
                double s = 0;
                for (int j = 0; j < dims_; ++j) s += static_cast<double>(q[j]) * docs[static_cast<std::size_t>(i)*dims_+j];
                hits.push_back({i, static_cast<float>(s)});
            }
            auto cmp=[](const Neighbor&a,const Neighbor&b){
                if(a.score != b.score) return a.score > b.score;
                return a.slot < b.slot;
            };
            if (hits.size() > static_cast<std::size_t>(k)) {
                std::partial_sort(hits.begin(),hits.begin()+k,hits.end(),cmp);
                hits.resize(k);
            } else std::sort(hits.begin(),hits.end(),cmp);
            all.push_back(std::move(hits));
        }
        return all;
    }
};
std::unique_ptr<Backend> make_backend(int, int dimensions, int, int) {
    return std::make_unique<CpuBackend>(dimensions);
}
