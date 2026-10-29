#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct Neighbor { int slot; float score; };

// An API boundary only; Docker always builds cuda_backend.cu, never cpu_backend.cpp.
class Backend {
public:
    virtual ~Backend() = default;
    virtual std::string name() const = 0;
    virtual void initialize(const float* docs, const std::uint8_t* active, int highwater) = 0;
    virtual void update(const float* docs, const std::uint8_t* active,
                        const std::vector<int>& changed_slots, int highwater) = 0;
    virtual std::vector<std::vector<Neighbor>> search(
        const float* host_docs, const std::uint8_t* active, int highwater,
        const std::vector<std::vector<float>>& queries, int k) = 0;
};

std::unique_ptr<Backend> make_backend(int capacity, int dimensions, int max_batch, int max_k);
