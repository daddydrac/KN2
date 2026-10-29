#pragma once
#include "engine.hpp"
#include <cstdint>
#include <atomic>
#include <future>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <thread>
#include <unordered_map>
#include <vector>
#include <string>

struct VectorInput { std::string id; std::vector<float> vector; std::string metadata = "{}"; };
struct Record { std::string id; std::vector<float> vector; std::string metadata; };
struct SearchInput { std::vector<float> vector; int k = 10; };
struct SearchHit { std::string id; float score; std::string metadata; };
struct BatchReport { int inserted = 0; int updated = 0; std::size_t count = 0; };
struct Stats { std::size_t count, capacity, dimensions, allocated_slots, wal_bytes; std::string backend; };

class Index {
public:
    Index(int capacity, int dimensions, int max_batch, int max_k, int batch_wait_us,
          const std::string& wal_path);
    ~Index();
    Index(const Index&) = delete;
    Index& operator=(const Index&) = delete;

    BatchReport upsert(std::vector<VectorInput> items);
    bool erase(const std::string& id);
    bool get(const std::string& id, Record& out) const;
    std::vector<std::vector<SearchHit>> search(const std::vector<SearchInput>& requests);
    Stats stats() const;
    int dimensions() const { return dims_.load(); }
    int max_k() const { return max_k_; }
    int max_batch() const { return batch_; }
private:
    struct Pending {
        std::vector<SearchInput> queries;
        std::promise<std::vector<std::vector<SearchHit>>> answer;
    };
    struct SlotInfo { std::string id; std::string metadata; };
    int capacity_, batch_, max_k_, batch_wait_us_;
    std::atomic<int> dims_; // zero means infer the dimension from the first insertion
    std::vector<float> docs_;
    std::vector<std::uint8_t> active_;
    std::vector<SlotInfo> slots_;
    std::unordered_map<std::string,int> by_id_;
    std::vector<int> free_;
    int highwater_ = 0;
    std::unique_ptr<Backend> backend_;
    int wal_fd_ = -1;
    std::size_t wal_bytes_ = 0;
    mutable std::mutex data_mutex_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::deque<std::shared_ptr<Pending>> queue_;
    bool stopping_ = false;
    std::thread worker_;

    static void normalize(std::vector<float>& v, int dimensions);
    void open_and_recover(const std::string& path);
    void initialize_storage(int dimensions);
    void write_wal_header();
    void append_wal(const std::vector<VectorInput>& items, bool is_delete);
    void worker_loop();
};
