#include "engine.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {
constexpr int THREADS=256;
void cuda_check(cudaError_t code,const char* label) {
    if(code!=cudaSuccess)throw std::runtime_error(std::string(label)+": "+cudaGetErrorString(code));
}
void blas_check(cublasStatus_t code,const char* label) {
    if(code!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error(std::string(label)+" returned cuBLAS status "+std::to_string(int(code)));
}
struct Pair { float score; int slot; int position; };
__device__ __forceinline__ bool better(const Pair& a,const Pair& b){
    if(a.slot<0)return false;
    if(b.slot<0)return true;
    if(a.score!=b.score)return a.score>b.score;
    return a.slot<b.slot;
}
__device__ __forceinline__ Pair warp_max(Pair p) {
    for(int offset=16;offset>0;offset>>=1){
        Pair other;
        other.score=__shfl_down_sync(0xffffffff,p.score,offset);
        other.slot=__shfl_down_sync(0xffffffff,p.slot,offset);
        other.position=__shfl_down_sync(0xffffffff,p.position,offset);
        if(better(other,p))p=other;
    }
    return p;
}
// Caller allocates scratch[8] and a shared winner, all 256 threads participate.
__device__ __forceinline__ Pair block_max(Pair local,Pair* scratch,Pair* winner){
    int tid=threadIdx.x;
    local=warp_max(local);
    if((tid&31)==0)scratch[tid>>5]=local;
    __syncthreads();
    if(tid<32){
        Pair candidate=tid<8?scratch[tid]:Pair{-INFINITY,-1,-1};
        candidate=warp_max(candidate);
        if(tid==0)*winner=candidate;
    }
    __syncthreads();
    return *winner;
}

__global__ void partial_topk(const float* __restrict__ scores,
                             const std::uint8_t* __restrict__ active,
                             int n,int tiles,int k,
                             float* __restrict__ partial_scores,int* __restrict__ partial_slots) {
    const int tile=blockIdx.x,q=blockIdx.y,tid=threadIdx.x;
    const int slot=tile*THREADS+tid;
    Pair p{slot<n && active[slot]?scores[std::size_t(q)*n+slot]:-INFINITY,
           slot<n && active[slot]?slot:-1,-1};
    __shared__ Pair scratch[8];__shared__ Pair winner;
    const std::size_t base=(std::size_t(q)*tiles+tile)*k;
    for(int round=0;round<k;++round){
        Pair win=block_max(p,scratch,&winner);
        if(tid==0){partial_scores[base+round]=win.score;partial_slots[base+round]=win.slot;}
        if(p.slot==win.slot){p.score=-INFINITY;p.slot=-1;}
        __syncthreads();
    }
}

__global__ void final_topk(float* __restrict__ part_scores,
                           const int* __restrict__ part_slots,int entries,int k,
                           int* __restrict__ out_slots,float* __restrict__ out_scores) {
    const int q=blockIdx.x,tid=threadIdx.x;
    const std::size_t base=std::size_t(q)*entries;
    __shared__ Pair scratch[8];__shared__ Pair winner;
    for(int round=0;round<k;++round){
        Pair best{-INFINITY,-1,-1};
        for(int j=tid;j<entries;j+=THREADS){
            Pair p{part_scores[base+j],part_slots[base+j],j};
            if(better(p,best))best=p;
        }
        Pair win=block_max(best,scratch,&winner);
        if(tid==0){
            out_slots[std::size_t(q)*k+round]=win.slot;
            out_scores[std::size_t(q)*k+round]=win.score;
            if(win.position>=0)part_scores[base+win.position]=-INFINITY;
        }
        __syncthreads();
    }
}

class CudaBackend final : public Backend {
    int capacity_,dim_,max_batch_,max_k_,tiles_;
    float *d_docs_=nullptr,*d_queries_=nullptr,*d_scores_=nullptr,*d_parts_=nullptr,*d_top_scores_=nullptr;
    int *d_indices_=nullptr,*d_top_indices_=nullptr;
    std::uint8_t* d_active_=nullptr;
    cudaStream_t stream_=nullptr;
    cublasHandle_t handle_=nullptr;
    std::string label_;
public:
    CudaBackend(int capacity,int dim,int batch,int k):capacity_(capacity),dim_(dim),max_batch_(batch),max_k_(k),tiles_((capacity+THREADS-1)/THREADS){
        try{
            int device=0;cuda_check(cudaGetDevice(&device),"cudaGetDevice");
            cudaDeviceProp props{};cuda_check(cudaGetDeviceProperties(&props,device),"cudaGetDeviceProperties");
            label_=std::string("CUDA: ")+props.name;
            cuda_check(cudaStreamCreateWithFlags(&stream_,cudaStreamNonBlocking),"cudaStreamCreate");
            blas_check(cublasCreate(&handle_),"cublasCreate");
            blas_check(cublasSetStream(handle_,stream_),"cublasSetStream");
            // SGEMM uses FP32 with standard accumulation, not approximate ANN.
            blas_check(cublasSetMathMode(handle_,CUBLAS_DEFAULT_MATH),"cublasSetMathMode");
            cuda_check(cudaMalloc(&d_docs_,std::size_t(capacity_)*dim_*sizeof(float)),"cudaMalloc docs");
            cuda_check(cudaMalloc(&d_active_,std::size_t(capacity_)),"cudaMalloc active");
            cuda_check(cudaMalloc(&d_queries_,std::size_t(max_batch_)*dim_*sizeof(float)),"cudaMalloc queries");
            cuda_check(cudaMalloc(&d_scores_,std::size_t(max_batch_)*capacity_*sizeof(float)),"cudaMalloc scores");
            cuda_check(cudaMalloc(&d_parts_,std::size_t(max_batch_)*tiles_*max_k_*sizeof(float)),"cudaMalloc partial scores");
            cuda_check(cudaMalloc(&d_indices_,std::size_t(max_batch_)*tiles_*max_k_*sizeof(int)),"cudaMalloc partial ids");
            cuda_check(cudaMalloc(&d_top_scores_,std::size_t(max_batch_)*max_k_*sizeof(float)),"cudaMalloc top scores");
            cuda_check(cudaMalloc(&d_top_indices_,std::size_t(max_batch_)*max_k_*sizeof(int)),"cudaMalloc top ids");
        }catch(...){release();throw;}
    }
    ~CudaBackend()override{release();}
    void release() noexcept {
        if(d_top_indices_)cudaFree(d_top_indices_);
        if(d_top_scores_)cudaFree(d_top_scores_);
        if(d_indices_)cudaFree(d_indices_);
        if(d_parts_)cudaFree(d_parts_);
        if(d_scores_)cudaFree(d_scores_);
        if(d_queries_)cudaFree(d_queries_);
        if(d_active_)cudaFree(d_active_);
        if(d_docs_)cudaFree(d_docs_);
        if(handle_)cublasDestroy(handle_);
        if(stream_)cudaStreamDestroy(stream_);
        d_top_indices_=nullptr;d_top_scores_=nullptr;d_indices_=nullptr;d_parts_=nullptr;
        d_scores_=nullptr;d_queries_=nullptr;d_active_=nullptr;d_docs_=nullptr;handle_=nullptr;stream_=nullptr;
    }
    std::string name() const override{return label_;}
    void initialize(const float* docs,const std::uint8_t* active,int highwater)override{
        if(highwater>0)cuda_check(cudaMemcpyAsync(d_docs_,docs,std::size_t(highwater)*dim_*sizeof(float),cudaMemcpyHostToDevice,stream_),"initial doc upload");
        cuda_check(cudaMemsetAsync(d_active_,0,capacity_,stream_),"active memset");
        if(highwater>0)cuda_check(cudaMemcpyAsync(d_active_,active,highwater,cudaMemcpyHostToDevice,stream_),"initial active upload");
        cuda_check(cudaStreamSynchronize(stream_),"initial GPU sync");
    }
    void update(const float* docs,const std::uint8_t* active,const std::vector<int>& changed,int highwater)override{
        if(!changed.empty()){
            auto sorted=changed;
            std::sort(sorted.begin(),sorted.end());
            sorted.erase(std::unique(sorted.begin(),sorted.end()),sorted.end());
            // Single H2D transfer per contiguous run. Dense bulk imports become large coalesced copies.
            std::size_t i=0;
            while(i<sorted.size()){
                std::size_t j=i+1;
                while(j<sorted.size()&&sorted[j]==sorted[j-1]+1)++j;
                const auto offset=std::size_t(sorted[i])*dim_;
                const auto bytes=(std::size_t(sorted[j-1]-sorted[i]+1)*dim_)*sizeof(float);
                cuda_check(cudaMemcpyAsync(d_docs_+offset,docs+offset,bytes,cudaMemcpyHostToDevice,stream_),"vector update");
                i=j;
            }
        }
        if(highwater>0)cuda_check(cudaMemcpyAsync(d_active_,active,highwater,cudaMemcpyHostToDevice,stream_),"active update");
        cuda_check(cudaStreamSynchronize(stream_),"update GPU sync");
    }
    std::vector<std::vector<Neighbor>> search(const float*,const std::uint8_t*,int n,
          const std::vector<std::vector<float>>& queries,int k)override{
        std::vector<std::vector<Neighbor>> found(queries.size());
        if(n==0||queries.empty())return found;
        const int qcount=static_cast<int>(queries.size());
        if(qcount>max_batch_||k>max_k_)throw std::invalid_argument("GPU query buffer capacity exceeded");
        std::vector<float> host_q(std::size_t(qcount)*dim_);
        for(int q=0;q<qcount;++q)std::copy(queries[q].begin(),queries[q].end(),host_q.begin()+std::size_t(q)*dim_);
        cuda_check(cudaMemcpyAsync(d_queries_,host_q.data(),host_q.size()*sizeof(float),cudaMemcpyHostToDevice,stream_),"query upload");
        float alpha=1.0f,beta=0.0f;
        // Col-major view: documents D x N, queries D x Q; scores = documents^T * queries, N x Q.
        blas_check(cublasSgemm(handle_,CUBLAS_OP_T,CUBLAS_OP_N,n,qcount,dim_,
            &alpha,d_docs_,dim_,d_queries_,dim_,&beta,d_scores_,n),"cublasSgemm");
        const int tiles=(n+THREADS-1)/THREADS;
        partial_topk<<<dim3(tiles,qcount),THREADS,0,stream_>>>(d_scores_,d_active_,n,tiles,k,d_parts_,d_indices_);
        cuda_check(cudaGetLastError(),"partial_topk launch");
        final_topk<<<qcount,THREADS,0,stream_>>>(d_parts_,d_indices_,tiles*k,k,d_top_indices_,d_top_scores_);
        cuda_check(cudaGetLastError(),"final_topk launch");
        std::vector<float> h_scores(std::size_t(qcount)*k);
        std::vector<int> h_slots(std::size_t(qcount)*k);
        cuda_check(cudaMemcpyAsync(h_scores.data(),d_top_scores_,h_scores.size()*sizeof(float),cudaMemcpyDeviceToHost,stream_),"top scores download");
        cuda_check(cudaMemcpyAsync(h_slots.data(),d_top_indices_,h_slots.size()*sizeof(int),cudaMemcpyDeviceToHost,stream_),"top ids download");
        cuda_check(cudaStreamSynchronize(stream_),"search GPU sync");
        for(int q=0;q<qcount;++q)for(int i=0;i<k;++i){
            const int slot=h_slots[std::size_t(q)*k+i];
            if(slot>=0)found[q].push_back({slot,h_scores[std::size_t(q)*k+i]});
        }
        return found;
    }
};
}
std::unique_ptr<Backend> make_backend(int capacity,int dimensions,int max_batch,int max_k){
    return std::make_unique<CudaBackend>(capacity,dimensions,max_batch,max_k);
}
