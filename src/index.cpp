#include "index.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cerrno>
#include <cctype>
#include <cstdlib>
#include <cstdio>
#include <fcntl.h>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>
#include <limits>
#include <memory>
#include <unordered_set>

namespace {
constexpr char MAGIC[8]={'K','N','N','W','A','L','0','2'};
constexpr std::size_t MAX_RECORD = 16 * 1024 * 1024;
std::uint32_t crc32(const std::uint8_t* p, std::size_t n) {
    static std::uint32_t table[256] = {};
    static const bool init = [](){for(std::uint32_t i=0;i<256;++i){auto c=i;for(int j=0;j<8;++j)c=(c>>1)^(0xedb88320U & (0U-(c&1U)));table[i]=c;}return true;}();
    (void)init;
    std::uint32_t c = ~0U;
    for (std::size_t i=0;i<n;++i) c=(c>>8)^table[(c^p[i])&255U];
    return ~c;
}
void put_u32(std::vector<std::uint8_t>& v, std::uint32_t x) {
    for(int j=0;j<4;++j) v.push_back(static_cast<std::uint8_t>(x>>(8*j)));
}
std::uint32_t get_u32(const std::uint8_t* p) {
    return std::uint32_t(p[0]) | (std::uint32_t(p[1])<<8) | (std::uint32_t(p[2])<<16) | (std::uint32_t(p[3])<<24);
}
void write_all(int fd, const void* ptr, std::size_t n) {
    const auto* p=static_cast<const std::uint8_t*>(ptr);
    while(n){ssize_t k=::write(fd,p,n);if(k<0 && errno==EINTR)continue;if(k<=0)throw std::runtime_error("WAL write failed");p+=k;n-=std::size_t(k);}
}
std::size_t read_all(int fd, void* ptr, std::size_t n) {
    auto* p=static_cast<std::uint8_t*>(ptr);std::size_t done=0;
    while(done<n){ssize_t k=::read(fd,p+done,n-done);if(k<0 && errno==EINTR)continue;if(k<0)throw std::runtime_error("WAL read failed");if(k==0)break;done+=std::size_t(k);}
    return done;
}
}

void Index::normalize(std::vector<float>& v,int dimensions){
    if(static_cast<int>(v.size())!=dimensions)throw std::invalid_argument("incorrect vector dimensions");
    double sum=0;
    for(float x:v){if(!std::isfinite(x))throw std::invalid_argument("vector contains non-finite number");sum+=double(x)*x;}
    if(!std::isfinite(sum)||sum<=1e-60)throw std::invalid_argument("zero or invalid vector norm");
    const double mul=1.0/std::sqrt(sum);
    for(auto& x:v)x=static_cast<float>(double(x)*mul);
}

Index::Index(int capacity,int dimensions,int max_batch,int max_k,int batch_wait_us,const std::string& wal_path)
    :capacity_(capacity),batch_(max_batch),max_k_(max_k),batch_wait_us_(batch_wait_us),dims_(dimensions),
    active_(capacity,0),slots_(capacity) {
    if(capacity<1 || dimensions<0 || max_batch<1 || max_batch>128 || max_k<1 || max_k>32 || batch_wait_us<0 || batch_wait_us>100000)
        throw std::invalid_argument("invalid configuration");
    if(dimensions>0 && (static_cast<std::size_t>(dimensions)*sizeof(float)+13+128+4096>MAX_RECORD))
        throw std::invalid_argument("dimension exceeds WAL per-record limit");
    open_and_recover(wal_path);
    if(dims_>0){
        backend_=make_backend(capacity_,dims_,batch_,max_k_);
        backend_->initialize(docs_.data(),active_.data(),highwater_);
    }
    worker_=std::thread(&Index::worker_loop,this);
}
// First insertion chooses the dimension once per index. GPU allocation is intentionally lazy.
void Index::initialize_storage(int dimensions){
    if(dimensions<=0 || static_cast<std::size_t>(dimensions)*sizeof(float)+13+128+4096>MAX_RECORD)
        throw std::invalid_argument("dimension must be positive and fit in a WAL record");
    const auto n=static_cast<std::size_t>(capacity_)*static_cast<std::size_t>(dimensions);
    if(n>std::numeric_limits<std::size_t>::max()/sizeof(float))
        throw std::invalid_argument("vector storage too large");
    std::vector<float> docs(n,0.0f);
    auto backend=make_backend(capacity_,dimensions,batch_,max_k_);
    backend->initialize(docs.data(),active_.data(),highwater_);
    docs_.swap(docs);
    backend_=std::move(backend);
    dims_=dimensions;
}
void Index::write_wal_header(){
    std::uint8_t hdr[4];const auto d=static_cast<std::uint32_t>(dims_.load());
    for(int j=0;j<4;++j)hdr[j]=static_cast<std::uint8_t>(d>>(8*j));
    // Header must be persisted BEFORE first vector record to make crash replay unambiguous.
    if(::pwrite(wal_fd_,hdr,4,8)!=4 || ::fdatasync(wal_fd_)<0){
        std::fprintf(stderr,"Fatal WAL dimension header write failed\n");
        std::abort();
    }
}
Index::~Index(){
    {std::lock_guard<std::mutex> l(queue_mutex_);stopping_=true;}
    queue_cv_.notify_all();
    if(worker_.joinable())worker_.join();
    if(wal_fd_>=0)::close(wal_fd_);
}

void Index::open_and_recover(const std::string& path) {
    wal_fd_=::open(path.c_str(), O_RDWR|O_CREAT|O_CLOEXEC,0640);
    if(wal_fd_<0)throw std::runtime_error("cannot open WAL (ensure /data is writable)");
    struct stat st{};if(::fstat(wal_fd_,&st)<0)throw std::runtime_error("cannot stat WAL");
    if(st.st_size==0){
        std::vector<std::uint8_t> head(MAGIC,MAGIC+8);
        put_u32(head,static_cast<std::uint32_t>(dims_.load()));put_u32(head,capacity_);
        write_all(wal_fd_,head.data(),head.size());
        if(::fdatasync(wal_fd_)<0)throw std::runtime_error("WAL sync failed");
        wal_bytes_=head.size();
        if(dims_>0)docs_.assign(static_cast<std::size_t>(capacity_)*dims_,0.0f);
        return;
    }
    std::uint8_t head[16];if(read_all(wal_fd_,head,16)!=16 || std::memcmp(head,MAGIC,8)!=0 || get_u32(head+12)!=static_cast<std::uint32_t>(capacity_))
        throw std::runtime_error("WAL format/capacity mismatch; preserve index capacity");
    const auto wal_dim=get_u32(head+8);
    if(wal_dim>static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
       (wal_dim>0 && static_cast<std::size_t>(wal_dim)*sizeof(float)+13+128+4096>MAX_RECORD) ||
       (dims_>0 && wal_dim>0 && wal_dim!=static_cast<std::uint32_t>(dims_)))
        throw std::runtime_error("WAL dimension mismatch/invalid dimensions");
    // An uninitialized WAL may have dimensions=0; the first insert selects them.
    if(wal_dim>0)dims_=static_cast<int>(wal_dim);
    else if(dims_>0 && st.st_size>16)throw std::runtime_error("nonempty uninitialized WAL");
    if(dims_>0)docs_.assign(static_cast<std::size_t>(capacity_)*dims_,0.0f);
    off_t good=16;
    while(true){
        std::uint8_t hdr[8];std::size_t read=read_all(wal_fd_,hdr,sizeof hdr);
        if(read==0)break;
        if(read!=8){if(::ftruncate(wal_fd_,good)<0)throw std::runtime_error("WAL truncate failed");break;}
        const auto len=get_u32(hdr), crc=get_u32(hdr+4);
        if(len<13||len>MAX_RECORD)throw std::runtime_error("WAL record size corruption");
        std::vector<std::uint8_t> payload(len);
        if(read_all(wal_fd_,payload.data(),len)!=len){if(::ftruncate(wal_fd_,good)<0)throw std::runtime_error("WAL truncate failed");break;}
        if(crc32(payload.data(),len)!=crc)throw std::runtime_error("WAL checksum mismatch");
        const std::uint8_t op=payload[0];const std::uint32_t idlen=get_u32(payload.data()+1),metalen=get_u32(payload.data()+5),dim=get_u32(payload.data()+9);
        if(idlen<1||idlen>128||metalen>4096||dim!=(op==1?static_cast<std::uint32_t>(dims_):0U)||
           (op!=1&&op!=2)||len!=13ULL+idlen+metalen+static_cast<std::size_t>(dim)*sizeof(float))throw std::runtime_error("WAL record corrupted");
        std::string id(reinterpret_cast<const char*>(payload.data()+13),idlen);
        std::string meta(reinterpret_cast<const char*>(payload.data()+13+idlen),metalen);
        auto it=by_id_.find(id);
        if(op==2){
            if(it!=by_id_.end()){int s=it->second;active_[s]=0;slots_[s]={};by_id_.erase(it);free_.push_back(s);}
        }else{
            int s;
            if(it!=by_id_.end())s=it->second;
            else if(!free_.empty()){s=free_.back();free_.pop_back();}
            else if(highwater_<capacity_)s=highwater_++;
            else throw std::runtime_error("WAL replay exceeds configured capacity");
            const auto* src=payload.data()+13+idlen+metalen;
            std::memcpy(docs_.data()+static_cast<std::size_t>(s)*dims_,src,static_cast<std::size_t>(dims_)*sizeof(float));
            by_id_[id]=s;slots_[s]={id,meta};active_[s]=1;
        }
        good+=static_cast<off_t>(8+len);
    }
    if(::lseek(wal_fd_,0,SEEK_END)<0)throw std::runtime_error("WAL seek failed");
    wal_bytes_=static_cast<std::size_t>(good);
}

void Index::append_wal(const std::vector<VectorInput>& items, bool is_delete) {
    std::vector<std::uint8_t> payload,group;
    for(const auto& v:items){
        payload.clear();payload.push_back(is_delete?2:1);
        put_u32(payload,static_cast<std::uint32_t>(v.id.size()));
        put_u32(payload,is_delete?0:static_cast<std::uint32_t>(v.metadata.size()));
        put_u32(payload,is_delete?0:static_cast<std::uint32_t>(dims_));
        payload.insert(payload.end(),v.id.begin(),v.id.end());
        if(!is_delete){
            payload.insert(payload.end(),v.metadata.begin(),v.metadata.end());
            const std::uint8_t* p=reinterpret_cast<const std::uint8_t*>(v.vector.data());
            payload.insert(payload.end(),p,p+static_cast<std::size_t>(dims_)*sizeof(float));
        }
        put_u32(group,static_cast<std::uint32_t>(payload.size()));put_u32(group,crc32(payload.data(),payload.size()));
        group.insert(group.end(),payload.begin(),payload.end());
    }
    try {
        write_all(wal_fd_,group.data(),group.size());
        if(::fdatasync(wal_fd_)<0)throw std::runtime_error("WAL fdatasync failed");
        wal_bytes_+=group.size();
    } catch (const std::exception& e) {
        // A partial append cannot safely be followed by another record in this process.
        std::fprintf(stderr,"Fatal WAL write error: %s\n",e.what());
        std::abort();
    }
}

BatchReport Index::upsert(std::vector<VectorInput> items) {
    if(items.empty()||items.size()>1024)throw std::invalid_argument("batch must have 1..1024 vectors");
    std::unordered_set<std::string> seen;
    for(auto& it:items){
        if(it.id.empty()||it.id.size()>128||!std::all_of(it.id.begin(),it.id.end(),[](unsigned char c){return std::isalnum(c)||c=='.'||c=='_'||c==':'||c=='-';}))
            throw std::invalid_argument("id must be 1..128 alphanumeric or . _ : -");
        if(!seen.insert(it.id).second)throw std::invalid_argument("duplicate id inside batch");
        if(it.metadata.size()>4096)throw std::invalid_argument("metadata exceeds 4096 bytes");
    }
    std::lock_guard<std::mutex> lk(data_mutex_);
    const int effective_dims=dims_>0 ? dims_.load() : static_cast<int>(items.front().vector.size());
    if(effective_dims<=0 || static_cast<std::size_t>(effective_dims)*sizeof(float)+13+128+4096>MAX_RECORD)
        throw std::invalid_argument("positive dimensions required; vector exceeds WAL record limit");
    BatchReport report;report.count=by_id_.size();
    for(const auto& it:items)if(by_id_.count(it.id))++report.updated;else ++report.inserted;
    if(by_id_.size()+report.inserted>static_cast<std::size_t>(capacity_))throw std::length_error("index at capacity");
    for(auto& it:items)normalize(it.vector,effective_dims);
    // Initialize the device only after all input/capacity checks have passed.
    if(dims_==0)initialize_storage(effective_dims);
    // Existing empty WALs may still have an uninitialized dimension header.
    std::uint8_t stored_dim[4];
    if(::pread(wal_fd_,stored_dim,4,8)!=4)throw std::runtime_error("cannot read WAL header");
    if(get_u32(stored_dim)==0)write_wal_header();
    // Write-ahead BEFORE modifying memory/GPU. Any device failure after WAL durability is fatal,
    // because restart must replay the journal to recover the known committed state.
    append_wal(items,false);
    std::vector<int> changed;changed.reserve(items.size());
    for(auto& it:items){
        int s;auto found=by_id_.find(it.id);
        if(found!=by_id_.end())s=found->second;
        else if(!free_.empty()){s=free_.back();free_.pop_back();}
        else s=highwater_++;
        std::memcpy(docs_.data()+static_cast<std::size_t>(s)*dims_,it.vector.data(),static_cast<std::size_t>(dims_)*sizeof(float));
        active_[s]=1;slots_[s]={it.id,it.metadata};by_id_[it.id]=s;changed.push_back(s);
    }
    try {backend_->update(docs_.data(),active_.data(),changed,highwater_);}
    catch (const std::exception& e) { std::fprintf(stderr,"Fatal GPU update after WAL commit: %s\n",e.what());std::abort(); }
    report.count=by_id_.size();return report;
}

bool Index::erase(const std::string& id) {
    std::lock_guard<std::mutex> lk(data_mutex_);
    auto it=by_id_.find(id);if(it==by_id_.end())return false;
    append_wal({VectorInput{id,{},""}},true);
    const int s=it->second;by_id_.erase(it);active_[s]=0;slots_[s]={};free_.push_back(s);
    try {backend_->update(docs_.data(),active_.data(),{},highwater_);}
    catch (const std::exception& e) { std::fprintf(stderr,"Fatal GPU delete after WAL commit: %s\n",e.what());std::abort(); }
    return true;
}

bool Index::get(const std::string& id,Record& out)const {
    std::lock_guard<std::mutex> lk(data_mutex_);
    auto it=by_id_.find(id);if(it==by_id_.end())return false;
    int s=it->second;out.id=id;out.metadata=slots_[s].metadata;
    out.vector.assign(docs_.begin()+static_cast<std::size_t>(s)*dims_,docs_.begin()+static_cast<std::size_t>(s+1)*dims_);
    return true;
}

Stats Index::stats()const {
    std::lock_guard<std::mutex> lk(data_mutex_);
    return {by_id_.size(),static_cast<std::size_t>(capacity_),static_cast<std::size_t>(dims_),static_cast<std::size_t>(highwater_),wal_bytes_,backend_?backend_->name():"uninitialized"};
}

std::vector<std::vector<SearchHit>> Index::search(const std::vector<SearchInput>& requests) {
    if(requests.empty()||requests.size()>static_cast<std::size_t>(batch_))throw std::invalid_argument("invalid query batch size");
    auto job=std::make_shared<Pending>();job->queries=requests;
    if(dims_==0)throw std::invalid_argument("index has no dimensions yet: insert a vector first");
    for(auto& q:job->queries){if(q.k<1||q.k>max_k_)throw std::invalid_argument("k out of range");normalize(q.vector,dims_);}
    auto f=job->answer.get_future();
    {
        std::lock_guard<std::mutex> lk(queue_mutex_);
        if(stopping_)throw std::runtime_error("engine stopping");
        if(queue_.size()>=2048)throw std::length_error("search queue saturated");
        queue_.push_back(job);
    }
    queue_cv_.notify_one();return f.get();
}

void Index::worker_loop(){
    for(;;){
        std::vector<std::shared_ptr<Pending>> jobs;
        std::vector<std::vector<float>> queries;
        std::vector<int> ks;
        {
            std::unique_lock<std::mutex> lk(queue_mutex_);
            queue_cv_.wait(lk,[this]{return stopping_||!queue_.empty();});
            if(stopping_&&queue_.empty())return;
            if(batch_wait_us_>0) queue_cv_.wait_for(lk,std::chrono::microseconds(batch_wait_us_),[this]{return stopping_;});
            while(!queue_.empty()){
                auto next=queue_.front();
                if(!jobs.empty()&&queries.size()+next->queries.size()>static_cast<std::size_t>(batch_))break;
                queue_.pop_front();
                for(const auto& q:next->queries){queries.push_back(q.vector);ks.push_back(q.k);}
                jobs.push_back(std::move(next));
                if(queries.size()==static_cast<std::size_t>(batch_))break;
            }
        }
        try{
            std::vector<std::vector<SearchHit>> results;
            {
                std::lock_guard<std::mutex> lk(data_mutex_);
                const int maxk=*std::max_element(ks.begin(),ks.end());
                auto found=by_id_.empty() ? std::vector<std::vector<Neighbor>>(queries.size())
                    : backend_->search(docs_.data(),active_.data(),highwater_,queries,maxk);
                for(std::size_t i=0;i<found.size();++i){
                    std::vector<SearchHit> hits;
                    for(const auto& m:found[i]){
                        if(static_cast<int>(hits.size())>=ks[i])break;
                        if(m.slot<0||m.slot>=highwater_||!active_[m.slot])continue;
                        hits.push_back({slots_[m.slot].id,m.score,slots_[m.slot].metadata});
                    }
                    results.push_back(std::move(hits));
                }
            }
            std::size_t at=0;
            for(auto& job:jobs){
                std::vector<std::vector<SearchHit>> response;
                for(std::size_t i=0;i<job->queries.size();++i)response.push_back(std::move(results.at(at++)));
                job->answer.set_value(std::move(response));
            }
        }catch(...){
            for(auto& job:jobs)job->answer.set_exception(std::current_exception());
        }
    }
}
