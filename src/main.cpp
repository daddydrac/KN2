#include "index.hpp"
#include <boost/asio.hpp>
#include <boost/asio/thread_pool.hpp>
#include <boost/beast.hpp>
#include <json-c/json.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

namespace net=boost::asio;
namespace beast=boost::beast;
namespace http=beast::http;
using tcp=net::ip::tcp;
using Json=std::unique_ptr<json_object,decltype(&json_object_put)>;

static int env_int(const char* name,int def,int lo,int hi) {
    const char* v=std::getenv(name);if(!v||!*v)return def;
    char* end=nullptr;long x=std::strtol(v,&end,10);
    if(end==v||*end||x<lo||x>hi)throw std::runtime_error(std::string("invalid ")+name);
    return static_cast<int>(x);
}
static const char* env_str(const char* name,const char* def){const char* v=std::getenv(name);return v&&*v?v:def;}
static json_object* field(json_object* obj,const char* key) {
    json_object* out=nullptr;
    if(!obj||!json_object_is_type(obj,json_type_object)||!json_object_object_get_ex(obj,key,&out))throw std::invalid_argument(std::string("missing field: ")+key);
    return out;
}
static std::string get_string(json_object* x,const char* label){
    if(!json_object_is_type(x,json_type_string))throw std::invalid_argument(std::string("invalid ")+label);
    return json_object_get_string(x);
}
static int get_int(json_object* x,const char* label){
    if(!json_object_is_type(x,json_type_int))throw std::invalid_argument(std::string("invalid ")+label);
    std::int64_t v=json_object_get_int64(x);if(v<-2147483647LL||v>2147483647LL)throw std::invalid_argument(std::string("invalid ")+label);
    return static_cast<int>(v);
}
static std::vector<float> get_vector(json_object* x,int dimensions){
    if(!json_object_is_type(x,json_type_array))throw std::invalid_argument("vector must be an array");
    const auto len=json_object_array_length(x);
    if(len<1 || len>static_cast<std::size_t>((16*1024*1024-13-128-4096)/sizeof(float)) ||
       (dimensions>0 && len!=static_cast<std::size_t>(dimensions)))
        throw std::invalid_argument("incorrect vector dimensions or vector exceeds WAL limit");
    std::vector<float> v;v.reserve(len);
    for(std::size_t i=0;i<len;++i){
        auto* y=json_object_array_get_idx(x,i);
        if(!(json_object_is_type(y,json_type_double)||json_object_is_type(y,json_type_int)))throw std::invalid_argument("vector must contain numbers");
        double d=json_object_get_double(y);
        if(!std::isfinite(d)||std::abs(d)>3.4e38)throw std::invalid_argument("vector contains non-finite/out-of-range number");
        v.push_back(static_cast<float>(d));
    }
    return v;
}
static Json parse_json(const std::string& body){
    json_tokener* tok=json_tokener_new();
    if(!tok)throw std::bad_alloc();
    json_object* obj=json_tokener_parse_ex(tok,body.data(),static_cast<int>(body.size()));
    auto err=json_tokener_get_error(tok);
    int end=json_tokener_get_parse_end(tok);
    json_tokener_free(tok);
    if(err!=json_tokener_success){if(obj)json_object_put(obj);throw std::invalid_argument("malformed JSON");}
    for(std::size_t i=end;i<body.size();++i)if(!std::isspace(static_cast<unsigned char>(body[i]))){if(obj)json_object_put(obj);throw std::invalid_argument("trailing JSON data");}
    if(!obj)throw std::invalid_argument("JSON body required");
    return Json(obj,&json_object_put);
}
static VectorInput parse_record(json_object* obj,int dimensions){
    VectorInput rec;
    rec.id=get_string(field(obj,"id"),"id");
    rec.vector=get_vector(field(obj,"vector"),dimensions);
    json_object* md=nullptr;
    if(json_object_object_get_ex(obj,"metadata",&md)){
        if(!json_object_is_type(md,json_type_object))throw std::invalid_argument("metadata must be an object");
        rec.metadata=json_object_to_json_string_ext(md,JSON_C_TO_STRING_PLAIN);
    }
    return rec;
}
static SearchInput parse_query(json_object* obj,int dimensions){
    SearchInput out;
    out.vector=get_vector(field(obj,"vector"),dimensions);
    json_object* k=nullptr;
    if(json_object_object_get_ex(obj,"k",&k))out.k=get_int(k,"k");
    return out;
}
static Json make_json(){return Json(json_object_new_object(),&json_object_put);}
static void set(json_object* obj,const char* key,int x){json_object_object_add(obj,key,json_object_new_int(x));}
static void set(json_object* obj,const char* key,std::int64_t x){json_object_object_add(obj,key,json_object_new_int64(x));}
static void set(json_object* obj,const char* key,const std::string& x){json_object_object_add(obj,key,json_object_new_string_len(x.c_str(),static_cast<int>(x.size())));}
static json_object* metadata(const std::string& str){
    json_object* obj=json_tokener_parse(str.c_str());
    return obj?obj:json_object_new_object();
}
static json_object* hits_to_json(const std::vector<SearchHit>& hits){
    json_object* arr=json_object_new_array_ext(static_cast<int>(hits.size()));
    for(const auto& hit:hits){
        auto* obj=json_object_new_object();set(obj,"id",hit.id);
        json_object_object_add(obj,"score",json_object_new_double(hit.score));
        json_object_object_add(obj,"metadata",metadata(hit.metadata));
        json_object_array_add(arr,obj);
    }
    return arr;
}
static std::string serialize(const Json& j){return json_object_to_json_string_ext(j.get(),JSON_C_TO_STRING_PLAIN);}
static http::response<http::string_body> response(http::status status,const std::string& body,unsigned ver){
    http::response<http::string_body> res{status,ver};
    res.set(http::field::server,"KN2-for-GPU/1.1");res.set(http::field::content_type,"application/json");
    res.keep_alive(false);
    res.body()=body;res.prepare_payload();return res;
}
static http::response<http::string_body> route(const http::request<http::string_body>& req,Index& idx){
    const std::string target(req.target());
    auto reply=[&](http::status status,Json j){return response(status,serialize(j),req.version());};
    try{
        if(req.method()==http::verb::get&&target=="/healthz"){
            auto j=make_json();set(j.get(),"status","ok");return reply(http::status::ok,std::move(j));
        }
        if(req.method()==http::verb::get&&target=="/v1/stats"){
            const auto s=idx.stats();auto j=make_json();
            set(j.get(),"count",std::int64_t(s.count));set(j.get(),"capacity",std::int64_t(s.capacity));
            set(j.get(),"dimensions",std::int64_t(s.dimensions));set(j.get(),"allocated_slots",std::int64_t(s.allocated_slots));
            set(j.get(),"wal_bytes",std::int64_t(s.wal_bytes));set(j.get(),"backend",s.backend);
            set(j.get(),"metric","cosine");return reply(http::status::ok,std::move(j));
        }
        if(target.rfind("/v1/vectors/",0)==0&&target!="/v1/vectors/batch"){
            const std::string id=target.substr(std::string("/v1/vectors/").size());
            if(id.empty()||id.find('/')!=std::string::npos||id.find('?')!=std::string::npos)throw std::invalid_argument("invalid id");
            if(req.method()==http::verb::get){
                Record rec;if(!idx.get(id,rec))return response(http::status::not_found,"{\"error\":\"not found\"}",req.version());
                auto j=make_json();set(j.get(),"id",rec.id);auto* a=json_object_new_array_ext(static_cast<int>(rec.vector.size()));
                for(float x:rec.vector)json_object_array_add(a,json_object_new_double(x));
                json_object_object_add(j.get(),"vector",a);json_object_object_add(j.get(),"metadata",metadata(rec.metadata));
                json_object_object_add(j.get(),"normalized",json_object_new_boolean(1));
                return reply(http::status::ok,std::move(j));
            }
            if(req.method()==http::verb::delete_){
                if(!idx.erase(id))return response(http::status::not_found,"{\"error\":\"not found\"}",req.version());
                auto j=make_json();json_object_object_add(j.get(),"deleted",json_object_new_boolean(1));return reply(http::status::ok,std::move(j));
            }
        }
        if(req.method()==http::verb::post){
            auto body=parse_json(req.body());
            if(target=="/v1/vectors"||target=="/v1/vectors/batch"){
                std::vector<VectorInput> docs;
                if(target=="/v1/vectors")docs.push_back(parse_record(body.get(),idx.dimensions()));
                else{
                    auto* a=field(body.get(),"vectors");
                    if(!json_object_is_type(a,json_type_array)||json_object_array_length(a)<1||json_object_array_length(a)>1024)
                        throw std::invalid_argument("vectors batch must have 1..1024 entries");
                    for(int i=0;i<static_cast<int>(json_object_array_length(a));++i)
                        docs.push_back(parse_record(json_object_array_get_idx(a,i),idx.dimensions()));
                }
                const auto r=idx.upsert(std::move(docs));auto j=make_json();
                set(j.get(),"inserted",r.inserted);set(j.get(),"updated",r.updated);set(j.get(),"count",std::int64_t(r.count));
                return reply(http::status::ok,std::move(j));
            }
            if(target=="/v1/search"||target=="/v1/search/batch"){
                std::vector<SearchInput> queries;
                if(target=="/v1/search")queries.push_back(parse_query(body.get(),idx.dimensions()));
                else{
                    auto* arr=field(body.get(),"queries");
                    if(!json_object_is_type(arr,json_type_array)||json_object_array_length(arr)<1||json_object_array_length(arr)>static_cast<std::size_t>(idx.max_batch()))
                        throw std::invalid_argument("invalid query batch size");
                    for(int i=0;i<static_cast<int>(json_object_array_length(arr));++i)
                        queries.push_back(parse_query(json_object_array_get_idx(arr,i),idx.dimensions()));
                }
                auto found=idx.search(queries);
                if(target=="/v1/search"){
                    auto j=make_json();set(j.get(),"k",queries[0].k);set(j.get(),"count",static_cast<int>(found[0].size()));
                    set(j.get(),"metric","cosine");json_object_object_add(j.get(),"results",hits_to_json(found[0]));
                    return reply(http::status::ok,std::move(j));
                }
                auto j=make_json();auto* arr=json_object_new_array_ext(static_cast<int>(found.size()));
                for(std::size_t i=0;i<found.size();++i){
                    auto* elem=json_object_new_object();set(elem,"k",queries[i].k);set(elem,"count",static_cast<int>(found[i].size()));
                    json_object_object_add(elem,"results",hits_to_json(found[i]));json_object_array_add(arr,elem);
                }
                set(j.get(),"metric","cosine");json_object_object_add(j.get(),"queries",arr);
                return reply(http::status::ok,std::move(j));
            }
        }
        return response(http::status::not_found,"{\"error\":\"route not found\"}",req.version());
    }catch(const std::invalid_argument& e){
        auto j=make_json();set(j.get(),"error",e.what());return reply(http::status::bad_request,std::move(j));
    }catch(const std::length_error& e){
        auto j=make_json();set(j.get(),"error",e.what());return reply(http::status::conflict,std::move(j));
    }catch(const std::exception& e){
        std::cerr<<"Request failed: "<<e.what()<<'\n';
        auto j=make_json();set(j.get(),"error","internal server error");return reply(http::status::internal_server_error,std::move(j));
    }
}
static void session(tcp::socket socket,Index& idx){
    beast::tcp_stream stream(std::move(socket));
    beast::flat_buffer buffer;
    try{
        // Reuse HTTP/1.1 connections for low-overhead search requests. Short idle
        // timeout protects the fixed HTTP worker pool from abandoned clients.
        for(int requests=0;requests<128;++requests){
            stream.expires_after(std::chrono::seconds(requests?2:30));
            http::request_parser<http::string_body> parser;
            parser.body_limit(64*1024*1024);
            beast::error_code ec;
            http::read(stream,buffer,parser,ec);
            if(ec==http::error::end_of_stream||ec==beast::error::timeout)break;
            if(ec)throw beast::system_error(ec);
            auto req=parser.release();
            auto res=route(req,idx);
            const bool keep_alive=req.keep_alive()&&requests<127;
            res.keep_alive(keep_alive);
            stream.expires_after(std::chrono::seconds(30));
            http::write(stream,res);
            if(!keep_alive)break;
        }
    }catch(const std::exception& e){std::cerr<<"HTTP session: "<<e.what()<<'\n';}
    beast::error_code ec;
    stream.socket().shutdown(tcp::socket::shutdown_both,ec);
}
int main(){
    try{
        const int cap=env_int("KNN_CAPACITY",100000,1,1000000);
        const int dim=env_int("KNN_DIMENSIONS",0,0,2147483647);
        const int batch=env_int("KNN_MAX_BATCH",32,1,128);
        const int maxk=env_int("KNN_MAX_K",32,1,32);
        const int wait=env_int("KNN_BATCH_WAIT_US",200,0,100000);
        const int threads=env_int("KNN_HTTP_THREADS",16,1,256);
        const int port=env_int("KNN_PORT",8080,1024,65535);
        Index idx(cap,dim,batch,maxk,wait,env_str("KNN_WAL_PATH","/data/index.wal"));
        std::cout<<"Started "<<idx.stats().backend<<" dims="<<idx.dimensions()<<" capacity="<<cap
                 <<" indexed="<<idx.stats().count<<" batch="<<batch<<" port="<<port<<std::endl;
        net::io_context ioc(1);tcp::acceptor acceptor(ioc,tcp::endpoint(tcp::v4(),static_cast<unsigned short>(port)));
        acceptor.set_option(net::socket_base::reuse_address(true));
        net::thread_pool pool(threads);
        for(;;){
            tcp::socket socket(ioc);acceptor.accept(socket);
            net::post(pool,[s=std::move(socket),&idx]()mutable{session(std::move(s),idx);});
        }
    }catch(const std::exception& e){std::cerr<<"FATAL: "<<e.what()<<std::endl;return 1;}
}
