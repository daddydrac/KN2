# C++17/CUDA only; no Go, no Python, no runtime source downloads.
# Pin CUDA to 12.8.1 for RTX 30/40 and Blackwell (sm_120) support.
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS build
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc-12 g++-12 cmake libboost-dev libjson-c-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*
ENV CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 CUDAHOSTCXX=/usr/bin/g++-12
WORKDIR /work
COPY CMakeLists.txt ./
COPY include ./include
COPY src ./src
# Native cubins for common NVIDIA GPUs + PTX fallback. Override as needed.
ARG CUDA_ARCHS=75-real;80-real;86-real;89-real;90-real;120-real;89-virtual
RUN cmake -S . -B /build -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
    -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 \
    -DKNN_CPU_ONLY=OFF && cmake --build /build -j"$(nproc)" \
    && test -s /build/knn_server

FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends libjson-c5 libstdc++6 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --home /data --shell /usr/sbin/nologin knn \
    && mkdir -p /data && chown 10001:10001 /data
COPY --from=build /build/knn_server /usr/local/bin/knn_server
USER 10001:10001
ENV KNN_WAL_PATH=/data/index.wal KNN_PORT=8080
EXPOSE 8080
VOLUME ["/data"]
CMD ["/usr/local/bin/knn_server"]
