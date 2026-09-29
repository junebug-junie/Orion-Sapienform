# Volta (sm_70) build of PrismML's llama.cpp fork for Ternary-Bonsai-2-27B.
#
# Stock llama.cpp rejects PQ2_0/PTQ1_0 (or loads Q2_0 and emits garbage: no
# Hadamard activation runtime). This compiles PrismML-Eng/llama.cpp `prism`
# @ 88c4bc6 with CUDA 12.8 (Prism: 13.3 builds segfault; Circe's host nvcc
# 13.x dropped Volta), then drops the binaries onto orion-llamacpp-host so the
# stock profile wrapper (app.main) still launches it.
#
# Build from repo root (worktree):
#   services/orion-llamacpp-bonsai-host/scripts/build-bonsai-volta.sh

# Global: an ARG used in a later FROM must be declared before the first FROM.
ARG HOST_IMAGE=orion-llamacpp-host:0.1.0
ARG CUDA_DEVEL_IMAGE=nvidia/cuda:12.8.1-devel-ubuntu24.04
FROM ${CUDA_DEVEL_IMAGE} AS build

ARG PRISM_REPO=https://github.com/PrismML-Eng/llama.cpp.git
ARG PRISM_REF=88c4bc60b9c9578f134385be9535e853f2db9b9f

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake build-essential git ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git init . \
    && git remote add origin "${PRISM_REPO}" \
    && git fetch --depth 1 origin "${PRISM_REF}" \
    && git checkout FETCH_HEAD

# Stubs are enough for the link when the daemon has no GPU in the build.
ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs

# FA_ALL_QUANTS: Prism KNOWN_ISSUES -- quantized KV types need it under flash attention.
RUN cmake -B build \
        -DGGML_CUDA=ON \
        -DGGML_NATIVE=OFF \
        -DGGML_CUDA_FA_ALL_QUANTS=ON \
        -DCMAKE_CUDA_ARCHITECTURES=70 \
        -DLLAMA_CURL=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=ON \
        -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --config Release --target llama-server llama-cli -j"$(nproc)" \
    && grep -rlaiq "pq2_0" build/bin \
    && ./build/bin/llama-server --help | grep -E -- "--ctx-checkpoints"

FROM ${HOST_IMAGE}

# Stock wrapper looks for /app/llama-server and sibling .so files.
COPY --from=build /src/build/bin/ /app/
ENV LD_LIBRARY_PATH=/app
