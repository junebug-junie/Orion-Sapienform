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
# Blobless, NOT --depth 1: llama.cpp's build number is `git rev-list --count HEAD`.
# A shallow clone reports build 1, and the wrapper then treats the binary as
# pre-b5332 and silently drops --flash-attn and --reasoning.
RUN git clone --filter=blob:none --branch prism "${PRISM_REPO}" . \
    && git checkout "${PRISM_REF}"

# Stubs are enough for the link when the daemon has no GPU in the build.
ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs
# ggml-cuda links the driver; stubs ship only libcuda.so, the binary wants .so.1.
RUN ln -sf libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1

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
    && ./build/bin/llama-server --version 2>&1 | tee /tmp/version \
    && test "$(grep -oE '\(build [0-9]+' /tmp/version | grep -oE '[0-9]+')" -gt 5332 \
    && ./build/bin/llama-server --help > /tmp/help \
    && for f in --ctx-checkpoints --jinja --reasoning --reasoning-format --chat-template-kwargs --flash-attn --no-context-shift; do \
         grep -qE -- "(^|[ ,])$f([ ,=]|$)" /tmp/help || { echo "missing $f"; exit 1; }; \
       done

FROM ${HOST_IMAGE}

# Stock wrapper looks for /app/llama-server and sibling .so files.
COPY --from=build /src/build/bin/ /app/
# The base image only supplies the wrapper's Python deps. Its baked wrapper and
# profiles are as old as its last rebuild: on Circe (2026-09-30) they lacked this
# profile (KeyError at boot) and the fork build-number parse (drops --flash-attn).
COPY services/orion-llamacpp-host/app /app/app
COPY config /app/config
COPY orion /app/orion
ENV LD_LIBRARY_PATH=/app
