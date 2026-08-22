# Base + dependency set + build steps below are the exact combination
# proven live on circe GPU2 (Tesla V100-32GB, Volta/sm_70) 2026-08-22 --
# see README.md "Provenance" for the debugging trail (flash-attn dropped:
# requires Ampere+/sm_80; torchvision/pytorchvideo shim; peft missing from
# a static import grep because it's a function-body import upstream).
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git build-essential libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

# torch/torchvision/torchaudio already pinned at 2.4.0+cu121 in the base
# image -- not reinstalling those (pip could otherwise silently swap in a
# non-CUDA wheel).
COPY services/orion-affectgpt-worker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Compat shim: pytorchvideo (unmaintained since ~2022) imports the removed
# public torchvision.transforms.functional_tensor module. This image's
# torchvision 0.19.0 still ships the same code privately as
# _functional_tensor -- confirmed live (has affine==True) -- so re-export it
# under the old public name rather than downgrading torchvision and risking
# a CUDA 12.1 wheel mismatch.
RUN SITE=$(python -c "import torchvision, os; print(os.path.dirname(torchvision.__file__))") \
    && echo "from torchvision.transforms._functional_tensor import *" > "$SITE/transforms/functional_tensor.py"

# Vendored AffectGPT source, pinned to the exact commit this service was
# built and proven against -- treat as an external dependency (like a pip
# package), never hand-edited in this repo's own checkout.
RUN git clone https://github.com/zeroQiaoba/AffectGPT.git /opt/affectgpt-src \
    && cd /opt/affectgpt-src \
    && git checkout fffca794c6792023234565370e3f69f0488aede9

# MER2025OV_Dataset.__init__ unconditionally reads this CSV even for
# single-sample zeroshot inference (upstream bug, worked around here rather
# than patched in the vendored source -- see README). Header-only stub.
RUN mkdir -p /opt/affectgpt-src/dataset/mer2025-dataset \
    && echo "name,english" > /opt/affectgpt-src/dataset/mer2025-dataset/subtitle_chieng.csv

WORKDIR /app
COPY services/orion-affectgpt-worker/app ./app
COPY orion ./orion

EXPOSE 6610
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "6610"]
