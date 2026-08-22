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

# Compat shim -- see vendor_patches/functional_tensor_shim.py for why this
# exists. Tracked file, not an inline `echo ...py` (review finding,
# 2026-08-22: the previous inline form was invisible to normal code
# review/diff and would silently ImportError again on any base-image or
# torchvision bump with no signal until the container failed to start).
COPY services/orion-affectgpt-worker/vendor_patches/functional_tensor_shim.py /tmp/functional_tensor_shim.py
RUN SITE=$(python -c "import torchvision, os; print(os.path.dirname(torchvision.__file__))") \
    && cp /tmp/functional_tensor_shim.py "$SITE/transforms/functional_tensor.py"

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

# Build-time import smoke check -- exercises the exact same deferred-import
# chain as app/model_runtime.py's load() (minus actual model-weight
# loading, which needs the volume-mounted weights this build stage doesn't
# have). Added after review (2026-08-22) found this dependency list had
# already broken twice from a trimmed-without-testing edit (missing
# webdataset, then missing openai/matplotlib) and shipped with no test that
# would catch a repeat -- a bad requirements.txt edit now fails `docker
# build` instead of surfacing only as a runtime crash-loop after `docker up`.
RUN python -c "\
import sys, os; \
sys.path.insert(0, '/opt/affectgpt-src/AffectGPT'); \
os.chdir('/opt/affectgpt-src/AffectGPT'); \
import config; \
import my_affectgpt.tasks; \
import my_affectgpt.models; \
import my_affectgpt.runners; \
import my_affectgpt.processors; \
import my_affectgpt.datasets.builders; \
import my_affectgpt.datasets.builders.image_text_pair_builder; \
from my_affectgpt.common.config import Config; \
from my_affectgpt.common.registry import registry; \
from my_affectgpt.conversation.conversation_video import Chat; \
from my_affectgpt.datasets.datasets.mer2025ov_dataset import MER2025OV_Dataset; \
from my_affectgpt.processors.base_processor import BaseProcessor; \
print('AFFECTGPT_IMPORT_CHAIN_OK')"

EXPOSE 6610
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "6610"]
