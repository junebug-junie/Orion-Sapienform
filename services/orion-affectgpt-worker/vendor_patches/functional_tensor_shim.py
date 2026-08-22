# Compat shim, installed by Dockerfile at build time to
# <torchvision>/transforms/functional_tensor.py.
#
# pytorchvideo (unmaintained since ~2022) imports the removed public
# torchvision.transforms.functional_tensor module. This image's pinned
# torchvision 0.19.0 (matching the base image's own 2.4.0+cu121 pairing)
# still ships the same code privately as _functional_tensor -- confirmed
# live 2026-08-22 (has affine==True) -- so this re-exports it under the old
# public name rather than downgrading torchvision and risking a CUDA 12.1
# wheel mismatch. Tracked here (not a Dockerfile-inline `echo ...py`) so a
# future torchvision bump that breaks this is a reviewable diff, not an
# invisible RUN step.
from torchvision.transforms._functional_tensor import *  # noqa: F401,F403
