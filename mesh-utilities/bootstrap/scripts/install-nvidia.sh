#!/usr/bin/env bash
set -euo pipefail

# Driver
DRIVER_PKG="${DRIVER_PACKAGE:-nvidia-driver-550-server}"
apt-get update
apt-get install -y "$DRIVER_PKG" || true

# NVIDIA Container Toolkit with fallbacks
apt-get install -y curl gnupg

install_from_ubuntu_repo() {
  if apt-cache policy nvidia-container-toolkit | grep -q Candidate; then
    apt-get install -y nvidia-container-toolkit && return 0
  fi
  return 1
}

add_generic_repo_list() {
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/libnvidia-container.list     | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#'     | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  apt-get update
}

add_hardcoded_repo_list() {
  echo "deb [arch=amd64 signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://nvidia.github.io/libnvidia-container/stable/ubuntu24.04/amd64/ /"     | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  apt-get update
}

if install_from_ubuntu_repo; then
  :
else
  if add_generic_repo_list && apt-get install -y nvidia-container-toolkit; then
    :
  else
    add_hardcoded_repo_list
    apt-get install -y nvidia-container-toolkit
  fi
fi

# Configure Docker runtime
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

# CUDA Toolkit 12.6
mkdir -p /etc/apt/keyrings
curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/3bf863cc.pub | gpg --dearmor -o /etc/apt/keyrings/cuda-archive-keyring.gpg
echo "deb [signed-by=/etc/apt/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/ /"   | tee /etc/apt/sources.list.d/cuda-ubuntu2404-x86_64.list >/dev/null
apt-get update
apt-get install -y cuda-toolkit-12-6 || apt-get install -y cuda-toolkit

# PATH + LD
echo 'export PATH=/usr/local/cuda/bin:$PATH' > /etc/profile.d/cuda.sh
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> /etc/profile.d/cuda.sh
chmod 644 /etc/profile.d/cuda.sh
