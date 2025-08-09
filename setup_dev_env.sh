#!/usr/bin/env bash
set -e

echo "[*] Updating package lists..."
sudo apt update

echo "[*] Installing core development tools..."
sudo apt install -y \
    build-essential \
    gcc \
    g++ \
    make \
    cmake \
    git \
    pkg-config \
    curl \
    wget \
    unzip \
    zip

echo "[*] Installing ncurses (for terminal UIs)..."
sudo apt install -y \
    libncurses5-dev \
    libncursesw5-dev

echo "[*] Installing OpenCL headers and loader..."
sudo apt install -y \
    ocl-icd-opencl-dev \
    clinfo

echo "[*] Installing math & plotting extras..."
sudo apt install -y \
    libgmp-dev \
    libmpfr-dev \
    libssl-dev

echo "[*] Installing NVIDIA CUDA toolkit..."
sudo apt install -y \
    nvidia-cuda-toolkit

echo "[*] Checking CUDA install..."
nvcc --version || echo "WARNING: nvcc not found in PATH."

echo "[*] Verifying disassembly tools..."
if ! command -v cuobjdump &>/dev/null; then
    echo "WARNING: cuobjdump not found. Ensure CUDA bin dir is in your PATH:"
    echo "    export PATH=/usr/local/cuda/bin:\$PATH"
fi
if ! command -v nvdisasm &>/dev/null; then
    echo "WARNING: nvdisasm not found. Ensure CUDA bin dir is in your PATH."
fi

echo "[*] All set! Recommended CUDA PATH setup:"
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
echo "Run: source ~/.bashrc"

