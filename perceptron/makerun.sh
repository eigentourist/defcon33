# Build both CPU and CUDA versions (for an RTX 4000 use ARCH=75)
make ARCH=75

# Run CPU version
make run_cpu

# Run CUDA version with each gate
make run_and
make run_or
make run_nand

# If your file names differ:
make CPU_SRC=my_cpu.c CUDA_HOST=my_cuda_host.cpp CUDA_KERN=my_kernel.cu ARCH=75

