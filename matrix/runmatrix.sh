# Build everything (adjust ARCH if needed)
make ARCH=75

# Run each
make run_cpu
make run_cuda
make run_compare   # the CPU-vs-GPU speedup demo

# Clean
make clean

