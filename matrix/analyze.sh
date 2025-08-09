# After building
make ARCH=75
make list_exec         # shows kernels/architectures embedded in matrix_cuda
make dump_exec_elf     # deeper sections/symbols view
make cubin && make list_cubin   # if you want to inspect the .cubin directly

