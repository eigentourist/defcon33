# Unified Makefile: CPU and CUDA builds for tiny CNN
# Files expected in this directory:
#   - cnn_cpu.c      (CPU-only version)
#   - cnn_cuda.cu    (CUDA forward pass, CPU weight updates)

# ---- toolchains ----
CC        ?= gcc
NVCC      ?= nvcc

# ---- CPU flags ----
CFLAGS    ?= -O3 -march=native -pipe -std=c11 -Wall -Wextra -Wpedantic -fno-omit-frame-pointer
LDFLAGS   ?=
LDLIBS    ?= -lm

# ---- CUDA arch (override at CLI: make ARCH=75) ----
ARCH      ?= 75
GENCODE    = -gencode arch=compute_$(ARCH),code=sm_$(ARCH) \
             -gencode arch=compute_$(ARCH),code=compute_$(ARCH)
NVCCFLAGS ?= -O3 -std=c++17 $(GENCODE)

# ---- sources / targets ----
CPU_SRC    ?= cnn_cpu.c
CPU_OBJ     = $(CPU_SRC:.c=.o)
CPU_TARGET ?= cnn_cpu

CUDA_SRC    ?= cnn_cuda.cu
CUDA_OBJ     = $(CUDA_SRC:.cu=.o)
CUDA_TARGET ?= cnn_cuda

.PHONY: all cpu cuda run_cpu run_cuda debug_cpu debug_cuda clean

# Build both by default
all: cpu cuda

# ----- CPU build -----
cpu: $(CPU_TARGET)
$(CPU_TARGET): $(CPU_OBJ)
	$(CC) $(CPU_OBJ) -o $@ $(LDFLAGS) $(LDLIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

run_cpu: $(CPU_TARGET)
	./$(CPU_TARGET)

debug_cpu: CFLAGS := -O0 -g3 -std=c11 -Wall -Wextra -Wpedantic -fsanitize=address,undefined
debug_cpu: LDFLAGS := -fsanitize=address,undefined
debug_cpu: clean cpu

# ----- CUDA build -----
cuda: $(CUDA_TARGET)
$(CUDA_TARGET): $(CUDA_OBJ)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

run_cuda: $(CUDA_TARGET)
	./$(CUDA_TARGET)

debug_cuda: NVCCFLAGS := -O0 -g -G -lineinfo -std=c++17 $(GENCODE)
debug_cuda: clean cuda

# ----- housekeeping -----
clean:
	rm -f $(CPU_OBJ) $(CUDA_OBJ) $(CPU_TARGET) $(CUDA_TARGET)

