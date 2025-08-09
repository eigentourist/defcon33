# Unified Makefile for matrix multiply (CPU + CUDA + compare)

# ---- toolchains ----
CC    ?= gcc
NVCC  ?= nvcc

# ---- flags ----
CFLAGS    ?= -O3 -std=c11 -Wall -Wextra -Wpedantic -fno-omit-frame-pointer
LDFLAGS   ?=
LDLIBS    ?= -lm

# ---- CUDA arch (RTX 4000 = 75; override: make ARCH=86, etc.) ----
ARCH      ?= 75
GEN       = -gencode arch=compute_$(ARCH),code=sm_$(ARCH) \
            -gencode arch=compute_$(ARCH),code=compute_$(ARCH)
NVCCFLAGS ?= -O3 -std=c++17 $(GEN)

# ---- sources/targets ----
CPU_SRC      ?= matrix_mult_cpu.c
CPU_OBJ       = $(CPU_SRC:.c=.o)
CPU_TARGET   ?= matrix_cpu

CUDA_HOST    ?= matrix_mult_cuda_host.cpp
CUDA_KERN    ?= matmul.cu
CUDA_OBJS     = $(CUDA_HOST:.cpp=.o) $(CUDA_KERN:.cu=.o)
CUDA_TARGET  ?= matrix_cuda

COMPARE_SRC  ?= matrix_mult_cuda_compare.cu
COMPARE_OBJ   = $(COMPARE_SRC:.cu=.o)
COMPARE_TGT  ?= matrix_compare

.PHONY: all cpu cuda compare run_cpu run_cuda run_compare debug_cuda clean

# Build everything
all: cpu cuda compare

# ---- CPU build ----
cpu: $(CPU_TARGET)
$(CPU_TARGET): $(CPU_OBJ)
	$(CC) $(CPU_OBJ) -o $@ $(LDFLAGS) $(LDLIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

run_cpu: $(CPU_TARGET)
	./$(CPU_TARGET)

# ---- CUDA (host+kernel) ----
cuda: $(CUDA_TARGET)
$(CUDA_TARGET): $(CUDA_OBJS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

%.o: %.cpp matmul.cuh
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

%.o: %.cu matmul.cuh
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

run_cuda: $(CUDA_TARGET)
	./$(CUDA_TARGET)

# ---- CUDA compare (single .cu TU) ----
compare: $(COMPARE_TGT)
$(COMPARE_TGT): $(COMPARE_OBJ)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

run_compare: $(COMPARE_TGT)
	./$(COMPARE_TGT)

# ---- debug (CUDA) ----
debug_cuda: NVCCFLAGS := -O0 -g -G -lineinfo -std=c++17 $(GEN)
debug_cuda: clean cuda compare

# ---- housekeeping ----
clean:
	rm -f $(CPU_OBJ) $(CUDA_OBJS) $(COMPARE_OBJ) \
	      $(CPU_TARGET) $(CUDA_TARGET) $(COMPARE_TGT)

