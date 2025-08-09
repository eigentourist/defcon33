# Makefile: CPU MLP + CUDA perceptron (and optional CUDA MLP) — same dir

# ---- Toolchain ----
CC        ?= gcc
CXX       ?= g++
NVCC      ?= nvcc

# ---- Flags ----
CFLAGS    ?= -O3 -march=native -pipe -std=c11 -Wall -Wextra -Wpedantic -fno-omit-frame-pointer
CXXFLAGS  ?= -O3 -std=c++17
LDFLAGS   ?=
LDLIBS    ?= -lm

# CUDA arch (RTX 4000 = 75)
ARCH      ?= 75
GENCODE    = -gencode arch=compute_$(ARCH),code=sm_$(ARCH) \
             -gencode arch=compute_$(ARCH),code=compute_$(ARCH)
NVCCFLAGS ?= -O3 -std=c++17 $(GENCODE)

# ---- CPU (pure C) ----
C_SRCS    := $(wildcard *.c)
C_OBJS    := $(C_SRCS:.c=.o)
CPU_TARGET ?= mlp_xor

# ---- CUDA perceptron ----
PERCU_SRCS := perceptron_kernel.cu
PERCU_OBJS := $(PERCU_SRCS:.cu=.o)
PERCP_HOST := perceptron_cuda_host.cpp
PERCP_HOBJ := $(PERCP_HOST:.cpp=.o)
PERCP_TGT  ?= perceptron_cuda

# ---- (Optional) CUDA MLP from earlier step ----
MLP_CUDA_KERNEL := mlp_kernels.cu
MLP_CUDA_KOBJS  := $(MLP_CUDA_KERNEL:.cu=.o)
MLP_CUDA_HOST   := mlp_cuda_host.cpp
MLP_CUDA_HOBJ   := $(MLP_CUDA_HOST:.cpp=.o)
MLP_CUDA_TGT    ?= mlp_cuda

.PHONY: all cpu perceptron mlp_cuda run_cpu run_perceptron debug clean

all: cpu perceptron

# CPU
cpu: $(CPU_TARGET)
$(CPU_TARGET): $(C_OBJS)
	$(CC) $(C_OBJS) -o $@ $(LDFLAGS) $(LDLIBS)
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# CUDA perceptron
perceptron: $(PERCP_TGT)
$(PERCP_TGT): $(PERCP_HOBJ) $(PERCU_OBJS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ -lcudart
%.o: %.cu perceptron_kernel.cuh
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
%.o: %.cpp perceptron_kernel.cuh
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Optional CUDA MLP target (if files exist)
mlp_cuda: $(MLP_CUDA_TGT)
$(MLP_CUDA_TGT): $(MLP_CUDA_HOBJ) $(MLP_CUDA_KOBJS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ -lcudart
$(MLP_CUDA_HOBJ): %.o : %.cpp mlp_kernels.cuh
	$(CXX) $(CXXFLAGS) -c $< -o $@
$(MLP_CUDA_KOBJS): %.o : %.cu mlp_kernels.cuh
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

run_cpu: $(CPU_TARGET)
	./$(CPU_TARGET)

run_perceptron: $(PERCP_TGT)
	./$(PERCP_TGT)

debug: CFLAGS := -O0 -g3 -std=c11 -Wall -Wextra -Wpedantic -fsanitize=address,undefined
debug: LDFLAGS := -fsanitize=address,undefined
debug: CXXFLAGS := -O0 -g3 -std=c++17
debug: NVCCFLAGS := -O0 -g -G -lineinfo -std=c++17 $(GENCODE)
debug: clean all

clean:
	rm -f $(C_OBJS) $(CPU_TARGET) \
	      $(PERCU_OBJS) $(PERCP_HOBJ) $(PERCP_TGT) \
	      $(MLP_CUDA_KOBJS) $(MLP_CUDA_HOBJ) $(MLP_CUDA_TGT)

