# Makefile for SIMPLE PERCEPTRON (CPU + CUDA)

# Filenames
CPU_SRC    ?= perceptron_cpu.c
CUDA_HOST  ?= perceptron_cuda_host.cpp
CUDA_KERN  ?= perceptron_kernel.cu

CPU_TARGET ?= perceptron_cpu
CUDA_TARGET?= perceptron_cuda

# Toolchains
CC    ?= gcc
CXX   ?= g++
NVCC  ?= nvcc

# Flags
CFLAGS    ?= -O3 -std=c11 -Wall -Wextra -Wpedantic -fno-omit-frame-pointer
CXXFLAGS  ?= -O3 -std=c++17 -Wall -Wextra -Wpedantic
LDFLAGS   ?=
LDLIBS    ?= -lm

# CUDA arch (default = 75)
ARCH      ?= 75
GENCODE    = -gencode arch=compute_$(ARCH),code=sm_$(ARCH) \
             -gencode arch=compute_$(ARCH),code=compute_$(ARCH)
NVCCFLAGS ?= -O3 -std=c++17 $(GENCODE)

# Objects
CPU_OBJ   := $(CPU_SRC:.c=.o)
CUDA_HOBJ := $(CUDA_HOST:.cpp=.o)
CUDA_KOBJ := $(CUDA_KERN:.cu=.o)

# Detect CPU source
CPU_EXISTS := $(wildcard $(CPU_SRC))

.PHONY: all cpu cuda run_cpu run_and run_or run_nand test debug_cpu debug_cuda clean

# Build both if possible
all: cuda
ifneq ($(CPU_EXISTS),)
all: cpu
endif

# ---- CPU build (only if file exists) ----
ifeq ($(CPU_EXISTS),)
cpu:
	@echo "Note: CPU source '$(CPU_SRC)' not found. Skipping CPU build."
else
cpu: $(CPU_TARGET)
$(CPU_TARGET): $(CPU_OBJ)
	$(CC) $(CPU_OBJ) -o $@ $(LDFLAGS) $(LDLIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@
endif

run_cpu: $(CPU_TARGET)
	./$(CPU_TARGET)

# ---- CUDA build ----
cuda: $(CUDA_TARGET)
$(CUDA_TARGET): $(CUDA_HOBJ) $(CUDA_KOBJ)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Handy run targets
run_and:  $(CUDA_TARGET) ; ./$(CUDA_TARGET) --and
run_or:   $(CUDA_TARGET) ; ./$(CUDA_TARGET) --or
run_nand: $(CUDA_TARGET) ; ./$(CUDA_TARGET) --nand

test: $(CUDA_TARGET)
	@echo "=== AND ===";  ./$(CUDA_TARGET) --and ; \
	 echo "=== OR  ==="; ./$(CUDA_TARGET) --or  ; \
	 echo "=== NAND ===";./$(CUDA_TARGET) --nand

# ---- debug builds ----
debug_cpu: CFLAGS   := -O0 -g3 -std=c11 -Wall -Wextra -Wpedantic -fsanitize=address,undefined
debug_cpu: LDFLAGS  := -fsanitize=address,undefined
debug_cpu: clean cpu

debug_cuda: NVCCFLAGS := -O0 -g -G -lineinfo -std=c++17 $(GENCODE)
debug_cuda: clean cuda

# ---- housekeeping ----
clean:
	rm -f $(CPU_OBJ) $(CUDA_HOBJ) $(CUDA_KOBJ) $(CPU_TARGET) $(CUDA_TARGET)

