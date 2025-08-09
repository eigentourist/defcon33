# Unified Makefile for matrix multiply (CPU + CUDA + compare + disassembly)

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

.PHONY: all cpu cuda compare run_cpu run_cuda run_compare debug_cuda clean \
        ptx cubin sass disasm dump_exec_ptx dump_exec_sass

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

# ===================== Disassembly / Inspection =====================
# Produce PTX/SASS from the kernel source and from the built executable.

# PTX directly from the kernel source
PTX_FILE   := $(basename $(CUDA_KERN)).ptx
ptx: $(PTX_FILE)
$(PTX_FILE): $(CUDA_KERN) matmul.cuh
	$(NVCC) $(NVCCFLAGS) -ptx $(CUDA_KERN) -o $(PTX_FILE)
	@echo "Wrote PTX -> $(PTX_FILE)"

# CUBIN + SASS (nvdisasm) from the kernel source
CUBIN_FILE := $(basename $(CUDA_KERN)).cubin
SASS_FILE  := $(basename $(CUDA_KERN)).sass

cubin: $(CUBIN_FILE)
$(CUBIN_FILE): $(CUDA_KERN) matmul.cuh
	$(NVCC) $(NVCCFLAGS) -cubin $(CUDA_KERN) -o $(CUBIN_FILE)
	@echo "Wrote cubin -> $(CUBIN_FILE)"

sass: $(SASS_FILE)
$(SASS_FILE): $(CUBIN_FILE)
	@which nvdisasm >/dev/null 2>&1 || (echo "nvdisasm not found. Install CUDA toolkit."; exit 1)
	nvdisasm $(CUBIN_FILE) > $(SASS_FILE)
	@echo "Wrote SASS -> $(SASS_FILE)"

# Convenience: do both PTX + SASS
disasm: ptx sass
	@echo "PTX:  $(PTX_FILE)"
	@echo "SASS: $(SASS_FILE)"

# Dump PTX/SASS from the built executable (fatbin) using cuobjdump
dump_exec_ptx: $(CUDA_TARGET)
	@which cuobjdump >/dev/null 2>&1 || (echo "cuobjdump not found. Install CUDA toolkit."; exit 1)
	cuobjdump --dump-ptx $(CUDA_TARGET) > $(CUDA_TARGET).ptx
	@echo "Wrote exec PTX -> $(CUDA_TARGET).ptx"

dump_exec_sass: $(CUDA_TARGET)
	@which cuobjdump >/dev/null 2>&1 || (echo "cuobjdump not found. Install CUDA toolkit."; exit 1)
	cuobjdump --dump-sass $(CUDA_TARGET) > $(CUDA_TARGET).sass
	@echo "Wrote exec SASS -> $(CUDA_TARGET).sass"

# ---- Listing helpers (what kernels/code objects are inside) ----
# List kernels / code objects embedded in the built executable (fatbin)
list_exec: $(CUDA_TARGET)
	@which cuobjdump >/dev/null 2>&1 || (echo "cuobjdump not found. Install CUDA toolkit."; exit 1)
	@echo "== cuobjdump --list $(CUDA_TARGET) =="
	cuobjdump --list $(CUDA_TARGET)

# List kernels / code objects inside a cubin (if you built one via `make cubin`)
list_cubin: $(CUBIN_FILE)
	@which cuobjdump >/dev/null 2>&1 || (echo "cuobjdump not found. Install CUDA toolkit."; exit 1)
	@echo "== cuobjdump --list $(CUBIN_FILE) =="
	cuobjdump --list $(CUBIN_FILE)

# Optional: quick ELF metadata view (sections, symbols) from the executable
dump_exec_elf: $(CUDA_TARGET)
	@which cuobjdump >/dev/null 2>&1 || (echo "cuobjdump not found. Install CUDA toolkit."; exit 1)
	@echo "== cuobjdump --dump-elf $(CUDA_TARGET) =="
	cuobjdump --dump-elf $(CUDA_TARGET)

# ---- housekeeping ----
clean:
	rm -f $(CPU_OBJ) $(CUDA_OBJS) $(COMPARE_OBJ) \
	      $(CPU_TARGET) $(CUDA_TARGET) $(COMPARE_TGT) \
	      $(PTX_FILE) $(CUBIN_FILE) $(SASS_FILE) \
	      $(CUDA_TARGET).ptx $(CUDA_TARGET).sass

