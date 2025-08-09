# Mixed toolchain: nvcc for .cu, g++ for host .cpp
NVCC      ?= nvcc
CXX       ?= g++

# For a Turing-first build with PTX fallback
ARCH ?= 75
GENCODE = -gencode arch=compute_$(ARCH),code=sm_$(ARCH) \
          -gencode arch=compute_$(ARCH),code=compute_$(ARCH)

CXXFLAGS  ?= -O3 -std=c++17
NVCCFLAGS ?= -O3 -std=c++17 $(GENCODE)

LIBS       = -lncurses
LDFLAGS    =

TARGET = life
CU_SRC = life.cu
CPP_SRC = life_ncurses.cpp
OBJ = $(CU_SRC:.cu=.cu.o) $(CPP_SRC:.cpp=.o)

.PHONY: all run benchmark debug clean

all: $(TARGET)

$(TARGET): $(OBJ)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LIBS) $(LDFLAGS)

%.cu.o: %.cu life_kernel.cuh
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

%.o: %.cpp life_kernel.cuh
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

benchmark: $(TARGET)
	./$(TARGET) --benchmark

debug: CXXFLAGS := -O0 -g3 -std=c++17
debug: NVCCFLAGS := -O0 -g -G -lineinfo -std=c++17 $(GENCODE)
debug: clean $(TARGET)

clean:
	rm -f $(OBJ) $(TARGET)

