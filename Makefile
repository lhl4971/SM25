# Makefile for MPI + CUDA + OpenMP
ARCH      ?= sm_60
HOST_COMP ?= mpicc

CUDA_HOME ?= $(shell readlink -f $(shell which nvcc) | sed 's:/bin/nvcc::')

MPI_INC_DIR := $(shell mpicc -show 2>/dev/null | sed -n 's/.*-I\([^ ]*\).*/\1/p' | head -n1)

NVCC := $(CUDA_HOME)/bin/nvcc
CXX  := mpicxx

CXXFLAGS  := -std=c++11 -O3
NVCCFLAGS := -std=c++11 -O3 -arch=$(ARCH) -Xcompiler "-fPIC"

INCLUDES  := -Iinclude -Isrc \
             -I/usr/lib/x86_64-linux-gnu/openmpi/include \
             -I$(MPI_INC_DIR) -I$(CUDA_HOME)/include

# ===== Sources =====
MAIN_SRC      := task_mpi_cuda.cpp

CPP_SRCS      := src/conjugate_gradient.cpp \
                 src/mpi_conjugate_gradient.cpp

CUDA_SRCS     := src/cuda_kernels.cu \
                 src/cuda_operators.cu \
                 src/mpi_cuda_conjugate_gradient.cu

# ===== Objects =====
OBJ_DIR       := obj

OBJS_CPP_MAIN  := $(OBJ_DIR)/$(MAIN_SRC:.cpp=.o)
OBJS_CPP_EXTRA := $(patsubst src/%.cpp,$(OBJ_DIR)/%.o,$(CPP_SRCS))
OBJS_CU        := $(patsubst src/%.cu,$(OBJ_DIR)/%.o,$(CUDA_SRCS))

OBJS          := $(OBJS_CPP_MAIN) $(OBJS_CPP_EXTRA) $(OBJS_CU)

TARGET        := task_mpi_cuda

.SUFFIXES:

all: $(TARGET)

$(TARGET): $(OBJS)
	@echo "===> Linking $(TARGET)"
	$(CXX) $(CXXFLAGS) -o $@ $^ -L$(CUDA_HOME)/lib64 -lcudart -lcuda -lm

# Rule for .cpp files in .
$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	@echo "===> CXX $<"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Rule for .cpp files under src/ =====
$(OBJ_DIR)/%.o: src/%.cpp
	@mkdir -p $(dir $@)
	@echo "===> CXX (src) $<"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Rule for CUDA files
$(OBJ_DIR)/%.o: src/%.cu
	@mkdir -p $(dir $@)
	@echo "===> NVCC $<"
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(TARGET)

info:
	@echo "ARCH       = $(ARCH)"
	@echo "HOST_COMP  = $(HOST_COMP)"
	@echo "CUDA_HOME  = $(CUDA_HOME)"
	@echo "MPI_INC_DIR= $(MPI_INC_DIR)"
	@echo "NVCC       = $(NVCC)"
