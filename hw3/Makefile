# Compiler settings
CXX = g++
NVCC = nvcc
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra
NVCCFLAGS = -std=c++17 -O3 -Xcompiler "-Wall -Wextra"

# Target executable name
TARGET = ssw_cuda

# Source files
CPP_SRC = main.cpp ssw_common.cpp ssw_xsimd.cpp
CUDA_SRC = ssw_cuda.cu

# Object files
CPP_OBJ = $(CPP_SRC:.cpp=.o)
CUDA_OBJ = $(CUDA_SRC:.cu=.o)
OBJ = $(CPP_OBJ) $(CUDA_OBJ)

# Header files
HEADERS = ssw_xsimd.h ssw_common.h ssw_cuda.h

# Detect operating system
UNAME_S := $(shell uname -s)

# Path to vcpkg includes
VCPKG_PATH = /home/nycu813/Wall/2025-Parallel-Program-Optimization/vcpkg/installed/x64-linux/include

# CUDA path
CUDA_PATH = /usr/local/cuda
CUDA_INCLUDES = -I$(CUDA_PATH)/include
CUDA_LIBS = -L$(CUDA_PATH)/lib64 -lcudart -lcuda

# macOS specific settings
ifeq ($(UNAME_S),Darwin)
    UNAME_M := $(shell uname -m)
    ifeq ($(UNAME_M),arm64)
        CXXFLAGS += -arch arm64
    endif
    
    BREW_PREFIX := $(shell brew --prefix xsimd 2>/dev/null)
    ifneq ($(BREW_PREFIX),)
        INCLUDES = -I$(BREW_PREFIX)/include
    endif
    
    # Check for CUDA on macOS
    ifeq ($(wildcard $(CUDA_PATH)),)
        CUDA_PATH = /usr/local/cuda
    endif
endif

# Linux specific settings
ifeq ($(UNAME_S),Linux)
    # Use the user-defined path for vcpkg
    INCLUDES = -I$(VCPKG_PATH)
endif

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $(CPP_OBJ) $(CUDA_OBJ) $(CUDA_LIBS)

%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(CUDA_INCLUDES) -c $< -o $@

%.o: %.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)

run: $(TARGET)
	./$(TARGET) target.fa query.fa