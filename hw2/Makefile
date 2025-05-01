CC = g++
CFLAGS = -std=c++17 -O3 -Wall -Wextra

# Target executable name
TARGET = ssw_xsimd

# Source files
SRC = main.cpp ssw_xsimd.cpp

# Object files
OBJ = $(SRC:.cpp=.o)

# Header files
HEADERS = ssw_xsimd.h

# Detect operating system
UNAME_S := $(shell uname -s)

VCPKG_PATH = /home/nycu813/Wall/2025-Parallel-Program-Optimization/vcpkg/installed/x64-linux/include

# macOS specific settings
ifeq ($(UNAME_S),Darwin)
    UNAME_M := $(shell uname -m)
    ifeq ($(UNAME_M),arm64)
        CFLAGS += -arch arm64
    endif
    
    BREW_PREFIX := $(shell brew --prefix xsimd 2>/dev/null)
    ifneq ($(BREW_PREFIX),)
        INCLUDES = -I$(BREW_PREFIX)/include
    endif
endif

# Linux specific settings
ifeq ($(UNAME_S),Linux)
    # Use the user-defined path
    INCLUDES = -I$(VCPKG_PATH)
endif


all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

%.o: %.cpp $(HEADERS)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)

run: $(TARGET)
	./$(TARGET) target.fa query.fa