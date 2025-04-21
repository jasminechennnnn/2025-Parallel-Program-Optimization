CC = g++
CFLAGS = -std=c++17 -O3 -Wall -Wextra
INCLUDES = -I$(shell brew --prefix xsimd)/include
LIBS = 

# For Apple Silicon
CFLAGS += -arch arm64

# Target executable name
TARGET = ssw_xsimd

# Source files
SRC = main.cpp ssw_xsimd.cpp

# Object files
OBJ = $(SRC:.cpp=.o)

# Header files
HEADERS = ssw_xsimd.h

# Default target
all: $(TARGET)

# Link the target executable
$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

# Compile source files
%.o: %.cpp $(HEADERS)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Clean build files
clean:
	rm -f $(OBJ) $(TARGET)

# Run the program
run: $(TARGET)
	./$(TARGET) target.fa query.fa