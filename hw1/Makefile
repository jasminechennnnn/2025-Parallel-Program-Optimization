CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O3 -pthread

all: 1_matrix 2_threadpool

1_matrix: 1_matrix.cpp
	$(CXX) $(CXXFLAGS) 1_matrix.cpp -o 1_matrix

2_threadpool: 2_threadpool.cpp
	$(CXX) $(CXXFLAGS) 2_threadpool.cpp -o 2_threadpool

clean:
	rm -f 1_matrix 2_threadpool