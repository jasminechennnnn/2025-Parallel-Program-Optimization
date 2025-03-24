# HW1: Matrix and Thread Pool

This homework consists of two parts:
1. **Matrix Class Implementation and Operations** (1_matrix.cpp)
2. **Thread Pool Implementation and Application** (2_threadpool.cpp)

## Repository Link

[2025-Parallel-Program-Optimization/hw1](https://github.com/jasminechennnnn/2025-Parallel-Program-Optimization/tree/main/hw1)

## 1 Usage

### 1.1 Part 1 - Matrix
```bash
make 1_matrix
./1_matrix
```

### 1.2 Part 2 - Thread Pool
```bash
make 2_threadpool
./2_threadpool
```

### Common Commands
```bash
make all    # Compile all programs
make clean  # Clean compiled files
```

## 2 My Notes

### 2.1 Part 1 - Matrix

#### Class Design

**Column_Major_Matrix Class**
```cpp
template <typename T>
class Column_Major_Matrix {
private:
    std::vector<std::vector<T>> all_column;
    size_t rows;
    size_t cols;
};
public:
-`Column_Major_Matrix(size_t rows, size_t cols)`: Constructor
- `Column_Major_Matrix(const Column_Major_Matrix& other)`: Copy constructor
- `Column_Major_Matrix(Column_Major_Matrix&& other) noexcept`: Move constructor
- `operator=(const Column_Major_Matrix& other)`: Copy assignment
- `operator=(Column_Major_Matrix&& other) noexcept`: Move assignment
- `const vector<T>& getColumn(size_t idx) const`: Get column
- `void setColumn(size_t idx, const vector<T>& column)`: Set column
- `vector<T> getRow(size_t idx) const`: Get row
- `void setRow(size_t idx, const vector<T>& row)`: Set row
- `T get(size_t i, size_t j) const`: Get element
- `void set(size_t i, size_t j, T value)`: Set element
- `size_t getRows() const`: Get row count
- `size_t getCols() const`: Get column count
- `operator*(const Row_Major_Matrix<T>& other) const`: Matrix multiplication
- `operator%(const Row_Major_Matrix<T>& other) const`: Multithreaded multiplication
- `operator Row_Major_Matrix<T>() const`: Type conversion
- `void print() const`: Print matrix (debugging)
```

**Row_Major_Matrix Class**
```cpp
template <typename T>
class Row_Major_Matrix {
private:
    std::vector<std::vector<T>> all_row;
    size_t rows;
    size_t cols;
};
public:
- `Row_Major_Matrix(size_t rows, size_t cols)`: Constructor
- `Row_Major_Matrix(const Row_Major_Matrix& other)`: Copy constructor
- `Row_Major_Matrix(Row_Major_Matrix&& other) noexcept`: Move constructor
- `operator=(const Row_Major_Matrix& other)`: Copy assignment
- `operator=(Row_Major_Matrix&& other) noexcept`: Move assignment
- `const vector<T>& getRow(size_t idx) const`: Get row
- `void setRow(size_t idx, const vector<T>& row)`: Set row
- `vector<T> getColumn(size_t idx) const`: Get column
- `void setColumn(size_t idx, const vector<T>& column)`: Set column
- `T get(size_t i, size_t j) const`: Get element
- `void set(size_t i, size_t j, T value)`: Set element
- `size_t getRows() const`: Get row count
- `size_t getCols() const`: Get column count
- `operator*(const Column_Major_Matrix<T>& other) const`: Matrix multiplication
- `operator%(const Column_Major_Matrix<T>& other) const`: Multithreaded multiplication
- `operator Column_Major_Matrix<T>() const`: Type conversion
- `void print() const`: Print matrix (debugging)
```

#### Constructor Implementation (Rule of 5)

- **Basic Constructor**:
  - Takes dimension parameters and fills the matrix with randomly generated values
  
- **Copy Constructor**:
  ```cpp
  // Copy constructor - deep copy (O(n)), the copied object is expected to be used again
  Column_Major_Matrix<int> cc2(cc1);  // Direct initialization
  Row_Major_Matrix<int> rr2 = rr1;    // Copy initialization
  ```

- **Move Constructor**:
  ```cpp
  // Move constructor - shallow copy (O(1)) then clear the original object, which is not expected to be used again
  Column_Major_Matrix<int> cc3 = std::move(cc2);
  ```

#### Assignment Operator Overloading

- **Copy Assignment Operator**:
  ```cpp
  // Clean the target object's existing resources, then copy resources from the source object
  cc2 = cc1;  // cc2 must be an existing object
  ```

- **Move Assignment Operator**:
  ```cpp
  // Clean the target object's existing resources, then move resources from the source object (no copying)
  cc2 = std::move(cc1);
  ```

#### Matrix Multiplication Operator Overloading

- **`operator*`**: Implements standard matrix multiplication
- **`operator%`**: Implements multi-threaded matrix multiplication (using 10 threads)

#### Type Conversion Operators

Implements implicit conversion between matrix types, as shown in the example:
```cpp
Column_Major_Matrix<int> cc(55, 1000);
Row_Major_Matrix<int> rr(1000, 66);
Row_Major_Matrix<int> result = cc * rr;  // Using implicit conversion
```

#### Getter/Setter Functions

Provides methods to access rows and columns by index:
- `getRow(size_t idx)`
- `setRow(size_t idx, const std::vector<T>& row)`
- `getColumn(size_t idx)`
- `setColumn(size_t idx, const std::vector<T>& column)`
- `get(size_t i, size_t j)`
- `set(size_t i, size_t j, T value)`

### 2.2 Part 2 - Thread Pool

#### Thread Pool Class Requirements

- [x] A. Allow users to send jobs into the pool - implemented via the `pool.enqueue()` method
- [x] B. Allow any kind of callable objects as jobs - using `std::function<void()>` and templated `enqueue` method
- [x] C. Maintain a job queue to store unfinished jobs - using `std::queue<std::function<void()>> tasks`
- [x] D. Have 5 threads always waiting for new jobs, each keeping a record of total running time - created in the constructor and using `std::chrono` to track running time
- [x] E. Threads are terminated (joined) only when the thread pool is destructed, showing total running time and thread ID - implemented in the destructor
- [x] F. Use condition variable and mutex to notify threads to do work - using `std::condition_variable` and `std::mutex`

#### Special Task Implementation

- [x] Implemented `print_1` function: generates a random integer number and prints '1' if the number is odd, otherwise '0'; protects cout as a shared resource
- [x] Implemented `print_2` functor: prints "2", ensuring it's only executed when there are no more `print_1` jobs running - using condition variable for the waiting mechanism
- [x] In main, first sends 496 functions and then 4 functors into the pool

#### Thread Pool Core Design

```cpp
class ThreadPool {
private:
    std::vector<std::thread> workers;                // Thread vector
    std::queue<std::function<void()>> tasks;         // Task queue
    std::mutex queue_mutex;                          // Task queue mutex
    std::condition_variable condition;               // Condition variable for thread notification
    std::condition_variable print1_done_condition;   // Condition variable for print_1 completion
    std::atomic<int> active_print1_tasks;            // Counter for active print_1 tasks
    bool stop;                                       // Stop flag
    
    // Thread running time and ID records
    std::vector<std::chrono::milliseconds> running_times;
    std::vector<std::thread::id> thread_ids;
};
```