#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <thread>
#include <mutex>
#include <functional>

using namespace std;

// Forward declaration
template <typename T>
class Row_Major_Matrix;

// Column Major Matrix class definition
template <typename T>
class Column_Major_Matrix {
private:
    vector<vector<T>> all_column;
    size_t rows;
    size_t cols;

public:
    // Constructor: initialize matrix with random values
    Column_Major_Matrix(size_t rows, size_t cols) : rows(rows), cols(cols) {
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<> dis(1, 100); // Generate random integers between 1 and 100 for simplicity

        // Initialize column vectors
        all_column.resize(cols);
        for (size_t j = 0; j < cols; ++j) {
            all_column[j].resize(rows);
            for (size_t i = 0; i < rows; ++i) {
                all_column[j][i] = static_cast<T>(dis(gen));
            }
        }
    }

    // Copy constructor
    Column_Major_Matrix(const Column_Major_Matrix& other) : rows(other.rows), cols(other.cols) {
        // std::vector handles deep copy!!!!!
        all_column = other.all_column;
    }

    // Move constructor: Transfer ownership of resources from 'other' to this object
    Column_Major_Matrix(Column_Major_Matrix&& other) noexcept : rows(other.rows), cols(other.cols) {
        // std::move enables move semantics to transfer resources instead of copying them.
        // The internal pointer of other.all_column is transferred to all_column.
        // other.all_column is cleared to ensure it no longer owns these resources.
        all_column = std::move(other.all_column);
        other.rows = 0;
        other.cols = 0;
    }

    // Copy assignment operator: Deep copy from 'other' to this object
    Column_Major_Matrix& operator=(const Column_Major_Matrix& other) {
        // Used when the object already exists; std::vector handles deep copy.
        if (this != &other) {
            rows = other.rows;
            cols = other.cols;
            all_column = other.all_column;
        }
        return *this;
    }

    // Move assignment operator: Transfer ownership of resources from 'other' to this object
    Column_Major_Matrix& operator=(Column_Major_Matrix&& other) noexcept {
        if (this != &other) {
            rows = other.rows;
            cols = other.cols;
            all_column = std::move(other.all_column);
            other.rows = 0;
            other.cols = 0;
        }
        return *this;
    }

    // Get column vector
    const vector<T>& getColumn(size_t idx) const {
        if (idx >= cols) {
            throw out_of_range("Column index out of range");
        }
        return all_column[idx];
    }

    // Set column vector
    void setColumn(size_t idx, const vector<T>& column) {
        if (idx >= cols) {
            throw out_of_range("Column index out of range");
        }
        if (column.size() != rows) {
            throw invalid_argument("Column size does not match matrix dimensions");
        }
        all_column[idx] = column;
    }

    // Get row vector
    vector<T> getRow(size_t idx) const {
        if (idx >= rows) {
            throw out_of_range("Row index out of range");
        }
        vector<T> row(cols);
        for (size_t j = 0; j < cols; ++j) {
            row[j] = all_column[j][idx];
        }
        return row;
    }

    // Set row vector
    void setRow(size_t idx, const vector<T>& row) {
        if (idx >= rows) {
            throw out_of_range("Row index out of range");
        }
        if (row.size() != cols) {
            throw invalid_argument("Row size does not match matrix dimensions");
        }
        for (size_t j = 0; j < cols; ++j) {
            all_column[j][idx] = row[j];
        }
    }

    // Get element at specified position
    T get(size_t i, size_t j) const {
        if (i >= rows || j >= cols) {
            throw out_of_range("Index out of range");
        }
        return all_column[j][i];
    }

    // Set element at specified position
    void set(size_t i, size_t j, T value) {
        if (i >= rows || j >= cols) {
            throw out_of_range("Index out of range");
        }
        all_column[j][i] = value;
    }

    // Get dimensions
    size_t getRows() const { return rows; }
    size_t getCols() const { return cols; }

    // Overload * operator, compute product with Row_Major_Matrix
    // allow calculation of the product of a Column_Major_Matrix instance to a Row_Major_Matrix
    // instance, and return the resultant product as a Column_Major_Matrix.
    // const: ensures the function does not modify the object state.
    Column_Major_Matrix operator*(const Row_Major_Matrix<T>& other) const;

    // Multithreaded matrix multiplication operator
    Column_Major_Matrix operator%(const Row_Major_Matrix<T>& other) const;

    // Type conversion operator
    // operator: defines a custom type conversion
    // Row_Major_Matrix<T>: target type
    // (): operator parameter list, since type conversion operator does not require parameters, it is empty
    operator Row_Major_Matrix<T>() const;

    // Print matrix (for debugging)
    void print() const {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                cout << all_column[j][i] << " ";
            }
            cout << endl;
        }
    }
};

// Row Major Matrix class definition
template <typename T>
class Row_Major_Matrix {
private:
    vector<vector<T>> all_row;
    size_t rows;
    size_t cols;

public:
    // Constructor, specify dimensions and fill with random values
    Row_Major_Matrix(size_t rows, size_t cols) : rows(rows), cols(cols) {
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<> dis(1, 100); // Generate random integers between 1 and 100 for simplicity

        // Initialize row vectors
        all_row.resize(rows);
        for (size_t i = 0; i < rows; ++i) {
            all_row[i].resize(cols);
            for (size_t j = 0; j < cols; ++j) {
                all_row[i][j] = static_cast<T>(dis(gen));
            }
        }
    }

    // Copy constructor
    Row_Major_Matrix(const Row_Major_Matrix& other) : rows(other.rows), cols(other.cols) {
        all_row = other.all_row;
    }

    // Move constructor
    Row_Major_Matrix(Row_Major_Matrix&& other) noexcept : rows(other.rows), cols(other.cols) {
        all_row = std::move(other.all_row);
        other.rows = 0;
        other.cols = 0;
    }

    // Copy assignment operator
    Row_Major_Matrix& operator=(const Row_Major_Matrix& other) {
        if (this != &other) {
            rows = other.rows;
            cols = other.cols;
            all_row = other.all_row;
        }
        return *this;
    }

    // Move assignment operator
    Row_Major_Matrix& operator=(Row_Major_Matrix&& other) noexcept {
        if (this != &other) {
            rows = other.rows;
            cols = other.cols;
            all_row = std::move(other.all_row);
            other.rows = 0;
            other.cols = 0;
        }
        return *this;
    }

    // Get row vector
    const vector<T>& getRow(size_t idx) const {
        if (idx >= rows) {
            throw out_of_range("Row index out of range");
        }
        return all_row[idx];
    }

    // Set row vector
    void setRow(size_t idx, const vector<T>& row) {
        if (idx >= rows) {
            throw out_of_range("Row index out of range");
        }
        if (row.size() != cols) {
            throw invalid_argument("Row size does not match matrix dimensions");
        }
        all_row[idx] = row;
    }

    // Get column vector
    vector<T> getColumn(size_t idx) const {
        if (idx >= cols) {
            throw out_of_range("Column index out of range");
        }
        vector<T> column(rows);
        for (size_t i = 0; i < rows; ++i) {
            column[i] = all_row[i][idx];
        }
        return column;
    }

    // Set column vector
    void setColumn(size_t idx, const vector<T>& column) {
        if (idx >= cols) {
            throw out_of_range("Column index out of range");
        }
        if (column.size() != rows) {
            throw invalid_argument("Column size does not match matrix dimensions");
        }
        for (size_t i = 0; i < rows; ++i) {
            all_row[i][idx] = column[i];
        }
    }

    // Get element at specified position
    T get(size_t i, size_t j) const {
        if (i >= rows || j >= cols) {
            throw out_of_range("Index out of range");
        }
        return all_row[i][j];
    }

    // Set element at specified position
    void set(size_t i, size_t j, T value) {
        if (i >= rows || j >= cols) {
            throw out_of_range("Index out of range");
        }
        all_row[i][j] = value;
    }

    // Get dimensions
    size_t getRows() const { return rows; }
    size_t getCols() const { return cols; }

    // Overload * operator, compute product with Column_Major_Matrix
    // allow calculation of the product of a Row_Major_Matrix instance to a Column_Major_Matrix
    // instance, and return the resultant product as a Row_Major_Matrix.
    // const: ensures the function does not modify the object state.
    Row_Major_Matrix operator*(const Column_Major_Matrix<T>& other) const;

    // Multithreaded matrix multiplication operator
    Row_Major_Matrix operator%(const Column_Major_Matrix<T>& other) const;

    // Type conversion operator: Convert to Column_Major_Matrix
    operator Column_Major_Matrix<T>() const;

    // Print matrix (for debugging)
    void print() const {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                cout << all_row[i][j] << " ";
            }
            cout << endl;
        }
    }
};

// Column_Major_Matrix type conversion operator implementation
template <typename T>
Column_Major_Matrix<T>::operator Row_Major_Matrix<T>() const {
    Row_Major_Matrix<T> result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        vector<T> row = this->getRow(i);
        result.setRow(i, row);
    }
    return result;
}

// Row_Major_Matrix type conversion operator implementation
template <typename T>
Row_Major_Matrix<T>::operator Column_Major_Matrix<T>() const {
    Column_Major_Matrix<T> result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        vector<T> column = this->getColumn(i);
        result.setColumn(i, column);
    }
    return result;
}

// Column_Major_Matrix matrix multiplication operator implementation
template <typename T>
Column_Major_Matrix<T> Column_Major_Matrix<T>::operator*(const Row_Major_Matrix<T>& other) const {
    if (this->cols != other.getRows()) { // getRows(): return dimensions of rows
        throw invalid_argument("Matrix dimensions do not match for multiplication");
    }

    // result dimensions: (m, p) * (p, n) = (m, n)
    size_t m = this->rows; 
    size_t n = other.getCols();
    size_t p = this->cols; // = other.getRows()

    Column_Major_Matrix<T> result(m, n);

    // Perform matrix multiplication
    for (size_t j = 0; j < n; ++j) {
        for (size_t i = 0; i < m; ++i) {
            T sum = 0;
            for (size_t k = 0; k < p; ++k) {
                sum += this->get(i, k) * other.get(k, j);
            }
            result.set(i, j, sum);
        }
    }

    return result;
}

// Row_Major_Matrix matrix multiplication operator implementation
template <typename T>
Row_Major_Matrix<T> Row_Major_Matrix<T>::operator*(const Column_Major_Matrix<T>& other) const {
    if (this->cols != other.getRows()) {
        throw invalid_argument("Matrix dimensions do not match for multiplication");
    }

    size_t m = this->rows;
    size_t n = other.getCols();
    size_t p = this->cols; // = other.getRows()

    Row_Major_Matrix<T> result(m, n);

    // Perform matrix multiplication
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            T sum = 0;
            for (size_t k = 0; k < p; ++k) {
                sum += this->get(i, k) * other.get(k, j);
            }
            result.set(i, j, sum);
        }
    }

    return result;
}

// Multithreaded helper function for partial matrix multiplication
template <typename T>
void multiplyPartial(
    const Column_Major_Matrix<T>& a, 
    const Row_Major_Matrix<T>& b, 
    Column_Major_Matrix<T>& result, 
    size_t start_row, 
    size_t end_row) {
    
    // result dimensions: (m, p) * (p, n) = (m, n)
    size_t n = b.getCols();
    size_t p = a.getCols(); // = b.getRows()

    for (size_t j = 0; j < n; ++j) {
        for (size_t i = start_row; i < end_row; ++i) {
            T sum = 0;
            for (size_t k = 0; k < p; ++k) {
                sum += a.get(i, k) * b.get(k, j);
            }
            result.set(i, j, sum);
        }
    }
}

// Column_Major_Matrix multithreaded matrix multiplication operator implementation
template <typename T>
Column_Major_Matrix<T> Column_Major_Matrix<T>::operator%(const Row_Major_Matrix<T>& other) const {
    if (this->cols != other.getRows()) {
        throw invalid_argument("Matrix dimensions do not match for multiplication");
    }
    
    // result dimensions: (m, p) * (p, n) = (m, n)
    size_t m = this->rows; 
    size_t n = other.getCols();
    
    Column_Major_Matrix<T> result(m, n);
    
    // Calculate rows per thread
    // num of threads = 10
    size_t rows_per_thread = m / 10;
    vector<thread> threads;

    // Create 10 threads
    for (size_t i = 0; i < 10; ++i) {
        size_t start_row = i * rows_per_thread;
        size_t end_row = (i == 9) ? m : (i + 1) * rows_per_thread; // Last thread handles remaining rows

        // vector.emplace_back: 將物件直接構造在容器中，避免複製
        // std::ref: 用於將物件轉換為引用(避免複製)，確保在multithread環境中正確共享物件
        threads.emplace_back(multiplyPartial<T>, std::ref(*this), std::ref(other), 
                            std::ref(result), start_row, end_row);
    }

    // Wait for all threads to finish
    for (auto& thread : threads) {
        thread.join();
    }

    return result;
}

// Multithreaded helper function for Row_Major_Matrix partial matrix multiplication
template <typename T>
void multiplyPartialRow(
    const Row_Major_Matrix<T>& a, 
    const Column_Major_Matrix<T>& b, 
    Row_Major_Matrix<T>& result, 
    size_t start_row, 
    size_t end_row) {
    
    // result dimensions: (m, p) * (p, n) = (m, n)
    size_t n = b.getCols();
    size_t p = a.getCols(); // = b.getRows()

    for (size_t i = start_row; i < end_row; ++i) {
        for (size_t j = 0; j < n; ++j) {
            T sum = 0;
            for (size_t k = 0; k < p; ++k) {
                sum += a.get(i, k) * b.get(k, j);
            }
            result.set(i, j, sum);
        }
    }
}

// Row_Major_Matrix multithreaded matrix multiplication operator implementation
template <typename T>
Row_Major_Matrix<T> Row_Major_Matrix<T>::operator%(const Column_Major_Matrix<T>& other) const {
    if (this->cols != other.getRows()) {
        throw invalid_argument("Matrix dimensions do not match for multiplication");
    }

    // result dimensions: (m, p) * (p, n) = (m, n)
    size_t m = this->rows;
    size_t n = other.getCols();
    
    Row_Major_Matrix<T> result(m, n);
    
    // Calculate rows per thread
    size_t rows_per_thread = m / 10;
    vector<thread> threads;

    // Create 10 threads
    for (size_t i = 0; i < 10; ++i) {
        size_t start_row = i * rows_per_thread;
        size_t end_row = (i == 9) ? m : (i + 1) * rows_per_thread; // Last thread handles remaining rows

        // std::ref: 用於將物件轉換為引用，確保在multithread環境中正確共享物件
        threads.emplace_back(multiplyPartialRow<T>, std::ref(*this), std::ref(other), 
                            std::ref(result), start_row, end_row);
    }

    // Wait for all threads to finish
    for (auto& thread : threads) {
        thread.join();
    }

    return result;
}

// Benchmark function: Compare single-threaded and multithreaded matrix multiplication
template <typename T>
void benchmark(const Column_Major_Matrix<T>& a, const Row_Major_Matrix<T>& b) {
    // Test single-threaded multiplication
    auto start = std::chrono::high_resolution_clock::now();
    auto result1 = a * b;
    auto end = std::chrono::high_resolution_clock::now();
    auto single_thread_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    // Test multithreaded multiplication
    start = std::chrono::high_resolution_clock::now();
    auto result2 = a % b;
    end = std::chrono::high_resolution_clock::now();
    auto multi_thread_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    cout << "Matrix multiplication benchmark:" << endl;
    cout << "Single thread time: " << single_thread_time << " ms" << endl;
    cout << "Multi thread time: " << multi_thread_time << " ms" << endl;
    cout << "Speedup: " << static_cast<double>(single_thread_time) / multi_thread_time << "x" << endl;
}

int main() {
    // Test Matrix class basic functionality
    cout << "Testing constructors..." << endl;
    Column_Major_Matrix<int> cc1(1000, 1000);
    Row_Major_Matrix<int> rr1(1000, 1000);
    
    // Test copy constructor
    cout << "\nTesting copy constructor and copy assignment operator..." << endl;
    Column_Major_Matrix<int> cc2(cc1);
    Row_Major_Matrix<int> rr2 = rr1;
    
    // Test move constructor
    cout << "\nTesting move constructor and move assignment operator..." << endl;
    Column_Major_Matrix<int> cc3 = std::move(cc2);
    Row_Major_Matrix<int> rr3 = std::move(rr2);
    
    // Test implicit type conversion
    cout << "\nTesting implicit type conversion & matrix multiplication...\nA = Column_Major_Matrix(55, 1000), B = Row_Major_Matrix(1000, 66)" << endl;
    Column_Major_Matrix<int> cc(55, 1000);
    Row_Major_Matrix<int> rr(1000, 66);
    Row_Major_Matrix<int> result = cc * rr;
    cout << "Result = Row_Major_Matrix(" << result.getRows() << ", " << result.getCols() << ")" << endl;
    
    // Benchmark performance
    cout << "\nBenchmarking performance..." << endl;
    Column_Major_Matrix<int> small_cc(500, 500);
    Row_Major_Matrix<int> small_rr(500, 500);
    benchmark(small_cc, small_rr);
    
    return 0;
}