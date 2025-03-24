# 1 Usage

## 1.1 part 1
```bash
make 1_matrix
```
## 1.2 part 2
```bash
make 2_thread_pool
```

# 2 Note
## 2.1 Part 1
**class Column_Major_Matrix**
```cpp
template <typename T>
class Column_Major_Matrix {
private:
    vector<vector<T>> all_column;
    size_t rows;
    size_t cols;
};
```

**class Row_Major_Matrix**
```cpp
template <typename T>
class Row_Major_Matrix {
private:
    vector<vector<T>> all_row;
    size_t rows;
    size_t cols;
};
```

### constructors
(Rule of 5)
- constructor: takes arguments to specify the dimensions, fill with random values
- copy constructor
    ```cpp
    // copy constructor, 把 obj1 的資源複製到 obj2  deep copy(O(n)), 被複製的東西預設還會再被使用
    MyClass obj2(obj1);  // 直接初始化
    MyClass obj2 = obj1; // 複製初始化
    ```
- move constructor
    移動操作通常應標記為 noexcept，因為它們通常只涉及指針轉移，而不應該失敗。這樣告訴編譯器此函數不會拋出異常，編譯器可以進行更積極的優化
    ```cpp
    // move constructor, 把 obj1 的資源轉移給 obj2, obj1 的資源被清除 先shallow copy(O(1))然後再清除, 被移動的東西預設不會再被使用
    MyClass obj2 = std::move(obj1);  // 直接初始化
    ```

### Overload assignment
- assignment operator
    ```cpp
    // assignment operator, 清理 obj2 的現有資源，然後複製 obj1 的資源到 obj2
    obj2 = obj1;  // obj2 必須是"已經存在"的物件
    ```
- move assignment

### overload operator*


### Overload operator%


### conversion operators
**show it work by:**
```cpp
Column_Major_Matrix<int> cc (55, 1000);
Row_Major_Matrix<int> rr (1000, 66);
Row_Major_Matrix<int> rr = cc*rr;
```

### getter/setter function
access each column and row by an index.


## 2.2 Part 2
### Class pool

1. Design a thread pool class with following features:
A. Allow users to send jobs into the pool
B. Allow any kind of callable objects as jobs
C. Maintain a job queue to store unfinished jobs
i. Hint: element type: std::function/std::bind or package_task
D. Have 5 threads always waiting for new jobs. Each thread will keep a record of total
running time throughout the lifespan of the thread.
E. Threads are terminated(joined) only when the thread pool is
destructed. The total running time of each thread will be shown on the screen
upon destruction along with the std::thread::id.
F. Use condition variable and mutex to notify threads
to do works
2. 3. 4. Write one function (named print_1), which can generate a random integer number and
then print out ‘1’ if the number is an odd number otherwise ‘0’
. Note that cout is also a
shared resource.
Write a print_2 functor, which simply prints “2” on the screen. Use conditional variable
to ensure that print_2 functor can only be executed when there is no more print_1 job
to be executed.
In main, first send 496 functions and then 4 functors into the pool.