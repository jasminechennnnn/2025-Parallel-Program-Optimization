#include <iostream>
#include <queue>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>
#include <chrono>
#include <random>
#include <iomanip>

using namespace std;

// Thread pool class
class ThreadPool {
private:
    // Worker threads
    std::vector<std::thread> workers;
    // Task queue
    std::queue<std::function<void()>> tasks;
    
    // Synchronization primitives
    std::mutex queue_mutex;
    std::condition_variable condition;
    std::condition_variable print1_done_condition;
    std::mutex print1_mutex;
    
    // Control state
    bool stop;
    std::atomic<int> active_print1_tasks;
    
    // Thread runtime tracking
    std::vector<std::chrono::milliseconds> running_times;
    std::vector<std::thread::id> thread_ids;

public:
    // Constructor: Initialize thread pool
    ThreadPool(size_t threads = 5) : stop(false), active_print1_tasks(0) {
        running_times.resize(threads);
        thread_ids.resize(threads);
        
        for (size_t i = 0; i < threads; ++i) {
            workers.emplace_back([this, i] {
                // Record thread ID
                thread_ids[i] = std::this_thread::get_id();
                
                // Record start time
                auto start_time = std::chrono::high_resolution_clock::now();
                
                while (true) {
                    std::function<void()> task;
                    
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex);
                        
                        // Wait for a task or stop signal
                        condition.wait(lock, [this] { 
                            return stop || !tasks.empty(); 
                        });
                        
                        // Exit if pool is stopped and no tasks remain
                        if (stop && tasks.empty()) {
                            break;
                        }
                        
                        // Get a task
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    
                    // Execute the task
                    task();
                }
                
                // Calculate total runtime
                auto end_time = std::chrono::high_resolution_clock::now();
                running_times[i] = std::chrono::duration_cast<std::chrono::milliseconds>(
                    end_time - start_time);
            });
        }
    }
    
    // Destructor: Wait for all threads to finish
    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        
        // Notify all threads
        condition.notify_all();
        
        // Join all threads
        for (auto& worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        
        // Output each thread's runtime
        std::cout << std::endl;
        for (size_t i = 0; i < workers.size(); ++i) {
            std::cout << "Thread " << i << " (ID: " << thread_ids[i] 
                      << ") total running time: " << running_times[i].count() 
                      << " ms" << std::endl;
        }
    }
    
    // Add a task to the thread pool
    template<class F>
    void enqueue(F&& f) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            
            // Reject new tasks if pool is stopped
            if (stop) {
                throw std::runtime_error("enqueue on stopped ThreadPool");
            }
            
            // Add task to queue
            tasks.emplace(std::forward<F>(f));
        }
        
        // Notify one thread
        condition.notify_one();
    }
    
    // Check if all print_1 tasks are done
    bool are_print1_tasks_done() const {
        return active_print1_tasks == 0;
    }
    
    // Wait for all print_1 tasks to complete
    void wait_for_print1_done() {
        std::unique_lock<std::mutex> lock(print1_mutex);
        print1_done_condition.wait(lock, [this] { 
            return this->are_print1_tasks_done(); 
        });
    }
    
    // Increment active print_1 task count
    void increment_print1_count() {
        active_print1_tasks++;
    }
    
    // Decrement active print_1 task count
    void decrement_print1_count() {
        if (--active_print1_tasks == 0) {
            std::unique_lock<std::mutex> lock(print1_mutex);
            print1_done_condition.notify_all();
        }
    }
};

// Mutex for protecting cout
std::mutex cout_mutex;

// print_1 function: Generate random integer, print 1 if odd, 0 if even
void print_1(ThreadPool& pool) {
    // Register print_1 task
    pool.increment_print1_count();
    
    // Generate random number
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 1000);
    int num = dis(gen);
    
    // Protect cout shared resource
    {
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cout << (num % 2 == 1 ? "1" : "0");
    }
    
    // Mark print_1 task as complete
    pool.decrement_print1_count();
}

// print_2 functor: Print 2 only after all print_1 tasks are done
class print_2 {
private:
    ThreadPool& pool;
    
public:
    explicit print_2(ThreadPool& p) : pool(p) {}
    
    void operator()() {
        // Wait for all print_1 tasks to complete
        pool.wait_for_print1_done();
        
        // Protect cout shared resource
        {
            std::lock_guard<std::mutex> lock(cout_mutex);
            std::cout << "2";
        }
    }
};

int main() {
    // Create thread pool
    ThreadPool pool;
    
    std::cout << "Starting 496 print_1 functions and 4 print_2 functors...\n";
    
    // Add 496 print_1 tasks
    for (int i = 0; i < 496; ++i) {
        pool.enqueue([&pool] {
            print_1(pool);
        });
    }
    
    // Add 4 print_2 tasks
    for (int i = 0; i < 4; ++i) {
        pool.enqueue(print_2(pool));
    }
    
    // Main thread waits to ensure all tasks complete
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    std::cout << "\nAll tasks should be completed now.\n";
    
    // Thread pool destructor waits for all threads and displays runtimes
    return 0;
}