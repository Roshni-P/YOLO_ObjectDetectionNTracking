#pragma once
#include <queue>
#include <mutex>
#include <condition_variable>

/// @brief This is thread safe queue that acts as a bounded queue
///     when a maxSize is specified. If maxSize is zero, it is an
///     unbounded queue.
/// @tparam T 
template<typename T>
class ProcessQueue
{
    private:
    std::mutex mtx;
    std::queue<T> queue;
    std::condition_variable cv;
    int maxSize;

    public:
    ProcessQueue(int qSize);
    bool pop(T& element);
    void push(T element);
    bool isEmpty();
};