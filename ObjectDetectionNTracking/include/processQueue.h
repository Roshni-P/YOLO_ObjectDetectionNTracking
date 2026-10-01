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
    ProcessQueue(int qSize): maxSize(qSize)
    {
    }

    bool isEmpty()
    {
        std::lock_guard<std::mutex> lock(mtx);
        
        return queue.empty();
    }

    void push(T element)
    {
        std::unique_lock<std::mutex> lock(mtx);
        if(maxSize>0 && !queue.empty())
        {
            queue.pop();
        }

        queue.push(std::move(element));
        cv.notify_one();
    }

    bool pop(T& element)
    {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [this](){return !queue.empty();});

        element = std::move(queue.front());
        queue.pop();

        return true;
    }
};