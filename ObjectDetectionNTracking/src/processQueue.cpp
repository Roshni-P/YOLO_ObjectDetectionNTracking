#include "processQueue.h"

template<typename T>
ProcessQueue<T>::ProcessQueue(int qSize): maxSize(qSize)
{
}

template<typename T>
void ProcessQueue<T>::push(T element)
{
    std::unique_lock<std::mutex> lock(mtx);
    if(maxSize>0 && !queue.isempty())
    {
        queue.pop();
    }

    queue.push(std::move(element));
    cv.notify_one();
}

template<typename T>
bool ProcessQueue<T>::pop(T& element)
{
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, [this](){return !queue.isEmpty();});

    element = std::move(queue.front());
    queue.pop();

    return true;
}
template<typename T>
bool ProcessQueue<T>::isEmpty()
{
    std::lock_guard<std::mutex> lock(mtx);
    
    return queue.empty();
}