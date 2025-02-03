#pragma once

#include "common_inc.hpp"

#include <thread>
#include <mutex>

// Abstract class to simplify working on shared data across multiple threads
class ParallelWorker {
public:
    class SharedData {
    public:
        // Hold onto the lock while writing
        // Make sure to release it after
        inline std::lock_guard<std::mutex> lock() { return std::lock_guard<std::mutex>(mutex); };
    private:
        std::mutex mutex;
    };

    ~ParallelWorker() {
        end();
        join();
    };
    
    void setData(SharedData& data) {
        _data = &data;
    }
    void doWork() {
        _end = false;
        _done = false;
        _thread = std::thread([this] {
            work(_data);
            _done = true;
        });
    };
    void end() {
        _end = true;
    };
    bool done() {
        return _done;
    }
    void join() {
        if (_thread.joinable())
            _thread.join();
    }
protected:
    // Override this function
    // End loop as soon as possible when _end is true
    virtual void work(SharedData* _d) = 0;
    bool _end = false;
private:
    SharedData* _data = NULL;
    std::thread _thread;
    bool _done = false;
};
