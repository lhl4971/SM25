#ifndef TIMER_HPP
#define TIMER_HPP

#include <chrono>
#include <unordered_map>
#include <string>

#ifdef POISSON_SOLVER_CUDA_ENABLE
#include <cuda_runtime.h>
#endif

class Timer {
private:
    struct Record {
        double total = 0.0;
        bool running = false;
        std::chrono::high_resolution_clock::time_point start_t;
    };

    std::unordered_map<std::string, Record> timers;

public:
    Timer() {}

    void reset(const std::string &name) {
        timers[name] = Record();
    }

    void start(const std::string &name) {
        Record &rec = timers[name];
        if (rec.running) return;
#ifdef POISSON_SOLVER_CUDA_ENABLE
        cudaDeviceSynchronize();
#endif
        rec.start_t = std::chrono::high_resolution_clock::now();
        rec.running = true;
    }

    void stop(const std::string &name) {
        Record &rec = timers[name];
        if (!rec.running) return;
#ifdef POISSON_SOLVER_CUDA_ENABLE
        cudaDeviceSynchronize();
#endif
        auto end_t = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double>(end_t - rec.start_t).count();
        rec.total += dt;
        rec.running = false;
    }

    double get(const std::string &name) const {
        std::unordered_map<std::string, Record>::const_iterator it = timers.find(name);
        if (it == timers.end()) return 0.0;
        return it->second.total;
    }
};

#endif // TIMER_HPP