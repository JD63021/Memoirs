#pragma once

#include <cuda_runtime.h>

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>

namespace memoirs {
namespace diagnostics {

inline void cuda_timing_check(cudaError_t e, const char* expr) {
    if (e != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA timing error: ") +
            expr + " -> " + cudaGetErrorString(e)
        );
    }
}

class CudaSynchronizedTimer {
public:
    void start() {
        cuda_timing_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize before timer");
        t0_ = clock::now();
        running_ = true;
    }

    double stop_seconds() {
        if (!running_) {
            return 0.0;
        }

        cuda_timing_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize after timer");
        const auto t1 = clock::now();
        running_ = false;

        return std::chrono::duration<double>(t1 - t0_).count();
    }

private:
    using clock = std::chrono::steady_clock;

    clock::time_point t0_;
    bool running_ = false;
};

struct CudaGmgTimingReport {
    double rhsSeconds = 0.0;
    double setupSeconds = 0.0;
    double solveTotalSeconds = 0.0;
    double solveAvgSeconds = 0.0;
    int solveRepeats = 0;

    void print(std::ostream& os) const {
        os << "--------------- CUDA GMG timing diagnostic ---------------\n";
        os << "cudaGmgRhsSeconds          = " << rhsSeconds << "\n";
        os << "cudaGmgSetupSeconds        = " << setupSeconds << "\n";
        os << "cudaGmgSolveTotalSeconds   = " << solveTotalSeconds << "\n";
        os << "cudaGmgSolveAvgSeconds     = " << solveAvgSeconds << "\n";
        os << "cudaGmgSolveRepeats        = " << solveRepeats << "\n";
        os << "----------------------------------------------------------\n";
    }
};

} // namespace diagnostics
} // namespace memoirs
