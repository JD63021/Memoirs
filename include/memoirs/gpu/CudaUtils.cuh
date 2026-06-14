#pragma once

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>
#include <string>

namespace memoirs {
namespace gpu {

inline void cuda_check(cudaError_t e, const char* file, int line, const char* expr) {
    if (e != cudaSuccess) {
        std::ostringstream oss;
        oss << "CUDA error at " << file << ":" << line
            << " expr=" << expr
            << " code=" << int(e)
            << " msg=" << cudaGetErrorString(e);
        throw std::runtime_error(oss.str());
    }
}

} // namespace gpu
} // namespace memoirs

#define MEMOIRS_CUDA_CHECK(expr) \
    ::memoirs::gpu::cuda_check((expr), __FILE__, __LINE__, #expr)

#define MEMOIRS_CUDA_KERNEL_CHECK() \
    do { \
        MEMOIRS_CUDA_CHECK(cudaGetLastError()); \
    } while (0)
