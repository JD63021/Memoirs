#pragma once

#include "memoirs/gpu/CudaUtils.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <utility>

namespace memoirs {
namespace gpu {

template <class T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(std::size_t n) {
        resize(n);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept {
        ptr_ = other.ptr_;
        n_ = other.n_;
        other.ptr_ = nullptr;
        other.n_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            release();
            ptr_ = other.ptr_;
            n_ = other.n_;
            other.ptr_ = nullptr;
            other.n_ = 0;
        }
        return *this;
    }

    ~DeviceBuffer() {
        release();
    }

    void resize(std::size_t n) {
        if (n == n_) return;
        release();
        n_ = n;
        if (n_ > 0) {
            MEMOIRS_CUDA_CHECK(cudaMalloc(&ptr_, n_ * sizeof(T)));
        }
    }

    void release() {
        if (ptr_) {
            cudaFree(ptr_);
            ptr_ = nullptr;
        }
        n_ = 0;
    }

    void zero() {
        if (ptr_ && n_ > 0) {
            MEMOIRS_CUDA_CHECK(cudaMemset(ptr_, 0, n_ * sizeof(T)));
        }
    }

    void copy_from_host(const T* h, std::size_t n) {
        resize(n);
        if (n > 0) {
            MEMOIRS_CUDA_CHECK(cudaMemcpy(ptr_, h, n * sizeof(T), cudaMemcpyHostToDevice));
        }
    }

    void copy_to_host(T* h, std::size_t n) const {
        if (n != n_) {
            throw std::runtime_error("DeviceBuffer::copy_to_host size mismatch");
        }
        if (n > 0) {
            MEMOIRS_CUDA_CHECK(cudaMemcpy(h, ptr_, n * sizeof(T), cudaMemcpyDeviceToHost));
        }
    }

    T* data() { return ptr_; }
    const T* data() const { return ptr_; }

    std::size_t size() const { return n_; }

    double mib() const {
        return double(n_) * double(sizeof(T)) / (1024.0 * 1024.0);
    }

private:
    T* ptr_ = nullptr;
    std::size_t n_ = 0;
};

} // namespace gpu
} // namespace memoirs
