#pragma once

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <dlfcn.h>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>

namespace memoirs_diag {

class NvmlMemorySampler {
public:
    using nvmlReturn_t = int;
    using nvmlDevice_t = void*;

    struct nvmlMemory_t {
        unsigned long long total;
        unsigned long long free;
        unsigned long long used;
    };

    using nvmlInit_v2_t = nvmlReturn_t (*)();
    using nvmlShutdown_t = nvmlReturn_t (*)();
    using nvmlDeviceGetHandleByIndex_v2_t = nvmlReturn_t (*)(unsigned int, nvmlDevice_t*);
    using nvmlDeviceGetMemoryInfo_t = nvmlReturn_t (*)(nvmlDevice_t, nvmlMemory_t*);

    NvmlMemorySampler() = default;

    NvmlMemorySampler(const NvmlMemorySampler&) = delete;
    NvmlMemorySampler& operator=(const NvmlMemorySampler&) = delete;

    ~NvmlMemorySampler() {
        close();
    }

    bool open(int deviceIndex) {
        close();

        lib_ = dlopen("libnvidia-ml.so.1", RTLD_LAZY | RTLD_LOCAL);
        if (!lib_) {
            status_ = std::string("dlopen(libnvidia-ml.so.1) failed: ") + dlerror();
            return false;
        }

        nvmlInit_v2_ = reinterpret_cast<nvmlInit_v2_t>(dlsym(lib_, "nvmlInit_v2"));
        nvmlShutdown_ = reinterpret_cast<nvmlShutdown_t>(dlsym(lib_, "nvmlShutdown"));
        nvmlDeviceGetHandleByIndex_v2_ =
            reinterpret_cast<nvmlDeviceGetHandleByIndex_v2_t>(
                dlsym(lib_, "nvmlDeviceGetHandleByIndex_v2"));
        nvmlDeviceGetMemoryInfo_ =
            reinterpret_cast<nvmlDeviceGetMemoryInfo_t>(
                dlsym(lib_, "nvmlDeviceGetMemoryInfo"));

        if (!nvmlInit_v2_ || !nvmlShutdown_ ||
            !nvmlDeviceGetHandleByIndex_v2_ || !nvmlDeviceGetMemoryInfo_) {
            status_ = "NVML symbol lookup failed";
            close_library_only();
            return false;
        }

        nvmlReturn_t rc = nvmlInit_v2_();
        if (rc != 0) {
            std::ostringstream oss;
            oss << "nvmlInit_v2 failed rc=" << rc;
            status_ = oss.str();
            close_library_only();
            return false;
        }
        initialized_ = true;

        rc = nvmlDeviceGetHandleByIndex_v2_(static_cast<unsigned int>(deviceIndex), &device_);
        if (rc != 0) {
            std::ostringstream oss;
            oss << "nvmlDeviceGetHandleByIndex_v2 failed rc=" << rc
                << " deviceIndex=" << deviceIndex;
            status_ = oss.str();
            close();
            return false;
        }

        status_ = "ok";
        return true;
    }

    bool sample_used_mib(double& usedMiB) const {
        if (!initialized_ || !nvmlDeviceGetMemoryInfo_) return false;

        nvmlMemory_t mem{};
        nvmlReturn_t rc = nvmlDeviceGetMemoryInfo_(device_, &mem);
        if (rc != 0) return false;

        usedMiB = double(mem.used) / (1024.0 * 1024.0);
        return true;
    }

    void close() {
        if (initialized_ && nvmlShutdown_) {
            nvmlShutdown_();
        }
        initialized_ = false;
        device_ = nullptr;
        close_library_only();
    }

    const std::string& status() const {
        return status_;
    }

private:
    void close_library_only() {
        if (lib_) {
            dlclose(lib_);
            lib_ = nullptr;
        }

        nvmlInit_v2_ = nullptr;
        nvmlShutdown_ = nullptr;
        nvmlDeviceGetHandleByIndex_v2_ = nullptr;
        nvmlDeviceGetMemoryInfo_ = nullptr;
    }

    void* lib_ = nullptr;
    bool initialized_ = false;
    nvmlDevice_t device_ = nullptr;
    std::string status_ = "not opened";

    nvmlInit_v2_t nvmlInit_v2_ = nullptr;
    nvmlShutdown_t nvmlShutdown_ = nullptr;
    nvmlDeviceGetHandleByIndex_v2_t nvmlDeviceGetHandleByIndex_v2_ = nullptr;
    nvmlDeviceGetMemoryInfo_t nvmlDeviceGetMemoryInfo_ = nullptr;
};

struct GpuMemorySnapshot {
    bool enabled = false;
    bool available = false;
    int deviceIndex = 0;
    int intervalMs = 20;
    double baselineMiB = -1.0;
    double peakMiB = -1.0;
    double lastMiB = -1.0;
    double netPeakMiB = -1.0;
    long long samples = 0;
    std::string backend = "nvml";
    std::string status = "disabled";
};

class GpuMemoryMonitor {
public:
    GpuMemoryMonitor() = default;

    GpuMemoryMonitor(const GpuMemoryMonitor&) = delete;
    GpuMemoryMonitor& operator=(const GpuMemoryMonitor&) = delete;

    ~GpuMemoryMonitor() {
        stop();
    }

    void configure_from_environment() {
        enabled_ = env_bool("MEMOIRS_GPU_MEM_DIAG", false) ||
                   env_bool("MEMOIRS_GPU_MEMORY_DIAG", false);

        deviceIndex_ = env_int("MEMOIRS_GPU_MEM_DEVICE", 0);
        intervalMs_ = env_int("MEMOIRS_GPU_MEM_INTERVAL_MS", 20);

        if (intervalMs_ < 5) intervalMs_ = 5;
    }

    void start() {
        configure_from_environment();

        if (!enabled_) return;
        if (running_) return;

        available_ = sampler_.open(deviceIndex_);
        if (!available_) {
            std::lock_guard<std::mutex> lock(mu_);
            status_ = sampler_.status();
            return;
        }

        double first = 0.0;
        if (!sampler_.sample_used_mib(first)) {
            std::lock_guard<std::mutex> lock(mu_);
            available_ = false;
            status_ = "initial NVML memory sample failed";
            sampler_.close();
            return;
        }

        {
            std::lock_guard<std::mutex> lock(mu_);
            baselineMiB_ = first;
            peakMiB_ = first;
            lastMiB_ = first;
            samples_ = 1;
            status_ = "ok";
        }

        running_ = true;
        worker_ = std::thread([this]() {
            while (running_) {
                sample_once();
                std::this_thread::sleep_for(std::chrono::milliseconds(intervalMs_));
            }
        });
    }

    void stop() {
        if (!enabled_) return;

        if (running_) {
            sample_once();
            running_ = false;

            if (worker_.joinable()) {
                worker_.join();
            }

            sample_once();
        }

        sampler_.close();
    }

    GpuMemorySnapshot snapshot() const {
        std::lock_guard<std::mutex> lock(mu_);

        GpuMemorySnapshot s;
        s.enabled = enabled_;
        s.available = available_;
        s.deviceIndex = deviceIndex_;
        s.intervalMs = intervalMs_;
        s.baselineMiB = baselineMiB_;
        s.peakMiB = peakMiB_;
        s.lastMiB = lastMiB_;
        s.netPeakMiB =
            (baselineMiB_ >= 0.0 && peakMiB_ >= 0.0) ? (peakMiB_ - baselineMiB_) : -1.0;
        s.samples = samples_;
        s.backend = "nvml";
        s.status = status_;
        return s;
    }

    void print(std::ostream& os) const {
        GpuMemorySnapshot s = snapshot();

        if (!s.enabled) return;

        os << "--------------- gpu memory diagnostic ---------------\n";
        os << "gpuMemDiag                = 1\n";
        os << "gpuMemBackend             = " << s.backend << "\n";
        os << "gpuMemStatus              = " << s.status << "\n";
        os << "gpuMemDevice              = " << s.deviceIndex << "\n";
        os << "gpuMemIntervalMs          = " << s.intervalMs << "\n";

        if (s.available) {
            os << std::fixed << std::setprecision(3);
            os << "gpuMemBaselineMiB         = " << s.baselineMiB << "\n";
            os << "gpuMemPeakMiB             = " << s.peakMiB << "\n";
            os << "gpuMemNetPeakMiB          = " << s.netPeakMiB << "\n";
            os << "gpuMemLastMiB             = " << s.lastMiB << "\n";
            os << "gpuMemSamples             = " << s.samples << "\n";
            os.unsetf(std::ios::floatfield);
        }

        os << "-----------------------------------------------------\n";
    }

private:
    static bool env_bool(const char* name, bool defval) {
        const char* e = std::getenv(name);
        if (!e || !*e) return defval;

        std::string v(e);
        for (char& c : v) c = char(std::tolower(static_cast<unsigned char>(c)));

        return !(v == "0" || v == "false" || v == "off" || v == "no");
    }

    static int env_int(const char* name, int defval) {
        const char* e = std::getenv(name);
        if (!e || !*e) return defval;
        return std::atoi(e);
    }

    void sample_once() {
        if (!available_) return;

        double used = 0.0;
        if (!sampler_.sample_used_mib(used)) return;

        std::lock_guard<std::mutex> lock(mu_);
        lastMiB_ = used;
        if (used > peakMiB_) peakMiB_ = used;
        ++samples_;
    }

    bool enabled_ = false;
    bool available_ = false;
    int deviceIndex_ = 0;
    int intervalMs_ = 20;

    mutable std::mutex mu_;
    double baselineMiB_ = -1.0;
    double peakMiB_ = -1.0;
    double lastMiB_ = -1.0;
    long long samples_ = 0;
    std::string status_ = "disabled";

    std::atomic<bool> running_{false};
    std::thread worker_;

    NvmlMemorySampler sampler_;
};

} // namespace memoirs_diag

using memoirs_diag::GpuMemoryMonitor;
