#pragma once

#include "memoirs/gpu/CudaUtils.cuh"
#include "memoirs/gpu/DeviceBuffer.cuh"
#include "memoirs/operators/StructuredQ1PoissonOperator.hpp"
#include "memoirs/solvers/DenseLu.hpp"
#include "memoirs/structured/StructuredGrid3D.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace memoirs {
namespace gpu {

inline const char* cublas_status_name(cublasStatus_t s) {
    switch (s) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
        default: return "CUBLAS_STATUS_UNKNOWN";
    }
}

inline void cublas_check(cublasStatus_t s, const char* file, int line, const char* expr) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string("cuBLAS error at ") + file + ":" + std::to_string(line) +
            " expr=" + expr + " status=" + cublas_status_name(s)
        );
    }
}

#define MEMOIRS_CUBLAS_CHECK(expr) \
    ::memoirs::gpu::cublas_check((expr), __FILE__, __LINE__, #expr)

class CublasHandle {
public:
    CublasHandle() {
        MEMOIRS_CUBLAS_CHECK(cublasCreate(&h_));
        MEMOIRS_CUBLAS_CHECK(cublasSetPointerMode(h_, CUBLAS_POINTER_MODE_HOST));
    }

    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;

    CublasHandle(CublasHandle&& other) noexcept {
        h_ = other.h_;
        other.h_ = nullptr;
    }

    CublasHandle& operator=(CublasHandle&& other) noexcept {
        if (this != &other) {
            if (h_) cublasDestroy(h_);
            h_ = other.h_;
            other.h_ = nullptr;
        }
        return *this;
    }

    ~CublasHandle() {
        if (h_) cublasDestroy(h_);
    }

    cublasHandle_t get() const { return h_; }

private:
    cublasHandle_t h_ = nullptr;
};

inline cublasStatus_t cublas_dot_dispatch(
    cublasHandle_t h,
    int n,
    const double* a,
    const double* b,
    double* out
) {
    return cublasDdot(h, n, a, 1, b, 1, out);
}

inline cublasStatus_t cublas_dot_dispatch(
    cublasHandle_t h,
    int n,
    const float* a,
    const float* b,
    float* out
) {
    return cublasSdot(h, n, a, 1, b, 1, out);
}


struct Grid3DCuda {
    int nx = 0;
    int ny = 0;
    int nz = 0;
    double hx = 0.0;
    double hy = 0.0;
    double hz = 0.0;

    __host__ __device__ int n_nodes_host() const {
        return nx * ny * nz;
    }
};

__host__ inline Grid3DCuda to_cuda_grid(const memoirs::structured::StructuredGrid3D& g) {
    Grid3DCuda d;
    d.nx = g.nx;
    d.ny = g.ny;
    d.nz = g.nz;
    d.hx = g.hx();
    d.hy = g.hy();
    d.hz = g.hz();
    return d;
}

__host__ __device__ inline int g_id(const Grid3DCuda g, int i, int j, int k) {
    return (k * g.ny + j) * g.nx + i;
}

__host__ __device__ inline bool g_boundary(const Grid3DCuda g, int i, int j, int k) {
    return i == 0 || j == 0 || k == 0 ||
           i == g.nx - 1 || j == g.ny - 1 || k == g.nz - 1;
}

template <class Real>
__device__ inline Real q1_weight(const Grid3DCuda g, int di, int dj, int dk) {
    const Real hx = Real(g.hx);
    const Real hy = Real(g.hy);
    const Real hz = Real(g.hz);

    const Real cx = hy * hz / (Real(2) * hx);
    const Real cy = hx * hz / (Real(2) * hy);
    const Real cz = hx * hy / (Real(2) * hz);

    const int ai = di + 1;
    const int aj = dj + 1;
    const int ak = dk + 1;

    const Real K[3] = {Real(-0.5), Real(1.0), Real(-0.5)};
    const Real M[3] = {Real(1.0 / 3.0), Real(4.0 / 3.0), Real(1.0 / 3.0)};

    return cx * K[ai] * M[aj] * M[ak]
         + cy * M[ai] * K[aj] * M[ak]
         + cz * M[ai] * M[aj] * K[ak];
}

template <class Real>
__device__ inline Real q1_diag(const Grid3DCuda g, int i, int j, int k) {
    if (g_boundary(g, i, j, k)) return Real(1);
    return q1_weight<Real>(g, 0, 0, 0);
}

template <class Real>
__global__ void kernel_fill_exact_sin(Grid3DCuda g, Real* x) {
    const int n = g.n_nodes_host();
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= n) return;

    const int i = q % g.nx;
    const int tmp = q / g.nx;
    const int j = tmp % g.ny;
    const int k = tmp / g.ny;

    const double pi = 3.141592653589793238462643383279502884;
    const double xx = double(i) * g.hx;
    const double yy = double(j) * g.hy;
    const double zz = double(k) * g.hz;

    x[q] = Real(sin(pi * xx) * sin(pi * yy) * sin(pi * zz));
}

template <class Real>
__global__ void kernel_apply_q1(Grid3DCuda g, const Real* x, Real* y) {
    const int n = g.n_nodes_host();
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= n) return;

    const int i = q % g.nx;
    const int tmp = q / g.nx;
    const int j = tmp % g.ny;
    const int k = tmp / g.ny;

    if (g_boundary(g, i, j, k)) {
        y[q] = x[q];
        return;
    }

    Real sum = Real(0);

    for (int dk = -1; dk <= 1; ++dk) {
        const int kk = k + dk;
        for (int dj = -1; dj <= 1; ++dj) {
            const int jj = j + dj;
            for (int di = -1; di <= 1; ++di) {
                const int ii = i + di;

                // Symmetric strong Dirichlet elimination.
                // Boundary columns are removed from interior rows.
                if (g_boundary(g, ii, jj, kk)) continue;

                sum += q1_weight<Real>(g, di, dj, dk) * x[g_id(g, ii, jj, kk)];
            }
        }
    }

    y[q] = sum;
}

template <class Real>
__global__ void kernel_subtract(int n, const Real* a, const Real* b, Real* r) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q < n) r[q] = a[q] - b[q];
}

template <class Real>
__global__ void kernel_copy(int n, const Real* a, Real* b) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q < n) b[q] = a[q];
}

template <class Real>
__global__ void kernel_zero(int n, Real* x) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q < n) x[q] = Real(0);
}

template <class Real>
__global__ void kernel_weighted_jacobi_update(Grid3DCuda g,
                                              Real omega,
                                              const Real* rhs,
                                              const Real* Ax,
                                              Real* x) {
    const int n = g.n_nodes_host();
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= n) return;

    const int i = q % g.nx;
    const int tmp = q / g.nx;
    const int j = tmp % g.ny;
    const int k = tmp / g.ny;

    const Real invD = Real(1) / q1_diag<Real>(g, i, j, k);
    x[q] += omega * invD * (rhs[q] - Ax[q]);
}

template <class Real>
__global__ void kernel_restrict_full_weighting(Grid3DCuda fg,
                                               Grid3DCuda cg,
                                               const Real* rf,
                                               Real* rc) {
    const int nc = cg.n_nodes_host();
    const int qc = blockIdx.x * blockDim.x + threadIdx.x;
    if (qc >= nc) return;

    const int I = qc % cg.nx;
    const int tmp = qc / cg.nx;
    const int J = tmp % cg.ny;
    const int K = tmp / cg.ny;

    if (g_boundary(cg, I, J, K)) {
        rc[qc] = Real(0);
        return;
    }

    const Real w1[3] = {Real(0.25), Real(0.5), Real(0.25)};

    const int ic = 2 * I;
    const int jc = 2 * J;
    const int kc = 2 * K;

    Real sum = Real(0);

    for (int dk = -1; dk <= 1; ++dk) {
        for (int dj = -1; dj <= 1; ++dj) {
            for (int di = -1; di <= 1; ++di) {
                const int i = ic + di;
                const int j = jc + dj;
                const int k = kc + dk;
                const Real w = w1[di + 1] * w1[dj + 1] * w1[dk + 1];
                sum += w * rf[g_id(fg, i, j, k)];
            }
        }
    }

    rc[qc] = sum;
}

template <class Real>
__global__ void kernel_prolongate_trilinear_add(Grid3DCuda cg,
                                                Grid3DCuda fg,
                                                const Real* ec,
                                                Real* xf) {
    const int nf = fg.n_nodes_host();
    const int qf = blockIdx.x * blockDim.x + threadIdx.x;
    if (qf >= nf) return;

    const int i = qf % fg.nx;
    const int tmp = qf / fg.nx;
    const int j = tmp % fg.ny;
    const int k = tmp / fg.ny;

    if (g_boundary(fg, i, j, k)) return;

    const int I0 = i / 2;
    const int J0 = j / 2;
    const int K0 = k / 2;

    const int oi = i & 1;
    const int oj = j & 1;
    const int ok = k & 1;

    Real val = Real(0);

    for (int dk = 0; dk <= ok; ++dk) {
        const int K = K0 + dk;
        const Real wk = ok ? Real(0.5) : Real(1);

        for (int dj = 0; dj <= oj; ++dj) {
            const int J = J0 + dj;
            const Real wj = oj ? Real(0.5) : Real(1);

            for (int di = 0; di <= oi; ++di) {
                const int I = I0 + di;
                const Real wi = oi ? Real(0.5) : Real(1);

                val += wi * wj * wk * ec[g_id(cg, I, J, K)];
            }
        }
    }

    xf[qf] += val;
}


template <class Real>
__global__ void kernel_dense_lu_solve_serial(int n,
                                             const Real* lu,
                                             const int* piv,
                                             const Real* rhs,
                                             Real* x) {
    // Tiny coarse solve. For the intended 2x2x2 coarse elements,
    // n = 3x3x3 = 27. A single CUDA thread is enough and avoids any
    // host transfer/synchronization inside the V-cycle.
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    for (int i = 0; i < n; ++i) {
        x[i] = rhs[i];
    }

    // Apply row pivots: PA = LU, so solve LU x = P rhs.
    for (int k = 0; k < n; ++k) {
        const int p = piv[k];
        if (p != k) {
            const Real tmp = x[k];
            x[k] = x[p];
            x[p] = tmp;
        }
    }

    // Forward solve L y = P rhs. L has implicit unit diagonal.
    for (int i = 0; i < n; ++i) {
        Real sum = x[i];

        for (int j = 0; j < i; ++j) {
            sum -= lu[i * n + j] * x[j];
        }

        x[i] = sum;
    }

    // Backward solve U x = y.
    for (int i = n - 1; i >= 0; --i) {
        Real sum = x[i];

        for (int j = i + 1; j < n; ++j) {
            sum -= lu[i * n + j] * x[j];
        }

        x[i] = sum / lu[i * n + i];
    }
}

template <class Real>
__global__ void kernel_dot_reduce(int n, const Real* a, const Real* b, double* partial) {
    extern __shared__ double sh[];

    const int tid = threadIdx.x;
    const int q = blockIdx.x * blockDim.x + threadIdx.x;

    double v = 0.0;
    if (q < n) {
        v = double(a[q]) * double(b[q]);
    }

    sh[tid] = v;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sh[tid] += sh[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial[blockIdx.x] = sh[0];
    }
}

template <class Real>
__global__ void kernel_pcg_update_x_r(int n,
                                      Real alpha,
                                      Real* x,
                                      Real* r,
                                      const Real* p,
                                      const Real* Ap) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q < n) {
        x[q] += alpha * p[q];
        r[q] -= alpha * Ap[q];
    }
}

template <class Real>
__global__ void kernel_pcg_update_p(int n,
                                    Real beta,
                                    const Real* z,
                                    Real* p) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q < n) {
        p[q] = z[q] + beta * p[q];
    }
}

inline int div_up(int n, int block) {
    return (n + block - 1) / block;
}

template <class Real>
class StructuredQ1GpuOperator {
public:
    explicit StructuredQ1GpuOperator(const memoirs::structured::StructuredGrid3D& g)
        : grid_(g), dg_(to_cuda_grid(g)) {}

    int size() const { return grid_.n_nodes(); }
    const memoirs::structured::StructuredGrid3D& grid() const { return grid_; }
    Grid3DCuda cuda_grid() const { return dg_; }

    void apply(const Real* x, Real* y) const {
        const int n = size();
        const int block = 256;
        kernel_apply_q1<Real><<<div_up(n, block), block>>>(dg_, x, y);
        MEMOIRS_CUDA_KERNEL_CHECK();
    }

    void fill_exact_sin(Real* x) const {
        const int n = size();
        const int block = 256;
        kernel_fill_exact_sin<Real><<<div_up(n, block), block>>>(dg_, x);
        MEMOIRS_CUDA_KERNEL_CHECK();
    }

private:
    memoirs::structured::StructuredGrid3D grid_;
    Grid3DCuda dg_;
};

template <class Real>
class GpuDotWorkspace {
public:
    // Kept for API compatibility with the earlier partial-reduction workspace.
    // cuBLAS performs the reduction on the GPU and returns only one scalar.
    void resize_for_n(int) {}

    double dot(int n, const Real* a, const Real* b) {
        Real out = Real(0);
        MEMOIRS_CUBLAS_CHECK(cublas_dot_dispatch(handle_.get(), n, a, b, &out));
        return double(out);
    }

private:
    CublasHandle handle_;
};

template <class Real>
struct StructuredGmgCudaOptions {
    int preSmooth = 2;
    int postSmooth = 2;
    Real omega = Real(0.70);
    int coarseMaxDofs = 256;
    int maxLevels = 32;
    int verbose = 1;
};

template <class Real>
struct StructuredGmgCudaLevel {
    memoirs::structured::StructuredGrid3D grid;
    Grid3DCuda dg;
    DeviceBuffer<Real> x;
    DeviceBuffer<Real> rhs;
    DeviceBuffer<Real> ax;
    DeviceBuffer<Real> residual;

    StructuredGmgCudaLevel(const memoirs::structured::StructuredGrid3D& g)
        : grid(g), dg(to_cuda_grid(g)) {
        const std::size_t n = g.size();
        x.resize(n);
        rhs.resize(n);
        ax.resize(n);
        residual.resize(n);
    }

    StructuredGmgCudaLevel(StructuredGmgCudaLevel&&) noexcept = default;
    StructuredGmgCudaLevel& operator=(StructuredGmgCudaLevel&&) noexcept = default;

    StructuredGmgCudaLevel(const StructuredGmgCudaLevel&) = delete;
    StructuredGmgCudaLevel& operator=(const StructuredGmgCudaLevel&) = delete;
};

template <class Real>
class StructuredGmgCudaPreconditioner {
public:
    using Grid = memoirs::structured::StructuredGrid3D;

    void setup(const Grid& finest, const StructuredGmgCudaOptions<Real>& opt) {
        opt_ = opt;
        levels_.clear();

        Grid g = finest;
        levels_.emplace_back(g);

        while (int(levels_.size()) < opt_.maxLevels &&
               g.n_nodes() > opt_.coarseMaxDofs &&
               g.can_coarsen_by_2()) {
            g = g.coarsened_by_2();
            levels_.emplace_back(g);
        }

        const Grid& cg = levels_.back().grid;
        const int nc = cg.n_nodes();

        memoirs::operators::StructuredQ1PoissonOperator<Real> cpuCoarseOp(cg);
        std::vector<Real> Ac = cpuCoarseOp.build_dense_matrix();
        coarseLu_.factor(nc, Ac);

        dCoarseLu_.copy_from_host(coarseLu_.lu_storage().data(),
                                  coarseLu_.lu_storage().size());
        dCoarsePiv_.copy_from_host(coarseLu_.pivots().data(),
                                   coarseLu_.pivots().size());

        if (opt_.verbose) print_hierarchy(std::cout);
    }

    void print_hierarchy(std::ostream& os) const {
        os << "--------------- CUDA structured GMG hierarchy ---------------\n";
        os << "gmgCudaLevels             = " << levels_.size() << "\n";
        os << "gmgCudaPreSmooth          = " << opt_.preSmooth << "\n";
        os << "gmgCudaPostSmooth         = " << opt_.postSmooth << "\n";
        os << "gmgCudaJacobiOmega        = " << double(opt_.omega) << "\n";
        os << "gmgCudaCoarseMaxDofs      = " << opt_.coarseMaxDofs << "\n";

        double vectorMiB = 0.0;

        for (std::size_t l = 0; l < levels_.size(); ++l) {
            const auto& lev = levels_[l];
            const double levMiB =
                4.0 * double(lev.grid.n_nodes()) * double(sizeof(Real)) / (1024.0 * 1024.0);
            vectorMiB += levMiB;

            os << "gmgCudaLevel[" << l << "]          = "
               << lev.grid.label()
               << " dofs=" << lev.grid.n_nodes()
               << " levelVec4MiB=" << levMiB
               << "\n";
        }

        const int nc = levels_.back().grid.n_nodes();
        const double coarseDenseMiB =
            double(nc) * double(nc) * double(sizeof(Real)) / (1024.0 * 1024.0);

        os << "gmgCudaCoarseSolver       = dense_lu_device_solve_cpu_factor\n";
        os << "gmgCudaCoarseDofs         = " << nc << "\n";
        os << "gmgCudaCoarseDenseMiB     = " << coarseDenseMiB << "\n";
        os << "gmgCudaLevelVectorMiB     = " << vectorMiB << "\n";
        os << "------------------------------------------------------------\n";
    }

    void apply(const Real* d_rhs, Real* d_z) {
        if (levels_.empty()) {
            throw std::runtime_error("StructuredGmgCudaPreconditioner::apply before setup");
        }

        auto& f = levels_.front();
        const int n = f.grid.n_nodes();

        MEMOIRS_CUDA_CHECK(cudaMemcpy(f.rhs.data(), d_rhs,
                                      static_cast<std::size_t>(n) * sizeof(Real),
                                      cudaMemcpyDeviceToDevice));
        f.x.zero();

        vcycle(0);

        MEMOIRS_CUDA_CHECK(cudaMemcpy(d_z, f.x.data(),
                                      static_cast<std::size_t>(n) * sizeof(Real),
                                      cudaMemcpyDeviceToDevice));
    }

private:
    void apply_level(std::size_t l, const Real* x, Real* y) {
        auto& lev = levels_[l];
        const int n = lev.grid.n_nodes();
        const int block = 256;
        kernel_apply_q1<Real><<<div_up(n, block), block>>>(lev.dg, x, y);
        MEMOIRS_CUDA_KERNEL_CHECK();
    }

    void smooth(std::size_t l, int sweeps) {
        auto& lev = levels_[l];
        const int n = lev.grid.n_nodes();
        const int block = 256;

        for (int s = 0; s < sweeps; ++s) {
            apply_level(l, lev.x.data(), lev.ax.data());

            kernel_weighted_jacobi_update<Real>
                <<<div_up(n, block), block>>>
                (lev.dg, opt_.omega, lev.rhs.data(), lev.ax.data(), lev.x.data());
            MEMOIRS_CUDA_KERNEL_CHECK();
        }
    }

    void residual(std::size_t l) {
        auto& lev = levels_[l];
        const int n = lev.grid.n_nodes();
        const int block = 256;

        apply_level(l, lev.x.data(), lev.ax.data());

        kernel_subtract<Real>
            <<<div_up(n, block), block>>>
            (n, lev.rhs.data(), lev.ax.data(), lev.residual.data());
        MEMOIRS_CUDA_KERNEL_CHECK();
    }

    void restrict_to_next(std::size_t l) {
        auto& fine = levels_[l];
        auto& coarse = levels_[l + 1];

        const int nc = coarse.grid.n_nodes();
        const int block = 256;

        kernel_restrict_full_weighting<Real>
            <<<div_up(nc, block), block>>>
            (fine.dg, coarse.dg, fine.residual.data(), coarse.rhs.data());
        MEMOIRS_CUDA_KERNEL_CHECK();

        coarse.x.zero();
    }

    void prolongate_add(std::size_t l) {
        auto& fine = levels_[l];
        auto& coarse = levels_[l + 1];

        const int nf = fine.grid.n_nodes();
        const int block = 256;

        kernel_prolongate_trilinear_add<Real>
            <<<div_up(nf, block), block>>>
            (coarse.dg, fine.dg, coarse.x.data(), fine.x.data());
        MEMOIRS_CUDA_KERNEL_CHECK();
    }

    void coarse_solve_device() {
        auto& c = levels_.back();
        const int nc = c.grid.n_nodes();

        kernel_dense_lu_solve_serial<Real>
            <<<1, 1>>>
            (nc, dCoarseLu_.data(), dCoarsePiv_.data(), c.rhs.data(), c.x.data());
        MEMOIRS_CUDA_KERNEL_CHECK();
    }

    void vcycle(std::size_t l) {
        if (l + 1 == levels_.size()) {
            coarse_solve_device();
            return;
        }

        smooth(l, opt_.preSmooth);
        residual(l);
        restrict_to_next(l);
        vcycle(l + 1);
        prolongate_add(l);
        smooth(l, opt_.postSmooth);
    }

    StructuredGmgCudaOptions<Real> opt_;
    std::vector<StructuredGmgCudaLevel<Real>> levels_;

    // CPU factorization is still used during setup, but the factor and pivots
    // are copied once to GPU. The V-cycle coarse solve itself is device-side.
    memoirs::solvers::DenseLu<Real> coarseLu_;
    DeviceBuffer<Real> dCoarseLu_;
    DeviceBuffer<int> dCoarsePiv_;
};

template <class Real>
struct CudaPcgReport {
    int iterations = 0;
    int converged = 0;
    int breakdown = 0;
    double initialResidual = 0.0;
    double finalResidual = 0.0;
    double finalRelativeResidual = 0.0;
};

template <class Real>
CudaPcgReport<Real> pcg_solve_cuda(
    const StructuredQ1GpuOperator<Real>& A,
    StructuredGmgCudaPreconditioner<Real>& M,
    const Real* d_b,
    Real* d_x,
    Real tol,
    int maxit,
    int printEvery
) {
    const int n = A.size();
    const int block = 256;

    DeviceBuffer<Real> d_Ax(static_cast<std::size_t>(n));
    DeviceBuffer<Real> d_r(static_cast<std::size_t>(n));
    DeviceBuffer<Real> d_z(static_cast<std::size_t>(n));
    DeviceBuffer<Real> d_p(static_cast<std::size_t>(n));
    DeviceBuffer<Real> d_Ap(static_cast<std::size_t>(n));

    GpuDotWorkspace<Real> dots;
    dots.resize_for_n(n);

    d_Ax.zero();
    A.apply(d_x, d_Ax.data());

    kernel_subtract<Real>
        <<<div_up(n, block), block>>>
        (n, d_b, d_Ax.data(), d_r.data());
    MEMOIRS_CUDA_KERNEL_CHECK();

    const double bnorm = std::max(std::sqrt(dots.dot(n, d_b, d_b)), 1.0);
    double rnorm = std::sqrt(dots.dot(n, d_r.data(), d_r.data()));

    CudaPcgReport<Real> rep;
    rep.initialResidual = rnorm;
    rep.finalResidual = rnorm;
    rep.finalRelativeResidual = rnorm / bnorm;

    if (rep.finalRelativeResidual <= double(tol)) {
        rep.converged = 1;
        return rep;
    }

    M.apply(d_r.data(), d_z.data());

    kernel_copy<Real>
        <<<div_up(n, block), block>>>
        (n, d_z.data(), d_p.data());
    MEMOIRS_CUDA_KERNEL_CHECK();

    double rz = dots.dot(n, d_r.data(), d_z.data());
    if (!(std::abs(rz) > 0.0)) {
        rep.breakdown = 1;
        return rep;
    }

    for (int it = 1; it <= maxit; ++it) {
        A.apply(d_p.data(), d_Ap.data());

        const double pAp = dots.dot(n, d_p.data(), d_Ap.data());
        if (!(pAp > 0.0)) {
            rep.breakdown = 1;
            rep.iterations = it - 1;
            return rep;
        }

        const Real alpha = Real(rz / pAp);

        kernel_pcg_update_x_r<Real>
            <<<div_up(n, block), block>>>
            (n, alpha, d_x, d_r.data(), d_p.data(), d_Ap.data());
        MEMOIRS_CUDA_KERNEL_CHECK();

        rnorm = std::sqrt(dots.dot(n, d_r.data(), d_r.data()));

        rep.iterations = it;
        rep.finalResidual = rnorm;
        rep.finalRelativeResidual = rnorm / bnorm;

        if (printEvery > 0 && (it == 1 || it % printEvery == 0)) {
            std::cout << "cuda pcg it=" << it
                      << " rel=" << rep.finalRelativeResidual
                      << "\n";
        }

        if (rep.finalRelativeResidual <= double(tol)) {
            rep.converged = 1;
            return rep;
        }

        M.apply(d_r.data(), d_z.data());

        const double rzNew = dots.dot(n, d_r.data(), d_z.data());
        if (!(std::abs(rzNew) > 0.0)) {
            rep.breakdown = 1;
            return rep;
        }

        const Real beta = Real(rzNew / rz);

        kernel_pcg_update_p<Real>
            <<<div_up(n, block), block>>>
            (n, beta, d_z.data(), d_p.data());
        MEMOIRS_CUDA_KERNEL_CHECK();

        rz = rzNew;
    }

    return rep;
}

template <class Real>
double device_l2_diff(int n, const Real* a, const Real* b) {
    GpuDotWorkspace<Real> dots;
    dots.resize_for_n(n);

    DeviceBuffer<Real> d_tmp(static_cast<std::size_t>(n));
    const int block = 256;

    kernel_subtract<Real>
        <<<div_up(n, block), block>>>
        (n, a, b, d_tmp.data());
    MEMOIRS_CUDA_KERNEL_CHECK();

    return std::sqrt(dots.dot(n, d_tmp.data(), d_tmp.data()) / double(n));
}

template <class Real>
double device_residual_rel(const StructuredQ1GpuOperator<Real>& A,
                           const Real* d_b,
                           const Real* d_x) {
    const int n = A.size();
    const int block = 256;

    DeviceBuffer<Real> d_Ax(static_cast<std::size_t>(n));
    DeviceBuffer<Real> d_r(static_cast<std::size_t>(n));

    A.apply(d_x, d_Ax.data());

    kernel_subtract<Real>
        <<<div_up(n, block), block>>>
        (n, d_b, d_Ax.data(), d_r.data());
    MEMOIRS_CUDA_KERNEL_CHECK();

    GpuDotWorkspace<Real> dots;
    dots.resize_for_n(n);

    const double r = std::sqrt(dots.dot(n, d_r.data(), d_r.data()));
    const double b = std::max(std::sqrt(dots.dot(n, d_b, d_b)), 1.0);
    return r / b;
}

} // namespace gpu
} // namespace memoirs
