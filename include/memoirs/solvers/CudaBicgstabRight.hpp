#pragma once

#include "memoirs/gpu/StructuredGmgCuda.cuh"

#include <cmath>
#include <iostream>

namespace memoirs {
namespace gpu {

template <class Real>
__global__ void kernel_bicgstab_combination3(int n,
                                             Real a,
                                             const Real* x,
                                             Real b,
                                             const Real* y,
                                             Real c,
                                             const Real* z,
                                             Real* out) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q < n) out[q] = a * x[q] + b * y[q] + c * z[q];
}

template <class Real>
__global__ void kernel_bicgstab_axpby(int n,
                                      Real a,
                                      const Real* x,
                                      Real b,
                                      Real* y) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q < n) y[q] = a * x[q] + b * y[q];
}

template <class Real>
struct CudaRightBicgstabReport {
    int iterations = 0;
    int converged = 0;
    int breakdown = 0;
    double initialResidual = 0.0;
    double finalResidual = 0.0;
    double finalRelativeResidual = 0.0;
};

template <class Real, class Operator, class RightPreconditioner>
CudaRightBicgstabReport<Real> bicgstab_right_prec_cuda(
    const Operator& A,
    RightPreconditioner& M,
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
    DeviceBuffer<Real> d_rhat(static_cast<std::size_t>(n));
    DeviceBuffer<Real> d_p(static_cast<std::size_t>(n));
    DeviceBuffer<Real> d_v(static_cast<std::size_t>(n));
    DeviceBuffer<Real> d_s(static_cast<std::size_t>(n));
    DeviceBuffer<Real> d_t(static_cast<std::size_t>(n));
    DeviceBuffer<Real> d_phat(static_cast<std::size_t>(n));
    DeviceBuffer<Real> d_shat(static_cast<std::size_t>(n));

    GpuDotWorkspace<Real> dots;
    dots.resize_for_n(n);

    A.apply(d_x, d_Ax.data());

    kernel_subtract<Real>
        <<<div_up(n, block), block>>>
        (n, d_b, d_Ax.data(), d_r.data());
    MEMOIRS_CUDA_KERNEL_CHECK();

    kernel_copy<Real>
        <<<div_up(n, block), block>>>
        (n, d_r.data(), d_rhat.data());
    MEMOIRS_CUDA_KERNEL_CHECK();

    d_p.zero();
    d_v.zero();

    const double bnorm = std::max(std::sqrt(dots.dot(n, d_b, d_b)), 1.0);
    double rnorm = std::sqrt(dots.dot(n, d_r.data(), d_r.data()));

    CudaRightBicgstabReport<Real> rep;
    rep.initialResidual = rnorm;
    rep.finalResidual = rnorm;
    rep.finalRelativeResidual = rnorm / bnorm;

    if (rep.finalRelativeResidual <= double(tol)) {
        rep.converged = 1;
        return rep;
    }

    double rhoOld = 1.0;
    double alpha = 1.0;
    double omega = 1.0;

    for (int it = 1; it <= maxit; ++it) {
        const double rhoNew = dots.dot(n, d_rhat.data(), d_r.data());

        if (!(std::abs(rhoNew) > 0.0)) {
            rep.breakdown = 1;
            rep.iterations = it - 1;
            return rep;
        }

        const double beta = (rhoNew / rhoOld) * (alpha / omega);

        kernel_bicgstab_combination3<Real>
            <<<div_up(n, block), block>>>
            (n,
             Real(1), d_r.data(),
             Real(beta), d_p.data(),
             Real(-beta * omega), d_v.data(),
             d_p.data());
        MEMOIRS_CUDA_KERNEL_CHECK();

        M.apply(d_p.data(), d_phat.data());

        A.apply(d_phat.data(), d_v.data());

        const double rhatv = dots.dot(n, d_rhat.data(), d_v.data());
        if (!(std::abs(rhatv) > 0.0)) {
            rep.breakdown = 1;
            rep.iterations = it - 1;
            return rep;
        }

        alpha = rhoNew / rhatv;

        kernel_bicgstab_combination3<Real>
            <<<div_up(n, block), block>>>
            (n,
             Real(1), d_r.data(),
             Real(-alpha), d_v.data(),
             Real(0), d_v.data(),
             d_s.data());
        MEMOIRS_CUDA_KERNEL_CHECK();

        const double snorm = std::sqrt(dots.dot(n, d_s.data(), d_s.data()));
        if (snorm / bnorm <= double(tol)) {
            kernel_bicgstab_axpby<Real>
                <<<div_up(n, block), block>>>
                (n, Real(alpha), d_phat.data(), Real(1), d_x);
            MEMOIRS_CUDA_KERNEL_CHECK();

            rep.iterations = it;
            rep.converged = 1;
            rep.finalResidual = snorm;
            rep.finalRelativeResidual = snorm / bnorm;
            return rep;
        }

        M.apply(d_s.data(), d_shat.data());

        A.apply(d_shat.data(), d_t.data());

        const double tt = dots.dot(n, d_t.data(), d_t.data());
        if (!(tt > 0.0)) {
            rep.breakdown = 1;
            rep.iterations = it - 1;
            return rep;
        }

        omega = dots.dot(n, d_t.data(), d_s.data()) / tt;

        if (!(std::abs(omega) > 0.0)) {
            rep.breakdown = 1;
            rep.iterations = it - 1;
            return rep;
        }

        kernel_bicgstab_combination3<Real>
            <<<div_up(n, block), block>>>
            (n,
             Real(1), d_x,
             Real(alpha), d_phat.data(),
             Real(omega), d_shat.data(),
             d_x);
        MEMOIRS_CUDA_KERNEL_CHECK();

        kernel_bicgstab_combination3<Real>
            <<<div_up(n, block), block>>>
            (n,
             Real(1), d_s.data(),
             Real(-omega), d_t.data(),
             Real(0), d_t.data(),
             d_r.data());
        MEMOIRS_CUDA_KERNEL_CHECK();

        rnorm = std::sqrt(dots.dot(n, d_r.data(), d_r.data()));

        rep.iterations = it;
        rep.finalResidual = rnorm;
        rep.finalRelativeResidual = rnorm / bnorm;

        if (printEvery > 0 && (it == 1 || it % printEvery == 0)) {
            std::cout << "cuda right-bicgstab it=" << it
                      << " rel=" << rep.finalRelativeResidual
                      << "\n";
        }

        if (rep.finalRelativeResidual <= double(tol)) {
            rep.converged = 1;
            return rep;
        }

        rhoOld = rhoNew;
    }

    return rep;
}

} // namespace gpu
} // namespace memoirs
