#ifndef MEMOIRS_PRECISION_float
#ifndef MEMOIRS_PRECISION_double
#define MEMOIRS_PRECISION_double
#endif
#endif

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// Avoid pulling HYPRE/MPI headers into nvcc compilation through common headers.
#ifdef MEMOIRS_USE_HYPRE
#define MEMOIRS_RESTORE_USE_HYPRE_AFTER_Q2Q1_CUDA_AUDIT 1
#undef MEMOIRS_USE_HYPRE
#endif

#include "memoirs/sections/00_common.hpp"
#include "memoirs/sections/01_options.hpp"
#include "memoirs/sections/02_polymesh.hpp"
#include "memoirs/sections/03_cell_topology.hpp"
#include "memoirs/sections/06_sparse_rows.hpp"
#include "memoirs/sections/16_q2q1_structured.hpp"
#include "memoirs/sections/18_q2q1_nse_cuda_audit.hpp"

#ifdef MEMOIRS_RESTORE_USE_HYPRE_AFTER_Q2Q1_CUDA_AUDIT
#define MEMOIRS_USE_HYPRE 1
#undef MEMOIRS_RESTORE_USE_HYPRE_AFTER_Q2Q1_CUDA_AUDIT
#endif

namespace {

#define Q2Q1_CUDA_CHECK(expr)                                                        \
    do {                                                                             \
        cudaError_t _e = (expr);                                                     \
        if (_e != cudaSuccess) {                                                     \
            throw std::runtime_error(std::string("CUDA error: ") + #expr +           \
                                     " -> " + cudaGetErrorString(_e));               \
        }                                                                            \
    } while (0)

static inline double q2q1_cuda_elapsed_seconds(cudaEvent_t a, cudaEvent_t b) {
    float ms = 0.0f;
    Q2Q1_CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    return 0.001 * (double)ms;
}

__global__ void q2q1_zero_values_rhs_kernel(
    Real* vals,
    long long nnz,
    Real* rhs,
    long long nRows
) {
    const long long tid = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    const long long stride = (long long)blockDim.x * (long long)gridDim.x;

    for (long long i = tid; i < nnz; i += stride) {
        vals[i] = Real(0);
    }

    for (long long i = tid; i < nRows; i += stride) {
        rhs[i] = Real(1);
    }
}

__global__ void q2q1_set_identity_diag_kernel(
    Real* vals,
    const int* diagSlots,
    long long nRows
) {
    const long long tid = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    const long long stride = (long long)blockDim.x * (long long)gridDim.x;

    for (long long r = tid; r < nRows; r += stride) {
        const int slot = diagSlots[r];
        if (slot >= 0) vals[slot] = Real(1);
    }
}

__global__ void q2q1_zero_values_zero_rhs_kernel(
    Real* vals,
    long long nnz,
    Real* rhs,
    long long nRows
) {
    const long long tid = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    const long long stride = (long long)blockDim.x * (long long)gridDim.x;

    for (long long i = tid; i < nnz; i += stride) {
        vals[i] = Real(0);
    }

    for (long long i = tid; i < nRows; i += stride) {
        rhs[i] = Real(0);
    }
}

} // namespace

Q2Q1CudaIdentityFillReport q2q1_cuda_fill_identity_values_rhs(
    const Q2Q1StructuredGrid& g,
    SparseRows& A,
    std::vector<Real>& b
) {
    Q2Q1CudaIdentityFillReport rep;
    rep.rows = g.nRows;
    rep.nnz = (long long)A.flatVals.size();

    if (!A.fixedPattern || !A.pattern) {
        throw std::runtime_error("q2q1_cuda_fill_identity_values_rhs requires fixed-pattern SparseRows.");
    }
    if ((long long)sparse_nrows(A) != g.nRows) {
        throw std::runtime_error("q2q1_cuda_fill_identity_values_rhs row mismatch.");
    }
    if (rep.nnz <= 0 || rep.rows <= 0) {
        throw std::runtime_error("q2q1_cuda_fill_identity_values_rhs empty matrix.");
    }

    std::vector<int> hDiagSlots((std::size_t)rep.rows, -1);
    long long missingDiag = 0;
    for (int r = 0; r < (int)rep.rows; ++r) {
        const int slot = sparse_lookup_flat_slot(A, r, r);
        hDiagSlots[(std::size_t)r] = slot;
        if (slot < 0) ++missingDiag;
    }

    if (missingDiag != 0) {
        throw std::runtime_error("q2q1_cuda_fill_identity_values_rhs found missing diagonal slots.");
    }

    Real* dVals = nullptr;
    Real* dRhs = nullptr;
    int* dDiagSlots = nullptr;

    cudaEvent_t t0, t1, t2, t3;
    Q2Q1_CUDA_CHECK(cudaEventCreate(&t0));
    Q2Q1_CUDA_CHECK(cudaEventCreate(&t1));
    Q2Q1_CUDA_CHECK(cudaEventCreate(&t2));
    Q2Q1_CUDA_CHECK(cudaEventCreate(&t3));

    try {
        Q2Q1_CUDA_CHECK(cudaMalloc((void**)&dVals, sizeof(Real) * (std::size_t)rep.nnz));
        Q2Q1_CUDA_CHECK(cudaMalloc((void**)&dRhs, sizeof(Real) * (std::size_t)rep.rows));
        Q2Q1_CUDA_CHECK(cudaMalloc((void**)&dDiagSlots, sizeof(int) * (std::size_t)rep.rows));

        Q2Q1_CUDA_CHECK(cudaMemcpy(
            dDiagSlots,
            hDiagSlots.data(),
            sizeof(int) * (std::size_t)rep.rows,
            cudaMemcpyHostToDevice
        ));

        const int block = 256;
        const int gridVals = (int)std::min<long long>(65535LL, std::max<long long>(1LL, (std::max(rep.nnz, rep.rows) + block - 1) / block));
        const int gridRows = (int)std::min<long long>(65535LL, std::max<long long>(1LL, (rep.rows + block - 1) / block));

        Q2Q1_CUDA_CHECK(cudaEventRecord(t0));
        q2q1_zero_values_rhs_kernel<<<gridVals, block>>>(dVals, rep.nnz, dRhs, rep.rows);
        Q2Q1_CUDA_CHECK(cudaGetLastError());
        Q2Q1_CUDA_CHECK(cudaEventRecord(t1));

        q2q1_set_identity_diag_kernel<<<gridRows, block>>>(dVals, dDiagSlots, rep.rows);
        Q2Q1_CUDA_CHECK(cudaGetLastError());
        Q2Q1_CUDA_CHECK(cudaEventRecord(t2));

        A.flatVals.assign((std::size_t)rep.nnz, Real(0));
        b.assign((std::size_t)rep.rows, Real(0));

        Q2Q1_CUDA_CHECK(cudaMemcpy(
            A.flatVals.data(),
            dVals,
            sizeof(Real) * (std::size_t)rep.nnz,
            cudaMemcpyDeviceToHost
        ));
        Q2Q1_CUDA_CHECK(cudaMemcpy(
            b.data(),
            dRhs,
            sizeof(Real) * (std::size_t)rep.rows,
            cudaMemcpyDeviceToHost
        ));
        Q2Q1_CUDA_CHECK(cudaEventRecord(t3));
        Q2Q1_CUDA_CHECK(cudaEventSynchronize(t3));

        rep.zeroSeconds = q2q1_cuda_elapsed_seconds(t0, t1);
        rep.diagSeconds = q2q1_cuda_elapsed_seconds(t1, t2);
        rep.copyBackSeconds = q2q1_cuda_elapsed_seconds(t2, t3);
        rep.totalSeconds = q2q1_cuda_elapsed_seconds(t0, t3);

        Q2Q1_CUDA_CHECK(cudaFree(dVals)); dVals = nullptr;
        Q2Q1_CUDA_CHECK(cudaFree(dRhs)); dRhs = nullptr;
        Q2Q1_CUDA_CHECK(cudaFree(dDiagSlots)); dDiagSlots = nullptr;
        Q2Q1_CUDA_CHECK(cudaEventDestroy(t0));
        Q2Q1_CUDA_CHECK(cudaEventDestroy(t1));
        Q2Q1_CUDA_CHECK(cudaEventDestroy(t2));
        Q2Q1_CUDA_CHECK(cudaEventDestroy(t3));

        return rep;
    } catch (...) {
        if (dVals) cudaFree(dVals);
        if (dRhs) cudaFree(dRhs);
        if (dDiagSlots) cudaFree(dDiagSlots);
        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
        cudaEventDestroy(t2);
        cudaEventDestroy(t3);
        throw;
    }
}


namespace {

__device__ __forceinline__ long long q2q1_d_vnode_id(int nv1, int I, int J, int K) {
    return (long long)I + (long long)nv1 * ((long long)J + (long long)nv1 * (long long)K);
}

__device__ __forceinline__ long long q2q1_d_pnode_id(int np1, int i, int j, int k) {
    return (long long)i + (long long)np1 * ((long long)j + (long long)np1 * (long long)k);
}

__device__ __forceinline__ long long q2q1_d_ux(long long v) { return v; }
__device__ __forceinline__ long long q2q1_d_uy(long long nV, long long v) { return nV + v; }
__device__ __forceinline__ long long q2q1_d_uz(long long nV, long long v) { return 2LL * nV + v; }
__device__ __forceinline__ long long q2q1_d_pr(long long nV, long long p) { return 3LL * nV + p; }

__device__ __forceinline__ double q2q1_d_dot(double ax, double ay, double az,
                                             double bx, double by, double bz) {
    return ax * bx + ay * by + az * bz;
}

__device__ __forceinline__ void q2q1_d_q2_1d(double x, double L[3], double dL[3], double ddL[3]) {
    L[0] = 0.5 * x * (x - 1.0);
    L[1] = 1.0 - x * x;
    L[2] = 0.5 * x * (x + 1.0);

    dL[0] = x - 0.5;
    dL[1] = -2.0 * x;
    dL[2] = x + 0.5;

    ddL[0] = 1.0;
    ddL[1] = -2.0;
    ddL[2] = 1.0;
}

__device__ __forceinline__ void q2q1_d_q2_basis(
    double xi, double eta, double zeta,
    double N[27],
    double gx[27], double gy[27], double gz[27],
    double lap[27],
    double invHx, double invHy, double invHz,
    double lapScaleX, double lapScaleY, double lapScaleZ
) {
    double Lx[3], Ly[3], Lz[3];
    double dLx[3], dLy[3], dLz[3];
    double ddLx[3], ddLy[3], ddLz[3];

    q2q1_d_q2_1d(xi, Lx, dLx, ddLx);
    q2q1_d_q2_1d(eta, Ly, dLy, ddLy);
    q2q1_d_q2_1d(zeta, Lz, dLz, ddLz);

    int a = 0;
    for (int kk = 0; kk < 3; ++kk) {
        for (int jj = 0; jj < 3; ++jj) {
            for (int ii = 0; ii < 3; ++ii) {
                const double v = Lx[ii] * Ly[jj] * Lz[kk];
                N[a] = v;
                gx[a] = dLx[ii] * Ly[jj] * Lz[kk] * invHx;
                gy[a] = Lx[ii] * dLy[jj] * Lz[kk] * invHy;
                gz[a] = Lx[ii] * Ly[jj] * dLz[kk] * invHz;

                lap[a] =
                    ddLx[ii] * Ly[jj] * Lz[kk] * lapScaleX +
                    Lx[ii] * ddLy[jj] * Lz[kk] * lapScaleY +
                    Lx[ii] * Ly[jj] * ddLz[kk] * lapScaleZ;

                ++a;
            }
        }
    }
}

__device__ __forceinline__ void q2q1_d_q1_basis(
    double xi, double eta, double zeta,
    double N[8],
    double gx[8], double gy[8], double gz[8],
    double invHx, double invHy, double invHz
) {
    const int sx[8] = {-1,  1,  1, -1, -1,  1,  1, -1};
    const int sy[8] = {-1, -1,  1,  1, -1, -1,  1,  1};
    const int sz[8] = {-1, -1, -1, -1,  1,  1,  1,  1};

    for (int a = 0; a < 8; ++a) {
        const double X = 1.0 + (double)sx[a] * xi;
        const double Y = 1.0 + (double)sy[a] * eta;
        const double Z = 1.0 + (double)sz[a] * zeta;

        N[a] = 0.125 * X * Y * Z;
        gx[a] = 0.125 * (double)sx[a] * Y * Z * invHx;
        gy[a] = 0.125 * X * (double)sy[a] * Z * invHy;
        gz[a] = 0.125 * X * Y * (double)sz[a] * invHz;
    }
}

__device__ __forceinline__ void q2q1_d_force(double x, double y, double z, double nu,
                                             double& fx, double& fy, double& fz) {
    const double pi = 3.141592653589793238462643383279502884;
    const double sx = sin(pi * x);
    const double sy = sin(pi * y);
    const double sz = sin(pi * z);

    const double cx = cos(pi * x);
    const double cy = cos(pi * y);
    const double cz = cos(pi * z);

    const double c2x = cos(2.0 * pi * x);
    const double c2y = cos(2.0 * pi * y);
    const double c2z = cos(2.0 * pi * z);

    const double A = sx * sx;
    const double B = sy * cy;
    const double C = sz * sz;
    const double D = sx * cx;
    const double E = sy * sy;

    const double lapUx =
        4.0 * pi * pi * pi * B * (c2x * C - 2.0 * A * C + A * c2z);

    const double lapUy =
        4.0 * pi * pi * pi * D * (2.0 * E * C - c2y * C - E * c2z);

    const double dpdx = -pi * sx * sy * sz;
    const double dpdy =  pi * cx * cy * sz;
    const double dpdz =  pi * cx * sy * cz;

    fx = -nu * lapUx + dpdx;
    fy = -nu * lapUy + dpdy;
    fz = dpdz;
}

__device__ __forceinline__ int q2q1_d_idx_vv(int comp, int a, int b) {
    return comp * 27 * 27 + a * 27 + b;
}

__device__ __forceinline__ int q2q1_d_idx_vp(int comp, int a, int bp) {
    return 2187 + comp * 27 * 8 + a * 8 + bp;
}

__device__ __forceinline__ int q2q1_d_idx_pv(int ap, int comp, int b) {
    return 2835 + ap * 3 * 27 + comp * 27 + b;
}

__device__ __forceinline__ int q2q1_d_idx_pp(int ap, int bp) {
    return 3483 + ap * 8 + bp;
}

__global__ void q2q1_stokes_pspg_continuous_kernel(
    Real* vals,
    Real* rhs,
    const int* cellSlots,
    int N,
    int nv1,
    int np1,
    long long nV,
    double hx,
    double hy,
    double hz,
    double nu,
    double tau,
    int continuousRhs
) {
    const long long cell = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    const long long nCells = (long long)N * (long long)N * (long long)N;
    if (cell >= nCells) return;

    const int i = (int)(cell % N);
    const int j = (int)((cell / N) % N);
    const int k = (int)(cell / ((long long)N * (long long)N));

    long long vnodes[27];
    int a = 0;
    for (int kk = 0; kk < 3; ++kk) {
        for (int jj = 0; jj < 3; ++jj) {
            for (int ii = 0; ii < 3; ++ii) {
                vnodes[a++] = q2q1_d_vnode_id(nv1, 2 * i + ii, 2 * j + jj, 2 * k + kk);
            }
        }
    }

    long long pnodes[8];
    pnodes[0] = q2q1_d_pnode_id(np1, i,     j,     k);
    pnodes[1] = q2q1_d_pnode_id(np1, i + 1, j,     k);
    pnodes[2] = q2q1_d_pnode_id(np1, i + 1, j + 1, k);
    pnodes[3] = q2q1_d_pnode_id(np1, i,     j + 1, k);
    pnodes[4] = q2q1_d_pnode_id(np1, i,     j,     k + 1);
    pnodes[5] = q2q1_d_pnode_id(np1, i + 1, j,     k + 1);
    pnodes[6] = q2q1_d_pnode_id(np1, i + 1, j + 1, k + 1);
    pnodes[7] = q2q1_d_pnode_id(np1, i,     j + 1, k + 1);

    const double invHx = 2.0 / hx;
    const double invHy = 2.0 / hy;
    const double invHz = 2.0 / hz;

    const double lapScaleX = invHx * invHx;
    const double lapScaleY = invHy * invHy;
    const double lapScaleZ = invHz * invHz;

    const double detJ = hx * hy * hz / 8.0;

    const double qa = sqrt((3.0 - 2.0 * sqrt(6.0 / 5.0)) / 7.0);
    const double qb = sqrt((3.0 + 2.0 * sqrt(6.0 / 5.0)) / 7.0);
    const double qx[4] = {-qb, -qa, qa, qb};
    const double wa = (18.0 + sqrt(30.0)) / 36.0;
    const double wb = (18.0 - sqrt(30.0)) / 36.0;
    const double qw[4] = {wb, wa, wa, wb};

    const int* slots = cellSlots + cell * 3547LL;

    for (int kq = 0; kq < 4; ++kq) {
        for (int jq = 0; jq < 4; ++jq) {
            for (int iq = 0; iq < 4; ++iq) {
                const double xi = qx[iq];
                const double eta = qx[jq];
                const double zeta = qx[kq];
                const double dV = qw[iq] * qw[jq] * qw[kq] * detJ;

                double Nv[27], gxV[27], gyV[27], gzV[27], lapV[27];
                double Np[8], gxP[8], gyP[8], gzP[8];

                q2q1_d_q2_basis(xi, eta, zeta, Nv, gxV, gyV, gzV, lapV,
                                 invHx, invHy, invHz, lapScaleX, lapScaleY, lapScaleZ);
                q2q1_d_q1_basis(xi, eta, zeta, Np, gxP, gyP, gzP, invHx, invHy, invHz);

                const double xq = ((double)i + 0.5 * (xi + 1.0)) * hx;
                const double yq = ((double)j + 0.5 * (eta + 1.0)) * hy;
                const double zq = ((double)k + 0.5 * (zeta + 1.0)) * hz;

                double fx = 0.0;
                double fy = 0.0;
                double fz = 0.0;
                if (continuousRhs) {
                    q2q1_d_force(xq, yq, zq, nu, fx, fy, fz);
                }

                for (int a = 0; a < 27; ++a) {
                    for (int b = 0; b < 27; ++b) {
                        const double vv = nu * q2q1_d_dot(gxV[a], gyV[a], gzV[a],
                                                           gxV[b], gyV[b], gzV[b]) * dV;

                        int fs = slots[q2q1_d_idx_vv(0, a, b)];
                        if (fs >= 0) atomicAdd(vals + fs, Real(vv));
                        fs = slots[q2q1_d_idx_vv(1, a, b)];
                        if (fs >= 0) atomicAdd(vals + fs, Real(vv));
                        fs = slots[q2q1_d_idx_vv(2, a, b)];
                        if (fs >= 0) atomicAdd(vals + fs, Real(vv));
                    }

                    for (int bp = 0; bp < 8; ++bp) {
                        int fs = slots[q2q1_d_idx_vp(0, a, bp)];
                        if (fs >= 0) atomicAdd(vals + fs, Real((-Np[bp] * gxV[a]) * dV));

                        fs = slots[q2q1_d_idx_vp(1, a, bp)];
                        if (fs >= 0) atomicAdd(vals + fs, Real((-Np[bp] * gyV[a]) * dV));

                        fs = slots[q2q1_d_idx_vp(2, a, bp)];
                        if (fs >= 0) atomicAdd(vals + fs, Real((-Np[bp] * gzV[a]) * dV));
                    }

                    const long long va = vnodes[a];
                    atomicAdd(rhs + q2q1_d_ux(va), Real(Nv[a] * fx * dV));
                    atomicAdd(rhs + q2q1_d_uy(nV, va), Real(Nv[a] * fy * dV));
                    atomicAdd(rhs + q2q1_d_uz(nV, va), Real(Nv[a] * fz * dV));
                }

                for (int ap = 0; ap < 8; ++ap) {
                    for (int b = 0; b < 27; ++b) {
                        int fs = slots[q2q1_d_idx_pv(ap, 0, b)];
                        if (fs >= 0) atomicAdd(vals + fs,
                            Real((Np[ap] * gxV[b] + tau * gxP[ap] * (-nu * lapV[b])) * dV));

                        fs = slots[q2q1_d_idx_pv(ap, 1, b)];
                        if (fs >= 0) atomicAdd(vals + fs,
                            Real((Np[ap] * gyV[b] + tau * gyP[ap] * (-nu * lapV[b])) * dV));

                        fs = slots[q2q1_d_idx_pv(ap, 2, b)];
                        if (fs >= 0) atomicAdd(vals + fs,
                            Real((Np[ap] * gzV[b] + tau * gzP[ap] * (-nu * lapV[b])) * dV));
                    }

                    for (int bp = 0; bp < 8; ++bp) {
                        const double pp = tau * q2q1_d_dot(gxP[ap], gyP[ap], gzP[ap],
                                                            gxP[bp], gyP[bp], gzP[bp]) * dV;
                        const int fs = slots[q2q1_d_idx_pp(ap, bp)];
                        if (fs >= 0) atomicAdd(vals + fs, Real(pp));
                    }

                    const long long pa = pnodes[ap];
                    const double rhsP = tau * (gxP[ap] * fx + gyP[ap] * fy + gzP[ap] * fz) * dV;
                    atomicAdd(rhs + q2q1_d_pr(nV, pa), Real(rhsP));
                }
            }
        }
    }
}

} // namespace

Q2Q1CudaStokesFillReport q2q1_cuda_fill_stokes_pspg_continuous_rhs(
    const Q2Q1StructuredGrid& g,
    const Q2Q1CellSlotCache& slots,
    SparseRows& A,
    std::vector<Real>& b,
    double nu,
    double tau,
    int continuousRhs
) {
    Q2Q1CudaStokesFillReport rep;
    rep.rows = g.nRows;
    rep.nnz = (long long)A.flatVals.size();
    rep.cells = g.nCells;
    rep.slotCount = (long long)slots.slot.size();

    if (!A.fixedPattern || !A.pattern) {
        throw std::runtime_error("q2q1_cuda_fill_stokes_pspg_continuous_rhs requires fixed-pattern SparseRows.");
    }
    if ((long long)sparse_nrows(A) != g.nRows) {
        throw std::runtime_error("q2q1_cuda_fill_stokes_pspg_continuous_rhs row mismatch.");
    }
    if (slots.nCells != g.nCells || slots.slotsPerCell != 3547) {
        throw std::runtime_error("q2q1_cuda_fill_stokes_pspg_continuous_rhs slot cache mismatch.");
    }

    Real* dVals = nullptr;
    Real* dRhs = nullptr;
    int* dSlots = nullptr;

    cudaEvent_t t0, t1, t2, t3;
    Q2Q1_CUDA_CHECK(cudaEventCreate(&t0));
    Q2Q1_CUDA_CHECK(cudaEventCreate(&t1));
    Q2Q1_CUDA_CHECK(cudaEventCreate(&t2));
    Q2Q1_CUDA_CHECK(cudaEventCreate(&t3));

    try {
        Q2Q1_CUDA_CHECK(cudaMalloc((void**)&dVals, sizeof(Real) * (std::size_t)rep.nnz));
        Q2Q1_CUDA_CHECK(cudaMalloc((void**)&dRhs, sizeof(Real) * (std::size_t)rep.rows));
        Q2Q1_CUDA_CHECK(cudaMalloc((void**)&dSlots, sizeof(int) * (std::size_t)rep.slotCount));

        Q2Q1_CUDA_CHECK(cudaMemcpy(
            dSlots,
            slots.slot.data(),
            sizeof(int) * (std::size_t)rep.slotCount,
            cudaMemcpyHostToDevice
        ));

        const int block = 128;
        const int gridZero = (int)std::min<long long>(
            65535LL,
            std::max<long long>(1LL, (std::max(rep.nnz, rep.rows) + block - 1) / block)
        );
        const int gridCells = (int)((rep.cells + block - 1) / block);

        Q2Q1_CUDA_CHECK(cudaEventRecord(t0));
        q2q1_zero_values_zero_rhs_kernel<<<gridZero, block>>>(dVals, rep.nnz, dRhs, rep.rows);
        Q2Q1_CUDA_CHECK(cudaGetLastError());
        Q2Q1_CUDA_CHECK(cudaEventRecord(t1));

        q2q1_stokes_pspg_continuous_kernel<<<gridCells, block>>>(
            dVals,
            dRhs,
            dSlots,
            g.N,
            g.nv1,
            g.np1,
            g.nVelocityNodes,
            g.hx,
            g.hy,
            g.hz,
            nu,
            tau,
            continuousRhs
        );
        Q2Q1_CUDA_CHECK(cudaGetLastError());
        Q2Q1_CUDA_CHECK(cudaEventRecord(t2));

        A.flatVals.assign((std::size_t)rep.nnz, Real(0));
        b.assign((std::size_t)rep.rows, Real(0));

        Q2Q1_CUDA_CHECK(cudaMemcpy(
            A.flatVals.data(),
            dVals,
            sizeof(Real) * (std::size_t)rep.nnz,
            cudaMemcpyDeviceToHost
        ));
        Q2Q1_CUDA_CHECK(cudaMemcpy(
            b.data(),
            dRhs,
            sizeof(Real) * (std::size_t)rep.rows,
            cudaMemcpyDeviceToHost
        ));
        Q2Q1_CUDA_CHECK(cudaEventRecord(t3));
        Q2Q1_CUDA_CHECK(cudaEventSynchronize(t3));

        rep.zeroSeconds = q2q1_cuda_elapsed_seconds(t0, t1);
        rep.cellKernelSeconds = q2q1_cuda_elapsed_seconds(t1, t2);
        rep.copyBackSeconds = q2q1_cuda_elapsed_seconds(t2, t3);
        rep.totalSeconds = q2q1_cuda_elapsed_seconds(t0, t3);

        Q2Q1_CUDA_CHECK(cudaFree(dVals)); dVals = nullptr;
        Q2Q1_CUDA_CHECK(cudaFree(dRhs)); dRhs = nullptr;
        Q2Q1_CUDA_CHECK(cudaFree(dSlots)); dSlots = nullptr;

        Q2Q1_CUDA_CHECK(cudaEventDestroy(t0));
        Q2Q1_CUDA_CHECK(cudaEventDestroy(t1));
        Q2Q1_CUDA_CHECK(cudaEventDestroy(t2));
        Q2Q1_CUDA_CHECK(cudaEventDestroy(t3));

        return rep;
    } catch (...) {
        if (dVals) cudaFree(dVals);
        if (dRhs) cudaFree(dRhs);
        if (dSlots) cudaFree(dSlots);
        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
        cudaEventDestroy(t2);
        cudaEventDestroy(t3);
        throw;
    }
}
