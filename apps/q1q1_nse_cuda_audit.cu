#ifndef MEMOIRS_PRECISION_float
#ifndef MEMOIRS_PRECISION_double
#define MEMOIRS_PRECISION_double
#endif
#endif

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>

// Avoid pulling HYPRE/MPI headers into nvcc compilation through 00_common.hpp.
#ifdef MEMOIRS_USE_HYPRE
#define MEMOIRS_RESTORE_USE_HYPRE_AFTER_CUDA_AUDIT 1
#undef MEMOIRS_USE_HYPRE
#endif

#include "memoirs/sections/00_common.hpp"
#include "memoirs/sections/01_options.hpp"
#include "memoirs/sections/02_polymesh.hpp"
#include "memoirs/sections/03_cell_topology.hpp"
#include "memoirs/sections/04_reference_elements.hpp"
#include "memoirs/sections/05_dof_map.hpp"
#include "memoirs/sections/06_sparse_rows.hpp"
#include "memoirs/sections/07_mms.hpp"
#include "memoirs/sections/08_laplacian_assembler.hpp"
#include "memoirs/sections/11_q1q1_nse_utils.hpp"
#include "memoirs/sections/12_q1q1_alg_matrix.hpp"
#include "memoirs/sections/13_q1q1_continuous_stokes_rhs.hpp"
#include "memoirs/sections/14_q1q1_nse_picard.hpp"
#include "memoirs/sections/15_q1q1_nse_cuda_audit.hpp"

#ifdef MEMOIRS_RESTORE_USE_HYPRE_AFTER_CUDA_AUDIT
#define MEMOIRS_USE_HYPRE 1
#undef MEMOIRS_RESTORE_USE_HYPRE_AFTER_CUDA_AUDIT
#endif

namespace {

#define Q1Q1_CUDA_CHECK(expr)                                                        \
    do {                                                                             \
        cudaError_t _e = (expr);                                                     \
        if (_e != cudaSuccess) {                                                     \
            throw std::runtime_error(std::string("CUDA error: ") + #expr +           \
                                     " -> " + cudaGetErrorString(_e));               \
        }                                                                            \
    } while (0)


static inline int q1q1_cuda_cavity_quad_order_from_env() {
    int q = 4; // default: old qp64 behavior
    if (const char* e = std::getenv("MEMOIRS_Q1Q1_CUDA_CAVITY_QUAD")) {
        q = std::atoi(e);
    }
    if (q != 2 && q != 3 && q != 4) {
        std::cerr << "MEMOIRS_Q1Q1_CUDA_CAVITY_QUAD must be 2, 3, or 4. Got " << q << "\n";
        std::exit(1);
    }
    return q;
}

static inline const char* q1q1_cuda_cavity_quad_label(int q) {
    if (q == 2) return "qp8";
    if (q == 3) return "qp27";
    return "qp64";
}


static inline double q1q1_cuda_elapsed_seconds(cudaEvent_t a, cudaEvent_t b) {
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, a, b);
    return 0.001 * (double)ms;
}

struct DVec3 {
    double x, y, z;
};

struct DMat3 {
    double a[3][3];
};

struct DExact {
    double ux, uy, uz, p;
};

struct DForce {
    double fx, fy, fz;
};

__device__ inline int d_row(int node, int field) {
    return 4 * node + field;
}

__device__ inline double d_dot3(DVec3 a, DVec3 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

__device__ inline void d_atomic_add(Real* ptr, double v) {
    atomicAdd(ptr, (Real)v);
}

__device__ inline double d_det3(const DMat3& M) {
    return
        M.a[0][0]*(M.a[1][1]*M.a[2][2] - M.a[1][2]*M.a[2][1])
      - M.a[0][1]*(M.a[1][0]*M.a[2][2] - M.a[1][2]*M.a[2][0])
      + M.a[0][2]*(M.a[1][0]*M.a[2][1] - M.a[1][1]*M.a[2][0]);
}

__device__ inline DMat3 d_inv3(const DMat3& M) {
    const double d = d_det3(M);

    DMat3 B;
    B.a[0][0] =  (M.a[1][1]*M.a[2][2] - M.a[1][2]*M.a[2][1]) / d;
    B.a[0][1] = -(M.a[0][1]*M.a[2][2] - M.a[0][2]*M.a[2][1]) / d;
    B.a[0][2] =  (M.a[0][1]*M.a[1][2] - M.a[0][2]*M.a[1][1]) / d;

    B.a[1][0] = -(M.a[1][0]*M.a[2][2] - M.a[1][2]*M.a[2][0]) / d;
    B.a[1][1] =  (M.a[0][0]*M.a[2][2] - M.a[0][2]*M.a[2][0]) / d;
    B.a[1][2] = -(M.a[0][0]*M.a[1][2] - M.a[0][2]*M.a[1][0]) / d;

    B.a[2][0] =  (M.a[1][0]*M.a[2][1] - M.a[1][1]*M.a[2][0]) / d;
    B.a[2][1] = -(M.a[0][0]*M.a[2][1] - M.a[0][1]*M.a[2][0]) / d;
    B.a[2][2] =  (M.a[0][0]*M.a[1][1] - M.a[0][1]*M.a[1][0]) / d;
    return B;
}

__device__ inline DVec3 d_invJT_mul(const DMat3& invJ, DVec3 gref) {
    DVec3 g;
    g.x = invJ.a[0][0]*gref.x + invJ.a[1][0]*gref.y + invJ.a[2][0]*gref.z;
    g.y = invJ.a[0][1]*gref.x + invJ.a[1][1]*gref.y + invJ.a[2][1]*gref.z;
    g.z = invJ.a[0][2]*gref.x + invJ.a[1][2]*gref.y + invJ.a[2][2]*gref.z;
    return g;
}

__device__ inline void d_hex_q1_basis(
    double xi,
    double eta,
    double zeta,
    double N[8],
    DVec3 dN[8]
) {
    const int sx[8] = {-1,  1,  1, -1, -1,  1,  1, -1};
    const int sy[8] = {-1, -1,  1,  1, -1, -1,  1,  1};
    const int sz[8] = {-1, -1, -1, -1,  1,  1,  1,  1};

    for (int a = 0; a < 8; ++a) {
        const double X = 1.0 + sx[a]*xi;
        const double Y = 1.0 + sy[a]*eta;
        const double Z = 1.0 + sz[a]*zeta;

        N[a] = 0.125 * X * Y * Z;
        dN[a].x = 0.125 * sx[a] * Y * Z;
        dN[a].y = 0.125 * X * sy[a] * Z;
        dN[a].z = 0.125 * X * Y * sz[a];
    }
}

__device__ inline DExact d_exact_mms_at(DVec3 X) {
    const double pi = 3.141592653589793238462643383279502884;

    const double sx = sin(pi * X.x);
    const double sy = sin(pi * X.y);
    const double sz = sin(pi * X.z);

    const double cx = cos(pi * X.x);
    const double cy = cos(pi * X.y);

    DExact e;
    e.ux =  2.0 * pi * sx * sx * sy * cy * sz * sz;
    e.uy = -2.0 * pi * sx * cx * sy * sy * sz * sz;
    e.uz =  0.0;
    e.p  =  cx * sy * sz;
    return e;
}

__device__ inline DForce d_stokes_force_at(DVec3 X, double nu) {
    const double pi = 3.141592653589793238462643383279502884;

    const double sx = sin(pi * X.x);
    const double sy = sin(pi * X.y);
    const double sz = sin(pi * X.z);

    const double cx = cos(pi * X.x);
    const double cy = cos(pi * X.y);
    const double cz = cos(pi * X.z);

    const double s2x = sin(2.0 * pi * X.x);
    const double s2y = sin(2.0 * pi * X.y);
    const double c2x = cos(2.0 * pi * X.x);
    const double c2y = cos(2.0 * pi * X.y);
    const double c2z = cos(2.0 * pi * X.z);

    const double A  = sx * sx;
    const double B  = sy * cy;
    const double C  = sz * sz;

    const double D  = sx * cx;
    const double E  = sy * sy;

    const double Add = 2.0 * pi * pi * c2x;
    const double Bdd = -2.0 * pi * pi * s2y;
    const double Cdd = 2.0 * pi * pi * c2z;

    const double Ddd = -2.0 * pi * pi * s2x;
    const double Edd = 2.0 * pi * pi * c2y;

    const double lapUx =  2.0 * pi * (Add * B * C + A * Bdd * C + A * B * Cdd);
    const double lapUy = -2.0 * pi * (Ddd * E * C + D * Edd * C + D * E * Cdd);
    const double lapUz =  0.0;

    const double dpdx = -pi * sx * sy * sz;
    const double dpdy =  pi * cx * cy * sz;
    const double dpdz =  pi * cx * sy * cz;

    DForce f;
    f.fx = -nu * lapUx + dpdx;
    f.fy = -nu * lapUy + dpdy;
    f.fz = -nu * lapUz + dpdz;
    return f;
}

__device__ inline void d_exact_velocity_gradient(
    DVec3 X,
    double& dux_dx,
    double& dux_dy,
    double& dux_dz,
    double& duy_dx,
    double& duy_dy,
    double& duy_dz
) {
    const double pi = 3.141592653589793238462643383279502884;

    const double sx = sin(pi * X.x);
    const double sy = sin(pi * X.y);
    const double sz = sin(pi * X.z);

    const double cx = cos(pi * X.x);
    const double cy = cos(pi * X.y);

    const double s2x = sin(2.0 * pi * X.x);
    const double s2y = sin(2.0 * pi * X.y);
    const double s2z = sin(2.0 * pi * X.z);

    const double c2x = cos(2.0 * pi * X.x);
    const double c2y = cos(2.0 * pi * X.y);

    const double A = sx * sx;
    const double B = sy * cy;
    const double C = sz * sz;

    const double D = sx * cx;
    const double E = sy * sy;

    dux_dx =  2.0 * pi * pi * s2x * B * C;
    dux_dy =  2.0 * pi * pi * A * c2y * C;
    dux_dz =  2.0 * pi * pi * A * B * s2z;

    duy_dx = -2.0 * pi * pi * c2x * E * C;
    duy_dy = -2.0 * pi * pi * D * s2y * C;
    duy_dz = -2.0 * pi * pi * D * E * s2z;
}

__device__ inline DForce d_nse_force_at(DVec3 X, double nu, double advScale) {
    DForce f = d_stokes_force_at(X, nu);
    DExact u = d_exact_mms_at(X);

    double dux_dx, dux_dy, dux_dz, duy_dx, duy_dy, duy_dz;
    d_exact_velocity_gradient(X, dux_dx, dux_dy, dux_dz, duy_dx, duy_dy, duy_dz);

    const double convx = u.ux * dux_dx + u.uy * dux_dy + u.uz * dux_dz;
    const double convy = u.ux * duy_dx + u.uy * duy_dy + u.uz * duy_dz;
    const double convz = 0.0;

    f.fx += advScale * convx;
    f.fy += advScale * convy;
    f.fz += advScale * convz;

    return f;
}

__global__ void q1q1_cuda_assemble_kernel(
    int nCells,
    const DVec3* __restrict__ points,
    const int* __restrict__ cellNodes,
    const unsigned char* __restrict__ isBnd,
    const Q1Q1NseCellSlots* __restrict__ cellSlots,
    const Real* __restrict__ beta,
    double nu,
    double dt,
    double advScale,
    double tau,
    double tauSupg,
    int pressurePinNode,
    Real* __restrict__ vals,
    Real* __restrict__ rhs
) {
    const int cellI = blockIdx.x * blockDim.x + threadIdx.x;
    if (cellI >= nCells) return;

    const int* hv = cellNodes + 8 * cellI;
    const Q1Q1NseCellSlots& cs = cellSlots[cellI];

    const double qp[4] = {
        -0.86113631159405257522,
        -0.33998104358485626480,
         0.33998104358485626480,
         0.86113631159405257522
    };
    const double qw[4] = {
         0.34785484513745385737,
         0.65214515486254614263,
         0.65214515486254614263,
         0.34785484513745385737
    };

    const double massCoeff = 1.0 / dt;

    for (int iq = 0; iq < 4; ++iq) {
        for (int jq = 0; jq < 4; ++jq) {
            for (int kq = 0; kq < 4; ++kq) {
                const double xi   = qp[iq];
                const double eta  = qp[jq];
                const double zeta = qp[kq];
                const double w = qw[iq] * qw[jq] * qw[kq];

                double N[8];
                DVec3 dNref[8];
                d_hex_q1_basis(xi, eta, zeta, N, dNref);

                DVec3 xq{0.0, 0.0, 0.0};
                DMat3 J;
                #pragma unroll
                for (int r = 0; r < 3; ++r) {
                    #pragma unroll
                    for (int c = 0; c < 3; ++c) J.a[r][c] = 0.0;
                }

                for (int a = 0; a < 8; ++a) {
                    const DVec3 X = points[hv[a]];

                    xq.x += N[a] * X.x;
                    xq.y += N[a] * X.y;
                    xq.z += N[a] * X.z;

                    J.a[0][0] += dNref[a].x * X.x;
                    J.a[1][0] += dNref[a].x * X.y;
                    J.a[2][0] += dNref[a].x * X.z;

                    J.a[0][1] += dNref[a].y * X.x;
                    J.a[1][1] += dNref[a].y * X.y;
                    J.a[2][1] += dNref[a].y * X.z;

                    J.a[0][2] += dNref[a].z * X.x;
                    J.a[1][2] += dNref[a].z * X.y;
                    J.a[2][2] += dNref[a].z * X.z;
                }

                const double dV = fabs(d_det3(J)) * w;
                const DMat3 invJ = d_inv3(J);

                DVec3 grad[8];
                for (int a = 0; a < 8; ++a) {
                    grad[a] = d_invJT_mul(invJ, dNref[a]);
                }

                double bx = 0.0, by = 0.0, bz = 0.0;
                double oldHx = 0.0, oldHy = 0.0, oldHz = 0.0;

                for (int b = 0; b < 8; ++b) {
                    const int nb = hv[b];
                    if (isBnd[nb]) continue;

                    bx += N[b] * (double)beta[d_row(nb, 0)];
                    by += N[b] * (double)beta[d_row(nb, 1)];
                    bz += N[b] * (double)beta[d_row(nb, 2)];

                    const DExact ex = d_exact_mms_at(points[nb]);
                    oldHx += N[b] * ex.ux;
                    oldHy += N[b] * ex.uy;
                    oldHz += N[b] * ex.uz;
                }

                DForce f = d_nse_force_at(xq, nu, advScale);
                const double fx = f.fx + massCoeff * oldHx;
                const double fy = f.fy + massCoeff * oldHy;
                const double fz = f.fz + massCoeff * oldHz;

                for (int a = 0; a < 8; ++a) {
                    const int na = hv[a];

                    const double streamNa =
                        advScale * (bx * grad[a].x + by * grad[a].y + bz * grad[a].z);
                    const double test = N[a] + tauSupg * streamNa;

                    if (!isBnd[na]) {
                        d_atomic_add(&rhs[d_row(na, 0)], test * fx * dV);
                        d_atomic_add(&rhs[d_row(na, 1)], test * fy * dV);
                        d_atomic_add(&rhs[d_row(na, 2)], test * fz * dV);
                    }

                    if (na != pressurePinNode) {
                        const double rp =
                            tau * (grad[a].x * fx + grad[a].y * fy + grad[a].z * fz);
                        d_atomic_add(&rhs[d_row(na, 3)], rp * dV);
                    }
                }

                for (int a = 0; a < 8; ++a) {
                    const int na = hv[a];

                    const double streamNa =
                        advScale * (bx * grad[a].x + by * grad[a].y + bz * grad[a].z);

                    for (int b = 0; b < 8; ++b) {
                        const int nb = hv[b];

                        const double stiff = d_dot3(grad[a], grad[b]) * dV;
                        const double betaGradNb =
                            advScale * (bx * grad[b].x +
                                        by * grad[b].y +
                                        bz * grad[b].z);

                        const double residualVelTrial = massCoeff * N[b] + betaGradNb;

                        if (!isBnd[na]) {
                            if (!isBnd[nb]) {
                                const double baseVel =
                                    N[a] * residualVelTrial * dV
                                  + nu * stiff
                                  + tauSupg * streamNa * residualVelTrial * dV;

                                d_atomic_add(&vals[cs.vvFlat[a][b][0]], baseVel);
                                d_atomic_add(&vals[cs.vvFlat[a][b][1]], baseVel);
                                d_atomic_add(&vals[cs.vvFlat[a][b][2]], baseVel);
                            }

                            if (nb != pressurePinNode) {
                                d_atomic_add(
                                    &vals[cs.vpFlat[a][b][0]],
                                    (-N[b] * grad[a].x + tauSupg * streamNa * grad[b].x) * dV);
                                d_atomic_add(
                                    &vals[cs.vpFlat[a][b][1]],
                                    (-N[b] * grad[a].y + tauSupg * streamNa * grad[b].y) * dV);
                                d_atomic_add(
                                    &vals[cs.vpFlat[a][b][2]],
                                    (-N[b] * grad[a].z + tauSupg * streamNa * grad[b].z) * dV);
                            }
                        }

                        if (na != pressurePinNode) {
                            if (!isBnd[nb]) {
                                d_atomic_add(
                                    &vals[cs.pvFlat[a][b][0]],
                                    (N[a] * grad[b].x + tau * grad[a].x * residualVelTrial) * dV);
                                d_atomic_add(
                                    &vals[cs.pvFlat[a][b][1]],
                                    (N[a] * grad[b].y + tau * grad[a].y * residualVelTrial) * dV);
                                d_atomic_add(
                                    &vals[cs.pvFlat[a][b][2]],
                                    (N[a] * grad[b].z + tau * grad[a].z * residualVelTrial) * dV);
                            }

                            if (nb != pressurePinNode) {
                                d_atomic_add(&vals[cs.ppFlat[a][b]], tau * stiff);
                            }
                        }
                    }
                }
            }
        }
    }
}


__global__ void q1q1_cuda_assemble_kernel_qp64(
    int nCells,
    const DVec3* __restrict__ points,
    const int* __restrict__ cellNodes,
    const unsigned char* __restrict__ isBnd,
    const Q1Q1NseCellSlots* __restrict__ cellSlots,
    const Real* __restrict__ beta,
    double nu,
    double dt,
    double advScale,
    double tau,
    double tauSupg,
    int pressurePinNode,
    Real* __restrict__ vals,
    Real* __restrict__ rhs
) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int cellI = gid >> 6;      // gid / 64
    const int qid   = gid & 63;      // gid % 64

    if (cellI >= nCells) return;

    const int iq = qid / 16;
    const int jq = (qid / 4) & 3;
    const int kq = qid & 3;

    const int* hv = cellNodes + 8 * cellI;
    const Q1Q1NseCellSlots& cs = cellSlots[cellI];

    const double qp[4] = {
        -0.86113631159405257522,
        -0.33998104358485626480,
         0.33998104358485626480,
         0.86113631159405257522
    };
    const double qw[4] = {
         0.34785484513745385737,
         0.65214515486254614263,
         0.65214515486254614263,
         0.34785484513745385737
    };

    const double xi   = qp[iq];
    const double eta  = qp[jq];
    const double zeta = qp[kq];
    const double w = qw[iq] * qw[jq] * qw[kq];

    const double massCoeff = 1.0 / dt;

    double N[8];
    DVec3 dNref[8];
    d_hex_q1_basis(xi, eta, zeta, N, dNref);

    DVec3 xq{0.0, 0.0, 0.0};
    DMat3 J;

    #pragma unroll
    for (int r = 0; r < 3; ++r) {
        #pragma unroll
        for (int c = 0; c < 3; ++c) J.a[r][c] = 0.0;
    }

    #pragma unroll
    for (int a = 0; a < 8; ++a) {
        const DVec3 X = points[hv[a]];

        xq.x += N[a] * X.x;
        xq.y += N[a] * X.y;
        xq.z += N[a] * X.z;

        J.a[0][0] += dNref[a].x * X.x;
        J.a[1][0] += dNref[a].x * X.y;
        J.a[2][0] += dNref[a].x * X.z;

        J.a[0][1] += dNref[a].y * X.x;
        J.a[1][1] += dNref[a].y * X.y;
        J.a[2][1] += dNref[a].y * X.z;

        J.a[0][2] += dNref[a].z * X.x;
        J.a[1][2] += dNref[a].z * X.y;
        J.a[2][2] += dNref[a].z * X.z;
    }

    const double dV = fabs(d_det3(J)) * w;
    const DMat3 invJ = d_inv3(J);

    DVec3 grad[8];

    #pragma unroll
    for (int a = 0; a < 8; ++a) {
        grad[a] = d_invJT_mul(invJ, dNref[a]);
    }

    double bx = 0.0, by = 0.0, bz = 0.0;
    double oldHx = 0.0, oldHy = 0.0, oldHz = 0.0;

    #pragma unroll
    for (int b = 0; b < 8; ++b) {
        const int nb = hv[b];

        bx += N[b] * (double)beta[d_row(nb, 0)];
        by += N[b] * (double)beta[d_row(nb, 1)];
        bz += N[b] * (double)beta[d_row(nb, 2)];

        if (isBnd[nb]) continue;

        const DExact ex = d_exact_mms_at(points[nb]);
        oldHx += N[b] * ex.ux;
        oldHy += N[b] * ex.uy;
        oldHz += N[b] * ex.uz;
    }

    DForce f = d_nse_force_at(xq, nu, advScale);

    const double fx = f.fx + massCoeff * oldHx;
    const double fy = f.fy + massCoeff * oldHy;
    const double fz = f.fz + massCoeff * oldHz;

    #pragma unroll
    for (int a = 0; a < 8; ++a) {
        const int na = hv[a];

        const double streamNa =
            advScale * (bx * grad[a].x + by * grad[a].y + bz * grad[a].z);
        const double test = N[a] + tauSupg * streamNa;

        if (!isBnd[na]) {
            d_atomic_add(&rhs[d_row(na, 0)], test * fx * dV);
            d_atomic_add(&rhs[d_row(na, 1)], test * fy * dV);
            d_atomic_add(&rhs[d_row(na, 2)], test * fz * dV);
        }

        if (na != pressurePinNode) {
            const double rp =
                tau * (grad[a].x * fx + grad[a].y * fy + grad[a].z * fz);
            d_atomic_add(&rhs[d_row(na, 3)], rp * dV);
        }
    }

    #pragma unroll
    for (int a = 0; a < 8; ++a) {
        const int na = hv[a];

        const double streamNa =
            advScale * (bx * grad[a].x + by * grad[a].y + bz * grad[a].z);

        #pragma unroll
        for (int b = 0; b < 8; ++b) {
            const int nb = hv[b];

            const double stiff = d_dot3(grad[a], grad[b]) * dV;
            const double betaGradNb =
                advScale * (bx * grad[b].x +
                            by * grad[b].y +
                            bz * grad[b].z);

            const double residualVelTrial = massCoeff * N[b] + betaGradNb;

            if (!isBnd[na]) {
                if (!isBnd[nb]) {
                    const double baseVel =
                        N[a] * residualVelTrial * dV
                      + nu * stiff
                      + tauSupg * streamNa * residualVelTrial * dV;

                    d_atomic_add(&vals[cs.vvFlat[a][b][0]], baseVel);
                    d_atomic_add(&vals[cs.vvFlat[a][b][1]], baseVel);
                    d_atomic_add(&vals[cs.vvFlat[a][b][2]], baseVel);
                }

                if (nb != pressurePinNode) {
                    d_atomic_add(
                        &vals[cs.vpFlat[a][b][0]],
                        (-N[b] * grad[a].x + tauSupg * streamNa * grad[b].x) * dV);
                    d_atomic_add(
                        &vals[cs.vpFlat[a][b][1]],
                        (-N[b] * grad[a].y + tauSupg * streamNa * grad[b].y) * dV);
                    d_atomic_add(
                        &vals[cs.vpFlat[a][b][2]],
                        (-N[b] * grad[a].z + tauSupg * streamNa * grad[b].z) * dV);
                }
            }

            if (na != pressurePinNode) {
                if (!isBnd[nb]) {
                    d_atomic_add(
                        &vals[cs.pvFlat[a][b][0]],
                        (N[a] * grad[b].x + tau * grad[a].x * residualVelTrial) * dV);
                    d_atomic_add(
                        &vals[cs.pvFlat[a][b][1]],
                        (N[a] * grad[b].y + tau * grad[a].y * residualVelTrial) * dV);
                    d_atomic_add(
                        &vals[cs.pvFlat[a][b][2]],
                        (N[a] * grad[b].z + tau * grad[a].z * residualVelTrial) * dV);
                }

                if (nb != pressurePinNode) {
                    d_atomic_add(&vals[cs.ppFlat[a][b]], tau * stiff);
                }
            }
        }
    }
}



__device__ __forceinline__ void q1q1_cuda_cavity_fill_quad_rule(
    int qOrder,
    double* qp,
    double* qw
) {
    if (qOrder == 2) {
        const double a = 0.57735026918962576451;
        qp[0] = -a; qp[1] = a; qp[2] = 0.0; qp[3] = 0.0;
        qw[0] = 1.0; qw[1] = 1.0; qw[2] = 0.0; qw[3] = 0.0;
    } else if (qOrder == 3) {
        const double a = 0.77459666924148337704;
        qp[0] = -a; qp[1] = 0.0; qp[2] = a; qp[3] = 0.0;
        qw[0] = 0.55555555555555555556;
        qw[1] = 0.88888888888888888889;
        qw[2] = 0.55555555555555555556;
        qw[3] = 0.0;
    } else {
        qp[0] = -0.86113631159405257522;
        qp[1] = -0.33998104358485626480;
        qp[2] =  0.33998104358485626480;
        qp[3] =  0.86113631159405257522;
        qw[0] = 0.34785484513745385737;
        qw[1] = 0.65214515486254614263;
        qw[2] = 0.65214515486254614263;
        qw[3] = 0.34785484513745385737;
    }
}

__global__ void q1q1_cuda_assemble_cavity_bdf1_qp64(
    int nCells,
    const DVec3* __restrict__ points,
    const int* __restrict__ cellNodes,
    const unsigned char* __restrict__ isBnd,
    const Q1Q1NseCellSlots* __restrict__ cellSlots,
    const Real* __restrict__ beta,
    const Real* __restrict__ oldState,
    const Real* __restrict__ bcUx,
    const Real* __restrict__ bcUy,
    const Real* __restrict__ bcUz,
    double nu,
    double dt,
    double advScale,
    double tau,
    double tauSupg,
    int pressurePinNode,
    Real* __restrict__ vals,
    Real* __restrict__ rhs
, int qOrder) {
    const int qOrder2 = qOrder * qOrder;
    const int nQp = qOrder2 * qOrder;

    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int cellI = gid / nQp;
    const int qid   = gid - cellI * nQp;

    if (cellI >= nCells) return;

    const int iq = qid / qOrder2;
    const int rem = qid - iq * qOrder2;
    const int jq = rem / qOrder;
    const int kq = rem - jq * qOrder;

    const int* hv = cellNodes + 8 * cellI;
    const Q1Q1NseCellSlots& cs = cellSlots[cellI];

    double qp[4];
    double qw[4];
    q1q1_cuda_cavity_fill_quad_rule(qOrder, qp, qw);

    const double xi   = qp[iq];
    const double eta  = qp[jq];
    const double zeta = qp[kq];
    const double w = qw[iq] * qw[jq] * qw[kq];

    const double massCoeff = 1.0 / dt;

    double N[8];
    DVec3 dNref[8];
    d_hex_q1_basis(xi, eta, zeta, N, dNref);

    DVec3 xq{0.0, 0.0, 0.0};
    DMat3 J;

    #pragma unroll
    for (int r = 0; r < 3; ++r) {
        #pragma unroll
        for (int c = 0; c < 3; ++c) J.a[r][c] = 0.0;
    }

    #pragma unroll
    for (int a = 0; a < 8; ++a) {
        const DVec3 X = points[hv[a]];

        xq.x += N[a] * X.x;
        xq.y += N[a] * X.y;
        xq.z += N[a] * X.z;

        J.a[0][0] += dNref[a].x * X.x;
        J.a[1][0] += dNref[a].x * X.y;
        J.a[2][0] += dNref[a].x * X.z;

        J.a[0][1] += dNref[a].y * X.x;
        J.a[1][1] += dNref[a].y * X.y;
        J.a[2][1] += dNref[a].y * X.z;

        J.a[0][2] += dNref[a].z * X.x;
        J.a[1][2] += dNref[a].z * X.y;
        J.a[2][2] += dNref[a].z * X.z;
    }

    const double dV = fabs(d_det3(J)) * w;
    const DMat3 invJ = d_inv3(J);

    DVec3 grad[8];

    #pragma unroll
    for (int a = 0; a < 8; ++a) {
        grad[a] = d_invJT_mul(invJ, dNref[a]);
    }

    double bx = 0.0, by = 0.0, bz = 0.0;
    double oldUx = 0.0, oldUy = 0.0, oldUz = 0.0;

    #pragma unroll
    for (int b = 0; b < 8; ++b) {
        const int nb = hv[b];

        bx += N[b] * (double)beta[d_row(nb, 0)];
        by += N[b] * (double)beta[d_row(nb, 1)];
        bz += N[b] * (double)beta[d_row(nb, 2)];

        oldUx += N[b] * (double)oldState[d_row(nb, 0)];
        oldUy += N[b] * (double)oldState[d_row(nb, 1)];
        oldUz += N[b] * (double)oldState[d_row(nb, 2)];
    }

    const double fx = massCoeff * oldUx;
    const double fy = massCoeff * oldUy;
    const double fz = massCoeff * oldUz;

    // BDF1 old-time RHS. No body force for driven cavity.
    #pragma unroll
    for (int a = 0; a < 8; ++a) {
        const int na = hv[a];

        const double streamNa =
            advScale * (bx * grad[a].x + by * grad[a].y + bz * grad[a].z);
        const double test = N[a] + tauSupg * streamNa;

        if (!isBnd[na]) {
            d_atomic_add(&rhs[d_row(na, 0)], test * fx * dV);
            d_atomic_add(&rhs[d_row(na, 1)], test * fy * dV);
            d_atomic_add(&rhs[d_row(na, 2)], test * fz * dV);
        }

        if (na != pressurePinNode) {
            const double rp =
                tau * (grad[a].x * fx + grad[a].y * fy + grad[a].z * fz);
            d_atomic_add(&rhs[d_row(na, 3)], rp * dV);
        }
    }

    #pragma unroll
    for (int a = 0; a < 8; ++a) {
        const int na = hv[a];

        const double streamNa =
            advScale * (bx * grad[a].x + by * grad[a].y + bz * grad[a].z);

        #pragma unroll
        for (int b = 0; b < 8; ++b) {
            const int nb = hv[b];

            const double stiff = d_dot3(grad[a], grad[b]) * dV;
            const double betaGradNb =
                advScale * (bx * grad[b].x +
                            by * grad[b].y +
                            bz * grad[b].z);

            const double residualVelTrial = massCoeff * N[b] + betaGradNb;

            if (!isBnd[na]) {
                if (!isBnd[nb]) {
                    const double baseVel =
                        N[a] * residualVelTrial * dV
                      + nu * stiff
                      + tauSupg * streamNa * residualVelTrial * dV;

                    d_atomic_add(&vals[cs.vvFlat[a][b][0]], baseVel);
                    d_atomic_add(&vals[cs.vvFlat[a][b][1]], baseVel);
                    d_atomic_add(&vals[cs.vvFlat[a][b][2]], baseVel);
                } else {
                    // Known velocity Dirichlet column correction for moving lid.
                    const double gux = (double)bcUx[nb];
                    const double guy = (double)bcUy[nb];
                    const double guz = (double)bcUz[nb];

                    if (gux != 0.0 || guy != 0.0 || guz != 0.0) {
                        const double baseVel =
                            N[a] * residualVelTrial * dV
                          + nu * stiff
                          + tauSupg * streamNa * residualVelTrial * dV;

                        d_atomic_add(&rhs[d_row(na,0)], -baseVel * gux);
                        d_atomic_add(&rhs[d_row(na,1)], -baseVel * guy);
                        d_atomic_add(&rhs[d_row(na,2)], -baseVel * guz);
                    }
                }

                if (nb != pressurePinNode) {
                    d_atomic_add(
                        &vals[cs.vpFlat[a][b][0]],
                        (-N[b] * grad[a].x + tauSupg * streamNa * grad[b].x) * dV);
                    d_atomic_add(
                        &vals[cs.vpFlat[a][b][1]],
                        (-N[b] * grad[a].y + tauSupg * streamNa * grad[b].y) * dV);
                    d_atomic_add(
                        &vals[cs.vpFlat[a][b][2]],
                        (-N[b] * grad[a].z + tauSupg * streamNa * grad[b].z) * dV);
                }
            }

            if (na != pressurePinNode) {
                if (!isBnd[nb]) {
                    d_atomic_add(
                        &vals[cs.pvFlat[a][b][0]],
                        (N[a] * grad[b].x + tau * grad[a].x * residualVelTrial) * dV);
                    d_atomic_add(
                        &vals[cs.pvFlat[a][b][1]],
                        (N[a] * grad[b].y + tau * grad[a].y * residualVelTrial) * dV);
                    d_atomic_add(
                        &vals[cs.pvFlat[a][b][2]],
                        (N[a] * grad[b].z + tau * grad[a].z * residualVelTrial) * dV);
                } else {
                    const double gux = (double)bcUx[nb];
                    const double guy = (double)bcUy[nb];
                    const double guz = (double)bcUz[nb];

                    if (gux != 0.0 || guy != 0.0 || guz != 0.0) {
                        const double cx =
                            (N[a] * grad[b].x + tau * grad[a].x * residualVelTrial) * dV;
                        const double cy =
                            (N[a] * grad[b].y + tau * grad[a].y * residualVelTrial) * dV;
                        const double cz =
                            (N[a] * grad[b].z + tau * grad[a].z * residualVelTrial) * dV;

                        d_atomic_add(&rhs[d_row(na,3)], -(cx*gux + cy*guy + cz*guz));
                    }
                }

                if (nb != pressurePinNode) {
                    d_atomic_add(&vals[cs.ppFlat[a][b]], tau * stiff);
                }
            }
        }
    }
}


__global__ void q1q1_cuda_residual_l2_kernel(
    int nRows,
    const int* __restrict__ rowPtr,
    const int* __restrict__ cols,
    const Real* __restrict__ vals,
    const Real* __restrict__ rhs,
    const Real* __restrict__ x,
    const unsigned char* __restrict__ isBnd,
    int pressurePinNode,
    double* __restrict__ sums
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nRows) return;

    const int node = row >> 2;
    const int field = row & 3;

    // Ignore strong Dirichlet velocity rows and pressure pin.
    if (field < 3 && isBnd[node]) return;
    if (field == 3 && node == pressurePinNode) return;

    double ax = 0.0;
    for (int k = rowPtr[row]; k < rowPtr[row + 1]; ++k) {
        ax += (double)vals[k] * (double)x[cols[k]];
    }

    const double b = (double)rhs[row];
    const double r = ax - b;

    atomicAdd(&sums[0], r * r);
    atomicAdd(&sums[1], b * b);
    atomicAdd(&sums[2], ax * ax);
}

__global__ void q1q1_cuda_apply_cavity_strong_rows_kernel(
    int nBoundary,
    const int* __restrict__ boundaryNodes,
    int pressurePinNode,
    const Real* __restrict__ bcUx,
    const Real* __restrict__ bcUy,
    const Real* __restrict__ bcUz,
    const int* __restrict__ rowPtr,
    const int* __restrict__ cols,
    Real* __restrict__ vals,
    Real* __restrict__ rhs
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < nBoundary * 3) {
        const int bidx = tid / 3;
        const int comp = tid - 3 * bidx;
        const int node = boundaryNodes[bidx];
        const int row = d_row(node, comp);

        for (int k = rowPtr[row]; k < rowPtr[row + 1]; ++k) {
            vals[k] = Real(0);
        }

        for (int k = rowPtr[row]; k < rowPtr[row + 1]; ++k) {
            if (cols[k] == row) {
                vals[k] = Real(1);
                break;
            }
        }

        if (comp == 0) rhs[row] = bcUx[node];
        if (comp == 1) rhs[row] = bcUy[node];
        if (comp == 2) rhs[row] = bcUz[node];
        return;
    }

    // Pressure pin p = 0.
    if (tid == nBoundary * 3) {
        const int row = d_row(pressurePinNode, 3);

        for (int k = rowPtr[row]; k < rowPtr[row + 1]; ++k) {
            vals[k] = Real(0);
        }

        for (int k = rowPtr[row]; k < rowPtr[row + 1]; ++k) {
            if (cols[k] == row) {
                vals[k] = Real(1);
                break;
            }
        }

        rhs[row] = Real(0);
    }
}


__global__ void q1q1_cuda_assemble_cavity_bdf_history_qp64(
    int nCells,
    const DVec3* __restrict__ points,
    const int* __restrict__ cellNodes,
    const unsigned char* __restrict__ isBnd,
    const Q1Q1NseCellSlots* __restrict__ cellSlots,
    const Real* __restrict__ beta,
    const Real* __restrict__ oldState,
    const Real* __restrict__ olderState,
    const Real* __restrict__ bcUx,
    const Real* __restrict__ bcUy,
    const Real* __restrict__ bcUz,
    double nu,
    double advScale,
    double tau,
    double tauSupg,
    double lhsCoeff,
    double oldCoeff,
    double olderCoeff,
    int pressurePinNode,
    Real* __restrict__ vals,
    Real* __restrict__ rhs
, int qOrder) {
    const int qOrder2 = qOrder * qOrder;
    const int nQp = qOrder2 * qOrder;

    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int cellI = gid / nQp;
    const int qid   = gid - cellI * nQp;

    if (cellI >= nCells) return;

    const int iq = qid / qOrder2;
    const int rem = qid - iq * qOrder2;
    const int jq = rem / qOrder;
    const int kq = rem - jq * qOrder;

    const int* hv = cellNodes + 8 * cellI;
    const Q1Q1NseCellSlots& cs = cellSlots[cellI];

    double qp[4];
    double qw[4];
    q1q1_cuda_cavity_fill_quad_rule(qOrder, qp, qw);

    const double xi   = qp[iq];
    const double eta  = qp[jq];
    const double zeta = qp[kq];
    const double w = qw[iq] * qw[jq] * qw[kq];

    double N[8];
    DVec3 dNref[8];
    d_hex_q1_basis(xi, eta, zeta, N, dNref);

    DVec3 xq{0.0, 0.0, 0.0};
    DMat3 J;

    #pragma unroll
    for (int r = 0; r < 3; ++r) {
        #pragma unroll
        for (int c = 0; c < 3; ++c) J.a[r][c] = 0.0;
    }

    #pragma unroll
    for (int a = 0; a < 8; ++a) {
        const DVec3 X = points[hv[a]];

        xq.x += N[a] * X.x;
        xq.y += N[a] * X.y;
        xq.z += N[a] * X.z;

        J.a[0][0] += dNref[a].x * X.x;
        J.a[1][0] += dNref[a].x * X.y;
        J.a[2][0] += dNref[a].x * X.z;

        J.a[0][1] += dNref[a].y * X.x;
        J.a[1][1] += dNref[a].y * X.y;
        J.a[2][1] += dNref[a].y * X.z;

        J.a[0][2] += dNref[a].z * X.x;
        J.a[1][2] += dNref[a].z * X.y;
        J.a[2][2] += dNref[a].z * X.z;
    }

    const double dV = fabs(d_det3(J)) * w;
    const DMat3 invJ = d_inv3(J);

    DVec3 grad[8];

    #pragma unroll
    for (int a = 0; a < 8; ++a) {
        grad[a] = d_invJT_mul(invJ, dNref[a]);
    }

    double bx = 0.0, by = 0.0, bz = 0.0;
    double histUx = 0.0, histUy = 0.0, histUz = 0.0;

    #pragma unroll
    for (int b = 0; b < 8; ++b) {
        const int nb = hv[b];

        bx += N[b] * (double)beta[d_row(nb, 0)];
        by += N[b] * (double)beta[d_row(nb, 1)];
        bz += N[b] * (double)beta[d_row(nb, 2)];

        histUx += N[b] * (
            oldCoeff   * (double)oldState[d_row(nb, 0)]
          + olderCoeff * (double)olderState[d_row(nb, 0)]);
        histUy += N[b] * (
            oldCoeff   * (double)oldState[d_row(nb, 1)]
          + olderCoeff * (double)olderState[d_row(nb, 1)]);
        histUz += N[b] * (
            oldCoeff   * (double)oldState[d_row(nb, 2)]
          + olderCoeff * (double)olderState[d_row(nb, 2)]);
    }

    const double fx = histUx;
    const double fy = histUy;
    const double fz = histUz;

    // History RHS.
    #pragma unroll
    for (int a = 0; a < 8; ++a) {
        const int na = hv[a];

        const double streamNa =
            advScale * (bx * grad[a].x + by * grad[a].y + bz * grad[a].z);
        const double test = N[a] + tauSupg * streamNa;

        if (!isBnd[na]) {
            d_atomic_add(&rhs[d_row(na, 0)], test * fx * dV);
            d_atomic_add(&rhs[d_row(na, 1)], test * fy * dV);
            d_atomic_add(&rhs[d_row(na, 2)], test * fz * dV);
        }

        if (na != pressurePinNode) {
            const double rp =
                tau * (grad[a].x * fx + grad[a].y * fy + grad[a].z * fz);
            d_atomic_add(&rhs[d_row(na, 3)], rp * dV);
        }
    }

    #pragma unroll
    for (int a = 0; a < 8; ++a) {
        const int na = hv[a];

        const double streamNa =
            advScale * (bx * grad[a].x + by * grad[a].y + bz * grad[a].z);

        #pragma unroll
        for (int b = 0; b < 8; ++b) {
            const int nb = hv[b];

            const double stiff = d_dot3(grad[a], grad[b]) * dV;
            const double betaGradNb =
                advScale * (bx * grad[b].x +
                            by * grad[b].y +
                            bz * grad[b].z);

            const double residualVelTrial = lhsCoeff * N[b] + betaGradNb;

            if (!isBnd[na]) {
                if (!isBnd[nb]) {
                    const double baseVel =
                        N[a] * residualVelTrial * dV
                      + nu * stiff
                      + tauSupg * streamNa * residualVelTrial * dV;

                    d_atomic_add(&vals[cs.vvFlat[a][b][0]], baseVel);
                    d_atomic_add(&vals[cs.vvFlat[a][b][1]], baseVel);
                    d_atomic_add(&vals[cs.vvFlat[a][b][2]], baseVel);
                } else {
                    const double gux = (double)bcUx[nb];
                    const double guy = (double)bcUy[nb];
                    const double guz = (double)bcUz[nb];

                    if (gux != 0.0 || guy != 0.0 || guz != 0.0) {
                        const double baseVel =
                            N[a] * residualVelTrial * dV
                          + nu * stiff
                          + tauSupg * streamNa * residualVelTrial * dV;

                        d_atomic_add(&rhs[d_row(na,0)], -baseVel * gux);
                        d_atomic_add(&rhs[d_row(na,1)], -baseVel * guy);
                        d_atomic_add(&rhs[d_row(na,2)], -baseVel * guz);
                    }
                }

                if (nb != pressurePinNode) {
                    d_atomic_add(
                        &vals[cs.vpFlat[a][b][0]],
                        (-N[b] * grad[a].x + tauSupg * streamNa * grad[b].x) * dV);
                    d_atomic_add(
                        &vals[cs.vpFlat[a][b][1]],
                        (-N[b] * grad[a].y + tauSupg * streamNa * grad[b].y) * dV);
                    d_atomic_add(
                        &vals[cs.vpFlat[a][b][2]],
                        (-N[b] * grad[a].z + tauSupg * streamNa * grad[b].z) * dV);
                }
            }

            if (na != pressurePinNode) {
                if (!isBnd[nb]) {
                    d_atomic_add(
                        &vals[cs.pvFlat[a][b][0]],
                        (N[a] * grad[b].x + tau * grad[a].x * residualVelTrial) * dV);
                    d_atomic_add(
                        &vals[cs.pvFlat[a][b][1]],
                        (N[a] * grad[b].y + tau * grad[a].y * residualVelTrial) * dV);
                    d_atomic_add(
                        &vals[cs.pvFlat[a][b][2]],
                        (N[a] * grad[b].z + tau * grad[a].z * residualVelTrial) * dV);
                } else {
                    const double gux = (double)bcUx[nb];
                    const double guy = (double)bcUy[nb];
                    const double guz = (double)bcUz[nb];

                    if (gux != 0.0 || guy != 0.0 || guz != 0.0) {
                        const double cx =
                            (N[a] * grad[b].x + tau * grad[a].x * residualVelTrial) * dV;
                        const double cy =
                            (N[a] * grad[b].y + tau * grad[a].y * residualVelTrial) * dV;
                        const double cz =
                            (N[a] * grad[b].z + tau * grad[a].z * residualVelTrial) * dV;

                        d_atomic_add(&rhs[d_row(na,3)], -(cx*gux + cy*guy + cz*guz));
                    }
                }

                if (nb != pressurePinNode) {
                    d_atomic_add(&vals[cs.ppFlat[a][b]], tau * stiff);
                }
            }
        }
    }
}

__global__ void q1q1_cuda_apply_strong_rows_kernel(
    int nBoundary,
    const int* __restrict__ boundaryNodes,
    int pressurePinNode,
    const DVec3* __restrict__ points,
    const int* __restrict__ rowPtr,
    const int* __restrict__ cols,
    Real* __restrict__ vals,
    Real* __restrict__ rhs
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary velocity identity rows.
    if (tid < nBoundary * 3) {
        const int bidx = tid / 3;
        const int comp = tid - 3 * bidx;
        const int node = boundaryNodes[bidx];
        const int row = d_row(node, comp);

        for (int k = rowPtr[row]; k < rowPtr[row + 1]; ++k) {
            vals[k] = Real(0);
        }

        for (int k = rowPtr[row]; k < rowPtr[row + 1]; ++k) {
            if (cols[k] == row) {
                vals[k] = Real(1);
                break;
            }
        }

        const DExact ex = d_exact_mms_at(points[node]);
        rhs[row] = (comp == 0) ? Real(ex.ux) : ((comp == 1) ? Real(ex.uy) : Real(ex.uz));
        return;
    }

    // Pressure pin identity row.
    if (tid == nBoundary * 3) {
        const int row = d_row(pressurePinNode, 3);

        for (int k = rowPtr[row]; k < rowPtr[row + 1]; ++k) {
            vals[k] = Real(0);
        }

        for (int k = rowPtr[row]; k < rowPtr[row + 1]; ++k) {
            if (cols[k] == row) {
                vals[k] = Real(1);
                break;
            }
        }

        const DExact ex = d_exact_mms_at(points[pressurePinNode]);
        rhs[row] = Real(ex.p);
    }
}

template <class T>
T* device_alloc_and_copy(const std::vector<T>& h) {
    T* d = nullptr;
    if (!h.empty()) {
        Q1Q1_CUDA_CHECK(cudaMalloc(&d, sizeof(T) * h.size()));
        Q1Q1_CUDA_CHECK(cudaMemcpy(d, h.data(), sizeof(T) * h.size(), cudaMemcpyHostToDevice));
    }
    return d;
}


struct Q1Q1CudaAssemblyCache {
    bool valid = false;
    int nNodes = 0;
    int nRows = 0;
    int nCells = 0;
    int nnz = 0;
    int nBoundary = 0;
    const void* patternPtr = nullptr;

    DVec3* dPoints = nullptr;
    int* dCellNodes = nullptr;
    unsigned char* dIsBnd = nullptr;
    Q1Q1NseCellSlots* dCellSlots = nullptr;
    Real* dBeta = nullptr;
    int* dBoundaryNodes = nullptr;
    int* dRowPtr = nullptr;
    int* dCols = nullptr;
    Real* dVals = nullptr;
    Real* dRhs = nullptr;

    void clear() {
        cudaFree(dPoints);        dPoints = nullptr;
        cudaFree(dCellNodes);     dCellNodes = nullptr;
        cudaFree(dIsBnd);         dIsBnd = nullptr;
        cudaFree(dCellSlots);     dCellSlots = nullptr;
        cudaFree(dBeta);          dBeta = nullptr;
        cudaFree(dBoundaryNodes); dBoundaryNodes = nullptr;
        cudaFree(dRowPtr);        dRowPtr = nullptr;
        cudaFree(dCols);          dCols = nullptr;
        cudaFree(dVals);          dVals = nullptr;
        cudaFree(dRhs);           dRhs = nullptr;

        valid = false;
        nNodes = nRows = nCells = nnz = nBoundary = 0;
        patternPtr = nullptr;
    }

    ~Q1Q1CudaAssemblyCache() {
        clear();
    }
};

static Q1Q1CudaAssemblyCache& q1q1_cuda_assembly_cache_singleton() {
    static Q1Q1CudaAssemblyCache cache;
    return cache;
}

static void q1q1_cuda_prepare_assembly_cache(
    Q1Q1CudaAssemblyCache& cache,
    const PolyMesh& mesh,
    const LinearCgDofMap& dm,
    const AssembledSystem& hostSys,
    const std::vector<Q1Q1NseCellSlots>& cellSlots
) {
    const int nNodes = dm.nDofs;
    const int nRows = 4 * nNodes;
    const int nCells = (int)mesh.cells.size();
    const int nnz = (int)hostSys.A.flatVals.size();
    const int nBoundary = (int)dm.boundaryDofs.size();
    const void* patternPtr = hostSys.A.pattern ? (const void*)hostSys.A.pattern.get() : nullptr;

    if (cache.valid &&
        cache.nNodes == nNodes &&
        cache.nRows == nRows &&
        cache.nCells == nCells &&
        cache.nnz == nnz &&
        cache.nBoundary == nBoundary &&
        cache.patternPtr == patternPtr) {
        return;
    }

    cache.clear();

    cache.nNodes = nNodes;
    cache.nRows = nRows;
    cache.nCells = nCells;
    cache.nnz = nnz;
    cache.nBoundary = nBoundary;
    cache.patternPtr = patternPtr;

    std::vector<DVec3> hPoints(nNodes);
    for (int i = 0; i < nNodes; ++i) {
        hPoints[i] = {mesh.points[i].x, mesh.points[i].y, mesh.points[i].z};
    }

    std::vector<int> hCellNodes(8 * nCells);
    for (int c = 0; c < nCells; ++c) {
        const auto hv = ordered_hex_vertices_axis_aligned(mesh, mesh.cells[c]);
        for (int a = 0; a < 8; ++a) hCellNodes[8*c + a] = hv[a];
    }

    std::vector<unsigned char> hIsBnd(nNodes, 0);
    for (int node : dm.boundaryDofs) {
        if (node >= 0 && node < nNodes) hIsBnd[node] = 1;
    }

    std::vector<int> hBoundaryNodes(dm.boundaryDofs.begin(), dm.boundaryDofs.end());

    std::vector<int> hFlatCols;
    hFlatCols.reserve(hostSys.A.pattern->nnz());
    for (int r = 0; r < hostSys.A.pattern->nRows(); ++r) {
        const auto& c = sparse_cols_row(hostSys.A, r);
        for (int k = 0; k < (int)c.size(); ++k) hFlatCols.push_back(c[k]);
    }

    cache.dPoints = device_alloc_and_copy(hPoints);
    cache.dCellNodes = device_alloc_and_copy(hCellNodes);
    cache.dIsBnd = device_alloc_and_copy(hIsBnd);
    cache.dBoundaryNodes = device_alloc_and_copy(hBoundaryNodes);
    cache.dRowPtr = device_alloc_and_copy(hostSys.A.pattern->rowPtr);
    cache.dCols = device_alloc_and_copy(hFlatCols);

    Q1Q1_CUDA_CHECK(cudaMalloc(&cache.dCellSlots, sizeof(Q1Q1NseCellSlots) * cellSlots.size()));
    Q1Q1_CUDA_CHECK(cudaMemcpy(
        cache.dCellSlots,
        cellSlots.data(),
        sizeof(Q1Q1NseCellSlots) * cellSlots.size(),
        cudaMemcpyHostToDevice));

    Q1Q1_CUDA_CHECK(cudaMalloc(&cache.dBeta, sizeof(Real) * nRows));
    Q1Q1_CUDA_CHECK(cudaMalloc(&cache.dVals, sizeof(Real) * nnz));
    Q1Q1_CUDA_CHECK(cudaMalloc(&cache.dRhs, sizeof(Real) * nRows));

    cache.valid = true;
}

} // namespace

void q1q1_cuda_audit_nse_assembly(
    const PolyMesh& mesh,
    const LinearCgDofMap& dm,
    const std::vector<Real>& beta,
    const Q1Q1NsePicardOptions& opt,
    const Q1Q1AlgInfo& info,
    AssembledSystem& hostSys
) {
    if (!hostSys.A.fixedPattern || !hostSys.A.pattern || hostSys.A.flatVals.empty()) {
        std::cout << "q1q1CudaAuditAssembly     = skipped_not_fixed_flat_pattern\n";
        return;
    }

    const int nNodes = dm.nDofs;
    const int nRows = 4 * nNodes;
    const int nCells = (int)mesh.cells.size();
    const int nnz = (int)hostSys.A.flatVals.size();

    const auto& cellSlots = q1q1_get_nse_cell_slots_cached(
        mesh, dm, info.pressurePinNode, hostSys.A
    );

    const bool cacheBuffers = memoirs_env_bool("MEMOIRS_Q1Q1_CUDA_CACHE_BUFFERS", true);

    Q1Q1CudaAssemblyCache localCache;
    Q1Q1CudaAssemblyCache& cache =
        cacheBuffers ? q1q1_cuda_assembly_cache_singleton() : localCache;

    q1q1_cuda_prepare_assembly_cache(cache, mesh, dm, hostSys, cellSlots);

    cudaEvent_t ev0 = nullptr, ev1 = nullptr;

    try {
        cudaEvent_t evBeta = nullptr;
        cudaEvent_t evZero = nullptr;
        cudaEvent_t evKernel = nullptr;
        cudaEvent_t evCopy = nullptr;

        Q1Q1_CUDA_CHECK(cudaEventCreate(&evBeta));
        Q1Q1_CUDA_CHECK(cudaEventCreate(&evZero));
        Q1Q1_CUDA_CHECK(cudaEventCreate(&evKernel));
        Q1Q1_CUDA_CHECK(cudaEventCreate(&evCopy));

        Q1Q1_CUDA_CHECK(cudaEventCreate(&ev0));
        Q1Q1_CUDA_CHECK(cudaEventCreate(&ev1));

        Q1Q1_CUDA_CHECK(cudaEventRecord(ev0));

        Q1Q1_CUDA_CHECK(cudaMemcpy(
            cache.dBeta,
            beta.data(),
            sizeof(Real) * nRows,
            cudaMemcpyHostToDevice));
        Q1Q1_CUDA_CHECK(cudaEventRecord(evBeta));

        Q1Q1_CUDA_CHECK(cudaMemset(cache.dVals, 0, sizeof(Real) * nnz));
        Q1Q1_CUDA_CHECK(cudaMemset(cache.dRhs, 0, sizeof(Real) * nRows));
        Q1Q1_CUDA_CHECK(cudaEventRecord(evZero));

        const double tauSupg = (opt.supg != 0) ? info.tau * opt.supgTauScale : 0.0;

        const bool useQpKernel = memoirs_env_bool("MEMOIRS_Q1Q1_CUDA_QP_KERNEL", true);

        if (useQpKernel) {
            const int block = 256;
            const int total = nCells * 64;
            const int grid = (total + block - 1) / block;

            q1q1_cuda_assemble_kernel_qp64<<<grid, block>>>(
                nCells,
                cache.dPoints,
                cache.dCellNodes,
                cache.dIsBnd,
                cache.dCellSlots,
                cache.dBeta,
                opt.nu,
                opt.dt,
                opt.advScale,
                info.tau,
                tauSupg,
                info.pressurePinNode,
                cache.dVals,
                cache.dRhs
            );
        } else {
            const int block = 128;
            const int grid = (nCells + block - 1) / block;

            q1q1_cuda_assemble_kernel<<<grid, block>>>(
                nCells,
                cache.dPoints,
                cache.dCellNodes,
                cache.dIsBnd,
                cache.dCellSlots,
                cache.dBeta,
                opt.nu,
                opt.dt,
                opt.advScale,
                info.tau,
                tauSupg,
                info.pressurePinNode,
                cache.dVals,
                cache.dRhs
            );
        }

        Q1Q1_CUDA_CHECK(cudaGetLastError());

        const int bcThreads = cache.nBoundary * 3 + 1;
        const int bcBlock = 128;
        const int bcGrid = (bcThreads + bcBlock - 1) / bcBlock;

        q1q1_cuda_apply_strong_rows_kernel<<<bcGrid, bcBlock>>>(
            cache.nBoundary,
            cache.dBoundaryNodes,
            info.pressurePinNode,
            cache.dPoints,
            cache.dRowPtr,
            cache.dCols,
            cache.dVals,
            cache.dRhs
        );
        Q1Q1_CUDA_CHECK(cudaGetLastError());
        Q1Q1_CUDA_CHECK(cudaEventRecord(evKernel));

        std::vector<Real> gpuVals(nnz);
        std::vector<Real> gpuRhs(nRows);

        Q1Q1_CUDA_CHECK(cudaMemcpy(gpuVals.data(), cache.dVals, sizeof(Real) * nnz, cudaMemcpyDeviceToHost));
        Q1Q1_CUDA_CHECK(cudaMemcpy(gpuRhs.data(), cache.dRhs, sizeof(Real) * nRows, cudaMemcpyDeviceToHost));
        Q1Q1_CUDA_CHECK(cudaEventRecord(evCopy));

        Q1Q1_CUDA_CHECK(cudaEventRecord(ev1));
        Q1Q1_CUDA_CHECK(cudaEventSynchronize(ev1));

        const double betaSeconds = q1q1_cuda_elapsed_seconds(ev0, evBeta);
        const double zeroSeconds = q1q1_cuda_elapsed_seconds(evBeta, evZero);
        const double kernelSecondsOnly = q1q1_cuda_elapsed_seconds(evZero, evKernel);
        const double copySeconds = q1q1_cuda_elapsed_seconds(evKernel, evCopy);
        const double totalSeconds = q1q1_cuda_elapsed_seconds(ev0, ev1);

        cudaEventDestroy(evBeta);
        cudaEventDestroy(evZero);
        cudaEventDestroy(evKernel);
        cudaEventDestroy(evCopy);

        const double ms = 1000.0 * kernelSecondsOnly;

        if (memoirs_env_bool("MEMOIRS_Q1Q1_CUDA_ASSEMBLY", false)) {
            hostSys.A.flatVals.swap(gpuVals);
            hostSys.b.swap(gpuRhs);

            std::cout << "--------------- q1q1 CUDA assembly use ---------------\n";
            std::cout << "q1q1CudaAssemblyUsed      = 1\n";
            std::cout << "q1q1CudaAssemblyRows      = " << nRows << "\n";
            std::cout << "q1q1CudaAssemblyNnz       = " << nnz << "\n";
            std::cout << "q1q1CudaAssemblyCells     = " << nCells << "\n";
            std::cout << "q1q1CudaAssemblyCacheBuffers = " << (cacheBuffers ? 1 : 0) << "\n";
            std::cout << "q1q1CudaAssemblyKernelMode = " << (memoirs_env_bool("MEMOIRS_Q1Q1_CUDA_QP_KERNEL", true) ? "qp64" : "cell") << "\n";
            std::cout << "q1q1CudaAssemblyBetaCopySeconds = " << std::setprecision(16) << betaSeconds << "\n";
            std::cout << "q1q1CudaAssemblyZeroSeconds = " << std::setprecision(16) << zeroSeconds << "\n";
            std::cout << "q1q1CudaAssemblyKernelSeconds = " << std::setprecision(16) << kernelSecondsOnly << "\n";
            std::cout << "q1q1CudaAssemblyD2HSeconds = " << std::setprecision(16) << copySeconds << "\n";
            std::cout << "q1q1CudaAssemblyTotalSeconds = " << std::setprecision(16) << totalSeconds << "\n";
            std::cout << "-----------------------------------------------------\n";
            cudaEventDestroy(ev0);
            cudaEventDestroy(ev1);
            return;
        }

        std::cout << "--------------- q1q1 CUDA assembly audit -------------\n";
        std::cout << "q1q1CudaAuditAssembly     = 1\n";
        std::cout << "q1q1CudaAuditRows         = " << nRows << "\n";
        std::cout << "q1q1CudaAuditNnz          = " << nnz << "\n";
        std::cout << "q1q1CudaAuditCells        = " << nCells << "\n";
        std::cout << "q1q1CudaAuditCacheBuffers = " << (cacheBuffers ? 1 : 0) << "\n";
        std::cout << "q1q1CudaAuditKernelMode   = " << (memoirs_env_bool("MEMOIRS_Q1Q1_CUDA_QP_KERNEL", true) ? "qp64" : "cell") << "\n";

        double maxA = 0.0, maxAref = 0.0;
        int maxAi = -1;
        for (int i = 0; i < nnz; ++i) {
            const double ref = (double)hostSys.A.flatVals[i];
            const double got = (double)gpuVals[i];
            const double diff = std::abs(got - ref);
            if (diff > maxA) {
                maxA = diff;
                maxAi = i;
            }
            maxAref = std::max(maxAref, std::abs(ref));
        }

        double maxB = 0.0, maxBref = 0.0;
        int maxBi = -1;
        for (int i = 0; i < nRows; ++i) {
            const double ref = (double)hostSys.b[i];
            const double got = (double)gpuRhs[i];
            const double diff = std::abs(got - ref);
            if (diff > maxB) {
                maxB = diff;
                maxBi = i;
            }
            maxBref = std::max(maxBref, std::abs(ref));
        }

        std::cout << "q1q1CudaAuditBetaCopySeconds = " << std::setprecision(16) << betaSeconds << "\n";
        std::cout << "q1q1CudaAuditZeroSeconds = " << std::setprecision(16) << zeroSeconds << "\n";
        std::cout << "q1q1CudaAuditKernelSeconds= " << std::setprecision(16) << kernelSecondsOnly << "\n";
        std::cout << "q1q1CudaAuditD2HSeconds  = " << std::setprecision(16) << copySeconds << "\n";
        std::cout << "q1q1CudaAuditTotalSeconds= " << std::setprecision(16) << totalSeconds << "\n";
        std::cout << "q1q1CudaAuditAmaxAbs      = " << std::setprecision(16) << maxA << "\n";
        std::cout << "q1q1CudaAuditAmaxRel      = " << std::setprecision(16) << (maxA / std::max(maxAref, 1.0e-300)) << "\n";
        std::cout << "q1q1CudaAuditAmaxIndex    = " << maxAi << "\n";
        std::cout << "q1q1CudaAuditBmaxAbs      = " << std::setprecision(16) << maxB << "\n";
        std::cout << "q1q1CudaAuditBmaxRel      = " << std::setprecision(16) << (maxB / std::max(maxBref, 1.0e-300)) << "\n";
        std::cout << "q1q1CudaAuditBmaxIndex    = " << maxBi << "\n";
        std::cout << "-----------------------------------------------------\n";

        cudaEventDestroy(ev0);
        cudaEventDestroy(ev1);
    } catch (...) {
        if (ev0) cudaEventDestroy(ev0);
        if (ev1) cudaEventDestroy(ev1);
        throw;
    }
}


void q1q1_cuda_assemble_cavity_bdf1(
    const PolyMesh& mesh,
    const LinearCgDofMap& dm,
    const std::vector<Real>& beta,
    const std::vector<Real>& oldState,
    const Q1Q1NsePicardOptions& opt,
    const Q1Q1AlgInfo& info,
    double lidUx,
    double lidUy,
    double lidUz,
    AssembledSystem& hostSys
) {
    if (!hostSys.A.fixedPattern || !hostSys.A.pattern || hostSys.A.flatVals.empty()) {
        throw std::runtime_error("CUDA cavity BDF1 requires fixed flat pattern.");
    }

    const int nNodes = dm.nDofs;
    const int nRows = 4 * nNodes;
    const int nCells = (int)mesh.cells.size();
    const int nnz = (int)hostSys.A.flatVals.size();

    if ((int)beta.size() != nRows) {
        throw std::runtime_error("Bad beta size in CUDA cavity BDF1.");
    }
    if ((int)oldState.size() != nRows) {
        throw std::runtime_error("Bad oldState size in CUDA cavity BDF1.");
    }

    const auto& cellSlots = q1q1_get_nse_cell_slots_cached(
        mesh, dm, info.pressurePinNode, hostSys.A
    );

    Q1Q1CudaAssemblyCache& cache = q1q1_cuda_assembly_cache_singleton();
    q1q1_cuda_prepare_assembly_cache(cache, mesh, dm, hostSys, cellSlots);

    std::vector<Real> hBcUx(nNodes, Real(0));
    std::vector<Real> hBcUy(nNodes, Real(0));
    std::vector<Real> hBcUz(nNodes, Real(0));

    int lidCount = 0;
    int wallCount = 0;

    for (int node : dm.boundaryDofs) {
        const Vec3 g = q1q1_cavity_lid_velocity_at_node(mesh, node, lidUx, lidUy, lidUz);

        hBcUx[node] = Real(g.x);
        hBcUy[node] = Real(g.y);
        hBcUz[node] = Real(g.z);

        if (g.x != 0.0 || g.y != 0.0 || g.z != 0.0) {
            lidCount++;
        } else {
            wallCount++;
        }
    }

    Real* dOldState = nullptr;
    Real* dBcUx = nullptr;
    Real* dBcUy = nullptr;
    Real* dBcUz = nullptr;

    cudaEvent_t ev0 = nullptr;
    cudaEvent_t evCopyIn = nullptr;
    cudaEvent_t evZero = nullptr;
    cudaEvent_t evKernel = nullptr;
    cudaEvent_t evCopyOut = nullptr;
    cudaEvent_t ev1 = nullptr;

    try {
        Q1Q1_CUDA_CHECK(cudaMalloc(&dOldState, sizeof(Real) * nRows));
        Q1Q1_CUDA_CHECK(cudaMalloc(&dBcUx, sizeof(Real) * nNodes));
        Q1Q1_CUDA_CHECK(cudaMalloc(&dBcUy, sizeof(Real) * nNodes));
        Q1Q1_CUDA_CHECK(cudaMalloc(&dBcUz, sizeof(Real) * nNodes));

        Q1Q1_CUDA_CHECK(cudaEventCreate(&ev0));
        Q1Q1_CUDA_CHECK(cudaEventCreate(&evCopyIn));
        Q1Q1_CUDA_CHECK(cudaEventCreate(&evZero));
        Q1Q1_CUDA_CHECK(cudaEventCreate(&evKernel));
        Q1Q1_CUDA_CHECK(cudaEventCreate(&evCopyOut));
        Q1Q1_CUDA_CHECK(cudaEventCreate(&ev1));

        Q1Q1_CUDA_CHECK(cudaEventRecord(ev0));

        Q1Q1_CUDA_CHECK(cudaMemcpy(cache.dBeta, beta.data(), sizeof(Real) * nRows, cudaMemcpyHostToDevice));
        Q1Q1_CUDA_CHECK(cudaMemcpy(dOldState, oldState.data(), sizeof(Real) * nRows, cudaMemcpyHostToDevice));
        Q1Q1_CUDA_CHECK(cudaMemcpy(dBcUx, hBcUx.data(), sizeof(Real) * nNodes, cudaMemcpyHostToDevice));
        Q1Q1_CUDA_CHECK(cudaMemcpy(dBcUy, hBcUy.data(), sizeof(Real) * nNodes, cudaMemcpyHostToDevice));
        Q1Q1_CUDA_CHECK(cudaMemcpy(dBcUz, hBcUz.data(), sizeof(Real) * nNodes, cudaMemcpyHostToDevice));
        Q1Q1_CUDA_CHECK(cudaEventRecord(evCopyIn));

        Q1Q1_CUDA_CHECK(cudaMemset(cache.dVals, 0, sizeof(Real) * nnz));
        Q1Q1_CUDA_CHECK(cudaMemset(cache.dRhs, 0, sizeof(Real) * nRows));
        Q1Q1_CUDA_CHECK(cudaEventRecord(evZero));

        const int block = 256;
        const int qOrder = q1q1_cuda_cavity_quad_order_from_env();
        const int total = nCells * qOrder * qOrder * qOrder;
        const int grid = (total + block - 1) / block;

        const double tauSupg = (opt.supg != 0) ? info.tau * opt.supgTauScale : 0.0;

        q1q1_cuda_assemble_cavity_bdf1_qp64<<<grid, block>>>(
            nCells,
            cache.dPoints,
            cache.dCellNodes,
            cache.dIsBnd,
            cache.dCellSlots,
            cache.dBeta,
            dOldState,
            dBcUx,
            dBcUy,
            dBcUz,
            opt.nu,
            opt.dt,
            opt.advScale,
            info.tau,
            tauSupg,
            info.pressurePinNode,
            cache.dVals,
            cache.dRhs
        , qOrder);
        Q1Q1_CUDA_CHECK(cudaGetLastError());

        const int bcThreads = cache.nBoundary * 3 + 1;
        const int bcBlock = 128;
        const int bcGrid = (bcThreads + bcBlock - 1) / bcBlock;

        q1q1_cuda_apply_cavity_strong_rows_kernel<<<bcGrid, bcBlock>>>(
            cache.nBoundary,
            cache.dBoundaryNodes,
            info.pressurePinNode,
            dBcUx,
            dBcUy,
            dBcUz,
            cache.dRowPtr,
            cache.dCols,
            cache.dVals,
            cache.dRhs
        );
        Q1Q1_CUDA_CHECK(cudaGetLastError());
        Q1Q1_CUDA_CHECK(cudaEventRecord(evKernel));

        hostSys.A.flatVals.assign(nnz, Real(0));
        hostSys.b.assign(nRows, Real(0));

        Q1Q1_CUDA_CHECK(cudaMemcpy(hostSys.A.flatVals.data(), cache.dVals, sizeof(Real) * nnz, cudaMemcpyDeviceToHost));
        Q1Q1_CUDA_CHECK(cudaMemcpy(hostSys.b.data(), cache.dRhs, sizeof(Real) * nRows, cudaMemcpyDeviceToHost));
        Q1Q1_CUDA_CHECK(cudaEventRecord(evCopyOut));

        Q1Q1_CUDA_CHECK(cudaEventRecord(ev1));
        Q1Q1_CUDA_CHECK(cudaEventSynchronize(ev1));

        const double copyInSeconds = q1q1_cuda_elapsed_seconds(ev0, evCopyIn);
        const double zeroSeconds = q1q1_cuda_elapsed_seconds(evCopyIn, evZero);
        const double kernelSeconds = q1q1_cuda_elapsed_seconds(evZero, evKernel);
        const double d2hSeconds = q1q1_cuda_elapsed_seconds(evKernel, evCopyOut);
        const double totalSeconds = q1q1_cuda_elapsed_seconds(ev0, ev1);

        if (memoirs_env_bool("MEMOIRS_Q1Q1_CAVITY_ASSEMBLY_DIAG", false)) {
        std::cout << "--------------- q1q1 CUDA cavity BDF1 assembly --------\n";
        std::cout << "q1q1CudaCavityAssemblyUsed = 1\n";
        std::cout << "q1q1CudaCavityRows      = " << nRows << "\n";
        std::cout << "q1q1CudaCavityNnz       = " << nnz << "\n";
        std::cout << "q1q1CudaCavityCells     = " << nCells << "\n";
        std::cout << "q1q1CudaCavityKernelMode = " << q1q1_cuda_cavity_quad_label(q1q1_cuda_cavity_quad_order_from_env()) << "\n";
        std::cout << "q1q1CudaCavityLidInteriorNodes = " << lidCount << "\n";
        std::cout << "q1q1CudaCavityWallBoundaryNodes = " << wallCount << "\n";
        std::cout << "q1q1CavityRimPolicy     = top_rim_no_slip_wall\n";
        std::cout << "q1q1CudaCavityCopyInSeconds = " << std::setprecision(16) << copyInSeconds << "\n";
        std::cout << "q1q1CudaCavityZeroSeconds = " << std::setprecision(16) << zeroSeconds << "\n";
        std::cout << "q1q1CudaCavityKernelSeconds = " << std::setprecision(16) << kernelSeconds << "\n";
        std::cout << "q1q1CudaCavityD2HSeconds = " << std::setprecision(16) << d2hSeconds << "\n";
        std::cout << "q1q1CudaCavityTotalSeconds = " << std::setprecision(16) << totalSeconds << "\n";
        std::cout << "-----------------------------------------------------\n";
        }

    } catch (...) {
        cudaFree(dOldState);
        cudaFree(dBcUx);
        cudaFree(dBcUy);
        cudaFree(dBcUz);

        if (ev0) cudaEventDestroy(ev0);
        if (evCopyIn) cudaEventDestroy(evCopyIn);
        if (evZero) cudaEventDestroy(evZero);
        if (evKernel) cudaEventDestroy(evKernel);
        if (evCopyOut) cudaEventDestroy(evCopyOut);
        if (ev1) cudaEventDestroy(ev1);
        throw;
    }

    cudaFree(dOldState);
    cudaFree(dBcUx);
    cudaFree(dBcUy);
    cudaFree(dBcUz);

    if (ev0) cudaEventDestroy(ev0);
    if (evCopyIn) cudaEventDestroy(evCopyIn);
    if (evZero) cudaEventDestroy(evZero);
    if (evKernel) cudaEventDestroy(evKernel);
    if (evCopyOut) cudaEventDestroy(evCopyOut);
    if (ev1) cudaEventDestroy(ev1);
}


void q1q1_cuda_assemble_cavity_bdf_history(
    const PolyMesh& mesh,
    const LinearCgDofMap& dm,
    const std::vector<Real>& beta,
    const std::vector<Real>& oldState,
    const std::vector<Real>& olderState,
    const Q1Q1NsePicardOptions& opt,
    const Q1Q1AlgInfo& info,
    double lidUx,
    double lidUy,
    double lidUz,
    bool useBdf2,
    AssembledSystem& hostSys
) {
    if (!hostSys.A.fixedPattern || !hostSys.A.pattern || hostSys.A.flatVals.empty()) {
        throw std::runtime_error("CUDA cavity BDF history requires fixed flat pattern.");
    }

    const int nNodes = dm.nDofs;
    const int nRows = 4 * nNodes;
    const int nCells = (int)mesh.cells.size();
    const int nnz = (int)hostSys.A.flatVals.size();

    if ((int)beta.size() != nRows) throw std::runtime_error("Bad beta size in CUDA cavity BDF history.");
    if ((int)oldState.size() != nRows) throw std::runtime_error("Bad oldState size in CUDA cavity BDF history.");
    if ((int)olderState.size() != nRows) throw std::runtime_error("Bad olderState size in CUDA cavity BDF history.");

    const double invDt = 1.0 / opt.dt;
    const double lhsCoeff   = useBdf2 ? (1.5 * invDt)  : invDt;
    const double oldCoeff   = useBdf2 ? (2.0 * invDt)  : invDt;
    const double olderCoeff = useBdf2 ? (-0.5 * invDt) : 0.0;

    const auto& cellSlots = q1q1_get_nse_cell_slots_cached(
        mesh, dm, info.pressurePinNode, hostSys.A
    );

    Q1Q1CudaAssemblyCache& cache = q1q1_cuda_assembly_cache_singleton();
    q1q1_cuda_prepare_assembly_cache(cache, mesh, dm, hostSys, cellSlots);

    std::vector<Real> hBcUx(nNodes, Real(0));
    std::vector<Real> hBcUy(nNodes, Real(0));
    std::vector<Real> hBcUz(nNodes, Real(0));

    int lidCount = 0;
    int wallCount = 0;

    for (int node : dm.boundaryDofs) {
        const Vec3 g = q1q1_cavity_lid_velocity_at_node(mesh, node, lidUx, lidUy, lidUz);

        hBcUx[node] = Real(g.x);
        hBcUy[node] = Real(g.y);
        hBcUz[node] = Real(g.z);

        if (g.x != 0.0 || g.y != 0.0 || g.z != 0.0) lidCount++;
        else wallCount++;
    }

    Real* dOldState = nullptr;
    Real* dOlderState = nullptr;
    Real* dBcUx = nullptr;
    Real* dBcUy = nullptr;
    Real* dBcUz = nullptr;

    cudaEvent_t ev0 = nullptr;
    cudaEvent_t evCopyIn = nullptr;
    cudaEvent_t evZero = nullptr;
    cudaEvent_t evKernel = nullptr;
    cudaEvent_t evCopyOut = nullptr;
    cudaEvent_t ev1 = nullptr;

    try {
        Q1Q1_CUDA_CHECK(cudaMalloc(&dOldState, sizeof(Real) * nRows));
        Q1Q1_CUDA_CHECK(cudaMalloc(&dOlderState, sizeof(Real) * nRows));
        Q1Q1_CUDA_CHECK(cudaMalloc(&dBcUx, sizeof(Real) * nNodes));
        Q1Q1_CUDA_CHECK(cudaMalloc(&dBcUy, sizeof(Real) * nNodes));
        Q1Q1_CUDA_CHECK(cudaMalloc(&dBcUz, sizeof(Real) * nNodes));

        Q1Q1_CUDA_CHECK(cudaEventCreate(&ev0));
        Q1Q1_CUDA_CHECK(cudaEventCreate(&evCopyIn));
        Q1Q1_CUDA_CHECK(cudaEventCreate(&evZero));
        Q1Q1_CUDA_CHECK(cudaEventCreate(&evKernel));
        Q1Q1_CUDA_CHECK(cudaEventCreate(&evCopyOut));
        Q1Q1_CUDA_CHECK(cudaEventCreate(&ev1));

        Q1Q1_CUDA_CHECK(cudaEventRecord(ev0));

        Q1Q1_CUDA_CHECK(cudaMemcpy(cache.dBeta, beta.data(), sizeof(Real) * nRows, cudaMemcpyHostToDevice));
        Q1Q1_CUDA_CHECK(cudaMemcpy(dOldState, oldState.data(), sizeof(Real) * nRows, cudaMemcpyHostToDevice));
        Q1Q1_CUDA_CHECK(cudaMemcpy(dOlderState, olderState.data(), sizeof(Real) * nRows, cudaMemcpyHostToDevice));
        Q1Q1_CUDA_CHECK(cudaMemcpy(dBcUx, hBcUx.data(), sizeof(Real) * nNodes, cudaMemcpyHostToDevice));
        Q1Q1_CUDA_CHECK(cudaMemcpy(dBcUy, hBcUy.data(), sizeof(Real) * nNodes, cudaMemcpyHostToDevice));
        Q1Q1_CUDA_CHECK(cudaMemcpy(dBcUz, hBcUz.data(), sizeof(Real) * nNodes, cudaMemcpyHostToDevice));
        Q1Q1_CUDA_CHECK(cudaEventRecord(evCopyIn));

        Q1Q1_CUDA_CHECK(cudaMemset(cache.dVals, 0, sizeof(Real) * nnz));
        Q1Q1_CUDA_CHECK(cudaMemset(cache.dRhs, 0, sizeof(Real) * nRows));
        Q1Q1_CUDA_CHECK(cudaEventRecord(evZero));

        const int block = 256;
        const int qOrder = q1q1_cuda_cavity_quad_order_from_env();
        const int total = nCells * qOrder * qOrder * qOrder;
        const int grid = (total + block - 1) / block;

        const double tauSupg = (opt.supg != 0) ? info.tau * opt.supgTauScale : 0.0;

        q1q1_cuda_assemble_cavity_bdf_history_qp64<<<grid, block>>>(
            nCells,
            cache.dPoints,
            cache.dCellNodes,
            cache.dIsBnd,
            cache.dCellSlots,
            cache.dBeta,
            dOldState,
            dOlderState,
            dBcUx,
            dBcUy,
            dBcUz,
            opt.nu,
            opt.advScale,
            info.tau,
            tauSupg,
            lhsCoeff,
            oldCoeff,
            olderCoeff,
            info.pressurePinNode,
            cache.dVals,
            cache.dRhs
        , qOrder);
        Q1Q1_CUDA_CHECK(cudaGetLastError());

        const int bcThreads = cache.nBoundary * 3 + 1;
        const int bcBlock = 128;
        const int bcGrid = (bcThreads + bcBlock - 1) / bcBlock;

        q1q1_cuda_apply_cavity_strong_rows_kernel<<<bcGrid, bcBlock>>>(
            cache.nBoundary,
            cache.dBoundaryNodes,
            info.pressurePinNode,
            dBcUx,
            dBcUy,
            dBcUz,
            cache.dRowPtr,
            cache.dCols,
            cache.dVals,
            cache.dRhs
        );
        Q1Q1_CUDA_CHECK(cudaGetLastError());
        Q1Q1_CUDA_CHECK(cudaEventRecord(evKernel));

        hostSys.A.flatVals.assign(nnz, Real(0));
        hostSys.b.assign(nRows, Real(0));

        Q1Q1_CUDA_CHECK(cudaMemcpy(hostSys.A.flatVals.data(), cache.dVals, sizeof(Real) * nnz, cudaMemcpyDeviceToHost));
        Q1Q1_CUDA_CHECK(cudaMemcpy(hostSys.b.data(), cache.dRhs, sizeof(Real) * nRows, cudaMemcpyDeviceToHost));
        Q1Q1_CUDA_CHECK(cudaEventRecord(evCopyOut));

        Q1Q1_CUDA_CHECK(cudaEventRecord(ev1));
        Q1Q1_CUDA_CHECK(cudaEventSynchronize(ev1));

        const double copyInSeconds = q1q1_cuda_elapsed_seconds(ev0, evCopyIn);
        const double zeroSeconds = q1q1_cuda_elapsed_seconds(evCopyIn, evZero);
        const double kernelSeconds = q1q1_cuda_elapsed_seconds(evZero, evKernel);
        const double d2hSeconds = q1q1_cuda_elapsed_seconds(evKernel, evCopyOut);
        const double totalSeconds = q1q1_cuda_elapsed_seconds(ev0, ev1);

        if (memoirs_env_bool("MEMOIRS_Q1Q1_CAVITY_ASSEMBLY_DIAG", false)) {
        std::cout << "--------------- q1q1 CUDA cavity BDF assembly --------\n";
        std::cout << "q1q1CudaCavityAssemblyUsed = 1\n";
        std::cout << "q1q1CudaCavityTimeScheme = " << (useBdf2 ? "bdf2" : "bdf1_startup") << "\n";
        std::cout << "q1q1CudaCavityRows      = " << nRows << "\n";
        std::cout << "q1q1CudaCavityNnz       = " << nnz << "\n";
        std::cout << "q1q1CudaCavityCells     = " << nCells << "\n";
        std::cout << "q1q1CudaCavityKernelMode = " << q1q1_cuda_cavity_quad_label(q1q1_cuda_cavity_quad_order_from_env()) << "\n";
        std::cout << "q1q1CudaCavityLidInteriorNodes = " << lidCount << "\n";
        std::cout << "q1q1CudaCavityWallBoundaryNodes = " << wallCount << "\n";
        std::cout << "q1q1CavityRimPolicy     = top_rim_no_slip_wall\n";
        std::cout << "q1q1CudaCavityLhsCoeff  = " << std::setprecision(16) << lhsCoeff << "\n";
        std::cout << "q1q1CudaCavityOldCoeff  = " << std::setprecision(16) << oldCoeff << "\n";
        std::cout << "q1q1CudaCavityOlderCoeff = " << std::setprecision(16) << olderCoeff << "\n";
        std::cout << "q1q1CudaCavityCopyInSeconds = " << std::setprecision(16) << copyInSeconds << "\n";
        std::cout << "q1q1CudaCavityZeroSeconds = " << std::setprecision(16) << zeroSeconds << "\n";
        std::cout << "q1q1CudaCavityKernelSeconds = " << std::setprecision(16) << kernelSeconds << "\n";
        std::cout << "q1q1CudaCavityD2HSeconds = " << std::setprecision(16) << d2hSeconds << "\n";
        std::cout << "q1q1CudaCavityTotalSeconds = " << std::setprecision(16) << totalSeconds << "\n";
        std::cout << "-----------------------------------------------------\n";
        }

    } catch (...) {
        cudaFree(dOldState);
        cudaFree(dOlderState);
        cudaFree(dBcUx);
        cudaFree(dBcUy);
        cudaFree(dBcUz);

        if (ev0) cudaEventDestroy(ev0);
        if (evCopyIn) cudaEventDestroy(evCopyIn);
        if (evZero) cudaEventDestroy(evZero);
        if (evKernel) cudaEventDestroy(evKernel);
        if (evCopyOut) cudaEventDestroy(evCopyOut);
        if (ev1) cudaEventDestroy(ev1);
        throw;
    }

    cudaFree(dOldState);
    cudaFree(dOlderState);
    cudaFree(dBcUx);
    cudaFree(dBcUy);
    cudaFree(dBcUz);

    if (ev0) cudaEventDestroy(ev0);
    if (evCopyIn) cudaEventDestroy(evCopyIn);
    if (evZero) cudaEventDestroy(evZero);
    if (evKernel) cudaEventDestroy(evKernel);
    if (evCopyOut) cudaEventDestroy(evCopyOut);
    if (ev1) cudaEventDestroy(ev1);
}


Q1Q1NonlinearResidualReport q1q1_cuda_cavity_nonlinear_residual(
    const PolyMesh& mesh,
    const LinearCgDofMap& dm,
    const std::vector<Real>& solution,
    const std::vector<Real>& oldState,
    const std::vector<Real>& olderState,
    const Q1Q1NsePicardOptions& opt,
    const Q1Q1AlgInfo& info,
    double lidUx,
    double lidUy,
    double lidUz,
    bool useBdf2,
    const AssembledSystem& patternSys
) {
    if (!patternSys.A.fixedPattern || !patternSys.A.pattern || patternSys.A.flatVals.empty()) {
        throw std::runtime_error("CUDA nonlinear residual requires fixed flat pattern.");
    }

    const int nNodes = dm.nDofs;
    const int nRows = 4 * nNodes;
    const int nCells = (int)mesh.cells.size();
    const int nnz = (int)patternSys.A.flatVals.size();

    if ((int)solution.size() != nRows) throw std::runtime_error("Bad solution size in nonlinear residual.");
    if ((int)oldState.size() != nRows) throw std::runtime_error("Bad oldState size in nonlinear residual.");
    if ((int)olderState.size() != nRows) throw std::runtime_error("Bad olderState size in nonlinear residual.");

    const double invDt = 1.0 / opt.dt;
    const double lhsCoeff   = useBdf2 ? (1.5 * invDt)  : invDt;
    const double oldCoeff   = useBdf2 ? (2.0 * invDt)  : invDt;
    const double olderCoeff = useBdf2 ? (-0.5 * invDt) : 0.0;

    const auto& cellSlots = q1q1_get_nse_cell_slots_cached(
        mesh, dm, info.pressurePinNode, patternSys.A
    );

    Q1Q1CudaAssemblyCache& cache = q1q1_cuda_assembly_cache_singleton();
    q1q1_cuda_prepare_assembly_cache(cache, mesh, dm, patternSys, cellSlots);

    std::vector<Real> hBcUx(nNodes, Real(0));
    std::vector<Real> hBcUy(nNodes, Real(0));
    std::vector<Real> hBcUz(nNodes, Real(0));

    for (int node : dm.boundaryDofs) {
        const Vec3 g = q1q1_cavity_lid_velocity_at_node(mesh, node, lidUx, lidUy, lidUz);
        hBcUx[node] = Real(g.x);
        hBcUy[node] = Real(g.y);
        hBcUz[node] = Real(g.z);
    }

    Real* dOldState = nullptr;
    Real* dOlderState = nullptr;
    Real* dBcUx = nullptr;
    Real* dBcUy = nullptr;
    Real* dBcUz = nullptr;
    Real* dX = nullptr;
    double* dSums = nullptr;

    cudaEvent_t ev0 = nullptr;
    cudaEvent_t ev1 = nullptr;

    Q1Q1NonlinearResidualReport rep;

    try {
        Q1Q1_CUDA_CHECK(cudaMalloc(&dOldState, sizeof(Real) * nRows));
        Q1Q1_CUDA_CHECK(cudaMalloc(&dOlderState, sizeof(Real) * nRows));
        Q1Q1_CUDA_CHECK(cudaMalloc(&dBcUx, sizeof(Real) * nNodes));
        Q1Q1_CUDA_CHECK(cudaMalloc(&dBcUy, sizeof(Real) * nNodes));
        Q1Q1_CUDA_CHECK(cudaMalloc(&dBcUz, sizeof(Real) * nNodes));
        Q1Q1_CUDA_CHECK(cudaMalloc(&dX, sizeof(Real) * nRows));
        Q1Q1_CUDA_CHECK(cudaMalloc(&dSums, sizeof(double) * 3));

        Q1Q1_CUDA_CHECK(cudaEventCreate(&ev0));
        Q1Q1_CUDA_CHECK(cudaEventCreate(&ev1));
        Q1Q1_CUDA_CHECK(cudaEventRecord(ev0));

        // For nonlinear residual, beta is the solved velocity itself.
        Q1Q1_CUDA_CHECK(cudaMemcpy(cache.dBeta, solution.data(), sizeof(Real) * nRows, cudaMemcpyHostToDevice));
        Q1Q1_CUDA_CHECK(cudaMemcpy(dX, solution.data(), sizeof(Real) * nRows, cudaMemcpyHostToDevice));
        Q1Q1_CUDA_CHECK(cudaMemcpy(dOldState, oldState.data(), sizeof(Real) * nRows, cudaMemcpyHostToDevice));
        Q1Q1_CUDA_CHECK(cudaMemcpy(dOlderState, olderState.data(), sizeof(Real) * nRows, cudaMemcpyHostToDevice));
        Q1Q1_CUDA_CHECK(cudaMemcpy(dBcUx, hBcUx.data(), sizeof(Real) * nNodes, cudaMemcpyHostToDevice));
        Q1Q1_CUDA_CHECK(cudaMemcpy(dBcUy, hBcUy.data(), sizeof(Real) * nNodes, cudaMemcpyHostToDevice));
        Q1Q1_CUDA_CHECK(cudaMemcpy(dBcUz, hBcUz.data(), sizeof(Real) * nNodes, cudaMemcpyHostToDevice));

        Q1Q1_CUDA_CHECK(cudaMemset(cache.dVals, 0, sizeof(Real) * nnz));
        Q1Q1_CUDA_CHECK(cudaMemset(cache.dRhs, 0, sizeof(Real) * nRows));

        const int block = 256;
        const int qOrder = q1q1_cuda_cavity_quad_order_from_env();
        const int total = nCells * qOrder * qOrder * qOrder;
        const int grid = (total + block - 1) / block;

        const double tauSupg = (opt.supg != 0) ? info.tau * opt.supgTauScale : 0.0;

        q1q1_cuda_assemble_cavity_bdf_history_qp64<<<grid, block>>>(
            nCells,
            cache.dPoints,
            cache.dCellNodes,
            cache.dIsBnd,
            cache.dCellSlots,
            cache.dBeta,
            dOldState,
            dOlderState,
            dBcUx,
            dBcUy,
            dBcUz,
            opt.nu,
            opt.advScale,
            info.tau,
            tauSupg,
            lhsCoeff,
            oldCoeff,
            olderCoeff,
            info.pressurePinNode,
            cache.dVals,
            cache.dRhs
        , qOrder);
        Q1Q1_CUDA_CHECK(cudaGetLastError());

        const int bcThreads = cache.nBoundary * 3 + 1;
        const int bcBlock = 128;
        const int bcGrid = (bcThreads + bcBlock - 1) / bcBlock;

        q1q1_cuda_apply_cavity_strong_rows_kernel<<<bcGrid, bcBlock>>>(
            cache.nBoundary,
            cache.dBoundaryNodes,
            info.pressurePinNode,
            dBcUx,
            dBcUy,
            dBcUz,
            cache.dRowPtr,
            cache.dCols,
            cache.dVals,
            cache.dRhs
        );
        Q1Q1_CUDA_CHECK(cudaGetLastError());

        Q1Q1_CUDA_CHECK(cudaMemset(dSums, 0, sizeof(double) * 3));

        const int rBlock = 256;
        const int rGrid = (nRows + rBlock - 1) / rBlock;

        q1q1_cuda_residual_l2_kernel<<<rGrid, rBlock>>>(
            nRows,
            cache.dRowPtr,
            cache.dCols,
            cache.dVals,
            cache.dRhs,
            dX,
            cache.dIsBnd,
            info.pressurePinNode,
            dSums
        );
        Q1Q1_CUDA_CHECK(cudaGetLastError());

        double hSums[3] = {0.0, 0.0, 0.0};
        Q1Q1_CUDA_CHECK(cudaMemcpy(hSums, dSums, sizeof(double) * 3, cudaMemcpyDeviceToHost));

        Q1Q1_CUDA_CHECK(cudaEventRecord(ev1));
        Q1Q1_CUDA_CHECK(cudaEventSynchronize(ev1));

        rep.absL2 = std::sqrt(std::max(0.0, hSums[0]));
        rep.rhsL2 = std::sqrt(std::max(0.0, hSums[1]));
        rep.axL2  = std::sqrt(std::max(0.0, hSums[2]));

        const double denom = std::max({1.0e-300, rep.rhsL2, rep.axL2});
        rep.relL2 = rep.absL2 / denom;
        rep.seconds = q1q1_cuda_elapsed_seconds(ev0, ev1);

    } catch (...) {
        cudaFree(dOldState);
        cudaFree(dOlderState);
        cudaFree(dBcUx);
        cudaFree(dBcUy);
        cudaFree(dBcUz);
        cudaFree(dX);
        cudaFree(dSums);

        if (ev0) cudaEventDestroy(ev0);
        if (ev1) cudaEventDestroy(ev1);
        throw;
    }

    cudaFree(dOldState);
    cudaFree(dOlderState);
    cudaFree(dBcUx);
    cudaFree(dBcUy);
    cudaFree(dBcUz);
    cudaFree(dX);
    cudaFree(dSums);

    if (ev0) cudaEventDestroy(ev0);
    if (ev1) cudaEventDestroy(ev1);

    return rep;
}
