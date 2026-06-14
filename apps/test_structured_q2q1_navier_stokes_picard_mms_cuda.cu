#include "memoirs/gpu/CudaUtils.cuh"
#include "memoirs/gpu/DeviceBuffer.cuh"
#include "memoirs/gpu/StructuredGmgCuda.cuh"
#include "memoirs/solvers/DenseLu.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

#if defined(MEMOIRS_PRECISION_single)
using Real = float;
static constexpr const char* kPrecisionName = "single";
#else
using Real = double;
static constexpr const char* kPrecisionName = "double";
#endif

namespace {

struct Cli {
    int nElem = 8;
    int maxit = 4000;
    int gmresRestart = 60;
    double tol = 1e-10;
    double nu = 1.0;
    double tauScale = 1.0;
    std::string tauMode = "metric";
    double tauC = 4.0;
    double tauDt = 0.0;
    double tauAdvMag = 0.0;
    int printEvery = 100;
    int repeats = 1;
    int rhsMode = 0; // 0 continuous, 1 algebraic A*x_exact for the current Picard operator
    std::string solver = "gmres";
    double dt = 1.0;
    double advScale = 0.05;
    int maxPicard = 20;
    double picardTol = 1e-8;
    int betaInitial = 0; // 0 zero, 1 exact
    int supg = 0; // 0 off, 1 on
    double supgTauScale = 1.0;
    std::string prec = "block_gmg"; // block_gmg, operator_inverse_diagonal, or coupled_gmg
    int mgPre = 2;
    int mgPost = 2;
    double mgOmega = 0.70;
    int mgCoarseElem = 2;
    int mgVerbose = 0;
    std::string mgRebuildPolicy = "every_picard"; // every_picard or every_step
    int timing = 1;
};

static int arg_int(int& i, int argc, char** argv) {
    if (i + 1 >= argc) throw std::runtime_error("missing integer argument");
    return std::atoi(argv[++i]);
}

static double arg_double(int& i, int argc, char** argv) {
    if (i + 1 >= argc) throw std::runtime_error("missing floating argument");
    return std::atof(argv[++i]);
}

static Cli parse_cli(int argc, char** argv) {
    Cli c;
    for (int i = 1; i < argc; ++i) {
        const std::string a(argv[i]);
        if (a == "-n" || a == "-nElem") c.nElem = arg_int(i, argc, argv);
        else if (a == "-maxit") c.maxit = arg_int(i, argc, argv);
        else if (a == "-gmresRestart") c.gmresRestart = arg_int(i, argc, argv);
        else if (a == "-tol") c.tol = arg_double(i, argc, argv);
        else if (a == "-nu") c.nu = arg_double(i, argc, argv);
        else if (a == "-tauScale") c.tauScale = arg_double(i, argc, argv);
        else if (a == "-tauMode") {
            if (i + 1 >= argc) throw std::runtime_error("missing string argument for -tauMode");
            c.tauMode = argv[++i];
        }
        else if (a == "-tauC") c.tauC = arg_double(i, argc, argv);
        else if (a == "-tauDt") c.tauDt = arg_double(i, argc, argv);
        else if (a == "-tauAdvMag") c.tauAdvMag = arg_double(i, argc, argv);
        else if (a == "-solver") {
            if (i + 1 >= argc) throw std::runtime_error("missing string argument for -solver");
            c.solver = argv[++i];
        }
        else if (a == "-prec" || a == "-preconditioner" || a == "-rightPreconditioner") {
            if (i + 1 >= argc) throw std::runtime_error("missing string argument for -prec");
            c.prec = argv[++i];
        }
        else if (a == "-dt") c.dt = arg_double(i, argc, argv);
        else if (a == "-advScale") c.advScale = arg_double(i, argc, argv);
        else if (a == "-maxPicard") c.maxPicard = arg_int(i, argc, argv);
        else if (a == "-picardTol") c.picardTol = arg_double(i, argc, argv);
        else if (a == "-betaInitial") c.betaInitial = arg_int(i, argc, argv);
        else if (a == "-supg") c.supg = arg_int(i, argc, argv);
        else if (a == "-supgTauScale") c.supgTauScale = arg_double(i, argc, argv);
        else if (a == "-mgPre") c.mgPre = arg_int(i, argc, argv);
        else if (a == "-mgPost") c.mgPost = arg_int(i, argc, argv);
        else if (a == "-mgOmega") c.mgOmega = arg_double(i, argc, argv);
        else if (a == "-mgCoarseElem") c.mgCoarseElem = arg_int(i, argc, argv);
        else if (a == "-mgVerbose") c.mgVerbose = arg_int(i, argc, argv);
        else if (a == "-mgRebuildPolicy") {
            if (i + 1 >= argc) throw std::runtime_error("missing string argument for -mgRebuildPolicy");
            c.mgRebuildPolicy = argv[++i];
        }
        else if (a == "-timing") c.timing = arg_int(i, argc, argv);
        else if (a == "-rhsMode") c.rhsMode = arg_int(i, argc, argv);
        else if (a == "-printEvery") c.printEvery = arg_int(i, argc, argv);
        else if (a == "-repeats") c.repeats = arg_int(i, argc, argv);
        else if (a == "-pGradSign" || a == "-pspgRhsSign") {
            // Accepted for command compatibility with Q1/Q1 app; Q2/Q1 reference uses + signs.
            (void)arg_double(i, argc, argv);
        }
        else if (a == "-pspgForceMode") {
            // Accepted for command compatibility. Q2/Q1 PSPG reference always uses fullF.
            if (i + 1 >= argc) throw std::runtime_error("missing string argument for -pspgForceMode");
            ++i;
        }
        else throw std::runtime_error("unknown argument: " + a);
    }
    if (c.nElem < 2) throw std::runtime_error("-n must be >= 2");
    if (c.maxit < 1) throw std::runtime_error("-maxit must be >= 1");
    if (c.gmresRestart < 2) throw std::runtime_error("-gmresRestart must be >= 2");
    if (c.repeats < 1) throw std::runtime_error("-repeats must be >= 1");
    if (!(c.tol > 0)) throw std::runtime_error("-tol must be positive");
    if (!(c.nu > 0)) throw std::runtime_error("-nu must be positive");
    if (!(c.tauScale > 0)) throw std::runtime_error("-tauScale must be positive");
    if (!(c.tauC > 0)) throw std::runtime_error("-tauC must be positive");
    if (c.tauMode != "metric" && c.tauMode != "simple" && c.tauMode != "sphere" && c.tauMode != "tezduyar") {
        throw std::runtime_error("unknown -tauMode; use metric, simple, sphere, or tezduyar");
    }
    if (c.rhsMode != 0 && c.rhsMode != 1) throw std::runtime_error("-rhsMode must be 0 or 1");
    if (!(c.dt > 0)) throw std::runtime_error("-dt must be positive");
    if (c.maxPicard < 1) throw std::runtime_error("-maxPicard must be >= 1");
    if (!(c.picardTol > 0)) throw std::runtime_error("-picardTol must be positive");
    if (c.betaInitial != 0 && c.betaInitial != 1) throw std::runtime_error("-betaInitial must be 0 or 1");
    if (c.supg != 0 && c.supg != 1) throw std::runtime_error("-supg must be 0 or 1");
    if (!(c.supgTauScale >= 0.0)) throw std::runtime_error("-supgTauScale must be nonnegative");
    if (c.prec == "inverse_diagonal" || c.prec == "operator_diag") c.prec = "operator_inverse_diagonal";
    if (c.prec == "monolithic_gmg" || c.prec == "coupled_monolithic_gmg") c.prec = "coupled_gmg";
    if (c.prec != "block_gmg" && c.prec != "operator_inverse_diagonal" && c.prec != "coupled_gmg") {
        throw std::runtime_error("unknown -prec; use block_gmg, operator_inverse_diagonal, or coupled_gmg");
    }
    if (c.mgPre < 0 || c.mgPost < 0) throw std::runtime_error("-mgPre/-mgPost must be nonnegative");
    if (!(c.mgOmega > 0.0)) throw std::runtime_error("-mgOmega must be positive");
    if (c.mgCoarseElem != 2) throw std::runtime_error("v1 coupled_gmg supports -mgCoarseElem 2 only");
    if (c.mgRebuildPolicy != "every_picard" && c.mgRebuildPolicy != "every_step") {
        throw std::runtime_error("unknown -mgRebuildPolicy; use every_picard or every_step");
    }
    if (c.timing != 0 && c.timing != 1) throw std::runtime_error("-timing must be 0 or 1");
    if (c.solver != "gmres") throw std::runtime_error("Q2/Q1 NSE Picard v1 supports -solver gmres only");
    return c;
}

} // namespace

namespace memoirs { namespace gpu {

inline int div_up_local(int n, int block) { return (n + block - 1) / block; }

template <class RealT>
__global__ void kernel_zero_local(int n, RealT* x) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = RealT(0);
}

template <class RealT>
__global__ void kernel_copy_local(int n, const RealT* a, RealT* b) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = a[i];
}

template <class RealT>
__global__ void kernel_subtract_local(int n, const RealT* a, const RealT* b, RealT* c) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] - b[i];
}

template <class RealT>
__global__ void kernel_axpby_local(int n, RealT alpha, const RealT* x, RealT beta, RealT* y) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = alpha * x[i] + beta * y[i];
}

struct Q2Q1GridCuda {
    int N = 0;
    int nvx = 0, nvy = 0, nvz = 0;
    int npx = 0, npy = 0, npz = 0;
    int nv = 0, np = 0, ndof = 0;
    double hx = 0, hy = 0, hz = 0;
};

__host__ inline Q2Q1GridCuda make_q2q1_grid(int N) {
    Q2Q1GridCuda g;
    g.N = N;
    g.nvx = 2 * N + 1; g.nvy = 2 * N + 1; g.nvz = 2 * N + 1;
    g.npx = N + 1; g.npy = N + 1; g.npz = N + 1;
    g.nv = g.nvx * g.nvy * g.nvz;
    g.np = g.npx * g.npy * g.npz;
    g.ndof = 3 * g.nv + g.np;
    g.hx = 1.0 / double(N); g.hy = 1.0 / double(N); g.hz = 1.0 / double(N);
    return g;
}

__host__ __device__ inline int v_id(const Q2Q1GridCuda g, int i, int j, int k) {
    return (k * g.nvy + j) * g.nvx + i;
}
__host__ __device__ inline int p_id(const Q2Q1GridCuda g, int i, int j, int k) {
    return (k * g.npy + j) * g.npx + i;
}
__host__ __device__ inline bool v_boundary(const Q2Q1GridCuda g, int i, int j, int k) {
    return i == 0 || j == 0 || k == 0 || i == g.nvx - 1 || j == g.nvy - 1 || k == g.nvz - 1;
}
__host__ __device__ inline int elem_from_linear(const Q2Q1GridCuda g, int e, int& ex, int& ey, int& ez) {
    ex = e % g.N;
    const int tmp = e / g.N;
    ey = tmp % g.N;
    ez = tmp / g.N;
    return 0;
}

__host__ __device__ inline void q2_1d(double r, double N[3], double dNdr[3], double d2Ndr2[3]) {
    N[0] = 0.5 * r * (r - 1.0);
    N[1] = 1.0 - r * r;
    N[2] = 0.5 * r * (r + 1.0);
    dNdr[0] = r - 0.5;
    dNdr[1] = -2.0 * r;
    dNdr[2] = r + 0.5;
    d2Ndr2[0] = 1.0;
    d2Ndr2[1] = -2.0;
    d2Ndr2[2] = 1.0;
}

__host__ __device__ inline void q1_1d(double r, double N[2], double dNdr[2]) {
    N[0] = 0.5 * (1.0 - r);
    N[1] = 0.5 * (1.0 + r);
    dNdr[0] = -0.5;
    dNdr[1] = 0.5;
}

__host__ __device__ inline int q2_lid(int ax, int ay, int az) { return ax + 3 * ay + 9 * az; }
__host__ __device__ inline int q1_lid(int ax, int ay, int az) { return ax + 2 * ay + 4 * az; }

__host__ __device__ inline void stokes_exact(double x, double y, double z,
                                             double& u, double& v, double& w, double& p) {
    const double pi = 3.141592653589793238462643383279502884;
    const double sx = sin(pi*x), sy = sin(pi*y), sz = sin(pi*z);
    const double cx = cos(pi*x), cy = cos(pi*y);
    u =  2.0 * pi * sx * sx * sy * cy * sz * sz;
    v = -2.0 * pi * sx * cx * sy * sy * sz * sz;
    w = 0.0;
    p = cos(pi*x) * sy * sz; // mean-zero in x; p(0,0,0)=0
}

__host__ __device__ inline void stokes_gradp(double x, double y, double z,
                                             double& px, double& py, double& pz) {
    const double pi = 3.141592653589793238462643383279502884;
    const double sx = sin(pi*x), sy = sin(pi*y), sz = sin(pi*z);
    const double cx = cos(pi*x), cy = cos(pi*y), cz = cos(pi*z);
    px = -pi * sx * sy * sz;
    py =  pi * cx * cy * sz;
    pz =  pi * cx * sy * cz;
}

__host__ __device__ inline void stokes_force(double x, double y, double z, double nu,
                                             double& fx, double& fy, double& fz) {
    const double pi = 3.141592653589793238462643383279502884;
    const double sx = sin(pi*x), sy = sin(pi*y), sz = sin(pi*z);
    const double cx = cos(pi*x), cy = cos(pi*y);
    const double c2x = cos(2.0*pi*x), c2y = cos(2.0*pi*y), c2z = cos(2.0*pi*z);
    const double A = sx*sx;
    const double B = sy*cy;
    const double C = sz*sz;
    const double D = sx*cx;
    const double E = sy*sy;
    const double lap_u = 4.0*pi*pi*pi*B * (c2x*C - 2.0*A*C + A*c2z);
    const double lap_v = 8.0*pi*pi*pi*D*E*C - 4.0*pi*pi*pi*D*c2y*C - 4.0*pi*pi*pi*D*E*c2z;
    double px, py, pz;
    stokes_gradp(x,y,z,px,py,pz);
    fx = -nu * lap_u + px;
    fy = -nu * lap_v + py;
    fz = pz;
}

__host__ __device__ inline void stokes_grad_velocity(double x, double y, double z,
                                                     double gu[3], double gv[3], double gw[3]) {
    const double pi = 3.141592653589793238462643383279502884;
    const double sx = sin(pi*x), sy = sin(pi*y), sz = sin(pi*z);
    const double cx = cos(pi*x), cy = cos(pi*y), cz = cos(pi*z);
    const double c2x = cos(2.0*pi*x), c2y = cos(2.0*pi*y);
    gu[0] =  4.0*pi*pi*sx*cx*sy*cy*sz*sz;
    gu[1] =  2.0*pi*pi*sx*sx*c2y*sz*sz;
    gu[2] =  4.0*pi*pi*sx*sx*sy*cy*sz*cz;
    gv[0] = -2.0*pi*pi*c2x*sy*sy*sz*sz;
    gv[1] = -4.0*pi*pi*sx*cx*sy*cy*sz*sz;
    gv[2] = -4.0*pi*pi*sx*cx*sy*sy*sz*cz;
    gw[0] = 0.0; gw[1] = 0.0; gw[2] = 0.0;
}

__host__ __device__ inline void nse_picard_mms_force(double x, double y, double z,
                                                     double nu, double advScale,
                                                     double& fx, double& fy, double& fz) {
    // Stationary divergence-free MMS: du/dt=0.
    // f = advScale*(u.grad)u - nu*Delta u + grad p.
    stokes_force(x,y,z,nu,fx,fy,fz);
    double u,v,w,p;
    stokes_exact(x,y,z,u,v,w,p);
    double gu[3], gv[3], gw[3];
    stokes_grad_velocity(x,y,z,gu,gv,gw);
    fx += advScale * (u*gu[0] + v*gu[1] + w*gu[2]);
    fy += advScale * (u*gv[0] + v*gv[1] + w*gv[2]);
    fz += advScale * (u*gw[0] + v*gw[1] + w*gw[2]);
}

inline double compute_tau(const Q2Q1GridCuda& g, double nu, double tauScale,
                          const std::string& mode, double tauC, double tauDt, double tauAdvMag) {
    const double hx = g.hx, hy = g.hy, hz = g.hz;
    const double hmin = std::min({hx,hy,hz});
    const double pi = 3.141592653589793238462643383279502884;
    const double hEq = std::pow(6.0 * hx * hy * hz / pi, 1.0/3.0);
    if (mode == "simple") return tauScale * hmin * hmin / nu;
    if (mode == "sphere") return tauScale * hEq * hEq / (tauC * nu);
    if (mode == "tezduyar") {
        double inv2 = 0.0;
        if (tauDt > 0) { const double t = 2.0 / tauDt; inv2 += t*t; }
        if (tauAdvMag > 0) { const double a = tauAdvMag / hEq; inv2 += a*a; }
        const double d = tauC * nu / (hEq*hEq);
        inv2 += d*d;
        return tauScale / std::sqrt(inv2);
    }
    // MATLAB-style diffusive metric tau, extended to 3D:
    // tau = tauScale / sqrt((tauC*nu)^2 * [(2/hx)^4 + (2/hy)^4 + (2/hz)^4])
    const double gx = 2.0/hx, gy = 2.0/hy, gz = 2.0/hz;
    const double metric4 = gx*gx*gx*gx + gy*gy*gy*gy + gz*gz*gz*gz;
    return tauScale / ((tauC * nu) * std::sqrt(metric4));
}

template <class RealT>
__global__ void kernel_fill_exact_q2q1(Q2Q1GridCuda g, RealT* x) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q < g.nv) {
        const int i = q % g.nvx;
        const int tmp = q / g.nvx;
        const int j = tmp % g.nvy;
        const int k = tmp / g.nvy;
        const double xx = double(i) / double(2*g.N);
        const double yy = double(j) / double(2*g.N);
        const double zz = double(k) / double(2*g.N);
        double u,v,w,p;
        stokes_exact(xx,yy,zz,u,v,w,p);
        x[q] = RealT(u);
        x[g.nv + q] = RealT(v);
        x[2*g.nv + q] = RealT(w);
    }
    if (q < g.np) {
        const int i = q % g.npx;
        const int tmp = q / g.npx;
        const int j = tmp % g.npy;
        const int k = tmp / g.npy;
        const double xx = double(i) / double(g.N);
        const double yy = double(j) / double(g.N);
        const double zz = double(k) / double(g.N);
        double u,v,w,pv;
        stokes_exact(xx,yy,zz,u,v,w,pv);
        x[3*g.nv + q] = RealT(pv);
    }
}

template <class RealT>
__global__ void kernel_apply_q2q1_elements(Q2Q1GridCuda g, RealT nu, RealT tau, RealT massCoeff, RealT advScale, int supgEnabled, RealT supgTauScale, const RealT* beta, const RealT* x, RealT* y) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    const int nElem = g.N * g.N * g.N;
    if (e >= nElem) return;
    int ex,ey,ez; elem_from_linear(g,e,ex,ey,ez);
    constexpr int pPin = 0;

    const double qp[4] = {-0.86113631159405257522, -0.33998104358485626480,
                           0.33998104358485626480,  0.86113631159405257522};
    const double qw[4] = { 0.34785484513745385737,  0.65214515486254614263,
                           0.65214515486254614263,  0.34785484513745385737};
    const double hx = g.hx, hy = g.hy, hz = g.hz;
    const double jac = hx*hy*hz/8.0;

    // Tensor-product basis tables for the 4-point Gauss rule.  This is the
    // first cheap sum-factorization step: avoid rebuilding the same 1D Q2/Q1
    // basis values inside the 4x4x4 quadrature loops.  The algebra below is
    // unchanged; the kernel still assembles by element and atomically scatters.
    double B2[4][3], D2[4][3], DD2[4][3];
    double B1[4][2], D1[4][2];
    for (int q=0; q<4; ++q) {
        q2_1d(qp[q], B2[q], D2[q], DD2[q]);
        q1_1d(qp[q], B1[q], D1[q]);
    }

    int vgid[27];
    bool vbnd[27];
    double ux[27], uyv[27], uzv[27];
    double bx[27], byv[27], bzv[27];
    for (int az=0; az<3; ++az) for (int ay=0; ay<3; ++ay) for (int ax=0; ax<3; ++ax) {
        const int a = q2_lid(ax,ay,az);
        const int gi = 2*ex + ax, gj = 2*ey + ay, gk = 2*ez + az;
        const int id = v_id(g,gi,gj,gk);
        vgid[a] = id;
        vbnd[a] = v_boundary(g,gi,gj,gk);
        ux[a] = double(x[id]);
        uyv[a] = double(x[g.nv + id]);
        uzv[a] = double(x[2*g.nv + id]);
        bx[a] = beta ? double(beta[id]) : 0.0;
        byv[a] = beta ? double(beta[g.nv + id]) : 0.0;
        bzv[a] = beta ? double(beta[2*g.nv + id]) : 0.0;
    }

    int pgid[8];
    double pp[8];
    for (int az=0; az<2; ++az) for (int ay=0; ay<2; ++ay) for (int ax=0; ax<2; ++ax) {
        const int a = q1_lid(ax,ay,az);
        const int id = p_id(g,ex+ax,ey+ay,ez+az);
        pgid[a] = id;
        pp[a] = double(x[3*g.nv + id]);
    }

    for (int iq=0; iq<4; ++iq) {
        const double* Q2x = B2[iq]; const double* dQ2x = D2[iq]; const double* ddQ2x = DD2[iq];
        const double* Q1x = B1[iq]; const double* dQ1x = D1[iq];
        for (int jq=0; jq<4; ++jq) {
            const double* Q2y = B2[jq]; const double* dQ2y = D2[jq]; const double* ddQ2y = DD2[jq];
            const double* Q1y = B1[jq]; const double* dQ1y = D1[jq];
            for (int kq=0; kq<4; ++kq) {
                const double* Q2z = B2[kq]; const double* dQ2z = D2[kq]; const double* ddQ2z = DD2[kq];
                const double* Q1z = B1[kq]; const double* dQ1z = D1[kq];
                const double wq = qw[iq]*qw[jq]*qw[kq]*jac;

                double Nv[27], dVx[27], dVy[27], dVz[27], lapV[27];
                for (int az=0; az<3; ++az) for (int ay=0; ay<3; ++ay) for (int ax=0; ax<3; ++ax) {
                    const int a = q2_lid(ax,ay,az);
                    Nv[a] = Q2x[ax]*Q2y[ay]*Q2z[az];
                    dVx[a] = (2.0/hx)*dQ2x[ax]*Q2y[ay]*Q2z[az];
                    dVy[a] = Q2x[ax]*(2.0/hy)*dQ2y[ay]*Q2z[az];
                    dVz[a] = Q2x[ax]*Q2y[ay]*(2.0/hz)*dQ2z[az];
                    lapV[a] = (4.0/(hx*hx))*ddQ2x[ax]*Q2y[ay]*Q2z[az]
                            + Q2x[ax]*(4.0/(hy*hy))*ddQ2y[ay]*Q2z[az]
                            + Q2x[ax]*Q2y[ay]*(4.0/(hz*hz))*ddQ2z[az];
                }
                double Np[8], dPx[8], dPy[8], dPz[8];
                for (int az=0; az<2; ++az) for (int ay=0; ay<2; ++ay) for (int ax=0; ax<2; ++ax) {
                    const int a = q1_lid(ax,ay,az);
                    Np[a] = Q1x[ax]*Q1y[ay]*Q1z[az];
                    dPx[a] = (2.0/hx)*dQ1x[ax]*Q1y[ay]*Q1z[az];
                    dPy[a] = Q1x[ax]*(2.0/hy)*dQ1y[ay]*Q1z[az];
                    dPz[a] = Q1x[ax]*Q1y[ay]*(2.0/hz)*dQ1z[az];
                }

                double gradUx[3]={0,0,0}, gradUy[3]={0,0,0}, gradUz[3]={0,0,0};
                double lapUx=0, lapUy=0, lapUz=0;
                double divU=0;
                double uh=0, vh=0, wh=0;
                double betah[3]={0,0,0};
                for (int b=0; b<27; ++b) if (!vbnd[b]) {
                    uh += Nv[b]*ux[b]; vh += Nv[b]*uyv[b]; wh += Nv[b]*uzv[b];
                    betah[0] += Nv[b]*bx[b]; betah[1] += Nv[b]*byv[b]; betah[2] += Nv[b]*bzv[b];
                    gradUx[0] += dVx[b]*ux[b]; gradUx[1] += dVy[b]*ux[b]; gradUx[2] += dVz[b]*ux[b];
                    gradUy[0] += dVx[b]*uyv[b]; gradUy[1] += dVy[b]*uyv[b]; gradUy[2] += dVz[b]*uyv[b];
                    gradUz[0] += dVx[b]*uzv[b]; gradUz[1] += dVy[b]*uzv[b]; gradUz[2] += dVz[b]*uzv[b];
                    lapUx += lapV[b]*ux[b]; lapUy += lapV[b]*uyv[b]; lapUz += lapV[b]*uzv[b];
                    divU += dVx[b]*ux[b] + dVy[b]*uyv[b] + dVz[b]*uzv[b];
                }
                const double convUx = double(advScale) * (betah[0]*gradUx[0] + betah[1]*gradUx[1] + betah[2]*gradUx[2]);
                const double convUy = double(advScale) * (betah[0]*gradUy[0] + betah[1]*gradUy[1] + betah[2]*gradUy[2]);
                const double convUz = double(advScale) * (betah[0]*gradUz[0] + betah[1]*gradUz[1] + betah[2]*gradUz[2]);
                double ph=0, gradP[3]={0,0,0};
                for (int b=0; b<8; ++b) if (pgid[b] != pPin) {
                    ph += Np[b]*pp[b];
                    gradP[0] += dPx[b]*pp[b]; gradP[1] += dPy[b]*pp[b]; gradP[2] += dPz[b]*pp[b];
                }

                const double rmx = double(massCoeff)*uh + convUx - double(nu)*lapUx + gradP[0];
                const double rmy = double(massCoeff)*vh + convUy - double(nu)*lapUy + gradP[1];
                const double rmz = double(massCoeff)*wh + convUz - double(nu)*lapUz + gradP[2];
                const double tauSupg = supgEnabled ? double(tau)*double(supgTauScale) : 0.0;

                // Momentum rows: Galerkin weak form plus optional SUPG.
                // SUPG component i: tau_supg*(a.grad v_test)*R_i, with a = advScale*beta.
                for (int a=0; a<27; ++a) if (!vbnd[a]) {
                    const int row = vgid[a];
                    const double Na = Nv[a];
                    const double streamNa = double(advScale)*(betah[0]*dVx[a] + betah[1]*dVy[a] + betah[2]*dVz[a]);
                    const double ku = double(massCoeff)*Na*uh
                                    + double(nu) * (dVx[a]*gradUx[0] + dVy[a]*gradUx[1] + dVz[a]*gradUx[2])
                                    + Na*convUx - ph*dVx[a]
                                    + tauSupg*streamNa*rmx;
                    const double kv = double(massCoeff)*Na*vh
                                    + double(nu) * (dVx[a]*gradUy[0] + dVy[a]*gradUy[1] + dVz[a]*gradUy[2])
                                    + Na*convUy - ph*dVy[a]
                                    + tauSupg*streamNa*rmy;
                    const double kw = double(massCoeff)*Na*wh
                                    + double(nu) * (dVx[a]*gradUz[0] + dVy[a]*gradUz[1] + dVz[a]*gradUz[2])
                                    + Na*convUz - ph*dVz[a]
                                    + tauSupg*streamNa*rmz;
                    atomicAdd(&y[row], RealT(wq*ku));
                    atomicAdd(&y[g.nv + row], RealT(wq*kv));
                    atomicAdd(&y[2*g.nv + row], RealT(wq*kw));
                }

                // Pressure rows: (q, div u) + tau*(grad q, u/dt + beta.grad u - nu*Delta u + grad p).
                for (int a=0; a<8; ++a) if (pgid[a] != pPin) {
                    const double r0 = Np[a]*divU + double(tau)*(dPx[a]*rmx + dPy[a]*rmy + dPz[a]*rmz);
                    atomicAdd(&y[3*g.nv + pgid[a]], RealT(wq*r0));
                }
            }
        }
    }
}

template <class RealT>
__global__ void kernel_apply_q2q1_strong_rows(Q2Q1GridCuda g, const RealT* x, RealT* y) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q < g.nv) {
        const int i = q % g.nvx;
        const int tmp = q / g.nvx;
        const int j = tmp % g.nvy;
        const int k = tmp / g.nvy;
        if (v_boundary(g,i,j,k)) {
            y[q] = x[q];
            y[g.nv + q] = x[g.nv + q];
            y[2*g.nv + q] = x[2*g.nv + q];
        }
    }
    if (q == 0) y[3*g.nv] = x[3*g.nv];
}

template <class RealT>
__global__ void kernel_assemble_rhs_q2q1(Q2Q1GridCuda g, RealT nu, RealT tau, RealT massCoeff, RealT advScale, int supgEnabled, RealT supgTauScale, const RealT* beta, const RealT* uold, RealT* rhs) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    const int nElem = g.N*g.N*g.N;
    if (e >= nElem) return;
    int ex,ey,ez; elem_from_linear(g,e,ex,ey,ez);
    const double qp[4] = {-0.86113631159405257522, -0.33998104358485626480,
                           0.33998104358485626480,  0.86113631159405257522};
    const double qw[4] = { 0.34785484513745385737,  0.65214515486254614263,
                           0.65214515486254614263,  0.34785484513745385737};
    const double hx=g.hx, hy=g.hy, hz=g.hz, jac=hx*hy*hz/8.0;

    for (int iq=0; iq<4; ++iq) {
        double Q2x[3], dQ2x[3], ddQ2x[3], Q1x[2], dQ1x[2]; q2_1d(qp[iq],Q2x,dQ2x,ddQ2x); q1_1d(qp[iq],Q1x,dQ1x);
        const double xx = (double(ex)+0.5*(1.0+qp[iq]))*hx;
        for (int jq=0; jq<4; ++jq) {
            double Q2y[3], dQ2y[3], ddQ2y[3], Q1y[2], dQ1y[2]; q2_1d(qp[jq],Q2y,dQ2y,ddQ2y); q1_1d(qp[jq],Q1y,dQ1y);
            const double yy = (double(ey)+0.5*(1.0+qp[jq]))*hy;
            for (int kq=0; kq<4; ++kq) {
                double Q2z[3], dQ2z[3], ddQ2z[3], Q1z[2], dQ1z[2]; q2_1d(qp[kq],Q2z,dQ2z,ddQ2z); q1_1d(qp[kq],Q1z,dQ1z);
                const double zz = (double(ez)+0.5*(1.0+qp[kq]))*hz;
                const double wq = qw[iq]*qw[jq]*qw[kq]*jac;
                double fx,fy,fz; nse_picard_mms_force(xx,yy,zz,double(nu),double(advScale),fx,fy,fz);
                double uoldh=0.0, voldh=0.0, woldh=0.0;
                double betah[3] = {0.0, 0.0, 0.0};
                for (int bz=0; bz<3; ++bz) for (int by=0; by<3; ++by) for (int bx_=0; bx_<3; ++bx_) {
                    const int gid = v_id(g,2*ex+bx_,2*ey+by,2*ez+bz);
                    const double Nb = Q2x[bx_]*Q2y[by]*Q2z[bz];
                    if (!v_boundary(g,2*ex+bx_,2*ey+by,2*ez+bz)) {
                        uoldh += Nb*double(uold[gid]);
                        voldh += Nb*double(uold[g.nv + gid]);
                        woldh += Nb*double(uold[2*g.nv + gid]);
                        if (beta) {
                            betah[0] += Nb*double(beta[gid]);
                            betah[1] += Nb*double(beta[g.nv + gid]);
                            betah[2] += Nb*double(beta[2*g.nv + gid]);
                        }
                    }
                }
                fx += double(massCoeff)*uoldh;
                fy += double(massCoeff)*voldh;
                fz += double(massCoeff)*woldh;
                const double tauSupg = supgEnabled ? double(tau)*double(supgTauScale) : 0.0;

                for (int az=0; az<3; ++az) for (int ay=0; ay<3; ++ay) for (int ax=0; ax<3; ++ax) {
                    const int a = q2_lid(ax,ay,az);
                    const int gi = 2*ex+ax, gj=2*ey+ay, gk=2*ez+az;
                    if (!v_boundary(g,gi,gj,gk)) {
                        const int row = v_id(g,gi,gj,gk);
                        const double Na = Q2x[ax]*Q2y[ay]*Q2z[az];
                        const double dNaX = (2.0/hx)*dQ2x[ax]*Q2y[ay]*Q2z[az];
                        const double dNaY = Q2x[ax]*(2.0/hy)*dQ2y[ay]*Q2z[az];
                        const double dNaZ = Q2x[ax]*Q2y[ay]*(2.0/hz)*dQ2z[az];
                        const double streamNa = double(advScale)*(betah[0]*dNaX + betah[1]*dNaY + betah[2]*dNaZ);
                        const double testWeight = Na + tauSupg*streamNa;
                        atomicAdd(&rhs[row], RealT(wq*testWeight*fx));
                        atomicAdd(&rhs[g.nv + row], RealT(wq*testWeight*fy));
                        atomicAdd(&rhs[2*g.nv + row], RealT(wq*testWeight*fz));
                    }
                }
                for (int az=0; az<2; ++az) for (int ay=0; ay<2; ++ay) for (int ax=0; ax<2; ++ax) {
                    const int a = q1_lid(ax,ay,az);
                    const int row = p_id(g,ex+ax,ey+ay,ez+az);
                    if (row != 0) {
                        const double dNaX = (2.0/hx)*dQ1x[ax]*Q1y[ay]*Q1z[az];
                        const double dNaY = Q1x[ax]*(2.0/hy)*dQ1y[ay]*Q1z[az];
                        const double dNaZ = Q1x[ax]*Q1y[ay]*(2.0/hz)*dQ1z[az];
                        const double val = double(tau)*(dNaX*fx + dNaY*fy + dNaZ*fz);
                        atomicAdd(&rhs[3*g.nv + row], RealT(wq*val));
                    }
                }
            }
        }
    }
}

template <class RealT>
__global__ void kernel_rhs_strong_q2q1(Q2Q1GridCuda g, RealT* rhs) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q < g.nv) {
        const int i = q % g.nvx;
        const int tmp = q / g.nvx;
        const int j = tmp % g.nvy;
        const int k = tmp / g.nvy;
        if (v_boundary(g,i,j,k)) {
            rhs[q] = RealT(0); rhs[g.nv+q] = RealT(0); rhs[2*g.nv+q] = RealT(0);
        }
    }
    if (q == 0) rhs[3*g.nv] = RealT(0);
}


template <class RealT>
__global__ void kernel_assemble_operator_diag_q2q1(Q2Q1GridCuda g, RealT nu, RealT tau, RealT massCoeff, RealT advScale, int supgEnabled, RealT supgTauScale, const RealT* beta, RealT* diag) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    const int nElem = g.N * g.N * g.N;
    if (e >= nElem) return;
    int ex,ey,ez; elem_from_linear(g,e,ex,ey,ez);
    constexpr int pPin = 0;

    const double qp[4] = {-0.86113631159405257522, -0.33998104358485626480,
                           0.33998104358485626480,  0.86113631159405257522};
    const double qw[4] = { 0.34785484513745385737,  0.65214515486254614263,
                           0.65214515486254614263,  0.34785484513745385737};
    const double hx = g.hx, hy = g.hy, hz = g.hz;
    const double jac = hx*hy*hz/8.0;

    int vgid[27];
    bool vbnd[27];
    double bx[27], byv[27], bzv[27];
    for (int az=0; az<3; ++az) for (int ay=0; ay<3; ++ay) for (int ax=0; ax<3; ++ax) {
        const int a = q2_lid(ax,ay,az);
        const int gi = 2*ex + ax, gj = 2*ey + ay, gk = 2*ez + az;
        const int id = v_id(g,gi,gj,gk);
        vgid[a] = id;
        vbnd[a] = v_boundary(g,gi,gj,gk);
        bx[a] = beta ? double(beta[id]) : 0.0;
        byv[a] = beta ? double(beta[g.nv + id]) : 0.0;
        bzv[a] = beta ? double(beta[2*g.nv + id]) : 0.0;
    }

    int pgid[8];
    for (int az=0; az<2; ++az) for (int ay=0; ay<2; ++ay) for (int ax=0; ax<2; ++ax) {
        const int a = q1_lid(ax,ay,az);
        pgid[a] = p_id(g,ex+ax,ey+ay,ez+az);
    }

    for (int iq=0; iq<4; ++iq) {
        double Q2x[3], dQ2x[3], ddQ2x[3], Q1x[2], dQ1x[2];
        q2_1d(qp[iq], Q2x, dQ2x, ddQ2x);
        q1_1d(qp[iq], Q1x, dQ1x);
        for (int jq=0; jq<4; ++jq) {
            double Q2y[3], dQ2y[3], ddQ2y[3], Q1y[2], dQ1y[2];
            q2_1d(qp[jq], Q2y, dQ2y, ddQ2y);
            q1_1d(qp[jq], Q1y, dQ1y);
            for (int kq=0; kq<4; ++kq) {
                double Q2z[3], dQ2z[3], ddQ2z[3], Q1z[2], dQ1z[2];
                q2_1d(qp[kq], Q2z, dQ2z, ddQ2z);
                q1_1d(qp[kq], Q1z, dQ1z);
                const double wq = qw[iq]*qw[jq]*qw[kq]*jac;

                double Nv[27], dVx[27], dVy[27], dVz[27], lapV[27];
                for (int az=0; az<3; ++az) for (int ay=0; ay<3; ++ay) for (int ax=0; ax<3; ++ax) {
                    const int a = q2_lid(ax,ay,az);
                    Nv[a] = Q2x[ax]*Q2y[ay]*Q2z[az];
                    dVx[a] = (2.0/hx)*dQ2x[ax]*Q2y[ay]*Q2z[az];
                    dVy[a] = Q2x[ax]*(2.0/hy)*dQ2y[ay]*Q2z[az];
                    dVz[a] = Q2x[ax]*Q2y[ay]*(2.0/hz)*dQ2z[az];
                    lapV[a] = (4.0/(hx*hx))*ddQ2x[ax]*Q2y[ay]*Q2z[az]
                            + Q2x[ax]*(4.0/(hy*hy))*ddQ2y[ay]*Q2z[az]
                            + Q2x[ax]*Q2y[ay]*(4.0/(hz*hz))*ddQ2z[az];
                }
                double dPx[8], dPy[8], dPz[8];
                for (int az=0; az<2; ++az) for (int ay=0; ay<2; ++ay) for (int ax=0; ax<2; ++ax) {
                    const int a = q1_lid(ax,ay,az);
                    dPx[a] = (2.0/hx)*dQ1x[ax]*Q1y[ay]*Q1z[az];
                    dPy[a] = Q1x[ax]*(2.0/hy)*dQ1y[ay]*Q1z[az];
                    dPz[a] = Q1x[ax]*Q1y[ay]*(2.0/hz)*dQ1z[az];
                }

                double betah[3]={0,0,0};
                for (int b=0; b<27; ++b) if (!vbnd[b]) {
                    betah[0] += Nv[b]*bx[b];
                    betah[1] += Nv[b]*byv[b];
                    betah[2] += Nv[b]*bzv[b];
                }

                // Literal velocity-row diagonal of the current Picard operator:
                // (v,u/dt) + nu*(grad v,grad u) + (v,beta.grad u), plus optional SUPG.
                // Pressure-gradient coupling is velocity-row/pressure-column and is off-diagonal.
                const double tauSupg = supgEnabled ? double(tau)*double(supgTauScale) : 0.0;
                for (int a=0; a<27; ++a) if (!vbnd[a]) {
                    const double betaGradNa = betah[0]*dVx[a] + betah[1]*dVy[a] + betah[2]*dVz[a];
                    const double streamNa = double(advScale)*betaGradNa;
                    const double strongDiagNa = double(massCoeff)*Nv[a]
                                              + double(advScale)*betaGradNa
                                              - double(nu)*lapV[a];
                    const double d = double(massCoeff)*Nv[a]*Nv[a]
                                   + double(nu)*(dVx[a]*dVx[a] + dVy[a]*dVy[a] + dVz[a]*dVz[a])
                                   + Nv[a]*double(advScale)*betaGradNa
                                   + tauSupg*streamNa*strongDiagNa;
                    const RealT val = RealT(wq*d);
                    const int row = vgid[a];
                    atomicAdd(&diag[row], val);
                    atomicAdd(&diag[g.nv + row], val);
                    atomicAdd(&diag[2*g.nv + row], val);
                }

                // Literal pressure-row/pressure-column diagonal from PSPG tau*(grad q, grad p).
                // The velocity-residual PSPG terms are pressure-row/velocity-column and are off-diagonal.
                for (int a=0; a<8; ++a) if (pgid[a] != pPin) {
                    const double d = double(tau)*(dPx[a]*dPx[a] + dPy[a]*dPy[a] + dPz[a]*dPz[a]);
                    atomicAdd(&diag[3*g.nv + pgid[a]], RealT(wq*d));
                }
            }
        }
    }
}

template <class RealT>
__global__ void kernel_operator_diag_strong_rows_q2q1(Q2Q1GridCuda g, RealT* diag) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q < g.nv) {
        const int i = q % g.nvx;
        const int tmp = q / g.nvx;
        const int j = tmp % g.nvy;
        const int k = tmp / g.nvy;
        if (v_boundary(g,i,j,k)) {
            diag[q] = RealT(1);
            diag[g.nv + q] = RealT(1);
            diag[2*g.nv + q] = RealT(1);
        }
    }
    if (q == 0) diag[3*g.nv] = RealT(1);
}

template <class RealT>
__global__ void kernel_apply_operator_inverse_diag_q2q1(int n, const RealT* diag, const RealT* r, RealT* z) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double d = double(diag[i]);
    const double ad = fabs(d);
    if (ad > 1.0e-300 && ad < 1.0e300) z[i] = RealT(double(r[i]) / d);
    else z[i] = r[i];
}

template <class RealT>
__global__ void kernel_diag_prec_q2q1(Q2Q1GridCuda g, RealT nu, RealT tau, const RealT* r, RealT* z) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q < g.nv) {
        const int i = q % g.nvx;
        const int tmp = q / g.nvx;
        const int j = tmp % g.nvy;
        const int k = tmp / g.nvy;
        if (v_boundary(g,i,j,k)) {
            z[q] = r[q]; z[g.nv+q] = r[g.nv+q]; z[2*g.nv+q] = r[2*g.nv+q];
            return;
        }
        // Cheap robust diagonal proxy for Q2 stiffness; enough for GMRES development.
        const RealT d = max(RealT(1e-30), RealT(nu) * RealT(g.hx*g.hy*g.hz) * RealT(64.0/(g.hx*g.hx)));
        z[q] = r[q]/d; z[g.nv+q] = r[g.nv+q]/d; z[2*g.nv+q] = r[2*g.nv+q]/d;
    }
    if (q < g.np) {
        if (q == 0) z[3*g.nv] = r[3*g.nv];
        else {
            const RealT d = max(RealT(1e-30), RealT(tau) * RealT(g.hx*g.hy*g.hz) * RealT(12.0/(g.hx*g.hx)));
            z[3*g.nv + q] = r[3*g.nv + q]/d;
        }
    }
}

template <class RealT>
class Q2Q1NsePicardOperator {
public:
    Q2Q1NsePicardOperator(Q2Q1GridCuda g, RealT nu, RealT tau, RealT massCoeff, RealT advScale, int supgEnabled, RealT supgTauScale, const RealT* beta)
        : g_(g), nu_(nu), tau_(tau), massCoeff_(massCoeff), advScale_(advScale), supgEnabled_(supgEnabled), supgTauScale_(supgTauScale), beta_(beta) {}
    int size() const { return g_.ndof; }
    int nv() const { return g_.nv; }
    int np() const { return g_.np; }
    Q2Q1GridCuda grid() const { return g_; }
    RealT tau() const { return tau_; }
    void apply(const RealT* x, RealT* y) const {
        const int block=128;
        kernel_zero_local<RealT><<<div_up_local(g_.ndof,256),256>>>(g_.ndof,y);
        MEMOIRS_CUDA_KERNEL_CHECK();
        kernel_apply_q2q1_elements<RealT><<<div_up_local(g_.N*g_.N*g_.N,block),block>>>(g_,nu_,tau_,massCoeff_,advScale_,supgEnabled_,supgTauScale_,beta_,x,y);
        MEMOIRS_CUDA_KERNEL_CHECK();
        kernel_apply_q2q1_strong_rows<RealT><<<div_up_local(std::max(g_.nv,g_.np),256),256>>>(g_,x,y);
        MEMOIRS_CUDA_KERNEL_CHECK();
    }
    void assemble_rhs_from_old(const RealT* oldState, RealT* rhs) const {
        const int block=128;
        kernel_zero_local<RealT><<<div_up_local(g_.ndof,256),256>>>(g_.ndof,rhs);
        MEMOIRS_CUDA_KERNEL_CHECK();
        kernel_assemble_rhs_q2q1<RealT><<<div_up_local(g_.N*g_.N*g_.N,block),block>>>(g_,nu_,tau_,massCoeff_,advScale_,supgEnabled_,supgTauScale_,beta_,oldState,rhs);
        MEMOIRS_CUDA_KERNEL_CHECK();
        kernel_rhs_strong_q2q1<RealT><<<div_up_local(std::max(g_.nv,g_.np),256),256>>>(g_,rhs);
        MEMOIRS_CUDA_KERNEL_CHECK();
    }
private:
    Q2Q1GridCuda g_;
    RealT nu_, tau_, massCoeff_, advScale_;
    int supgEnabled_;
    RealT supgTauScale_;
    const RealT* beta_;
};


template <class RealT>
__global__ void kernel_extract_q2q1_component(Q2Q1GridCuda g,
                                               int comp,
                                               const RealT* r,
                                               RealT* scalar) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (comp < 3) {
        if (q < g.nv) scalar[q] = r[comp * g.nv + q];
    } else {
        if (q < g.np) scalar[q] = r[3 * g.nv + q];
    }
}

template <class RealT>
__global__ void kernel_scatter_q2q1_component_scaled(Q2Q1GridCuda g,
                                                      int comp,
                                                      RealT scale,
                                                      const RealT* scalar,
                                                      RealT* z) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (comp < 3) {
        if (q < g.nv) z[comp * g.nv + q] = scale * scalar[q];
    } else {
        if (q < g.np) z[3 * g.nv + q] = scale * scalar[q];
    }
}

template <class RealT>
class Q2Q1BlockGmgPrec {
public:
    Q2Q1BlockGmgPrec(Q2Q1GridCuda g, RealT nu, RealT tau)
        : g_(g), nu_(nu), tau_(tau),
          velRhs_(static_cast<std::size_t>(g.nv)),
          velSol_(static_cast<std::size_t>(g.nv)),
          pRhs_(static_cast<std::size_t>(g.np)),
          pSol_(static_cast<std::size_t>(g.np)) {

        memoirs::gpu::StructuredGmgCudaOptions<RealT> opt;
        opt.preSmooth = 3;
        opt.postSmooth = 3;
        opt.omega = RealT(0.70);
        opt.coarseMaxDofs = 256;
        opt.maxLevels = 32;
        opt.verbose = 0;

        memoirs::structured::StructuredGrid3D vg(g_.nvx, g_.nvy, g_.nvz, 1.0, 1.0, 1.0);
        memoirs::structured::StructuredGrid3D pg(g_.npx, g_.npy, g_.npz, 1.0, 1.0, 1.0);
        velGmg_.setup(vg, opt);
        pGmg_.setup(pg, opt);
    }

    void rebuild(RealT, RealT, RealT, RealT, int, RealT, const RealT*) {
        // Block-GMG is independent of the current Picard beta/SUPG in this development preconditioner.
    }

    void apply(const RealT* r, RealT* z) {
        const int block = 256;
        kernel_zero_local<RealT><<<div_up_local(g_.ndof, block), block>>>(g_.ndof, z);
        MEMOIRS_CUDA_KERNEL_CHECK();

        // Velocity block approximation:
        //   (nu * Q2 stiffness)^{-1} ≈ (1/nu) * (Q1 Poisson on refined Q2 grid)^{-1}.
        // This is not an exact Q2 block, but it is a real multilevel diffusion inverse.
        for (int c = 0; c < 3; ++c) {
            kernel_extract_q2q1_component<RealT><<<div_up_local(g_.nv, block), block>>>
                (g_, c, r, velRhs_.data());
            MEMOIRS_CUDA_KERNEL_CHECK();
            velGmg_.apply(velRhs_.data(), velSol_.data());
            kernel_scatter_q2q1_component_scaled<RealT><<<div_up_local(g_.nv, block), block>>>
                (g_, c, RealT(1) / nu_, velSol_.data(), z);
            MEMOIRS_CUDA_KERNEL_CHECK();
        }

        // Pressure block approximation:
        //   (tau * Q1 pressure Laplacian)^{-1} ≈ (1/tau) * scalar Q1 GMG.
        // Note: this uses the existing Dirichlet-style scalar GMG as a preconditioner
        // for the pinned pressure block. It is intentionally approximate.
        kernel_extract_q2q1_component<RealT><<<div_up_local(g_.np, block), block>>>
            (g_, 3, r, pRhs_.data());
        MEMOIRS_CUDA_KERNEL_CHECK();
        pGmg_.apply(pRhs_.data(), pSol_.data());
        kernel_scatter_q2q1_component_scaled<RealT><<<div_up_local(g_.np, block), block>>>
            (g_, 3, RealT(1) / tau_, pSol_.data(), z);
        MEMOIRS_CUDA_KERNEL_CHECK();
    }

private:
    Q2Q1GridCuda g_;
    RealT nu_, tau_;
    memoirs::gpu::StructuredGmgCudaPreconditioner<RealT> velGmg_;
    memoirs::gpu::StructuredGmgCudaPreconditioner<RealT> pGmg_;
    DeviceBuffer<RealT> velRhs_, velSol_, pRhs_, pSol_;
};

template <class RealT>
class Q2Q1OperatorInverseDiagPrec {
public:
    explicit Q2Q1OperatorInverseDiagPrec(Q2Q1GridCuda g)
        : g_(g), diag_(static_cast<std::size_t>(g.ndof)) {}

    void rebuild(RealT nu, RealT tau, RealT massCoeff, RealT advScale, int supgEnabled, RealT supgTauScale, const RealT* beta) {
        const int block = 256;
        diag_.zero();
        kernel_assemble_operator_diag_q2q1<RealT><<<div_up_local(g_.N*g_.N*g_.N, block), block>>>(
            g_, nu, tau, massCoeff, advScale, supgEnabled, supgTauScale, beta, diag_.data());
        MEMOIRS_CUDA_KERNEL_CHECK();
        kernel_operator_diag_strong_rows_q2q1<RealT><<<div_up_local(std::max(g_.nv,g_.np), block), block>>>(g_, diag_.data());
        MEMOIRS_CUDA_KERNEL_CHECK();
    }

    void apply(const RealT* r, RealT* z) {
        const int block = 256;
        kernel_apply_operator_inverse_diag_q2q1<RealT><<<div_up_local(g_.ndof, block), block>>>(
            g_.ndof, diag_.data(), r, z);
        MEMOIRS_CUDA_KERNEL_CHECK();
    }

private:
    Q2Q1GridCuda g_;
    DeviceBuffer<RealT> diag_;
};


template <class RealT>
__global__ void kernel_jacobi_update_q2q1(int n, RealT omega, const RealT* diag, const RealT* residual, RealT* x) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const RealT d = diag[i];
    const RealT ad = d >= RealT(0) ? d : -d;
    if (ad > RealT(1e-30)) x[i] += omega * residual[i] / d;
    else x[i] += omega * residual[i];
}

__host__ __device__ inline void q2_prolong_1d_weights(int fineIdx, int coarseElemN, int& i0, double w[3]) {
    // Fine level has Nf=2*Nc elements and Q2 indices 0..4*Nc.
    // Evaluate the coarse Q2 function at x = fineIdx/(4*Nc).
    const int maxFineIdx = 4 * coarseElemN;
    if (fineIdx >= maxFineIdx) {
        i0 = 2 * (coarseElemN - 1);
        double dN[3], dd[3];
        q2_1d(1.0, w, dN, dd);
        return;
    }
    const int ec = fineIdx / 4;
    const double sLocal = 0.5 * double(fineIdx) - 2.0 * double(ec); // in [0,2)
    const double r = sLocal - 1.0;
    i0 = 2 * ec;
    double dN[3], dd[3];
    q2_1d(r, w, dN, dd);
}

__host__ __device__ inline void q1_prolong_1d_weights(int fineIdx, int coarseElemN, int& i0, double w[2]) {
    // Fine level has Nf=2*Nc elements and Q1 indices 0..2*Nc.
    const int maxFineIdx = 2 * coarseElemN;
    if (fineIdx >= maxFineIdx) {
        i0 = coarseElemN - 1;
        double dN[2];
        q1_1d(1.0, w, dN);
        return;
    }
    const int ec = fineIdx / 2;
    const double sLocal = 0.5 * double(fineIdx) - double(ec); // in [0,1)
    const double r = 2.0 * sLocal - 1.0;
    i0 = ec;
    double dN[2];
    q1_1d(r, w, dN);
}

template <class RealT>
__global__ void kernel_prolong_add_q2q1(Q2Q1GridCuda fine, Q2Q1GridCuda coarse, const RealT* xc, RealT* xf) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q < fine.nv) {
        const int fi = q % fine.nvx;
        const int tmp = q / fine.nvx;
        const int fj = tmp % fine.nvy;
        const int fk = tmp / fine.nvy;
        int i0,j0,k0; double wx[3], wy[3], wz[3];
        q2_prolong_1d_weights(fi, coarse.N, i0, wx);
        q2_prolong_1d_weights(fj, coarse.N, j0, wy);
        q2_prolong_1d_weights(fk, coarse.N, k0, wz);
        double val[3] = {0.0, 0.0, 0.0};
        for (int az=0; az<3; ++az) for (int ay=0; ay<3; ++ay) for (int ax=0; ax<3; ++ax) {
            const double w = wx[ax] * wy[ay] * wz[az];
            const int cid = v_id(coarse, i0+ax, j0+ay, k0+az);
            val[0] += w * double(xc[cid]);
            val[1] += w * double(xc[coarse.nv + cid]);
            val[2] += w * double(xc[2*coarse.nv + cid]);
        }
        xf[q] += RealT(val[0]);
        xf[fine.nv + q] += RealT(val[1]);
        xf[2*fine.nv + q] += RealT(val[2]);
    }
    if (q < fine.np) {
        const int fi = q % fine.npx;
        const int tmp = q / fine.npx;
        const int fj = tmp % fine.npy;
        const int fk = tmp / fine.npy;
        int i0,j0,k0; double wx[2], wy[2], wz[2];
        q1_prolong_1d_weights(fi, coarse.N, i0, wx);
        q1_prolong_1d_weights(fj, coarse.N, j0, wy);
        q1_prolong_1d_weights(fk, coarse.N, k0, wz);
        double val = 0.0;
        for (int az=0; az<2; ++az) for (int ay=0; ay<2; ++ay) for (int ax=0; ax<2; ++ax) {
            const double w = wx[ax] * wy[ay] * wz[az];
            const int cid = p_id(coarse, i0+ax, j0+ay, k0+az);
            val += w * double(xc[3*coarse.nv + cid]);
        }
        xf[3*fine.nv + q] += RealT(val);
    }
}

template <class RealT>
__global__ void kernel_restrict_add_q2q1(Q2Q1GridCuda fine, Q2Q1GridCuda coarse, const RealT* rf, RealT* rc) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q < fine.nv) {
        const int fi = q % fine.nvx;
        const int tmp = q / fine.nvx;
        const int fj = tmp % fine.nvy;
        const int fk = tmp / fine.nvy;
        int i0,j0,k0; double wx[3], wy[3], wz[3];
        q2_prolong_1d_weights(fi, coarse.N, i0, wx);
        q2_prolong_1d_weights(fj, coarse.N, j0, wy);
        q2_prolong_1d_weights(fk, coarse.N, k0, wz);
        const double rv0 = double(rf[q]);
        const double rv1 = double(rf[fine.nv + q]);
        const double rv2 = double(rf[2*fine.nv + q]);
        for (int az=0; az<3; ++az) for (int ay=0; ay<3; ++ay) for (int ax=0; ax<3; ++ax) {
            const RealT w = RealT(wx[ax] * wy[ay] * wz[az]);
            const int cid = v_id(coarse, i0+ax, j0+ay, k0+az);
            atomicAdd(&rc[cid], w * RealT(rv0));
            atomicAdd(&rc[coarse.nv + cid], w * RealT(rv1));
            atomicAdd(&rc[2*coarse.nv + cid], w * RealT(rv2));
        }
    }
    if (q < fine.np) {
        const int fi = q % fine.npx;
        const int tmp = q / fine.npx;
        const int fj = tmp % fine.npy;
        const int fk = tmp / fine.npy;
        int i0,j0,k0; double wx[2], wy[2], wz[2];
        q1_prolong_1d_weights(fi, coarse.N, i0, wx);
        q1_prolong_1d_weights(fj, coarse.N, j0, wy);
        q1_prolong_1d_weights(fk, coarse.N, k0, wz);
        const double rp = double(rf[3*fine.nv + q]);
        for (int az=0; az<2; ++az) for (int ay=0; ay<2; ++ay) for (int ax=0; ax<2; ++ax) {
            const RealT w = RealT(wx[ax] * wy[ay] * wz[az]);
            const int cid = p_id(coarse, i0+ax, j0+ay, k0+az);
            atomicAdd(&rc[3*coarse.nv + cid], w * RealT(rp));
        }
    }
}

template <class RealT>
__global__ void kernel_inject_beta_q2q1(Q2Q1GridCuda fine, Q2Q1GridCuda coarse, const RealT* betaFine, RealT* betaCoarse) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= coarse.nv) return;
    const int ci = q % coarse.nvx;
    const int tmp = q / coarse.nvx;
    const int cj = tmp % coarse.nvy;
    const int ck = tmp / coarse.nvy;
    const int fid = v_id(fine, 2*ci, 2*cj, 2*ck);
    betaCoarse[q] = betaFine[fid];
    betaCoarse[coarse.nv + q] = betaFine[fine.nv + fid];
    betaCoarse[2*coarse.nv + q] = betaFine[2*fine.nv + fid];
}

template <class RealT>
class Q2Q1CoupledGmgPrec {
public:
    Q2Q1CoupledGmgPrec(Q2Q1GridCuda g, int preSmooth, int postSmooth, RealT omega, int coarseElem, int verbose)
        : preSmooth_(preSmooth), postSmooth_(postSmooth), omega_(omega), coarseElem_(coarseElem), verbose_(verbose) {
        if (coarseElem_ != 2) throw std::runtime_error("Q2Q1CoupledGmgPrec v1 requires coarseElem=2");
        if (g.N < coarseElem_) throw std::runtime_error("Q2Q1CoupledGmgPrec requires N >= coarseElem");
        int n = g.N;
        while (n > coarseElem_) {
            if ((n % 2) != 0) throw std::runtime_error("Q2Q1CoupledGmgPrec requires power-of-two hierarchy down to N=2");
            n /= 2;
        }
        if (n != coarseElem_) throw std::runtime_error("Q2Q1CoupledGmgPrec hierarchy did not reach N=2");
        for (int nl = g.N; ; nl /= 2) {
            levels_.emplace_back(make_q2q1_grid(nl));
            if (nl == coarseElem_) break;
        }
    }

    void rebuild(RealT nu, RealT tau, RealT massCoeff, RealT advScale, int supgEnabled, RealT supgTauScale, const RealT* beta) {
        nu_ = nu; tau_ = tau; massCoeff_ = massCoeff; advScale_ = advScale; supgEnabled_ = supgEnabled; supgTauScale_ = supgTauScale;
        const int block = 256;
        kernel_copy_local<RealT><<<div_up_local(3*levels_[0].g.nv, block), block>>>(3*levels_[0].g.nv, beta, levels_[0].beta.data());
        MEMOIRS_CUDA_KERNEL_CHECK();
        for (std::size_t l=1; l<levels_.size(); ++l) {
            kernel_inject_beta_q2q1<RealT><<<div_up_local(levels_[l].g.nv, block), block>>>(
                levels_[l-1].g, levels_[l].g, levels_[l-1].beta.data(), levels_[l].beta.data());
            MEMOIRS_CUDA_KERNEL_CHECK();
        }
        const RealT h0 = RealT(levels_[0].g.hx);
        for (auto& lev : levels_) {
            const RealT hrel = RealT(lev.g.hx) / h0;
            lev.tau = tau_ * hrel * hrel; // rediscretized h^2 tau scaling on coarse levels
            lev.diag.zero();
            kernel_assemble_operator_diag_q2q1<RealT><<<div_up_local(lev.g.N*lev.g.N*lev.g.N, block), block>>>(
                lev.g, nu_, lev.tau, massCoeff_, advScale_, supgEnabled_, supgTauScale_, lev.beta.data(), lev.diag.data());
            MEMOIRS_CUDA_KERNEL_CHECK();
            kernel_operator_diag_strong_rows_q2q1<RealT><<<div_up_local(std::max(lev.g.nv, lev.g.np), block), block>>>(lev.g, lev.diag.data());
            MEMOIRS_CUDA_KERNEL_CHECK();
        }
        build_coarse_lu();
    }

    void apply(const RealT* r, RealT* z) {
        const int block = 256;
        kernel_zero_local<RealT><<<div_up_local(levels_[0].g.ndof, block), block>>>(levels_[0].g.ndof, z);
        MEMOIRS_CUDA_KERNEL_CHECK();
        vcycle(0, r, z);
    }

private:
    struct Level {
        explicit Level(Q2Q1GridCuda gg)
            : g(gg), diag(static_cast<std::size_t>(gg.ndof)), beta(static_cast<std::size_t>(3*gg.nv)),
              Ax(static_cast<std::size_t>(gg.ndof)), residual(static_cast<std::size_t>(gg.ndof)),
              b(static_cast<std::size_t>(gg.ndof)), x(static_cast<std::size_t>(gg.ndof)) {}
        Q2Q1GridCuda g;
        RealT tau = RealT(0);
        DeviceBuffer<RealT> diag, beta, Ax, residual, b, x;
    };

    void apply_level(int l, const RealT* x, RealT* y) {
        const int block = 128;
        auto& lev = levels_[static_cast<std::size_t>(l)];
        kernel_zero_local<RealT><<<div_up_local(lev.g.ndof,256),256>>>(lev.g.ndof,y);
        MEMOIRS_CUDA_KERNEL_CHECK();
        kernel_apply_q2q1_elements<RealT><<<div_up_local(lev.g.N*lev.g.N*lev.g.N,block),block>>>(
            lev.g,nu_,lev.tau,massCoeff_,advScale_,supgEnabled_,supgTauScale_,lev.beta.data(),x,y);
        MEMOIRS_CUDA_KERNEL_CHECK();
        kernel_apply_q2q1_strong_rows<RealT><<<div_up_local(std::max(lev.g.nv,lev.g.np),256),256>>>(lev.g,x,y);
        MEMOIRS_CUDA_KERNEL_CHECK();
    }

    void smooth(int l, const RealT* b, RealT* x, int sweeps) {
        auto& lev = levels_[static_cast<std::size_t>(l)];
        const int block = 256;
        for (int s=0; s<sweeps; ++s) {
            apply_level(l, x, lev.Ax.data());
            kernel_subtract_local<RealT><<<div_up_local(lev.g.ndof,block),block>>>(lev.g.ndof,b,lev.Ax.data(),lev.residual.data());
            MEMOIRS_CUDA_KERNEL_CHECK();
            kernel_jacobi_update_q2q1<RealT><<<div_up_local(lev.g.ndof,block),block>>>(lev.g.ndof,omega_,lev.diag.data(),lev.residual.data(),x);
            MEMOIRS_CUDA_KERNEL_CHECK();
        }
    }

    void vcycle(int l, const RealT* b, RealT* x) {
        auto& lev = levels_[static_cast<std::size_t>(l)];
        const int block = 256;
        if (l == int(levels_.size()) - 1) {
            solve_coarse(b, x);
            return;
        }
        smooth(l, b, x, preSmooth_);
        apply_level(l, x, lev.Ax.data());
        kernel_subtract_local<RealT><<<div_up_local(lev.g.ndof,block),block>>>(lev.g.ndof,b,lev.Ax.data(),lev.residual.data());
        MEMOIRS_CUDA_KERNEL_CHECK();
        auto& coarse = levels_[static_cast<std::size_t>(l+1)];
        coarse.b.zero();
        kernel_restrict_add_q2q1<RealT><<<div_up_local(std::max(lev.g.nv, lev.g.np),block),block>>>(lev.g,coarse.g,lev.residual.data(),coarse.b.data());
        MEMOIRS_CUDA_KERNEL_CHECK();
        coarse.x.zero();
        vcycle(l+1, coarse.b.data(), coarse.x.data());
        kernel_prolong_add_q2q1<RealT><<<div_up_local(std::max(lev.g.nv, lev.g.np),block),block>>>(lev.g,coarse.g,coarse.x.data(),x);
        MEMOIRS_CUDA_KERNEL_CHECK();
        smooth(l, b, x, postSmooth_);
    }

    void build_coarse_lu() {
        auto& c = levels_.back();
        const int n = c.g.ndof;
        DeviceBuffer<RealT> basis(static_cast<std::size_t>(n)), y(static_cast<std::size_t>(n));
        std::vector<RealT> h_basis(static_cast<std::size_t>(n), RealT(0));
        std::vector<RealT> h_y(static_cast<std::size_t>(n));
        std::vector<RealT> A(static_cast<std::size_t>(n)*static_cast<std::size_t>(n), RealT(0));
        for (int j=0; j<n; ++j) {
            std::fill(h_basis.begin(), h_basis.end(), RealT(0));
            h_basis[static_cast<std::size_t>(j)] = RealT(1);
            basis.copy_from_host(h_basis.data(), h_basis.size());
            apply_level(int(levels_.size())-1, basis.data(), y.data());
            y.copy_to_host(h_y.data(), h_y.size());
            for (int i=0; i<n; ++i) A[static_cast<std::size_t>(i)*static_cast<std::size_t>(n)+static_cast<std::size_t>(j)] = h_y[static_cast<std::size_t>(i)];
        }
        coarseLu_.factor(n, A);
        dCoarseLu_.copy_from_host(coarseLu_.lu_storage().data(), coarseLu_.lu_storage().size());
        dCoarsePiv_.copy_from_host(coarseLu_.pivots().data(), coarseLu_.pivots().size());
        if (verbose_) std::cout << "coupled_gmg coarse dense LU ndof=" << n << " levels=" << levels_.size()
                                << " fineTau=" << double(levels_.front().tau)
                                << " coarseTau=" << double(levels_.back().tau) << "\n";
    }

    void solve_coarse(const RealT* b, RealT* x) {
        const int n = levels_.back().g.ndof;
        kernel_dense_lu_solve_serial<RealT><<<1, 1>>>(n, dCoarseLu_.data(), dCoarsePiv_.data(), b, x);
        MEMOIRS_CUDA_KERNEL_CHECK();
    }

    std::vector<Level> levels_;
    int preSmooth_ = 2, postSmooth_ = 2, coarseElem_ = 2, verbose_ = 0;
    RealT omega_ = RealT(0.70);
    RealT nu_ = RealT(1), tau_ = RealT(1), massCoeff_ = RealT(1), advScale_ = RealT(0), supgTauScale_ = RealT(1);
    int supgEnabled_ = 0;
    memoirs::solvers::DenseLu<RealT> coarseLu_;
    DeviceBuffer<RealT> dCoarseLu_;
    DeviceBuffer<int> dCoarsePiv_;
};

struct SolverReport {
    int iterations=0, converged=0, breakdown=0, breakdownCode=0;
    double breakdownValue=0, initialResidual=0, finalResidual=0, finalRelativeResidual=0;
};

template <class RealT>
struct GmresRightPrecWorkspace {
    int n = 0;
    int restart = 0;
    DeviceBuffer<RealT> d_Ax, d_r, d_w;
    std::vector<DeviceBuffer<RealT>> V, Z;
    std::vector<std::vector<double>> H;
    std::vector<double> cs, sn, g, y;
    GpuDotWorkspace<RealT> dots;

    void resize(int nn, int rr) {
        rr = std::max(2, rr);
        if (n == nn && restart == rr) return;
        n = nn; restart = rr;
        d_Ax.resize(static_cast<std::size_t>(n));
        d_r.resize(static_cast<std::size_t>(n));
        d_w.resize(static_cast<std::size_t>(n));
        V.clear(); Z.clear();
        V.reserve(static_cast<std::size_t>(restart + 1));
        Z.reserve(static_cast<std::size_t>(restart));
        for (int i=0; i<restart+1; ++i) V.emplace_back(static_cast<std::size_t>(n));
        for (int i=0; i<restart; ++i) Z.emplace_back(static_cast<std::size_t>(n));
        H.assign(static_cast<std::size_t>(restart+1), std::vector<double>(static_cast<std::size_t>(restart), 0.0));
        cs.assign(static_cast<std::size_t>(restart), 0.0);
        sn.assign(static_cast<std::size_t>(restart), 0.0);
        g.assign(static_cast<std::size_t>(restart+1), 0.0);
        y.assign(static_cast<std::size_t>(restart), 0.0);
    }
};

template <class RealT, class Operator, class RightPreconditioner>
SolverReport gmres_right_prec_ws(const Operator& A, RightPreconditioner& M, const RealT* d_b, RealT* d_x,
                                 RealT tol, int maxit, int restart, int printEvery,
                                 GmresRightPrecWorkspace<RealT>& ws) {
    const int n=A.size(); const int block=256;
    restart = std::max(2, restart);
    ws.resize(n, restart);
    auto& d_Ax = ws.d_Ax;
    auto& d_r = ws.d_r;
    auto& d_w = ws.d_w;
    auto& V = ws.V;
    auto& Z = ws.Z;
    auto& H = ws.H;
    auto& cs = ws.cs;
    auto& sn = ws.sn;
    auto& g = ws.g;
    auto& y = ws.y;
    auto& dots = ws.dots;
    SolverReport rep;
    const double bnorm = std::max(std::sqrt(dots.dot(n,d_b,d_b)),1.0);
    A.apply(d_x,d_Ax.data());
    kernel_subtract_local<RealT><<<div_up_local(n,block),block>>>(n,d_b,d_Ax.data(),d_r.data()); MEMOIRS_CUDA_KERNEL_CHECK();
    double beta = std::sqrt(dots.dot(n,d_r.data(),d_r.data()));
    rep.initialResidual=beta; rep.finalResidual=beta; rep.finalRelativeResidual=beta/bnorm;
    if (rep.finalRelativeResidual <= double(tol)) { rep.converged=1; return rep; }

    int totalIt=0;
    while(totalIt<maxit) {
        A.apply(d_x,d_Ax.data());
        kernel_subtract_local<RealT><<<div_up_local(n,block),block>>>(n,d_b,d_Ax.data(),d_r.data()); MEMOIRS_CUDA_KERNEL_CHECK();
        beta = std::sqrt(dots.dot(n,d_r.data(),d_r.data()));
        rep.finalResidual=beta; rep.finalRelativeResidual=beta/bnorm;
        if (rep.finalRelativeResidual <= double(tol)) { rep.converged=1; return rep; }
        for (auto& row : H) std::fill(row.begin(), row.end(), 0.0);
        std::fill(cs.begin(), cs.end(), 0.0);
        std::fill(sn.begin(), sn.end(), 0.0);
        std::fill(y.begin(), y.end(), 0.0);
        std::fill(g.begin(),g.end(),0.0); g[0]=beta;
        kernel_axpby_local<RealT><<<div_up_local(n,block),block>>>(n,RealT(1.0/beta),d_r.data(),RealT(0),V[0].data()); MEMOIRS_CUDA_KERNEL_CHECK();
        int lastJ=-1;
        for(int j=0;j<restart && totalIt<maxit;++j) {
            lastJ=j;
            M.apply(V[j].data(),Z[j].data());
            A.apply(Z[j].data(),d_w.data());
            for(int i=0;i<=j;++i) {
                H[i][j]=dots.dot(n,V[i].data(),d_w.data());
                kernel_axpby_local<RealT><<<div_up_local(n,block),block>>>(n,RealT(-H[i][j]),V[i].data(),RealT(1),d_w.data()); MEMOIRS_CUDA_KERNEL_CHECK();
            }
            H[j+1][j]=std::sqrt(dots.dot(n,d_w.data(),d_w.data()));
            if (H[j+1][j] > 0.0 && std::isfinite(H[j+1][j])) {
                kernel_axpby_local<RealT><<<div_up_local(n,block),block>>>(n,RealT(1.0/H[j+1][j]),d_w.data(),RealT(0),V[j+1].data()); MEMOIRS_CUDA_KERNEL_CHECK();
            } else H[j+1][j]=0.0;
            for(int i=0;i<j;++i) {
                const double hij=H[i][j], hi1j=H[i+1][j];
                H[i][j]=cs[i]*hij+sn[i]*hi1j;
                H[i+1][j]=-sn[i]*hij+cs[i]*hi1j;
            }
            const double h0=H[j][j], h1=H[j+1][j], den=std::hypot(h0,h1);
            if (!(den>0.0) || !std::isfinite(den)) { rep.breakdown=1; rep.breakdownCode=10; rep.breakdownValue=den; rep.iterations=totalIt; return rep; }
            cs[j]=h0/den; sn[j]=h1/den;
            H[j][j]=cs[j]*h0+sn[j]*h1; H[j+1][j]=0.0;
            const double gj=g[j], gj1=g[j+1];
            g[j]=cs[j]*gj+sn[j]*gj1; g[j+1]=-sn[j]*gj+cs[j]*gj1;
            ++totalIt;
            rep.iterations=totalIt; rep.finalResidual=std::abs(g[j+1]); rep.finalRelativeResidual=rep.finalResidual/bnorm;
            if(printEvery>0 && (totalIt==1 || totalIt%printEvery==0)) std::cout<<"cuda q2q1 right-gmres it="<<totalIt<<" rel="<<rep.finalRelativeResidual<<"\n";
            if(rep.finalRelativeResidual <= double(tol)) break;
        }
        if(lastJ<0) break;
        const int k=lastJ;
        for(int i=0;i<=k;++i) y[i]=g[i];
        for(int i=k;i>=0;--i) {
            double sum=y[i];
            for(int j=i+1;j<=k;++j) sum-=H[i][j]*y[j];
            if(!(std::abs(H[i][i])>0.0) || !std::isfinite(H[i][i])) { rep.breakdown=1; rep.breakdownCode=11; rep.breakdownValue=H[i][i]; rep.iterations=totalIt; return rep; }
            y[i]=sum/H[i][i];
        }
        for(int i=0;i<=k;++i) {
            kernel_axpby_local<RealT><<<div_up_local(n,block),block>>>(n,RealT(y[i]),Z[i].data(),RealT(1),d_x); MEMOIRS_CUDA_KERNEL_CHECK();
        }
        A.apply(d_x,d_Ax.data());
        kernel_subtract_local<RealT><<<div_up_local(n,block),block>>>(n,d_b,d_Ax.data(),d_r.data()); MEMOIRS_CUDA_KERNEL_CHECK();
        const double rnorm=std::sqrt(dots.dot(n,d_r.data(),d_r.data()));
        rep.finalResidual=rnorm; rep.finalRelativeResidual=rnorm/bnorm; rep.iterations=totalIt;
        if(printEvery>0) std::cout<<"cuda q2q1 right-gmres restart-end it="<<totalIt<<" rel="<<rep.finalRelativeResidual<<"\n";
        if(rep.finalRelativeResidual <= double(tol)) { rep.converged=1; return rep; }
    }
    return rep;
}

template <class RealT, class Operator, class RightPreconditioner>
SolverReport gmres_right_prec(const Operator& A, RightPreconditioner& M, const RealT* d_b, RealT* d_x,
                              RealT tol, int maxit, int restart, int printEvery) {
    GmresRightPrecWorkspace<RealT> ws;
    return gmres_right_prec_ws(A, M, d_b, d_x, tol, maxit, restart, printEvery, ws);
}

template <class RealT>
struct CheckedResidualWorkspace {
    int n = 0;
    DeviceBuffer<RealT> d_Ax, d_r;
    GpuDotWorkspace<RealT> dots;
    void resize(int nn) {
        if (n == nn) return;
        n = nn;
        d_Ax.resize(static_cast<std::size_t>(n));
        d_r.resize(static_cast<std::size_t>(n));
    }
};

template <class RealT, class Operator>
double checked_rel_residual_ws(const Operator& A, const RealT* d_b, const RealT* d_x,
                               CheckedResidualWorkspace<RealT>& ws) {
    const int n=A.size(); const int block=256;
    ws.resize(n);
    A.apply(d_x,ws.d_Ax.data());
    kernel_subtract_local<RealT><<<div_up_local(n,block),block>>>(n,d_b,ws.d_Ax.data(),ws.d_r.data()); MEMOIRS_CUDA_KERNEL_CHECK();
    return std::sqrt(ws.dots.dot(n,ws.d_r.data(),ws.d_r.data()))/std::max(std::sqrt(ws.dots.dot(n,d_b,d_b)),1.0);
}

template <class RealT, class Operator>
double checked_rel_residual(const Operator& A, const RealT* d_b, const RealT* d_x) {
    const int n=A.size(); const int block=256;
    DeviceBuffer<RealT> d_Ax(n), d_r(n);
    GpuDotWorkspace<RealT> dots;
    A.apply(d_x,d_Ax.data());
    kernel_subtract_local<RealT><<<div_up_local(n,block),block>>>(n,d_b,d_Ax.data(),d_r.data()); MEMOIRS_CUDA_KERNEL_CHECK();
    return std::sqrt(dots.dot(n,d_r.data(),d_r.data()))/std::max(std::sqrt(dots.dot(n,d_b,d_b)),1.0);
}

struct Q2Q1Errors { double velocityL2=0, pressureL2=0, pressureL2MeanShifted=0, pressureMeanShift=0; };

template <class RealT>
Q2Q1Errors compute_errors_host(Q2Q1GridCuda g, const std::vector<RealT>& x) {
    const double qp[4] = {-0.86113631159405257522, -0.33998104358485626480,
                           0.33998104358485626480,  0.86113631159405257522};
    const double qw[4] = { 0.34785484513745385737,  0.65214515486254614263,
                           0.65214515486254614263,  0.34785484513745385737};
    const double hx=g.hx, hy=g.hy, hz=g.hz, jac=hx*hy*hz/8.0;
    double intPdiff=0.0, vol=0.0;
    for(int ez=0;ez<g.N;++ez) for(int ey=0;ey<g.N;++ey) for(int ex=0;ex<g.N;++ex) {
        for(int iq=0;iq<4;++iq) { double Q1x[2],dQ1x[2]; q1_1d(qp[iq],Q1x,dQ1x); const double xx=(ex+0.5*(1+qp[iq]))*hx;
        for(int jq=0;jq<4;++jq) { double Q1y[2],dQ1y[2]; q1_1d(qp[jq],Q1y,dQ1y); const double yy=(ey+0.5*(1+qp[jq]))*hy;
        for(int kq=0;kq<4;++kq) { double Q1z[2],dQ1z[2]; q1_1d(qp[kq],Q1z,dQ1z); const double zz=(ez+0.5*(1+qp[kq]))*hz; const double wq=qw[iq]*qw[jq]*qw[kq]*jac;
            double ph=0; for(int az=0;az<2;++az) for(int ay=0;ay<2;++ay) for(int ax=0;ax<2;++ax) { const int a=q1_lid(ax,ay,az); const int id=p_id(g,ex+ax,ey+ay,ez+az); ph += Q1x[ax]*Q1y[ay]*Q1z[az]*double(x[3*g.nv+id]); }
            double u,v,w,p; stokes_exact(xx,yy,zz,u,v,w,p); intPdiff += wq*(ph-p); vol += wq;
        }}}
    }
    const double shift=intPdiff/vol;
    double ev2=0, ep2=0, eps2=0;
    for(int ez=0;ez<g.N;++ez) for(int ey=0;ey<g.N;++ey) for(int ex=0;ex<g.N;++ex) {
        for(int iq=0;iq<4;++iq) { double Q2x[3],dQ2x[3],ddQ2x[3], Q1x[2],dQ1x[2]; q2_1d(qp[iq],Q2x,dQ2x,ddQ2x); q1_1d(qp[iq],Q1x,dQ1x); const double xx=(ex+0.5*(1+qp[iq]))*hx;
        for(int jq=0;jq<4;++jq) { double Q2y[3],dQ2y[3],ddQ2y[3], Q1y[2],dQ1y[2]; q2_1d(qp[jq],Q2y,dQ2y,ddQ2y); q1_1d(qp[jq],Q1y,dQ1y); const double yy=(ey+0.5*(1+qp[jq]))*hy;
        for(int kq=0;kq<4;++kq) { double Q2z[3],dQ2z[3],ddQ2z[3], Q1z[2],dQ1z[2]; q2_1d(qp[kq],Q2z,dQ2z,ddQ2z); q1_1d(qp[kq],Q1z,dQ1z); const double zz=(ez+0.5*(1+qp[kq]))*hz; const double wq=qw[iq]*qw[jq]*qw[kq]*jac;
            double uh=0,vh=0,wh=0,ph=0;
            for(int az=0;az<3;++az) for(int ay=0;ay<3;++ay) for(int ax=0;ax<3;++ax) { const int a=q2_lid(ax,ay,az); const int id=v_id(g,2*ex+ax,2*ey+ay,2*ez+az); const double Na=Q2x[ax]*Q2y[ay]*Q2z[az]; uh += Na*double(x[id]); vh += Na*double(x[g.nv+id]); wh += Na*double(x[2*g.nv+id]); }
            for(int az=0;az<2;++az) for(int ay=0;ay<2;++ay) for(int ax=0;ax<2;++ax) { const int id=p_id(g,ex+ax,ey+ay,ez+az); ph += Q1x[ax]*Q1y[ay]*Q1z[az]*double(x[3*g.nv+id]); }
            double ue,ve,we,pe; stokes_exact(xx,yy,zz,ue,ve,we,pe);
            const double du=uh-ue, dv=vh-ve, dw=wh-we, dp=ph-pe, dps=(ph-shift)-pe;
            ev2 += wq*(du*du+dv*dv+dw*dw); ep2 += wq*dp*dp; eps2 += wq*dps*dps;
        }}}
    }
    Q2Q1Errors e; e.velocityL2=std::sqrt(ev2); e.pressureL2=std::sqrt(ep2); e.pressureL2MeanShifted=std::sqrt(eps2); e.pressureMeanShift=shift; return e;
}

}} // namespace memoirs::gpu

int main(int argc, char** argv) {
    try {
        const Cli cli = parse_cli(argc, argv);
        const auto appWallT0 = std::chrono::high_resolution_clock::now();
        const auto g = memoirs::gpu::make_q2q1_grid(cli.nElem);
        const Real tau = Real(memoirs::gpu::compute_tau(g, cli.nu, cli.tauScale, cli.tauMode, cli.tauC, cli.dt, cli.advScale));
        const Real massCoeff = Real(1.0 / cli.dt);
        using Op = memoirs::gpu::Q2Q1NsePicardOperator<Real>;

        memoirs::gpu::DeviceBuffer<Real> d_x(g.ndof), d_b(g.ndof), d_exact(g.ndof), d_old(g.ndof), d_beta(g.ndof);
        d_x.zero(); d_beta.zero();
        const int block=256;
        memoirs::gpu::kernel_fill_exact_q2q1<Real><<<memoirs::gpu::div_up_local(std::max(g.nv,g.np),block),block>>>(g,d_exact.data());
        MEMOIRS_CUDA_KERNEL_CHECK();
        memoirs::gpu::kernel_copy_local<Real><<<memoirs::gpu::div_up_local(g.ndof,block),block>>>(g.ndof,d_exact.data(),d_old.data());
        MEMOIRS_CUDA_KERNEL_CHECK();
        if (cli.betaInitial == 1) {
            memoirs::gpu::kernel_copy_local<Real><<<memoirs::gpu::div_up_local(g.ndof,block),block>>>(g.ndof,d_exact.data(),d_beta.data());
            MEMOIRS_CUDA_KERNEL_CHECK();
        }

        const Real supgTauScale = Real(cli.supgTauScale);
        Op A0(g, Real(cli.nu), tau, massCoeff, Real(cli.advScale), cli.supg, supgTauScale, d_beta.data());
        if (cli.rhsMode == 1) A0.apply(d_exact.data(), d_b.data());
        else A0.assemble_rhs_from_old(d_old.data(), d_b.data());

        std::cout << "Memoirs CUDA structured Q2/Q1 Navier-Stokes Picard PSPG MMS development test\n";
        std::cout << "precision                 = " << kPrecisionName << "\n";
        std::cout << "velocityGrid              = " << g.nvx << "x" << g.nvy << "x" << g.nvz << " Q2 nodes\n";
        std::cout << "pressureGrid              = " << g.npx << "x" << g.npy << "x" << g.npz << " Q1 nodes\n";
        std::cout << "elements                  = " << g.N << "x" << g.N << "x" << g.N << "\n";
        std::cout << "velocityDofsPerComponent  = " << g.nv << "\n";
        std::cout << "pressureDofs              = " << g.np << "\n";
        std::cout << "unknowns                  = " << g.ndof << "\n";
        std::cout << "operator                  = monolithic_q2q1_nse_picard_pspg_supg_matrix_free_v2\n";
        std::cout << "mms                       = stationary_divergence_free_sine_velocity_mild_convection\n";
        std::cout << "velocityBC                = homogeneous_dirichlet_all_faces\n";
        std::cout << "pressurePin               = p_node_0_exact_value_0\n";
        std::cout << "timeDiscretization        = backward_euler_single_step_mms_old_exact\n";
        std::cout << "linearization             = Picard_beta_dot_grad_u\n";
        std::cout << "supg                      = " << (cli.supg ? "on" : "off") << "\n";
        std::cout << "supgTauScale              = " << cli.supgTauScale << "\n";
        std::cout << "pspg                      = full_residual_time_convection_diffusion_pressure\n";
        std::cout << "solver                    = right_gmres\n";
        std::cout << "gmresRestart              = " << cli.gmresRestart << "\n";
        std::cout << "rightPreconditioner       = ";
        if (cli.prec == "block_gmg") std::cout << "block_scalar_gmg_velocity_q2grid_pressure_q1grid\n";
        else if (cli.prec == "operator_inverse_diagonal") std::cout << "assembled_operator_inverse_diagonal\n";
        else std::cout << "coupled_q2q1_monolithic_gmg_rediscretized_v1\n";
        if (cli.prec == "operator_inverse_diagonal") {
            std::cout << "diagIncludes              = mass+diffusion+picard_convection"
                      << (cli.supg ? "+supg" : "")
                      << "+pspg_pressure_block\n";
        }
        if (cli.prec == "coupled_gmg") {
            std::cout << "coupledGmgHierarchy       = power_of_two_to_2x2x2_elements\n";
            std::cout << "coupledGmgSmoother        = monolithic_weighted_jacobi_on_operator_diagonal\n";
            std::cout << "coupledGmgCoarseSolve     = dense_lu_device_solve_cpu_factor_full_monolithic_coarse_operator\n";
            std::cout << "coupledGmgPrePostOmega    = " << cli.mgPre << "/" << cli.mgPost << "/" << cli.mgOmega << "\n";
        }
        std::cout << "rhsMode                   = " << (cli.rhsMode==0?"continuous_quadrature":"algebraic_Ax_exact_current_picard") << "\n";
        std::cout << "nu                        = " << cli.nu << "\n";
        std::cout << "dt                        = " << cli.dt << "\n";
        std::cout << "massCoeff                 = " << double(massCoeff) << "\n";
        std::cout << "advScale                  = " << cli.advScale << "\n";
        std::cout << "maxPicard                 = " << cli.maxPicard << "\n";
        std::cout << "picardTol                 = " << cli.picardTol << "\n";
        std::cout << "betaInitial               = " << (cli.betaInitial==0?"zero":"exact") << "\n";
        std::cout << "tauScale                  = " << cli.tauScale << "\n";
        std::cout << "tauMode                   = " << cli.tauMode << "\n";
        std::cout << "tauC                      = " << cli.tauC << "\n";
        std::cout << "tau                       = " << std::setprecision(8) << double(tau) << "\n";
        std::cout << "linearTol                 = " << cli.tol << "\n";
        std::cout << "linearMaxit               = " << cli.maxit << "\n";
        std::cout << "mgRebuildPolicy           = " << cli.mgRebuildPolicy << "\n";
        std::cout << "timing                    = " << (cli.timing ? "on" : "off") << "\n";

        memoirs::gpu::SolverReport rep;
        double totalSeconds=0.0;
        double rhsSeconds=0.0, rebuildSeconds=0.0, linearSeconds=0.0, nonlinearResidualSeconds=0.0, betaUpdateSeconds=0.0;
        double finalResidualCheckSeconds=0.0, errorSeconds=0.0;
        int preconditionerRebuilds=0;
        int totalLinearIterations=0;
        double nlRel=1e300;
        int picardUsed=0;
        memoirs::gpu::GmresRightPrecWorkspace<Real> gmresWs;
        memoirs::gpu::CheckedResidualWorkspace<Real> nonlinearResidualWs;
        memoirs::gpu::CheckedResidualWorkspace<Real> finalResidualWs;

        auto run_picard_with_preconditioner = [&](auto& M) {
            bool preconditionerBuilt = false;
            for (int piter=0; piter<cli.maxPicard; ++piter) {
                ++picardUsed;
                Op A(g, Real(cli.nu), tau, massCoeff, Real(cli.advScale), cli.supg, supgTauScale, d_beta.data());
                const auto rhsT0=std::chrono::high_resolution_clock::now();
                if (cli.rhsMode == 1) A.apply(d_exact.data(), d_b.data());
                else A.assemble_rhs_from_old(d_old.data(), d_b.data());
                const auto rhsT1=std::chrono::high_resolution_clock::now();
                rhsSeconds += std::chrono::duration<double>(rhsT1-rhsT0).count();

                const bool rebuildThisPicard = (cli.mgRebuildPolicy == "every_picard") || !preconditionerBuilt;
                if (rebuildThisPicard) {
                    const auto rbT0=std::chrono::high_resolution_clock::now();
                    M.rebuild(Real(cli.nu), tau, massCoeff, Real(cli.advScale), cli.supg, supgTauScale, d_beta.data());
                    const auto rbT1=std::chrono::high_resolution_clock::now();
                    rebuildSeconds += std::chrono::duration<double>(rbT1-rbT0).count();
                    preconditionerBuilt = true;
                    ++preconditionerRebuilds;
                }

                const auto t0=std::chrono::high_resolution_clock::now();
                rep = memoirs::gpu::gmres_right_prec_ws<Real>(A,M,d_b.data(),d_x.data(),Real(cli.tol),cli.maxit,cli.gmresRestart,cli.printEvery,gmresWs);
                const auto t1=std::chrono::high_resolution_clock::now();
                const double linSec = std::chrono::duration<double>(t1-t0).count();
                totalSeconds += linSec;
                linearSeconds += linSec;
                totalLinearIterations += rep.iterations;

                Op ANL(g, Real(cli.nu), tau, massCoeff, Real(cli.advScale), cli.supg, supgTauScale, d_x.data());
                const auto nlT0=std::chrono::high_resolution_clock::now();
                nlRel = memoirs::gpu::checked_rel_residual_ws<Real>(ANL,d_b.data(),d_x.data(),nonlinearResidualWs);
                const auto nlT1=std::chrono::high_resolution_clock::now();
                nonlinearResidualSeconds += std::chrono::duration<double>(nlT1-nlT0).count();
                std::cout << "picard                    = " << piter+1
                          << " linearIts=" << rep.iterations
                          << " linearRel=" << rep.finalRelativeResidual
                          << " nonlinearAlgebraicRel=" << std::setprecision(12) << nlRel
                          << " precondRebuilt=" << rebuildThisPicard
                          << " converged=" << rep.converged << "\n";
                if (!rep.converged) break;
                if (nlRel <= cli.picardTol) break;
                const auto buT0=std::chrono::high_resolution_clock::now();
                memoirs::gpu::kernel_copy_local<Real><<<memoirs::gpu::div_up_local(g.ndof,block),block>>>(g.ndof,d_x.data(),d_beta.data());
                MEMOIRS_CUDA_KERNEL_CHECK();
                const auto buT1=std::chrono::high_resolution_clock::now();
                betaUpdateSeconds += std::chrono::duration<double>(buT1-buT0).count();
            }
        };

        if (cli.prec == "operator_inverse_diagonal") {
            memoirs::gpu::Q2Q1OperatorInverseDiagPrec<Real> M(g);
            run_picard_with_preconditioner(M);
        } else if (cli.prec == "coupled_gmg") {
            memoirs::gpu::Q2Q1CoupledGmgPrec<Real> M(g, cli.mgPre, cli.mgPost, Real(cli.mgOmega), cli.mgCoarseElem, cli.mgVerbose);
            run_picard_with_preconditioner(M);
        } else {
            memoirs::gpu::Q2Q1BlockGmgPrec<Real> M(g, Real(cli.nu), tau);
            run_picard_with_preconditioner(M);
        }

        Op Afinal(g, Real(cli.nu), tau, massCoeff, Real(cli.advScale), cli.supg, supgTauScale, d_x.data());
        const auto frcT0=std::chrono::high_resolution_clock::now();
        const double checked = memoirs::gpu::checked_rel_residual_ws<Real>(Afinal,d_b.data(),d_x.data(),finalResidualWs);
        const auto frcT1=std::chrono::high_resolution_clock::now();
        finalResidualCheckSeconds += std::chrono::duration<double>(frcT1-frcT0).count();

        const auto errT0=std::chrono::high_resolution_clock::now();
        std::vector<Real> h_x(g.ndof);
        MEMOIRS_CUDA_CHECK(cudaMemcpy(h_x.data(), d_x.data(), sizeof(Real)*std::size_t(g.ndof), cudaMemcpyDeviceToHost));
        const auto err = memoirs::gpu::compute_errors_host(g,h_x);
        const auto errT1=std::chrono::high_resolution_clock::now();
        errorSeconds += std::chrono::duration<double>(errT1-errT0).count();

        std::cout << "--------------- q2q1 nse picard pspg solve report ---------------\n";
        std::cout << "picardIterations          = " << picardUsed << "\n";
        std::cout << "linearIterationsLast      = " << rep.iterations << "\n";
        std::cout << "linearConvergedLast       = " << rep.converged << "\n";
        std::cout << "breakdown                 = " << rep.breakdown << "\n";
        std::cout << "breakdownCode             = " << rep.breakdownCode << "\n";
        std::cout << std::setprecision(17);
        std::cout << "breakdownValue            = " << rep.breakdownValue << "\n";
        std::cout << "initialResidualLast       = " << rep.initialResidual << "\n";
        std::cout << "finalResidualLast         = " << rep.finalResidual << "\n";
        std::cout << "finalRelativeResidualLast = " << rep.finalRelativeResidual << "\n";
        std::cout << "nonlinearAlgebraicRelativeResidual = " << checked << "\n";
        std::cout << "velocityL2Quadrature      = " << err.velocityL2 << "\n";
        std::cout << "pressureL2Quadrature      = " << err.pressureL2 << "\n";
        std::cout << "pressureMeanShift         = " << err.pressureMeanShift << "\n";
        std::cout << "pressureL2MeanShiftedQuadrature = " << err.pressureL2MeanShifted << "\n";
        const auto appWallT1 = std::chrono::high_resolution_clock::now();
        const double wallSeconds = std::chrono::duration<double>(appWallT1-appWallT0).count();
        std::cout << "------------------------------------------------------------------\n";
        std::cout << "preconditionerRebuilds    = " << preconditionerRebuilds << "\n";
        std::cout << "totalLinearIterations     = " << totalLinearIterations << "\n";
        std::cout << "solveSeconds              = " << totalSeconds / double(std::max(picardUsed,1)) << "\n";
        if (cli.timing) {
            std::cout << "timingRhsSeconds          = " << rhsSeconds << "\n";
            std::cout << "timingRebuildSeconds      = " << rebuildSeconds << "\n";
            std::cout << "timingLinearSeconds       = " << linearSeconds << "\n";
            std::cout << "timingNonlinearResidualSeconds = " << nonlinearResidualSeconds << "\n";
            std::cout << "timingBetaUpdateSeconds   = " << betaUpdateSeconds << "\n";
            std::cout << "timingFinalResidualSeconds = " << finalResidualCheckSeconds << "\n";
            std::cout << "timingErrorSeconds        = " << errorSeconds << "\n";
            std::cout << "timingWallSeconds         = " << wallSeconds << "\n";
        }
        return (rep.converged && checked <= cli.picardTol*10.0) ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "FATAL: " << e.what() << "\n";
        return 1;
    }
}
