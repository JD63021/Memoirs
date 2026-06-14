#include "memoirs/diagnostics/GpuMemoryMonitor.hpp"
#include "memoirs/diagnostics/CudaTiming.hpp"
#include "memoirs/diagnostics/StructuredQ1StokesErrorDiagnostics.hpp"
#include "memoirs/gpu/CudaUtils.cuh"
#include "memoirs/gpu/DeviceBuffer.cuh"
#include "memoirs/structured/StructuredGrid3D.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
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
    int nElem = 16;
    int maxit = 1000;
    int gmresRestart = 50;
    std::string solver = "bicgstab";
    double tol = 1e-10;
    double nu = 1.0;
    double tauScale = 0.25;
    std::string tauMode = "simple";
    double tauC = 4.0;
    double tauDt = 0.0;
    double tauAdvMag = 0.0;
    double pGradSign = 1.0;
    double pspgRhsSign = 1.0;
    int pspgForceMode = 0; // 0=fullF, 1=gradPOnly, 2=none
    int printEvery = 25;
    int repeats = 1;
    // 0 = continuous quadrature RHS, 1 = algebraic A*x_exact RHS smoke test.
    int rhsMode = 0;
    double dt = 1.0;
    double advScale = 1.0;
    int maxPicard = 20;
    double picardTol = 1e-6;
    int betaInitial = 0; // 0 zero, 1 exact
    int supg = 1;
    double supgTauScale = 1.0;
    int timing = 1;
    std::string prec = "diagonal";
    int mgPre = 2;
    int mgPost = 2;
    double mgOmega = 0.70;
    int mgVerbose = 0;
    std::string blockGmgPressure = "pspg";   // pspg, mass, mass_pspg
    int blockGmgConvection = 0;              // add constant central convection surrogate to velocity block
    int blockGmgSupg = 0;                    // add SUPG streamline-diffusion surrogate to velocity block
    double blockGmgConvScale = 1.0;
    int pcdKpMaxit = 30;
    double pcdKpTol = 1e-8;
    double pcdSign = -1.0;
    std::string pcdVelSolver = "gmg"; // gmg, gmres_diag
    int pcdVelIters = 0;              // fixed inner GMRES iterations when pcdVelSolver=gmres_diag
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
        else if (a == "-solver") {
            if (i + 1 >= argc) throw std::runtime_error("missing string argument for -solver");
            c.solver = argv[++i];
        }
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
        else if (a == "-pGradSign") c.pGradSign = arg_double(i, argc, argv);
        else if (a == "-pspgRhsSign") c.pspgRhsSign = arg_double(i, argc, argv);
        else if (a == "-pspgForceMode") {
            if (i + 1 >= argc) throw std::runtime_error("missing string argument for -pspgForceMode");
            const std::string m = argv[++i];
            if (m == "fullF") c.pspgForceMode = 0;
            else if (m == "gradPOnly") c.pspgForceMode = 1;
            else if (m == "none") c.pspgForceMode = 2;
            else throw std::runtime_error("unknown -pspgForceMode; use fullF, gradPOnly, or none");
        }
        else if (a == "-printEvery") c.printEvery = arg_int(i, argc, argv);
        else if (a == "-repeats") c.repeats = arg_int(i, argc, argv);
        else if (a == "-rhsMode") c.rhsMode = arg_int(i, argc, argv);
        else if (a == "-dt") c.dt = arg_double(i, argc, argv);
        else if (a == "-advScale") c.advScale = arg_double(i, argc, argv);
        else if (a == "-maxPicard") c.maxPicard = arg_int(i, argc, argv);
        else if (a == "-picardTol") c.picardTol = arg_double(i, argc, argv);
        else if (a == "-betaInitial") c.betaInitial = arg_int(i, argc, argv);
        else if (a == "-supg") c.supg = arg_int(i, argc, argv);
        else if (a == "-supgTauScale") c.supgTauScale = arg_double(i, argc, argv);
        else if (a == "-timing") c.timing = arg_int(i, argc, argv);
        else if (a == "-prec") {
            if (i + 1 >= argc) throw std::runtime_error("missing string argument for -prec");
            c.prec = argv[++i];
        }
        else if (a == "-mgPre") c.mgPre = arg_int(i, argc, argv);
        else if (a == "-mgPost") c.mgPost = arg_int(i, argc, argv);
        else if (a == "-mgOmega") c.mgOmega = arg_double(i, argc, argv);
        else if (a == "-mgVerbose") c.mgVerbose = arg_int(i, argc, argv);
        else if (a == "-blockGmgPressure" || a == "-pPrec" || a == "-pressurePrec") {
            if (i + 1 >= argc) throw std::runtime_error("missing string argument for -blockGmgPressure");
            c.blockGmgPressure = argv[++i];
            if (!(c.blockGmgPressure == "pspg" || c.blockGmgPressure == "mass" || c.blockGmgPressure == "mass_pspg" || c.blockGmgPressure == "pspg_mass"))
                throw std::runtime_error("unknown -blockGmgPressure; use pspg, mass, or mass_pspg");
            if (c.blockGmgPressure == "pspg_mass") c.blockGmgPressure = "mass_pspg";
        }
        else if (a == "-blockGmgConvection" || a == "-velPrecConvection") c.blockGmgConvection = arg_int(i, argc, argv);
        else if (a == "-blockGmgSupg" || a == "-velPrecSupg") c.blockGmgSupg = arg_int(i, argc, argv);
        else if (a == "-blockGmgConvScale") c.blockGmgConvScale = arg_double(i, argc, argv);
        else if (a == "-pcdKpMaxit") c.pcdKpMaxit = arg_int(i, argc, argv);
        else if (a == "-pcdKpTol") c.pcdKpTol = arg_double(i, argc, argv);
        else if (a == "-pcdSign") c.pcdSign = arg_double(i, argc, argv);
        else if (a == "-pcdVelSolver") {
            if (i + 1 >= argc) throw std::runtime_error("missing string argument for -pcdVelSolver");
            c.pcdVelSolver = argv[++i];
        }
        else if (a == "-pcdVelIters") c.pcdVelIters = arg_int(i, argc, argv);
        else throw std::runtime_error("unknown argument: " + a);
    }

    if (c.nElem < 2) throw std::runtime_error("-n must be >= 2");
    if (c.maxit < 1) throw std::runtime_error("-maxit must be >= 1");
    if (c.gmresRestart < 2) throw std::runtime_error("-gmresRestart must be >= 2");
    if (c.solver != "bicgstab" && c.solver != "gmres") {
        throw std::runtime_error("unknown -solver; use bicgstab or gmres");
    }
    if (c.repeats < 1) throw std::runtime_error("-repeats must be >= 1");
    if (!(c.nu > 0.0)) throw std::runtime_error("-nu must be positive");
    if (!(c.tauScale > 0.0)) throw std::runtime_error("-tauScale must be positive");
    if (!(c.tauC > 0.0)) throw std::runtime_error("-tauC must be positive");
    if (c.tauDt < 0.0) throw std::runtime_error("-tauDt must be >= 0");
    if (c.tauAdvMag < 0.0) throw std::runtime_error("-tauAdvMag must be >= 0");
    if (c.tauMode != "simple" &&
        c.tauMode != "sphere" &&
        c.tauMode != "sphereC" &&
        c.tauMode != "tezduyar" &&
        c.tauMode != "metric") {
        throw std::runtime_error("unknown -tauMode; use simple, sphere, sphereC, tezduyar, or metric");
    }
    if (c.rhsMode != 0 && c.rhsMode != 1) {
        throw std::runtime_error("-rhsMode must be 0 (continuous) or 1 (algebraic)");
    }
    if (!(c.dt > 0.0)) throw std::runtime_error("-dt must be positive");
    if (c.maxPicard < 1) throw std::runtime_error("-maxPicard must be >= 1");
    if (!(c.picardTol > 0.0)) throw std::runtime_error("-picardTol must be positive");
    if (c.betaInitial != 0 && c.betaInitial != 1) throw std::runtime_error("-betaInitial must be 0 or 1");
    if (c.supg != 0 && c.supg != 1) throw std::runtime_error("-supg must be 0 or 1");
    if (!(c.supgTauScale >= 0.0)) throw std::runtime_error("-supgTauScale must be nonnegative");
    if (c.timing != 0 && c.timing != 1) throw std::runtime_error("-timing must be 0 or 1");
    if (c.prec != "diagonal" && c.prec != "gmg" && c.prec != "coupled_gmg" && c.prec != "block_gmg" && c.prec != "field_gmg" && c.prec != "pcd" && c.prec != "pcd_gmg") {
        throw std::runtime_error("unknown -prec; use diagonal, gmg/coupled_gmg, block_gmg, or pcd");
    }
    if (c.mgPre < 0 || c.mgPost < 0) throw std::runtime_error("-mgPre/-mgPost must be >= 0");
    if (!(c.mgOmega > 0.0)) throw std::runtime_error("-mgOmega must be positive");
    if (c.mgVerbose != 0 && c.mgVerbose != 1) throw std::runtime_error("-mgVerbose must be 0 or 1");
    if (c.pcdKpMaxit < 1) throw std::runtime_error("-pcdKpMaxit must be >= 1");
    if (!(c.pcdKpTol > 0.0)) throw std::runtime_error("-pcdKpTol must be positive");
    if (!(c.pcdSign == -1.0 || c.pcdSign == 1.0)) throw std::runtime_error("-pcdSign must be -1 or 1");
    if (c.pcdVelSolver != "gmg" && c.pcdVelSolver != "gmres_diag") {
        throw std::runtime_error("unknown -pcdVelSolver; use gmg or gmres_diag");
    }
    if (c.pcdVelIters < 0) throw std::runtime_error("-pcdVelIters must be >= 0");
    if (c.pcdVelSolver == "gmres_diag" && c.pcdVelIters < 1) {
        throw std::runtime_error("-pcdVelIters must be >= 1 when -pcdVelSolver gmres_diag");
    }

    return c;
}

} // namespace

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
            " expr=" + expr + " status=" + cublas_status_name(s));
    }
}

#define MEMOIRS_LOCAL_CUBLAS_CHECK(expr) \
    ::memoirs::gpu::cublas_check((expr), __FILE__, __LINE__, #expr)

class CublasHandleLocal {
public:
    CublasHandleLocal() {
        MEMOIRS_LOCAL_CUBLAS_CHECK(cublasCreate(&h_));
        MEMOIRS_LOCAL_CUBLAS_CHECK(cublasSetPointerMode(h_, CUBLAS_POINTER_MODE_HOST));
    }

    CublasHandleLocal(const CublasHandleLocal&) = delete;
    CublasHandleLocal& operator=(const CublasHandleLocal&) = delete;

    ~CublasHandleLocal() {
        if (h_) cublasDestroy(h_);
    }

    cublasHandle_t get() const { return h_; }

private:
    cublasHandle_t h_ = nullptr;
};

inline cublasStatus_t cublas_dot_dispatch_local(cublasHandle_t h,
                                                int n,
                                                const double* a,
                                                const double* b,
                                                double* out) {
    return cublasDdot(h, n, a, 1, b, 1, out);
}

inline cublasStatus_t cublas_dot_dispatch_local(cublasHandle_t h,
                                                int n,
                                                const float* a,
                                                const float* b,
                                                float* out) {
    return cublasSdot(h, n, a, 1, b, 1, out);
}

template <class RealT>
class GpuDotLocal {
public:
    double dot(int n, const RealT* a, const RealT* b) {
        RealT out = RealT(0);
        MEMOIRS_LOCAL_CUBLAS_CHECK(cublas_dot_dispatch_local(handle_.get(), n, a, b, &out));
        return double(out);
    }

private:
    CublasHandleLocal handle_;
};

struct Grid3DCudaLocal {
    int nx = 0;
    int ny = 0;
    int nz = 0;
    double hx = 0.0;
    double hy = 0.0;
    double hz = 0.0;

    __host__ __device__ int n_nodes_host() const { return nx * ny * nz; }
};

__host__ inline Grid3DCudaLocal to_cuda_grid_local(
    const memoirs::structured::StructuredGrid3D& g) {
    Grid3DCudaLocal d;
    d.nx = g.nx;
    d.ny = g.ny;
    d.nz = g.nz;
    d.hx = g.hx();
    d.hy = g.hy();
    d.hz = g.hz();
    return d;
}

__host__ __device__ inline int g_id_local(const Grid3DCudaLocal g,
                                          int i,
                                          int j,
                                          int k) {
    return (k * g.ny + j) * g.nx + i;
}

__host__ __device__ inline bool g_boundary_local(const Grid3DCudaLocal g,
                                                 int i,
                                                 int j,
                                                 int k) {
    return i == 0 || j == 0 || k == 0 ||
           i == g.nx - 1 || j == g.ny - 1 || k == g.nz - 1;
}

inline int div_up_local(int n, int block) {
    return (n + block - 1) / block;
}

template <class RealT>
__global__ void kernel_zero_local(int n, RealT* x) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q < n) x[q] = RealT(0);
}

template <class RealT>
__global__ void kernel_copy_local(int n, const RealT* a, RealT* b) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q < n) b[q] = a[q];
}

template <class RealT>
__global__ void kernel_subtract_local(int n, const RealT* a, const RealT* b, RealT* r) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q < n) r[q] = a[q] - b[q];
}

template <class RealT>
__global__ void kernel_axpby_local(int n,
                                   RealT a,
                                   const RealT* x,
                                   RealT b,
                                   RealT* y) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q < n) y[q] = a * x[q] + b * y[q];
}

template <class RealT>
__global__ void kernel_combination3_local(int n,
                                          RealT a,
                                          const RealT* x,
                                          RealT b,
                                          const RealT* y,
                                          RealT c,
                                          const RealT* z,
                                          RealT* out) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q < n) out[q] = a * x[q] + b * y[q] + c * z[q];
}

template <class RealT>
__host__ __device__ inline RealT stokes_local_ke_q1(const Grid3DCudaLocal g,
                                           int ra,
                                           int ca) {
    const int rix = ra & 1;
    const int riy = (ra >> 1) & 1;
    const int riz = (ra >> 2) & 1;

    const int cix = ca & 1;
    const int ciy = (ca >> 1) & 1;
    const int ciz = (ca >> 2) & 1;

    const RealT M[2][2] = {
        {RealT(2.0 / 3.0), RealT(1.0 / 3.0)},
        {RealT(1.0 / 3.0), RealT(2.0 / 3.0)}
    };

    const RealT K[2][2] = {
        {RealT(0.5), RealT(-0.5)},
        {RealT(-0.5), RealT(0.5)}
    };

    const RealT hx = RealT(g.hx);
    const RealT hy = RealT(g.hy);
    const RealT hz = RealT(g.hz);

    const RealT cx = hy * hz / (RealT(2) * hx);
    const RealT cy = hx * hz / (RealT(2) * hy);
    const RealT cz = hx * hy / (RealT(2) * hz);

    return cx * K[rix][cix] * M[riy][ciy] * M[riz][ciz] +
           cy * M[rix][cix] * K[riy][ciy] * M[riz][ciz] +
           cz * M[rix][cix] * M[riy][ciy] * K[riz][ciz];
}

__device__ inline int stokes_local_node_id(const Grid3DCudaLocal g,
                                           int ex,
                                           int ey,
                                           int ez,
                                           int a) {
    const int ix = a & 1;
    const int iy = (a >> 1) & 1;
    const int iz = (a >> 2) & 1;
    return g_id_local(g, ex + ix, ey + iy, ez + iz);
}

template <class RealT>
__host__ __device__ inline RealT stokes_local_b_q1(const Grid3DCudaLocal g,
                                          int qa,
                                          int ub,
                                          int comp) {
    const RealT gp = RealT(0.577350269189625764509148780501957456);
    const RealT q[2] = {-gp, gp};

    const RealT hx = RealT(g.hx);
    const RealT hy = RealT(g.hy);
    const RealT hz = RealT(g.hz);
    const RealT jac = hx * hy * hz / RealT(8);

    RealT val = RealT(0);

    for (int iq = 0; iq < 2; ++iq) {
        const RealT xi = q[iq];
        const RealT Nx[2] = {RealT(0.5) * (RealT(1) - xi),
                             RealT(0.5) * (RealT(1) + xi)};
        const RealT dNx[2] = {RealT(-1) / hx, RealT(1) / hx};

        for (int jq = 0; jq < 2; ++jq) {
            const RealT eta = q[jq];
            const RealT Ny[2] = {RealT(0.5) * (RealT(1) - eta),
                                 RealT(0.5) * (RealT(1) + eta)};
            const RealT dNy[2] = {RealT(-1) / hy, RealT(1) / hy};

            for (int kq = 0; kq < 2; ++kq) {
                const RealT zeta = q[kq];
                const RealT Nz[2] = {RealT(0.5) * (RealT(1) - zeta),
                                     RealT(0.5) * (RealT(1) + zeta)};
                const RealT dNz[2] = {RealT(-1) / hz, RealT(1) / hz};

                const int qx = qa & 1;
                const int qy = (qa >> 1) & 1;
                const int qz = (qa >> 2) & 1;

                const int ux = ub & 1;
                const int uy = (ub >> 1) & 1;
                const int uz = (ub >> 2) & 1;

                const RealT Na = Nx[qx] * Ny[qy] * Nz[qz];

                RealT dNb = RealT(0);
                if (comp == 0) dNb = dNx[ux] * Ny[uy] * Nz[uz];
                else if (comp == 1) dNb = Nx[ux] * dNy[uy] * Nz[uz];
                else dNb = Nx[ux] * Ny[uy] * dNz[uz];

                val += Na * dNb * jac;
            }
        }
    }

    return val;
}

__device__ inline void stokes_mms_exact_device(double x,
                                               double y,
                                               double z,
                                               double& u,
                                               double& v,
                                               double& w,
                                               double& p) {
    const double pi = 3.141592653589793238462643383279502884;

    const double sx = sin(pi * x);
    const double sy = sin(pi * y);
    const double sz = sin(pi * z);
    const double cx = cos(pi * x);
    const double cy = cos(pi * y);

    u = 2.0 * pi * sx * sx * sy * cy * sz * sz;
    v = -2.0 * pi * sx * cx * sy * sy * sz * sz;
    w = 0.0;
    // Mean-zero pressure, analogous to MATLAB p=cos(pi*x)*sin(pi*y).
    // In 3D: integral over x is zero, and p(0,0,0)=0 for the pressure pin.
    p = cx * sy * sz;
}


__device__ inline void stokes_mms_gradp_device(double x,
                                               double y,
                                               double z,
                                               double pGradSign,
                                               double& px,
                                               double& py,
                                               double& pz) {
    const double pi = 3.141592653589793238462643383279502884;
    const double sx = sin(pi * x);
    const double sy = sin(pi * y);
    const double sz = sin(pi * z);
    const double cx = cos(pi * x);
    const double cy = cos(pi * y);
    const double cz = cos(pi * z);

    px = pGradSign * (-pi * sx * sy * sz);
    py = pGradSign * ( pi * cx * cy * sz);
    pz = pGradSign * ( pi * cx * sy * cz);
}

__device__ inline void stokes_mms_force_device(double x,
                                               double y,
                                               double z,
                                               double nu,
                                               double pGradSign,
                                               double& fx,
                                               double& fy,
                                               double& fz) {
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

    const double lap_u =
        4.0 * pi * pi * pi * B *
        (c2x * C - 2.0 * A * C + A * c2z);

    const double lap_v =
        8.0 * pi * pi * pi * D * E * C -
        4.0 * pi * pi * pi * D * c2y * C -
        4.0 * pi * pi * pi * D * E * c2z;

    const double px = -pi * sx * sy * sz;
    const double py =  pi * cx * cy * sz;
    const double pz =  pi * cx * sy * cz;

    fx = -nu * lap_u + pGradSign * px;
    fy = -nu * lap_v + pGradSign * py;
    fz = pGradSign * pz;
}

template <class RealT>
__global__ void kernel_fill_stokes_exact(Grid3DCudaLocal g, RealT* x) {
    const int nn = g.n_nodes_host();
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= nn) return;

    const int i = q % g.nx;
    const int tmp = q / g.nx;
    const int j = tmp % g.ny;
    const int k = tmp / g.ny;

    const double xx = double(i) * g.hx;
    const double yy = double(j) * g.hy;
    const double zz = double(k) * g.hz;

    double u, v, w, p;
    stokes_mms_exact_device(xx, yy, zz, u, v, w, p);

    x[q] = RealT(u);
    x[nn + q] = RealT(v);
    x[2 * nn + q] = RealT(w);
    x[3 * nn + q] = RealT(p);
}

template <class RealT>
__global__ void kernel_apply_stokes_pspg_elements(Grid3DCudaLocal g,
                                                  RealT nu,
                                                  RealT tau,
                                                  const RealT* x,
                                                  RealT* y) {
    const int nn = g.n_nodes_host();
    const int exN = g.nx - 1;
    const int eyN = g.ny - 1;
    const int ezN = g.nz - 1;
    const int nElem = exN * eyN * ezN;

    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nElem) return;

    const int ex = e % exN;
    const int tmp = e / exN;
    const int ey = tmp % eyN;
    const int ez = tmp / eyN;

    constexpr int pPin = 0;

    for (int a = 0; a < 8; ++a) {
        const int rowNode = stokes_local_node_id(g, ex, ey, ez, a);
        const int ai = rowNode % g.nx;
        const int atmp = rowNode / g.nx;
        const int aj = atmp % g.ny;
        const int ak = atmp / g.ny;

        const bool rowBoundary = g_boundary_local(g, ai, aj, ak);

        if (!rowBoundary) {
            RealT yu[3] = {RealT(0), RealT(0), RealT(0)};

            for (int b = 0; b < 8; ++b) {
                const int colNode = stokes_local_node_id(g, ex, ey, ez, b);
                const int bi = colNode % g.nx;
                const int btmp = colNode / g.nx;
                const int bj = btmp % g.ny;
                const int bk = btmp / g.ny;
                const bool colBoundary = g_boundary_local(g, bi, bj, bk);

                const RealT ke = stokes_local_ke_q1<RealT>(g, a, b);

                if (!colBoundary) {
                    yu[0] += nu * ke * x[colNode];
                    yu[1] += nu * ke * x[nn + colNode];
                    yu[2] += nu * ke * x[2 * nn + colNode];
                }

                if (colNode != pPin) {
                    const RealT bx = stokes_local_b_q1<RealT>(g, b, a, 0);
                    const RealT by = stokes_local_b_q1<RealT>(g, b, a, 1);
                    const RealT bz = stokes_local_b_q1<RealT>(g, b, a, 2);
                    const RealT pp = x[3 * nn + colNode];

                    // Weak momentum pressure coupling: -(p, div v).
                    yu[0] -= bx * pp;
                    yu[1] -= by * pp;
                    yu[2] -= bz * pp;
                }
            }

            atomicAdd(&y[rowNode], yu[0]);
            atomicAdd(&y[nn + rowNode], yu[1]);
            atomicAdd(&y[2 * nn + rowNode], yu[2]);
        }

        if (rowNode != pPin) {
            RealT yp = RealT(0);

            for (int b = 0; b < 8; ++b) {
                const int colNode = stokes_local_node_id(g, ex, ey, ez, b);
                const int bi = colNode % g.nx;
                const int btmp = colNode / g.nx;
                const int bj = btmp % g.ny;
                const int bk = btmp / g.ny;
                const bool colBoundary = g_boundary_local(g, bi, bj, bk);

                if (!colBoundary) {
                    const RealT bx = stokes_local_b_q1<RealT>(g, a, b, 0);
                    const RealT by = stokes_local_b_q1<RealT>(g, a, b, 1);
                    const RealT bz = stokes_local_b_q1<RealT>(g, a, b, 2);

                    yp += bx * x[colNode];
                    yp += by * x[nn + colNode];
                    yp += bz * x[2 * nn + colNode];
                }

                if (colNode != pPin) {
                    const RealT ke = stokes_local_ke_q1<RealT>(g, a, b);
                    yp += tau * ke * x[3 * nn + colNode];
                }
            }

            atomicAdd(&y[3 * nn + rowNode], yp);
        }
    }
}

template <class RealT>
__global__ void kernel_apply_stokes_strong_rows(Grid3DCudaLocal g,
                                                const RealT* x,
                                                RealT* y) {
    const int nn = g.n_nodes_host();
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= nn) return;

    const int i = q % g.nx;
    const int tmp = q / g.nx;
    const int j = tmp % g.ny;
    const int k = tmp / g.ny;

    if (g_boundary_local(g, i, j, k)) {
        y[q] = x[q];
        y[nn + q] = x[nn + q];
        y[2 * nn + q] = x[2 * nn + q];
    }

    if (q == 0) {
        y[3 * nn + q] = x[3 * nn + q];
    }
}

template <class RealT>
__global__ void kernel_assemble_stokes_rhs_elements(Grid3DCudaLocal g,
                                                    RealT nu,
                                                    RealT tau,
                                                    RealT pGradSign,
                                                    RealT pspgRhsSign,
                                                    int pspgForceMode,
                                                    RealT* rhs) {
    const int nn = g.n_nodes_host();
    const int exN = g.nx - 1;
    const int eyN = g.ny - 1;
    const int ezN = g.nz - 1;
    const int nElem = exN * eyN * ezN;

    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nElem) return;

    const int ex = e % exN;
    const int tmp = e / exN;
    const int ey = tmp % eyN;
    const int ez = tmp / eyN;

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

    const double hx = g.hx;
    const double hy = g.hy;
    const double hz = g.hz;
    const double jac = hx * hy * hz / 8.0;

    for (int iq = 0; iq < 4; ++iq) {
        const double xi = qp[iq];
        const double wx = qw[iq];
        const double Nx[2] = {0.5 * (1.0 - xi), 0.5 * (1.0 + xi)};
        const double dNx[2] = {-1.0 / hx, 1.0 / hx};
        const double xx = (double(ex) + 0.5 * (1.0 + xi)) * hx;

        for (int jq = 0; jq < 4; ++jq) {
            const double eta = qp[jq];
            const double wy = qw[jq];
            const double Ny[2] = {0.5 * (1.0 - eta), 0.5 * (1.0 + eta)};
            const double dNy[2] = {-1.0 / hy, 1.0 / hy};
            const double yy = (double(ey) + 0.5 * (1.0 + eta)) * hy;

            for (int kq = 0; kq < 4; ++kq) {
                const double zeta = qp[kq];
                const double wz = qw[kq];
                const double Nz[2] = {0.5 * (1.0 - zeta), 0.5 * (1.0 + zeta)};
                const double dNz[2] = {-1.0 / hz, 1.0 / hz};
                const double zz = (double(ez) + 0.5 * (1.0 + zeta)) * hz;

                double fx, fy, fz;
                stokes_mms_force_device(xx, yy, zz, double(nu), double(pGradSign), fx, fy, fz);

                double gpx, gpy, gpz;
                stokes_mms_gradp_device(xx, yy, zz, double(pGradSign), gpx, gpy, gpz);

                double pfx = fx;
                double pfy = fy;
                double pfz = fz;

                if (pspgForceMode == 1) {
                    // Discrete Q1 PSPG pressure-row test:
                    // use only tau*(grad q, grad p_exact).
                    pfx = gpx;
                    pfy = gpy;
                    pfz = gpz;
                } else if (pspgForceMode == 2) {
                    pfx = 0.0;
                    pfy = 0.0;
                    pfz = 0.0;
                }

                const double wq = wx * wy * wz * jac;

                for (int a = 0; a < 8; ++a) {
                    const int ax = a & 1;
                    const int ay = (a >> 1) & 1;
                    const int az = (a >> 2) & 1;

                    const int rowNode = stokes_local_node_id(g, ex, ey, ez, a);
                    const int ri = rowNode % g.nx;
                    const int rtmp = rowNode / g.nx;
                    const int rj = rtmp % g.ny;
                    const int rk = rtmp / g.ny;
                    const bool rowBoundary = g_boundary_local(g, ri, rj, rk);

                    const double Na = Nx[ax] * Ny[ay] * Nz[az];
                    const double dNadx = dNx[ax] * Ny[ay] * Nz[az];
                    const double dNady = Nx[ax] * dNy[ay] * Nz[az];
                    const double dNadz = Nx[ax] * Ny[ay] * dNz[az];

                    if (!rowBoundary) {
                        atomicAdd(&rhs[rowNode], RealT(Na * fx * wq));
                        atomicAdd(&rhs[nn + rowNode], RealT(Na * fy * wq));
                        atomicAdd(&rhs[2 * nn + rowNode], RealT(Na * fz * wq));
                    }

                    if (rowNode != 0) {
                        // Q1 has zero elementwise second derivatives, so PSPG contributes
                        // tau * (grad q, grad p - f) = 0 as tau*(grad q, grad p) on the
                        // left and tau*(grad q, f) on the right for the Stokes MMS.
                        const double pspg = double(pspgRhsSign) * double(tau) *
                            (dNadx * pfx + dNady * pfy + dNadz * pfz) * wq;
                        atomicAdd(&rhs[3 * nn + rowNode], RealT(pspg));
                    }
                }
            }
        }
    }
}

template <class RealT>
__global__ void kernel_apply_stokes_rhs_strong_rows(Grid3DCudaLocal g, RealT* rhs) {
    const int nn = g.n_nodes_host();
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= nn) return;

    const int i = q % g.nx;
    const int tmp = q / g.nx;
    const int j = tmp % g.ny;
    const int k = tmp / g.ny;

    if (g_boundary_local(g, i, j, k)) {
        rhs[q] = RealT(0);
        rhs[nn + q] = RealT(0);
        rhs[2 * nn + q] = RealT(0);
    }

    if (q == 0) rhs[3 * nn + q] = RealT(0);
}


template <class RealT>
__global__ void kernel_stokes_diag_prec(Grid3DCudaLocal g,
                                        RealT nu,
                                        RealT tau,
                                        const RealT* r,
                                        RealT* z) {
    const int nn = g.n_nodes_host();
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= nn) return;

    const int i = q % g.nx;
    const int tmp = q / g.nx;
    const int j = tmp % g.ny;
    const int k = tmp / g.ny;

    const bool bnd = g_boundary_local(g, i, j, k);
    constexpr int pPin = 0;

    RealT kdiag = RealT(0);
    RealT B[3] = {RealT(0), RealT(0), RealT(0)};
    RealT G[3] = {RealT(0), RealT(0), RealT(0)};

    const int ex0 = max(0, i - 1);
    const int ex1 = min(g.nx - 2, i);
    const int ey0 = max(0, j - 1);
    const int ey1 = min(g.ny - 2, j);
    const int ez0 = max(0, k - 1);
    const int ez1 = min(g.nz - 2, k);

    for (int ez = ez0; ez <= ez1; ++ez) {
        for (int ey = ey0; ey <= ey1; ++ey) {
            for (int ex = ex0; ex <= ex1; ++ex) {
                const int lx = i - ex;
                const int ly = j - ey;
                const int lz = k - ez;
                const int a = lx + 2 * ly + 4 * lz;

                kdiag += stokes_local_ke_q1<RealT>(g, a, a);

                if (!bnd && q != pPin) {
                    const RealT bx = stokes_local_b_q1<RealT>(g, a, a, 0);
                    const RealT by = stokes_local_b_q1<RealT>(g, a, a, 1);
                    const RealT bz = stokes_local_b_q1<RealT>(g, a, a, 2);

                    B[0] += bx;
                    B[1] += by;
                    B[2] += bz;

                    // Momentum pressure block uses -(p, div v).
                    G[0] -= bx;
                    G[1] -= by;
                    G[2] -= bz;
                }
            }
        }
    }

    const RealT dU = bnd ? RealT(1) : max(nu * kdiag, RealT(1e-30));
    const RealT dP = (q == pPin) ? RealT(1) : max(tau * kdiag, RealT(1e-30));

    if (bnd || q == pPin) {
        if (bnd) {
            z[q] = r[q];
            z[nn + q] = r[nn + q];
            z[2 * nn + q] = r[2 * nn + q];
        } else {
            z[q] = r[q] / dU;
            z[nn + q] = r[nn + q] / dU;
            z[2 * nn + q] = r[2 * nn + q] / dU;
        }

        if (q == pPin) z[3 * nn + q] = r[3 * nn + q];
        else z[3 * nn + q] = r[3 * nn + q] / dP;
        return;
    }

    // Local nodal 4x4 block:
    //
    // [dU 0  0  Gx] [zu]   [ru]
    // [0 dU  0  Gy] [zv] = [rv]
    // [0  0 dU  Gz] [zw]   [rw]
    // [Bx By Bz dP] [zp]   [rp]
    const RealT ru0 = r[q];
    const RealT ru1 = r[nn + q];
    const RealT ru2 = r[2 * nn + q];
    const RealT rp = r[3 * nn + q];

    const RealT invDU = RealT(1) / dU;

    RealT schur = dP - (B[0] * G[0] + B[1] * G[1] + B[2] * G[2]) * invDU;
    const RealT absSchur = schur >= RealT(0) ? schur : -schur;
    if (!(absSchur > RealT(1e-30))) {
        schur = (schur >= RealT(0) ? RealT(1e-30) : RealT(-1e-30));
    }

    const RealT rhsP = rp - (B[0] * ru0 + B[1] * ru1 + B[2] * ru2) * invDU;
    const RealT zp = rhsP / schur;

    z[q] = (ru0 - G[0] * zp) * invDU;
    z[nn + q] = (ru1 - G[1] * zp) * invDU;
    z[2 * nn + q] = (ru2 - G[2] * zp) * invDU;
    z[3 * nn + q] = zp;
}

template <class RealT>
void stokes_print_nodal4_diag_audit(const memoirs::structured::StructuredGrid3D& grid,
                                    Grid3DCudaLocal g,
                                    RealT nu,
                                    RealT tau,
                                    std::ostream& os) {
    const int nn = grid.n_nodes();
    constexpr int pPin = 0;

    double maxAbsB = 0.0;
    double maxAbsG = 0.0;
    double maxAbsBG = 0.0;
    double minAbsSchur = std::numeric_limits<double>::infinity();
    double maxAbsSchur = 0.0;
    int coupledNodes = 0;
    int freeNodes = 0;

    for (int q = 0; q < nn; ++q) {
        const int i = q % g.nx;
        const int tmp = q / g.nx;
        const int j = tmp % g.ny;
        const int k = tmp / g.ny;

        const bool bnd = g_boundary_local(g, i, j, k);

        double kdiag = 0.0;
        double B[3] = {0.0, 0.0, 0.0};
        double Gv[3] = {0.0, 0.0, 0.0};

        const int ex0 = std::max(0, i - 1);
        const int ex1 = std::min(g.nx - 2, i);
        const int ey0 = std::max(0, j - 1);
        const int ey1 = std::min(g.ny - 2, j);
        const int ez0 = std::max(0, k - 1);
        const int ez1 = std::min(g.nz - 2, k);

        for (int ez = ez0; ez <= ez1; ++ez) {
            for (int ey = ey0; ey <= ey1; ++ey) {
                for (int ex = ex0; ex <= ex1; ++ex) {
                    const int lx = i - ex;
                    const int ly = j - ey;
                    const int lz = k - ez;
                    const int a = lx + 2 * ly + 4 * lz;

                    kdiag += double(stokes_local_ke_q1<RealT>(g, a, a));

                    if (!bnd && q != pPin) {
                        const double bx = double(stokes_local_b_q1<RealT>(g, a, a, 0));
                        const double by = double(stokes_local_b_q1<RealT>(g, a, a, 1));
                        const double bz = double(stokes_local_b_q1<RealT>(g, a, a, 2));

                        B[0] += bx;
                        B[1] += by;
                        B[2] += bz;

                        Gv[0] -= bx;
                        Gv[1] -= by;
                        Gv[2] -= bz;
                    }
                }
            }
        }

        if (!bnd && q != pPin) {
            ++freeNodes;

            const double dU = std::max(double(nu) * kdiag, 1e-300);
            const double dP = std::max(double(tau) * kdiag, 1e-300);
            const double Bnorm = std::sqrt(B[0]*B[0] + B[1]*B[1] + B[2]*B[2]);
            const double Gnorm = std::sqrt(Gv[0]*Gv[0] + Gv[1]*Gv[1] + Gv[2]*Gv[2]);
            const double BG = B[0]*Gv[0] + B[1]*Gv[1] + B[2]*Gv[2];
            const double schur = dP - BG / dU;

            maxAbsB = std::max(maxAbsB, Bnorm);
            maxAbsG = std::max(maxAbsG, Gnorm);
            maxAbsBG = std::max(maxAbsBG, std::abs(BG));
            minAbsSchur = std::min(minAbsSchur, std::abs(schur));
            maxAbsSchur = std::max(maxAbsSchur, std::abs(schur));

            if (Bnorm > 1e-30 || Gnorm > 1e-30) ++coupledNodes;
        }
    }

    if (!std::isfinite(minAbsSchur)) minAbsSchur = 0.0;

    os << "--------------- stokes nodal preconditioner audit ---------------\n";
    os << "preconditionerType        = nodal_4x4_inverse_diagonal\n";
    os << std::setprecision(17);
    os << "freeInteriorPressureNodes = " << freeNodes << "\n";
    os << "sameNodeCoupledNodes      = " << coupledNodes << "\n";
    os << "maxSameNodeBNorm          = " << maxAbsB << "\n";
    os << "maxSameNodeGNorm          = " << maxAbsG << "\n";
    os << "maxAbsBdotG               = " << maxAbsBG << "\n";
    os << "minAbsLocalSchur          = " << minAbsSchur << "\n";
    os << "maxAbsLocalSchur          = " << maxAbsSchur << "\n";
    if (coupledNodes == 0) {
        os << "auditNote                 = same-node Q1 grad/div coupling cancels; this block is effectively scalar diagonal on interiors\n";
    }
    os << "------------------------------------------------------------------\n";
}


template <class RealT>
RealT stokes_compute_tau_host(const memoirs::structured::StructuredGrid3D& grid,
                              RealT nu,
                              RealT tauScale,
                              const std::string& tauMode,
                              RealT tauC,
                              RealT tauDt,
                              RealT tauAdvMag) {
    const RealT hx = RealT(grid.hx());
    const RealT hy = RealT(grid.hy());
    const RealT hz = RealT(grid.hz());

    const RealT hmin = std::min({hx, hy, hz});
    const RealT volume = hx * hy * hz;
    const RealT pi = RealT(3.141592653589793238462643383279502884);
    const RealT hEq = std::pow(RealT(6) * volume / pi, RealT(1.0 / 3.0));

    if (tauMode == "simple") {
        return tauScale * hmin * hmin / nu;
    }

    if (tauMode == "sphere" || tauMode == "sphereC") {
        return tauScale * hEq * hEq / (tauC * nu);
    }

    if (tauMode == "metric") {
        // Tezduyar/Shakib-style steady diffusive metric form.
        //
        // 2D MATLAB analogue:
        // tau = 1 / sqrt( (CI*mu)^2 * [ (2/hx)^4 + (2/hy)^4 ] )
        //
        // 3D structured hex analogue:
        // tau = tauScale / sqrt( (tauC*nu)^2 *
        //                         [ (2/hx)^4 + (2/hy)^4 + (2/hz)^4 ] )
        const RealT gx = RealT(2) / hx;
        const RealT gy = RealT(2) / hy;
        const RealT gz = RealT(2) / hz;
        const RealT metric4 = gx*gx*gx*gx + gy*gy*gy*gy + gz*gz*gz*gz;
        return tauScale / ((tauC * nu) * std::sqrt(metric4));
    }

    if (tauMode == "tezduyar") {
        RealT invTau2 = RealT(0);

        if (tauDt > RealT(0)) {
            const RealT t = RealT(2) / tauDt;
            invTau2 += t * t;
        }

        if (tauAdvMag > RealT(0)) {
            const RealT a = tauAdvMag / hEq;
            invTau2 += a * a;
        }

        const RealT d = tauC * nu / (hEq * hEq);
        invTau2 += d * d;

        return tauScale / std::sqrt(invTau2);
    }

    return tauScale * hmin * hmin / nu;
}

template <class RealT>
RealT stokes_hmin_host(const memoirs::structured::StructuredGrid3D& grid) {
    return RealT(std::min({grid.hx(), grid.hy(), grid.hz()}));
}

template <class RealT>
RealT stokes_heq_host(const memoirs::structured::StructuredGrid3D& grid) {
    const RealT hx = RealT(grid.hx());
    const RealT hy = RealT(grid.hy());
    const RealT hz = RealT(grid.hz());
    const RealT volume = hx * hy * hz;
    const RealT pi = RealT(3.141592653589793238462643383279502884);
    return std::pow(RealT(6) * volume / pi, RealT(1.0 / 3.0));
}

template <class RealT>
class StructuredStokesPspgOperator {
public:
    StructuredStokesPspgOperator(const memoirs::structured::StructuredGrid3D& grid,
                                 RealT nu,
                                 RealT tauScale,
                                 const std::string& tauMode,
                                 RealT tauC,
                                 RealT tauDt,
                                 RealT tauAdvMag,
                                 RealT pGradSign,
                                 RealT pspgRhsSign,
                                 int pspgForceMode)
        : grid_(grid),
          dg_(to_cuda_grid_local(grid)),
          nu_(nu),
          tauScale_(tauScale),
          tauMode_(tauMode),
          tauC_(tauC),
          tauDt_(tauDt),
          tauAdvMag_(tauAdvMag),
          pGradSign_(pGradSign),
          pspgRhsSign_(pspgRhsSign),
          pspgForceMode_(pspgForceMode) {
        hMin_ = stokes_hmin_host<RealT>(grid_);
        hEq_ = stokes_heq_host<RealT>(grid_);
        tau_ = stokes_compute_tau_host<RealT>(
            grid_, nu_, tauScale_, tauMode_, tauC_, tauDt_, tauAdvMag_);
    }

    int size() const { return 4 * grid_.n_nodes(); }
    int scalar_size() const { return grid_.n_nodes(); }
    RealT tau() const { return tau_; }

    const memoirs::structured::StructuredGrid3D& grid() const { return grid_; }

    void apply(const RealT* x, RealT* y) const {
        const int nn = grid_.n_nodes();
        const int total = 4 * nn;
        const int nElem = (grid_.nx - 1) * (grid_.ny - 1) * (grid_.nz - 1);
        const int block = 256;

        kernel_zero_local<RealT><<<div_up_local(total, block), block>>>(total, y);
        MEMOIRS_CUDA_KERNEL_CHECK();

        kernel_apply_stokes_pspg_elements<RealT>
            <<<div_up_local(nElem, block), block>>>(dg_, nu_, tau_, x, y);
        MEMOIRS_CUDA_KERNEL_CHECK();

        kernel_apply_stokes_strong_rows<RealT>
            <<<div_up_local(nn, block), block>>>(dg_, x, y);
        MEMOIRS_CUDA_KERNEL_CHECK();
    }

    void assemble_rhs_continuous(RealT* rhs) const {
        const int nn = grid_.n_nodes();
        const int total = 4 * nn;
        const int nElem = (grid_.nx - 1) * (grid_.ny - 1) * (grid_.nz - 1);
        const int block = 256;

        kernel_zero_local<RealT><<<div_up_local(total, block), block>>>(total, rhs);
        MEMOIRS_CUDA_KERNEL_CHECK();

        kernel_assemble_stokes_rhs_elements<RealT>
            <<<div_up_local(nElem, block), block>>>(dg_, nu_, tau_, pGradSign_, pspgRhsSign_, pspgForceMode_, rhs);
        MEMOIRS_CUDA_KERNEL_CHECK();

        kernel_apply_stokes_rhs_strong_rows<RealT>
            <<<div_up_local(nn, block), block>>>(dg_, rhs);
        MEMOIRS_CUDA_KERNEL_CHECK();
    }

    void fill_exact(RealT* x) const {
        const int nn = grid_.n_nodes();
        const int block = 256;
        kernel_fill_stokes_exact<RealT><<<div_up_local(nn, block), block>>>(dg_, x);
        MEMOIRS_CUDA_KERNEL_CHECK();
    }

private:
    memoirs::structured::StructuredGrid3D grid_;
    Grid3DCudaLocal dg_;
    RealT nu_;
    RealT tau_;
    RealT tauScale_;
    std::string tauMode_;
    RealT tauC_;
    RealT tauDt_;
    RealT tauAdvMag_;
    RealT hMin_;
    RealT hEq_;
    RealT pGradSign_;
    RealT pspgRhsSign_;
    int pspgForceMode_;
};

template <class RealT>
class StructuredStokesDiagonalPreconditioner {
public:
    StructuredStokesDiagonalPreconditioner(const memoirs::structured::StructuredGrid3D& grid,
                                           RealT nu,
                                           RealT tau)
        : grid_(grid), dg_(to_cuda_grid_local(grid)), nu_(nu), tau_(tau) {}

    void apply(const RealT* r, RealT* z) {
        const int n = grid_.n_nodes();
        const int block = 256;
        kernel_stokes_diag_prec<RealT>
            <<<div_up_local(n, block), block>>>(dg_, nu_, tau_, r, z);
        MEMOIRS_CUDA_KERNEL_CHECK();
    }

    void print_audit(std::ostream& os) const {
        stokes_print_nodal4_diag_audit<RealT>(grid_, dg_, nu_, tau_, os);
    }

private:
    memoirs::structured::StructuredGrid3D grid_;
    Grid3DCudaLocal dg_;
    RealT nu_;
    RealT tau_;
    RealT tauScale_;
    std::string tauMode_;
    RealT tauC_;
    RealT tauDt_;
    RealT tauAdvMag_;
    RealT hMin_;
    RealT hEq_;
    RealT pGradSign_;
    RealT pspgRhsSign_;
    int pspgForceMode_;
};

template <class RealT>
struct CudaRightBicgstabReportLocal {
    int iterations = 0;
    int converged = 0;
    int breakdown = 0;
    int breakdownCode = 0;
    double breakdownValue = 0.0;
    double initialResidual = 0.0;
    double finalResidual = 0.0;
    double finalRelativeResidual = 0.0;
};

template <class RealT, class Operator, class RightPreconditioner>
CudaRightBicgstabReportLocal<RealT> bicgstab_right_prec_cuda_local(
    const Operator& A,
    RightPreconditioner& M,
    const RealT* d_b,
    RealT* d_x,
    RealT tol,
    int maxit,
    int printEvery) {

    const int n = A.size();
    const int block = 256;

    DeviceBuffer<RealT> d_Ax(static_cast<std::size_t>(n));
    DeviceBuffer<RealT> d_r(static_cast<std::size_t>(n));
    DeviceBuffer<RealT> d_rhat(static_cast<std::size_t>(n));
    DeviceBuffer<RealT> d_p(static_cast<std::size_t>(n));
    DeviceBuffer<RealT> d_v(static_cast<std::size_t>(n));
    DeviceBuffer<RealT> d_s(static_cast<std::size_t>(n));
    DeviceBuffer<RealT> d_t(static_cast<std::size_t>(n));
    DeviceBuffer<RealT> d_phat(static_cast<std::size_t>(n));
    DeviceBuffer<RealT> d_shat(static_cast<std::size_t>(n));

    GpuDotLocal<RealT> dots;

    A.apply(d_x, d_Ax.data());

    kernel_subtract_local<RealT><<<div_up_local(n, block), block>>>
        (n, d_b, d_Ax.data(), d_r.data());
    MEMOIRS_CUDA_KERNEL_CHECK();

    kernel_copy_local<RealT><<<div_up_local(n, block), block>>>
        (n, d_r.data(), d_rhat.data());
    MEMOIRS_CUDA_KERNEL_CHECK();

    d_p.zero();
    d_v.zero();

    const double bnorm = std::max(std::sqrt(dots.dot(n, d_b, d_b)), 1.0);
    double rnorm = std::sqrt(dots.dot(n, d_r.data(), d_r.data()));

    CudaRightBicgstabReportLocal<RealT> rep;
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
            rep.breakdownCode = 1; // rhoNew = rhat dot r
            rep.breakdownValue = rhoNew;
            rep.iterations = it - 1;
            return rep;
        }

        const double beta = (rhoNew / rhoOld) * (alpha / omega);

        kernel_combination3_local<RealT><<<div_up_local(n, block), block>>>
            (n, RealT(1), d_r.data(), RealT(beta), d_p.data(),
             RealT(-beta * omega), d_v.data(), d_p.data());
        MEMOIRS_CUDA_KERNEL_CHECK();

        M.apply(d_p.data(), d_phat.data());
        A.apply(d_phat.data(), d_v.data());

        const double rhatv = dots.dot(n, d_rhat.data(), d_v.data());
        if (!(std::abs(rhatv) > 0.0)) {
            rep.breakdown = 1;
            rep.breakdownCode = 2; // rhat dot v
            rep.breakdownValue = rhatv;
            rep.iterations = it - 1;
            return rep;
        }

        alpha = rhoNew / rhatv;

        kernel_combination3_local<RealT><<<div_up_local(n, block), block>>>
            (n, RealT(1), d_r.data(), RealT(-alpha), d_v.data(),
             RealT(0), d_v.data(), d_s.data());
        MEMOIRS_CUDA_KERNEL_CHECK();

        const double snorm = std::sqrt(dots.dot(n, d_s.data(), d_s.data()));
        if (snorm / bnorm <= double(tol)) {
            kernel_axpby_local<RealT><<<div_up_local(n, block), block>>>
                (n, RealT(alpha), d_phat.data(), RealT(1), d_x);
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
            rep.breakdownCode = 3; // t dot t
            rep.breakdownValue = tt;
            rep.iterations = it - 1;
            return rep;
        }

        omega = dots.dot(n, d_t.data(), d_s.data()) / tt;
        if (!(std::abs(omega) > 0.0)) {
            rep.breakdown = 1;
            rep.breakdownCode = 4; // omega = (t dot s)/(t dot t)
            rep.breakdownValue = omega;
            rep.iterations = it - 1;
            return rep;
        }

        kernel_combination3_local<RealT><<<div_up_local(n, block), block>>>
            (n, RealT(1), d_x, RealT(alpha), d_phat.data(),
             RealT(omega), d_shat.data(), d_x);
        MEMOIRS_CUDA_KERNEL_CHECK();

        kernel_combination3_local<RealT><<<div_up_local(n, block), block>>>
            (n, RealT(1), d_s.data(), RealT(-omega), d_t.data(),
             RealT(0), d_t.data(), d_r.data());
        MEMOIRS_CUDA_KERNEL_CHECK();

        rnorm = std::sqrt(dots.dot(n, d_r.data(), d_r.data()));

        rep.iterations = it;
        rep.finalResidual = rnorm;
        rep.finalRelativeResidual = rnorm / bnorm;

        if (printEvery > 0 && (it == 1 || it % printEvery == 0)) {
            std::cout << "cuda stokes right-bicgstab it=" << it
                      << " rel=" << rep.finalRelativeResidual << "\n";
        }

        if (rep.finalRelativeResidual <= double(tol)) {
            rep.converged = 1;
            return rep;
        }

        rhoOld = rhoNew;
    }

    return rep;
}


template <class RealT, class Operator, class RightPreconditioner>
CudaRightBicgstabReportLocal<RealT> gmres_right_prec_cuda_local(
    const Operator& A,
    RightPreconditioner& M,
    const RealT* d_b,
    RealT* d_x,
    RealT tol,
    int maxit,
    int restart,
    int printEvery) {

    const int n = A.size();
    const int block = 256;

    DeviceBuffer<RealT> d_Ax(static_cast<std::size_t>(n));
    DeviceBuffer<RealT> d_r(static_cast<std::size_t>(n));
    DeviceBuffer<RealT> d_w(static_cast<std::size_t>(n));

    GpuDotLocal<RealT> dots;

    CudaRightBicgstabReportLocal<RealT> rep;

    const double bnorm = std::max(std::sqrt(dots.dot(n, d_b, d_b)), 1.0);

    A.apply(d_x, d_Ax.data());
    kernel_subtract_local<RealT><<<div_up_local(n, block), block>>>
        (n, d_b, d_Ax.data(), d_r.data());
    MEMOIRS_CUDA_KERNEL_CHECK();

    double beta = std::sqrt(dots.dot(n, d_r.data(), d_r.data()));
    rep.initialResidual = beta;
    rep.finalResidual = beta;
    rep.finalRelativeResidual = beta / bnorm;

    if (rep.finalRelativeResidual <= double(tol)) {
        rep.converged = 1;
        return rep;
    }

    restart = std::max(2, restart);

    std::vector<DeviceBuffer<RealT>> V;
    std::vector<DeviceBuffer<RealT>> Z;
    V.reserve(static_cast<std::size_t>(restart + 1));
    Z.reserve(static_cast<std::size_t>(restart));

    for (int i = 0; i < restart + 1; ++i) {
        V.emplace_back(static_cast<std::size_t>(n));
    }
    for (int i = 0; i < restart; ++i) {
        Z.emplace_back(static_cast<std::size_t>(n));
    }

    std::vector<std::vector<double>> H(
        static_cast<std::size_t>(restart + 1),
        std::vector<double>(static_cast<std::size_t>(restart), 0.0));
    std::vector<double> cs(static_cast<std::size_t>(restart), 0.0);
    std::vector<double> sn(static_cast<std::size_t>(restart), 0.0);
    std::vector<double> g(static_cast<std::size_t>(restart + 1), 0.0);
    std::vector<double> y(static_cast<std::size_t>(restart), 0.0);

    int totalIt = 0;

    while (totalIt < maxit) {
        A.apply(d_x, d_Ax.data());
        kernel_subtract_local<RealT><<<div_up_local(n, block), block>>>
            (n, d_b, d_Ax.data(), d_r.data());
        MEMOIRS_CUDA_KERNEL_CHECK();

        beta = std::sqrt(dots.dot(n, d_r.data(), d_r.data()));

        rep.finalResidual = beta;
        rep.finalRelativeResidual = beta / bnorm;

        if (rep.finalRelativeResidual <= double(tol)) {
            rep.converged = 1;
            return rep;
        }

        for (int i = 0; i < restart + 1; ++i) g[static_cast<std::size_t>(i)] = 0.0;
        g[0] = beta;

        // V0 = r / beta
        kernel_axpby_local<RealT><<<div_up_local(n, block), block>>>
            (n, RealT(1.0 / beta), d_r.data(), RealT(0), V[0].data());
        MEMOIRS_CUDA_KERNEL_CHECK();

        int lastJ = -1;

        for (int j = 0; j < restart && totalIt < maxit; ++j) {
            lastJ = j;

            // Right preconditioning: z_j = M^{-1} v_j
            M.apply(V[static_cast<std::size_t>(j)].data(), Z[static_cast<std::size_t>(j)].data());

            // w = A z_j
            A.apply(Z[static_cast<std::size_t>(j)].data(), d_w.data());

            // Modified Gram-Schmidt
            for (int i = 0; i <= j; ++i) {
                H[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                    dots.dot(n, V[static_cast<std::size_t>(i)].data(), d_w.data());

                kernel_axpby_local<RealT><<<div_up_local(n, block), block>>>
                    (n,
                     RealT(-H[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)]),
                     V[static_cast<std::size_t>(i)].data(),
                     RealT(1),
                     d_w.data());
                MEMOIRS_CUDA_KERNEL_CHECK();
            }

            H[static_cast<std::size_t>(j + 1)][static_cast<std::size_t>(j)] =
                std::sqrt(dots.dot(n, d_w.data(), d_w.data()));

            const double hnext = H[static_cast<std::size_t>(j + 1)][static_cast<std::size_t>(j)];
            if (!(std::isfinite(hnext)) || hnext <= 0.0) {
                // Happy breakdown or numerical Arnoldi breakdown.
                // Continue to least-squares update with current subspace.
                H[static_cast<std::size_t>(j + 1)][static_cast<std::size_t>(j)] = 0.0;
            } else {
                kernel_axpby_local<RealT><<<div_up_local(n, block), block>>>
                    (n, RealT(1.0 / hnext), d_w.data(), RealT(0),
                     V[static_cast<std::size_t>(j + 1)].data());
                MEMOIRS_CUDA_KERNEL_CHECK();
            }

            // Apply previous Givens rotations.
            for (int i = 0; i < j; ++i) {
                const double hij  = H[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
                const double hi1j = H[static_cast<std::size_t>(i + 1)][static_cast<std::size_t>(j)];
                H[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                    cs[static_cast<std::size_t>(i)] * hij +
                    sn[static_cast<std::size_t>(i)] * hi1j;
                H[static_cast<std::size_t>(i + 1)][static_cast<std::size_t>(j)] =
                    -sn[static_cast<std::size_t>(i)] * hij +
                     cs[static_cast<std::size_t>(i)] * hi1j;
            }

            // New Givens rotation.
            const double h0 = H[static_cast<std::size_t>(j)][static_cast<std::size_t>(j)];
            const double h1 = H[static_cast<std::size_t>(j + 1)][static_cast<std::size_t>(j)];
            const double denom = std::hypot(h0, h1);

            if (!(std::isfinite(denom)) || denom == 0.0) {
                rep.breakdown = 1;
                rep.breakdownCode = 10; // GMRES zero Hessenberg diagonal
                rep.breakdownValue = denom;
                rep.iterations = totalIt;
                return rep;
            }

            cs[static_cast<std::size_t>(j)] = h0 / denom;
            sn[static_cast<std::size_t>(j)] = h1 / denom;

            H[static_cast<std::size_t>(j)][static_cast<std::size_t>(j)] =
                cs[static_cast<std::size_t>(j)] * h0 +
                sn[static_cast<std::size_t>(j)] * h1;
            H[static_cast<std::size_t>(j + 1)][static_cast<std::size_t>(j)] = 0.0;

            const double gj  = g[static_cast<std::size_t>(j)];
            const double gj1 = g[static_cast<std::size_t>(j + 1)];
            g[static_cast<std::size_t>(j)] =
                cs[static_cast<std::size_t>(j)] * gj +
                sn[static_cast<std::size_t>(j)] * gj1;
            g[static_cast<std::size_t>(j + 1)] =
                -sn[static_cast<std::size_t>(j)] * gj +
                 cs[static_cast<std::size_t>(j)] * gj1;

            ++totalIt;

            rep.iterations = totalIt;
            rep.finalResidual = std::abs(g[static_cast<std::size_t>(j + 1)]);
            rep.finalRelativeResidual = rep.finalResidual / bnorm;

            if (printEvery > 0 && (totalIt == 1 || totalIt % printEvery == 0)) {
                std::cout << "cuda stokes right-gmres it=" << totalIt
                          << " rel=" << rep.finalRelativeResidual << "\n";
            }

            if (rep.finalRelativeResidual <= double(tol)) {
                break;
            }
        }

        const int k = lastJ;
        if (k < 0) break;

        // Solve upper triangular system H(0:k,0:k) y = g(0:k).
        for (int i = 0; i <= k; ++i) {
            y[static_cast<std::size_t>(i)] = g[static_cast<std::size_t>(i)];
        }

        for (int i = k; i >= 0; --i) {
            double sum = y[static_cast<std::size_t>(i)];
            for (int j = i + 1; j <= k; ++j) {
                sum -= H[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] *
                       y[static_cast<std::size_t>(j)];
            }

            const double diag = H[static_cast<std::size_t>(i)][static_cast<std::size_t>(i)];
            if (!(std::isfinite(diag)) || std::abs(diag) == 0.0) {
                rep.breakdown = 1;
                rep.breakdownCode = 11; // GMRES singular triangular solve
                rep.breakdownValue = diag;
                rep.iterations = totalIt;
                return rep;
            }

            y[static_cast<std::size_t>(i)] = sum / diag;
        }

        // x += Z*y
        for (int i = 0; i <= k; ++i) {
            kernel_axpby_local<RealT><<<div_up_local(n, block), block>>>
                (n, RealT(y[static_cast<std::size_t>(i)]),
                 Z[static_cast<std::size_t>(i)].data(),
                 RealT(1),
                 d_x);
            MEMOIRS_CUDA_KERNEL_CHECK();
        }

        // Check true residual after the restart/update.
        A.apply(d_x, d_Ax.data());
        kernel_subtract_local<RealT><<<div_up_local(n, block), block>>>
            (n, d_b, d_Ax.data(), d_r.data());
        MEMOIRS_CUDA_KERNEL_CHECK();

        const double rnorm = std::sqrt(dots.dot(n, d_r.data(), d_r.data()));
        rep.finalResidual = rnorm;
        rep.finalRelativeResidual = rnorm / bnorm;
        rep.iterations = totalIt;

        if (printEvery > 0) {
            std::cout << "cuda stokes right-gmres restart-end it=" << totalIt
                      << " rel=" << rep.finalRelativeResidual << "\n";
        }

        if (rep.finalRelativeResidual <= double(tol)) {
            rep.converged = 1;
            return rep;
        }
    }

    return rep;
}


template <class RealT, class Operator>
double device_residual_rel_local(const Operator& A, const RealT* d_b, const RealT* d_x) {
    const int n = A.size();
    const int block = 256;
    DeviceBuffer<RealT> d_Ax(static_cast<std::size_t>(n));
    DeviceBuffer<RealT> d_r(static_cast<std::size_t>(n));
    GpuDotLocal<RealT> dots;

    A.apply(d_x, d_Ax.data());
    kernel_subtract_local<RealT><<<div_up_local(n, block), block>>>
        (n, d_b, d_Ax.data(), d_r.data());
    MEMOIRS_CUDA_KERNEL_CHECK();

    const double r = std::sqrt(dots.dot(n, d_r.data(), d_r.data()));
    const double b = std::max(std::sqrt(dots.dot(n, d_b, d_b)), 1.0);
    return r / b;
}


template <class RealT, class Operator>
void print_exact_rhs_defect_local(const Operator& A,
                                  const RealT* d_b,
                                  const RealT* d_exact,
                                  int nn,
                                  std::ostream& os) {
    const int n = A.size();
    const int block = 256;

    DeviceBuffer<RealT> d_Ax(static_cast<std::size_t>(n));
    DeviceBuffer<RealT> d_def(static_cast<std::size_t>(n));
    GpuDotLocal<RealT> dots;

    A.apply(d_exact, d_Ax.data());

    // def = A*x_exact_interpolated - b_continuous
    kernel_subtract_local<RealT><<<div_up_local(n, block), block>>>
        (n, d_Ax.data(), d_b, d_def.data());
    MEMOIRS_CUDA_KERNEL_CHECK();

    const double bTotal = std::sqrt(dots.dot(n, d_b, d_b));
    const double dTotal = std::sqrt(dots.dot(n, d_def.data(), d_def.data()));
    const double denomTotal = std::max(bTotal, 1.0);

    const double bUx = std::sqrt(dots.dot(nn, d_b, d_b));
    const double bUy = std::sqrt(dots.dot(nn, d_b + nn, d_b + nn));
    const double bUz = std::sqrt(dots.dot(nn, d_b + 2 * nn, d_b + 2 * nn));
    const double bP  = std::sqrt(dots.dot(nn, d_b + 3 * nn, d_b + 3 * nn));

    const double dUx = std::sqrt(dots.dot(nn, d_def.data(), d_def.data()));
    const double dUy = std::sqrt(dots.dot(nn, d_def.data() + nn, d_def.data() + nn));
    const double dUz = std::sqrt(dots.dot(nn, d_def.data() + 2 * nn, d_def.data() + 2 * nn));
    const double dP  = std::sqrt(dots.dot(nn, d_def.data() + 3 * nn, d_def.data() + 3 * nn));

    os << "--------------- stokes exact-RHS consistency audit ---------------\n";
    os << std::setprecision(17);
    os << "exactDefectDefinition     = A*x_exact_interpolated_minus_b_continuous\n";
    os << "rhsNormTotal              = " << bTotal << "\n";
    os << "exactDefectNormTotal      = " << dTotal << "\n";
    os << "exactDefectRelTotal       = " << dTotal / denomTotal << "\n";

    os << "rhsNormUx                 = " << bUx << "\n";
    os << "rhsNormUy                 = " << bUy << "\n";
    os << "rhsNormUz                 = " << bUz << "\n";
    os << "rhsNormP                  = " << bP << "\n";

    os << "exactDefectNormUx         = " << dUx << "\n";
    os << "exactDefectNormUy         = " << dUy << "\n";
    os << "exactDefectNormUz         = " << dUz << "\n";
    os << "exactDefectNormP          = " << dP << "\n";

    os << "exactDefectRelUx          = " << dUx / std::max(bUx, 1.0) << "\n";
    os << "exactDefectRelUy          = " << dUy / std::max(bUy, 1.0) << "\n";
    os << "exactDefectRelUz          = " << dUz / std::max(bUz, 1.0) << "\n";
    os << "exactDefectRelP           = " << dP / std::max(bP, 1.0) << "\n";
    os << "exactDefectRelUx_byBlockRhs = " << dUx / std::max(bUx, 1.0e-300) << "\n";
    os << "exactDefectRelUy_byBlockRhs = " << dUy / std::max(bUy, 1.0e-300) << "\n";
    os << "exactDefectRelUz_byBlockRhs = " << dUz / std::max(bUz, 1.0e-300) << "\n";
    os << "exactDefectRelP_byBlockRhs  = " << dP / std::max(bP, 1.0e-300) << "\n";
    os << "------------------------------------------------------------------\n";
}


} // namespace gpu
} // namespace memoirs



namespace memoirs { namespace gpu {

__device__ inline void q1q1_mms_grad_velocity_device(double x,
                                                     double y,
                                                     double z,
                                                     double gu[3],
                                                     double gv[3],
                                                     double gw[3]) {
    const double pi = 3.141592653589793238462643383279502884;
    const double sx = sin(pi*x), cx = cos(pi*x);
    const double sy = sin(pi*y), cy = cos(pi*y);
    const double sz = sin(pi*z), cz = cos(pi*z);

    gu[0] = 4.0*pi*pi*sx*cx*sy*cy*sz*sz;
    gu[1] = 2.0*pi*pi*sx*sx*(cy*cy - sy*sy)*sz*sz;
    gu[2] = 4.0*pi*pi*sx*sx*sy*cy*sz*cz;

    gv[0] = -2.0*pi*pi*(cx*cx - sx*sx)*sy*sy*sz*sz;
    gv[1] = -4.0*pi*pi*sx*cx*sy*cy*sz*sz;
    gv[2] = -4.0*pi*pi*sx*cx*sy*sy*sz*cz;

    gw[0] = 0.0;
    gw[1] = 0.0;
    gw[2] = 0.0;
}

__device__ inline void q1q1_nse_mms_force_device(double x,
                                                 double y,
                                                 double z,
                                                 double nu,
                                                 double advScale,
                                                 double& fx,
                                                 double& fy,
                                                 double& fz) {
    // Stationary MMS: f = advScale*(u.grad)u - nu*Delta u + grad p.
    stokes_mms_force_device(x, y, z, nu, 1.0, fx, fy, fz);
    double u, v, w, p;
    stokes_mms_exact_device(x, y, z, u, v, w, p);
    double gu[3], gv[3], gw[3];
    q1q1_mms_grad_velocity_device(x, y, z, gu, gv, gw);
    fx += advScale * (u*gu[0] + v*gu[1] + w*gu[2]);
    fy += advScale * (u*gv[0] + v*gv[1] + w*gv[2]);
    fz += advScale * (u*gw[0] + v*gw[1] + w*gw[2]);
}

template <class RealT>
__device__ inline void q1_shape_phys(const Grid3DCudaLocal g,
                                     double xi, double eta, double zeta,
                                     double N[8], double dX[8], double dY[8], double dZ[8]) {
    const double Nx[2] = {0.5*(1.0-xi), 0.5*(1.0+xi)};
    const double Ny[2] = {0.5*(1.0-eta), 0.5*(1.0+eta)};
    const double Nz[2] = {0.5*(1.0-zeta), 0.5*(1.0+zeta)};
    const double dNx[2] = {-1.0/g.hx, 1.0/g.hx};
    const double dNy[2] = {-1.0/g.hy, 1.0/g.hy};
    const double dNz[2] = {-1.0/g.hz, 1.0/g.hz};
    for (int az=0; az<2; ++az) for (int ay=0; ay<2; ++ay) for (int ax=0; ax<2; ++ax) {
        const int a = ax + 2*ay + 4*az;
        N[a]  = Nx[ax]*Ny[ay]*Nz[az];
        dX[a] = dNx[ax]*Ny[ay]*Nz[az];
        dY[a] = Nx[ax]*dNy[ay]*Nz[az];
        dZ[a] = Nx[ax]*Ny[ay]*dNz[az];
    }
}

template <class RealT>
__global__ void kernel_apply_q1q1_nse_elements(Grid3DCudaLocal g,
                                               RealT nu,
                                               RealT tau,
                                               RealT massCoeff,
                                               RealT advScale,
                                               int supgEnabled,
                                               RealT supgTauScale,
                                               const RealT* beta,
                                               const RealT* x,
                                               RealT* y) {
    const int nn = g.n_nodes_host();
    const int exN = g.nx - 1;
    const int eyN = g.ny - 1;
    const int ezN = g.nz - 1;
    const int nElem = exN * eyN * ezN;
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nElem) return;

    const int ex = e % exN;
    const int tmp = e / exN;
    const int ey = tmp % eyN;
    const int ez = tmp / eyN;
    constexpr int pPin = 0;

    const double qp[4] = {-0.86113631159405257522, -0.33998104358485626480,
                           0.33998104358485626480,  0.86113631159405257522};
    const double qw[4] = { 0.34785484513745385737,  0.65214515486254614263,
                           0.65214515486254614263,  0.34785484513745385737};
    const double jac = g.hx*g.hy*g.hz/8.0;

    int gid[8]; bool bnd[8];
    double ux[8], uy[8], uz[8], pp[8], bx[8], by[8], bz[8];
    for (int a=0; a<8; ++a) {
        const int id = stokes_local_node_id(g, ex, ey, ez, a);
        gid[a] = id;
        const int gi = id % g.nx;
        const int t = id / g.nx;
        const int gj = t % g.ny;
        const int gk = t / g.ny;
        bnd[a] = g_boundary_local(g, gi, gj, gk);
        ux[a] = double(x[id]);
        uy[a] = double(x[nn + id]);
        uz[a] = double(x[2*nn + id]);
        pp[a] = double(x[3*nn + id]);
        bx[a] = beta ? double(beta[id]) : 0.0;
        by[a] = beta ? double(beta[nn + id]) : 0.0;
        bz[a] = beta ? double(beta[2*nn + id]) : 0.0;
    }

    for (int iq=0; iq<4; ++iq) for (int jq=0; jq<4; ++jq) for (int kq=0; kq<4; ++kq) {
        double N[8], dX[8], dY[8], dZ[8];
        q1_shape_phys<RealT>(g, qp[iq], qp[jq], qp[kq], N, dX, dY, dZ);
        const double wq = qw[iq]*qw[jq]*qw[kq]*jac;

        double uh=0.0, vh=0.0, wh=0.0, ph=0.0;
        double gradUx[3]={0,0,0}, gradUy[3]={0,0,0}, gradUz[3]={0,0,0};
        double gradP[3]={0,0,0};
        double betah[3]={0,0,0};
        double divU=0.0;
        for (int b=0; b<8; ++b) {
            if (!bnd[b]) {
                uh += N[b]*ux[b]; vh += N[b]*uy[b]; wh += N[b]*uz[b];
                betah[0] += N[b]*bx[b]; betah[1] += N[b]*by[b]; betah[2] += N[b]*bz[b];
                gradUx[0] += dX[b]*ux[b]; gradUx[1] += dY[b]*ux[b]; gradUx[2] += dZ[b]*ux[b];
                gradUy[0] += dX[b]*uy[b]; gradUy[1] += dY[b]*uy[b]; gradUy[2] += dZ[b]*uy[b];
                gradUz[0] += dX[b]*uz[b]; gradUz[1] += dY[b]*uz[b]; gradUz[2] += dZ[b]*uz[b];
                divU += dX[b]*ux[b] + dY[b]*uy[b] + dZ[b]*uz[b];
            }
            if (gid[b] != pPin) {
                ph += N[b]*pp[b];
                gradP[0] += dX[b]*pp[b]; gradP[1] += dY[b]*pp[b]; gradP[2] += dZ[b]*pp[b];
            }
        }
        const double convUx = double(advScale)*(betah[0]*gradUx[0] + betah[1]*gradUx[1] + betah[2]*gradUx[2]);
        const double convUy = double(advScale)*(betah[0]*gradUy[0] + betah[1]*gradUy[1] + betah[2]*gradUy[2]);
        const double convUz = double(advScale)*(betah[0]*gradUz[0] + betah[1]*gradUz[1] + betah[2]*gradUz[2]);
        // Q1 strong Laplacian is zero elementwise; Galerkin diffusion remains in weak form.
        const double rmx = double(massCoeff)*uh + convUx + gradP[0];
        const double rmy = double(massCoeff)*vh + convUy + gradP[1];
        const double rmz = double(massCoeff)*wh + convUz + gradP[2];
        const double tauSupg = supgEnabled ? double(tau)*double(supgTauScale) : 0.0;

        for (int a=0; a<8; ++a) if (!bnd[a]) {
            const double streamNa = double(advScale)*(betah[0]*dX[a] + betah[1]*dY[a] + betah[2]*dZ[a]);
            const double ku = double(massCoeff)*N[a]*uh
                            + double(nu)*(dX[a]*gradUx[0] + dY[a]*gradUx[1] + dZ[a]*gradUx[2])
                            + N[a]*convUx - ph*dX[a]
                            + tauSupg*streamNa*rmx;
            const double kv = double(massCoeff)*N[a]*vh
                            + double(nu)*(dX[a]*gradUy[0] + dY[a]*gradUy[1] + dZ[a]*gradUy[2])
                            + N[a]*convUy - ph*dY[a]
                            + tauSupg*streamNa*rmy;
            const double kw = double(massCoeff)*N[a]*wh
                            + double(nu)*(dX[a]*gradUz[0] + dY[a]*gradUz[1] + dZ[a]*gradUz[2])
                            + N[a]*convUz - ph*dZ[a]
                            + tauSupg*streamNa*rmz;
            const int row = gid[a];
            atomicAdd(&y[row], RealT(wq*ku));
            atomicAdd(&y[nn + row], RealT(wq*kv));
            atomicAdd(&y[2*nn + row], RealT(wq*kw));
        }

        for (int a=0; a<8; ++a) if (gid[a] != pPin) {
            const double rp = N[a]*divU + double(tau)*(dX[a]*rmx + dY[a]*rmy + dZ[a]*rmz);
            atomicAdd(&y[3*nn + gid[a]], RealT(wq*rp));
        }
    }
}

template <class RealT>
__global__ void kernel_assemble_rhs_q1q1_nse_elements(Grid3DCudaLocal g,
                                                      RealT nu,
                                                      RealT tau,
                                                      RealT massCoeff,
                                                      RealT advScale,
                                                      int supgEnabled,
                                                      RealT supgTauScale,
                                                      const RealT* beta,
                                                      const RealT* uold,
                                                      RealT* rhs) {
    const int nn = g.n_nodes_host();
    const int exN = g.nx - 1;
    const int eyN = g.ny - 1;
    const int ezN = g.nz - 1;
    const int nElem = exN * eyN * ezN;
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nElem) return;
    const int ex = e % exN;
    const int tmp = e / exN;
    const int ey = tmp % eyN;
    const int ez = tmp / eyN;
    constexpr int pPin = 0;
    const double qp[4] = {-0.86113631159405257522, -0.33998104358485626480,
                           0.33998104358485626480,  0.86113631159405257522};
    const double qw[4] = { 0.34785484513745385737,  0.65214515486254614263,
                           0.65214515486254614263,  0.34785484513745385737};
    const double jac = g.hx*g.hy*g.hz/8.0;

    int gid[8]; bool bnd[8];
    for (int a=0; a<8; ++a) {
        const int id = stokes_local_node_id(g, ex, ey, ez, a);
        gid[a] = id;
        const int gi = id % g.nx;
        const int t = id / g.nx;
        const int gj = t % g.ny;
        const int gk = t / g.ny;
        bnd[a] = g_boundary_local(g, gi, gj, gk);
    }

    for (int iq=0; iq<4; ++iq) for (int jq=0; jq<4; ++jq) for (int kq=0; kq<4; ++kq) {
        double N[8], dX[8], dY[8], dZ[8];
        q1_shape_phys<RealT>(g, qp[iq], qp[jq], qp[kq], N, dX, dY, dZ);
        const double xx = (double(ex)+0.5*(1.0+qp[iq]))*g.hx;
        const double yy = (double(ey)+0.5*(1.0+qp[jq]))*g.hy;
        const double zz = (double(ez)+0.5*(1.0+qp[kq]))*g.hz;
        const double wq = qw[iq]*qw[jq]*qw[kq]*jac;
        double fx, fy, fz;
        q1q1_nse_mms_force_device(xx, yy, zz, double(nu), double(advScale), fx, fy, fz);
        double uoldh=0.0, voldh=0.0, woldh=0.0;
        double betah[3]={0,0,0};
        for (int b=0; b<8; ++b) if (!bnd[b]) {
            const int id=gid[b];
            uoldh += N[b]*double(uold[id]);
            voldh += N[b]*double(uold[nn+id]);
            woldh += N[b]*double(uold[2*nn+id]);
            if (beta) {
                betah[0] += N[b]*double(beta[id]);
                betah[1] += N[b]*double(beta[nn+id]);
                betah[2] += N[b]*double(beta[2*nn+id]);
            }
        }
        fx += double(massCoeff)*uoldh;
        fy += double(massCoeff)*voldh;
        fz += double(massCoeff)*woldh;
        const double tauSupg = supgEnabled ? double(tau)*double(supgTauScale) : 0.0;

        for (int a=0; a<8; ++a) if (!bnd[a]) {
            const double streamNa = double(advScale)*(betah[0]*dX[a] + betah[1]*dY[a] + betah[2]*dZ[a]);
            const double test = N[a] + tauSupg*streamNa;
            const int row=gid[a];
            atomicAdd(&rhs[row], RealT(wq*test*fx));
            atomicAdd(&rhs[nn+row], RealT(wq*test*fy));
            atomicAdd(&rhs[2*nn+row], RealT(wq*test*fz));
        }
        for (int a=0; a<8; ++a) if (gid[a] != pPin) {
            const double rp = double(tau)*(dX[a]*fx + dY[a]*fy + dZ[a]*fz);
            atomicAdd(&rhs[3*nn + gid[a]], RealT(wq*rp));
        }
    }
}



template <class RealT>
__global__ void kernel_set_index_value_local(int idx, RealT val, RealT* x) {
    if (blockIdx.x == 0 && threadIdx.x == 0) x[idx] = val;
}

template <class RealT>
__global__ void kernel_q1_lumped_mass_inverse_apply(Grid3DCudaLocal g,
                                                     const RealT* r,
                                                     RealT* z,
                                                     RealT sign) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    const int nn = g.n_nodes_host();
    if (q >= nn) return;
    constexpr int pPin = 0;
    if (q == pPin) {
        z[q] = RealT(0);
        return;
    }
    const int i = q % g.nx;
    const int tmp = q / g.nx;
    const int j = tmp % g.ny;
    const int k = tmp / (g.ny);
    const int cx = (i > 0 ? 1 : 0) + (i < g.nx-1 ? 1 : 0);
    const int cy = (j > 0 ? 1 : 0) + (j < g.ny-1 ? 1 : 0);
    const int cz = (k > 0 ? 1 : 0) + (k < g.nz-1 ? 1 : 0);
    const double lump = double(cx*cy*cz) * double(g.hx*g.hy*g.hz) / 8.0;
    z[q] = RealT(double(sign) * double(r[q]) / max(lump, 1.0e-300));
}

template <class RealT>
__global__ void kernel_q1_pressure_ap_full_apply(Grid3DCudaLocal g,
                                                  RealT nu,
                                                  RealT tau,
                                                  RealT massCoeff,
                                                  RealT advScale,
                                                  int supgEnabled,
                                                  RealT supgTauScale,
                                                  const RealT* beta,
                                                  const RealT* p,
                                                  RealT* y) {
    const int nn = g.n_nodes_host();
    const int exN = g.nx - 1;
    const int eyN = g.ny - 1;
    const int ezN = g.nz - 1;
    const int nElem = exN * eyN * ezN;
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nElem) return;

    const int ex = e % exN;
    const int tmp = e / exN;
    const int ey = tmp % eyN;
    const int ez = tmp / eyN;
    constexpr int pPin = 0;

    const double qp[4] = {-0.86113631159405257522, -0.33998104358485626480,
                           0.33998104358485626480,  0.86113631159405257522};
    const double qw[4] = { 0.34785484513745385737,  0.65214515486254614263,
                           0.65214515486254614263,  0.34785484513745385737};
    const double jac = g.hx*g.hy*g.hz/8.0;

    int gid[8];
    double pp[8], bx[8], by[8], bz[8];
    for (int a=0; a<8; ++a) {
        const int id = stokes_local_node_id(g, ex, ey, ez, a);
        gid[a] = id;
        pp[a] = (id == pPin) ? 0.0 : double(p[id]);
        bx[a] = beta ? double(beta[id]) : 0.0;
        by[a] = beta ? double(beta[nn + id]) : 0.0;
        bz[a] = beta ? double(beta[2*nn + id]) : 0.0;
    }

    const double tauSupg = supgEnabled ? double(tau)*double(supgTauScale) : 0.0;
    for (int iq=0; iq<4; ++iq) for (int jq=0; jq<4; ++jq) for (int kq=0; kq<4; ++kq) {
        double N[8], dX[8], dY[8], dZ[8];
        q1_shape_phys<RealT>(g, qp[iq], qp[jq], qp[kq], N, dX, dY, dZ);
        const double wq = qw[iq]*qw[jq]*qw[kq]*jac;

        double ph=0.0;
        double gradP[3]={0,0,0};
        double betah[3]={0,0,0};
        for (int b=0; b<8; ++b) {
            ph += N[b]*pp[b];
            gradP[0] += dX[b]*pp[b];
            gradP[1] += dY[b]*pp[b];
            gradP[2] += dZ[b]*pp[b];
            betah[0] += N[b]*bx[b];
            betah[1] += N[b]*by[b];
            betah[2] += N[b]*bz[b];
        }
        const double ax = double(advScale)*betah[0];
        const double ay = double(advScale)*betah[1];
        const double az = double(advScale)*betah[2];
        const double convP = ax*gradP[0] + ay*gradP[1] + az*gradP[2];

        for (int a=0; a<8; ++a) if (gid[a] != pPin) {
            const double streamNa = ax*dX[a] + ay*dY[a] + az*dZ[a];
            const double val = double(massCoeff)*N[a]*ph
                             + double(nu)*(dX[a]*gradP[0] + dY[a]*gradP[1] + dZ[a]*gradP[2])
                             + N[a]*convP
                             + tauSupg*streamNa*(convP + double(massCoeff)*ph);
            atomicAdd(&y[gid[a]], RealT(wq*val));
        }
    }
}

template <class RealT>
__global__ void kernel_q1q1_pcd_apply_B_velocity(Grid3DCudaLocal g,
                                                  RealT tau,
                                                  RealT massCoeff,
                                                  RealT advScale,
                                                  const RealT* beta,
                                                  const RealT* u,
                                                  RealT* bp) {
    const int nn = g.n_nodes_host();
    const int exN = g.nx - 1;
    const int eyN = g.ny - 1;
    const int ezN = g.nz - 1;
    const int nElem = exN * eyN * ezN;
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nElem) return;

    const int ex = e % exN;
    const int tmp = e / exN;
    const int ey = tmp % eyN;
    const int ez = tmp / eyN;
    constexpr int pPin = 0;

    const double qp[4] = {-0.86113631159405257522, -0.33998104358485626480,
                           0.33998104358485626480,  0.86113631159405257522};
    const double qw[4] = { 0.34785484513745385737,  0.65214515486254614263,
                           0.65214515486254614263,  0.34785484513745385737};
    const double jac = g.hx*g.hy*g.hz/8.0;

    int gid[8]; bool bnd[8];
    double ux[8], uy[8], uz[8], bx[8], by[8], bz[8];
    for (int a=0; a<8; ++a) {
        const int id = stokes_local_node_id(g, ex, ey, ez, a);
        gid[a] = id;
        const int gi = id % g.nx;
        const int t = id / g.nx;
        const int gj = t % g.ny;
        const int gk = t / g.ny;
        bnd[a] = g_boundary_local(g, gi, gj, gk);
        ux[a] = double(u[id]);
        uy[a] = double(u[nn + id]);
        uz[a] = double(u[2*nn + id]);
        bx[a] = beta ? double(beta[id]) : 0.0;
        by[a] = beta ? double(beta[nn + id]) : 0.0;
        bz[a] = beta ? double(beta[2*nn + id]) : 0.0;
    }

    for (int iq=0; iq<4; ++iq) for (int jq=0; jq<4; ++jq) for (int kq=0; kq<4; ++kq) {
        double N[8], dX[8], dY[8], dZ[8];
        q1_shape_phys<RealT>(g, qp[iq], qp[jq], qp[kq], N, dX, dY, dZ);
        const double wq = qw[iq]*qw[jq]*qw[kq]*jac;

        double uh=0.0, vh=0.0, wh=0.0;
        double gradUx[3]={0,0,0}, gradUy[3]={0,0,0}, gradUz[3]={0,0,0};
        double betah[3]={0,0,0};
        double divU=0.0;
        for (int b=0; b<8; ++b) if (!bnd[b]) {
            uh += N[b]*ux[b]; vh += N[b]*uy[b]; wh += N[b]*uz[b];
            betah[0] += N[b]*bx[b]; betah[1] += N[b]*by[b]; betah[2] += N[b]*bz[b];
            gradUx[0] += dX[b]*ux[b]; gradUx[1] += dY[b]*ux[b]; gradUx[2] += dZ[b]*ux[b];
            gradUy[0] += dX[b]*uy[b]; gradUy[1] += dY[b]*uy[b]; gradUy[2] += dZ[b]*uy[b];
            gradUz[0] += dX[b]*uz[b]; gradUz[1] += dY[b]*uz[b]; gradUz[2] += dZ[b]*uz[b];
            divU += dX[b]*ux[b] + dY[b]*uy[b] + dZ[b]*uz[b];
        }
        const double convUx = double(advScale)*(betah[0]*gradUx[0] + betah[1]*gradUx[1] + betah[2]*gradUx[2]);
        const double convUy = double(advScale)*(betah[0]*gradUy[0] + betah[1]*gradUy[1] + betah[2]*gradUy[2]);
        const double convUz = double(advScale)*(betah[0]*gradUz[0] + betah[1]*gradUz[1] + betah[2]*gradUz[2]);
        const double rmx = double(massCoeff)*uh + convUx;
        const double rmy = double(massCoeff)*vh + convUy;
        const double rmz = double(massCoeff)*wh + convUz;

        for (int a=0; a<8; ++a) if (gid[a] != pPin) {
            const double val = N[a]*divU + double(tau)*(dX[a]*rmx + dY[a]*rmy + dZ[a]*rmz);
            atomicAdd(&bp[gid[a]], RealT(wq*val));
        }
    }
}

template <class RealT>
__global__ void kernel_negate_local(int n, RealT* x) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q < n) x[q] = -x[q];
}

template <class RealT>
__global__ void kernel_q1q1_nse_scalar_diag_prec(Grid3DCudaLocal g,
                                                 RealT nu,
                                                 RealT tau,
                                                 RealT massCoeff,
                                                 const RealT* r,
                                                 RealT* z) {
    const int nn = g.n_nodes_host();
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= nn) return;
    const int i = q % g.nx;
    const int tmp = q / g.nx;
    const int j = tmp % g.ny;
    const int k = tmp / g.ny;
    const bool bnd = g_boundary_local(g,i,j,k);
    constexpr int pPin=0;
    if (bnd) {
        z[q]=r[q]; z[nn+q]=r[nn+q]; z[2*nn+q]=r[2*nn+q];
    }
    double mdiag=0.0, kdiag=0.0;
    const double gp = 0.577350269189625764509148780501957456;
    const double qpts[2] = {-gp, gp};
    const double jac = g.hx*g.hy*g.hz/8.0;
    const int ex0=max(0,i-1), ex1=min(g.nx-2,i);
    const int ey0=max(0,j-1), ey1=min(g.ny-2,j);
    const int ez0=max(0,k-1), ez1=min(g.nz-2,k);
    for (int ez=ez0; ez<=ez1; ++ez) for (int ey=ey0; ey<=ey1; ++ey) for (int ex=ex0; ex<=ex1; ++ex) {
        const int ax=i-ex, ay=j-ey, az=k-ez;
        const int a=ax+2*ay+4*az;
        (void)a;
        for (int iq=0;iq<2;++iq) for (int jq=0;jq<2;++jq) for (int kq=0;kq<2;++kq) {
            double N[8],dX[8],dY[8],dZ[8];
            q1_shape_phys<RealT>(g,qpts[iq],qpts[jq],qpts[kq],N,dX,dY,dZ);
            mdiag += N[a]*N[a]*jac;
            kdiag += (dX[a]*dX[a]+dY[a]*dY[a]+dZ[a]*dZ[a])*jac;
        }
    }
    const RealT dU = bnd ? RealT(1) : max(RealT(massCoeff*mdiag + nu*kdiag), RealT(1e-30));
    const RealT dP = (q==pPin) ? RealT(1) : max(RealT(tau*kdiag), RealT(1e-30));
    if (!bnd) {
        z[q]=r[q]/dU; z[nn+q]=r[nn+q]/dU; z[2*nn+q]=r[2*nn+q]/dU;
    }
    if (q==pPin) z[3*nn+q]=r[3*nn+q];
    else z[3*nn+q]=r[3*nn+q]/dP;
}

template <class RealT>
class StructuredQ1Q1NsePicardOperator {
public:
    StructuredQ1Q1NsePicardOperator(const memoirs::structured::StructuredGrid3D& grid,
                                    RealT nu,
                                    RealT tau,
                                    RealT massCoeff,
                                    RealT advScale,
                                    int supgEnabled,
                                    RealT supgTauScale,
                                    const RealT* beta)
        : grid_(grid), dg_(to_cuda_grid_local(grid)), nu_(nu), tau_(tau), massCoeff_(massCoeff),
          advScale_(advScale), supgEnabled_(supgEnabled), supgTauScale_(supgTauScale), beta_(beta) {}
    int size() const { return 4*grid_.n_nodes(); }
    RealT tau() const { return tau_; }
    const memoirs::structured::StructuredGrid3D& grid() const { return grid_; }
    void apply(const RealT* x, RealT* y) const {
        const int nn=grid_.n_nodes(); const int total=4*nn; const int block=256;
        const int nElem=(grid_.nx-1)*(grid_.ny-1)*(grid_.nz-1);
        kernel_zero_local<RealT><<<div_up_local(total,block),block>>>(total,y); MEMOIRS_CUDA_KERNEL_CHECK();
        kernel_apply_q1q1_nse_elements<RealT><<<div_up_local(nElem,block),block>>>(dg_,nu_,tau_,massCoeff_,advScale_,supgEnabled_,supgTauScale_,beta_,x,y); MEMOIRS_CUDA_KERNEL_CHECK();
        kernel_apply_stokes_strong_rows<RealT><<<div_up_local(nn,block),block>>>(dg_,x,y); MEMOIRS_CUDA_KERNEL_CHECK();
    }
    void assemble_rhs_from_old(const RealT* uold, RealT* rhs) const {
        const int nn=grid_.n_nodes(); const int total=4*nn; const int block=256;
        const int nElem=(grid_.nx-1)*(grid_.ny-1)*(grid_.nz-1);
        kernel_zero_local<RealT><<<div_up_local(total,block),block>>>(total,rhs); MEMOIRS_CUDA_KERNEL_CHECK();
        kernel_assemble_rhs_q1q1_nse_elements<RealT><<<div_up_local(nElem,block),block>>>(dg_,nu_,tau_,massCoeff_,advScale_,supgEnabled_,supgTauScale_,beta_,uold,rhs); MEMOIRS_CUDA_KERNEL_CHECK();
        kernel_apply_stokes_rhs_strong_rows<RealT><<<div_up_local(nn,block),block>>>(dg_,rhs); MEMOIRS_CUDA_KERNEL_CHECK();
    }
    void fill_exact(RealT* x) const {
        const int nn=grid_.n_nodes(); const int block=256;
        kernel_fill_stokes_exact<RealT><<<div_up_local(nn,block),block>>>(dg_,x); MEMOIRS_CUDA_KERNEL_CHECK();
    }
private:
    memoirs::structured::StructuredGrid3D grid_;
    Grid3DCudaLocal dg_;
    RealT nu_, tau_, massCoeff_, advScale_, supgTauScale_;
    int supgEnabled_;
    const RealT* beta_;
};

template <class RealT>
class StructuredQ1Q1NseScalarDiagonalPreconditioner {
public:
    StructuredQ1Q1NseScalarDiagonalPreconditioner(const memoirs::structured::StructuredGrid3D& grid) : grid_(grid), dg_(to_cuda_grid_local(grid)) {}
    void rebuild(RealT nu, RealT tau, RealT massCoeff, RealT advScale, int supgEnabled, RealT supgTauScale, const RealT* beta) {
        (void)advScale; (void)supgEnabled; (void)supgTauScale; (void)beta;
        nu_=nu; tau_=tau; massCoeff_=massCoeff;
    }
    void apply(const RealT* r, RealT* z) {
        const int nn=grid_.n_nodes(); const int block=256;
        kernel_q1q1_nse_scalar_diag_prec<RealT><<<div_up_local(nn,block),block>>>(dg_,nu_,tau_,massCoeff_,r,z); MEMOIRS_CUDA_KERNEL_CHECK();
    }
private:
    memoirs::structured::StructuredGrid3D grid_;
    Grid3DCudaLocal dg_;
    RealT nu_=RealT(1), tau_=RealT(1), massCoeff_=RealT(1);
};


template <class RealT>
__global__ void kernel_build_q1q1_nse_scalar_diag(Grid3DCudaLocal g,
                                                  RealT nu,
                                                  RealT tau,
                                                  RealT massCoeff,
                                                  RealT* diag) {
    const int nn = g.n_nodes_host();
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= nn) return;
    const int i = q % g.nx;
    const int tmp = q / g.nx;
    const int j = tmp % g.ny;
    const int k = tmp / g.ny;
    const bool bnd = g_boundary_local(g,i,j,k);
    constexpr int pPin = 0;
    double mdiag = 0.0, kdiag = 0.0;
    const double gp = 0.577350269189625764509148780501957456;
    const double qpts[2] = {-gp, gp};
    const double jac = g.hx*g.hy*g.hz/8.0;
    const int ex0=max(0,i-1), ex1=min(g.nx-2,i);
    const int ey0=max(0,j-1), ey1=min(g.ny-2,j);
    const int ez0=max(0,k-1), ez1=min(g.nz-2,k);
    for (int ez=ez0; ez<=ez1; ++ez) for (int ey=ey0; ey<=ey1; ++ey) for (int ex=ex0; ex<=ex1; ++ex) {
        const int ax=i-ex, ay=j-ey, az=k-ez;
        const int a=ax+2*ay+4*az;
        for (int iq=0;iq<2;++iq) for (int jq=0;jq<2;++jq) for (int kq=0;kq<2;++kq) {
            double N[8],dX[8],dY[8],dZ[8];
            q1_shape_phys<RealT>(g,qpts[iq],qpts[jq],qpts[kq],N,dX,dY,dZ);
            mdiag += N[a]*N[a]*jac;
            kdiag += (dX[a]*dX[a]+dY[a]*dY[a]+dZ[a]*dZ[a])*jac;
        }
    }
    const RealT dU = bnd ? RealT(1) : max(RealT(massCoeff*mdiag + nu*kdiag), RealT(1e-30));
    const RealT dP = (q==pPin) ? RealT(1) : max(RealT(tau*kdiag), RealT(1e-30));
    diag[q] = dU;
    diag[nn+q] = dU;
    diag[2*nn+q] = dU;
    diag[3*nn+q] = dP;
}

template <class RealT>
__global__ void kernel_jacobi_update_with_diag(int n, RealT omega, const RealT* residual, const RealT* diag, RealT* x) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q < n) x[q] += omega * residual[q] / diag[q];
}

template <class RealT>
__global__ void kernel_q1q1_restrict_full_weighting(Grid3DCudaLocal gf, Grid3DCudaLocal gc, const RealT* fine, RealT* coarse) {
    const int nnf = gf.n_nodes_host();
    const int nnc = gc.n_nodes_host();
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= nnf) return;
    const int fi = q % gf.nx;
    const int tmp = q / gf.nx;
    const int fj = tmp % gf.ny;
    const int fk = tmp / gf.ny;
    int ci[2], cj[2], ck[2];
    double wi[2], wj[2], wk[2];
    int ni=1,nj=1,nk=1;
    if ((fi & 1)==0) { ci[0]=fi/2; wi[0]=1.0; } else { ni=2; ci[0]=fi/2; ci[1]=ci[0]+1; wi[0]=0.5; wi[1]=0.5; }
    if ((fj & 1)==0) { cj[0]=fj/2; wj[0]=1.0; } else { nj=2; cj[0]=fj/2; cj[1]=cj[0]+1; wj[0]=0.5; wj[1]=0.5; }
    if ((fk & 1)==0) { ck[0]=fk/2; wk[0]=1.0; } else { nk=2; ck[0]=fk/2; ck[1]=ck[0]+1; wk[0]=0.5; wk[1]=0.5; }
    for (int c=0; c<4; ++c) {
        const RealT val = fine[c*nnf + q];
        for (int kk=0; kk<nk; ++kk) for (int jj=0; jj<nj; ++jj) for (int ii=0; ii<ni; ++ii) {
            if (ci[ii] < 0 || ci[ii] >= gc.nx || cj[jj] < 0 || cj[jj] >= gc.ny || ck[kk] < 0 || ck[kk] >= gc.nz) continue;
            const int cq = ci[ii] + gc.nx*(cj[jj] + gc.ny*ck[kk]);
            const double wt = 0.125 * wi[ii]*wj[jj]*wk[kk];
            atomicAdd(&coarse[c*nnc + cq], RealT(wt)*val);
        }
    }
}

template <class RealT>
__global__ void kernel_q1q1_prolong_add(Grid3DCudaLocal gc, Grid3DCudaLocal gf, const RealT* coarse, RealT* fine) {
    const int nnf = gf.n_nodes_host();
    const int nnc = gc.n_nodes_host();
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= nnf) return;
    const int fi = q % gf.nx;
    const int tmp = q / gf.nx;
    const int fj = tmp % gf.ny;
    const int fk = tmp / gf.ny;
    int ci[2], cj[2], ck[2];
    double wi[2], wj[2], wk[2];
    int ni=1,nj=1,nk=1;
    if ((fi & 1)==0) { ci[0]=fi/2; wi[0]=1.0; } else { ni=2; ci[0]=fi/2; ci[1]=ci[0]+1; wi[0]=0.5; wi[1]=0.5; }
    if ((fj & 1)==0) { cj[0]=fj/2; wj[0]=1.0; } else { nj=2; cj[0]=fj/2; cj[1]=cj[0]+1; wj[0]=0.5; wj[1]=0.5; }
    if ((fk & 1)==0) { ck[0]=fk/2; wk[0]=1.0; } else { nk=2; ck[0]=fk/2; ck[1]=ck[0]+1; wk[0]=0.5; wk[1]=0.5; }
    for (int c=0; c<4; ++c) {
        double acc = 0.0;
        for (int kk=0; kk<nk; ++kk) for (int jj=0; jj<nj; ++jj) for (int ii=0; ii<ni; ++ii) {
            if (ci[ii] < 0 || ci[ii] >= gc.nx || cj[jj] < 0 || cj[jj] >= gc.ny || ck[kk] < 0 || ck[kk] >= gc.nz) continue;
            const int cq = ci[ii] + gc.nx*(cj[jj] + gc.ny*ck[kk]);
            acc += wi[ii]*wj[jj]*wk[kk] * double(coarse[c*nnc + cq]);
        }
        fine[c*nnf + q] += RealT(acc);
    }
}

template <class RealT>
__global__ void kernel_q1q1_inject_beta_level(Grid3DCudaLocal gf, Grid3DCudaLocal gc, const RealT* bf, RealT* bc) {
    const int nnf = gf.n_nodes_host();
    const int nnc = gc.n_nodes_host();
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= nnc) return;
    const int ci = q % gc.nx;
    const int tmp = q / gc.nx;
    const int cj = tmp % gc.ny;
    const int ck = tmp / gc.ny;
    const int fq = (2*ci) + gf.nx*((2*cj) + gf.ny*(2*ck));
    bc[q] = bf[fq];
    bc[nnc+q] = bf[nnf+fq];
    bc[2*nnc+q] = bf[2*nnf+fq];
    bc[3*nnc+q] = RealT(0);
}

template <class RealT>
__global__ void kernel_set_basis_vector_local(int n, int col, RealT* x) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q < n) x[q] = (q == col) ? RealT(1) : RealT(0);
}

template <class RealT>
__global__ void kernel_dense_lu_solve_small_device(int n, const RealT* lu, const int* piv, const RealT* b, RealT* x) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    RealT tmp[512];
    if (n > 512) return;
    for (int i=0; i<n; ++i) tmp[i] = b[i];
    for (int k=0; k<n; ++k) {
        const int pk = piv[k];
        if (pk != k) { const RealT t = tmp[k]; tmp[k] = tmp[pk]; tmp[pk] = t; }
    }
    for (int i=0; i<n; ++i) {
        RealT s = tmp[i];
        for (int j=0; j<i; ++j) s -= lu[i*n+j] * tmp[j];
        tmp[i] = s;
    }
    for (int i=n-1; i>=0; --i) {
        RealT s = tmp[i];
        for (int j=i+1; j<n; ++j) s -= lu[i*n+j] * tmp[j];
        tmp[i] = s / lu[i*n+i];
    }
    for (int i=0; i<n; ++i) x[i] = tmp[i];
}

template <class RealT>
class StructuredQ1Q1CoupledGmgPreconditioner {
public:
    StructuredQ1Q1CoupledGmgPreconditioner(const memoirs::structured::StructuredGrid3D& grid,
                                           int preSweeps,
                                           int postSweeps,
                                           RealT omega,
                                           int verbose)
        : pre_(preSweeps), post_(postSweeps), omega_(omega), verbose_(verbose) {
        int nElem = grid.nx - 1;
        if (grid.nx != grid.ny || grid.nx != grid.nz) throw std::runtime_error("Q1/Q1 GMG requires cubic structured grid");
        if (nElem < 2) throw std::runtime_error("Q1/Q1 GMG requires at least 2 elements per direction");
        while (true) {
            const int nodes = nElem + 1;
            levels_.emplace_back();
            Level& lev = levels_.back();
            lev.grid = memoirs::structured::StructuredGrid3D(nodes,nodes,nodes);
            lev.dg = to_cuda_grid_local(lev.grid);
            const int total = 4*lev.grid.n_nodes();
            lev.x.resize(total); lev.b.resize(total); lev.Ax.resize(total); lev.res.resize(total); lev.corr.resize(total); lev.beta.resize(total); lev.diag.resize(total);
            if (nElem <= 2 || (nElem % 2) != 0) break;
            nElem /= 2;
        }
        const int nc = 4*levels_.back().grid.n_nodes();
        if (nc > 512) throw std::runtime_error("Q1/Q1 GMG coarse grid too large for small dense device solve; use power-of-two N or smaller odd coarse");
        d_lu_.resize(static_cast<std::size_t>(nc)*static_cast<std::size_t>(nc));
        d_piv_.resize(static_cast<std::size_t>(nc));
        if (verbose_) {
            std::cout << "q1q1 coupled_gmg levels=" << levels_.size()
                      << " coarseElements=" << (levels_.back().grid.nx-1)
                      << " coarseNdof=" << nc << "\n";
        }
    }

    void rebuild(RealT nu, RealT tauFine, RealT massCoeff, RealT advScale, int supgEnabled, RealT supgTauScale, const RealT* betaFine) {
        nu_ = nu; tauFine_ = tauFine; massCoeff_ = massCoeff; advScale_ = advScale; supgEnabled_ = supgEnabled; supgTauScale_ = supgTauScale;
        const int block = 256;
        const int total0 = 4*levels_[0].grid.n_nodes();
        if (betaFine) {
            kernel_copy_local<RealT><<<div_up_local(total0,block),block>>>(total0,betaFine,levels_[0].beta.data()); MEMOIRS_CUDA_KERNEL_CHECK();
        } else {
            levels_[0].beta.zero();
        }
        for (std::size_t l=1; l<levels_.size(); ++l) {
            kernel_q1q1_inject_beta_level<RealT><<<div_up_local(levels_[l].grid.n_nodes(),block),block>>>(levels_[l-1].dg,levels_[l].dg,levels_[l-1].beta.data(),levels_[l].beta.data()); MEMOIRS_CUDA_KERNEL_CHECK();
        }
        const double hf = levels_[0].grid.hx();
        for (auto& lev : levels_) {
            const double ratio = lev.grid.hx()/hf;
            lev.tau = RealT(double(tauFine_) * ratio * ratio);
            kernel_build_q1q1_nse_scalar_diag<RealT><<<div_up_local(lev.grid.n_nodes(),block),block>>>(lev.dg,nu_,lev.tau,massCoeff_,lev.diag.data()); MEMOIRS_CUDA_KERNEL_CHECK();
        }
        build_coarse_lu();
    }

    void apply(const RealT* r, RealT* z) {
        kernel_zero_local<RealT><<<div_up_local(4*levels_[0].grid.n_nodes(),256),256>>>(4*levels_[0].grid.n_nodes(),z); MEMOIRS_CUDA_KERNEL_CHECK();
        vcycle(0,r,z);
    }

private:
    struct Level {
        memoirs::structured::StructuredGrid3D grid;
        Grid3DCudaLocal dg;
        RealT tau = RealT(0);
        DeviceBuffer<RealT> x,b,Ax,res,corr,beta,diag;
    };

    void apply_level(int l, const RealT* x, RealT* y) {
        const Level& lev = levels_[static_cast<std::size_t>(l)];
        StructuredQ1Q1NsePicardOperator<RealT> A(lev.grid,nu_,lev.tau,massCoeff_,advScale_,supgEnabled_,supgTauScale_,lev.beta.data());
        A.apply(x,y);
    }

    void smooth(int l, const RealT* b, RealT* x, int sweeps) {
        Level& lev = levels_[static_cast<std::size_t>(l)];
        const int n = 4*lev.grid.n_nodes();
        const int block = 256;
        for (int s=0; s<sweeps; ++s) {
            apply_level(l,x,lev.Ax.data());
            kernel_subtract_local<RealT><<<div_up_local(n,block),block>>>(n,b,lev.Ax.data(),lev.res.data()); MEMOIRS_CUDA_KERNEL_CHECK();
            kernel_jacobi_update_with_diag<RealT><<<div_up_local(n,block),block>>>(n,omega_,lev.res.data(),lev.diag.data(),x); MEMOIRS_CUDA_KERNEL_CHECK();
        }
    }

    void vcycle(int l, const RealT* b, RealT* x) {
        if (l == int(levels_.size())-1) { solve_coarse(b,x); return; }
        Level& fine = levels_[static_cast<std::size_t>(l)];
        Level& coarse = levels_[static_cast<std::size_t>(l+1)];
        const int nf = 4*fine.grid.n_nodes();
        const int nc = 4*coarse.grid.n_nodes();
        const int block = 256;
        smooth(l,b,x,pre_);
        apply_level(l,x,fine.Ax.data());
        kernel_subtract_local<RealT><<<div_up_local(nf,block),block>>>(nf,b,fine.Ax.data(),fine.res.data()); MEMOIRS_CUDA_KERNEL_CHECK();
        kernel_zero_local<RealT><<<div_up_local(nc,block),block>>>(nc,coarse.b.data()); MEMOIRS_CUDA_KERNEL_CHECK();
        kernel_zero_local<RealT><<<div_up_local(nc,block),block>>>(nc,coarse.x.data()); MEMOIRS_CUDA_KERNEL_CHECK();
        kernel_q1q1_restrict_full_weighting<RealT><<<div_up_local(fine.grid.n_nodes(),block),block>>>(fine.dg,coarse.dg,fine.res.data(),coarse.b.data()); MEMOIRS_CUDA_KERNEL_CHECK();
        vcycle(l+1,coarse.b.data(),coarse.x.data());
        kernel_q1q1_prolong_add<RealT><<<div_up_local(fine.grid.n_nodes(),block),block>>>(coarse.dg,fine.dg,coarse.x.data(),x); MEMOIRS_CUDA_KERNEL_CHECK();
        smooth(l,b,x,post_);
    }

    void build_coarse_lu() {
        Level& c = levels_.back();
        const int nc = 4*c.grid.n_nodes();
        const int block = 256;
        std::vector<RealT> hA(static_cast<std::size_t>(nc)*static_cast<std::size_t>(nc));
        std::vector<RealT> hy(static_cast<std::size_t>(nc));
        for (int col=0; col<nc; ++col) {
            kernel_set_basis_vector_local<RealT><<<div_up_local(nc,block),block>>>(nc,col,c.x.data()); MEMOIRS_CUDA_KERNEL_CHECK();
            apply_level(int(levels_.size())-1,c.x.data(),c.Ax.data());
            c.Ax.copy_to_host(hy.data(),hy.size());
            for (int row=0; row<nc; ++row) hA[static_cast<std::size_t>(row)*nc + col] = hy[static_cast<std::size_t>(row)];
        }
        hPiv_.assign(static_cast<std::size_t>(nc),0);
        for (int k=0; k<nc; ++k) {
            int piv=k; double best=std::abs(double(hA[static_cast<std::size_t>(k)*nc+k]));
            for (int i=k+1; i<nc; ++i) {
                const double v=std::abs(double(hA[static_cast<std::size_t>(i)*nc+k]));
                if (v>best) { best=v; piv=i; }
            }
            if (!(best > 0.0) || !std::isfinite(best)) throw std::runtime_error("Q1/Q1 GMG coarse LU singular/nonfinite pivot");
            hPiv_[static_cast<std::size_t>(k)] = piv;
            if (piv != k) {
                for (int j=0; j<nc; ++j) std::swap(hA[static_cast<std::size_t>(k)*nc+j], hA[static_cast<std::size_t>(piv)*nc+j]);
            }
            const RealT akk = hA[static_cast<std::size_t>(k)*nc+k];
            for (int i=k+1; i<nc; ++i) {
                hA[static_cast<std::size_t>(i)*nc+k] /= akk;
                const RealT lik = hA[static_cast<std::size_t>(i)*nc+k];
                for (int j=k+1; j<nc; ++j) hA[static_cast<std::size_t>(i)*nc+j] -= lik*hA[static_cast<std::size_t>(k)*nc+j];
            }
        }
        d_lu_.copy_from_host(hA.data(),hA.size());
        d_piv_.copy_from_host(hPiv_.data(),hPiv_.size());
    }

    void solve_coarse(const RealT* b, RealT* x) {
        const int nc = 4*levels_.back().grid.n_nodes();
        kernel_dense_lu_solve_small_device<RealT><<<1,1>>>(nc,d_lu_.data(),d_piv_.data(),b,x); MEMOIRS_CUDA_KERNEL_CHECK();
    }

    std::vector<Level> levels_;
    int pre_=2, post_=2, verbose_=0;
    RealT omega_=RealT(0.7), nu_=RealT(1), tauFine_=RealT(1), massCoeff_=RealT(1), advScale_=RealT(0), supgTauScale_=RealT(1);
    int supgEnabled_=1;
    DeviceBuffer<RealT> d_lu_;
    DeviceBuffer<int> d_piv_;
    std::vector<int> hPiv_;
};



template <class RealT>
__device__ inline double q1_assembled_mass1d_coeff(int n, int i, int j, double h) {
    if (j < 0 || j >= n) return 0.0;
    const int d = j - i;
    if (d == 0) return (i == 0 || i == n-1) ? (h/3.0) : (2.0*h/3.0);
    if (d == -1 || d == 1) return h/6.0;
    return 0.0;
}

template <class RealT>
__device__ inline double q1_assembled_stiff1d_coeff(int n, int i, int j, double h) {
    if (j < 0 || j >= n) return 0.0;
    const int d = j - i;
    if (d == 0) return (i == 0 || i == n-1) ? (1.0/h) : (2.0/h);
    if (d == -1 || d == 1) return -1.0/h;
    return 0.0;
}

template <class RealT>
__device__ inline double q1_assembled_adv1d_coeff(int n, int i, int j) {
    // Central Galerkin advection matrix entry int phi_i * d(phi_j)/dx dx.
    // The h cancels in 1D: derivative gives 1/h, dx gives h.
    if (j < 0 || j >= n) return 0.0;
    const int d = j - i;
    if (d == -1) return -0.5;
    if (d ==  1) return  0.5;
    if (d == 0) {
        if (i == 0) return -0.5;
        if (i == n-1) return 0.5;
    }
    return 0.0;
}

template <class RealT>
__global__ void kernel_q1_scalar_fe_apply_stencil(Grid3DCudaLocal g,
                                                  RealT alphaMass,
                                                  RealT betaStiff,
                                                  RealT convX,
                                                  RealT convY,
                                                  RealT convZ,
                                                  int strongBoundary,
                                                  int strongPin,
                                                  const RealT* x,
                                                  RealT* y) {
    const int nn = g.n_nodes_host();
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= nn) return;
    const int i = q % g.nx;
    const int tmp = q / g.nx;
    const int j = tmp % g.ny;
    const int k = tmp / g.ny;
    if (strongBoundary && g_boundary_local(g,i,j,k)) { y[q] = x[q]; return; }
    if (strongPin && q == 0) { y[q] = x[q]; return; }

    double acc = 0.0;
    for (int dk=-1; dk<=1; ++dk) {
        const int kk = k + dk;
        if (kk < 0 || kk >= g.nz) continue;
        const double mz = q1_assembled_mass1d_coeff<RealT>(g.nz,k,kk,g.hz);
        const double kz = q1_assembled_stiff1d_coeff<RealT>(g.nz,k,kk,g.hz);
        for (int dj=-1; dj<=1; ++dj) {
            const int jj = j + dj;
            if (jj < 0 || jj >= g.ny) continue;
            const double my = q1_assembled_mass1d_coeff<RealT>(g.ny,j,jj,g.hy);
            const double ky = q1_assembled_stiff1d_coeff<RealT>(g.ny,j,jj,g.hy);
            for (int di=-1; di<=1; ++di) {
                const int ii = i + di;
                if (ii < 0 || ii >= g.nx) continue;
                if (strongBoundary && g_boundary_local(g,ii,jj,kk)) continue;
                const int qq = ii + g.nx*(jj + g.ny*kk);
                if (strongPin && qq == 0) continue;
                const double mx = q1_assembled_mass1d_coeff<RealT>(g.nx,i,ii,g.hx);
                const double kx = q1_assembled_stiff1d_coeff<RealT>(g.nx,i,ii,g.hx);
                const double gx = q1_assembled_adv1d_coeff<RealT>(g.nx,i,ii);
                const double gy = q1_assembled_adv1d_coeff<RealT>(g.ny,j,jj);
                const double gz = q1_assembled_adv1d_coeff<RealT>(g.nz,k,kk);
                const double m3 = mx*my*mz;
                const double k3 = kx*my*mz + mx*ky*mz + mx*my*kz;
                const double c3 = double(convX)*gx*my*mz + double(convY)*mx*gy*mz + double(convZ)*mx*my*gz;
                acc += (double(alphaMass)*m3 + double(betaStiff)*k3 + c3) * double(x[qq]);
            }
        }
    }
    y[q] = RealT(acc);
}

template <class RealT>
__global__ void kernel_q1_scalar_fe_diag_stencil(Grid3DCudaLocal g,
                                                 RealT alphaMass,
                                                 RealT betaStiff,
                                                 RealT convX,
                                                 RealT convY,
                                                 RealT convZ,
                                                 int strongBoundary,
                                                 int strongPin,
                                                 RealT* diag) {
    const int nn = g.n_nodes_host();
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= nn) return;
    const int i = q % g.nx;
    const int tmp = q / g.nx;
    const int j = tmp % g.ny;
    const int k = tmp / g.ny;
    if (strongBoundary && g_boundary_local(g,i,j,k)) { diag[q] = RealT(1); return; }
    if (strongPin && q == 0) { diag[q] = RealT(1); return; }
    const double mx = q1_assembled_mass1d_coeff<RealT>(g.nx,i,i,g.hx);
    const double my = q1_assembled_mass1d_coeff<RealT>(g.ny,j,j,g.hy);
    const double mz = q1_assembled_mass1d_coeff<RealT>(g.nz,k,k,g.hz);
    const double kx = q1_assembled_stiff1d_coeff<RealT>(g.nx,i,i,g.hx);
    const double ky = q1_assembled_stiff1d_coeff<RealT>(g.ny,j,j,g.hy);
    const double kz = q1_assembled_stiff1d_coeff<RealT>(g.nz,k,k,g.hz);
    const double gx = q1_assembled_adv1d_coeff<RealT>(g.nx,i,i);
    const double gy = q1_assembled_adv1d_coeff<RealT>(g.ny,j,j);
    const double gz = q1_assembled_adv1d_coeff<RealT>(g.nz,k,k);
    const double m3 = mx*my*mz;
    const double k3 = kx*my*mz + mx*ky*mz + mx*my*kz;
    const double c3 = double(convX)*gx*my*mz + double(convY)*mx*gy*mz + double(convZ)*mx*my*gz;
    const double d = double(alphaMass)*m3 + double(betaStiff)*k3 + c3;
    diag[q] = RealT((d > 1e-300 && isfinite(d)) ? d : 1.0);
}

template <class RealT>
__global__ void kernel_q1_scalar_jacobi_update(int n, RealT omega, const RealT* rhs, const RealT* ax, const RealT* diag, RealT* x) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q < n) x[q] += omega * (rhs[q] - ax[q]) / diag[q];
}

template <class RealT>
__global__ void kernel_q1_scalar_restrict_full_weighting(Grid3DCudaLocal gf, Grid3DCudaLocal gc, const RealT* fine, RealT* coarse) {
    const int nnf = gf.n_nodes_host();
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= nnf) return;
    const int fi = q % gf.nx;
    const int tmp = q / gf.nx;
    const int fj = tmp % gf.ny;
    const int fk = tmp / gf.ny;
    int ci[2], cj[2], ck[2];
    double wi[2], wj[2], wk[2];
    int ni=1,nj=1,nk=1;
    if ((fi & 1)==0) { ci[0]=fi/2; wi[0]=1.0; } else { ni=2; ci[0]=fi/2; ci[1]=ci[0]+1; wi[0]=0.5; wi[1]=0.5; }
    if ((fj & 1)==0) { cj[0]=fj/2; wj[0]=1.0; } else { nj=2; cj[0]=fj/2; cj[1]=cj[0]+1; wj[0]=0.5; wj[1]=0.5; }
    if ((fk & 1)==0) { ck[0]=fk/2; wk[0]=1.0; } else { nk=2; ck[0]=fk/2; ck[1]=ck[0]+1; wk[0]=0.5; wk[1]=0.5; }
    const RealT val = fine[q];
    for (int kk=0; kk<nk; ++kk) for (int jj=0; jj<nj; ++jj) for (int ii=0; ii<ni; ++ii) {
        if (ci[ii] < 0 || ci[ii] >= gc.nx || cj[jj] < 0 || cj[jj] >= gc.ny || ck[kk] < 0 || ck[kk] >= gc.nz) continue;
        const int cq = ci[ii] + gc.nx*(cj[jj] + gc.ny*ck[kk]);
        const double wt = 0.125 * wi[ii]*wj[jj]*wk[kk];
        atomicAdd(&coarse[cq], RealT(wt)*val);
    }
}

template <class RealT>
__global__ void kernel_q1_scalar_prolong_add(Grid3DCudaLocal gc, Grid3DCudaLocal gf, const RealT* coarse, RealT* fine) {
    const int nnf = gf.n_nodes_host();
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= nnf) return;
    const int fi = q % gf.nx;
    const int tmp = q / gf.nx;
    const int fj = tmp % gf.ny;
    const int fk = tmp / gf.ny;
    int ci[2], cj[2], ck[2];
    double wi[2], wj[2], wk[2];
    int ni=1,nj=1,nk=1;
    if ((fi & 1)==0) { ci[0]=fi/2; wi[0]=1.0; } else { ni=2; ci[0]=fi/2; ci[1]=ci[0]+1; wi[0]=0.5; wi[1]=0.5; }
    if ((fj & 1)==0) { cj[0]=fj/2; wj[0]=1.0; } else { nj=2; cj[0]=fj/2; cj[1]=cj[0]+1; wj[0]=0.5; wj[1]=0.5; }
    if ((fk & 1)==0) { ck[0]=fk/2; wk[0]=1.0; } else { nk=2; ck[0]=fk/2; ck[1]=ck[0]+1; wk[0]=0.5; wk[1]=0.5; }
    double acc = 0.0;
    for (int kk=0; kk<nk; ++kk) for (int jj=0; jj<nj; ++jj) for (int ii=0; ii<ni; ++ii) {
        if (ci[ii] < 0 || ci[ii] >= gc.nx || cj[jj] < 0 || cj[jj] >= gc.ny || ck[kk] < 0 || ck[kk] >= gc.nz) continue;
        const int cq = ci[ii] + gc.nx*(cj[jj] + gc.ny*ck[kk]);
        acc += wi[ii]*wj[jj]*wk[kk] * double(coarse[cq]);
    }
    fine[q] += RealT(acc);
}

template <class RealT>
class StructuredQ1ScalarFeGmgPreconditioner {
public:
    StructuredQ1ScalarFeGmgPreconditioner(const memoirs::structured::StructuredGrid3D& grid,
                                          int preSweeps,
                                          int postSweeps,
                                          RealT omega,
                                          int verbose)
        : pre_(preSweeps), post_(postSweeps), omega_(omega), verbose_(verbose) {
        int nElem = grid.nx - 1;
        while (true) {
            const int nodes = nElem + 1;
            levels_.emplace_back();
            Level& lev = levels_.back();
            lev.grid = memoirs::structured::StructuredGrid3D(nodes,nodes,nodes);
            lev.dg = to_cuda_grid_local(lev.grid);
            const int n = lev.grid.n_nodes();
            lev.x.resize(n); lev.rhs.resize(n); lev.ax.resize(n); lev.res.resize(n); lev.diag.resize(n);
            if (nElem <= 2 || (nElem % 2) != 0) break;
            nElem /= 2;
        }
        const int nc = levels_.back().grid.n_nodes();
        if (nc > 512) throw std::runtime_error("Q1 scalar FE GMG coarse grid too large");
        d_lu_.resize(static_cast<std::size_t>(nc)*static_cast<std::size_t>(nc));
        d_piv_.resize(static_cast<std::size_t>(nc));
        if (verbose_) std::cout << "q1 scalar FE gmg levels=" << levels_.size() << " coarseDofs=" << nc << "\n";
    }
    void rebuild(RealT alphaMassFine, RealT betaStiffFine, RealT convXFine, RealT convYFine, RealT convZFine, int betaScalesH2, int strongBoundary, int strongPin) {
        alphaMassFine_ = alphaMassFine; betaStiffFine_ = betaStiffFine; convXFine_ = convXFine; convYFine_ = convYFine; convZFine_ = convZFine; betaScalesH2_ = betaScalesH2; strongBoundary_ = strongBoundary; strongPin_ = strongPin;
        const int block = 256; const double hf = levels_[0].grid.hx();
        for (auto& lev : levels_) {
            const double ratio = lev.grid.hx()/hf;
            lev.alpha = alphaMassFine_;
            lev.beta = RealT(double(betaStiffFine_) * (betaScalesH2_ ? ratio*ratio : 1.0));
            lev.convX = convXFine_;
            lev.convY = convYFine_;
            lev.convZ = convZFine_;
            kernel_q1_scalar_fe_diag_stencil<RealT><<<div_up_local(lev.grid.n_nodes(),block),block>>>(lev.dg,lev.alpha,lev.beta,lev.convX,lev.convY,lev.convZ,strongBoundary_,strongPin_,lev.diag.data()); MEMOIRS_CUDA_KERNEL_CHECK();
        }
        build_coarse_lu();
    }
    void apply(const RealT* r, RealT* z) {
        Level& f = levels_[0]; const int n = f.grid.n_nodes();
        MEMOIRS_CUDA_CHECK(cudaMemcpy(f.rhs.data(), r, static_cast<std::size_t>(n)*sizeof(RealT), cudaMemcpyDeviceToDevice));
        f.x.zero(); vcycle(0);
        MEMOIRS_CUDA_CHECK(cudaMemcpy(z, f.x.data(), static_cast<std::size_t>(n)*sizeof(RealT), cudaMemcpyDeviceToDevice));
    }
    void apply_operator_on_fine(const RealT* x, RealT* y) {
        apply_level(0, x, y);
    }
private:
    struct Level { memoirs::structured::StructuredGrid3D grid; Grid3DCudaLocal dg; RealT alpha=RealT(0), beta=RealT(0), convX=RealT(0), convY=RealT(0), convZ=RealT(0); DeviceBuffer<RealT> x,rhs,ax,res,diag; };
    void apply_level(int l, const RealT* x, RealT* y) { Level& lev=levels_[static_cast<std::size_t>(l)]; const int block=256; kernel_q1_scalar_fe_apply_stencil<RealT><<<div_up_local(lev.grid.n_nodes(),block),block>>>(lev.dg,lev.alpha,lev.beta,lev.convX,lev.convY,lev.convZ,strongBoundary_,strongPin_,x,y); MEMOIRS_CUDA_KERNEL_CHECK(); }
    void smooth(int l, int sweeps) { Level& lev=levels_[static_cast<std::size_t>(l)]; const int n=lev.grid.n_nodes(); const int block=256; for (int s=0;s<sweeps;++s) { apply_level(l,lev.x.data(),lev.ax.data()); kernel_q1_scalar_jacobi_update<RealT><<<div_up_local(n,block),block>>>(n,omega_,lev.rhs.data(),lev.ax.data(),lev.diag.data(),lev.x.data()); MEMOIRS_CUDA_KERNEL_CHECK(); } }
    void residual(int l) { Level& lev=levels_[static_cast<std::size_t>(l)]; const int n=lev.grid.n_nodes(); const int block=256; apply_level(l,lev.x.data(),lev.ax.data()); kernel_subtract_local<RealT><<<div_up_local(n,block),block>>>(n,lev.rhs.data(),lev.ax.data(),lev.res.data()); MEMOIRS_CUDA_KERNEL_CHECK(); }
    void vcycle(int l) { if (l == int(levels_.size())-1) { solve_coarse(levels_[l].rhs.data(),levels_[l].x.data()); return; } Level& fine=levels_[static_cast<std::size_t>(l)]; Level& coarse=levels_[static_cast<std::size_t>(l+1)]; const int block=256; smooth(l,pre_); residual(l); kernel_zero_local<RealT><<<div_up_local(coarse.grid.n_nodes(),block),block>>>(coarse.grid.n_nodes(),coarse.rhs.data()); MEMOIRS_CUDA_KERNEL_CHECK(); kernel_zero_local<RealT><<<div_up_local(coarse.grid.n_nodes(),block),block>>>(coarse.grid.n_nodes(),coarse.x.data()); MEMOIRS_CUDA_KERNEL_CHECK(); kernel_q1_scalar_restrict_full_weighting<RealT><<<div_up_local(fine.grid.n_nodes(),block),block>>>(fine.dg,coarse.dg,fine.res.data(),coarse.rhs.data()); MEMOIRS_CUDA_KERNEL_CHECK(); vcycle(l+1); kernel_q1_scalar_prolong_add<RealT><<<div_up_local(fine.grid.n_nodes(),block),block>>>(coarse.dg,fine.dg,coarse.x.data(),fine.x.data()); MEMOIRS_CUDA_KERNEL_CHECK(); smooth(l,post_); }
    void build_coarse_lu() { Level& c=levels_.back(); const int nc=c.grid.n_nodes(); const int block=256; std::vector<RealT> hA(static_cast<std::size_t>(nc)*static_cast<std::size_t>(nc)); std::vector<RealT> hy(static_cast<std::size_t>(nc)); for (int col=0; col<nc; ++col) { kernel_set_basis_vector_local<RealT><<<div_up_local(nc,block),block>>>(nc,col,c.x.data()); MEMOIRS_CUDA_KERNEL_CHECK(); apply_level(int(levels_.size())-1,c.x.data(),c.ax.data()); c.ax.copy_to_host(hy.data(),hy.size()); for (int row=0; row<nc; ++row) hA[static_cast<std::size_t>(row)*nc + col] = hy[static_cast<std::size_t>(row)]; } hPiv_.assign(static_cast<std::size_t>(nc),0); for (int k=0;k<nc;++k) { int piv=k; double best=std::abs(double(hA[static_cast<std::size_t>(k)*nc+k])); for (int i=k+1;i<nc;++i) { const double v=std::abs(double(hA[static_cast<std::size_t>(i)*nc+k])); if (v>best){best=v;piv=i;} } if (!(best>0.0)||!std::isfinite(best)) throw std::runtime_error("Q1 scalar FE GMG coarse LU singular/nonfinite pivot"); hPiv_[static_cast<std::size_t>(k)] = piv; if (piv!=k) for (int j=0;j<nc;++j) std::swap(hA[static_cast<std::size_t>(k)*nc+j], hA[static_cast<std::size_t>(piv)*nc+j]); const RealT akk=hA[static_cast<std::size_t>(k)*nc+k]; for (int i=k+1;i<nc;++i) { hA[static_cast<std::size_t>(i)*nc+k]/=akk; const RealT lik=hA[static_cast<std::size_t>(i)*nc+k]; for (int j=k+1;j<nc;++j) hA[static_cast<std::size_t>(i)*nc+j]-=lik*hA[static_cast<std::size_t>(k)*nc+j]; } } d_lu_.copy_from_host(hA.data(),hA.size()); d_piv_.copy_from_host(hPiv_.data(),hPiv_.size()); }
    void solve_coarse(const RealT* b, RealT* x) { const int nc=levels_.back().grid.n_nodes(); kernel_dense_lu_solve_small_device<RealT><<<1,1>>>(nc,d_lu_.data(),d_piv_.data(),b,x); MEMOIRS_CUDA_KERNEL_CHECK(); }
    std::vector<Level> levels_; int pre_=2, post_=2, verbose_=0; RealT omega_=RealT(0.7), alphaMassFine_=RealT(0), betaStiffFine_=RealT(0), convXFine_=RealT(0), convYFine_=RealT(0), convZFine_=RealT(0); int betaScalesH2_=0, strongBoundary_=0, strongPin_=0; DeviceBuffer<RealT> d_lu_; DeviceBuffer<int> d_piv_; std::vector<int> hPiv_;
};

template <class RealT>
class StructuredQ1Q1BlockScalarGmgPreconditioner {
public:
    StructuredQ1Q1BlockScalarGmgPreconditioner(const memoirs::structured::StructuredGrid3D& grid, int preSweeps, int postSweeps, RealT omega, int verbose, const std::string& pressureMode, int includeConvection, int includeSupg, RealT convScale)
        : grid_(grid), pressureMode_(pressureMode), includeConvection_(includeConvection), includeSupg_(includeSupg), convScale_(convScale), ux_(grid,preSweeps,postSweeps,omega,verbose), uy_(grid,preSweeps,postSweeps,omega,0), uz_(grid,preSweeps,postSweeps,omega,0), pp_(grid,preSweeps,postSweeps,omega,0) {}
    void rebuild(RealT nu, RealT tau, RealT massCoeff, RealT advScale, int supgEnabled, RealT supgTauScale, const RealT* beta) {
        (void)beta;
        RealT velStiff = nu;
        if (includeSupg_ && supgEnabled) {
            // Cheap constant-coefficient SUPG surrogate: tau*(a.grad v)*(a.grad u) -> tau*|a|^2*K isotropic.
            velStiff = velStiff + RealT(double(tau) * double(supgTauScale) * double(advScale) * double(advScale));
        }
        const RealT cx = includeConvection_ ? RealT(double(convScale_) * double(advScale)) : RealT(0);
        const RealT cy = includeConvection_ ? RealT(double(convScale_) * double(advScale)) : RealT(0);
        const RealT cz = includeConvection_ ? RealT(double(convScale_) * double(advScale)) : RealT(0);
        ux_.rebuild(massCoeff,velStiff,cx,cy,cz,0,1,0);
        uy_.rebuild(massCoeff,velStiff,cx,cy,cz,0,1,0);
        uz_.rebuild(massCoeff,velStiff,cx,cy,cz,0,1,0);
        RealT pMass = RealT(0), pStiff = RealT(0); int pStiffScalesH2 = 0;
        if (pressureMode_ == "mass" || pressureMode_ == "mass_pspg") pMass = RealT(1);
        if (pressureMode_ == "pspg" || pressureMode_ == "mass_pspg") { pStiff = tau; pStiffScalesH2 = 1; }
        pp_.rebuild(pMass,pStiff,RealT(0),RealT(0),RealT(0),pStiffScalesH2,0,1);
    }
    void apply(const RealT* r, RealT* z) { const int nn=grid_.n_nodes(); ux_.apply(r,z); uy_.apply(r+nn,z+nn); uz_.apply(r+2*nn,z+2*nn); pp_.apply(r+3*nn,z+3*nn); }
private:
    memoirs::structured::StructuredGrid3D grid_;
    std::string pressureMode_ = "pspg";
    int includeConvection_ = 0;
    int includeSupg_ = 0;
    RealT convScale_ = RealT(1);
    StructuredQ1ScalarFeGmgPreconditioner<RealT> ux_, uy_, uz_, pp_;
};



template <class RealT>
__global__ void kernel_q1_velocity_full_apply_component(Grid3DCudaLocal g,
                                                        RealT nu,
                                                        RealT tau,
                                                        RealT massCoeff,
                                                        RealT advScale,
                                                        int supgEnabled,
                                                        RealT supgTauScale,
                                                        const RealT* beta,
                                                        const RealT* u,
                                                        RealT* y) {
    const int nn = g.n_nodes_host();
    const int exN = g.nx - 1;
    const int eyN = g.ny - 1;
    const int ezN = g.nz - 1;
    const int nElem = exN * eyN * ezN;
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nElem) return;

    const int ex = e % exN;
    const int tmp = e / exN;
    const int ey = tmp % eyN;
    const int ez = tmp / eyN;

    const double qp[4] = {-0.86113631159405257522, -0.33998104358485626480,
                           0.33998104358485626480,  0.86113631159405257522};
    const double qw[4] = { 0.34785484513745385737,  0.65214515486254614263,
                           0.65214515486254614263,  0.34785484513745385737};
    const double jac = g.hx*g.hy*g.hz/8.0;

    int gid[8]; bool bnd[8];
    double uu[8], bx[8], by[8], bz[8];
    for (int a=0; a<8; ++a) {
        const int id = stokes_local_node_id(g, ex, ey, ez, a);
        gid[a] = id;
        const int gi = id % g.nx;
        const int t = id / g.nx;
        const int gj = t % g.ny;
        const int gk = t / g.ny;
        bnd[a] = g_boundary_local(g, gi, gj, gk);
        uu[a] = bnd[a] ? 0.0 : double(u[id]);
        bx[a] = beta ? double(beta[id]) : 0.0;
        by[a] = beta ? double(beta[nn + id]) : 0.0;
        bz[a] = beta ? double(beta[2*nn + id]) : 0.0;
    }

    const double tauSupg = supgEnabled ? double(tau)*double(supgTauScale) : 0.0;
    for (int iq=0; iq<4; ++iq) for (int jq=0; jq<4; ++jq) for (int kq=0; kq<4; ++kq) {
        double N[8], dX[8], dY[8], dZ[8];
        q1_shape_phys<RealT>(g, qp[iq], qp[jq], qp[kq], N, dX, dY, dZ);
        const double wq = qw[iq]*qw[jq]*qw[kq]*jac;

        double uh = 0.0;
        double gradU[3] = {0.0,0.0,0.0};
        double betah[3] = {0.0,0.0,0.0};
        for (int b=0; b<8; ++b) {
            uh += N[b]*uu[b];
            gradU[0] += dX[b]*uu[b];
            gradU[1] += dY[b]*uu[b];
            gradU[2] += dZ[b]*uu[b];
            betah[0] += N[b]*bx[b];
            betah[1] += N[b]*by[b];
            betah[2] += N[b]*bz[b];
        }
        const double ax = double(advScale)*betah[0];
        const double ay = double(advScale)*betah[1];
        const double az = double(advScale)*betah[2];
        const double convU = ax*gradU[0] + ay*gradU[1] + az*gradU[2];
        const double strongR = double(massCoeff)*uh + convU;

        for (int a=0; a<8; ++a) if (!bnd[a]) {
            const double streamNa = ax*dX[a] + ay*dY[a] + az*dZ[a];
            const double val = double(massCoeff)*N[a]*uh
                             + double(nu)*(dX[a]*gradU[0] + dY[a]*gradU[1] + dZ[a]*gradU[2])
                             + N[a]*convU
                             + tauSupg*streamNa*strongR;
            atomicAdd(&y[gid[a]], RealT(wq*val));
        }
    }
}

template <class RealT>
__global__ void kernel_q1_velocity_full_strong_boundary(Grid3DCudaLocal g, const RealT* x, RealT* y) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    const int nn = g.n_nodes_host();
    if (q >= nn) return;
    const int i = q % g.nx;
    const int tmp = q / g.nx;
    const int j = tmp % g.ny;
    const int k = tmp / g.ny;
    if (g_boundary_local(g,i,j,k)) y[q] = x[q];
}

template <class RealT>
__global__ void kernel_q1_velocity_full_diag(Grid3DCudaLocal g,
                                             RealT nu,
                                             RealT tau,
                                             RealT massCoeff,
                                             RealT advScale,
                                             int supgEnabled,
                                             RealT supgTauScale,
                                             const RealT* beta,
                                             RealT* diag) {
    const int nn = g.n_nodes_host();
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= nn) return;
    const int i = q % g.nx;
    const int tmp = q / g.nx;
    const int j = tmp % g.ny;
    const int k = tmp / g.ny;
    if (g_boundary_local(g,i,j,k)) { diag[q] = RealT(1); return; }

    const double qp[4] = {-0.86113631159405257522, -0.33998104358485626480,
                           0.33998104358485626480,  0.86113631159405257522};
    const double qw[4] = { 0.34785484513745385737,  0.65214515486254614263,
                           0.65214515486254614263,  0.34785484513745385737};
    const double tauSupg = supgEnabled ? double(tau)*double(supgTauScale) : 0.0;
    double d = 0.0;
    const int ex0=max(0,i-1), ex1=min(g.nx-2,i);
    const int ey0=max(0,j-1), ey1=min(g.ny-2,j);
    const int ez0=max(0,k-1), ez1=min(g.nz-2,k);
    const double jac = g.hx*g.hy*g.hz/8.0;
    for (int ez=ez0; ez<=ez1; ++ez) for (int ey=ey0; ey<=ey1; ++ey) for (int ex=ex0; ex<=ex1; ++ex) {
        const int axl=i-ex, ayl=j-ey, azl=k-ez;
        const int a=axl + 2*ayl + 4*azl;
        int gid[8]; double bx[8],by[8],bz[8];
        for (int b=0; b<8; ++b) {
            const int id = stokes_local_node_id(g, ex, ey, ez, b);
            gid[b] = id;
            bx[b] = beta ? double(beta[id]) : 0.0;
            by[b] = beta ? double(beta[nn+id]) : 0.0;
            bz[b] = beta ? double(beta[2*nn+id]) : 0.0;
        }
        for (int iq=0;iq<4;++iq) for (int jq=0;jq<4;++jq) for (int kq=0;kq<4;++kq) {
            double N[8],dX[8],dY[8],dZ[8];
            q1_shape_phys<RealT>(g,qp[iq],qp[jq],qp[kq],N,dX,dY,dZ);
            double betah[3]={0,0,0};
            for (int b=0; b<8; ++b) {
                betah[0] += N[b]*bx[b];
                betah[1] += N[b]*by[b];
                betah[2] += N[b]*bz[b];
            }
            const double vx = double(advScale)*betah[0];
            const double vy = double(advScale)*betah[1];
            const double vz = double(advScale)*betah[2];
            const double streamNa = vx*dX[a] + vy*dY[a] + vz*dZ[a];
            const double val = double(massCoeff)*N[a]*N[a]
                             + double(nu)*(dX[a]*dX[a] + dY[a]*dY[a] + dZ[a]*dZ[a])
                             + N[a]*streamNa
                             + tauSupg*streamNa*(double(massCoeff)*N[a] + streamNa);
            d += qw[iq]*qw[jq]*qw[kq]*jac*val;
        }
    }
    if (!(d > 1e-300) || !isfinite(d)) d = 1.0;
    diag[q] = RealT(d);
}

template <class RealT>
__global__ void kernel_apply_diag_inverse_local(int n, const RealT* r, const RealT* diag, RealT* z) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q < n) z[q] = r[q] / diag[q];
}

template <class RealT>
class StructuredQ1Q1PcdGmgPreconditioner {
public:
    StructuredQ1Q1PcdGmgPreconditioner(const memoirs::structured::StructuredGrid3D& grid,
                                        int preSweeps,
                                        int postSweeps,
                                        RealT omega,
                                        int verbose,
                                        int kpMaxit,
                                        RealT kpTol,
                                        RealT pcdSign,
                                        const std::string& velSolver,
                                        int velIters)
        : grid_(grid),
          dg_(to_cuda_grid_local(grid)),
          kpMaxit_(kpMaxit),
          kpTol_(kpTol),
          pcdSign_(pcdSign),
          velSolver_(velSolver),
          velIters_(velIters),
          ux_(grid,preSweeps,postSweeps,omega,verbose),
          uy_(grid,preSweeps,postSweeps,omega,0),
          uz_(grid,preSweeps,postSweeps,omega,0),
          kp_(grid,preSweeps,postSweeps,omega,0) {
        const int nn = grid_.n_nodes();
        ptmp1_.resize(nn); ptmp2_.resize(nn); ptmp3_.resize(nn);
        kpR_.resize(nn); kpZ_.resize(nn); kpP_.resize(nn); kpAp_.resize(nn);
        velDiag_.resize(nn); velR_.resize(nn); velW_.resize(nn); velAx_.resize(nn);
        if (velSolver_ == "gmres_diag") {
            const int m = std::max(1, velIters_);
            velV_.reserve(static_cast<std::size_t>(m + 1));
            velZ_.reserve(static_cast<std::size_t>(m));
            for (int i=0; i<m+1; ++i) velV_.emplace_back(static_cast<std::size_t>(nn));
            for (int i=0; i<m; ++i) velZ_.emplace_back(static_cast<std::size_t>(nn));
        }
        if (verbose) std::cout << "q1q1 pcd full pressure buffers=" << nn << " kpMaxit=" << kpMaxit_ << " kpTol=" << double(kpTol_) << " sign=" << double(pcdSign_) << " velSolver=" << velSolver_ << " velIters=" << velIters_ << "\n";
    }

    void rebuild(RealT nu, RealT tau, RealT massCoeff, RealT advScale, int supgEnabled, RealT supgTauScale, const RealT* beta) {
        nu_ = nu;
        tau_ = tau;
        massCoeff_ = massCoeff;
        advScale_ = advScale;
        supgEnabled_ = supgEnabled;
        supgTauScale_ = supgTauScale;
        beta_ = beta;

        // F^{-1}: default is one cheap scalar GMG V-cycle for each velocity component.
        // Optional pcdVelSolver=gmres_diag uses a fixed inner GMRES solve on the full
        // mass+diffusion+Picard-convection+SUPG velocity block, preconditioned by its diagonal.
        ux_.rebuild(massCoeff_,nu_,RealT(0),RealT(0),RealT(0),0,1,0);
        uy_.rebuild(massCoeff_,nu_,RealT(0),RealT(0),RealT(0),0,1,0);
        uz_.rebuild(massCoeff_,nu_,RealT(0),RealT(0),RealT(0),0,1,0);
        if (velSolver_ == "gmres_diag") {
            const int block = 256;
            kernel_q1_velocity_full_diag<RealT><<<div_up_local(grid_.n_nodes(),block),block>>>(dg_,nu_,tau_,massCoeff_,advScale_,supgEnabled_,supgTauScale_,beta_,velDiag_.data());
            MEMOIRS_CUDA_KERNEL_CHECK();
        }

        // Kp^{-1}: pressure Laplacian GMG used as the preconditioner inside a real pressure PCG solve.
        kp_.rebuild(RealT(0),RealT(1),RealT(0),RealT(0),RealT(0),0,0,1);
    }

    void apply(const RealT* r, RealT* z) {
        const int nn = grid_.n_nodes();
        const int block = 256;

        // Lower-triangular PCD application:
        //   y_u = F^{-1} r_u
        //   rhs_p = r_p - B y_u
        //   y_p = sign * M_p^{-1} A_p K_p^{-1} rhs_p
        // where Kp^{-1} is solved by pressure PCG+GMG, Ap is the full unsteady pressure
        // convection-diffusion/SUPG operator, and Mp^{-1} is lumped diagonal.
        apply_velocity_inverse_component(r, z);
        apply_velocity_inverse_component(r+nn, z+nn);
        apply_velocity_inverse_component(r+2*nn, z+2*nn);

        kernel_zero_local<RealT><<<div_up_local(nn,block),block>>>(nn,ptmp1_.data()); MEMOIRS_CUDA_KERNEL_CHECK();
        const int nElem = (grid_.nx - 1) * (grid_.ny - 1) * (grid_.nz - 1);
        kernel_q1q1_pcd_apply_B_velocity<RealT><<<div_up_local(nElem,block),block>>>(dg_,tau_,massCoeff_,advScale_,beta_,z,ptmp1_.data()); MEMOIRS_CUDA_KERNEL_CHECK();
        kernel_subtract_local<RealT><<<div_up_local(nn,block),block>>>(nn,r+3*nn,ptmp1_.data(),ptmp1_.data()); MEMOIRS_CUDA_KERNEL_CHECK();
        kernel_set_index_value_local<RealT><<<1,1>>>(0,RealT(0),ptmp1_.data()); MEMOIRS_CUDA_KERNEL_CHECK();

        solve_kp_pcg(ptmp1_.data(), ptmp2_.data());

        kernel_zero_local<RealT><<<div_up_local(nn,block),block>>>(nn,ptmp3_.data()); MEMOIRS_CUDA_KERNEL_CHECK();
        kernel_q1_pressure_ap_full_apply<RealT><<<div_up_local(nElem,block),block>>>(dg_,nu_,tau_,massCoeff_,advScale_,supgEnabled_,supgTauScale_,beta_,ptmp2_.data(),ptmp3_.data()); MEMOIRS_CUDA_KERNEL_CHECK();
        kernel_set_index_value_local<RealT><<<1,1>>>(0,RealT(0),ptmp3_.data()); MEMOIRS_CUDA_KERNEL_CHECK();

        kernel_q1_lumped_mass_inverse_apply<RealT><<<div_up_local(nn,block),block>>>(dg_,ptmp3_.data(),z+3*nn,pcdSign_); MEMOIRS_CUDA_KERNEL_CHECK();
    }

private:
    void apply_velocity_block_component(const RealT* x, RealT* y) {
        const int nn = grid_.n_nodes();
        const int block = 256;
        const int nElem = (grid_.nx - 1) * (grid_.ny - 1) * (grid_.nz - 1);
        kernel_zero_local<RealT><<<div_up_local(nn,block),block>>>(nn,y); MEMOIRS_CUDA_KERNEL_CHECK();
        kernel_q1_velocity_full_apply_component<RealT><<<div_up_local(nElem,block),block>>>(dg_,nu_,tau_,massCoeff_,advScale_,supgEnabled_,supgTauScale_,beta_,x,y); MEMOIRS_CUDA_KERNEL_CHECK();
        kernel_q1_velocity_full_strong_boundary<RealT><<<div_up_local(nn,block),block>>>(dg_,x,y); MEMOIRS_CUDA_KERNEL_CHECK();
    }

    void apply_velocity_diag_inverse(const RealT* r, RealT* z) {
        const int nn = grid_.n_nodes();
        const int block = 256;
        kernel_apply_diag_inverse_local<RealT><<<div_up_local(nn,block),block>>>(nn,r,velDiag_.data(),z); MEMOIRS_CUDA_KERNEL_CHECK();
    }

    void apply_velocity_inverse_component(const RealT* r, RealT* z) {
        if (velSolver_ == "gmres_diag") solve_velocity_gmres_diag(r,z);
        else ux_.apply(r,z);
    }

    void solve_velocity_gmres_diag(const RealT* b, RealT* x) {
        const int nn = grid_.n_nodes();
        const int block = 256;
        const int m = std::max(1, velIters_);
        GpuDotLocal<RealT> dots;
        kernel_zero_local<RealT><<<div_up_local(nn,block),block>>>(nn,x); MEMOIRS_CUDA_KERNEL_CHECK();
        kernel_copy_local<RealT><<<div_up_local(nn,block),block>>>(nn,b,velR_.data()); MEMOIRS_CUDA_KERNEL_CHECK();
        const double beta0 = std::sqrt(dots.dot(nn, velR_.data(), velR_.data()));
        if (!(beta0 > 0.0) || !std::isfinite(beta0)) return;

        if (int(velV_.size()) < m+1 || int(velZ_.size()) < m) {
            apply_velocity_diag_inverse(b,x);
            return;
        }

        kernel_axpby_local<RealT><<<div_up_local(nn,block),block>>>(nn,RealT(1.0/beta0),velR_.data(),RealT(0),velV_[0].data()); MEMOIRS_CUDA_KERNEL_CHECK();

        std::vector<std::vector<double>> H(static_cast<std::size_t>(m+1), std::vector<double>(static_cast<std::size_t>(m),0.0));
        std::vector<double> cs(static_cast<std::size_t>(m),0.0), sn(static_cast<std::size_t>(m),0.0), g(static_cast<std::size_t>(m+1),0.0), y(static_cast<std::size_t>(m),0.0);
        g[0] = beta0;
        int lastJ = -1;
        for (int j=0; j<m; ++j) {
            lastJ = j;
            apply_velocity_diag_inverse(velV_[static_cast<std::size_t>(j)].data(), velZ_[static_cast<std::size_t>(j)].data());
            apply_velocity_block_component(velZ_[static_cast<std::size_t>(j)].data(), velW_.data());

            for (int i=0; i<=j; ++i) {
                H[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = dots.dot(nn, velV_[static_cast<std::size_t>(i)].data(), velW_.data());
                kernel_axpby_local<RealT><<<div_up_local(nn,block),block>>>(nn,RealT(-H[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)]),velV_[static_cast<std::size_t>(i)].data(),RealT(1),velW_.data()); MEMOIRS_CUDA_KERNEL_CHECK();
            }
            H[static_cast<std::size_t>(j+1)][static_cast<std::size_t>(j)] = std::sqrt(dots.dot(nn, velW_.data(), velW_.data()));
            const double hnext = H[static_cast<std::size_t>(j+1)][static_cast<std::size_t>(j)];
            if (std::isfinite(hnext) && hnext > 0.0 && j+1 < m+1) {
                kernel_axpby_local<RealT><<<div_up_local(nn,block),block>>>(nn,RealT(1.0/hnext),velW_.data(),RealT(0),velV_[static_cast<std::size_t>(j+1)].data()); MEMOIRS_CUDA_KERNEL_CHECK();
            } else {
                H[static_cast<std::size_t>(j+1)][static_cast<std::size_t>(j)] = 0.0;
            }

            for (int i=0; i<j; ++i) {
                const double hij = H[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
                const double hi1j = H[static_cast<std::size_t>(i+1)][static_cast<std::size_t>(j)];
                H[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = cs[static_cast<std::size_t>(i)]*hij + sn[static_cast<std::size_t>(i)]*hi1j;
                H[static_cast<std::size_t>(i+1)][static_cast<std::size_t>(j)] = -sn[static_cast<std::size_t>(i)]*hij + cs[static_cast<std::size_t>(i)]*hi1j;
            }
            const double h0 = H[static_cast<std::size_t>(j)][static_cast<std::size_t>(j)];
            const double h1 = H[static_cast<std::size_t>(j+1)][static_cast<std::size_t>(j)];
            const double denom = std::hypot(h0,h1);
            if (!(std::isfinite(denom)) || denom == 0.0) { lastJ = j-1; break; }
            cs[static_cast<std::size_t>(j)] = h0/denom;
            sn[static_cast<std::size_t>(j)] = h1/denom;
            H[static_cast<std::size_t>(j)][static_cast<std::size_t>(j)] = cs[static_cast<std::size_t>(j)]*h0 + sn[static_cast<std::size_t>(j)]*h1;
            H[static_cast<std::size_t>(j+1)][static_cast<std::size_t>(j)] = 0.0;
            const double gj = g[static_cast<std::size_t>(j)];
            const double gj1 = g[static_cast<std::size_t>(j+1)];
            g[static_cast<std::size_t>(j)] = cs[static_cast<std::size_t>(j)]*gj + sn[static_cast<std::size_t>(j)]*gj1;
            g[static_cast<std::size_t>(j+1)] = -sn[static_cast<std::size_t>(j)]*gj + cs[static_cast<std::size_t>(j)]*gj1;
        }
        const int k = lastJ;
        if (k < 0) { apply_velocity_diag_inverse(b,x); return; }
        for (int i=0; i<=k; ++i) y[static_cast<std::size_t>(i)] = g[static_cast<std::size_t>(i)];
        for (int i=k; i>=0; --i) {
            double sum = y[static_cast<std::size_t>(i)];
            for (int j=i+1; j<=k; ++j) sum -= H[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] * y[static_cast<std::size_t>(j)];
            const double diag = H[static_cast<std::size_t>(i)][static_cast<std::size_t>(i)];
            if (!(std::isfinite(diag)) || std::abs(diag) == 0.0) { apply_velocity_diag_inverse(b,x); return; }
            y[static_cast<std::size_t>(i)] = sum / diag;
        }
        kernel_zero_local<RealT><<<div_up_local(nn,block),block>>>(nn,x); MEMOIRS_CUDA_KERNEL_CHECK();
        for (int i=0; i<=k; ++i) {
            kernel_axpby_local<RealT><<<div_up_local(nn,block),block>>>(nn,RealT(y[static_cast<std::size_t>(i)]),velZ_[static_cast<std::size_t>(i)].data(),RealT(1),x); MEMOIRS_CUDA_KERNEL_CHECK();
        }
    }

    void solve_kp_pcg(const RealT* b, RealT* x) {
        const int nn = grid_.n_nodes();
        const int block = 256;
        GpuDotLocal<RealT> dots;
        kernel_zero_local<RealT><<<div_up_local(nn,block),block>>>(nn,x); MEMOIRS_CUDA_KERNEL_CHECK();
        kernel_copy_local<RealT><<<div_up_local(nn,block),block>>>(nn,b,kpR_.data()); MEMOIRS_CUDA_KERNEL_CHECK();
        kernel_set_index_value_local<RealT><<<1,1>>>(0,RealT(0),kpR_.data()); MEMOIRS_CUDA_KERNEL_CHECK();

        const double bnorm = std::max(std::sqrt(dots.dot(nn,b,b)), 1.0);
        kp_.apply(kpR_.data(), kpZ_.data());
        kernel_copy_local<RealT><<<div_up_local(nn,block),block>>>(nn,kpZ_.data(),kpP_.data()); MEMOIRS_CUDA_KERNEL_CHECK();
        double rz = dots.dot(nn,kpR_.data(),kpZ_.data());
        if (!(rz > 0.0) || !std::isfinite(rz)) {
            kernel_copy_local<RealT><<<div_up_local(nn,block),block>>>(nn,kpZ_.data(),x); MEMOIRS_CUDA_KERNEL_CHECK();
            return;
        }
        for (int it=0; it<kpMaxit_; ++it) {
            kp_.apply_operator_on_fine(kpP_.data(), kpAp_.data());
            const double pAp = dots.dot(nn,kpP_.data(),kpAp_.data());
            if (!(pAp > 0.0) || !std::isfinite(pAp)) break;
            const double alpha = rz / pAp;
            kernel_axpby_local<RealT><<<div_up_local(nn,block),block>>>(nn,RealT(alpha),kpP_.data(),RealT(1),x); MEMOIRS_CUDA_KERNEL_CHECK();
            kernel_axpby_local<RealT><<<div_up_local(nn,block),block>>>(nn,RealT(-alpha),kpAp_.data(),RealT(1),kpR_.data()); MEMOIRS_CUDA_KERNEL_CHECK();
            kernel_set_index_value_local<RealT><<<1,1>>>(0,RealT(0),kpR_.data()); MEMOIRS_CUDA_KERNEL_CHECK();
            const double rnorm = std::sqrt(dots.dot(nn,kpR_.data(),kpR_.data()));
            if (rnorm / bnorm <= double(kpTol_)) break;
            kp_.apply(kpR_.data(), kpZ_.data());
            const double rzNew = dots.dot(nn,kpR_.data(),kpZ_.data());
            if (!(rzNew > 0.0) || !std::isfinite(rzNew)) break;
            const double beta = rzNew / rz;
            kernel_combination3_local<RealT><<<div_up_local(nn,block),block>>>(nn,RealT(1),kpZ_.data(),RealT(beta),kpP_.data(),RealT(0),kpP_.data(),kpP_.data()); MEMOIRS_CUDA_KERNEL_CHECK();
            rz = rzNew;
        }
        kernel_set_index_value_local<RealT><<<1,1>>>(0,RealT(0),x); MEMOIRS_CUDA_KERNEL_CHECK();
    }

    memoirs::structured::StructuredGrid3D grid_;
    Grid3DCudaLocal dg_;
    int kpMaxit_ = 30;
    RealT kpTol_ = RealT(1e-8);
    RealT pcdSign_ = RealT(-1);
    std::string velSolver_ = "gmg";
    int velIters_ = 0;
    RealT nu_=RealT(1), tau_=RealT(0), massCoeff_=RealT(1), advScale_=RealT(0), supgTauScale_=RealT(1);
    int supgEnabled_=1;
    const RealT* beta_ = nullptr;
    StructuredQ1ScalarFeGmgPreconditioner<RealT> ux_, uy_, uz_, kp_;
    DeviceBuffer<RealT> ptmp1_, ptmp2_, ptmp3_;
    DeviceBuffer<RealT> kpR_, kpZ_, kpP_, kpAp_;
    DeviceBuffer<RealT> velDiag_, velR_, velW_, velAx_;
    std::vector<DeviceBuffer<RealT>> velV_, velZ_;
};

template <class RealT>
class StructuredQ1Q1NseSelectablePreconditioner {
public:
    StructuredQ1Q1NseSelectablePreconditioner(const memoirs::structured::StructuredGrid3D& grid,
                                               const std::string& mode,
                                               int mgPre,
                                               int mgPost,
                                               RealT mgOmega,
                                               int mgVerbose,
                                               const std::string& blockGmgPressure,
                                               int blockGmgConvection,
                                               int blockGmgSupg,
                                               RealT blockGmgConvScale,
                                               int pcdKpMaxit,
                                               RealT pcdKpTol,
                                               RealT pcdSign,
                                               const std::string& pcdVelSolver,
                                               int pcdVelIters)
        : mode_(mode),
          diagonal_(grid),
          gmg_(grid,mgPre,mgPost,mgOmega,mgVerbose),
          blockGmg_(grid,mgPre,mgPost,mgOmega,mgVerbose,blockGmgPressure,blockGmgConvection,blockGmgSupg,blockGmgConvScale) {
        if (mode_ == "pcd" || mode_ == "pcd_gmg") {
            pcd_.reset(new StructuredQ1Q1PcdGmgPreconditioner<RealT>(grid,mgPre,mgPost,mgOmega,mgVerbose,pcdKpMaxit,pcdKpTol,pcdSign,pcdVelSolver,pcdVelIters));
        }
    }
    void rebuild(RealT nu, RealT tau, RealT massCoeff, RealT advScale, int supgEnabled, RealT supgTauScale, const RealT* beta) {
        if (mode_ == "gmg" || mode_ == "coupled_gmg") gmg_.rebuild(nu,tau,massCoeff,advScale,supgEnabled,supgTauScale,beta);
        else if (mode_ == "block_gmg" || mode_ == "field_gmg") blockGmg_.rebuild(nu,tau,massCoeff,advScale,supgEnabled,supgTauScale,beta);
        else if (mode_ == "pcd" || mode_ == "pcd_gmg") pcd_->rebuild(nu,tau,massCoeff,advScale,supgEnabled,supgTauScale,beta);
        else diagonal_.rebuild(nu,tau,massCoeff,advScale,supgEnabled,supgTauScale,beta);
    }
    void apply(const RealT* r, RealT* z) {
        if (mode_ == "gmg" || mode_ == "coupled_gmg") gmg_.apply(r,z);
        else if (mode_ == "block_gmg" || mode_ == "field_gmg") blockGmg_.apply(r,z);
        else if (mode_ == "pcd" || mode_ == "pcd_gmg") pcd_->apply(r,z);
        else diagonal_.apply(r,z);
    }
private:
    std::string mode_ = "diagonal";
    StructuredQ1Q1NseScalarDiagonalPreconditioner<RealT> diagonal_;
    StructuredQ1Q1CoupledGmgPreconditioner<RealT> gmg_;
    StructuredQ1Q1BlockScalarGmgPreconditioner<RealT> blockGmg_;
    std::unique_ptr<StructuredQ1Q1PcdGmgPreconditioner<RealT>> pcd_;
};

}} // namespace memoirs::gpu

int main(int argc, char** argv) {
    GpuMemoryMonitor gpuMemMon;
    try {
        const Cli cli = parse_cli(argc, argv);
        gpuMemMon.start();
        using Grid = memoirs::structured::StructuredGrid3D;
        using Op = memoirs::gpu::StructuredQ1Q1NsePicardOperator<Real>;
        using Prec = memoirs::gpu::StructuredQ1Q1NseSelectablePreconditioner<Real>;
        const Grid grid(cli.nElem + 1, cli.nElem + 1, cli.nElem + 1);
        const Real tau = memoirs::gpu::stokes_compute_tau_host<Real>(grid, Real(cli.nu), Real(cli.tauScale), cli.tauMode, Real(cli.tauC), Real(cli.dt), Real(cli.advScale));
        const Real massCoeff = Real(1.0 / cli.dt);
        const int nn = grid.n_nodes();
        const int total = 4*nn;
        memoirs::gpu::DeviceBuffer<Real> d_x(total), d_b(total), d_exact(total), d_old(total), d_beta(total);
        d_x.zero(); d_beta.zero();
        Op Ainit(grid, Real(cli.nu), tau, massCoeff, Real(cli.advScale), cli.supg, Real(cli.supgTauScale), d_beta.data());
        Ainit.fill_exact(d_exact.data());
        memoirs::gpu::kernel_copy_local<Real><<<memoirs::gpu::div_up_local(total,256),256>>>(total,d_exact.data(),d_old.data());
        MEMOIRS_CUDA_KERNEL_CHECK();
        if (cli.betaInitial == 1) {
            memoirs::gpu::kernel_copy_local<Real><<<memoirs::gpu::div_up_local(total,256),256>>>(total,d_exact.data(),d_beta.data());
            MEMOIRS_CUDA_KERNEL_CHECK();
        }
        Prec M(grid, cli.prec, cli.mgPre, cli.mgPost, Real(cli.mgOmega), cli.mgVerbose,
               cli.blockGmgPressure, cli.blockGmgConvection, cli.blockGmgSupg, Real(cli.blockGmgConvScale),
               cli.pcdKpMaxit, Real(cli.pcdKpTol), Real(cli.pcdSign),
               cli.pcdVelSolver, cli.pcdVelIters);

        std::cout << "Memoirs CUDA structured Q1/Q1 Navier-Stokes Picard PSPG/SUPG MMS development test\n";
        std::cout << "precision                 = " << kPrecisionName << "\n";
        std::cout << "grid                      = " << grid.label() << " Q1 nodes\n";
        std::cout << "elements                  = " << cli.nElem << "x" << cli.nElem << "x" << cli.nElem << "\n";
        std::cout << "velocityDofsPerComponent  = " << nn << "\n";
        std::cout << "pressureDofs              = " << nn << "\n";
        std::cout << "unknowns                  = " << total << "\n";
        std::cout << "operator                  = monolithic_q1q1_nse_picard_pspg_supg_matrix_free_v1\n";
        std::cout << "mms                       = stationary_divergence_free_sine_velocity_mild_convection\n";
        std::cout << "velocityBC                = homogeneous_dirichlet_all_faces\n";
        std::cout << "pressurePin               = p_node_0_exact_value_0\n";
        std::cout << "timeDiscretization        = backward_euler_single_step_mms_old_exact\n";
        std::cout << "linearization             = Picard_beta_dot_grad_u\n";
        std::cout << "supg                      = " << (cli.supg ? "on" : "off") << "\n";
        std::cout << "supgTauScale              = " << cli.supgTauScale << "\n";
        std::cout << "pspg                      = q1_full_residual_time_convection_pressure_diffusion_weak\n";
        std::cout << "solver                    = " << (cli.solver == "bicgstab" ? "right_bicgstab" : "right_gmres") << "\n";
        if (cli.solver == "gmres") {
            std::cout << "gmresRestart              = " << cli.gmresRestart << "\n";
        }
        std::cout << "rightPreconditioner       = ";
        if (cli.prec == "diagonal") std::cout << "scalar_operator_inverse_diagonal_q1q1_v1\n";
        else if (cli.prec == "block_gmg" || cli.prec == "field_gmg") std::cout << "block_scalar_q1q1_structured_gmg_helmholtz_pspg_v1\n";
        else if (cli.prec == "pcd" || cli.prec == "pcd_gmg") std::cout << "pcd_q1q1_full_ap_kp_pcg_lumped_mass_v2\n";
        else std::cout << "coupled_q1q1_structured_gmg_v1\n";
        if (cli.prec == "gmg" || cli.prec == "coupled_gmg") {
            std::cout << "q1q1GmgHierarchy          = rediscretized_structured_q1q1_to_small_dense_coarse\n";
            std::cout << "q1q1GmgSmoother           = scalar_diagonal_weighted_jacobi\n";
            std::cout << "q1q1GmgCoarseSolve        = dense_lu_device_solve_cpu_factor\n";
            std::cout << "q1q1GmgPrePostOmega       = " << cli.mgPre << "/" << cli.mgPost << "/" << cli.mgOmega << "\n";
        }
        if (cli.prec == "block_gmg" || cli.prec == "field_gmg") {
            std::cout << "q1q1BlockGmgPressure      = " << cli.blockGmgPressure << "\n";
            std::cout << "q1q1BlockGmgConvection    = " << (cli.blockGmgConvection ? "constant_central_surrogate" : "off") << "\n";
            std::cout << "q1q1BlockGmgSupg          = " << (cli.blockGmgSupg ? "streamline_diffusion_surrogate" : "off") << "\n";
            std::cout << "q1q1BlockGmgConvScale     = " << cli.blockGmgConvScale << "\n";
            std::cout << "q1q1GmgPrePostOmega       = " << cli.mgPre << "/" << cli.mgPost << "/" << cli.mgOmega << "\n";
        }
        if (cli.prec == "pcd" || cli.prec == "pcd_gmg") {
            std::cout << "q1q1PcdVelocityInverse    = " << (cli.pcdVelSolver == "gmres_diag" ? "fixed_inner_gmres_diag_full_mass_diffusion_convection_supg_F_inverse" : "scalar_gmg_mass_plus_diffusion_F_inverse") << "\n";
            if (cli.pcdVelSolver == "gmres_diag") std::cout << "q1q1PcdVelocityIters      = " << cli.pcdVelIters << "\n";
            std::cout << "q1q1PcdPressureSchur      = -Mp_inverse_Ap_Kp_inverse\n";
            std::cout << "q1q1PcdAp                 = full_unsteady_pressure_convection_supg_operator\n";
            std::cout << "q1q1PcdB                  = pressure_row_velocity_subblock_with_pspg_time_convection\n";
            std::cout << "q1q1GmgPrePostOmega       = " << cli.mgPre << "/" << cli.mgPost << "/" << cli.mgOmega << "\n";
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
        std::cout << "timing                    = " << (cli.timing ? "on" : "off") << "\n";

        memoirs::gpu::CudaRightBicgstabReportLocal<Real> rep;
        double rhsSeconds=0.0, rebuildSeconds=0.0, linearSeconds=0.0, nonlinearResidualSeconds=0.0, betaUpdateSeconds=0.0;
        int totalLinearIterations=0;
        int picardUsed=0;
        double nlRel=1e300;
        const auto wall0 = std::chrono::high_resolution_clock::now();
        for (int piter=0; piter<cli.maxPicard; ++piter) {
            ++picardUsed;
            Op A(grid, Real(cli.nu), tau, massCoeff, Real(cli.advScale), cli.supg, Real(cli.supgTauScale), d_beta.data());
            auto t0=std::chrono::high_resolution_clock::now();
            if (cli.rhsMode == 1) A.apply(d_exact.data(), d_b.data());
            else A.assemble_rhs_from_old(d_old.data(), d_b.data());
            auto t1=std::chrono::high_resolution_clock::now();
            rhsSeconds += std::chrono::duration<double>(t1-t0).count();
            auto rb0=std::chrono::high_resolution_clock::now();
            M.rebuild(Real(cli.nu), tau, massCoeff, Real(cli.advScale), cli.supg, Real(cli.supgTauScale), d_beta.data());
            auto rb1=std::chrono::high_resolution_clock::now();
            rebuildSeconds += std::chrono::duration<double>(rb1-rb0).count();
            auto l0=std::chrono::high_resolution_clock::now();
            if (cli.solver == "bicgstab") {
                rep = memoirs::gpu::bicgstab_right_prec_cuda_local<Real>(A,M,d_b.data(),d_x.data(),Real(cli.tol),cli.maxit,cli.printEvery);
            } else {
                rep = memoirs::gpu::gmres_right_prec_cuda_local<Real>(A,M,d_b.data(),d_x.data(),Real(cli.tol),cli.maxit,cli.gmresRestart,cli.printEvery);
            }
            auto l1=std::chrono::high_resolution_clock::now();
            linearSeconds += std::chrono::duration<double>(l1-l0).count();
            totalLinearIterations += rep.iterations;
            Op ANL(grid, Real(cli.nu), tau, massCoeff, Real(cli.advScale), cli.supg, Real(cli.supgTauScale), d_x.data());
            auto nr0=std::chrono::high_resolution_clock::now();
            nlRel = memoirs::gpu::device_residual_rel_local(ANL, d_b.data(), d_x.data());
            auto nr1=std::chrono::high_resolution_clock::now();
            nonlinearResidualSeconds += std::chrono::duration<double>(nr1-nr0).count();
            std::cout << "picard                    = " << piter+1
                      << " linearIts=" << rep.iterations
                      << " linearRel=" << rep.finalRelativeResidual
                      << " nonlinearAlgebraicRel=" << std::setprecision(12) << nlRel
                      << " converged=" << rep.converged << "\n";
            if (!rep.converged) break;
            if (nlRel <= cli.picardTol) break;
            auto bu0=std::chrono::high_resolution_clock::now();
            memoirs::gpu::kernel_copy_local<Real><<<memoirs::gpu::div_up_local(total,256),256>>>(total,d_x.data(),d_beta.data());
            MEMOIRS_CUDA_KERNEL_CHECK();
            auto bu1=std::chrono::high_resolution_clock::now();
            betaUpdateSeconds += std::chrono::duration<double>(bu1-bu0).count();
        }
        double checked = 0.0;
        {
            Op Afinal(grid, Real(cli.nu), tau, massCoeff, Real(cli.advScale), cli.supg, Real(cli.supgTauScale), d_x.data());
            checked = memoirs::gpu::device_residual_rel_local(Afinal, d_b.data(), d_x.data());
        }
        std::vector<Real> h_x(total);
        d_x.copy_to_host(h_x.data(), h_x.size());
        const auto err = memoirs::diagnostics::structured_q1_l2_error_stokes_mms<Real>(grid, h_x);
        const auto wall1 = std::chrono::high_resolution_clock::now();
        const double wallSeconds = std::chrono::duration<double>(wall1-wall0).count();
        std::cout << "--------------- q1q1 nse picard pspg/supg solve report ---------------\n";
        std::cout << "picardIterations          = " << picardUsed << "\n";
        std::cout << "linearIterationsLast      = " << rep.iterations << "\n";
        std::cout << "linearConvergedLast       = " << rep.converged << "\n";
        std::cout << "breakdown                 = " << rep.breakdown << "\n";
        std::cout << "breakdownCode             = " << rep.breakdownCode << "\n";
        std::cout << "initialResidualLast       = " << std::setprecision(17) << rep.initialResidual << "\n";
        std::cout << "finalResidualLast         = " << rep.finalResidual << "\n";
        std::cout << "finalRelativeResidualLast = " << rep.finalRelativeResidual << "\n";
        std::cout << "nonlinearAlgebraicRelativeResidual = " << checked << "\n";
        std::cout << "velocityL2Quadrature      = " << err.velocityL2 << "\n";
        std::cout << "pressureL2Quadrature      = " << err.pressureL2 << "\n";
        std::cout << "pressureMeanShift         = " << err.pressureMeanShift << "\n";
        std::cout << "pressureL2MeanShiftedQuadrature = " << err.pressureL2MeanShifted << "\n";
        std::cout << "------------------------------------------------------------------\n";
        std::cout << "totalLinearIterations     = " << totalLinearIterations << "\n";
        std::cout << "solveSeconds              = " << linearSeconds / std::max(1,picardUsed) << "\n";
        if (cli.timing) {
            std::cout << "timingRhsSeconds          = " << rhsSeconds << "\n";
            std::cout << "timingRebuildSeconds      = " << rebuildSeconds << "\n";
            std::cout << "timingLinearSeconds       = " << linearSeconds << "\n";
            std::cout << "timingNonlinearResidualSeconds = " << nonlinearResidualSeconds << "\n";
            std::cout << "timingBetaUpdateSeconds   = " << betaUpdateSeconds << "\n";
            std::cout << "timingWallSeconds         = " << wallSeconds << "\n";
        }
        gpuMemMon.stop();
        return (rep.converged && !rep.breakdown) ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
