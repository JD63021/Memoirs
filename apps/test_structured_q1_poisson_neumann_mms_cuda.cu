#include "memoirs/diagnostics/GpuMemoryMonitor.hpp"
#include "memoirs/diagnostics/CudaTiming.hpp"
#include "memoirs/gpu/StructuredGmgCuda.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
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
    int nElem = 32;
    int maxit = 200;
    double tol = 1e-10;
    int pre = 2;
    int post = 2;
    double omega = 0.70;
    int coarseMaxDofs = 64;
    int maxLevels = 32;
    int verbose = 1;
    int printEvery = 10;
    int repeats = 1;
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
        std::string a(argv[i]);

        if (a == "-n" || a == "-nElem") c.nElem = arg_int(i, argc, argv);
        else if (a == "-maxit") c.maxit = arg_int(i, argc, argv);
        else if (a == "-tol") c.tol = arg_double(i, argc, argv);
        else if (a == "-pre") c.pre = arg_int(i, argc, argv);
        else if (a == "-post") c.post = arg_int(i, argc, argv);
        else if (a == "-omega") c.omega = arg_double(i, argc, argv);
        else if (a == "-coarseMaxDofs") c.coarseMaxDofs = arg_int(i, argc, argv);
        else if (a == "-maxLevels") c.maxLevels = arg_int(i, argc, argv);
        else if (a == "-verbose") c.verbose = arg_int(i, argc, argv);
        else if (a == "-printEvery") c.printEvery = arg_int(i, argc, argv);
        else if (a == "-repeats") c.repeats = arg_int(i, argc, argv);
        else throw std::runtime_error("unknown argument: " + a);
    }

    if (c.nElem < 2) throw std::runtime_error("-n must be >= 2");
    if (c.repeats < 1) throw std::runtime_error("-repeats must be >= 1");

    return c;
}

} // namespace

namespace memoirs {
namespace gpu {

template <class Real>
__device__ inline Real local_ke_q1(const Grid3DCuda g,
                                   int ra, int ca) {
    const int rix = ra & 1;
    const int riy = (ra >> 1) & 1;
    const int riz = (ra >> 2) & 1;

    const int cix = ca & 1;
    const int ciy = (ca >> 1) & 1;
    const int ciz = (ca >> 2) & 1;

    const Real M[2][2] = {
        {Real(2.0 / 3.0), Real(1.0 / 3.0)},
        {Real(1.0 / 3.0), Real(2.0 / 3.0)}
    };

    const Real K[2][2] = {
        {Real(0.5), Real(-0.5)},
        {Real(-0.5), Real(0.5)}
    };

    const Real hx = Real(g.hx);
    const Real hy = Real(g.hy);
    const Real hz = Real(g.hz);

    const Real cx = hy * hz / (Real(2) * hx);
    const Real cy = hx * hz / (Real(2) * hy);
    const Real cz = hx * hy / (Real(2) * hz);

    return
        cx * K[rix][cix] * M[riy][ciy] * M[riz][ciz] +
        cy * M[rix][cix] * K[riy][ciy] * M[riz][ciz] +
        cz * M[rix][cix] * M[riy][ciy] * K[riz][ciz];
}

__device__ inline int local_node_id(const Grid3DCuda g,
                                    int ex,
                                    int ey,
                                    int ez,
                                    int a) {
    const int ix = a & 1;
    const int iy = (a >> 1) & 1;
    const int iz = (a >> 2) & 1;
    return g_id(g, ex + ix, ey + iy, ez + iz);
}

template <class Real>
__global__ void kernel_fill_neumann_cos_exact(Grid3DCuda g, Real* u) {
    const int n = g.n_nodes_host();
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= n) return;

    const int i = q % g.nx;
    const int tmp = q / g.nx;
    const int j = tmp % g.ny;
    const int k = tmp / g.ny;

    const double pi = 3.141592653589793238462643383279502884;
    const double x = double(i) * g.hx;
    const double y = double(j) * g.hy;
    const double z = double(k) * g.hz;

    u[q] = Real(cos(pi * x) * cos(pi * y) * cos(pi * z));
}

template <class Real>
__global__ void kernel_apply_neumann_pinned_elements(Grid3DCuda g,
                                                     const Real* x,
                                                     Real* y) {
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

    const int pin = 0;

    for (int a = 0; a < 8; ++a) {
        const int row = local_node_id(g, ex, ey, ez, a);
        if (row == pin) continue;

        Real sum = Real(0);

        for (int b = 0; b < 8; ++b) {
            const int col = local_node_id(g, ex, ey, ez, b);
            if (col == pin) continue;

            sum += local_ke_q1<Real>(g, a, b) * x[col];
        }

        atomicAdd(&y[row], sum);
    }
}

template <class Real>
__global__ void kernel_set_pin_row(int pin, const Real* x, Real* y) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        y[pin] = x[pin];
    }
}

template <class Real>
__global__ void kernel_assemble_neumann_cos_rhs(Grid3DCuda g, Real* rhs) {
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

    const double pi = 3.141592653589793238462643383279502884;
    const Real hx = Real(g.hx);
    const Real hy = Real(g.hy);
    const Real hz = Real(g.hz);
    const Real jacw = hx * hy * hz / Real(8);

    const Real gp = Real(0.577350269189625764509148780501957456);
    const Real q[2] = { -gp, gp };

    for (int iq = 0; iq < 2; ++iq) {
        const Real xi = q[iq];
        const Real Nx[2] = { Real(0.5) * (Real(1) - xi),
                             Real(0.5) * (Real(1) + xi) };

        const double x = (double(ex) + 0.5 * (1.0 + double(xi))) * g.hx;

        for (int jq = 0; jq < 2; ++jq) {
            const Real eta = q[jq];
            const Real Ny[2] = { Real(0.5) * (Real(1) - eta),
                                 Real(0.5) * (Real(1) + eta) };

            const double y = (double(ey) + 0.5 * (1.0 + double(eta))) * g.hy;

            for (int kq = 0; kq < 2; ++kq) {
                const Real zeta = q[kq];
                const Real Nz[2] = { Real(0.5) * (Real(1) - zeta),
                                     Real(0.5) * (Real(1) + zeta) };

                const double z = (double(ez) + 0.5 * (1.0 + double(zeta))) * g.hz;

                const Real f = Real(3.0 * pi * pi *
                    cos(pi * x) * cos(pi * y) * cos(pi * z));

                for (int a = 0; a < 8; ++a) {
                    const int ix = a & 1;
                    const int iy = (a >> 1) & 1;
                    const int iz = (a >> 2) & 1;

                    const Real Na = Nx[ix] * Ny[iy] * Nz[iz];
                    const int row = local_node_id(g, ex, ey, ez, a);

                    atomicAdd(&rhs[row], Na * f * jacw);
                }
            }
        }
    }
}

template <class Real>
__global__ void kernel_apply_pin_rhs_correction(Grid3DCuda g, Real* rhs) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    // Pin node (0,0,0), exact value = 1.
    // Only the first element contains this node.
    const int ex = 0;
    const int ey = 0;
    const int ez = 0;
    const int pinLocal = 0;
    const int pin = 0;
    const Real uPin = Real(1);

    for (int a = 0; a < 8; ++a) {
        const int row = local_node_id(g, ex, ey, ez, a);
        if (row == pin) continue;

        rhs[row] -= local_ke_q1<Real>(g, a, pinLocal) * uPin;
    }

    rhs[pin] = uPin;
}

template <class Real>
class StructuredPoissonNeumannPinnedGpuOperator {
public:
    explicit StructuredPoissonNeumannPinnedGpuOperator(
        const memoirs::structured::StructuredGrid3D& grid
    )
        : grid_(grid), dg_(to_cuda_grid(grid)) {}

    int size() const { return grid_.n_nodes(); }
    const memoirs::structured::StructuredGrid3D& grid() const { return grid_; }

    void apply(const Real* x, Real* y) const {
        const int n = size();
        const int block = 256;
        const int nElem = (grid_.nx - 1) * (grid_.ny - 1) * (grid_.nz - 1);

        MEMOIRS_CUDA_CHECK(cudaMemset(y, 0, static_cast<std::size_t>(n) * sizeof(Real)));

        kernel_apply_neumann_pinned_elements<Real>
            <<<div_up(nElem, block), block>>>
            (dg_, x, y);
        MEMOIRS_CUDA_KERNEL_CHECK();

        kernel_set_pin_row<Real><<<1, 1>>>(0, x, y);
        MEMOIRS_CUDA_KERNEL_CHECK();
    }

    void fill_exact(Real* u) const {
        const int n = size();
        const int block = 256;

        kernel_fill_neumann_cos_exact<Real>
            <<<div_up(n, block), block>>>
            (dg_, u);
        MEMOIRS_CUDA_KERNEL_CHECK();
    }

    void assemble_rhs(Real* rhs) const {
        const int n = size();
        const int block = 256;
        const int nElem = (grid_.nx - 1) * (grid_.ny - 1) * (grid_.nz - 1);

        MEMOIRS_CUDA_CHECK(cudaMemset(rhs, 0, static_cast<std::size_t>(n) * sizeof(Real)));

        kernel_assemble_neumann_cos_rhs<Real>
            <<<div_up(nElem, block), block>>>
            (dg_, rhs);
        MEMOIRS_CUDA_KERNEL_CHECK();

        kernel_apply_pin_rhs_correction<Real><<<1, 1>>>(dg_, rhs);
        MEMOIRS_CUDA_KERNEL_CHECK();
    }

private:
    memoirs::structured::StructuredGrid3D grid_;
    Grid3DCuda dg_;
};

template <class Real, class Operator, class Preconditioner>
CudaPcgReport<Real> pcg_solve_cuda_generic(
    const Operator& A,
    Preconditioner& M,
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
            std::cout << "cuda neumann-pcg it=" << it
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

} // namespace gpu
} // namespace memoirs

static double host_nodal_l2_error_neumann_cos(
    const memoirs::structured::StructuredGrid3D& g,
    const std::vector<Real>& x
) {
    const double pi = std::acos(-1.0);
    double s = 0.0;

    for (int k = 0; k < g.nz; ++k) {
        for (int j = 0; j < g.ny; ++j) {
            for (int i = 0; i < g.nx; ++i) {
                const int q = g.id(i, j, k);
                const double ue =
                    std::cos(pi * g.x(i)) *
                    std::cos(pi * g.y(j)) *
                    std::cos(pi * g.z(k));

                const double d = double(x[static_cast<std::size_t>(q)]) - ue;
                s += d * d;
            }
        }
    }

    return std::sqrt(s / double(g.n_nodes()));
}

int main(int argc, char** argv) {
    GpuMemoryMonitor gpuMemMon;

    try {
        const Cli cli = parse_cli(argc, argv);
        gpuMemMon.start();

        using Grid = memoirs::structured::StructuredGrid3D;
        using NeumannOp = memoirs::gpu::StructuredPoissonNeumannPinnedGpuOperator<Real>;
        using GpuGmg = memoirs::gpu::StructuredGmgCudaPreconditioner<Real>;

        const Grid grid(cli.nElem + 1, cli.nElem + 1, cli.nElem + 1);
        const NeumannOp A(grid);

        memoirs::gpu::StructuredGmgCudaOptions<Real> gopt;
        gopt.preSmooth = cli.pre;
        gopt.postSmooth = cli.post;
        gopt.omega = Real(cli.omega);
        gopt.coarseMaxDofs = cli.coarseMaxDofs;
        gopt.maxLevels = cli.maxLevels;
        gopt.verbose = cli.verbose;

        GpuGmg M;
        M.setup(grid, gopt);

        memoirs::gpu::DeviceBuffer<Real> d_b(grid.size());
        memoirs::gpu::DeviceBuffer<Real> d_x(grid.size());

        A.assemble_rhs(d_b.data());

        std::cout << "Memoirs CUDA structured Q1 Poisson Neumann MMS test\n";
        std::cout << "precision                 = " << kPrecisionName << "\n";
        std::cout << "grid                      = " << grid.label() << "\n";
        std::cout << "operator                  = poisson_neumann_natural_plus_reference_pin\n";
        std::cout << "mms                       = poisson_neumann_cos\n";
        std::cout << "rhs                       = continuous_q1_volume_rhs_homogeneous_neumann\n";
        std::cout << "pin                       = node_0_exact_value_1\n";
        std::cout << "solver                    = pcg_gmg\n";
        std::cout << "preconditioner            = diffusion_gmg_vcycle_dirichlet_approx\n";
        std::cout << "tol                       = " << cli.tol << "\n";
        std::cout << "maxit                     = " << cli.maxit << "\n";

        memoirs::diagnostics::CudaGmgTimingReport timingReport;
        memoirs::diagnostics::CudaSynchronizedTimer timer;

        memoirs::gpu::CudaPcgReport<Real> rep;

        for (int r = 0; r < cli.repeats; ++r) {
            d_x.zero();

            timer.start();
            rep = memoirs::gpu::pcg_solve_cuda_generic<Real>(
                A, M, d_b.data(), d_x.data(),
                Real(cli.tol), cli.maxit, cli.printEvery);
            timingReport.solveTotalSeconds += timer.stop_seconds();
            timingReport.solveRepeats += 1;

            std::vector<Real> h_x(grid.size());
            d_x.copy_to_host(h_x.data(), h_x.size());

            const double l2 = host_nodal_l2_error_neumann_cos(grid, h_x);

            std::cout << "--------------- neumann poisson solve report ---------------\n";
            std::cout << "repeat                    = " << r << "\n";
            std::cout << "iterations                = " << rep.iterations << "\n";
            std::cout << "converged                 = " << rep.converged << "\n";
            std::cout << "breakdown                 = " << rep.breakdown << "\n";
            std::cout << std::setprecision(17);
            std::cout << "initialResidual           = " << rep.initialResidual << "\n";
            std::cout << "finalResidual             = " << rep.finalResidual << "\n";
            std::cout << "finalRelativeResidual     = " << rep.finalRelativeResidual << "\n";
            std::cout << "l2DiscreteError           = " << l2 << "\n";
            std::cout << "------------------------------------------------------------\n";

            if (!rep.converged || rep.breakdown) {
                gpuMemMon.stop();
                gpuMemMon.print(std::cout);
                return 2;
            }
        }

        if (timingReport.solveRepeats > 0) {
            timingReport.solveAvgSeconds =
                timingReport.solveTotalSeconds / double(timingReport.solveRepeats);
        }

        timingReport.print(std::cout);

        gpuMemMon.stop();
        gpuMemMon.print(std::cout);

        return 0;
    } catch (const std::exception& e) {
        gpuMemMon.stop();
        gpuMemMon.print(std::cout);
        std::cerr << "FATAL: " << e.what() << "\n";
        return 1;
    }
}
