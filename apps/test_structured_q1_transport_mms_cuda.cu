#include "memoirs/diagnostics/GpuMemoryMonitor.hpp"
#include "memoirs/diagnostics/CudaTiming.hpp"
#include "memoirs/diagnostics/StructuredQ1ErrorDiagnostics.hpp"
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
    double kappa = 0.1;
    double ax = 1.0;
    double ay = 2.0;
    double az = 3.0;
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
        else if (a == "-kappa") c.kappa = arg_double(i, argc, argv);
        else if (a == "-ax") c.ax = arg_double(i, argc, argv);
        else if (a == "-ay") c.ay = arg_double(i, argc, argv);
        else if (a == "-az") c.az = arg_double(i, argc, argv);
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
__device__ inline Real q1_adv_weight(const Grid3DCuda g,
                                     Real ax, Real ay, Real az,
                                     int di, int dj, int dk) {
    const Real hx = Real(g.hx);
    const Real hy = Real(g.hy);
    const Real hz = Real(g.hz);

    const int ai = di + 1;
    const int aj = dj + 1;
    const int ak = dk + 1;

    const Real M[3] = {
        Real(1.0 / 3.0),
        Real(4.0 / 3.0),
        Real(1.0 / 3.0)
    };

    // Assembled central derivative stencil:
    // col i-1 -> -1/2, col i -> 0, col i+1 -> +1/2.
    const Real D[3] = {
        Real(-0.5),
        Real(0.0),
        Real(0.5)
    };

    // Physical tensor factors:
    // x-advection: ax * D_x * (hy/2 M_y) * (hz/2 M_z)
    // so coefficient hy*hz/4.
    return
        ax * (hy * hz / Real(4)) * D[ai] * M[aj] * M[ak] +
        ay * (hx * hz / Real(4)) * M[ai] * D[aj] * M[ak] +
        az * (hx * hy / Real(4)) * M[ai] * M[aj] * D[ak];
}

template <class Real>
__global__ void kernel_fill_transport_exact(Grid3DCuda g, Real* u) {
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

    u[q] = Real(std::sin(pi * x) * std::sin(pi * y) * std::sin(pi * z));
}


template <class Real>
__global__ void kernel_assemble_transport_sin_rhs(Grid3DCuda g,
                                                  Real kappa,
                                                  Real ax,
                                                  Real ay,
                                                  Real az,
                                                  Real* rhs) {
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
        const Real Nx[2] = {
            Real(0.5) * (Real(1) - xi),
            Real(0.5) * (Real(1) + xi)
        };
        const double x = (double(ex) + 0.5 * (1.0 + double(xi))) * g.hx;

        const double sx = sin(pi * x);
        const double cx = cos(pi * x);

        for (int jq = 0; jq < 2; ++jq) {
            const Real eta = q[jq];
            const Real Ny[2] = {
                Real(0.5) * (Real(1) - eta),
                Real(0.5) * (Real(1) + eta)
            };
            const double y = (double(ey) + 0.5 * (1.0 + double(eta))) * g.hy;

            const double sy = sin(pi * y);
            const double cy = cos(pi * y);

            for (int kq = 0; kq < 2; ++kq) {
                const Real zeta = q[kq];
                const Real Nz[2] = {
                    Real(0.5) * (Real(1) - zeta),
                    Real(0.5) * (Real(1) + zeta)
                };
                const double z = (double(ez) + 0.5 * (1.0 + double(zeta))) * g.hz;

                const double sz = sin(pi * z);
                const double cz = cos(pi * z);

                const double u = sx * sy * sz;

                const double ux = pi * cx * sy * sz;
                const double uy = pi * sx * cy * sz;
                const double uz = pi * sx * sy * cz;

                const Real f = Real(
                    3.0 * double(kappa) * pi * pi * u
                    + double(ax) * ux
                    + double(ay) * uy
                    + double(az) * uz
                );

                for (int a = 0; a < 8; ++a) {
                    const int ix = a & 1;
                    const int iy = (a >> 1) & 1;
                    const int iz = (a >> 2) & 1;

                    const Real Na = Nx[ix] * Ny[iy] * Nz[iz];

                    const int row =
                        g_id(g, ex + ix, ey + iy, ez + iz);

                    atomicAdd(&rhs[row], Na * f * jacw);
                }
            }
        }
    }
}

template <class Real>
__global__ void kernel_set_transport_dirichlet_rhs(Grid3DCuda g, Real* rhs) {
    const int n = g.n_nodes_host();
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= n) return;

    const int i = q % g.nx;
    const int tmp = q / g.nx;
    const int j = tmp % g.ny;
    const int k = tmp / g.ny;

    if (g_boundary(g, i, j, k)) {
        // u_exact = sin(pi x) sin(pi y) sin(pi z) = 0 on cube boundary.
        rhs[q] = Real(0);
    }
}

template <class Real>
__global__ void kernel_apply_transport_dirichlet(Grid3DCuda g,
                                                 Real kappa,
                                                 Real ax,
                                                 Real ay,
                                                 Real az,
                                                 const Real* x,
                                                 Real* y) {
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

                // Strong homogeneous Dirichlet: remove boundary columns.
                if (g_boundary(g, ii, jj, kk)) continue;

                const int col = g_id(g, ii, jj, kk);

                const Real diff = kappa * q1_weight<Real>(g, di, dj, dk);
                const Real adv  = q1_adv_weight<Real>(g, ax, ay, az, di, dj, dk);

                sum += (diff + adv) * x[col];
            }
        }
    }

    y[q] = sum;
}

template <class Real>
class StructuredTransportDirichletGpuOperator {
public:
    explicit StructuredTransportDirichletGpuOperator(
        const memoirs::structured::StructuredGrid3D& grid,
        Real kappa,
        Real ax,
        Real ay,
        Real az
    )
        : grid_(grid),
          dg_(to_cuda_grid(grid)),
          kappa_(kappa),
          ax_(ax),
          ay_(ay),
          az_(az) {}

    int size() const { return grid_.n_nodes(); }
    const memoirs::structured::StructuredGrid3D& grid() const { return grid_; }

    void apply(const Real* x, Real* y) const {
        const int n = size();
        const int block = 256;

        kernel_apply_transport_dirichlet<Real>
            <<<div_up(n, block), block>>>
            (dg_, kappa_, ax_, ay_, az_, x, y);

        MEMOIRS_CUDA_KERNEL_CHECK();
    }

    void fill_exact(Real* u) const {
        const int n = size();
        const int block = 256;

        kernel_fill_transport_exact<Real>
            <<<div_up(n, block), block>>>
            (dg_, u);

        MEMOIRS_CUDA_KERNEL_CHECK();
    }

    void assemble_rhs(Real* rhs) const {
        const int n = size();
        const int block = 256;
        const int nElem =
            (grid_.nx - 1) * (grid_.ny - 1) * (grid_.nz - 1);

        MEMOIRS_CUDA_CHECK(cudaMemset(
            rhs, 0, static_cast<std::size_t>(n) * sizeof(Real)));

        kernel_assemble_transport_sin_rhs<Real>
            <<<div_up(nElem, block), block>>>
            (dg_, kappa_, ax_, ay_, az_, rhs);
        MEMOIRS_CUDA_KERNEL_CHECK();

        kernel_set_transport_dirichlet_rhs<Real>
            <<<div_up(n, block), block>>>
            (dg_, rhs);
        MEMOIRS_CUDA_KERNEL_CHECK();
    }

private:
    memoirs::structured::StructuredGrid3D grid_;
    Grid3DCuda dg_;
    Real kappa_;
    Real ax_;
    Real ay_;
    Real az_;
};

template <class Real>
__global__ void kernel_combination3(int n,
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
__global__ void kernel_axpby(int n,
                             Real a,
                             const Real* x,
                             Real b,
                             Real* y) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q < n) y[q] = a * x[q] + b * y[q];
}

template <class Real>
__global__ void kernel_xpay(int n,
                            const Real* x,
                            Real beta,
                            Real* y) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q < n) y[q] = x[q] + beta * y[q];
}

template <class Real>
struct CudaBicgstabReport {
    int iterations = 0;
    int converged = 0;
    int breakdown = 0;
    double initialResidual = 0.0;
    double finalResidual = 0.0;
    double finalRelativeResidual = 0.0;
};

template <class Real, class Operator, class RightPreconditioner>
CudaBicgstabReport<Real> bicgstab_right_prec_cuda(
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

    // Right-preconditioned temporary vectors:
    // phat = M^{-1} p
    // shat = M^{-1} s
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

    CudaBicgstabReport<Real> rep;
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

        // p = r + beta * (p - omega v)
        kernel_combination3<Real>
            <<<div_up(n, block), block>>>
            (n, Real(1), d_r.data(),
                Real(beta), d_p.data(),
                Real(-beta * omega), d_v.data(),
                d_p.data());
        MEMOIRS_CUDA_KERNEL_CHECK();

        // phat = M^{-1} p
        M.apply(d_p.data(), d_phat.data());

        // v = A phat
        A.apply(d_phat.data(), d_v.data());

        const double rhatv = dots.dot(n, d_rhat.data(), d_v.data());
        if (!(std::abs(rhatv) > 0.0)) {
            rep.breakdown = 1;
            rep.iterations = it - 1;
            return rep;
        }

        alpha = rhoNew / rhatv;

        // s = r - alpha v
        kernel_combination3<Real>
            <<<div_up(n, block), block>>>
            (n, Real(1), d_r.data(),
                Real(-alpha), d_v.data(),
                Real(0), d_v.data(),
                d_s.data());
        MEMOIRS_CUDA_KERNEL_CHECK();

        const double snorm = std::sqrt(dots.dot(n, d_s.data(), d_s.data()));
        if (snorm / bnorm <= double(tol)) {
            // x = x + alpha phat
            kernel_axpby<Real>
                <<<div_up(n, block), block>>>
                (n, Real(alpha), d_phat.data(), Real(1), d_x);
            MEMOIRS_CUDA_KERNEL_CHECK();

            rep.iterations = it;
            rep.converged = 1;
            rep.finalResidual = snorm;
            rep.finalRelativeResidual = snorm / bnorm;
            return rep;
        }

        // shat = M^{-1} s
        M.apply(d_s.data(), d_shat.data());

        // t = A shat
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

        // x = x + alpha phat + omega shat
        kernel_combination3<Real>
            <<<div_up(n, block), block>>>
            (n, Real(1), d_x,
                Real(alpha), d_phat.data(),
                Real(omega), d_shat.data(),
                d_x);
        MEMOIRS_CUDA_KERNEL_CHECK();

        // r = s - omega t
        kernel_combination3<Real>
            <<<div_up(n, block), block>>>
            (n, Real(1), d_s.data(),
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


int main(int argc, char** argv) {
    GpuMemoryMonitor gpuMemMon;

    try {
        const Cli cli = parse_cli(argc, argv);
        gpuMemMon.start();

        using Grid = memoirs::structured::StructuredGrid3D;
        using TransportOp = memoirs::gpu::StructuredTransportDirichletGpuOperator<Real>;
        using GpuGmg = memoirs::gpu::StructuredGmgCudaPreconditioner<Real>;

        const Grid grid(cli.nElem + 1, cli.nElem + 1, cli.nElem + 1);

        const TransportOp A(
            grid,
            Real(cli.kappa),
            Real(cli.ax),
            Real(cli.ay),
            Real(cli.az)
        );

        // Preconditioner: diffusion GMG only.
        // This is intentional for weak advection and later PSPG-like systems.
        memoirs::gpu::StructuredGmgCudaOptions<Real> gopt;
        gopt.preSmooth = cli.pre;
        gopt.postSmooth = cli.post;
        gopt.omega = Real(cli.omega);
        gopt.coarseMaxDofs = cli.coarseMaxDofs;
        gopt.maxLevels = cli.maxLevels;
        gopt.verbose = cli.verbose;

        GpuGmg M;
        M.setup(grid, gopt);

        memoirs::gpu::DeviceBuffer<Real> d_xExact(grid.size());
        memoirs::gpu::DeviceBuffer<Real> d_b(grid.size());
        memoirs::gpu::DeviceBuffer<Real> d_x(grid.size());

        A.fill_exact(d_xExact.data());

        // Continuous Q1 quadrature MMS RHS:
        // b_i = integral N_i * f dOmega, with homogeneous Dirichlet rows.
        A.assemble_rhs(d_b.data());

        std::cout << "Memoirs CUDA structured Q1 scalar transport MMS test\n";
        std::cout << "precision                 = " << kPrecisionName << "\n";
        std::cout << "grid                      = " << grid.label() << "\n";
        std::cout << "operator                  = -kappa_laplacian_plus_a_dot_grad\n";
        std::cout << "mms                       = transport_dirichlet_sin\n";
        std::cout << "rhs                       = continuous_q1_volume_rhs\n";
        std::cout << "solver                    = right_bicgstab_gmg\n";
        std::cout << "rightPreconditioner       = diffusion_gmg_vcycle\n";
        std::cout << "kappa                     = " << cli.kappa << "\n";
        std::cout << "a                         = ("
                  << cli.ax << "," << cli.ay << "," << cli.az << ")\n";
        std::cout << "tol                       = " << cli.tol << "\n";
        std::cout << "maxit                     = " << cli.maxit << "\n";

        memoirs::diagnostics::CudaGmgTimingReport timingReport;
        memoirs::diagnostics::CudaSynchronizedTimer timer;

        memoirs::gpu::CudaBicgstabReport<Real> rep;

        for (int r = 0; r < cli.repeats; ++r) {
            d_x.zero();

            timer.start();
            rep = memoirs::gpu::bicgstab_right_prec_cuda<Real>(
                A, M, d_b.data(), d_x.data(),
                Real(cli.tol), cli.maxit, cli.printEvery);
            timingReport.solveTotalSeconds += timer.stop_seconds();
            timingReport.solveRepeats += 1;

            const double xL2 =
                memoirs::gpu::device_l2_diff<Real>(
                    grid.n_nodes(), d_x.data(), d_xExact.data());

            std::vector<Real> h_x(grid.size());
            d_x.copy_to_host(h_x.data(), h_x.size());

            const double xL2Quad =
                memoirs::diagnostics::structured_q1_l2_error_transport_sin<Real>(grid, h_x);

            std::cout << "--------------- transport solve report ---------------\n";
            std::cout << "repeat                    = " << r << "\n";
            std::cout << "iterations                = " << rep.iterations << "\n";
            std::cout << "converged                 = " << rep.converged << "\n";
            std::cout << "breakdown                 = " << rep.breakdown << "\n";
            std::cout << std::setprecision(17);
            std::cout << "initialResidual           = " << rep.initialResidual << "\n";
            std::cout << "finalResidual             = " << rep.finalResidual << "\n";
            std::cout << "finalRelativeResidual     = " << rep.finalRelativeResidual << "\n";
            std::cout << "xErrorL2Discrete          = " << xL2 << "\n";
            std::cout << "xErrorL2Quadrature        = " << xL2Quad << "\n";
            std::cout << "-----------------------------------------------------\n";

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
