#include "memoirs/diagnostics/GpuMemoryMonitor.hpp"
#include "memoirs/diagnostics/CudaTiming.hpp"
#include "memoirs/gpu/StructuredGmgCuda.cuh"

#include <cuda_runtime.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#if defined(MEMOIRS_PRECISION_single)
using Real = float;
static constexpr const char* kPrecisionName = "single";
#else
using Real = double;
static constexpr const char* kPrecisionName = "double";
#endif

struct Cli {
    int nElem = 64;
    int maxit = 100;
    double tol = 1e-10;
    int pre = 2;
    int post = 2;
    double omega = 0.70;
    int coarseMaxDofs = 256;
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

int main(int argc, char** argv) {
    GpuMemoryMonitor gpuMemMon;

    try {
        const Cli cli = parse_cli(argc, argv);

        gpuMemMon.start();

        using Grid = memoirs::structured::StructuredGrid3D;
        using GpuOp = memoirs::gpu::StructuredQ1GpuOperator<Real>;
        using GpuGmg = memoirs::gpu::StructuredGmgCudaPreconditioner<Real>;

        const Grid grid(cli.nElem + 1, cli.nElem + 1, cli.nElem + 1);
        const GpuOp A(grid);

        memoirs::diagnostics::CudaGmgTimingReport timingReport;
        memoirs::diagnostics::CudaSynchronizedTimer timingTimer;

        memoirs::gpu::DeviceBuffer<Real> d_xExact(grid.size());
        memoirs::gpu::DeviceBuffer<Real> d_b(grid.size());
        memoirs::gpu::DeviceBuffer<Real> d_x(grid.size());

        timingTimer.start();
        A.fill_exact_sin(d_xExact.data());
        A.apply(d_xExact.data(), d_b.data());
        timingReport.rhsSeconds = timingTimer.stop_seconds();

        memoirs::gpu::StructuredGmgCudaOptions<Real> gopt;
        gopt.preSmooth = cli.pre;
        gopt.postSmooth = cli.post;
        gopt.omega = Real(cli.omega);
        gopt.coarseMaxDofs = cli.coarseMaxDofs;
        gopt.maxLevels = cli.maxLevels;
        gopt.verbose = cli.verbose;

        GpuGmg M;
        timingTimer.start();
        M.setup(grid, gopt);
        timingReport.setupSeconds = timingTimer.stop_seconds();

        std::cout << "Memoirs CUDA structured matrix-free PCG+GMG test\n";
        std::cout << "precision                 = " << kPrecisionName << "\n";
        std::cout << "grid                      = " << grid.label() << "\n";
        std::cout << "operator                  = cuda_structured_q1_poisson_matrix_free_27pt_tensor\n";
        std::cout << "boundary                  = homogeneous_dirichlet_eliminated\n";
        std::cout << "rhs                       = gpu_algebraic_A_times_sin_exact\n";
        std::cout << "pcgTol                    = " << cli.tol << "\n";
        std::cout << "pcgMaxit                  = " << cli.maxit << "\n";
        std::cout << "repeats                   = " << cli.repeats << "\n";
        std::cout << "coarseSolverNote          = CPU DenseLu temporary; large levels are GPU-resident\n";

        memoirs::gpu::CudaPcgReport<Real> rep;

        for (int r = 0; r < cli.repeats; ++r) {
            d_x.zero();

            timingTimer.start();
            rep = memoirs::gpu::pcg_solve_cuda<Real>(
                A, M, d_b.data(), d_x.data(),
                Real(cli.tol), cli.maxit, cli.printEvery);
            timingReport.solveTotalSeconds += timingTimer.stop_seconds();
            timingReport.solveRepeats += 1;

            const double rel =
                memoirs::gpu::device_residual_rel<Real>(A, d_b.data(), d_x.data());
            const double xL2 =
                memoirs::gpu::device_l2_diff<Real>(
                    grid.n_nodes(), d_x.data(), d_xExact.data());

            std::cout << "--------------- CUDA structured solve report ---------------\n";
            std::cout << "repeat                    = " << r << "\n";
            std::cout << "solver                    = pcg_cuda_host_control\n";
            std::cout << "precond                   = structured_gmg_cuda_vcycle\n";
            std::cout << "coarseSolve               = dense_lu_cpu_exact_TEMPORARY\n";
            std::cout << "iterations                = " << rep.iterations << "\n";
            std::cout << "converged                 = " << rep.converged << "\n";
            std::cout << "breakdown                 = " << rep.breakdown << "\n";
            std::cout << std::setprecision(17);
            std::cout << "initialResidual           = " << rep.initialResidual << "\n";
            std::cout << "finalResidual             = " << rep.finalResidual << "\n";
            std::cout << "finalRelativeResidual     = " << rep.finalRelativeResidual << "\n";
            std::cout << "checkedRelativeResidual   = " << rel << "\n";
            std::cout << "xErrorL2Discrete          = " << xL2 << "\n";
            std::cout << "------------------------------------------------------------\n";

            if (!rep.converged || rep.breakdown) {
                gpuMemMon.stop();
                gpuMemMon.print(std::cout);
                return 2;
            }
        }

        MEMOIRS_CUDA_CHECK(cudaDeviceSynchronize());

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
