#include "memoirs/operators/StructuredQ1PoissonOperator.hpp"
#include "memoirs/solvers/GeometricMultigrid.hpp"
#include "memoirs/solvers/PcgSolver.hpp"
#include "memoirs/structured/StructuredGrid3D.hpp"

#include <algorithm>
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

struct Cli {
    int nElem = 32;
    int maxit = 100;
    double tol = 1e-10;
    int pre = 2;
    int post = 2;
    double omega = 0.70;
    int coarseMaxDofs = 256;
    int maxLevels = 32;
    int verbose = 1;
    int printEvery = 10;
};

static int parse_int_arg(int& i, int argc, char** argv) {
    if (i + 1 >= argc) throw std::runtime_error("missing integer argument");
    return std::atoi(argv[++i]);
}

static double parse_double_arg(int& i, int argc, char** argv) {
    if (i + 1 >= argc) throw std::runtime_error("missing floating argument");
    return std::atof(argv[++i]);
}

static Cli parse_cli(int argc, char** argv) {
    Cli c;

    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);

        if (a == "-n" || a == "-nElem") {
            c.nElem = parse_int_arg(i, argc, argv);
        } else if (a == "-maxit") {
            c.maxit = parse_int_arg(i, argc, argv);
        } else if (a == "-tol") {
            c.tol = parse_double_arg(i, argc, argv);
        } else if (a == "-pre") {
            c.pre = parse_int_arg(i, argc, argv);
        } else if (a == "-post") {
            c.post = parse_int_arg(i, argc, argv);
        } else if (a == "-omega") {
            c.omega = parse_double_arg(i, argc, argv);
        } else if (a == "-coarseMaxDofs") {
            c.coarseMaxDofs = parse_int_arg(i, argc, argv);
        } else if (a == "-maxLevels") {
            c.maxLevels = parse_int_arg(i, argc, argv);
        } else if (a == "-verbose") {
            c.verbose = parse_int_arg(i, argc, argv);
        } else if (a == "-printEvery") {
            c.printEvery = parse_int_arg(i, argc, argv);
        } else {
            throw std::runtime_error("unknown argument: " + a);
        }
    }

    if (c.nElem < 2) {
        throw std::runtime_error("-n must be >= 2");
    }

    return c;
}

static Real exact_u(double x, double y, double z) {
    const double pi = std::acos(-1.0);
    return Real(std::sin(pi * x) * std::sin(pi * y) * std::sin(pi * z));
}

static Real inf_error(const std::vector<Real>& a, const std::vector<Real>& b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("inf_error size mismatch");
    }

    Real e = Real(0);
    for (std::size_t i = 0; i < a.size(); ++i) {
        e = std::max(e, Real(std::abs(a[i] - b[i])));
    }
    return e;
}

static Real l2_error_discrete(const std::vector<Real>& a, const std::vector<Real>& b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("l2_error_discrete size mismatch");
    }

    Real s = Real(0);
    for (std::size_t i = 0; i < a.size(); ++i) {
        const Real d = a[i] - b[i];
        s += d * d;
    }
    return std::sqrt(s / Real(a.size()));
}

int main(int argc, char** argv) {
    try {
        const Cli cli = parse_cli(argc, argv);

        using Grid = memoirs::structured::StructuredGrid3D;
        using Op = memoirs::operators::StructuredQ1PoissonOperator<Real>;
        using Gmg = memoirs::solvers::StructuredGmgPreconditioner<Real>;

        const Grid grid(cli.nElem + 1, cli.nElem + 1, cli.nElem + 1);
        const Op A(grid);

        std::vector<Real> xExact(grid.size(), Real(0));
        std::vector<Real> b;
        std::vector<Real> x(grid.size(), Real(0));

        for (int k = 0; k < grid.nz; ++k) {
            for (int j = 0; j < grid.ny; ++j) {
                for (int i = 0; i < grid.nx; ++i) {
                    const int q = grid.id(i, j, k);
                    xExact[static_cast<std::size_t>(q)] =
                        exact_u(grid.x(i), grid.y(j), grid.z(k));
                }
            }
        }

        // Algebraic test first:
        // b = A * u_exact using the same matrix-free operator.
        // This isolates PCG+GMG correctness before adding continuous MMS RHS.
        A.apply(xExact, b);

        typename Gmg::Options gopt;
        gopt.preSmooth = cli.pre;
        gopt.postSmooth = cli.post;
        gopt.omega = Real(cli.omega);
        gopt.coarseMaxDofs = cli.coarseMaxDofs;
        gopt.maxLevels = cli.maxLevels;
        gopt.verbose = cli.verbose;

        Gmg M;
        M.setup(grid, gopt);

        std::cout << "Memoirs structured matrix-free PCG+GMG test\n";
        std::cout << "precision                 = " << kPrecisionName << "\n";
        std::cout << "grid                      = " << grid.label() << "\n";
        std::cout << "operator                  = structured_q1_poisson_matrix_free_27pt_tensor\n";
        std::cout << "boundary                  = homogeneous_dirichlet_eliminated\n";
        std::cout << "rhs                       = algebraic_A_times_sin_exact\n";
        std::cout << "pcgTol                    = " << cli.tol << "\n";
        std::cout << "pcgMaxit                  = " << cli.maxit << "\n";
        std::cout << "estimatedFineVectors8MiB  = "
                  << A.estimated_matrix_free_vector_mib(8) << "\n";

        const auto rep = memoirs::solvers::pcg_solve<Real>(
            A, M, b, x, Real(cli.tol), cli.maxit, cli.printEvery);

        std::vector<Real> Ax;
        A.apply(x, Ax);

        std::vector<Real> r(b.size(), Real(0));
        for (std::size_t i = 0; i < b.size(); ++i) {
            r[i] = b[i] - Ax[i];
        }

        const Real rnorm = memoirs::solvers::norm2(r);
        const Real bnorm = std::max(memoirs::solvers::norm2(b), Real(1));
        const Real xInf = inf_error(x, xExact);
        const Real xL2 = l2_error_discrete(x, xExact);

        std::cout << "--------------- structured solve report ---------------\n";
        std::cout << "solver                    = pcg\n";
        std::cout << "precond                   = structured_gmg_vcycle\n";
        std::cout << "coarseSolve               = dense_lu_cpu_exact\n";
        std::cout << "iterations                = " << rep.iterations << "\n";
        std::cout << "converged                 = " << rep.converged << "\n";
        std::cout << "breakdown                 = " << rep.breakdown << "\n";
        std::cout << std::setprecision(17);
        std::cout << "initialResidual           = " << double(rep.initialResidual) << "\n";
        std::cout << "finalResidual             = " << double(rnorm) << "\n";
        std::cout << "finalRelativeResidual     = " << double(rnorm / bnorm) << "\n";
        std::cout << "xErrorInf                 = " << double(xInf) << "\n";
        std::cout << "xErrorL2Discrete          = " << double(xL2) << "\n";
        std::cout << "-------------------------------------------------------\n";

        if (!rep.converged || rep.breakdown) {
            return 2;
        }

        const Real xtol = sizeof(Real) == sizeof(float) ? Real(5e-4) : Real(1e-8);
        if (!(xInf < xtol)) {
            std::cerr << "xInf error too large: " << double(xInf)
                      << " xtol=" << double(xtol) << "\n";
            return 3;
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FATAL: " << e.what() << "\n";
        return 1;
    }
}
