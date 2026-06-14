#include "memoirs/solvers/DenseLu.hpp"

#include <algorithm>
#include <cmath>
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

template <class Scalar>
static Scalar inf_norm_diff(const std::vector<Scalar>& a, const std::vector<Scalar>& b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("inf_norm_diff size mismatch");
    }

    Scalar e = Scalar(0);
    for (std::size_t i = 0; i < a.size(); ++i) {
        e = std::max(e, Scalar(std::abs(a[i] - b[i])));
    }
    return e;
}

template <class Scalar>
static void run_known_3x3() {
    using memoirs::solvers::DenseLu;
    using memoirs::solvers::dense_lu_residual_inf_norm;

    const int n = 3;

    // A has enough off-diagonal coupling to exercise elimination.
    std::vector<Scalar> A = {
        Scalar(3), Scalar(2),  Scalar(-1),
        Scalar(2), Scalar(-2), Scalar(4),
        Scalar(-1),Scalar(0.5),Scalar(-1)
    };

    std::vector<Scalar> xExact = {
        Scalar(1), Scalar(-2), Scalar(-2)
    };

    std::vector<Scalar> b(n, Scalar(0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            b[static_cast<std::size_t>(i)] +=
                A[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) + static_cast<std::size_t>(j)]
                * xExact[static_cast<std::size_t>(j)];
        }
    }

    DenseLu<Scalar> lu;
    lu.factor(n, A);
    std::vector<Scalar> x = lu.solve(b);

    const Scalar xerr = inf_norm_diff(x, xExact);
    const Scalar rerr = dense_lu_residual_inf_norm(n, A, x, b);

    std::cout << "known3x3 scalar=" << (sizeof(Scalar) == sizeof(float) ? "float" : "double")
              << " xInfErr=" << std::setprecision(17) << double(xerr)
              << " rInfErr=" << std::setprecision(17) << double(rerr)
              << "\n";

    const Scalar tol = sizeof(Scalar) == sizeof(float) ? Scalar(5e-5) : Scalar(1e-12);
    if (!(xerr < tol && rerr < tol)) {
        throw std::runtime_error("known3x3 DenseLu test failed");
    }
}

template <class Scalar>
static void run_poisson_coarse(int nxNodes, int nyNodes, int nzNodes) {
    using memoirs::solvers::DenseLu;
    using memoirs::solvers::build_dirichlet_poisson_3d_dense_matrix;
    using memoirs::solvers::dense_lu_residual_inf_norm;

    const int n = nxNodes * nyNodes * nzNodes;
    std::vector<Scalar> A =
        build_dirichlet_poisson_3d_dense_matrix<Scalar>(nxNodes, nyNodes, nzNodes);

    std::vector<Scalar> xExact(static_cast<std::size_t>(n), Scalar(0));
    for (int q = 0; q < n; ++q) {
        xExact[static_cast<std::size_t>(q)] =
            Scalar(0.25) + Scalar(0.01) * Scalar((q % 17) - 8);
    }

    std::vector<Scalar> b(static_cast<std::size_t>(n), Scalar(0));
    for (int i = 0; i < n; ++i) {
        Scalar sum = Scalar(0);
        for (int j = 0; j < n; ++j) {
            sum += A[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) + static_cast<std::size_t>(j)]
                 * xExact[static_cast<std::size_t>(j)];
        }
        b[static_cast<std::size_t>(i)] = sum;
    }

    DenseLu<Scalar> lu;
    lu.factor(n, A);
    std::vector<Scalar> x = lu.solve(b);

    const Scalar xerr = inf_norm_diff(x, xExact);
    const Scalar rerr = dense_lu_residual_inf_norm(n, A, x, b);

    const double denseMatrixMiB =
        double(n) * double(n) * double(sizeof(Scalar)) / (1024.0 * 1024.0);

    std::cout << "poissonCoarse"
              << " nodes=" << nxNodes << "x" << nyNodes << "x" << nzNodes
              << " n=" << n
              << " scalar=" << (sizeof(Scalar) == sizeof(float) ? "float" : "double")
              << " denseMatrixMiB=" << std::fixed << std::setprecision(6) << denseMatrixMiB
              << " xInfErr=" << std::scientific << std::setprecision(17) << double(xerr)
              << " rInfErr=" << std::scientific << std::setprecision(17) << double(rerr)
              << "\n";

    const Scalar tol = sizeof(Scalar) == sizeof(float) ? Scalar(1e-3) : Scalar(1e-10);
    if (!(xerr < tol && rerr < tol)) {
        throw std::runtime_error("poisson coarse DenseLu test failed");
    }
}

int main() {
    try {
        std::cout << "Memoirs DenseLu CPU test\n";
        std::cout << "configured precision = " << kPrecisionName << "\n";

        run_known_3x3<double>();
        run_known_3x3<float>();

        // Coarse GMG examples:
        //
        // 2x2x2 coarse elements -> 3x3x3 nodes = 27 unknowns.
        // 4x4x4 coarse elements -> 5x5x5 nodes = 125 unknowns.
        run_poisson_coarse<double>(3, 3, 3);
        run_poisson_coarse<double>(5, 5, 5);

        // Test Real too, so the build precision path is exercised.
        run_poisson_coarse<Real>(3, 3, 3);

        std::cout << "DenseLu tests PASSED\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "DenseLu tests FAILED: " << e.what() << "\n";
        return 1;
    }
}
