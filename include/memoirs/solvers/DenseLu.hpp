#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace memoirs {
namespace solvers {

template <class Scalar>
class DenseLu {
public:
    DenseLu() = default;

    void factor(int n, const std::vector<Scalar>& ArowMajor) {
        if (n <= 0) {
            throw std::runtime_error("DenseLu::factor requires n > 0");
        }

        const std::size_t nn = static_cast<std::size_t>(n);
        if (ArowMajor.size() != nn * nn) {
            throw std::runtime_error("DenseLu::factor received wrong matrix size");
        }

        n_ = n;
        lu_ = ArowMajor;
        piv_.resize(nn);

        for (int i = 0; i < n_; ++i) {
            piv_[static_cast<std::size_t>(i)] = i;
        }

        Scalar maxAbsA = Scalar(0);
        for (Scalar v : lu_) {
            maxAbsA = std::max(maxAbsA, Scalar(std::abs(v)));
        }

        if (maxAbsA == Scalar(0)) {
            throw std::runtime_error("DenseLu::factor got an all-zero matrix");
        }

        const Scalar eps = std::numeric_limits<Scalar>::epsilon();
        const Scalar pivotTol = Scalar(64) * eps * maxAbsA;

        for (int k = 0; k < n_; ++k) {
            int p = k;
            Scalar best = Scalar(std::abs(lu_[idx(k, k)]));

            for (int i = k + 1; i < n_; ++i) {
                const Scalar cand = Scalar(std::abs(lu_[idx(i, k)]));
                if (cand > best) {
                    best = cand;
                    p = i;
                }
            }

            if (!(best > pivotTol)) {
                std::ostringstream oss;
                oss << "DenseLu::factor singular or near-singular pivot at k=" << k
                    << " pivot=" << std::setprecision(17) << best
                    << " tol=" << pivotTol
                    << " maxAbsA=" << maxAbsA;
                throw std::runtime_error(oss.str());
            }

            piv_[static_cast<std::size_t>(k)] = p;

            if (p != k) {
                for (int j = 0; j < n_; ++j) {
                    std::swap(lu_[idx(k, j)], lu_[idx(p, j)]);
                }
            }

            const Scalar akk = lu_[idx(k, k)];

            for (int i = k + 1; i < n_; ++i) {
                lu_[idx(i, k)] /= akk;
                const Scalar lik = lu_[idx(i, k)];

                for (int j = k + 1; j < n_; ++j) {
                    lu_[idx(i, j)] -= lik * lu_[idx(k, j)];
                }
            }
        }

        factored_ = true;
    }

    void solve(const std::vector<Scalar>& b, std::vector<Scalar>& x) const {
        require_factored();

        const std::size_t nn = static_cast<std::size_t>(n_);
        if (b.size() != nn) {
            throw std::runtime_error("DenseLu::solve received wrong RHS size");
        }

        x = b;

        // Apply the same row interchanges as used during factorization:
        // PA = LU, so solve LU x = P b.
        for (int k = 0; k < n_; ++k) {
            const int p = piv_[static_cast<std::size_t>(k)];
            if (p != k) {
                std::swap(x[static_cast<std::size_t>(k)],
                          x[static_cast<std::size_t>(p)]);
            }
        }

        // Forward solve L y = P b. L has implicit unit diagonal.
        for (int i = 0; i < n_; ++i) {
            Scalar sum = x[static_cast<std::size_t>(i)];

            for (int j = 0; j < i; ++j) {
                sum -= lu_[idx(i, j)] * x[static_cast<std::size_t>(j)];
            }

            x[static_cast<std::size_t>(i)] = sum;
        }

        // Backward solve U x = y.
        for (int i = n_ - 1; i >= 0; --i) {
            Scalar sum = x[static_cast<std::size_t>(i)];

            for (int j = i + 1; j < n_; ++j) {
                sum -= lu_[idx(i, j)] * x[static_cast<std::size_t>(j)];
            }

            x[static_cast<std::size_t>(i)] = sum / lu_[idx(i, i)];
        }
    }

    std::vector<Scalar> solve(const std::vector<Scalar>& b) const {
        std::vector<Scalar> x;
        solve(b, x);
        return x;
    }

    int n() const {
        return n_;
    }

    bool factored() const {
        return factored_;
    }

    const std::vector<Scalar>& lu_storage() const {
        return lu_;
    }

    const std::vector<int>& pivots() const {
        return piv_;
    }

private:
    std::size_t idx(int i, int j) const {
        return static_cast<std::size_t>(i) * static_cast<std::size_t>(n_)
             + static_cast<std::size_t>(j);
    }

    void require_factored() const {
        if (!factored_) {
            throw std::runtime_error("DenseLu::solve called before factor");
        }
    }

    int n_ = 0;
    bool factored_ = false;
    std::vector<Scalar> lu_;
    std::vector<int> piv_;
};

template <class Scalar>
Scalar dense_lu_residual_inf_norm(
    int n,
    const std::vector<Scalar>& ArowMajor,
    const std::vector<Scalar>& x,
    const std::vector<Scalar>& b
) {
    const std::size_t nn = static_cast<std::size_t>(n);
    if (ArowMajor.size() != nn * nn || x.size() != nn || b.size() != nn) {
        throw std::runtime_error("dense_lu_residual_inf_norm size mismatch");
    }

    Scalar rinf = Scalar(0);

    for (int i = 0; i < n; ++i) {
        Scalar ax = Scalar(0);

        for (int j = 0; j < n; ++j) {
            ax += ArowMajor[static_cast<std::size_t>(i) * nn + static_cast<std::size_t>(j)]
                * x[static_cast<std::size_t>(j)];
        }

        rinf = std::max(rinf, Scalar(std::abs(ax - b[static_cast<std::size_t>(i)])));
    }

    return rinf;
}

template <class Scalar>
std::vector<Scalar> build_dirichlet_poisson_3d_dense_matrix(
    int nxNodes,
    int nyNodes,
    int nzNodes
) {
    if (nxNodes < 2 || nyNodes < 2 || nzNodes < 2) {
        throw std::runtime_error("build_dirichlet_poisson_3d_dense_matrix requires at least 2 nodes per direction");
    }

    const int n = nxNodes * nyNodes * nzNodes;
    std::vector<Scalar> A(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), Scalar(0));

    auto id = [=](int i, int j, int k) {
        return (k * nyNodes + j) * nxNodes + i;
    };

    auto boundary = [=](int i, int j, int k) {
        return i == 0 || j == 0 || k == 0 ||
               i == nxNodes - 1 || j == nyNodes - 1 || k == nzNodes - 1;
    };

    for (int k = 0; k < nzNodes; ++k) {
        for (int j = 0; j < nyNodes; ++j) {
            for (int i = 0; i < nxNodes; ++i) {
                const int row = id(i, j, k);
                const std::size_t r = static_cast<std::size_t>(row) * static_cast<std::size_t>(n);

                if (boundary(i, j, k)) {
                    A[r + static_cast<std::size_t>(row)] = Scalar(1);
                } else {
                    A[r + static_cast<std::size_t>(row)] = Scalar(6);
                    A[r + static_cast<std::size_t>(id(i - 1, j, k))] = Scalar(-1);
                    A[r + static_cast<std::size_t>(id(i + 1, j, k))] = Scalar(-1);
                    A[r + static_cast<std::size_t>(id(i, j - 1, k))] = Scalar(-1);
                    A[r + static_cast<std::size_t>(id(i, j + 1, k))] = Scalar(-1);
                    A[r + static_cast<std::size_t>(id(i, j, k - 1))] = Scalar(-1);
                    A[r + static_cast<std::size_t>(id(i, j, k + 1))] = Scalar(-1);
                }
            }
        }
    }

    return A;
}

} // namespace solvers
} // namespace memoirs
