#pragma once

#include "memoirs/structured/StructuredGrid3D.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace memoirs {
namespace operators {

template <class Real>
class StructuredQ1PoissonOperator {
public:
    using Grid = memoirs::structured::StructuredGrid3D;

    explicit StructuredQ1PoissonOperator(const Grid& g)
        : grid_(g) {
        setup_weights();
    }

    const Grid& grid() const { return grid_; }

    int size() const {
        return grid_.n_nodes();
    }

    Real diagonal_value(int i, int j, int k) const {
        if (grid_.is_boundary(i, j, k)) {
            return Real(1);
        }
        return diagInterior_;
    }

    Real inverse_diagonal_value(int i, int j, int k) const {
        return Real(1) / diagonal_value(i, j, k);
    }

    void apply(const std::vector<Real>& x, std::vector<Real>& y) const {
        const std::size_t n = grid_.size();

        if (x.size() != n) {
            throw std::runtime_error("StructuredQ1PoissonOperator::apply size mismatch");
        }

        y.assign(n, Real(0));

        for (int k = 0; k < grid_.nz; ++k) {
            for (int j = 0; j < grid_.ny; ++j) {
                for (int i = 0; i < grid_.nx; ++i) {
                    const int row = grid_.id(i, j, k);
                    const std::size_t r = static_cast<std::size_t>(row);

                    if (grid_.is_boundary(i, j, k)) {
                        // Strong Dirichlet row, identity.
                        y[r] = x[r];
                        continue;
                    }

                    Real sum = Real(0);

                    for (int dk = -1; dk <= 1; ++dk) {
                        const int kk = k + dk;
                        for (int dj = -1; dj <= 1; ++dj) {
                            const int jj = j + dj;
                            for (int di = -1; di <= 1; ++di) {
                                const int ii = i + di;

                                // Symmetric Dirichlet elimination:
                                // interior rows do not keep boundary columns.
                                if (grid_.is_boundary(ii, jj, kk)) {
                                    continue;
                                }

                                const int col = grid_.id(ii, jj, kk);
                                sum += weight(di, dj, dk) *
                                       x[static_cast<std::size_t>(col)];
                            }
                        }
                    }

                    y[r] = sum;
                }
            }
        }
    }

    std::vector<Real> build_dense_matrix() const {
        const int n = size();
        std::vector<Real> A(static_cast<std::size_t>(n) * static_cast<std::size_t>(n),
                            Real(0));
        std::vector<Real> e(static_cast<std::size_t>(n), Real(0));
        std::vector<Real> Ae;

        for (int c = 0; c < n; ++c) {
            std::fill(e.begin(), e.end(), Real(0));
            e[static_cast<std::size_t>(c)] = Real(1);

            apply(e, Ae);

            for (int r = 0; r < n; ++r) {
                A[static_cast<std::size_t>(r) * static_cast<std::size_t>(n) +
                  static_cast<std::size_t>(c)] = Ae[static_cast<std::size_t>(r)];
            }
        }

        return A;
    }

    double estimated_matrix_free_vector_mib(int nVectors) const {
        return double(size()) * double(sizeof(Real)) * double(nVectors)
             / (1024.0 * 1024.0);
    }

private:
    void setup_weights() {
        const Real hx = Real(grid_.hx());
        const Real hy = Real(grid_.hy());
        const Real hz = Real(grid_.hz());

        cx_ = hy * hz / (Real(2) * hx);
        cy_ = hx * hz / (Real(2) * hy);
        cz_ = hx * hy / (Real(2) * hz);

        // Assembled 1D tensor factors for interior nodes.
        K_[0] = Real(-0.5); K_[1] = Real(1.0);     K_[2] = Real(-0.5);
        M_[0] = Real( 1.0 / 3.0);
        M_[1] = Real( 4.0 / 3.0);
        M_[2] = Real( 1.0 / 3.0);

        diagInterior_ =
            cx_ * K_[1] * M_[1] * M_[1] +
            cy_ * M_[1] * K_[1] * M_[1] +
            cz_ * M_[1] * M_[1] * K_[1];
    }

    Real weight(int di, int dj, int dk) const {
        const int ai = di + 1;
        const int aj = dj + 1;
        const int ak = dk + 1;

        return cx_ * K_[ai] * M_[aj] * M_[ak] +
               cy_ * M_[ai] * K_[aj] * M_[ak] +
               cz_ * M_[ai] * M_[aj] * K_[ak];
    }

    Grid grid_;

    Real cx_ = Real(0);
    Real cy_ = Real(0);
    Real cz_ = Real(0);

    Real K_[3] = {};
    Real M_[3] = {};

    Real diagInterior_ = Real(0);
};

} // namespace operators
} // namespace memoirs
