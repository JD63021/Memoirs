#pragma once

#include "memoirs/operators/StructuredQ1PoissonOperator.hpp"
#include "memoirs/solvers/DenseLu.hpp"
#include "memoirs/structured/StructuredGrid3D.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace memoirs {
namespace solvers {

template <class Real>
class StructuredGmgPreconditioner {
public:
    using Grid = memoirs::structured::StructuredGrid3D;
    using Op = memoirs::operators::StructuredQ1PoissonOperator<Real>;

    struct Options {
        int preSmooth = 2;
        int postSmooth = 2;
        Real omega = Real(0.70);
        int coarseMaxDofs = 256;
        int maxLevels = 32;
        int verbose = 0;
    };

    struct Level {
        Grid grid;
        Op op;
        std::vector<Real> Ax;
        std::vector<Real> residual;
        std::vector<Real> coarseRhs;
        std::vector<Real> coarseCorrection;

        explicit Level(const Grid& g)
            : grid(g), op(g) {}
    };

    StructuredGmgPreconditioner() = default;

    void setup(const Grid& finest, const Options& opt) {
        opt_ = opt;
        levels_.clear();

        Grid g = finest;
        levels_.emplace_back(g);

        while (int(levels_.size()) < opt_.maxLevels &&
               g.n_nodes() > opt_.coarseMaxDofs &&
               g.can_coarsen_by_2()) {
            g = g.coarsened_by_2();
            levels_.emplace_back(g);
        }

        if (levels_.empty()) {
            throw std::runtime_error("StructuredGmgPreconditioner setup produced no levels");
        }

        for (std::size_t l = 0; l < levels_.size(); ++l) {
            Level& lev = levels_[l];
            lev.Ax.assign(lev.grid.size(), Real(0));
            lev.residual.assign(lev.grid.size(), Real(0));

            if (l + 1 < levels_.size()) {
                const Grid& cg = levels_[l + 1].grid;
                lev.coarseRhs.assign(cg.size(), Real(0));
                lev.coarseCorrection.assign(cg.size(), Real(0));
            }
        }

        const Level& coarse = levels_.back();
        const int nc = coarse.grid.n_nodes();
        std::vector<Real> Ac = coarse.op.build_dense_matrix();
        coarseLu_.factor(nc, Ac);

        if (opt_.verbose) {
            print_hierarchy(std::cout);
        }
    }

    void print_hierarchy(std::ostream& os) const {
        os << "--------------- structured GMG hierarchy ---------------\n";
        os << "gmgLevels                 = " << levels_.size() << "\n";
        os << "gmgPreSmooth              = " << opt_.preSmooth << "\n";
        os << "gmgPostSmooth             = " << opt_.postSmooth << "\n";
        os << "gmgJacobiOmega            = " << double(opt_.omega) << "\n";
        os << "gmgCoarseMaxDofs          = " << opt_.coarseMaxDofs << "\n";

        for (std::size_t l = 0; l < levels_.size(); ++l) {
            const auto& g = levels_[l].grid;
            os << "gmgLevel[" << l << "]              = "
               << g.label()
               << " dofs=" << g.n_nodes()
               << " h=(" << g.hx() << "," << g.hy() << "," << g.hz() << ")"
               << "\n";
        }

        const int nc = levels_.back().grid.n_nodes();
        const double coarseDenseMiB =
            double(nc) * double(nc) * double(sizeof(Real)) / (1024.0 * 1024.0);

        os << "gmgCoarseSolver           = dense_lu_cpu_exact\n";
        os << "gmgCoarseDofs             = " << nc << "\n";
        os << "gmgCoarseDenseMatrixMiB   = " << coarseDenseMiB << "\n";
        os << "--------------------------------------------------------\n";
    }

    int levels() const {
        return static_cast<int>(levels_.size());
    }

    const Grid& finest_grid() const {
        return levels_.front().grid;
    }

    void apply(const std::vector<Real>& rhs, std::vector<Real>& z) {
        if (levels_.empty()) {
            throw std::runtime_error("StructuredGmgPreconditioner::apply before setup");
        }

        if (rhs.size() != levels_.front().grid.size()) {
            throw std::runtime_error("StructuredGmgPreconditioner::apply RHS size mismatch");
        }

        z.assign(rhs.size(), Real(0));
        vcycle(0, rhs, z);
    }

private:
    void smooth(Level& lev,
                const std::vector<Real>& rhs,
                std::vector<Real>& x,
                int sweeps) {
        const Grid& g = lev.grid;

        for (int sweep = 0; sweep < sweeps; ++sweep) {
            lev.op.apply(x, lev.Ax);

            for (int k = 0; k < g.nz; ++k) {
                for (int j = 0; j < g.ny; ++j) {
                    for (int i = 0; i < g.nx; ++i) {
                        const int q = g.id(i, j, k);
                        const std::size_t qs = static_cast<std::size_t>(q);

                        const Real invD = lev.op.inverse_diagonal_value(i, j, k);
                        x[qs] += opt_.omega * invD * (rhs[qs] - lev.Ax[qs]);
                    }
                }
            }
        }
    }

    void compute_residual(Level& lev,
                          const std::vector<Real>& rhs,
                          const std::vector<Real>& x) {
        lev.op.apply(x, lev.Ax);

        for (std::size_t i = 0; i < rhs.size(); ++i) {
            lev.residual[i] = rhs[i] - lev.Ax[i];
        }
    }

    void restrict_full_weighting(const Grid& fg,
                                 const Grid& cg,
                                 const std::vector<Real>& rf,
                                 std::vector<Real>& rc) const {
        rc.assign(cg.size(), Real(0));

        const Real w1[3] = {Real(0.25), Real(0.5), Real(0.25)};

        for (int K = 0; K < cg.nz; ++K) {
            for (int J = 0; J < cg.ny; ++J) {
                for (int I = 0; I < cg.nx; ++I) {
                    const int qc = cg.id(I, J, K);
                    const std::size_t qcs = static_cast<std::size_t>(qc);

                    // Error equation has zero Dirichlet boundary correction.
                    if (cg.is_boundary(I, J, K)) {
                        rc[qcs] = Real(0);
                        continue;
                    }

                    const int ic = 2 * I;
                    const int jc = 2 * J;
                    const int kc = 2 * K;

                    Real sum = Real(0);

                    for (int dk = -1; dk <= 1; ++dk) {
                        for (int dj = -1; dj <= 1; ++dj) {
                            for (int di = -1; di <= 1; ++di) {
                                const int i = ic + di;
                                const int j = jc + dj;
                                const int k = kc + dk;

                                const Real w =
                                    w1[di + 1] * w1[dj + 1] * w1[dk + 1];

                                sum += w * rf[static_cast<std::size_t>(fg.id(i, j, k))];
                            }
                        }
                    }

                    rc[qcs] = sum;
                }
            }
        }
    }

    void prolongate_trilinear_add(const Grid& cg,
                                  const Grid& fg,
                                  const std::vector<Real>& ec,
                                  std::vector<Real>& xf) const {
        for (int k = 0; k < fg.nz; ++k) {
            for (int j = 0; j < fg.ny; ++j) {
                for (int i = 0; i < fg.nx; ++i) {
                    if (fg.is_boundary(i, j, k)) {
                        continue;
                    }

                    const int qf = fg.id(i, j, k);
                    const std::size_t qfs = static_cast<std::size_t>(qf);

                    const int I0 = i / 2;
                    const int J0 = j / 2;
                    const int K0 = k / 2;

                    const int oi = i & 1;
                    const int oj = j & 1;
                    const int ok = k & 1;

                    Real val = Real(0);

                    for (int dk = 0; dk <= ok; ++dk) {
                        const int K = K0 + dk;
                        const Real wk = ok ? Real(0.5) : Real(1);

                        for (int dj = 0; dj <= oj; ++dj) {
                            const int J = J0 + dj;
                            const Real wj = oj ? Real(0.5) : Real(1);

                            for (int di = 0; di <= oi; ++di) {
                                const int I = I0 + di;
                                const Real wi = oi ? Real(0.5) : Real(1);

                                val += wi * wj * wk *
                                       ec[static_cast<std::size_t>(cg.id(I, J, K))];
                            }
                        }
                    }

                    xf[qfs] += val;
                }
            }
        }
    }

    void vcycle(std::size_t l,
                const std::vector<Real>& rhs,
                std::vector<Real>& x) {
        Level& lev = levels_[l];

        if (l + 1 == levels_.size()) {
            coarseLu_.solve(rhs, x);
            return;
        }

        smooth(lev, rhs, x, opt_.preSmooth);
        compute_residual(lev, rhs, x);

        Level& clev = levels_[l + 1];
        restrict_full_weighting(lev.grid, clev.grid, lev.residual, lev.coarseRhs);

        std::fill(lev.coarseCorrection.begin(), lev.coarseCorrection.end(), Real(0));
        vcycle(l + 1, lev.coarseRhs, lev.coarseCorrection);

        prolongate_trilinear_add(clev.grid, lev.grid, lev.coarseCorrection, x);
        smooth(lev, rhs, x, opt_.postSmooth);
    }

    Options opt_;
    std::vector<Level> levels_;
    DenseLu<Real> coarseLu_;
};

} // namespace solvers
} // namespace memoirs
