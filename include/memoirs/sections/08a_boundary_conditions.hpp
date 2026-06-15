#pragma once
#include <string>
#include <stdexcept>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>

// ============================================================================
// SECTION 8a: Boundary condition application
// ============================================================================
// First modular extraction from the one-shot Poisson assembler.  CG Poisson
// uses strong Dirichlet elimination.  DG/SIPG will use a separate weak boundary
// integrator and should not call this function.
// ============================================================================

struct StrongDirichletData {
    std::vector<int> dofs;
    std::vector<Real> values;
};


static inline bool memoirs_env_flag_dirichlet_audit(const char* name, bool def = false) {
    const char* v = std::getenv(name);
    if (!v) return def;
    std::string s(v);
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return char(std::tolower(c)); });
    return !(s.empty() || s == "0" || s == "false" || s == "off" || s == "no");
}

static inline void audit_strong_dirichlet_after_apply(
    const StrongDirichletData& bc,
    const SparseRows& A,
    const std::vector<Real>& b
) {
    if (!memoirs_env_flag_dirichlet_audit("MEMOIRS_DIRICHLET_AUDIT", false)) return;

    const int n = sparse_nrows(A);
    std::vector<char> isDirichlet(n, 0);
    for (int d : bc.dofs) {
        if (d >= 0 && d < n) isDirichlet[d] = 1;
    }

    int badBoundaryColumns = 0;
    double maxBoundaryColumnAbs = 0.0;
    int badBoundaryRows = 0;
    double minDiag = std::numeric_limits<double>::infinity();
    double maxDiag = 0.0;
    double maxAbsEntry = 0.0;
    int nonFiniteEntries = 0;
    int nonFiniteRhs = 0;

    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(double(b[i]))) ++nonFiniteRhs;

        bool diagPresent = false;
        const double dabs = sparse_diag_abs_or_zero(A, i, diagPresent);
        if (diagPresent) {
            minDiag = std::min(minDiag, dabs);
            maxDiag = std::max(maxDiag, dabs);
        }

        if (A.fixedPattern) {
            const auto& cols = sparse_cols_row(A, i);
            int rowNnzAbs = 0;
            for (int k = 0; k < (int)cols.size(); ++k) {
                const int j = cols[k];
                const double aij = double(sparse_value_at(A, i, k));
                if (!std::isfinite(aij)) ++nonFiniteEntries;
                const double aa = std::abs(aij);
                maxAbsEntry = std::max(maxAbsEntry, aa);
                if (aa > 0.0) ++rowNnzAbs;
                if (!isDirichlet[i] && j >= 0 && j < n && isDirichlet[j] && aa > 0.0) {
                    ++badBoundaryColumns;
                    maxBoundaryColumnAbs = std::max(maxBoundaryColumnAbs, aa);
                }
            }
            if (isDirichlet[i]) {
                Real aii = Real(0);
                sparse_get_value(A, i, i, aii);
                if (std::abs(double(aii) - 1.0) > 1.0e-12 || rowNnzAbs != 1) {
                    ++badBoundaryRows;
                }
            }
        } else {
            int rowNnzAbs = 0;
            for (const auto& kv : A.rows[i]) {
                const int j = kv.first;
                const double aij = double(kv.second);
                if (!std::isfinite(aij)) ++nonFiniteEntries;
                const double aa = std::abs(aij);
                maxAbsEntry = std::max(maxAbsEntry, aa);
                if (aa > 0.0) ++rowNnzAbs;
                if (!isDirichlet[i] && j >= 0 && j < n && isDirichlet[j] && aa > 0.0) {
                    ++badBoundaryColumns;
                    maxBoundaryColumnAbs = std::max(maxBoundaryColumnAbs, aa);
                }
            }
            if (isDirichlet[i]) {
                auto it = A.rows[i].find(i);
                const double aii = (it == A.rows[i].end()) ? 0.0 : double(it->second);
                if (std::abs(aii - 1.0) > 1.0e-12 || rowNnzAbs != 1) {
                    ++badBoundaryRows;
                }
            }
        }
    }

    const double symMax = sparse_symmetry_max_abs(A);
    std::cerr << "---------------- Dirichlet SPD audit ----------------\n";
    std::cerr << "dirichletDofs              = " << bc.dofs.size() << "\n";
    std::cerr << "badBoundaryColumns         = " << badBoundaryColumns << "\n";
    std::cerr << "maxBoundaryColumnAbs       = " << maxBoundaryColumnAbs << "\n";
    std::cerr << "badBoundaryRows            = " << badBoundaryRows << "\n";
    std::cerr << "matrixSymmetryMaxAbs       = " << symMax << "\n";
    std::cerr << "matrixDiagAbsMinMax        = " << minDiag << " / " << maxDiag << "\n";
    std::cerr << "matrixMaxAbsEntry          = " << maxAbsEntry << "\n";
    std::cerr << "matrixNonFiniteEntries     = " << nonFiniteEntries << "\n";
    std::cerr << "rhsNonFiniteEntries        = " << nonFiniteRhs << "\n";
    std::cerr << "-----------------------------------------------------\n";
}

static StrongDirichletData build_mms_strong_dirichlet_data(
    const PolyMesh& m,
    const LinearCgDofMap& dm,
    const std::string& mms
) {
    StrongDirichletData bc;
    bc.dofs.reserve(dm.boundaryDofs.size());
    bc.values.reserve(dm.boundaryDofs.size());

    for (int d : dm.boundaryDofs) {
        if (d < 0 || d >= dm.nDofs) {
            throw std::runtime_error("Dirichlet dof out of range while building MMS BC data.");
        }
        bc.dofs.push_back(d);
        bc.values.push_back(Real(mms_exact_value(cg_dof_coordinate(m, dm, d), mms)));
    }
    return bc;
}

static void apply_strong_dirichlet_by_dof_values(
    const StrongDirichletData& bc,
    SparseRows& A,
    std::vector<Real>& b
) {
    const int n = sparse_nrows(A);
    if ((int)b.size() != n) {
        throw std::runtime_error("apply_strong_dirichlet_by_dof_values size mismatch.");
    }
    if (bc.dofs.size() != bc.values.size()) {
        throw std::runtime_error("StrongDirichletData dof/value size mismatch.");
    }

    std::vector<char> isDirichlet(n, 0);
    std::vector<Real> g(n, Real(0));

    for (std::size_t k = 0; k < bc.dofs.size(); ++k) {
        const int d = bc.dofs[k];
        if (d < 0 || d >= n) throw std::runtime_error("Dirichlet dof out of range.");
        isDirichlet[d] = 1;
        g[d] = bc.values[k];
    }

    std::string mode = "symmetric_elimination";
    if (const char* e = std::getenv("MEMOIRS_STRONG_DIRICHLET_MODE")) {
        if (*e) mode = std::string(e);
    }

    const bool rowIdentity =
        mode == "row_identity" ||
        mode == "row" ||
        mode == "row_only" ||
        mode == "constraint" ||
        mode == "matlab";

    const bool symmetricElimination =
        mode == "symmetric_elimination" ||
        mode == "symmetric" ||
        mode == "spd" ||
        mode == "default";

    if (!rowIdentity && !symmetricElimination) {
        throw std::runtime_error(
            "Unsupported MEMOIRS_STRONG_DIRICHLET_MODE='" + mode +
            "'. Valid values: symmetric_elimination, row_identity."
        );
    }

    if (rowIdentity) {
        // Algebraic row-identity strong constraint:
        //   A[d,:] = 0, A[d,d] = 1, b[d] = g[d].
        // This intentionally leaves A[i,d] in interior rows.  It enforces the
        // boundary value strongly but the full matrix is generally nonsymmetric.
        // This mode is meant for diagnostics and for nonsymmetric Krylov solvers.
        if (A.fixedPattern) {
            for (int d : bc.dofs) {
                sparse_set_row_identity(A, d);
                b[d] = g[d];
            }
        } else {
            for (int d : bc.dofs) {
                A.rows[d].clear();
                A.rows[d][d] = Real(1);
                b[d] = g[d];
            }
        }
        return;
    }

    // Symmetry-preserving strong elimination:
    //   b_i <- b_i - A_i,d g_d
    //   A_i,d <- 0 for all interior rows i and boundary dofs d
    //   A[d,:] <- e_d^T, b_d <- g_d
    // The boundary coupling contribution is retained on the RHS.
    if (A.fixedPattern) {
        for (int i = 0; i < n; ++i) {
            if (isDirichlet[i]) continue;

            const auto& cols = sparse_cols_row(A, i);
            for (int k = 0; k < (int)cols.size(); ++k) {
                const int j = cols[k];
                if (j >= 0 && j < n && isDirichlet[j]) {
                    const Real aij = sparse_value_at(A, i, k);
                    b[i] -= aij * g[j];
                    sparse_set_value_at(A, i, k, Real(0));
                }
            }
        }

        for (int d : bc.dofs) {
            sparse_set_row_identity(A, d);
            b[d] = g[d];
        }
    } else {
        for (int i = 0; i < n; ++i) {
            if (isDirichlet[i]) continue;

            auto& row = A.rows[i];
            for (auto it = row.begin(); it != row.end(); ) {
                const int j = it->first;
                if (j >= 0 && j < n && isDirichlet[j]) {
                    b[i] -= it->second * g[j];
                    it = row.erase(it);
                } else {
                    ++it;
                }
            }
        }

        for (int d : bc.dofs) {
            A.rows[d].clear();
            A.rows[d][d] = Real(1);
            b[d] = g[d];
        }
    }
}

static void apply_mms_strong_dirichlet_all_boundary(
    const PolyMesh& m,
    const LinearCgDofMap& dm,
    const std::string& mms,
    SparseRows& A,
    std::vector<Real>& b
) {
    apply_strong_dirichlet_by_dof_values(
        build_mms_strong_dirichlet_data(m, dm, mms),
        A,
        b
    );
}
