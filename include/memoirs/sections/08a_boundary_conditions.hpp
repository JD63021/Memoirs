#pragma once

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

static StrongDirichletData build_mms_strong_dirichlet_data(
    const PolyMesh& m,
    const LinearCgDofMap& dm,
    const std::string& mms
) {
    StrongDirichletData bc;
    bc.dofs.reserve(dm.boundaryDofs.size());
    bc.values.reserve(dm.boundaryDofs.size());

    for (int d : dm.boundaryDofs) {
        if (d < 0 || d >= (int)m.points.size()) {
            throw std::runtime_error("Dirichlet dof out of range while building MMS BC data.");
        }
        bc.dofs.push_back(d);
        bc.values.push_back(Real(mms_exact_value(m.points[d], mms)));
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

    // Row-wise elimination:
    //   b_i <- b_i - A_i,d g_d
    //   remove/zero boundary columns from interior rows
    //   set boundary rows to identity.
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
