#pragma once

// ============================================================================
// SECTION 6: CSR builder / sparse row storage
// ============================================================================
//
// Legacy mode:
//   rows[i] is std::map<int, Real>.
//
// Fixed-pattern mode:
//   pattern stores immutable CSR-like graph:
//     rowPtr
//     cols
//     slot maps col -> local slot
//
//   numeric values are stored flat:
//     flatVals[rowPtr[row] + localSlot]
//
// This is the bridge to GPU device assembly: CUDA kernels can fill flatVals
// directly using precomputed local element slot IDs.
// ============================================================================

#include <algorithm>
#include <map>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <vector>


struct CsrPatternHost {
    std::vector<int> rowPtr;
    std::vector<int> colInd;
};

struct CsrMatrixHost {
    CsrPatternHost pattern;
    std::vector<Real> values;
};

struct SparseFixedPattern {
    std::vector<int> rowPtr;
    std::vector<std::vector<int>> cols;
    std::vector<std::unordered_map<int, int>> slot;

    int nRows() const { return (int)cols.size(); }
    int nnz() const { return rowPtr.empty() ? 0 : rowPtr.back(); }
};

struct SparseRows {
    // Legacy path.
    std::vector<std::map<int, Real>> rows;

    // Fixed-pattern path.
    bool fixedPattern = false;

    // Compatibility graph for non-shared fixed pattern.
    std::vector<std::vector<int>> cols;
    std::vector<std::unordered_map<int, int>> slot;
    std::vector<int> rowPtr;

    // Preferred cached immutable graph.
    std::shared_ptr<const SparseFixedPattern> pattern;

    // Preferred numeric values for fixed pattern.
    std::vector<Real> flatVals;

    // Compatibility only. Kept so accidental old code can be found/removed.
    std::vector<std::vector<Real>> vals;
};

static inline int sparse_nrows(const SparseRows& A) {
    if (!A.fixedPattern) return (int)A.rows.size();
    if (A.pattern) return A.pattern->nRows();
    return (int)A.cols.size();
}

static inline int sparse_row_start(const SparseRows& A, int row) {
    if (A.pattern) return A.pattern->rowPtr[row];
    return A.rowPtr[row];
}

static inline int sparse_row_end(const SparseRows& A, int row) {
    if (A.pattern) return A.pattern->rowPtr[row + 1];
    return A.rowPtr[row + 1];
}

static inline int sparse_nnz_flat(const SparseRows& A) {
    if (!A.fixedPattern) return (int)sparse_nrows(A); // not meaningful for legacy
    if (A.pattern) return A.pattern->nnz();
    return A.rowPtr.empty() ? 0 : A.rowPtr.back();
}

static inline const std::vector<int>& sparse_cols_row(const SparseRows& A, int i) {
    if (A.pattern) return A.pattern->cols[i];
    return A.cols[i];
}

static inline const std::unordered_map<int, int>& sparse_slot_row(const SparseRows& A, int i) {
    if (A.pattern) return A.pattern->slot[i];
    return A.slot[i];
}

static inline std::shared_ptr<const SparseFixedPattern>
sparse_make_fixed_pattern(std::vector<std::vector<int>> rowCols) {
    auto pat = std::make_shared<SparseFixedPattern>();

    const int n = (int)rowCols.size();
    pat->cols.resize(n);
    pat->slot.resize(n);
    pat->rowPtr.assign(n + 1, 0);

    int nnz = 0;
    for (int i = 0; i < n; ++i) {
        auto& c = rowCols[i];
        std::sort(c.begin(), c.end());
        c.erase(std::unique(c.begin(), c.end()), c.end());

        pat->cols[i] = std::move(c);
        pat->rowPtr[i] = nnz;
        nnz += (int)pat->cols[i].size();

        pat->slot[i].reserve(pat->cols[i].size() * 2 + 1);
        for (int k = 0; k < (int)pat->cols[i].size(); ++k) {
            pat->slot[i][pat->cols[i][k]] = k;
        }
    }
    pat->rowPtr[n] = nnz;

    return pat;
}

static inline void sparse_init_from_fixed_pattern(
    SparseRows& A,
    std::shared_ptr<const SparseFixedPattern> pat
) {
    A.fixedPattern = true;
    A.rows.clear();

    A.pattern = std::move(pat);

    A.cols.clear();
    A.slot.clear();
    A.rowPtr.clear();
    A.vals.clear();

    A.flatVals.assign(A.pattern->nnz(), Real(0));
}

static inline void sparse_init_fixed_pattern(
    SparseRows& A,
    std::vector<std::vector<int>> rowCols
) {
    sparse_init_from_fixed_pattern(A, sparse_make_fixed_pattern(std::move(rowCols)));
}


static inline std::vector<std::vector<int>> build_scalar_cg_cell_row_columns(
    const PolyMesh& m,
    const LinearCgDofMap& dm
) {
    std::vector<std::vector<int>> rowCols(dm.nDofs);

    for (int i = 0; i < dm.nDofs; ++i) {
        rowCols[i].push_back(i);
    }

    auto add_clique = [&](const std::vector<int>& dofs) {
        for (int I : dofs) {
            if (I < 0 || I >= dm.nDofs) {
                throw std::runtime_error("CG sparsity builder found row dof out of range.");
            }
            auto& cols = rowCols[I];
            for (int J : dofs) {
                if (J < 0 || J >= dm.nDofs) {
                    throw std::runtime_error("CG sparsity builder found column dof out of range.");
                }
                cols.push_back(J);
            }
        }
    };

    if (dm.resolvedSpace == "cg_hex_q1") {
        for (const auto& c : m.cells) {
            const auto hv = ordered_hex_vertices_axis_aligned(m, c);
            std::vector<int> dofs(hv.begin(), hv.end());
            add_clique(dofs);
        }
    } else if (dm.resolvedSpace == "cg_tet_p1") {
        for (const auto& c : m.cells) {
            if (c.verts.size() != 4) {
                throw std::runtime_error("CG tet sparsity builder found non-tet cell.");
            }
            std::vector<int> dofs = {c.verts[0], c.verts[1], c.verts[2], c.verts[3]};
            add_clique(dofs);
        }
    } else {
        throw std::runtime_error("CG sparsity builder unsupported for space: " + dm.resolvedSpace);
    }

    return rowCols;
}

static inline void sparse_init_scalar_cg_fixed_pattern(
    SparseRows& A,
    const PolyMesh& m,
    const LinearCgDofMap& dm
) {
    sparse_init_fixed_pattern(A, build_scalar_cg_cell_row_columns(m, dm));
}

static inline void sparse_zero_values(SparseRows& A) {
    if (A.fixedPattern) {
        std::fill(A.flatVals.begin(), A.flatVals.end(), Real(0));
    } else {
        for (auto& r : A.rows) r.clear();
    }
}

static inline int sparse_lookup_slot(const SparseRows& A, int row, int col) {
    if (!A.fixedPattern) return -1;

    const auto& sr = sparse_slot_row(A, row);
    auto it = sr.find(col);
    if (it == sr.end()) {
        std::ostringstream oss;
        oss << "SparseRows fixed pattern missing slot row=" << row << " col=" << col;
        throw std::runtime_error(oss.str());
    }
    return it->second;
}

static inline int sparse_lookup_flat_slot(const SparseRows& A, int row, int col) {
    return sparse_row_start(A, row) + sparse_lookup_slot(A, row, col);
}

static inline void sparse_add_slot(SparseRows& A, int row, int slot, Real v) {
    if (slot < 0) {
        throw std::runtime_error("sparse_add_slot called with negative slot");
    }
    A.flatVals[sparse_row_start(A, row) + slot] += v;
}

static inline void sparse_add_flat_slot(SparseRows& A, int flatSlot, Real v) {
    if (flatSlot < 0 || flatSlot >= (int)A.flatVals.size()) {
        throw std::runtime_error("sparse_add_flat_slot called with invalid slot");
    }
    A.flatVals[flatSlot] += v;
}

static inline Real sparse_value_at(const SparseRows& A, int row, int localSlot) {
    return A.flatVals[sparse_row_start(A, row) + localSlot];
}

static inline void sparse_set_value_at(SparseRows& A, int row, int localSlot, Real v) {
    A.flatVals[sparse_row_start(A, row) + localSlot] = v;
}

static inline void sparse_add(SparseRows& A, int i, int j, Real v) {
    if (A.fixedPattern) {
        const int slot = sparse_lookup_slot(A, i, j);
        sparse_add_slot(A, i, slot, v);
    } else {
        A.rows[i][j] += v;
    }
}

static inline void sparse_set_row_identity(SparseRows& A, int row) {
    if (A.fixedPattern) {
        for (int k = sparse_row_start(A, row); k < sparse_row_end(A, row); ++k) {
            A.flatVals[k] = Real(0);
        }

        const int slot = sparse_lookup_slot(A, row, row);
        sparse_set_value_at(A, row, slot, Real(1));
    } else {
        A.rows[row].clear();
        A.rows[row][row] = Real(1);
    }
}

static inline std::size_t sparse_nnz(const SparseRows& A) {
    if (A.fixedPattern) return (std::size_t)sparse_nnz_flat(A);

    std::size_t n = 0;
    for (const auto& r : A.rows) n += r.size();
    return n;
}

static inline Real sparse_matvec_row(const SparseRows& A, int i, const std::vector<Real>& x) {
    Real s = Real(0);

    if (A.fixedPattern) {
        const auto& c = sparse_cols_row(A, i);
        for (int k = 0; k < (int)c.size(); ++k) {
            s += sparse_value_at(A, i, k) * x[c[k]];
        }
        return s;
    }

    for (const auto& kv : A.rows[i]) s += kv.second * x[kv.first];
    return s;
}


static inline bool sparse_get_value(const SparseRows& A, int row, int col, Real& value) {
    if (A.fixedPattern) {
        const auto& sr = sparse_slot_row(A, row);
        auto it = sr.find(col);
        if (it == sr.end()) {
            value = Real(0);
            return false;
        }
        value = sparse_value_at(A, row, it->second);
        return true;
    }

    auto it = A.rows[row].find(col);
    if (it == A.rows[row].end()) {
        value = Real(0);
        return false;
    }
    value = it->second;
    return true;
}

static inline double sparse_diag_abs_or_zero(const SparseRows& A, int row, bool& present) {
    Real v = Real(0);
    present = sparse_get_value(A, row, row, v);
    return std::abs(double(v));
}

static inline double sparse_symmetry_max_abs(const SparseRows& A) {
    const int n = sparse_nrows(A);
    double symMax = 0.0;

    if (A.fixedPattern) {
        for (int i = 0; i < n; ++i) {
            const auto& c = sparse_cols_row(A, i);
            for (int k = 0; k < (int)c.size(); ++k) {
                const int j = c[k];
                const double aij = double(sparse_value_at(A, i, k));
                Real ajiR = Real(0);
                sparse_get_value(A, j, i, ajiR);
                symMax = std::max(symMax, std::abs(aij - double(ajiR)));
            }
        }
        return symMax;
    }

    for (int i = 0; i < n; ++i) {
        for (const auto& kv : A.rows[i]) {
            int j = kv.first;
            double aij = double(kv.second);
            double aji = 0.0;
            auto it = A.rows[j].find(i);
            if (it != A.rows[j].end()) aji = double(it->second);
            symMax = std::max(symMax, std::abs(aij - aji));
        }
    }
    return symMax;
}

