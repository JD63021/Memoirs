#pragma once

// ============================================================================
// SECTION 6A: Rectangular CSR operators
// ============================================================================
//
// Needed for pressure-correction / SIMPLE-style Schur operators:
//
//     Lp = B * H * B^T
//
// B is generally rectangular: pressure rows x velocity rows.
// ============================================================================

#include <algorithm>
#include <sstream>
#include <unordered_map>
#include <vector>

struct RectCsrPatternHost {
    int nRows = 0;
    int nCols = 0;
    std::vector<int> rowPtr;
    std::vector<int> colInd;
    std::vector<std::unordered_map<int, int>> slot;

    int nnz() const { return rowPtr.empty() ? 0 : rowPtr.back(); }
};

struct RectCsrHost {
    RectCsrPatternHost pattern;
    std::vector<Real> values;

    int nRows() const { return pattern.nRows; }
    int nCols() const { return pattern.nCols; }
    int nnz() const { return pattern.nnz(); }
};

static inline RectCsrPatternHost rect_csr_make_pattern(
    int nRows,
    int nCols,
    std::vector<std::vector<int>> rowCols
) {
    if ((int)rowCols.size() != nRows) {
        throw std::runtime_error("rect_csr_make_pattern row count mismatch");
    }

    RectCsrPatternHost pat;
    pat.nRows = nRows;
    pat.nCols = nCols;
    pat.rowPtr.assign(nRows + 1, 0);
    pat.slot.resize(nRows);

    int nnz = 0;
    for (int i = 0; i < nRows; ++i) {
        auto& c = rowCols[i];

        for (int col : c) {
            if (col < 0 || col >= nCols) {
                std::ostringstream oss;
                oss << "rect_csr_make_pattern column out of range row=" << i
                    << " col=" << col << " nCols=" << nCols;
                throw std::runtime_error(oss.str());
            }
        }

        std::sort(c.begin(), c.end());
        c.erase(std::unique(c.begin(), c.end()), c.end());

        pat.rowPtr[i] = nnz;
        nnz += (int)c.size();
        pat.colInd.insert(pat.colInd.end(), c.begin(), c.end());

        pat.slot[i].reserve(c.size() * 2 + 1);
        for (int k = 0; k < (int)c.size(); ++k) {
            pat.slot[i][c[k]] = k;
        }
    }

    pat.rowPtr[nRows] = nnz;
    return pat;
}

static inline RectCsrHost rect_csr_make(
    int nRows,
    int nCols,
    std::vector<std::vector<int>> rowCols
) {
    RectCsrHost A;
    A.pattern = rect_csr_make_pattern(nRows, nCols, std::move(rowCols));
    A.values.assign(A.pattern.nnz(), Real(0));
    return A;
}

static inline void rect_csr_zero(RectCsrHost& A) {
    std::fill(A.values.begin(), A.values.end(), Real(0));
}

static inline int rect_csr_lookup_slot(const RectCsrHost& A, int row, int col) {
    if (row < 0 || row >= A.pattern.nRows) {
        throw std::runtime_error("rect_csr_lookup_slot row out of range");
    }

    auto it = A.pattern.slot[row].find(col);
    if (it == A.pattern.slot[row].end()) {
        std::ostringstream oss;
        oss << "RectCsrHost missing slot row=" << row << " col=" << col;
        throw std::runtime_error(oss.str());
    }

    return it->second;
}

static inline int rect_csr_flat_slot(const RectCsrHost& A, int row, int col) {
    return A.pattern.rowPtr[row] + rect_csr_lookup_slot(A, row, col);
}

static inline void rect_csr_add(RectCsrHost& A, int row, int col, Real value) {
    A.values[rect_csr_flat_slot(A, row, col)] += value;
}

static inline void rect_csr_set(RectCsrHost& A, int row, int col, Real value) {
    A.values[rect_csr_flat_slot(A, row, col)] = value;
}

static inline Real rect_csr_get(const RectCsrHost& A, int row, int col) {
    if (row < 0 || row >= A.pattern.nRows || col < 0 || col >= A.pattern.nCols) {
        return Real(0);
    }

    auto it = A.pattern.slot[row].find(col);
    if (it == A.pattern.slot[row].end()) return Real(0);

    return A.values[A.pattern.rowPtr[row] + it->second];
}

static inline Real rect_csr_matvec_row(
    const RectCsrHost& A,
    int row,
    const std::vector<Real>& x
) {
    if ((int)x.size() != A.pattern.nCols) {
        throw std::runtime_error("rect_csr_matvec_row x size mismatch");
    }

    Real s = Real(0);
    for (int k = A.pattern.rowPtr[row]; k < A.pattern.rowPtr[row + 1]; ++k) {
        s += A.values[k] * x[A.pattern.colInd[k]];
    }
    return s;
}

static inline std::vector<Real> rect_csr_matvec(
    const RectCsrHost& A,
    const std::vector<Real>& x
) {
    if ((int)x.size() != A.pattern.nCols) {
        throw std::runtime_error("rect_csr_matvec x size mismatch");
    }

    std::vector<Real> y(A.pattern.nRows, Real(0));
    for (int i = 0; i < A.pattern.nRows; ++i) {
        y[i] = rect_csr_matvec_row(A, i, x);
    }
    return y;
}

static inline std::vector<Real> rect_csr_transpose_matvec(
    const RectCsrHost& A,
    const std::vector<Real>& x
) {
    if ((int)x.size() != A.pattern.nRows) {
        throw std::runtime_error("rect_csr_transpose_matvec x size mismatch");
    }

    std::vector<Real> y(A.pattern.nCols, Real(0));
    for (int i = 0; i < A.pattern.nRows; ++i) {
        const Real xi = x[i];
        for (int k = A.pattern.rowPtr[i]; k < A.pattern.rowPtr[i + 1]; ++k) {
            y[A.pattern.colInd[k]] += A.values[k] * xi;
        }
    }
    return y;
}

static inline RectCsrHost rect_csr_transpose_pattern_only(const RectCsrHost& A) {
    std::vector<std::vector<int>> rowCols(A.pattern.nCols);

    for (int i = 0; i < A.pattern.nRows; ++i) {
        for (int k = A.pattern.rowPtr[i]; k < A.pattern.rowPtr[i + 1]; ++k) {
            rowCols[A.pattern.colInd[k]].push_back(i);
        }
    }

    return rect_csr_make(A.pattern.nCols, A.pattern.nRows, std::move(rowCols));
}

static inline RectCsrHost rect_csr_transpose(const RectCsrHost& A) {
    RectCsrHost AT = rect_csr_transpose_pattern_only(A);

    for (int i = 0; i < A.pattern.nRows; ++i) {
        for (int k = A.pattern.rowPtr[i]; k < A.pattern.rowPtr[i + 1]; ++k) {
            rect_csr_set(AT, A.pattern.colInd[k], i, A.values[k]);
        }
    }

    return AT;
}
