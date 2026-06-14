#pragma once

// ============================================================================
// SECTION 6B: Small local block inverses and B H B^T helpers
// ============================================================================
//
// The old DG2/DG1 SIMPLE pressure solve used a full local velocity mass inverse,
// not just a diagonal inverse.  This module captures that reusable pattern:
//
//     Lp = B * H * B^T
//
// where H can be a cell-local dense inverse.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <sstream>
#include <vector>

struct LocalBlockInverseHost {
    int nBlocks = 0;
    int blockSize = 0;

    // Block-major global dofs: globalDofs[block*blockSize + local]
    std::vector<int> globalDofs;

    // Row-major inverse blocks:
    // invBlocks[block*bs*bs + i*bs + j]
    std::vector<Real> invBlocks;

    int blockOffset(int b) const { return b * blockSize * blockSize; }
    int dofOffset(int b) const { return b * blockSize; }
};

static inline std::vector<Real> dense_inverse_gauss_jordan(
    const std::vector<Real>& A,
    int n,
    double pivotTol = 1.0e-30
) {
    if ((int)A.size() != n * n) {
        throw std::runtime_error("dense_inverse_gauss_jordan size mismatch");
    }

    std::vector<Real> aug(n * 2 * n, Real(0));
    auto at = [&](int r, int c) -> Real& { return aug[r * (2 * n) + c]; };

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) at(i, j) = A[i * n + j];
        at(i, n + i) = Real(1);
    }

    for (int col = 0; col < n; ++col) {
        int piv = col;
        double pivAbs = std::abs(double(at(piv, col)));

        for (int r = col + 1; r < n; ++r) {
            const double a = std::abs(double(at(r, col)));
            if (a > pivAbs) {
                pivAbs = a;
                piv = r;
            }
        }

        if (pivAbs <= pivotTol) {
            std::ostringstream oss;
            oss << "dense inverse singular block at column " << col
                << " pivotAbs=" << pivAbs;
            throw std::runtime_error(oss.str());
        }

        if (piv != col) {
            for (int c = 0; c < 2 * n; ++c) std::swap(at(piv, c), at(col, c));
        }

        const Real invPiv = Real(1) / at(col, col);
        for (int c = 0; c < 2 * n; ++c) at(col, c) *= invPiv;

        for (int r = 0; r < n; ++r) {
            if (r == col) continue;

            const Real f = at(r, col);
            if (f == Real(0)) continue;

            for (int c = 0; c < 2 * n; ++c) at(r, c) -= f * at(col, c);
        }
    }

    std::vector<Real> inv(n * n, Real(0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) inv[i * n + j] = at(i, n + j);
    }
    return inv;
}

static inline LocalBlockInverseHost local_block_inverse_from_dense_blocks(
    int blockSize,
    const std::vector<std::vector<int>>& blockDofs,
    const std::vector<std::vector<Real>>& denseBlocks,
    double pivotTol = 1.0e-30
) {
    if (blockSize <= 0) throw std::runtime_error("invalid blockSize");

    if (blockDofs.size() != denseBlocks.size()) {
        throw std::runtime_error("local_block_inverse_from_dense_blocks count mismatch");
    }

    LocalBlockInverseHost H;
    H.nBlocks = (int)blockDofs.size();
    H.blockSize = blockSize;
    H.globalDofs.resize(H.nBlocks * blockSize);
    H.invBlocks.resize(H.nBlocks * blockSize * blockSize);

    for (int b = 0; b < H.nBlocks; ++b) {
        if ((int)blockDofs[b].size() != blockSize) {
            throw std::runtime_error("local block dof size mismatch");
        }
        if ((int)denseBlocks[b].size() != blockSize * blockSize) {
            throw std::runtime_error("local dense block size mismatch");
        }

        std::copy(blockDofs[b].begin(), blockDofs[b].end(), H.globalDofs.begin() + H.dofOffset(b));

        auto inv = dense_inverse_gauss_jordan(denseBlocks[b], blockSize, pivotTol);
        std::copy(inv.begin(), inv.end(), H.invBlocks.begin() + H.blockOffset(b));
    }

    return H;
}

static inline LocalBlockInverseHost local_block_inverse_from_diagonal(
    const std::vector<Real>& diagonal,
    double zeroTol = 0.0
) {
    LocalBlockInverseHost H;
    H.nBlocks = (int)diagonal.size();
    H.blockSize = 1;
    H.globalDofs.resize(H.nBlocks);
    H.invBlocks.resize(H.nBlocks);

    for (int i = 0; i < H.nBlocks; ++i) {
        if (std::abs(double(diagonal[i])) <= zeroTol) {
            throw std::runtime_error("local_block_inverse_from_diagonal found zero diagonal");
        }

        H.globalDofs[i] = i;
        H.invBlocks[i] = Real(1) / diagonal[i];
    }

    return H;
}

static inline std::vector<Real> local_block_apply(
    const LocalBlockInverseHost& H,
    const std::vector<Real>& x,
    int globalSize
) {
    if ((int)x.size() != globalSize) {
        throw std::runtime_error("local_block_apply input size mismatch");
    }

    std::vector<Real> y(globalSize, Real(0));
    const int bs = H.blockSize;

    for (int blk = 0; blk < H.nBlocks; ++blk) {
        const int dof0 = H.dofOffset(blk);
        const int inv0 = H.blockOffset(blk);

        for (int i = 0; i < bs; ++i) {
            const int gi = H.globalDofs[dof0 + i];
            if (gi < 0 || gi >= globalSize) {
                throw std::runtime_error("local_block_apply output dof out of range");
            }

            Real s = Real(0);
            for (int j = 0; j < bs; ++j) {
                const int gj = H.globalDofs[dof0 + j];
                if (gj < 0 || gj >= globalSize) {
                    throw std::runtime_error("local_block_apply input dof out of range");
                }

                s += H.invBlocks[inv0 + i * bs + j] * x[gj];
            }

            y[gi] += s;
        }
    }

    return y;
}

static inline std::vector<std::vector<int>> build_BHBt_pressure_pattern(
    const RectCsrHost& B,
    const LocalBlockInverseHost& H
) {
    std::vector<std::vector<int>> rowCols(B.nRows());
    for (int i = 0; i < B.nRows(); ++i) rowCols[i].push_back(i);

    const int bs = H.blockSize;

    for (int blk = 0; blk < H.nBlocks; ++blk) {
        const int dof0 = H.dofOffset(blk);
        std::vector<int> pRows;

        for (int p = 0; p < B.nRows(); ++p) {
            bool touches = false;

            for (int k = B.pattern.rowPtr[p]; k < B.pattern.rowPtr[p + 1] && !touches; ++k) {
                const int v = B.pattern.colInd[k];

                for (int a = 0; a < bs; ++a) {
                    if (v == H.globalDofs[dof0 + a]) {
                        touches = true;
                        break;
                    }
                }
            }

            if (touches) pRows.push_back(p);
        }

        for (int r : pRows) {
            auto& cols = rowCols[r];
            cols.insert(cols.end(), pRows.begin(), pRows.end());
        }
    }

    return rowCols;
}

static inline void assemble_B_localBlockInverse_BT(
    const RectCsrHost& B,
    const LocalBlockInverseHost& H,
    SparseRows& L
) {
    // B has pressure rows x velocity rows.
    // H acts in velocity space.
    // L must already be initialized with a compatible pressure-pressure pattern.
    if (sparse_nrows(L) != B.nRows()) {
        throw std::runtime_error("assemble_B_localBlockInverse_BT row mismatch");
    }

    const int bs = H.blockSize;

    for (int blk = 0; blk < H.nBlocks; ++blk) {
        const int dof0 = H.dofOffset(blk);
        const int inv0 = H.blockOffset(blk);

        std::vector<int> pRows;

        for (int p = 0; p < B.nRows(); ++p) {
            bool touches = false;

            for (int k = B.pattern.rowPtr[p]; k < B.pattern.rowPtr[p + 1] && !touches; ++k) {
                const int v = B.pattern.colInd[k];

                for (int a = 0; a < bs; ++a) {
                    if (v == H.globalDofs[dof0 + a]) {
                        touches = true;
                        break;
                    }
                }
            }

            if (touches) pRows.push_back(p);
        }

        for (int pr : pRows) {
            for (int pc : pRows) {
                Real acc = Real(0);

                for (int a = 0; a < bs; ++a) {
                    const int va = H.globalDofs[dof0 + a];
                    const Real Bra = rect_csr_get(B, pr, va);
                    if (Bra == Real(0)) continue;

                    for (int b = 0; b < bs; ++b) {
                        const int vb = H.globalDofs[dof0 + b];
                        const Real Bcb = rect_csr_get(B, pc, vb);
                        if (Bcb == Real(0)) continue;

                        acc += Bra * H.invBlocks[inv0 + a * bs + b] * Bcb;
                    }
                }

                if (acc != Real(0)) sparse_add(L, pr, pc, acc);
            }
        }
    }
}
