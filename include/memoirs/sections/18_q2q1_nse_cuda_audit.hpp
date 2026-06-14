#include "memoirs/sections/00_common.hpp"
#include <vector>
#pragma once

// Q2/Q1 CUDA assembly scaffold.

struct Q2Q1StructuredGrid;

//
// This is deliberately a smoke layer only. It does not assemble FE physics yet.
// It verifies that the new Q2/Q1 HYPRE-IJ target can use a CUDA source to fill
// the fixed-pattern flat values and RHS buffers, then solve through the same
// HYPRE IJ/ParCSR path as Q1/Q1.

struct Q2Q1CudaIdentityFillReport {
    long long rows = 0;
    long long nnz = 0;
    double zeroSeconds = -1.0;
    double diagSeconds = -1.0;
    double copyBackSeconds = -1.0;
    double totalSeconds = -1.0;
};

struct SparseRows;

Q2Q1CudaIdentityFillReport q2q1_cuda_fill_identity_values_rhs(
    const Q2Q1StructuredGrid& g,
    SparseRows& A,
    std::vector<Real>& b
);


struct Q2Q1CellSlotCache;

struct Q2Q1CudaStokesFillReport {
    long long rows = 0;
    long long nnz = 0;
    long long cells = 0;
    long long slotCount = 0;
    double zeroSeconds = -1.0;
    double cellKernelSeconds = -1.0;
    double copyBackSeconds = -1.0;
    double totalSeconds = -1.0;
};

Q2Q1CudaStokesFillReport q2q1_cuda_fill_stokes_pspg_continuous_rhs(
    const Q2Q1StructuredGrid& g,
    const Q2Q1CellSlotCache& slots,
    SparseRows& A,
    std::vector<Real>& b,
    double nu,
    double tau,
    int continuousRhs
);
