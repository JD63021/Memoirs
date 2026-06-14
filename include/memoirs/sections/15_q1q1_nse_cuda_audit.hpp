#pragma once

// CUDA assembly audit for Q1/Q1 transient Picard NSE.
// This declaration expects the previous section headers to already be included.

void q1q1_cuda_audit_nse_assembly(
    const PolyMesh& mesh,
    const LinearCgDofMap& dm,
    const std::vector<Real>& beta,
    const Q1Q1NsePicardOptions& opt,
    const Q1Q1AlgInfo& info,
    AssembledSystem& hostSys
);


void q1q1_cuda_assemble_cavity_bdf1(
    const PolyMesh& mesh,
    const LinearCgDofMap& dm,
    const std::vector<Real>& beta,
    const std::vector<Real>& oldState,
    const Q1Q1NsePicardOptions& opt,
    const Q1Q1AlgInfo& info,
    double lidUx,
    double lidUy,
    double lidUz,
    AssembledSystem& hostSys
);



void q1q1_cuda_assemble_cavity_bdf_history(
    const PolyMesh& mesh,
    const LinearCgDofMap& dm,
    const std::vector<Real>& beta,
    const std::vector<Real>& oldState,
    const std::vector<Real>& olderState,
    const Q1Q1NsePicardOptions& opt,
    const Q1Q1AlgInfo& info,
    double lidUx,
    double lidUy,
    double lidUz,
    bool useBdf2,
    AssembledSystem& hostSys
);



struct Q1Q1NonlinearResidualReport {
    double absL2 = -1.0;
    double rhsL2 = -1.0;
    double axL2 = -1.0;
    double relL2 = -1.0;
    double seconds = -1.0;
};

Q1Q1NonlinearResidualReport q1q1_cuda_cavity_nonlinear_residual(
    const PolyMesh& mesh,
    const LinearCgDofMap& dm,
    const std::vector<Real>& solution,
    const std::vector<Real>& oldState,
    const std::vector<Real>& olderState,
    const Q1Q1NsePicardOptions& opt,
    const Q1Q1AlgInfo& info,
    double lidUx,
    double lidUy,
    double lidUz,
    bool useBdf2,
    const AssembledSystem& patternSys
);

