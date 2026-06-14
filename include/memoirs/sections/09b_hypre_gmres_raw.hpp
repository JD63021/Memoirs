#pragma once

#if defined(MEMOIRS_USE_HYPRE)

// ============================================================================
// SECTION 09B: raw HYPRE IJ/ParCSR GMRES solve
// ============================================================================
//
// This is intentionally separate from the existing PCG/AMG path.
// It is used for mixed H(div) saddle systems:
//
//   [ M  -B^T ] [sigma] = [r]
//   [-B    0  ] [u    ]   [-f]
//
// which are indefinite and should not be sent to PCG.
// ============================================================================

static SolveReport solve_hypre_ij_gmres_raw(
    const SparseRows& A,
    const std::vector<Real>& b,
    const Options& opt,
    std::vector<Real>& xOut
) {
    SolveReport rep;

    auto elapsed = [](const std::chrono::steady_clock::time_point& a,
                      const std::chrono::steady_clock::time_point& b) -> double {
        return std::chrono::duration<double>(b - a).count();
    };

    int mpiWasInitialized = 0;
    MPI_Initialized(&mpiWasInitialized);

    bool startedMPI = false;
    if (!mpiWasInitialized) {
        int argc0 = 0;
        char** argv0 = nullptr;
        MPI_Init(&argc0, &argv0);
        startedMPI = true;
    }

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (size != 1) {
        throw std::runtime_error("memoirs raw GMRES HYPRE path currently supports one MPI rank.");
    }

    const HYPRE_MemoryLocation requestedMem = parse_hypre_memory_location(opt.hypreMemory);
    const bool useDevice = (requestedMem == HYPRE_MEMORY_DEVICE);

    hypre_check(HYPRE_Init(), "HYPRE_Init");

    if (useDevice) {
        hypre_check(HYPRE_SetMemoryLocation(HYPRE_MEMORY_DEVICE),
                    "HYPRE_SetMemoryLocation DEVICE");
        hypre_check(HYPRE_SetExecutionPolicy(HYPRE_EXEC_DEVICE),
                    "HYPRE_SetExecutionPolicy DEVICE");
    } else {
        hypre_check(HYPRE_SetMemoryLocation(HYPRE_MEMORY_HOST),
                    "HYPRE_SetMemoryLocation HOST");
        hypre_check(HYPRE_SetExecutionPolicy(HYPRE_EXEC_HOST),
                    "HYPRE_SetExecutionPolicy HOST");
    }

    const HYPRE_Int n = (HYPRE_Int)sparse_nrows(A);
    if ((HYPRE_Int)b.size() != n) {
        throw std::runtime_error("GMRES raw solve size mismatch.");
    }

    const HYPRE_Int ilower = 0;
    const HYPRE_Int iupper = n - 1;

    HYPRE_IJMatrix ijA = nullptr;
    HYPRE_ParCSRMatrix parA = nullptr;

    auto tMat0 = std::chrono::steady_clock::now();

    hypre_check(HYPRE_IJMatrixCreate(comm, ilower, iupper, ilower, iupper, &ijA),
                "HYPRE_IJMatrixCreate");
    hypre_check(HYPRE_IJMatrixSetObjectType(ijA, HYPRE_PARCSR),
                "HYPRE_IJMatrixSetObjectType");

    std::vector<HYPRE_Int> ijNcols(n);
    std::vector<HYPRE_Int> ijRows(n);
    std::vector<HYPRE_Int> ijCols;
    std::vector<HYPRE_Complex> ijVals;

    std::size_t nnz = sparse_nnz(A);
    ijCols.reserve(nnz);
    ijVals.reserve(nnz);

    for (HYPRE_Int i = 0; i < n; ++i) {
        ijRows[i] = i;

        if (A.fixedPattern) {
            const auto& cols = sparse_cols_row(A, i);
            ijNcols[i] = (HYPRE_Int)cols.size();

            for (int k = 0; k < (int)cols.size(); ++k) {
                ijCols.push_back((HYPRE_Int)cols[k]);
                ijVals.push_back((HYPRE_Complex)sparse_value_at(A, i, k));
            }
        } else {
            const auto& row = A.rows[i];
            ijNcols[i] = (HYPRE_Int)row.size();

            for (const auto& kv : row) {
                ijCols.push_back((HYPRE_Int)kv.first);
                ijVals.push_back((HYPRE_Complex)kv.second);
            }
        }
    }

    hypre_check(HYPRE_IJMatrixSetRowSizes(ijA, ijNcols.data()),
                "HYPRE_IJMatrixSetRowSizes");
    hypre_initialize_ij_matrix_for_host_insertion(ijA);

    hypre_check(HYPRE_IJMatrixSetValues(
        ijA,
        n,
        ijNcols.data(),
        ijRows.data(),
        ijCols.data(),
        ijVals.data()),
        "HYPRE_IJMatrixSetValues bulk");

    hypre_check(HYPRE_IJMatrixAssemble(ijA), "HYPRE_IJMatrixAssemble");

    auto tMat1 = std::chrono::steady_clock::now();
    rep.hypreMatrixInsertSeconds = elapsed(tMat0, tMat1);

    auto tMatMig0 = std::chrono::steady_clock::now();

    if (useDevice) {
        hypre_check(HYPRE_IJMatrixMigrate(ijA, HYPRE_MEMORY_DEVICE),
                    "HYPRE_IJMatrixMigrate DEVICE");
    }

    hypre_check(HYPRE_IJMatrixGetObject(ijA, (void**)&parA),
                "HYPRE_IJMatrixGetObject");

    auto tMatMig1 = std::chrono::steady_clock::now();
    rep.hypreMatrixMigrateSeconds = elapsed(tMatMig0, tMatMig1);

    HYPRE_IJVector ijb = nullptr;
    HYPRE_IJVector ijx = nullptr;
    HYPRE_ParVector parb = nullptr;
    HYPRE_ParVector parx = nullptr;

    std::vector<HYPRE_Int> rows(n);
    std::vector<HYPRE_Complex> bvals(n);
    std::vector<HYPRE_Complex> xvals(n, HYPRE_Complex(0));

    for (HYPRE_Int i = 0; i < n; ++i) {
        rows[i] = i;
        bvals[i] = (HYPRE_Complex)b[i];
    }

    auto tVec0 = std::chrono::steady_clock::now();

    hypre_check(HYPRE_IJVectorCreate(comm, ilower, iupper, &ijb),
                "HYPRE_IJVectorCreate b");
    hypre_check(HYPRE_IJVectorSetObjectType(ijb, HYPRE_PARCSR),
                "HYPRE_IJVectorSetObjectType b");
    hypre_initialize_ij_vector_for_host_insertion(ijb);

    hypre_check(HYPRE_IJVectorCreate(comm, ilower, iupper, &ijx),
                "HYPRE_IJVectorCreate x");
    hypre_check(HYPRE_IJVectorSetObjectType(ijx, HYPRE_PARCSR),
                "HYPRE_IJVectorSetObjectType x");
    hypre_initialize_ij_vector_for_host_insertion(ijx);

    hypre_check(HYPRE_IJVectorSetValues(ijb, n, rows.data(), bvals.data()),
                "HYPRE_IJVectorSetValues b");
    hypre_check(HYPRE_IJVectorSetValues(ijx, n, rows.data(), xvals.data()),
                "HYPRE_IJVectorSetValues x");

    hypre_check(HYPRE_IJVectorAssemble(ijb), "HYPRE_IJVectorAssemble b");
    hypre_check(HYPRE_IJVectorAssemble(ijx), "HYPRE_IJVectorAssemble x");

    auto tVec1 = std::chrono::steady_clock::now();
    rep.hypreVectorInsertSeconds = elapsed(tVec0, tVec1);

    auto tVecMig0 = std::chrono::steady_clock::now();

    if (useDevice) {
        hypre_check(HYPRE_IJVectorMigrate(ijb, HYPRE_MEMORY_DEVICE),
                    "HYPRE_IJVectorMigrate b DEVICE");
        hypre_check(HYPRE_IJVectorMigrate(ijx, HYPRE_MEMORY_DEVICE),
                    "HYPRE_IJVectorMigrate x DEVICE");
    }

    hypre_check(HYPRE_IJVectorGetObject(ijb, (void**)&parb),
                "HYPRE_IJVectorGetObject b");
    hypre_check(HYPRE_IJVectorGetObject(ijx, (void**)&parx),
                "HYPRE_IJVectorGetObject x");

    auto tVecMig1 = std::chrono::steady_clock::now();
    rep.hypreVectorMigrateSeconds = elapsed(tVecMig0, tVecMig1);

    HYPRE_Solver gmres = nullptr;
    HYPRE_Solver precond = nullptr;

    auto tSetup0 = std::chrono::steady_clock::now();

    hypre_check(HYPRE_ParCSRGMRESCreate(comm, &gmres), "HYPRE_ParCSRGMRESCreate");
    hypre_check(HYPRE_GMRESSetKDim(gmres, memoirs_env_int("MEMOIRS_GMRES_KDIM", 80)),
                "HYPRE_GMRESSetKDim");
    hypre_check(HYPRE_GMRESSetMaxIter(gmres, opt.maxit), "HYPRE_GMRESSetMaxIter");
    hypre_check(HYPRE_GMRESSetTol(gmres, opt.tol), "HYPRE_GMRESSetTol");
    hypre_check(HYPRE_GMRESSetPrintLevel(gmres, opt.hyprePrint),
                "HYPRE_GMRESSetPrintLevel");
    hypre_check(HYPRE_GMRESSetLogging(gmres, 1), "HYPRE_GMRESSetLogging");

    const std::string pre = lower_copy(opt.precond);

    if (pre == "diagscale") {
        hypre_check(HYPRE_GMRESSetPrecond(
            gmres,
            (HYPRE_PtrToSolverFcn)HYPRE_ParCSRDiagScale,
            (HYPRE_PtrToSolverFcn)HYPRE_ParCSRDiagScaleSetup,
            nullptr),
            "HYPRE_GMRESSetPrecond DiagScale");
    } else if (pre == "none") {
        // no preconditioner
    } else {
        throw std::runtime_error("RT0 mixed saddle currently supports -precond diagscale or none.");
    }

    hypre_check(HYPRE_ParCSRGMRESSetup(gmres, parA, parb, parx),
                "HYPRE_ParCSRGMRESSetup");

    auto tSetup1 = std::chrono::steady_clock::now();
    rep.hypreSetupSeconds = elapsed(tSetup0, tSetup1);

    auto tSolve0 = std::chrono::steady_clock::now();

    hypre_check(HYPRE_ParCSRGMRESSolve(gmres, parA, parb, parx),
                "HYPRE_ParCSRGMRESSolve");

    auto tSolve1 = std::chrono::steady_clock::now();
    rep.hypreSolveOnlySeconds = elapsed(tSolve0, tSolve1);
    rep.solveSeconds = rep.hypreSolveOnlySeconds;

    hypre_check(HYPRE_GMRESGetNumIterations(gmres, &rep.iterations),
                "HYPRE_GMRESGetNumIterations");
    {
        HYPRE_Real finalRelResHypre = HYPRE_Real(0);
        hypre_check(HYPRE_GMRESGetFinalRelativeResidualNorm(gmres, &finalRelResHypre),
                    "HYPRE_GMRESGetFinalRelativeResidualNorm");
        rep.finalRelRes = double(finalRelResHypre);
    }

    auto tGet0 = std::chrono::steady_clock::now();

    if (useDevice) {
        hypre_check(HYPRE_IJVectorMigrate(ijx, HYPRE_MEMORY_HOST),
                    "HYPRE_IJVectorMigrate x HOST");
    }

    std::vector<HYPRE_Complex> sol(n);
    hypre_check(HYPRE_IJVectorGetValues(ijx, n, rows.data(), sol.data()),
                "HYPRE_IJVectorGetValues x");

    xOut.assign(n, Real(0));
    for (HYPRE_Int i = 0; i < n; ++i) {
        xOut[i] = Real(sol[i]);
    }

    auto tGet1 = std::chrono::steady_clock::now();
    rep.hypreGetSolutionSeconds = elapsed(tGet0, tGet1);

    auto tDestroy0 = std::chrono::steady_clock::now();

    if (precond) {
        hypre_check(HYPRE_BoomerAMGDestroy(precond), "HYPRE_BoomerAMGDestroy");
    }

    hypre_check(HYPRE_ParCSRGMRESDestroy(gmres), "HYPRE_ParCSRGMRESDestroy");
    hypre_check(HYPRE_IJVectorDestroy(ijx), "HYPRE_IJVectorDestroy x");
    hypre_check(HYPRE_IJVectorDestroy(ijb), "HYPRE_IJVectorDestroy b");
    hypre_check(HYPRE_IJMatrixDestroy(ijA), "HYPRE_IJMatrixDestroy A");
    hypre_check(HYPRE_Finalize(), "HYPRE_Finalize");

    if (startedMPI) {
        MPI_Finalize();
    }

    auto tDestroy1 = std::chrono::steady_clock::now();
    rep.hypreDestroyFinalizeSeconds = elapsed(tDestroy0, tDestroy1);

    return rep;
}

#endif // MEMOIRS_USE_HYPRE
