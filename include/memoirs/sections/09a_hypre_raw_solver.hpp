#pragma once

#if defined(MEMOIRS_USE_HYPRE)

// Reusable raw sparse solve: A x = b.
// This intentionally does not know about CG/DG, basis, BCs, or MMS.
// Apps/operators compute their own errors after x copyback.

static SolveReport solve_hypre_ij_pcg_raw(
    const SparseRows& A,
    const std::vector<Real>& b,
    const Options& opt,
    std::vector<Real>& xOut
) {
    if (lower_copy(opt.solver) != "pcg") {
        throw std::runtime_error("solve_hypre_ij_pcg_raw supports only -solver pcg");
    }

    SolveReport rep;

    auto elapsed = [](const auto& a, const auto& b) {
        return std::chrono::duration<double>(b - a).count();
    };

    const int nHost = sparse_nrows(A);
    if ((int)b.size() != nHost) {
        throw std::runtime_error("solve_hypre_ij_pcg_raw size mismatch.");
    }

    int mpiWasInitialized = 0;
    MPI_Initialized(&mpiWasInitialized);

    bool startedMPI = false;
    if (!mpiWasInitialized) {
        int argc = 0;
        char** argv = nullptr;
        MPI_Init(&argc, &argv);
        startedMPI = true;
    }

    MPI_Comm comm = MPI_COMM_WORLD;

    int size = 1;
    MPI_Comm_size(comm, &size);
    if (size != 1) {
        throw std::runtime_error("solve_hypre_ij_pcg_raw currently supports one MPI rank only.");
    }

    const HYPRE_MemoryLocation requestedMem = parse_hypre_memory_location(opt.hypreMemory);
    const bool useDevice = (requestedMem == HYPRE_MEMORY_DEVICE);

    hypre_check(HYPRE_Init(), "HYPRE_Init raw");

    if (useDevice) {
        hypre_check(HYPRE_SetMemoryLocation(HYPRE_MEMORY_DEVICE), "HYPRE_SetMemoryLocation DEVICE raw");
        hypre_check(HYPRE_SetExecutionPolicy(HYPRE_EXEC_DEVICE), "HYPRE_SetExecutionPolicy DEVICE raw");
    } else {
        hypre_check(HYPRE_SetMemoryLocation(HYPRE_MEMORY_HOST), "HYPRE_SetMemoryLocation HOST raw");
        hypre_check(HYPRE_SetExecutionPolicy(HYPRE_EXEC_HOST), "HYPRE_SetExecutionPolicy HOST raw");
    }

    const HYPRE_Int n = (HYPRE_Int)nHost;
    const HYPRE_BigInt ilower = 0;
    const HYPRE_BigInt iupper = n - 1;

    HYPRE_IJMatrix ijA = nullptr;
    HYPRE_ParCSRMatrix parA = nullptr;

    auto tMat0 = std::chrono::steady_clock::now();

    hypre_check(HYPRE_IJMatrixCreate(comm, ilower, iupper, ilower, iupper, &ijA), "HYPRE_IJMatrixCreate raw");
    hypre_check(HYPRE_IJMatrixSetObjectType(ijA, HYPRE_PARCSR), "HYPRE_IJMatrixSetObjectType raw");

    std::vector<HYPRE_Int> rows(n);
    std::vector<HYPRE_Int> ncols(n);
    std::vector<HYPRE_Int> cols;
    std::vector<HYPRE_Complex> vals;

    cols.reserve(sparse_nnz(A));
    vals.reserve(sparse_nnz(A));

    for (HYPRE_Int i = 0; i < n; ++i) {
        rows[i] = i;

        if (A.fixedPattern) {
            const auto& c = sparse_cols_row(A, i);
            ncols[i] = (HYPRE_Int)c.size();

            for (int k = 0; k < (int)c.size(); ++k) {
                cols.push_back((HYPRE_Int)c[k]);
                vals.push_back((HYPRE_Complex)sparse_value_at(A, i, k));
            }
        } else {
            ncols[i] = (HYPRE_Int)A.rows[i].size();

            for (const auto& kv : A.rows[i]) {
                cols.push_back((HYPRE_Int)kv.first);
                vals.push_back((HYPRE_Complex)kv.second);
            }
        }
    }

    hypre_check(HYPRE_IJMatrixSetRowSizes(ijA, ncols.data()), "HYPRE_IJMatrixSetRowSizes raw");
    hypre_initialize_ij_matrix_for_host_insertion(ijA);

    hypre_check(
        HYPRE_IJMatrixSetValues(ijA, n, ncols.data(), rows.data(), cols.data(), vals.data()),
        "HYPRE_IJMatrixSetValues raw"
    );

    hypre_check(HYPRE_IJMatrixAssemble(ijA), "HYPRE_IJMatrixAssemble raw");

    auto tMat1 = std::chrono::steady_clock::now();
    rep.hypreMatrixInsertSeconds = elapsed(tMat0, tMat1);

    auto tMatMig0 = std::chrono::steady_clock::now();

    if (useDevice) {
        hypre_check(HYPRE_IJMatrixMigrate(ijA, HYPRE_MEMORY_DEVICE), "HYPRE_IJMatrixMigrate raw");
    }

    hypre_check(HYPRE_IJMatrixGetObject(ijA, (void**)&parA), "HYPRE_IJMatrixGetObject raw");

    auto tMatMig1 = std::chrono::steady_clock::now();
    rep.hypreMatrixMigrateSeconds = elapsed(tMatMig0, tMatMig1);

    HYPRE_IJVector ijb = nullptr;
    HYPRE_IJVector ijx = nullptr;
    HYPRE_ParVector parb = nullptr;
    HYPRE_ParVector parx = nullptr;

    std::vector<HYPRE_Complex> bvals(n);
    std::vector<HYPRE_Complex> xvals(n, HYPRE_Complex(0));

    for (HYPRE_Int i = 0; i < n; ++i) {
        bvals[i] = (HYPRE_Complex)b[i];
    }

    auto tVec0 = std::chrono::steady_clock::now();

    hypre_check(HYPRE_IJVectorCreate(comm, ilower, iupper, &ijb), "HYPRE_IJVectorCreate b raw");
    hypre_check(HYPRE_IJVectorSetObjectType(ijb, HYPRE_PARCSR), "HYPRE_IJVectorSetObjectType b raw");
    hypre_initialize_ij_vector_for_host_insertion(ijb);

    hypre_check(HYPRE_IJVectorCreate(comm, ilower, iupper, &ijx), "HYPRE_IJVectorCreate x raw");
    hypre_check(HYPRE_IJVectorSetObjectType(ijx, HYPRE_PARCSR), "HYPRE_IJVectorSetObjectType x raw");
    hypre_initialize_ij_vector_for_host_insertion(ijx);

    hypre_check(HYPRE_IJVectorSetValues(ijb, n, rows.data(), bvals.data()), "HYPRE_IJVectorSetValues b raw");
    hypre_check(HYPRE_IJVectorSetValues(ijx, n, rows.data(), xvals.data()), "HYPRE_IJVectorSetValues x raw");

    hypre_check(HYPRE_IJVectorAssemble(ijb), "HYPRE_IJVectorAssemble b raw");
    hypre_check(HYPRE_IJVectorAssemble(ijx), "HYPRE_IJVectorAssemble x raw");

    auto tVec1 = std::chrono::steady_clock::now();
    rep.hypreVectorInsertSeconds = elapsed(tVec0, tVec1);

    auto tVecMig0 = std::chrono::steady_clock::now();

    if (useDevice) {
        hypre_check(HYPRE_IJVectorMigrate(ijb, HYPRE_MEMORY_DEVICE), "HYPRE_IJVectorMigrate b raw");
        hypre_check(HYPRE_IJVectorMigrate(ijx, HYPRE_MEMORY_DEVICE), "HYPRE_IJVectorMigrate x raw");
    }

    hypre_check(HYPRE_IJVectorGetObject(ijb, (void**)&parb), "HYPRE_IJVectorGetObject b raw");
    hypre_check(HYPRE_IJVectorGetObject(ijx, (void**)&parx), "HYPRE_IJVectorGetObject x raw");

    auto tVecMig1 = std::chrono::steady_clock::now();
    rep.hypreVectorMigrateSeconds = elapsed(tVecMig0, tVecMig1);

    HYPRE_Solver pcg = nullptr;
    HYPRE_Solver precond = nullptr;

    auto tSetup0 = std::chrono::steady_clock::now();

    hypre_check(HYPRE_ParCSRPCGCreate(comm, &pcg), "HYPRE_ParCSRPCGCreate raw");
    hypre_check(HYPRE_PCGSetMaxIter(pcg, opt.maxit), "HYPRE_PCGSetMaxIter raw");
    hypre_check(HYPRE_PCGSetTol(pcg, opt.tol), "HYPRE_PCGSetTol raw");
    hypre_check(HYPRE_PCGSetTwoNorm(pcg, 1), "HYPRE_PCGSetTwoNorm raw");
    hypre_check(HYPRE_PCGSetPrintLevel(pcg, opt.hyprePrint), "HYPRE_PCGSetPrintLevel raw");
    hypre_check(HYPRE_PCGSetLogging(pcg, 1), "HYPRE_PCGSetLogging raw");

    const std::string pre = lower_copy(opt.precond);

    if (pre == "diagscale") {
        hypre_check(
            HYPRE_PCGSetPrecond(
                pcg,
                (HYPRE_PtrToSolverFcn)HYPRE_ParCSRDiagScale,
                (HYPRE_PtrToSolverFcn)HYPRE_ParCSRDiagScaleSetup,
                nullptr
            ),
            "HYPRE_PCGSetPrecond diag raw"
        );
    } else if (pre == "amg" || pre == "boomeramg") {
        hypre_check(HYPRE_BoomerAMGCreate(&precond), "HYPRE_BoomerAMGCreate raw");
        hypre_check(HYPRE_BoomerAMGSetPrintLevel(precond, opt.hyprePrint), "HYPRE_BoomerAMGSetPrintLevel raw");
        hypre_check(HYPRE_BoomerAMGSetCoarsenType(precond, memoirs_env_int("MEMOIRS_AMG_COARSEN", 8)), "HYPRE_BoomerAMGSetCoarsenType raw");
        hypre_check(HYPRE_BoomerAMGSetInterpType(precond, memoirs_env_int("MEMOIRS_AMG_INTERP", 6)), "HYPRE_BoomerAMGSetInterpType raw");
        hypre_check(HYPRE_BoomerAMGSetRelaxType(precond, memoirs_env_int("MEMOIRS_AMG_RELAX", 18)), "HYPRE_BoomerAMGSetRelaxType raw");
        hypre_check(HYPRE_BoomerAMGSetNumSweeps(precond, memoirs_env_int("MEMOIRS_AMG_SWEEPS", 1)), "HYPRE_BoomerAMGSetNumSweeps raw");
        hypre_check(HYPRE_BoomerAMGSetTol(precond, 0.0), "HYPRE_BoomerAMGSetTol raw");
        hypre_check(HYPRE_BoomerAMGSetMaxIter(precond, 1), "HYPRE_BoomerAMGSetMaxIter raw");
        hypre_check(HYPRE_BoomerAMGSetRelaxOrder(precond, 0), "HYPRE_BoomerAMGSetRelaxOrder raw");
        hypre_check(HYPRE_BoomerAMGSetPMaxElmts(precond, memoirs_env_int("MEMOIRS_AMG_PMAX", 4)), "HYPRE_BoomerAMGSetPMaxElmts raw");
        hypre_check(HYPRE_BoomerAMGSetKeepTranspose(precond, memoirs_env_int("MEMOIRS_AMG_KEEP_TRANSPOSE", 1)), "HYPRE_BoomerAMGSetKeepTranspose raw");
        hypre_check(HYPRE_BoomerAMGSetTruncFactor(precond, memoirs_env_double("MEMOIRS_AMG_TRUNC", 0.0)), "HYPRE_BoomerAMGSetTruncFactor raw");
        hypre_check(HYPRE_BoomerAMGSetRAP2(precond, memoirs_env_int("MEMOIRS_AMG_RAP2", 0)), "HYPRE_BoomerAMGSetRAP2 raw");

        const int agg = memoirs_env_int("MEMOIRS_AMG_AGG_LEVELS", 0);
        if (agg > 0) {
            hypre_check(HYPRE_BoomerAMGSetAggNumLevels(precond, agg), "HYPRE_BoomerAMGSetAggNumLevels raw");
        }

        const double strong = memoirs_env_double("MEMOIRS_AMG_STRONG", -1.0);
        if (strong >= 0.0) {
            hypre_check(HYPRE_BoomerAMGSetStrongThreshold(precond, strong), "HYPRE_BoomerAMGSetStrongThreshold raw");
        }

        hypre_check(
            HYPRE_PCGSetPrecond(
                pcg,
                (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSolve,
                (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSetup,
                precond
            ),
            "HYPRE_PCGSetPrecond AMG raw"
        );
    } else if (pre == "none") {
        // no preconditioner
    } else {
        throw std::runtime_error("Unsupported -precond in raw solver: " + opt.precond);
    }

    hypre_check(HYPRE_ParCSRPCGSetup(pcg, parA, parb, parx), "HYPRE_ParCSRPCGSetup raw");

    auto tSetup1 = std::chrono::steady_clock::now();
    rep.hypreSetupSeconds = elapsed(tSetup0, tSetup1);

    auto tSolve0 = std::chrono::steady_clock::now();

    hypre_check(HYPRE_ParVectorSetConstantValues(parx, HYPRE_Complex(0)), "HYPRE_ParVectorSetConstantValues raw");
    hypre_check(HYPRE_ParCSRPCGSolve(pcg, parA, parb, parx), "HYPRE_ParCSRPCGSolve raw");

    auto tSolve1 = std::chrono::steady_clock::now();
    rep.hypreSolveOnlySeconds = elapsed(tSolve0, tSolve1);
    rep.hypreSolveOnlyAvgSeconds = rep.hypreSolveOnlySeconds;

    hypre_check(HYPRE_PCGGetNumIterations(pcg, &rep.iterations), "HYPRE_PCGGetNumIterations raw");

    HYPRE_Real finalRel = 0.0;
    hypre_check(HYPRE_PCGGetFinalRelativeResidualNorm(pcg, &finalRel), "HYPRE_PCGGetFinalRelativeResidualNorm raw");
    rep.finalRelRes = (double)finalRel;

    auto tGet0 = std::chrono::steady_clock::now();

    if (useDevice) {
        hypre_check(HYPRE_IJVectorMigrate(ijx, HYPRE_MEMORY_HOST), "HYPRE_IJVectorMigrate x HOST raw");
    }

    std::fill(xvals.begin(), xvals.end(), HYPRE_Complex(0));
    hypre_check(HYPRE_IJVectorGetValues(ijx, n, rows.data(), xvals.data()), "HYPRE_IJVectorGetValues x raw");

    xOut.assign(n, Real(0));
    for (HYPRE_Int i = 0; i < n; ++i) {
        xOut[i] = Real(xvals[i]);
    }

    auto tGet1 = std::chrono::steady_clock::now();
    rep.hypreGetSolutionSeconds = elapsed(tGet0, tGet1);

    auto tDestroy0 = std::chrono::steady_clock::now();

    if (precond) hypre_check(HYPRE_BoomerAMGDestroy(precond), "HYPRE_BoomerAMGDestroy raw");
    hypre_check(HYPRE_ParCSRPCGDestroy(pcg), "HYPRE_ParCSRPCGDestroy raw");
    hypre_check(HYPRE_IJVectorDestroy(ijx), "HYPRE_IJVectorDestroy x raw");
    hypre_check(HYPRE_IJVectorDestroy(ijb), "HYPRE_IJVectorDestroy b raw");
    hypre_check(HYPRE_IJMatrixDestroy(ijA), "HYPRE_IJMatrixDestroy A raw");
    hypre_check(HYPRE_Finalize(), "HYPRE_Finalize raw");

    if (startedMPI) MPI_Finalize();

    auto tDestroy1 = std::chrono::steady_clock::now();
    rep.hypreDestroyFinalizeSeconds = elapsed(tDestroy0, tDestroy1);

    return rep;
}

#endif
