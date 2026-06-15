#pragma once

#if defined(MEMOIRS_USE_HYPRE)

struct HypreIjCachedPattern {
    bool valid = false;
    const void* patternPtr = nullptr;
    HYPRE_Int n = 0;
    std::size_t nnz = 0;

    std::vector<HYPRE_Int> rows;
    std::vector<HYPRE_Int> ncols;
    std::vector<HYPRE_Int> cols;
    std::vector<HYPRE_Complex> vals;
};

static inline HypreIjCachedPattern& hypre_ij_cached_pattern_singleton() {
    static HypreIjCachedPattern cache;
    return cache;
}

static inline const void* hypre_sparse_pattern_identity(const SparseRows& A) {
    if (A.pattern) return (const void*)A.pattern.get();
    if (A.fixedPattern && !A.cols.empty()) return (const void*)A.cols.data();
    return nullptr;
}

static inline bool hypre_try_build_cached_ij_pattern(
    const SparseRows& A,
    HypreIjCachedPattern& cache
) {
    if (!A.fixedPattern) return false;

    const void* pid = hypre_sparse_pattern_identity(A);
    const HYPRE_Int n = (HYPRE_Int)sparse_nrows(A);
    const std::size_t nnz = sparse_nnz(A);

    if (cache.valid &&
        cache.patternPtr == pid &&
        cache.n == n &&
        cache.nnz == nnz) {
        return true;
    }

    cache.valid = true;
    cache.patternPtr = pid;
    cache.n = n;
    cache.nnz = nnz;

    cache.rows.resize(n);
    cache.ncols.resize(n);
    cache.cols.clear();
    cache.cols.reserve(nnz);
    cache.vals.resize(nnz);

    for (HYPRE_Int i = 0; i < n; ++i) {
        const auto& c = sparse_cols_row(A, i);
        cache.rows[i] = i;
        cache.ncols[i] = (HYPRE_Int)c.size();
        for (int k = 0; k < (int)c.size(); ++k) {
            cache.cols.push_back((HYPRE_Int)c[k]);
        }
    }

    return true;
}

static inline void hypre_fill_cached_ij_values(
    const SparseRows& A,
    HypreIjCachedPattern& cache
) {
    if (!A.fixedPattern || !cache.valid) {
        throw std::runtime_error("hypre_fill_cached_ij_values requires fixed-pattern cache");
    }

    const HYPRE_Int n = (HYPRE_Int)sparse_nrows(A);
    std::size_t pos = 0;

    for (HYPRE_Int i = 0; i < n; ++i) {
        const auto& c = sparse_cols_row(A, i);
        for (int k = 0; k < (int)c.size(); ++k) {
            cache.vals[pos++] = (HYPRE_Complex)sparse_value_at(A, i, k);
        }
    }
}


#endif // defined(MEMOIRS_USE_HYPRE)




// Auto-split from frozen patch014 one-shot source.
// This header is intentionally included in order by apps/poisson_mms.cpp.

// ============================================================================
// SECTION 9: HYPRE IJ/ParCSR solve
// ============================================================================
//
// Current scope:
//   - Serial run only: mpirun -n 1 or direct execution.
//   - HYPRE IJMatrix / IJVector -> ParCSR.
//   - PCG with none / DiagScale / BoomerAMG preconditioner.
//   - Host-side insertion for one-shot correctness.
// Future GPU-resident direction:
//   - preserve IJ/ParCSR backend
//   - avoid repeated matrix/vector reconstruction across NSE loops
//   - move to device-resident HYPRE memory path where supported
// ============================================================================

struct SolveReport {
    int iterations = -1;
    double finalRelRes = -1.0;
    double nodalErrorL2 = -1.0;
    double nodalErrorMax = -1.0;
    double L2Error = -1.0;
    double H1SemiError = -1.0;
    double assemblySeconds = -1.0;
    double solveSeconds = -1.0;

    // HYPRE backend breakdown.
    // These are intentionally kept in the solve report so this one-shot code
    // can later split cleanly into solvers/hypre_ij.{h,cpp}.
    double hypreMatrixInsertSeconds = -1.0;
    double hypreMatrixMigrateSeconds = -1.0;
    double hypreVectorInsertSeconds = -1.0;
    double hypreVectorMigrateSeconds = -1.0;
    double hypreSetupSeconds = -1.0;
    int solveRepeats = 1;
    std::string rhsUpdateMode = "none";

    int amgCoarsenType = 8;
    int amgInterpType = 6;
    int amgRelaxType = 18;
    int amgAggLevels = 0;
    int amgKeepTranspose = 1;
    int amgPmax = 4;
    int amgNumSweeps = 1;
    int amgRAP2 = 0;
    double amgTruncFactor = 0.0;
    double amgStrongThreshold = -1.0;

    int ijUseRowSizes = 1;
    int ijBulkInsert = 1;

    double rhsUpdateSeconds = 0.0;
    double rhsUpdateAvgSeconds = 0.0;
    double hypreSolveOnlySeconds = -1.0;
    double hypreSolveOnlyAvgSeconds = -1.0;
    double hypreGetSolutionSeconds = -1.0;
    double errorNormSeconds = -1.0;
    double hypreDestroyFinalizeSeconds = -1.0;
};

static void compute_nodal_error(
    const PolyMesh& m,
    const LinearCgDofMap& dm,
    const std::string& mms,
    const std::vector<Real>& x,
    SolveReport& rep
) {
    double e2 = 0.0;
    double emax = 0.0;
    for (int i = 0; i < (int)x.size(); ++i) {
        const double ex = mms_exact_value(cg_dof_coordinate(m, dm, i), mms);
        const double e = double(x[i]) - ex;
        e2 += e*e;
        emax = std::max(emax, std::abs(e));
    }
    rep.nodalErrorL2 = std::sqrt(e2);
    rep.nodalErrorMax = emax;
}

static ErrorNorms compute_error_norms(
    const PolyMesh& m,
    const LinearCgDofMap& dm,
    const std::string& mms,
    const std::vector<Real>& x
) {
    ErrorNorms en;

    // Nodal diagnostics.
    double nodal2 = 0.0;
    double nodalMax = 0.0;
    for (int i = 0; i < (int)x.size(); ++i) {
        const double ex = mms_exact_value(cg_dof_coordinate(m, dm, i), mms);
        const double e = double(x[i]) - ex;
        nodal2 += e*e;
        nodalMax = std::max(nodalMax, std::abs(e));
    }
    en.nodalL2 = std::sqrt(nodal2);
    en.nodalMax = nodalMax;

    double l2 = 0.0;
    double h1 = 0.0;

    if (dm.resolvedSpace == "cg_hex_q1") {
        for (const auto& c : m.cells) {
            const auto hv = ordered_hex_vertices_axis_aligned(m, c);

            for (const auto& qpt : hex_q1_error_quadrature_3x3x3()) {
                const HexQ1BasisAtPoint basis = eval_hex_q1_basis_at(qpt.xi, qpt.eta, qpt.zeta);

                const Vec3 xq = map_hex_q1_to_physical(m, hv, basis.N);
                const Mat3 J = jacobian_hex_q1(m, hv, basis.dNref);
                const double detJ = std::abs(det3(J));
                const Mat3 invJ = inv3(J);

                double uh = 0.0;
                Vec3 guh;
                for (int aLoc = 0; aLoc < 8; ++aLoc) {
                    const double xa = double(x[hv[aLoc]]);
                    uh += basis.N[aLoc] * xa;

                    const Vec3 g = invJT_mul(invJ, basis.dNref[aLoc]);
                    guh.x += xa * g.x;
                    guh.y += xa * g.y;
                    guh.z += xa * g.z;
                }

                const double ue = mms_exact_value(xq, mms);
                const Vec3 ge = mms_exact_grad(xq, mms);
                const double eu = uh - ue;
                const Vec3 eg = {guh.x - ge.x, guh.y - ge.y, guh.z - ge.z};

                l2 += qpt.weight * detJ * eu * eu;
                h1 += qpt.weight * detJ * dot3(eg, eg);
            }
        }
    } else if (dm.resolvedSpace == "cg_tet_p1") {
        for (const auto& c : m.cells) {
            if (c.verts.size() != 4) throw std::runtime_error("Tet error norm cell does not have 4 vertices.");

            std::array<int,4> tv = {c.verts[0], c.verts[1], c.verts[2], c.verts[3]};
            const Mat3 J = jacobian_tet_p1(m, tv);
            const double detJ = std::abs(det3(J));
            const Mat3 invJ = inv3(J);
            const TetP1BasisAtPoint basis0 = eval_tet_p1_basis_at(0.25, 0.25, 0.25);

            std::array<Vec3,4> grad;
            Vec3 guh;
            for (int aLoc = 0; aLoc < 4; ++aLoc) {
                grad[aLoc] = invJT_mul(invJ, basis0.dNref[aLoc]);
                const double xa = double(x[tv[aLoc]]);
                guh.x += xa * grad[aLoc].x;
                guh.y += xa * grad[aLoc].y;
                guh.z += xa * grad[aLoc].z;
            }

            for (const auto& qpt : tet_p1_error_quadrature()) {
                const TetP1BasisAtPoint basis = eval_tet_p1_basis_at(qpt.xi, qpt.eta, qpt.zeta);
                const Vec3 xq = map_tet_p1_to_physical(m, tv, basis.N);

                double uh = 0.0;
                for (int aLoc = 0; aLoc < 4; ++aLoc) {
                    uh += basis.N[aLoc] * double(x[tv[aLoc]]);
                }

                const double ue = mms_exact_value(xq, mms);
                const Vec3 ge = mms_exact_grad(xq, mms);
                const double eu = uh - ue;
                const Vec3 eg = {guh.x - ge.x, guh.y - ge.y, guh.z - ge.z};

                l2 += qpt.weight * detJ * eu * eu;
                h1 += qpt.weight * detJ * dot3(eg, eg);
            }
        }
    } else if (dm.resolvedSpace == "cg_tet_p2") {
        for (int cidx = 0; cidx < (int)m.cells.size(); ++cidx) {
            const auto& c = m.cells[cidx];
            if (c.verts.size() != 4) throw std::runtime_error("Tet P2 error norm cell does not have 4 vertices.");
            if (dm.cellDofs[cidx].size() != 10) throw std::runtime_error("Tet P2 error norm cell does not have 10 dofs.");

            std::array<int,4> tv = {c.verts[0], c.verts[1], c.verts[2], c.verts[3]};
            const Mat3 J = jacobian_tet_p1(m, tv);
            const double detJ = std::abs(det3(J));
            const Mat3 invJ = inv3(J);

            for (const auto& qpt : tet_volume_quadrature_by_order(4)) {
                const TetP2BasisAtPoint basis = eval_tet_p2_basis_at(qpt.xi, qpt.eta, qpt.zeta);
                const TetP1BasisAtPoint geomBasis = eval_tet_p1_basis_at(qpt.xi, qpt.eta, qpt.zeta);
                const Vec3 xq = map_tet_p1_to_physical(m, tv, geomBasis.N);

                double uh = 0.0;
                Vec3 guh;
                for (int aLoc = 0; aLoc < 10; ++aLoc) {
                    const double xa = double(x[dm.cellDofs[cidx][aLoc]]);
                    uh += basis.N[aLoc] * xa;
                    const Vec3 g = invJT_mul(invJ, basis.dNref[aLoc]);
                    guh.x += xa * g.x;
                    guh.y += xa * g.y;
                    guh.z += xa * g.z;
                }

                const double ue = mms_exact_value(xq, mms);
                const Vec3 ge = mms_exact_grad(xq, mms);
                const double eu = uh - ue;
                const Vec3 eg = {guh.x - ge.x, guh.y - ge.y, guh.z - ge.z};

                l2 += qpt.weight * detJ * eu * eu;
                h1 += qpt.weight * detJ * dot3(eg, eg);
            }
        }
    } else {
        throw std::runtime_error("Error norm unsupported for space: " + dm.resolvedSpace);
    }

    en.L2 = std::sqrt(l2);
    en.H1Semi = std::sqrt(h1);
    return en;
}

#if defined(MEMOIRS_USE_HYPRE)

static void hypre_check(int ierr, const std::string& where) {
    if (ierr) {
        std::ostringstream oss;
        oss << "HYPRE error " << ierr << " at " << where;
        throw std::runtime_error(oss.str());
    }
}


static inline bool memoirs_allow_hypre_solve_error_256() {
    const char* v = std::getenv("MEMOIRS_ALLOW_HYPRE_ERROR_256");
    if (!v) return false;
    std::string s(v);
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return char(std::tolower(c)); });
    return !(s.empty() || s == "0" || s == "false" || s == "off" || s == "no");
}

static inline void hypre_check_pcg_solve(int ierr, const std::string& where) {
    if (!ierr) return;
    if (ierr == 256 && memoirs_allow_hypre_solve_error_256()) {
        std::cerr << "WARNING: HYPRE returned error/status 256 at " << where
                  << "; continuing because MEMOIRS_ALLOW_HYPRE_ERROR_256=1.\n";
        // HYPRE stores error flags globally; clear them so follow-up getter calls
        // such as HYPRE_PCGGetNumIterations do not fail only because the solve
        // reported non-convergence/status 256.
        HYPRE_ClearAllErrors();
        return;
    }
    hypre_check(ierr, where);
}

static inline void hypre_check_pcg_getter(int ierr, const std::string& where) {
    if (!ierr) return;
    if (memoirs_allow_hypre_solve_error_256()) {
        std::cerr << "WARNING: HYPRE getter returned error/status " << ierr
                  << " at " << where
                  << "; clearing and continuing because MEMOIRS_ALLOW_HYPRE_ERROR_256=1.\n";
        HYPRE_ClearAllErrors();
        return;
    }
    hypre_check(ierr, where);
}

static HYPRE_MemoryLocation parse_hypre_memory_location(const std::string& s) {
    const std::string m = lower_copy(s);
    if (m == "host" || m == "cpu") return HYPRE_MEMORY_HOST;
    if (m == "device" || m == "gpu" || m == "cuda") return HYPRE_MEMORY_DEVICE;
    throw std::runtime_error("Unsupported -hypreMemory: " + s);
}

static const char* hypre_memory_location_name(HYPRE_MemoryLocation loc) {
    if (loc == HYPRE_MEMORY_HOST) return "host";
    if (loc == HYPRE_MEMORY_DEVICE) return "device";
    return "unknown";
}

static void hypre_initialize_ij_matrix_for_host_insertion(HYPRE_IJMatrix A) {
    // Important:
    // HYPRE_MEMORY_HOST is an enum value, not necessarily a preprocessor macro.
    // Do not guard this with #if defined(HYPRE_MEMORY_HOST).
    //
    // We assemble/insert from host std::vectors in the one-shot code, even when
    // the final solve is device-resident. Therefore IJ initialization must
    // explicitly use HOST memory for insertion, then we migrate after assemble.
    hypre_check(HYPRE_IJMatrixInitialize_v2(A, HYPRE_MEMORY_HOST),
                "HYPRE_IJMatrixInitialize_v2 HOST");
}

static void hypre_initialize_ij_vector_for_host_insertion(HYPRE_IJVector v) {
    // Same reasoning as matrix initialization: values are inserted from host
    // std::vectors first, then migrated to device if requested.
    hypre_check(HYPRE_IJVectorInitialize_v2(v, HYPRE_MEMORY_HOST),
                "HYPRE_IJVectorInitialize_v2 HOST");
}

static SolveReport solve_hypre_ij_pcg(
    const PolyMesh& m,
    const LinearCgDofMap& dm,
    const AssembledSystem& sys,
    const std::string& mms,
    const Options& opt,
    std::vector<Real>& xOut
) {
    const std::string solverName = lower_copy(opt.solver);
    if (!(solverName == "pcg" || solverName == "gmres")) {
        throw std::runtime_error("Supported -solver values in scalar HYPRE path: pcg, gmres");
    }
    const bool useGmres = (solverName == "gmres");
    const int gmresRestart = memoirs_env_int("MEMOIRS_GMRES_RESTART", 80);

    SolveReport rep;
    const bool computeError = memoirs_compute_error_enabled();
    const int solveRepeats = memoirs_solve_repeats();
    const std::string rhsUpdateMode = memoirs_rhs_update_mode();
    if (!(rhsUpdateMode == "none" ||
          rhsUpdateMode == "scale_host" ||
          rhsUpdateMode == "scale_device_inplace")) {
        throw std::runtime_error("Unsupported MEMOIRS_RHS_UPDATE_MODE: " + rhsUpdateMode);
    }
    rep.solveRepeats = solveRepeats;
    rep.rhsUpdateMode = rhsUpdateMode;

    // DG2/DG1 pressure-solver-inspired HYPRE controls.
    //
    // Defaults match the robust pressure AMG settings we used there:
    //   coarsen=8, interp=6, relax=18, aggLevels=0,
    //   keepTranspose=1, pmax=4, sweeps=1.
    //
    // These are environment-controlled for quick memory/robustness sweeps
    // without disturbing the simple CLI while this is still a one-shot file.
    const int amgCoarsenType = memoirs_env_int("MEMOIRS_AMG_COARSEN", 8);
    const int amgInterpType = memoirs_env_int("MEMOIRS_AMG_INTERP", 6);
    const int amgRelaxType = memoirs_env_int("MEMOIRS_AMG_RELAX", 18);
    const int amgAggLevels = memoirs_env_int("MEMOIRS_AMG_AGG_LEVELS", 0);
    const int amgKeepTranspose = memoirs_env_int("MEMOIRS_AMG_KEEP_TRANSPOSE", 1);
    const int amgPmax = memoirs_env_int("MEMOIRS_AMG_PMAX", 4);
    const int amgNumSweeps = memoirs_env_int("MEMOIRS_AMG_SWEEPS", 1);
    const int amgRAP2 = memoirs_env_int("MEMOIRS_AMG_RAP2", 0);
    const double amgTruncFactor = memoirs_env_double("MEMOIRS_AMG_TRUNC", 0.0);
    const double amgStrongThreshold = memoirs_env_double("MEMOIRS_AMG_STRONG", -1.0);

    const bool ijUseRowSizes = memoirs_env_bool("MEMOIRS_IJ_ROW_SIZES", true);
    const bool ijBulkInsert = memoirs_env_bool("MEMOIRS_IJ_BULK_INSERT", true);

    rep.amgCoarsenType = amgCoarsenType;
    rep.amgInterpType = amgInterpType;
    rep.amgRelaxType = amgRelaxType;
    rep.amgAggLevels = amgAggLevels;
    rep.amgKeepTranspose = amgKeepTranspose;
    rep.amgPmax = amgPmax;
    rep.amgNumSweeps = amgNumSweeps;
    rep.amgRAP2 = amgRAP2;
    rep.amgTruncFactor = amgTruncFactor;
    rep.amgStrongThreshold = amgStrongThreshold;
    rep.ijUseRowSizes = ijUseRowSizes ? 1 : 0;
    rep.ijBulkInsert = ijBulkInsert ? 1 : 0;

    auto elapsed = [](const std::chrono::steady_clock::time_point& a,
                      const std::chrono::steady_clock::time_point& b) -> double {
        return std::chrono::duration<double>(b - a).count();
    };

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

    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (size != 1) {
        throw std::runtime_error("Patch 004 one-shot HYPRE path supports only one MPI rank.");
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

    const HYPRE_Int n = (HYPRE_Int)sparse_nrows(sys.A);
    const HYPRE_Int ilower = 0;
    const HYPRE_Int iupper = n - 1;

    HYPRE_IJMatrix ijA = nullptr;
    HYPRE_ParCSRMatrix parA = nullptr;

    // ----------------------------------------------------------------------
    // Matrix create/insert/assemble.
    // Later modular split:
    //   HypreIJSystem::create_matrix_from_host_rows(...)
    // ----------------------------------------------------------------------
    auto tMat0 = std::chrono::steady_clock::now();

    hypre_check(HYPRE_IJMatrixCreate(comm, ilower, iupper, ilower, iupper, &ijA),
                "HYPRE_IJMatrixCreate");
    hypre_check(HYPRE_IJMatrixSetObjectType(ijA, HYPRE_PARCSR),
                "HYPRE_IJMatrixSetObjectType");
    // Convert SparseRows map storage to flat CSR-like arrays for HYPRE IJ.
    //
    // This mirrors the mature DG2/DG1 HYPRE path:
    //   SetRowSizes -> Initialize_v2(HOST) -> SetValues(all rows) -> Assemble -> Migrate.
    //
    // It avoids per-row temporary vector allocation and gives HYPRE row-size
    // preallocation information before insertion.
    std::vector<HYPRE_Int> ijNcols(n);
    std::vector<HYPRE_Int> ijRows(n);
    std::vector<HYPRE_Int> ijCols;
    std::vector<HYPRE_Complex> ijVals;

    size_t flatNnz = 0;
    if (sys.A.fixedPattern) {
        for (HYPRE_Int i = 0; i < n; ++i) {
            flatNnz += sparse_cols_row(sys.A, i).size();
        }
    } else {
        for (HYPRE_Int i = 0; i < n; ++i) {
            flatNnz += sys.A.rows[i].size();
        }
    }

    ijCols.reserve(flatNnz);
    ijVals.reserve(flatNnz);

    for (HYPRE_Int i = 0; i < n; ++i) {
        ijRows[i] = i;

        if (sys.A.fixedPattern) {
            const auto& c = sparse_cols_row(sys.A, i);
            ijNcols[i] = (HYPRE_Int)c.size();
            for (int k = 0; k < (int)c.size(); ++k) {
                ijCols.push_back((HYPRE_Int)c[k]);
                ijVals.push_back((HYPRE_Complex)sparse_value_at(sys.A, i, k));
            }
        } else {
            const auto& row = sys.A.rows[i];
            ijNcols[i] = (HYPRE_Int)row.size();
            for (const auto& kv : row) {
                ijCols.push_back((HYPRE_Int)kv.first);
                ijVals.push_back((HYPRE_Complex)kv.second);
            }
        }
    }

    if (ijUseRowSizes) {
        hypre_check(HYPRE_IJMatrixSetRowSizes(ijA, ijNcols.data()),
                    "HYPRE_IJMatrixSetRowSizes");
    }

    hypre_initialize_ij_matrix_for_host_insertion(ijA);

    if (ijBulkInsert) {
        hypre_check(HYPRE_IJMatrixSetValues(
            ijA, n, ijNcols.data(), ijRows.data(), ijCols.data(), ijVals.data()),
            "HYPRE_IJMatrixSetValues bulk");
    } else {
        size_t off = 0;
        for (HYPRE_Int i = 0; i < n; ++i) {
            HYPRE_Int rowId = i;
            HYPRE_Int ncols = ijNcols[i];
            hypre_check(HYPRE_IJMatrixSetValues(
                ijA, 1, &ncols, &rowId, ijCols.data() + off, ijVals.data() + off),
                "HYPRE_IJMatrixSetValues row");
            off += (size_t)ncols;
        }
    }

    hypre_check(HYPRE_IJMatrixAssemble(ijA), "HYPRE_IJMatrixAssemble");

    auto tMat1 = std::chrono::steady_clock::now();
    rep.hypreMatrixInsertSeconds = elapsed(tMat0, tMat1);

    // ----------------------------------------------------------------------
    // Matrix migrate and object extraction.
    // Later reusable-device path should migrate once and keep this resident.
    // ----------------------------------------------------------------------
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
        bvals[i] = (HYPRE_Complex)sys.b[i];
    }

    // ----------------------------------------------------------------------
    // Vector create/insert/assemble.
    // Later loop-ready path should update RHS/vector values without rebuild.
    // ----------------------------------------------------------------------
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

    // ----------------------------------------------------------------------
    // Vector migration and object extraction.
    // ----------------------------------------------------------------------
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

    // ----------------------------------------------------------------------
    // PCG/BoomerAMG create/configure/setup.
    // Later loop-ready path should separate:
    //   setup once if A is constant
    //   setup every nonlinear iteration if matrix values change substantially
    // ----------------------------------------------------------------------
    auto tSetup0 = std::chrono::steady_clock::now();

    HYPRE_Solver krylov = nullptr;
    if (useGmres) {
        hypre_check(HYPRE_ParCSRGMRESCreate(comm, &krylov),
                    "HYPRE_ParCSRGMRESCreate");
        hypre_check(HYPRE_GMRESSetMaxIter(krylov, opt.maxit),
                    "HYPRE_GMRESSetMaxIter");
        hypre_check(HYPRE_GMRESSetTol(krylov, opt.tol),
                    "HYPRE_GMRESSetTol");
        hypre_check(HYPRE_GMRESSetKDim(krylov, gmresRestart),
                    "HYPRE_GMRESSetKDim");
        hypre_check(HYPRE_GMRESSetPrintLevel(krylov, opt.hyprePrint),
                    "HYPRE_GMRESSetPrintLevel");
        hypre_check(HYPRE_GMRESSetLogging(krylov, 1),
                    "HYPRE_GMRESSetLogging");
    } else {
        hypre_check(HYPRE_ParCSRPCGCreate(comm, &krylov),
                    "HYPRE_ParCSRPCGCreate");
        hypre_check(HYPRE_PCGSetMaxIter(krylov, opt.maxit),
                    "HYPRE_PCGSetMaxIter");
        hypre_check(HYPRE_PCGSetTol(krylov, opt.tol),
                    "HYPRE_PCGSetTol");
        hypre_check(HYPRE_PCGSetTwoNorm(krylov, 1),
                    "HYPRE_PCGSetTwoNorm");
        hypre_check(HYPRE_PCGSetPrintLevel(krylov, opt.hyprePrint),
                    "HYPRE_PCGSetPrintLevel");
        hypre_check(HYPRE_PCGSetLogging(krylov, 1),
                    "HYPRE_PCGSetLogging");
    }

    HYPRE_Solver precond = nullptr;
    const std::string pre = lower_copy(opt.precond);

    if (pre == "diagscale") {
        if (useGmres) {
            hypre_check(HYPRE_GMRESSetPrecond(
                krylov,
                (HYPRE_PtrToSolverFcn)HYPRE_ParCSRDiagScale,
                (HYPRE_PtrToSolverFcn)HYPRE_ParCSRDiagScaleSetup,
                nullptr),
                "HYPRE_GMRESSetPrecond DiagScale");
        } else {
            hypre_check(HYPRE_PCGSetPrecond(
                krylov,
                (HYPRE_PtrToSolverFcn)HYPRE_ParCSRDiagScale,
                (HYPRE_PtrToSolverFcn)HYPRE_ParCSRDiagScaleSetup,
                nullptr),
                "HYPRE_PCGSetPrecond DiagScale");
        }
    } else if (pre == "amg" || pre == "boomeramg") {
        hypre_check(HYPRE_BoomerAMGCreate(&precond), "HYPRE_BoomerAMGCreate");
        hypre_check(HYPRE_BoomerAMGSetPrintLevel(precond, opt.hyprePrint),
                    "HYPRE_BoomerAMGSetPrintLevel");
        hypre_check(HYPRE_BoomerAMGSetCoarsenType(precond, amgCoarsenType),
                    "HYPRE_BoomerAMGSetCoarsenType");
        hypre_check(HYPRE_BoomerAMGSetInterpType(precond, amgInterpType),
                    "HYPRE_BoomerAMGSetInterpType");
        hypre_check(HYPRE_BoomerAMGSetRelaxType(precond, amgRelaxType),
                    "HYPRE_BoomerAMGSetRelaxType");
        hypre_check(HYPRE_BoomerAMGSetNumSweeps(precond, amgNumSweeps),
                    "HYPRE_BoomerAMGSetNumSweeps");
        hypre_check(HYPRE_BoomerAMGSetTol(precond, 0.0),
                    "HYPRE_BoomerAMGSetTol");
        hypre_check(HYPRE_BoomerAMGSetMaxIter(precond, 1),
                    "HYPRE_BoomerAMGSetMaxIter");

        // DG2/DG1 CUDA BoomerAMG robust settings.
        hypre_check(HYPRE_BoomerAMGSetRelaxOrder(precond, 0),
                    "HYPRE_BoomerAMGSetRelaxOrder");
        hypre_check(HYPRE_BoomerAMGSetPMaxElmts(precond, amgPmax),
                    "HYPRE_BoomerAMGSetPMaxElmts");
        hypre_check(HYPRE_BoomerAMGSetKeepTranspose(precond, amgKeepTranspose),
                    "HYPRE_BoomerAMGSetKeepTranspose");
        hypre_check(HYPRE_BoomerAMGSetTruncFactor(precond, amgTruncFactor),
                    "HYPRE_BoomerAMGSetTruncFactor");
        hypre_check(HYPRE_BoomerAMGSetRAP2(precond, amgRAP2),
                    "HYPRE_BoomerAMGSetRAP2");

        if (amgAggLevels > 0) {
            hypre_check(HYPRE_BoomerAMGSetAggNumLevels(precond, amgAggLevels),
                        "HYPRE_BoomerAMGSetAggNumLevels");
            hypre_check(HYPRE_BoomerAMGSetAggInterpType(precond, 4),
                        "HYPRE_BoomerAMGSetAggInterpType");
        }

        if (amgStrongThreshold >= 0.0) {
            hypre_check(HYPRE_BoomerAMGSetStrongThreshold(precond, amgStrongThreshold),
                        "HYPRE_BoomerAMGSetStrongThreshold");
        }

        if (useGmres) {
            hypre_check(HYPRE_GMRESSetPrecond(
                krylov,
                (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSolve,
                (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSetup,
                precond),
                "HYPRE_GMRESSetPrecond BoomerAMG");
        } else {
            hypre_check(HYPRE_PCGSetPrecond(
                krylov,
                (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSolve,
                (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSetup,
                precond),
                "HYPRE_PCGSetPrecond BoomerAMG");
        }
    } else if (pre == "none") {
        // no preconditioner
    } else {
        throw std::runtime_error("Unsupported -precond: " + opt.precond);
    }

    if (useGmres) {
        hypre_check(HYPRE_ParCSRGMRESSetup(krylov, parA, parb, parx),
                    "HYPRE_ParCSRGMRESSetup");
    } else {
        hypre_check(HYPRE_ParCSRPCGSetup(krylov, parA, parb, parx),
                    "HYPRE_ParCSRPCGSetup");
    }

    auto tSetup1 = std::chrono::steady_clock::now();
    rep.hypreSetupSeconds = elapsed(tSetup0, tSetup1);

    // ----------------------------------------------------------------------
    // Solve only, optionally updating RHS between solves.
    //
    // Repeated-solve mode:
    //   Setup is performed once above.
    //   The same resident matrix, solution vector, PCG object and AMG
    //   preconditioner are reused for solveRepeats solves.
    //
    // RHS update mode:
    //   none:
    //     reuse same resident b.
    //
    //   scale_host:
    //     update b through the existing IJ vector from host values, migrate b
    //     back to device, then solve. This tests the first Picard-like loop
    //     shape without rebuilding A or AMG.
    // ----------------------------------------------------------------------
    auto tSolve0 = std::chrono::steady_clock::now();
    double rhsUpdateAccum = 0.0;

    for (int r = 0; r < solveRepeats; ++r) {
        if (rhsUpdateMode == "scale_host") {
            auto tRhs0 = std::chrono::steady_clock::now();

            const double factor = 1.0 + 0.01 * double(r);
            for (HYPRE_Int i = 0; i < n; ++i) {
                bvals[i] = HYPRE_Complex(factor * double(sys.b[i]));
            }

            if (useDevice) {
                hypre_check(HYPRE_IJVectorMigrate(ijb, HYPRE_MEMORY_HOST),
                            "HYPRE_IJVectorMigrate b HOST for RHS update");
            }

            hypre_check(HYPRE_IJVectorSetValues(ijb, n, rows.data(), bvals.data()),
                        "HYPRE_IJVectorSetValues b update");
            hypre_check(HYPRE_IJVectorAssemble(ijb),
                        "HYPRE_IJVectorAssemble b update");

            if (useDevice) {
                hypre_check(HYPRE_IJVectorMigrate(ijb, HYPRE_MEMORY_DEVICE),
                            "HYPRE_IJVectorMigrate b DEVICE after RHS update");
            }

            hypre_check(HYPRE_IJVectorGetObject(ijb, (void**)&parb),
                        "HYPRE_IJVectorGetObject b after RHS update");

            auto tRhs1 = std::chrono::steady_clock::now();
            rhsUpdateAccum += elapsed(tRhs0, tRhs1);
        } else if (rhsUpdateMode == "scale_device_inplace") {
            auto tRhs0 = std::chrono::steady_clock::now();

            // Synthetic device-side RHS update.
            //
            // This scales the currently resident RHS vector in-place. It is not
            // the final physical source evaluation, but it verifies the loop
            // architecture:
            //
            //   b remains resident
            //   no IJ host insertion
            //   no b migration
            //   same A and AMG setup are reused
            //
            // Use a very small factor to avoid large cumulative RHS changes over
            // many repeats.
            const double factor = 1.0 + 1.0e-4 * double(r + 1);
            hypre_check(HYPRE_ParVectorScale(HYPRE_Complex(factor), parb),
                        "HYPRE_ParVectorScale b device inplace");

            auto tRhs1 = std::chrono::steady_clock::now();
            rhsUpdateAccum += elapsed(tRhs0, tRhs1);
        }

        hypre_check(HYPRE_ParVectorSetConstantValues(parx, HYPRE_Complex(0)),
                    "HYPRE_ParVectorSetConstantValues x=0");

        if (useGmres) {
            hypre_check_pcg_solve(HYPRE_ParCSRGMRESSolve(krylov, parA, parb, parx),
                                  "HYPRE_ParCSRGMRESSolve");
        } else {
            hypre_check_pcg_solve(HYPRE_ParCSRPCGSolve(krylov, parA, parb, parx),
                                  "HYPRE_ParCSRPCGSolve");
        }
    }

    auto tSolve1 = std::chrono::steady_clock::now();
    rep.hypreSolveOnlySeconds = elapsed(tSolve0, tSolve1);
    rep.hypreSolveOnlyAvgSeconds = rep.hypreSolveOnlySeconds / double(solveRepeats);
    rep.rhsUpdateSeconds = rhsUpdateAccum;
    rep.rhsUpdateAvgSeconds = rhsUpdateAccum / double(solveRepeats);

    hypre_check_pcg_getter(HYPRE_PCGGetNumIterations(krylov, &rep.iterations),
                           "HYPRE_PCGGetNumIterations");
    HYPRE_Real finalRel = 0;
    hypre_check_pcg_getter(HYPRE_PCGGetFinalRelativeResidualNorm(krylov, &finalRel),
                           "HYPRE_PCGGetFinalRelativeResidualNorm");
    rep.finalRelRes = double(finalRel);

    // ----------------------------------------------------------------------
    // Optional solution copyback and MMS diagnostics.
    //
    // Verification mode, default:
    //   MEMOIRS_COMPUTE_ERROR=1
    //   migrate x back to host, extract values, compute nodal/L2/H1 errors.
    //
    // Loop/device-resident mode:
    //   MEMOIRS_COMPUTE_ERROR=0
    //   do NOT migrate x back. The solution remains in HYPRE's resident vector.
    //   This is the first stepping stone toward scalar/NSE Picard loops where
    //   diagnostics/output are occasional, not per iteration.
    // ----------------------------------------------------------------------
    if (computeError) {
        auto tGet0 = std::chrono::steady_clock::now();

        if (useDevice) {
            hypre_check(HYPRE_IJVectorMigrate(ijx, HYPRE_MEMORY_HOST),
                        "HYPRE_IJVectorMigrate x HOST for GetValues");
        }

        std::fill(xvals.begin(), xvals.end(), HYPRE_Complex(0));
        hypre_check(HYPRE_IJVectorGetValues(ijx, n, rows.data(), xvals.data()),
                    "HYPRE_IJVectorGetValues x");

        xOut.assign(n, Real(0));
        for (HYPRE_Int i = 0; i < n; ++i) {
            xOut[i] = Real(xvals[i]);
        }

        auto tGet1 = std::chrono::steady_clock::now();
        rep.hypreGetSolutionSeconds = elapsed(tGet0, tGet1);

        auto tErr0 = std::chrono::steady_clock::now();

        compute_nodal_error(m, dm, mms, xOut, rep);
        ErrorNorms en = compute_error_norms(m, dm, mms, xOut);
        rep.L2Error = en.L2;
        rep.H1SemiError = en.H1Semi;

        auto tErr1 = std::chrono::steady_clock::now();
        rep.errorNormSeconds = elapsed(tErr0, tErr1);
    } else {
        xOut.clear();
        rep.hypreGetSolutionSeconds = 0.0;
        rep.errorNormSeconds = 0.0;
        rep.nodalErrorL2 = -1.0;
        rep.nodalErrorMax = -1.0;
        rep.L2Error = -1.0;
        rep.H1SemiError = -1.0;
    }

    // ----------------------------------------------------------------------
    // Destroy/finalize.
    // Later reusable object mode should not do this per iteration.
    // ----------------------------------------------------------------------
    auto tDestroy0 = std::chrono::steady_clock::now();

    if (precond) {
        hypre_check(HYPRE_BoomerAMGDestroy(precond), "HYPRE_BoomerAMGDestroy");
    }
    if (useGmres) {
        hypre_check(HYPRE_ParCSRGMRESDestroy(krylov),
                    "HYPRE_ParCSRGMRESDestroy");
    } else {
        hypre_check(HYPRE_ParCSRPCGDestroy(krylov),
                    "HYPRE_ParCSRPCGDestroy");
    }
    hypre_check(HYPRE_IJVectorDestroy(ijb), "HYPRE_IJVectorDestroy b");
    hypre_check(HYPRE_IJVectorDestroy(ijx), "HYPRE_IJVectorDestroy x");
    hypre_check(HYPRE_IJMatrixDestroy(ijA), "HYPRE_IJMatrixDestroy A");

    hypre_check(HYPRE_Finalize(), "HYPRE_Finalize");

    if (startedMPI) {
        MPI_Finalize();
    }

    auto tDestroy1 = std::chrono::steady_clock::now();
    rep.hypreDestroyFinalizeSeconds = elapsed(tDestroy0, tDestroy1);

    return rep;
}

#else

static SolveReport solve_hypre_ij_pcg(
    const PolyMesh&,
    const LinearCgDofMap&,
    const AssembledSystem&,
    const std::string&,
    const Options&,
    std::vector<Real>&
) {
    throw std::runtime_error(
        "This binary was built without HYPRE. Reconfigure with "
        "-DMEMOIRS_USE_HYPRE=ON -DHYPRE_ROOT=/path/to/hypre");
}

#endif

static void print_solve_report(
    const Options& opt,
    const LinearCgDofMap& dm,
    const SolveReport& rep
) {
    std::cout << "--------------- solve report ---------------\n";
    std::cout << "backend                   = HYPRE IJ/ParCSR\n";
    std::cout << "space                     = " << dm.resolvedSpace << "\n";
    std::cout << "solver                    = " << opt.solver << "\n";
    std::cout << "precond                   = " << opt.precond << "\n";
    std::cout << "hypreMemory               = " << opt.hypreMemory << "\n";
    std::cout << "computeError              = " << (memoirs_compute_error_enabled() ? 1 : 0) << "\n";
    std::cout << "tol                       = " << std::setprecision(16) << opt.tol << "\n";
    std::cout << "maxit                     = " << opt.maxit << "\n";
    std::cout << "solveRepeats              = " << rep.solveRepeats << "\n";
    std::cout << "rhsUpdateMode             = " << rep.rhsUpdateMode << "\n";
    std::cout << "assemblyMode              = " << memoirs_assembly_mode() << "\n";
    std::cout << "amgSettings               = coarsen " << rep.amgCoarsenType
              << " interp " << rep.amgInterpType
              << " relax " << rep.amgRelaxType
              << " aggLevels " << rep.amgAggLevels
              << " keepTranspose " << rep.amgKeepTranspose
              << " pmax " << rep.amgPmax
              << " sweeps " << rep.amgNumSweeps
              << " trunc " << std::setprecision(6) << rep.amgTruncFactor
              << " strong " << std::setprecision(6) << rep.amgStrongThreshold
              << "\n";
    std::cout << "ijSettings                = rowSizes " << rep.ijUseRowSizes
              << " bulkInsert " << rep.ijBulkInsert << "\n";
    std::cout << "iterations                = " << rep.iterations << "\n";
    std::cout << "finalRelativeResidual     = " << std::setprecision(16) << rep.finalRelRes << "\n";
    std::cout << "L2Error                   = " << std::setprecision(16) << rep.L2Error << "\n";
    std::cout << "H1SemiError               = " << std::setprecision(16) << rep.H1SemiError << "\n";
    std::cout << "nodalErrorL2              = " << std::setprecision(16) << rep.nodalErrorL2 << "\n";
    std::cout << "nodalErrorMax             = " << std::setprecision(16) << rep.nodalErrorMax << "\n";
    std::cout << "assemblySeconds           = " << std::setprecision(8) << rep.assemblySeconds << "\n";
    std::cout << "solveSeconds              = " << std::setprecision(8) << rep.solveSeconds << "\n";
    std::cout << "hypreMatrixInsertSeconds  = " << std::setprecision(8) << rep.hypreMatrixInsertSeconds << "\n";
    std::cout << "hypreMatrixMigrateSeconds = " << std::setprecision(8) << rep.hypreMatrixMigrateSeconds << "\n";
    std::cout << "hypreVectorInsertSeconds  = " << std::setprecision(8) << rep.hypreVectorInsertSeconds << "\n";
    std::cout << "hypreVectorMigrateSeconds = " << std::setprecision(8) << rep.hypreVectorMigrateSeconds << "\n";
    std::cout << "hypreSetupSeconds         = " << std::setprecision(8) << rep.hypreSetupSeconds << "\n";
    std::cout << "rhsUpdateSeconds          = " << std::setprecision(8) << rep.rhsUpdateSeconds << "\n";
    std::cout << "rhsUpdateAvgSeconds       = " << std::setprecision(8) << rep.rhsUpdateAvgSeconds << "\n";
    std::cout << "hypreSolveOnlySeconds     = " << std::setprecision(8) << rep.hypreSolveOnlySeconds << "\n";
    std::cout << "hypreSolveOnlyAvgSeconds  = " << std::setprecision(8) << rep.hypreSolveOnlyAvgSeconds << "\n";
    std::cout << "hypreGetSolutionSeconds   = " << std::setprecision(8) << rep.hypreGetSolutionSeconds << "\n";
    std::cout << "errorNormSeconds          = " << std::setprecision(8) << rep.errorNormSeconds << "\n";
    std::cout << "hypreDestroyFinalizeSec   = " << std::setprecision(8) << rep.hypreDestroyFinalizeSeconds << "\n";
    std::cout << "--------------------------------------------\n";
}
