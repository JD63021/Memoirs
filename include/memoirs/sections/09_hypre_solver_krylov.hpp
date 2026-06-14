#pragma once

#include <cstring>
#include <cuda_runtime_api.h>

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

    // HYPRE solve return code. 0 means converged/OK.
    // Nonzero is kept nonfatal so Krylov iteration counts can still be reported.
    int hypreSolveStatus = 0;
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
    int amgNumFunctions = 1;
    int amgNodal = 0;
    int amgNodalLevels = 1;
    int amgKeepSameSign = 0;
    int amgNodalDiag = 0;

    int ijUseRowSizes = 1;
    int ijBulkInsert = 1;

    int hypreFvReuseEnabled = 0;
    int hypreFvReuseBuild = 0;
    int hypreFvReuseHit = 0;
    int hypreFvReuseSetup = 0;


    int amgReuseEnabled = 0;
    int amgReuseHit = 0;
    int amgReuseBuild = 0;


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
    const std::string& mms,
    const std::vector<Real>& x,
    SolveReport& rep
) {
    double e2 = 0.0;
    double emax = 0.0;
    for (int i = 0; i < (int)x.size(); ++i) {
        const double ex = mms_exact_value(m.points[i], mms);
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
        const double ex = mms_exact_value(m.points[i], mms);
        const double e = double(x[i]) - ex;
        nodal2 += e*e;
        nodalMax = std::max(nodalMax, std::abs(e));
    }
    en.nodalL2 = std::sqrt(nodal2);
    en.nodalMax = nodalMax;

    double l2 = 0.0;
    double h1 = 0.0;

    if (dm.resolvedSpace == "cg_hex_q1") {
        const double a = std::sqrt(3.0/5.0);
        const double qp[3] = {-a, 0.0, a};
        const double qw[3] = {5.0/9.0, 8.0/9.0, 5.0/9.0};

        for (const auto& c : m.cells) {
            const auto hv = ordered_hex_vertices_axis_aligned(m, c);

            for (int ix = 0; ix < 3; ++ix) {
                for (int iy = 0; iy < 3; ++iy) {
                    for (int iz = 0; iz < 3; ++iz) {
                        const double xi = qp[ix];
                        const double eta = qp[iy];
                        const double zeta = qp[iz];
                        const double w = qw[ix] * qw[iy] * qw[iz];

                        std::array<double,8> N;
                        std::array<Vec3,8> dNref;
                        hex_q1_basis(xi, eta, zeta, N, dNref);

                        Vec3 xq;
                        Mat3 J;
                        double uh = 0.0;

                        for (int aLoc = 0; aLoc < 8; ++aLoc) {
                            const Vec3& X = m.points[hv[aLoc]];
                            const double xa = double(x[hv[aLoc]]);

                            uh += N[aLoc] * xa;

                            xq.x += N[aLoc] * X.x;
                            xq.y += N[aLoc] * X.y;
                            xq.z += N[aLoc] * X.z;

                            J.a[0][0] += dNref[aLoc].x * X.x;
                            J.a[1][0] += dNref[aLoc].x * X.y;
                            J.a[2][0] += dNref[aLoc].x * X.z;

                            J.a[0][1] += dNref[aLoc].y * X.x;
                            J.a[1][1] += dNref[aLoc].y * X.y;
                            J.a[2][1] += dNref[aLoc].y * X.z;

                            J.a[0][2] += dNref[aLoc].z * X.x;
                            J.a[1][2] += dNref[aLoc].z * X.y;
                            J.a[2][2] += dNref[aLoc].z * X.z;
                        }

                        const double detJ = std::abs(det3(J));
                        const Mat3 invJ = inv3(J);

                        Vec3 guh;
                        for (int aLoc = 0; aLoc < 8; ++aLoc) {
                            const Vec3 g = invJT_mul(invJ, dNref[aLoc]);
                            const double xa = double(x[hv[aLoc]]);
                            guh.x += xa * g.x;
                            guh.y += xa * g.y;
                            guh.z += xa * g.z;
                        }

                        const double ue = mms_exact_value(xq, mms);
                        const Vec3 ge = mms_exact_grad(xq, mms);
                        const double eu = uh - ue;
                        const Vec3 eg = {guh.x - ge.x, guh.y - ge.y, guh.z - ge.z};

                        l2 += w * detJ * eu * eu;
                        h1 += w * detJ * dot3(eg, eg);
                    }
                }
            }
        }
    } else if (dm.resolvedSpace == "cg_tet_p1") {
        // Symmetric 4-point tetra quadrature, degree 2.
        const double aa = 0.5854101966249685;
        const double bb = 0.1381966011250105;
        const double lam[4][4] = {
            {aa, bb, bb, bb},
            {bb, aa, bb, bb},
            {bb, bb, aa, bb},
            {bb, bb, bb, aa}
        };

        for (const auto& c : m.cells) {
            if (c.verts.size() != 4) throw std::runtime_error("Tet error norm cell does not have 4 vertices.");

            std::array<int,4> tv = {c.verts[0], c.verts[1], c.verts[2], c.verts[3]};

            const Vec3& X0 = m.points[tv[0]];
            const Vec3& X1 = m.points[tv[1]];
            const Vec3& X2 = m.points[tv[2]];
            const Vec3& X3 = m.points[tv[3]];

            Mat3 J;
            J.a[0][0] = X1.x - X0.x; J.a[1][0] = X1.y - X0.y; J.a[2][0] = X1.z - X0.z;
            J.a[0][1] = X2.x - X0.x; J.a[1][1] = X2.y - X0.y; J.a[2][1] = X2.z - X0.z;
            J.a[0][2] = X3.x - X0.x; J.a[1][2] = X3.y - X0.y; J.a[2][2] = X3.z - X0.z;

            const double volume = std::abs(det3(J)) / 6.0;
            const Mat3 invJ = inv3(J);
            const auto& gref = tet_p1_grad_ref();

            std::array<Vec3,4> grad;
            Vec3 guh;
            for (int aLoc = 0; aLoc < 4; ++aLoc) {
                grad[aLoc] = invJT_mul(invJ, gref[aLoc]);
                const double xa = double(x[tv[aLoc]]);
                guh.x += xa * grad[aLoc].x;
                guh.y += xa * grad[aLoc].y;
                guh.z += xa * grad[aLoc].z;
            }

            for (int q = 0; q < 4; ++q) {
                Vec3 xq;
                double uh = 0.0;
                for (int aLoc = 0; aLoc < 4; ++aLoc) {
                    const Vec3& X = m.points[tv[aLoc]];
                    const double L = lam[q][aLoc];
                    const double xa = double(x[tv[aLoc]]);
                    xq.x += L * X.x;
                    xq.y += L * X.y;
                    xq.z += L * X.z;
                    uh += L * xa;
                }

                const double ue = mms_exact_value(xq, mms);
                const Vec3 ge = mms_exact_grad(xq, mms);
                const double eu = uh - ue;
                const Vec3 eg = {guh.x - ge.x, guh.y - ge.y, guh.z - ge.z};

                const double w = volume / 4.0;
                l2 += w * eu * eu;
                h1 += w * dot3(eg, eg);
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


// -----------------------------------------------------------------------------
// HYPRE/MPI lifecycle for the copied Krylov helper.
//
// FV reference pattern:
//   - MPI_Init once
//   - HYPRE_Init / HYPRE_DeviceInitialize once
//   - repeated nonlinear/Picard solves
//   - each solve destroys only its own IJMatrix/IJVector/Krylov/AMG objects
//   - finalize only once at process exit
//
// This experimental header approximates that pattern with header-local one-time
// HYPRE initialization. It intentionally does not finalize MPI/HYPRE inside
// solve_hypre_ij_krylov(), because Picard calls the solve helper repeatedly.
// -----------------------------------------------------------------------------
static bool memoirs_hypre_krylov_runtime_hypre_initialized = false;
static bool memoirs_hypre_krylov_runtime_device_initialized = false;

static inline void memoirs_hypre_krylov_runtime_init_once(bool useDevice, int rank) {
    if (!memoirs_hypre_krylov_runtime_hypre_initialized) {
        hypre_check(HYPRE_Init(), "HYPRE_Init");
        memoirs_hypre_krylov_runtime_hypre_initialized = true;
    }

    if (useDevice) {
        if (!memoirs_hypre_krylov_runtime_device_initialized) {
            hypre_check(HYPRE_DeviceInitialize(), "HYPRE_DeviceInitialize");

            HYPRE_Int spgemm_status = HYPRE_SetSpGemmUseVendor(0);
            if (spgemm_status && rank == 0) {
                std::cerr << "WARNING: HYPRE_SetSpGemmUseVendor(0) returned "
                          << int(spgemm_status)
                          << "; continuing with default SpGEMM backend.\n";
            }

            memoirs_hypre_krylov_runtime_device_initialized = true;
        }

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
}


struct MemoirsHypreAmgReuseCache {
    bool valid = false;
    HYPRE_Solver precond = nullptr;
    HYPRE_IJMatrix ijAOwner = nullptr;
    HYPRE_Int n = 0;
    std::string key;
    long long builds = 0;
    long long hits = 0;
};

static inline MemoirsHypreAmgReuseCache& memoirs_hypre_amg_reuse_cache_singleton() {
    static MemoirsHypreAmgReuseCache cache;
    return cache;
}

static inline HYPRE_Int memoirs_hypre_amg_reuse_noop_setup(
    HYPRE_Solver,
    HYPRE_ParCSRMatrix,
    HYPRE_ParVector,
    HYPRE_ParVector
) {
    return 0;
}

static inline void memoirs_hypre_amg_reuse_reset() {
    MemoirsHypreAmgReuseCache& c = memoirs_hypre_amg_reuse_cache_singleton();

    if (c.precond) {
        hypre_check(HYPRE_BoomerAMGDestroy(c.precond),
                    "HYPRE_BoomerAMGDestroy cached reuse precond");
    }
    if (c.ijAOwner) {
        hypre_check(HYPRE_IJMatrixDestroy(c.ijAOwner),
                    "HYPRE_IJMatrixDestroy cached reuse A");
    }

    c.valid = false;
    c.precond = nullptr;
    c.ijAOwner = nullptr;
    c.n = 0;
    c.key.clear();
}

static inline std::string memoirs_hypre_amg_reuse_make_key(
    HYPRE_Int n,
    const Options& opt,
    int coarsen,
    int interp,
    int relax,
    int aggLevels,
    int keepTranspose,
    int pmax,
    int sweeps,
    int rap2,
    double trunc,
    double strong,
    int numFunctions,
    int nodal,
    int nodalLevels,
    int keepSameSign,
    int nodalDiag
) {
    std::ostringstream os;
    os << n
       << "|" << lower_copy(opt.solver)
       << "|" << lower_copy(opt.precond)
       << "|" << lower_copy(opt.hypreMemory)
       << "|" << coarsen
       << "|" << interp
       << "|" << relax
       << "|" << aggLevels
       << "|" << keepTranspose
       << "|" << pmax
       << "|" << sweeps
       << "|" << rap2
       << "|" << std::setprecision(17) << trunc
       << "|" << std::setprecision(17) << strong
       << "|" << numFunctions
       << "|" << nodal
       << "|" << nodalLevels
       << "|" << keepSameSign
       << "|" << nodalDiag;
    return os.str();
}



// ============================================================================
// FV-like persistent HYPRE IJ/ParCSR/Krylov/BoomerAMG reuse experiment
// ============================================================================
//
// This follows the working FV-code pattern more closely than the earlier
// detached-AMG/no-op setup experiment:
//
//   - one persistent IJMatrix / ParCSR object
//   - one persistent IJVector pair
//   - one persistent Krylov object
//   - one persistent BoomerAMG object
//   - matrix values updated directly in the same ParCSR diag data buffer
//   - RHS/x updated directly in the same ParVector data buffers
//   - Krylov/AMG setup only on cache build or explicit reset
//
// Limitations for this experiment:
//   - single MPI rank, diagonal ParCSR block only
//   - requires fixed SparseRows pattern
//   - intended first for Q1/Q1 cavity HYPRE_MEMORY=device
// ============================================================================

struct MemoirsHypreFvlikeReuseCache {
    bool valid = false;
    bool setupValid = false;
    bool useDevice = false;

    HYPRE_Int n = 0;
    std::size_t nnz = 0;
    const void* patternPtr = nullptr;
    std::string key;

    std::string solverKind;
    bool usePCG = false;
    bool useBiCGSTAB = false;
    bool useGMRES = false;
    bool useFGMRES = false;

    HYPRE_IJMatrix ijA = nullptr;
    HYPRE_ParCSRMatrix parA = nullptr;
    HYPRE_IJVector ijb = nullptr;
    HYPRE_IJVector ijx = nullptr;
    HYPRE_ParVector parb = nullptr;
    HYPRE_ParVector parx = nullptr;

    HYPRE_Solver krylov = nullptr;
    HYPRE_Solver precond = nullptr;

    HYPRE_Complex* Adata = nullptr;
    HYPRE_Complex* bdata = nullptr;
    HYPRE_Complex* xdata = nullptr;

    std::vector<HYPRE_Int> rows;
    std::vector<HYPRE_Int> ncols;
    std::vector<HYPRE_Int> cols;
    std::vector<HYPRE_Int> permPatternToHypre;
    std::vector<HYPRE_Complex> vals;
    std::vector<HYPRE_Complex> valsPermuted;
    std::vector<HYPRE_Complex> bvals;
    std::vector<HYPRE_Complex> xvals;
    bool permIdentity = true;

    long long builds = 0;
    long long hits = 0;
};

static inline MemoirsHypreFvlikeReuseCache&
memoirs_hypre_fvlike_reuse_cache_singleton() {
    static MemoirsHypreFvlikeReuseCache c;
    return c;
}

static inline void memoirs_cuda_check_fvlike(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::ostringstream os;
        os << what << " failed: " << cudaGetErrorString(e);
        throw std::runtime_error(os.str());
    }
}

static inline void memoirs_hypre_fvlike_destroy_solver(MemoirsHypreFvlikeReuseCache& c) {
    if (c.precond) {
        hypre_check(HYPRE_BoomerAMGDestroy(c.precond),
                    "HYPRE_BoomerAMGDestroy fvlike");
    }

    if (c.krylov) {
        if (c.usePCG) {
            hypre_check(HYPRE_ParCSRPCGDestroy(c.krylov),
                        "HYPRE_ParCSRPCGDestroy fvlike");
        } else if (c.useBiCGSTAB) {
            hypre_check(HYPRE_ParCSRBiCGSTABDestroy(c.krylov),
                        "HYPRE_ParCSRBiCGSTABDestroy fvlike");
        } else if (c.useGMRES) {
            hypre_check(HYPRE_ParCSRGMRESDestroy(c.krylov),
                        "HYPRE_ParCSRGMRESDestroy fvlike");
        } else if (c.useFGMRES) {
            hypre_check(HYPRE_ParCSRFlexGMRESDestroy(c.krylov),
                        "HYPRE_ParCSRFlexGMRESDestroy fvlike");
        }
    }

    if (c.ijb) hypre_check(HYPRE_IJVectorDestroy(c.ijb), "HYPRE_IJVectorDestroy b fvlike");
    if (c.ijx) hypre_check(HYPRE_IJVectorDestroy(c.ijx), "HYPRE_IJVectorDestroy x fvlike");
    if (c.ijA) hypre_check(HYPRE_IJMatrixDestroy(c.ijA), "HYPRE_IJMatrixDestroy A fvlike");

    c = MemoirsHypreFvlikeReuseCache{};
}

static inline void memoirs_hypre_fvlike_reuse_reset() {
    auto& c = memoirs_hypre_fvlike_reuse_cache_singleton();
    if (c.valid || c.krylov || c.precond || c.ijA || c.ijb || c.ijx) {
        memoirs_hypre_fvlike_destroy_solver(c);
    }
}

static inline std::string memoirs_hypre_fvlike_make_key(
    HYPRE_Int n,
    std::size_t nnz,
    const void* patternPtr,
    const Options& opt,
    int coarsen,
    int interp,
    int relax,
    int aggLevels,
    int keepTranspose,
    int pmax,
    int sweeps,
    int rap2,
    double trunc,
    double strong,
    int numFunctions,
    int nodal,
    int nodalLevels,
    int keepSameSign,
    int nodalDiag,
    bool useDevice
) {
    std::ostringstream os;
    os << n
       << "|" << nnz
       << "|" << patternPtr
       << "|" << lower_copy(opt.solver)
       << "|" << lower_copy(opt.precond)
       << "|" << lower_copy(opt.hypreMemory)
       << "|" << (useDevice ? 1 : 0)
       << "|" << coarsen
       << "|" << interp
       << "|" << relax
       << "|" << aggLevels
       << "|" << keepTranspose
       << "|" << pmax
       << "|" << sweeps
       << "|" << rap2
       << "|" << std::setprecision(17) << trunc
       << "|" << std::setprecision(17) << strong
       << "|" << numFunctions
       << "|" << nodal
       << "|" << nodalLevels
       << "|" << keepSameSign
       << "|" << nodalDiag;
    return os.str();
}

static inline void memoirs_hypre_fvlike_fill_pattern(
    const SparseRows& A,
    MemoirsHypreFvlikeReuseCache& c
) {
    c.rows.resize(c.n);
    c.ncols.resize(c.n);
    c.cols.clear();
    c.cols.reserve(c.nnz);

    for (HYPRE_Int i = 0; i < c.n; ++i) {
        const auto& rowCols = sparse_cols_row(A, i);
        c.rows[i] = i;
        c.ncols[i] = (HYPRE_Int)rowCols.size();

        for (int k = 0; k < (int)rowCols.size(); ++k) {
            c.cols.push_back((HYPRE_Int)rowCols[k]);
        }
    }
}

static inline void memoirs_hypre_fvlike_fill_values(
    const SparseRows& A,
    MemoirsHypreFvlikeReuseCache& c
) {
    c.vals.resize(c.nnz);

    if (A.fixedPattern &&
        sizeof(Real) == sizeof(HYPRE_Complex) &&
        !A.flatVals.empty()) {
        const HYPRE_Complex* p =
            reinterpret_cast<const HYPRE_Complex*>(A.flatVals.data());
        std::memcpy(c.vals.data(), p, c.nnz * sizeof(HYPRE_Complex));
        return;
    }

    std::size_t pos = 0;
    for (HYPRE_Int i = 0; i < c.n; ++i) {
        const auto& rowCols = sparse_cols_row(A, i);
        for (int k = 0; k < (int)rowCols.size(); ++k) {
            c.vals[pos++] = (HYPRE_Complex)sparse_value_at(A, i, k);
        }
    }
}

static inline void memoirs_hypre_fvlike_build_diag_permutation(
    MemoirsHypreFvlikeReuseCache& c
) {
    hypre_CSRMatrix* diag = hypre_ParCSRMatrixDiag(c.parA);
    hypre_CSRMatrix* offd = hypre_ParCSRMatrixOffd(c.parA);

    const HYPRE_Int diagNnz = hypre_CSRMatrixNumNonzeros(diag);
    const HYPRE_Int offdNnz = offd ? hypre_CSRMatrixNumNonzeros(offd) : 0;

    if ((std::size_t)diagNnz != c.nnz || offdNnz != 0) {
        std::ostringstream os;
        os << "FV-like reuse expected one-rank diag-only ParCSR layout. "
           << "diag nnz=" << diagNnz
           << " expected=" << c.nnz
           << " offd nnz=" << offdNnz;
        throw std::runtime_error(os.str());
    }

    HYPRE_Int* I = hypre_CSRMatrixI(diag);
    HYPRE_Int* J = hypre_CSRMatrixJ(diag);

    if (!I || !J) {
        throw std::runtime_error("FV-like reuse could not access host ParCSR diag I/J.");
    }

    c.permPatternToHypre.assign(c.nnz, -1);
    c.permIdentity = true;

    std::size_t p = 0;
    for (HYPRE_Int r = 0; r < c.n; ++r) {
        const HYPRE_Int p0 = p;
        const HYPRE_Int p1 = p0 + c.ncols[r];

        const HYPRE_Int h0 = I[r];
        const HYPRE_Int h1 = I[r + 1];

        if ((p1 - p0) != (h1 - h0)) {
            std::ostringstream os;
            os << "FV-like reuse row nnz mismatch at row " << r
               << ": pattern=" << (p1 - p0)
               << " hypre=" << (h1 - h0);
            throw std::runtime_error(os.str());
        }

        for (HYPRE_Int pp = p0; pp < p1; ++pp) {
            const HYPRE_Int col = c.cols[pp];
            HYPRE_Int found = -1;

            for (HYPRE_Int qq = h0; qq < h1; ++qq) {
                if (J[qq] == col) {
                    found = qq;
                    break;
                }
            }

            if (found < 0) {
                std::ostringstream os;
                os << "FV-like reuse could not match hypre column at row "
                   << r << " col " << col;
                throw std::runtime_error(os.str());
            }

            c.permPatternToHypre[pp] = found;
            if (found != pp) c.permIdentity = false;
        }

        p = p1;
    }

    c.valsPermuted.resize(c.nnz);
}

static inline void memoirs_hypre_fvlike_refresh_A_pointer(
    MemoirsHypreFvlikeReuseCache& c
) {
    hypre_CSRMatrix* diag = hypre_ParCSRMatrixDiag(c.parA);
    c.Adata = hypre_CSRMatrixData(diag);

    if (!c.Adata) {
        throw std::runtime_error("FV-like reuse could not access ParCSR diag data pointer.");
    }
}

static inline void memoirs_hypre_fvlike_update_matrix_values(
    const SparseRows& A,
    MemoirsHypreFvlikeReuseCache& c
) {
    memoirs_hypre_fvlike_fill_values(A, c);

    const HYPRE_Complex* src = c.vals.data();

    if (!c.permIdentity) {
        std::fill(c.valsPermuted.begin(), c.valsPermuted.end(), HYPRE_Complex(0));

        for (std::size_t p = 0; p < c.nnz; ++p) {
            c.valsPermuted[(std::size_t)c.permPatternToHypre[p]] = c.vals[p];
        }

        src = c.valsPermuted.data();
    }

    if (c.useDevice) {
        memoirs_cuda_check_fvlike(
            cudaMemcpy(c.Adata, src, c.nnz * sizeof(HYPRE_Complex),
                       cudaMemcpyHostToDevice),
            "cudaMemcpy ParCSR A values");
    } else {
        std::memcpy(c.Adata, src, c.nnz * sizeof(HYPRE_Complex));
    }
}

static inline void memoirs_hypre_fvlike_update_vectors(
    const AssembledSystem& sys,
    MemoirsHypreFvlikeReuseCache& c
) {
    c.bvals.resize(c.n);
    c.xvals.resize(c.n);

    for (HYPRE_Int i = 0; i < c.n; ++i) {
        c.bvals[i] = (HYPRE_Complex)sys.b[i];
        c.xvals[i] = HYPRE_Complex(0);
    }

    if (c.useDevice) {
        memoirs_cuda_check_fvlike(
            cudaMemcpy(c.bdata, c.bvals.data(), c.n * sizeof(HYPRE_Complex),
                       cudaMemcpyHostToDevice),
            "cudaMemcpy ParVector b");
        memoirs_cuda_check_fvlike(
            cudaMemcpy(c.xdata, c.xvals.data(), c.n * sizeof(HYPRE_Complex),
                       cudaMemcpyHostToDevice),
            "cudaMemcpy ParVector x");
    } else {
        std::memcpy(c.bdata, c.bvals.data(), c.n * sizeof(HYPRE_Complex));
        std::memcpy(c.xdata, c.xvals.data(), c.n * sizeof(HYPRE_Complex));
    }
}

static inline void memoirs_hypre_fvlike_copy_solution_to_host(
    MemoirsHypreFvlikeReuseCache& c,
    std::vector<Real>& xOut
) {
    c.xvals.resize(c.n);

    if (c.useDevice) {
        memoirs_cuda_check_fvlike(
            cudaMemcpy(c.xvals.data(), c.xdata, c.n * sizeof(HYPRE_Complex),
                       cudaMemcpyDeviceToHost),
            "cudaMemcpy ParVector x to host");
    } else {
        std::memcpy(c.xvals.data(), c.xdata, c.n * sizeof(HYPRE_Complex));
    }

    xOut.assign(c.n, Real(0));
    for (HYPRE_Int i = 0; i < c.n; ++i) {
        xOut[i] = (Real)c.xvals[i];
    }
}

static inline void memoirs_hypre_fvlike_create_vectors(
    MemoirsHypreFvlikeReuseCache& c
) {
    const HYPRE_BigInt ilower = 0;
    const HYPRE_BigInt iupper = c.n - 1;

    c.bvals.assign(c.n, HYPRE_Complex(0));
    c.xvals.assign(c.n, HYPRE_Complex(0));

    hypre_check(HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &c.ijb),
                "HYPRE_IJVectorCreate b fvlike");
    hypre_check(HYPRE_IJVectorSetObjectType(c.ijb, HYPRE_PARCSR),
                "HYPRE_IJVectorSetObjectType b fvlike");
    hypre_check(HYPRE_IJVectorInitialize_v2(
                    c.ijb, c.useDevice ? HYPRE_MEMORY_DEVICE : HYPRE_MEMORY_HOST),
                "HYPRE_IJVectorInitialize_v2 b fvlike");

    hypre_check(HYPRE_IJVectorSetValues(
                    c.ijb, c.n, c.rows.data(), c.bvals.data()),
                "HYPRE_IJVectorSetValues b fvlike");
    hypre_check(HYPRE_IJVectorAssemble(c.ijb),
                "HYPRE_IJVectorAssemble b fvlike");

    hypre_check(HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &c.ijx),
                "HYPRE_IJVectorCreate x fvlike");
    hypre_check(HYPRE_IJVectorSetObjectType(c.ijx, HYPRE_PARCSR),
                "HYPRE_IJVectorSetObjectType x fvlike");
    hypre_check(HYPRE_IJVectorInitialize_v2(
                    c.ijx, c.useDevice ? HYPRE_MEMORY_DEVICE : HYPRE_MEMORY_HOST),
                "HYPRE_IJVectorInitialize_v2 x fvlike");

    hypre_check(HYPRE_IJVectorSetValues(
                    c.ijx, c.n, c.rows.data(), c.xvals.data()),
                "HYPRE_IJVectorSetValues x fvlike");
    hypre_check(HYPRE_IJVectorAssemble(c.ijx),
                "HYPRE_IJVectorAssemble x fvlike");

    hypre_check(HYPRE_IJVectorGetObject(c.ijb, (void**)&c.parb),
                "HYPRE_IJVectorGetObject b fvlike");
    hypre_check(HYPRE_IJVectorGetObject(c.ijx, (void**)&c.parx),
                "HYPRE_IJVectorGetObject x fvlike");

    c.bdata = hypre_VectorData(hypre_ParVectorLocalVector(c.parb));
    c.xdata = hypre_VectorData(hypre_ParVectorLocalVector(c.parx));

    if (!c.bdata || !c.xdata) {
        throw std::runtime_error("FV-like reuse could not obtain ParVector data pointers.");
    }
}

static inline void memoirs_hypre_fvlike_create_solver(
    MemoirsHypreFvlikeReuseCache& c,
    const Options& opt,
    int gmresRestart,
    int amgCoarsenType,
    int amgInterpType,
    int amgRelaxType,
    int amgAggLevels,
    int amgKeepTranspose,
    int amgPmax,
    int amgNumSweeps,
    int amgRAP2,
    double amgTruncFactor,
    double amgStrongThreshold,
    int amgNumFunctions,
    int amgNodal,
    int amgNodalLevels,
    int amgKeepSameSign,
    int amgNodalDiag
) {
    c.solverKind = lower_copy(opt.solver);
    c.usePCG = (c.solverKind == "pcg");
    c.useBiCGSTAB = (c.solverKind == "bicgstab" || c.solverKind == "bcgs");
    c.useGMRES = (c.solverKind == "gmres");
    c.useFGMRES = (c.solverKind == "fgmres" || c.solverKind == "flexgmres");

    if (c.usePCG) {
        hypre_check(HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &c.krylov),
                    "HYPRE_ParCSRPCGCreate fvlike");
        hypre_check(HYPRE_PCGSetTol(c.krylov, opt.tol), "HYPRE_PCGSetTol fvlike");
        hypre_check(HYPRE_PCGSetMaxIter(c.krylov, opt.maxit), "HYPRE_PCGSetMaxIter fvlike");
        hypre_check(HYPRE_PCGSetTwoNorm(c.krylov, 1), "HYPRE_PCGSetTwoNorm fvlike");
        hypre_check(HYPRE_PCGSetPrintLevel(c.krylov, opt.hyprePrint), "HYPRE_PCGSetPrintLevel fvlike");
        hypre_check(HYPRE_PCGSetLogging(c.krylov, 1), "HYPRE_PCGSetLogging fvlike");
    } else if (c.useBiCGSTAB) {
        hypre_check(HYPRE_ParCSRBiCGSTABCreate(MPI_COMM_WORLD, &c.krylov),
                    "HYPRE_ParCSRBiCGSTABCreate fvlike");
        hypre_check(HYPRE_ParCSRBiCGSTABSetTol(c.krylov, opt.tol), "HYPRE_BiCGSTABSetTol fvlike");
        hypre_check(HYPRE_ParCSRBiCGSTABSetAbsoluteTol(c.krylov, 0.0), "HYPRE_BiCGSTABSetAbsTol fvlike");
        hypre_check(HYPRE_ParCSRBiCGSTABSetMaxIter(c.krylov, opt.maxit), "HYPRE_BiCGSTABSetMaxIter fvlike");
        hypre_check(HYPRE_ParCSRBiCGSTABSetPrintLevel(c.krylov, opt.hyprePrint), "HYPRE_BiCGSTABSetPrintLevel fvlike");
        hypre_check(HYPRE_ParCSRBiCGSTABSetLogging(c.krylov, 1), "HYPRE_BiCGSTABSetLogging fvlike");
    } else if (c.useGMRES) {
        hypre_check(HYPRE_ParCSRGMRESCreate(MPI_COMM_WORLD, &c.krylov),
                    "HYPRE_ParCSRGMRESCreate fvlike");
        hypre_check(HYPRE_ParCSRGMRESSetTol(c.krylov, opt.tol), "HYPRE_GMRESSetTol fvlike");
        hypre_check(HYPRE_ParCSRGMRESSetAbsoluteTol(c.krylov, 0.0), "HYPRE_GMRESSetAbsTol fvlike");
        hypre_check(HYPRE_ParCSRGMRESSetMaxIter(c.krylov, opt.maxit), "HYPRE_GMRESSetMaxIter fvlike");
        hypre_check(HYPRE_ParCSRGMRESSetKDim(c.krylov, gmresRestart), "HYPRE_GMRESSetKDim fvlike");
        hypre_check(HYPRE_ParCSRGMRESSetPrintLevel(c.krylov, opt.hyprePrint), "HYPRE_GMRESSetPrintLevel fvlike");
        hypre_check(HYPRE_ParCSRGMRESSetLogging(c.krylov, 1), "HYPRE_GMRESSetLogging fvlike");
    } else if (c.useFGMRES) {
        hypre_check(HYPRE_ParCSRFlexGMRESCreate(MPI_COMM_WORLD, &c.krylov),
                    "HYPRE_ParCSRFlexGMRESCreate fvlike");
        hypre_check(HYPRE_ParCSRFlexGMRESSetTol(c.krylov, opt.tol), "HYPRE_FGMRESSetTol fvlike");
        hypre_check(HYPRE_ParCSRFlexGMRESSetMaxIter(c.krylov, opt.maxit), "HYPRE_FGMRESSetMaxIter fvlike");
        hypre_check(HYPRE_ParCSRFlexGMRESSetKDim(c.krylov, gmresRestart), "HYPRE_FGMRESSetKDim fvlike");
        hypre_check(HYPRE_ParCSRFlexGMRESSetPrintLevel(c.krylov, opt.hyprePrint), "HYPRE_FGMRESSetPrintLevel fvlike");
        hypre_check(HYPRE_ParCSRFlexGMRESSetLogging(c.krylov, 1), "HYPRE_FGMRESSetLogging fvlike");
    } else {
        throw std::runtime_error("FV-like reuse supports pcg|bicgstab|gmres|fgmres.");
    }

    const std::string pre = lower_copy(opt.precond);
    if (pre == "amg" || pre == "boomeramg") {
        hypre_check(HYPRE_BoomerAMGCreate(&c.precond),
                    "HYPRE_BoomerAMGCreate fvlike");
        hypre_check(HYPRE_BoomerAMGSetPrintLevel(c.precond, opt.hyprePrint),
                    "HYPRE_BoomerAMGSetPrintLevel fvlike");
        hypre_check(HYPRE_BoomerAMGSetCoarsenType(c.precond, amgCoarsenType),
                    "HYPRE_BoomerAMGSetCoarsenType fvlike");
        hypre_check(HYPRE_BoomerAMGSetInterpType(c.precond, amgInterpType),
                    "HYPRE_BoomerAMGSetInterpType fvlike");
        hypre_check(HYPRE_BoomerAMGSetRelaxType(c.precond, amgRelaxType),
                    "HYPRE_BoomerAMGSetRelaxType fvlike");
        hypre_check(HYPRE_BoomerAMGSetNumSweeps(c.precond, amgNumSweeps),
                    "HYPRE_BoomerAMGSetNumSweeps fvlike");
        hypre_check(HYPRE_BoomerAMGSetTol(c.precond, 0.0),
                    "HYPRE_BoomerAMGSetTol fvlike");
        hypre_check(HYPRE_BoomerAMGSetMaxIter(c.precond, 1),
                    "HYPRE_BoomerAMGSetMaxIter fvlike");
        hypre_check(HYPRE_BoomerAMGSetRelaxOrder(c.precond, 0),
                    "HYPRE_BoomerAMGSetRelaxOrder fvlike");
        hypre_check(HYPRE_BoomerAMGSetPMaxElmts(c.precond, amgPmax),
                    "HYPRE_BoomerAMGSetPMaxElmts fvlike");
        hypre_check(HYPRE_BoomerAMGSetKeepTranspose(c.precond, amgKeepTranspose),
                    "HYPRE_BoomerAMGSetKeepTranspose fvlike");
        hypre_check(HYPRE_BoomerAMGSetTruncFactor(c.precond, amgTruncFactor),
                    "HYPRE_BoomerAMGSetTruncFactor fvlike");
        hypre_check(HYPRE_BoomerAMGSetRAP2(c.precond, amgRAP2),
                    "HYPRE_BoomerAMGSetRAP2 fvlike");

        if (amgNumFunctions > 1) {
            hypre_check(HYPRE_BoomerAMGSetNumFunctions(c.precond, amgNumFunctions),
                        "HYPRE_BoomerAMGSetNumFunctions fvlike");
            if (amgNodal != 0) {
                hypre_check(HYPRE_BoomerAMGSetNodal(c.precond, amgNodal),
                            "HYPRE_BoomerAMGSetNodal fvlike");
                hypre_check(HYPRE_BoomerAMGSetNodalLevels(c.precond, amgNodalLevels),
                            "HYPRE_BoomerAMGSetNodalLevels fvlike");
                hypre_check(HYPRE_BoomerAMGSetKeepSameSign(c.precond, amgKeepSameSign),
                            "HYPRE_BoomerAMGSetKeepSameSign fvlike");
                hypre_check(HYPRE_BoomerAMGSetNodalDiag(c.precond, amgNodalDiag),
                            "HYPRE_BoomerAMGSetNodalDiag fvlike");
            }
        }

        if (amgAggLevels > 0) {
            hypre_check(HYPRE_BoomerAMGSetAggNumLevels(c.precond, amgAggLevels),
                        "HYPRE_BoomerAMGSetAggNumLevels fvlike");
            hypre_check(HYPRE_BoomerAMGSetAggInterpType(c.precond, 4),
                        "HYPRE_BoomerAMGSetAggInterpType fvlike");
        }

        if (amgStrongThreshold >= 0.0) {
            hypre_check(HYPRE_BoomerAMGSetStrongThreshold(c.precond, amgStrongThreshold),
                        "HYPRE_BoomerAMGSetStrongThreshold fvlike");
        }

        if (c.usePCG) {
            hypre_check(HYPRE_ParCSRPCGSetPrecond(
                            c.krylov,
                            (HYPRE_PtrToParSolverFcn)HYPRE_BoomerAMGSolve,
                            (HYPRE_PtrToParSolverFcn)HYPRE_BoomerAMGSetup,
                            c.precond),
                        "HYPRE_PCGSetPrecond AMG fvlike");
        } else if (c.useBiCGSTAB) {
            hypre_check(HYPRE_ParCSRBiCGSTABSetPrecond(
                            c.krylov,
                            (HYPRE_PtrToParSolverFcn)HYPRE_BoomerAMGSolve,
                            (HYPRE_PtrToParSolverFcn)HYPRE_BoomerAMGSetup,
                            c.precond),
                        "HYPRE_BiCGSTABSetPrecond AMG fvlike");
        } else if (c.useGMRES) {
            hypre_check(HYPRE_ParCSRGMRESSetPrecond(
                            c.krylov,
                            (HYPRE_PtrToParSolverFcn)HYPRE_BoomerAMGSolve,
                            (HYPRE_PtrToParSolverFcn)HYPRE_BoomerAMGSetup,
                            c.precond),
                        "HYPRE_GMRESSetPrecond AMG fvlike");
        } else {
            hypre_check(HYPRE_ParCSRFlexGMRESSetPrecond(
                            c.krylov,
                            (HYPRE_PtrToParSolverFcn)HYPRE_BoomerAMGSolve,
                            (HYPRE_PtrToParSolverFcn)HYPRE_BoomerAMGSetup,
                            c.precond),
                        "HYPRE_FGMRESSetPrecond AMG fvlike");
        }
    } else if (pre == "diag" || pre == "diagscale") {
        if (c.usePCG) {
            hypre_check(HYPRE_ParCSRPCGSetPrecond(
                            c.krylov,
                            (HYPRE_PtrToParSolverFcn)HYPRE_ParCSRDiagScale,
                            (HYPRE_PtrToParSolverFcn)HYPRE_ParCSRDiagScaleSetup,
                            nullptr),
                        "HYPRE_PCGSetPrecond DiagScale fvlike");
        } else if (c.useBiCGSTAB) {
            hypre_check(HYPRE_ParCSRBiCGSTABSetPrecond(
                            c.krylov,
                            (HYPRE_PtrToParSolverFcn)HYPRE_ParCSRDiagScale,
                            (HYPRE_PtrToParSolverFcn)HYPRE_ParCSRDiagScaleSetup,
                            nullptr),
                        "HYPRE_BiCGSTABSetPrecond DiagScale fvlike");
        } else if (c.useGMRES) {
            hypre_check(HYPRE_ParCSRGMRESSetPrecond(
                            c.krylov,
                            (HYPRE_PtrToParSolverFcn)HYPRE_ParCSRDiagScale,
                            (HYPRE_PtrToParSolverFcn)HYPRE_ParCSRDiagScaleSetup,
                            nullptr),
                        "HYPRE_GMRESSetPrecond DiagScale fvlike");
        } else {
            hypre_check(HYPRE_ParCSRFlexGMRESSetPrecond(
                            c.krylov,
                            (HYPRE_PtrToParSolverFcn)HYPRE_ParCSRDiagScale,
                            (HYPRE_PtrToParSolverFcn)HYPRE_ParCSRDiagScaleSetup,
                            nullptr),
                        "HYPRE_FGMRESSetPrecond DiagScale fvlike");
        }
    } else if (pre == "none") {
        // no preconditioner
    } else {
        throw std::runtime_error("Unsupported -precond for FV-like reuse: " + opt.precond);
    }
}

static SolveReport solve_hypre_ij_krylov_fvlike_reuse(
    const PolyMesh& m,
    const LinearCgDofMap& dm,
    const AssembledSystem& sys,
    const std::string& mms,
    const Options& opt,
    std::vector<Real>& xOut
) {
    (void)m;
    (void)dm;
    (void)mms;

    SolveReport rep;
    rep.hypreFvReuseEnabled = 1;

    const std::string solver = lower_copy(opt.solver);
    const bool usePCG = (solver == "pcg");
    const bool useBiCGSTAB = (solver == "bicgstab" || solver == "bcgs");
    const bool useGMRES = (solver == "gmres");
    const bool useFGMRES = (solver == "fgmres" || solver == "flexgmres");

    if (!(usePCG || useBiCGSTAB || useGMRES || useFGMRES)) {
        throw std::runtime_error("FV-like reuse supports -solver pcg|bicgstab|gmres|fgmres");
    }

    if (!sys.A.fixedPattern) {
        throw std::runtime_error("FV-like reuse requires SparseRows fixedPattern=true.");
    }

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
    const int amgNumFunctions = memoirs_env_int("MEMOIRS_AMG_NUM_FUNCTIONS", 1);
    const int amgNodal = memoirs_env_int("MEMOIRS_AMG_NODAL", 0);
    const int amgNodalLevels = memoirs_env_int("MEMOIRS_AMG_NODAL_LEVELS", 1);
    const int amgKeepSameSign = memoirs_env_int("MEMOIRS_AMG_KEEP_SAME_SIGN", 0);
    const int amgNodalDiag = memoirs_env_int("MEMOIRS_AMG_NODAL_DIAG", 0);
    const int gmresRestart = memoirs_env_int("MEMOIRS_GMRES_RESTART", 80);

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
    rep.amgNumFunctions = amgNumFunctions;
    rep.amgNodal = amgNodal;
    rep.amgNodalLevels = amgNodalLevels;
    rep.amgKeepSameSign = amgKeepSameSign;
    rep.amgNodalDiag = amgNodalDiag;

    auto elapsed = [](const std::chrono::steady_clock::time_point& a,
                      const std::chrono::steady_clock::time_point& b) -> double {
        return std::chrono::duration<double>(b - a).count();
    };

    int mpiWasFinalized = 0;
    MPI_Finalized(&mpiWasFinalized);
    if (mpiWasFinalized) {
        throw std::runtime_error("MPI already finalized before FV-like reuse solve.");
    }

    int mpiWasInitialized = 0;
    MPI_Initialized(&mpiWasInitialized);
    if (!mpiWasInitialized) {
        int argc = 0;
        char** argv = nullptr;
        MPI_Init(&argc, &argv);
    }

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 1) {
        throw std::runtime_error("FV-like reuse currently supports one MPI rank only.");
    }

    const HYPRE_MemoryLocation requestedMem =
        parse_hypre_memory_location(opt.hypreMemory);
    const bool useDevice = (requestedMem == HYPRE_MEMORY_DEVICE);

    memoirs_hypre_krylov_runtime_init_once(useDevice, rank);

    const HYPRE_Int n = (HYPRE_Int)sparse_nrows(sys.A);
    const std::size_t nnz = sparse_nnz(sys.A);
    const void* pid = hypre_sparse_pattern_identity(sys.A);

    const std::string key = memoirs_hypre_fvlike_make_key(
        n, nnz, pid, opt,
        amgCoarsenType, amgInterpType, amgRelaxType,
        amgAggLevels, amgKeepTranspose, amgPmax,
        amgNumSweeps, amgRAP2, amgTruncFactor, amgStrongThreshold,
        amgNumFunctions, amgNodal, amgNodalLevels,
        amgKeepSameSign, amgNodalDiag, useDevice);

    auto& c = memoirs_hypre_fvlike_reuse_cache_singleton();

    const bool hit =
        c.valid &&
        c.n == n &&
        c.nnz == nnz &&
        c.patternPtr == pid &&
        c.key == key;

    if (!hit) {
        memoirs_hypre_fvlike_reuse_reset();

        c.valid = true;
        c.setupValid = false;
        c.useDevice = useDevice;
        c.n = n;
        c.nnz = nnz;
        c.patternPtr = pid;
        c.key = key;

        c.usePCG = usePCG;
        c.useBiCGSTAB = useBiCGSTAB;
        c.useGMRES = useGMRES;
        c.useFGMRES = useFGMRES;

        ++c.builds;
        rep.hypreFvReuseBuild = 1;
    } else {
        ++c.hits;
        rep.hypreFvReuseHit = 1;
    }

    auto tMat0 = std::chrono::steady_clock::now();

    if (rep.hypreFvReuseBuild) {
        memoirs_hypre_fvlike_fill_pattern(sys.A, c);
        memoirs_hypre_fvlike_fill_values(sys.A, c);

        const HYPRE_BigInt ilower = 0;
        const HYPRE_BigInt iupper = n - 1;

        hypre_check(HYPRE_IJMatrixCreate(
                        MPI_COMM_WORLD, ilower, iupper, ilower, iupper, &c.ijA),
                    "HYPRE_IJMatrixCreate fvlike");
        hypre_check(HYPRE_IJMatrixSetObjectType(c.ijA, HYPRE_PARCSR),
                    "HYPRE_IJMatrixSetObjectType fvlike");
        hypre_check(HYPRE_IJMatrixSetRowSizes(c.ijA, c.ncols.data()),
                    "HYPRE_IJMatrixSetRowSizes fvlike");
        hypre_check(HYPRE_IJMatrixInitialize_v2(c.ijA, HYPRE_MEMORY_HOST),
                    "HYPRE_IJMatrixInitialize_v2 A HOST fvlike");
        hypre_check(HYPRE_IJMatrixSetValues(
                        c.ijA, n, c.ncols.data(),
                        c.rows.data(), c.cols.data(), c.vals.data()),
                    "HYPRE_IJMatrixSetValues A fvlike");
        hypre_check(HYPRE_IJMatrixAssemble(c.ijA),
                    "HYPRE_IJMatrixAssemble A fvlike");

        hypre_check(HYPRE_IJMatrixGetObject(c.ijA, (void**)&c.parA),
                    "HYPRE_IJMatrixGetObject A host fvlike");

        memoirs_hypre_fvlike_build_diag_permutation(c);

        if (useDevice) {
            hypre_check(HYPRE_IJMatrixMigrate(c.ijA, HYPRE_MEMORY_DEVICE),
                        "HYPRE_IJMatrixMigrate A DEVICE fvlike");
            hypre_check(HYPRE_IJMatrixGetObject(c.ijA, (void**)&c.parA),
                        "HYPRE_IJMatrixGetObject A device fvlike");
        }

        memoirs_hypre_fvlike_refresh_A_pointer(c);
        memoirs_hypre_fvlike_update_matrix_values(sys.A, c);

        memoirs_hypre_fvlike_create_vectors(c);
        memoirs_hypre_fvlike_update_vectors(sys, c);

        memoirs_hypre_fvlike_create_solver(
            c, opt, gmresRestart,
            amgCoarsenType, amgInterpType, amgRelaxType,
            amgAggLevels, amgKeepTranspose, amgPmax,
            amgNumSweeps, amgRAP2, amgTruncFactor, amgStrongThreshold,
            amgNumFunctions, amgNodal, amgNodalLevels,
            amgKeepSameSign, amgNodalDiag);
    } else {
        memoirs_hypre_fvlike_update_matrix_values(sys.A, c);
        memoirs_hypre_fvlike_update_vectors(sys, c);
    }

    auto tMat1 = std::chrono::steady_clock::now();
    rep.hypreMatrixInsertSeconds = elapsed(tMat0, tMat1);
    rep.hypreMatrixMigrateSeconds = 0.0;
    rep.hypreVectorInsertSeconds = 0.0;
    rep.hypreVectorMigrateSeconds = 0.0;

    auto tSetup0 = std::chrono::steady_clock::now();

    if (!c.setupValid) {
        rep.hypreFvReuseSetup = 1;

        if (c.usePCG) {
            hypre_check(HYPRE_ParCSRPCGSetup(c.krylov, c.parA, c.parb, c.parx),
                        "HYPRE_ParCSRPCGSetup fvlike");
        } else if (c.useBiCGSTAB) {
            hypre_check(HYPRE_ParCSRBiCGSTABSetup(c.krylov, c.parA, c.parb, c.parx),
                        "HYPRE_ParCSRBiCGSTABSetup fvlike");
        } else if (c.useGMRES) {
            hypre_check(HYPRE_ParCSRGMRESSetup(c.krylov, c.parA, c.parb, c.parx),
                        "HYPRE_ParCSRGMRESSetup fvlike");
        } else {
            hypre_check(HYPRE_ParCSRFlexGMRESSetup(c.krylov, c.parA, c.parb, c.parx),
                        "HYPRE_ParCSRFlexGMRESSetup fvlike");
        }

        c.setupValid = true;
    }

    auto tSetup1 = std::chrono::steady_clock::now();
    rep.hypreSetupSeconds = elapsed(tSetup0, tSetup1);

    auto tSolve0 = std::chrono::steady_clock::now();

    HYPRE_Int solveStatus = 0;
    if (c.usePCG) {
        solveStatus = HYPRE_ParCSRPCGSolve(c.krylov, c.parA, c.parb, c.parx);
    } else if (c.useBiCGSTAB) {
        solveStatus = HYPRE_ParCSRBiCGSTABSolve(c.krylov, c.parA, c.parb, c.parx);
    } else if (c.useGMRES) {
        solveStatus = HYPRE_ParCSRGMRESSolve(c.krylov, c.parA, c.parb, c.parx);
    } else {
        solveStatus = HYPRE_ParCSRFlexGMRESSolve(c.krylov, c.parA, c.parb, c.parx);
    }

    auto tSolve1 = std::chrono::steady_clock::now();
    rep.hypreSolveOnlySeconds = elapsed(tSolve0, tSolve1);
    rep.hypreSolveOnlyAvgSeconds = rep.hypreSolveOnlySeconds;
    rep.hypreSolveStatus = int(solveStatus);

    if (solveStatus != 0 && rank == 0) {
        std::cerr << "WARNING: FV-like HYPRE Krylov solve returned status "
                  << int(solveStatus)
                  << "; querying iterations/final residual anyway.\n";
    }

    HYPRE_Real finalRel = 0.0;
    if (c.usePCG) {
        hypre_check(HYPRE_PCGGetNumIterations(c.krylov, &rep.iterations),
                    "HYPRE_PCGGetNumIterations fvlike");
        hypre_check(HYPRE_PCGGetFinalRelativeResidualNorm(c.krylov, &finalRel),
                    "HYPRE_PCGGetFinalRelativeResidualNorm fvlike");
    } else if (c.useBiCGSTAB) {
        hypre_check(HYPRE_ParCSRBiCGSTABGetNumIterations(c.krylov, &rep.iterations),
                    "HYPRE_BiCGSTABGetNumIterations fvlike");
        hypre_check(HYPRE_ParCSRBiCGSTABGetFinalRelativeResidualNorm(c.krylov, &finalRel),
                    "HYPRE_BiCGSTABGetFinalRelativeResidualNorm fvlike");
    } else if (c.useGMRES) {
        hypre_check(HYPRE_ParCSRGMRESGetNumIterations(c.krylov, &rep.iterations),
                    "HYPRE_GMRESGetNumIterations fvlike");
        hypre_check(HYPRE_ParCSRGMRESGetFinalRelativeResidualNorm(c.krylov, &finalRel),
                    "HYPRE_GMRESGetFinalRelativeResidualNorm fvlike");
    } else {
        hypre_check(HYPRE_ParCSRFlexGMRESGetNumIterations(c.krylov, &rep.iterations),
                    "HYPRE_FGMRESGetNumIterations fvlike");
        hypre_check(HYPRE_ParCSRFlexGMRESGetFinalRelativeResidualNorm(c.krylov, &finalRel),
                    "HYPRE_FGMRESGetFinalRelativeResidualNorm fvlike");
    }

    rep.finalRelRes = (double)finalRel;

    auto tGet0 = std::chrono::steady_clock::now();
    memoirs_hypre_fvlike_copy_solution_to_host(c, xOut);
    auto tGet1 = std::chrono::steady_clock::now();
    rep.hypreGetSolutionSeconds = elapsed(tGet0, tGet1);

    rep.errorNormSeconds = 0.0;
    rep.nodalErrorL2 = -1.0;
    rep.nodalErrorMax = -1.0;
    rep.L2Error = -1.0;
    rep.H1SemiError = -1.0;
    rep.hypreDestroyFinalizeSeconds = 0.0;
    rep.solveRepeats = 1;
    rep.rhsUpdateMode = "fvlike_reuse";

    return rep;
}

static SolveReport solve_hypre_ij_krylov(
    const PolyMesh& m,
    const LinearCgDofMap& dm,
    const AssembledSystem& sys,
    const std::string& mms,
    const Options& opt,
    std::vector<Real>& xOut
) {
    const std::string solver = lower_copy(opt.solver);
    const bool usePCG = (solver == "pcg");
    const bool useBiCGSTAB = (solver == "bicgstab" || solver == "bcgs");
    const bool useGMRES = (solver == "gmres");
    const bool useFGMRES = (solver == "fgmres" || solver == "flexgmres");
    if (!(usePCG || useBiCGSTAB || useGMRES || useFGMRES)) {
        throw std::runtime_error("HYPRE IJ Krylov helper supports -solver pcg|bicgstab|gmres|fgmres");
    }

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

    if (memoirs_env_bool("MEMOIRS_HYPRE_FVLIKE_REUSE_WITHIN_STEP", false) &&
        sys.A.fixedPattern) {
        return solve_hypre_ij_krylov_fvlike_reuse(m, dm, sys, mms, opt, xOut);
    }

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
    const int amgNumFunctions = memoirs_env_int("MEMOIRS_AMG_NUM_FUNCTIONS", 1);
    const int amgNodal = memoirs_env_int("MEMOIRS_AMG_NODAL", 0);
    const int amgNodalLevels = memoirs_env_int("MEMOIRS_AMG_NODAL_LEVELS", 1);
    const int amgKeepSameSign = memoirs_env_int("MEMOIRS_AMG_KEEP_SAME_SIGN", 0);
    const int amgNodalDiag = memoirs_env_int("MEMOIRS_AMG_NODAL_DIAG", 0);
    const int gmresRestart = memoirs_env_int("MEMOIRS_GMRES_RESTART", 80);
    bool amgReuseWithinStep = memoirs_env_bool("MEMOIRS_HYPRE_AMG_REUSE_WITHIN_STEP", false);
    if (amgReuseWithinStep) {
        std::cerr
            << "WARNING: MEMOIRS_HYPRE_AMG_REUSE_WITHIN_STEP is currently disabled. "
            << "The cached BoomerAMG/no-op setup experiment crashed inside hypre_BoomerAMGSolve. "
            << "Falling back to AMG rebuild every Picard.\n";
        amgReuseWithinStep = false;
    }

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
    rep.amgNumFunctions = amgNumFunctions;
    rep.amgNodal = amgNodal;
    rep.amgNodalLevels = amgNodalLevels;
    rep.amgKeepSameSign = amgKeepSameSign;
    rep.amgNodalDiag = amgNodalDiag;
    rep.ijUseRowSizes = ijUseRowSizes ? 1 : 0;
    rep.ijBulkInsert = ijBulkInsert ? 1 : 0;
    rep.amgReuseEnabled = amgReuseWithinStep ? 1 : 0;

    auto elapsed = [](const std::chrono::steady_clock::time_point& a,
                      const std::chrono::steady_clock::time_point& b) -> double {
        return std::chrono::duration<double>(b - a).count();
    };

    int mpiWasFinalized = 0;
    MPI_Finalized(&mpiWasFinalized);
    if (mpiWasFinalized) {
        throw std::runtime_error("MPI was already finalized before solve_hypre_ij_krylov; runtime lifecycle is invalid.");
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

    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (size != 1) {
        throw std::runtime_error("Patch 004 one-shot HYPRE path supports only one MPI rank.");
    }

    const HYPRE_MemoryLocation requestedMem = parse_hypre_memory_location(opt.hypreMemory);
    const bool useDevice = (requestedMem == HYPRE_MEMORY_DEVICE);

    memoirs_hypre_krylov_runtime_init_once(useDevice, rank);

    const HYPRE_Int n = (HYPRE_Int)sparse_nrows(sys.A);
    const std::string amgReuseKey = memoirs_hypre_amg_reuse_make_key(
        n, opt,
        amgCoarsenType, amgInterpType, amgRelaxType,
        amgAggLevels, amgKeepTranspose, amgPmax,
        amgNumSweeps, amgRAP2, amgTruncFactor, amgStrongThreshold,
        amgNumFunctions, amgNodal, amgNodalLevels,
        amgKeepSameSign, amgNodalDiag);
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
    HypreIjCachedPattern* cachedIj = nullptr;
    const HYPRE_Complex* cachedIjDirectVals = nullptr;

    std::vector<HYPRE_Int> ijNcols;
    std::vector<HYPRE_Int> ijRows;
    std::vector<HYPRE_Int> ijCols;
    std::vector<HYPRE_Complex> ijVals;

    if (sys.A.fixedPattern &&
        memoirs_env_bool("MEMOIRS_Q1Q1_CACHE_IJ_PATTERN", true)) {
        auto& cache = hypre_ij_cached_pattern_singleton();
        hypre_try_build_cached_ij_pattern(sys.A, cache);

        const bool directIjValues =
            memoirs_env_bool("MEMOIRS_Q1Q1_DIRECT_IJ_VALUES", true) &&
            sizeof(Real) == sizeof(HYPRE_Complex);

        if (directIjValues) {
            cachedIjDirectVals =
                reinterpret_cast<const HYPRE_Complex*>(sys.A.flatVals.data());
        } else {
            hypre_fill_cached_ij_values(sys.A, cache);
            cachedIjDirectVals = cache.vals.data();
        }

        cachedIj = &cache;
    } else {
        ijNcols.resize(n);
        ijRows.resize(n);

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
    }

    const HYPRE_Int setN = cachedIj ? cachedIj->n : n;
    const HYPRE_Int* setNcols = cachedIj ? cachedIj->ncols.data() : ijNcols.data();
    const HYPRE_Int* setRows = cachedIj ? cachedIj->rows.data() : ijRows.data();
    const HYPRE_Int* setCols = cachedIj ? cachedIj->cols.data() : ijCols.data();
    const HYPRE_Complex* setVals = cachedIj ? cachedIjDirectVals : ijVals.data();

    if (ijUseRowSizes) {
        hypre_check(HYPRE_IJMatrixSetRowSizes(ijA, const_cast<HYPRE_Int*>(setNcols)),
                    "HYPRE_IJMatrixSetRowSizes");
    }

    hypre_initialize_ij_matrix_for_host_insertion(ijA);

    if (ijBulkInsert) {
        hypre_check(HYPRE_IJMatrixSetValues(
            ijA,
            setN,
            const_cast<HYPRE_Int*>(setNcols),
            const_cast<HYPRE_Int*>(setRows),
            const_cast<HYPRE_Int*>(setCols),
            const_cast<HYPRE_Complex*>(setVals)),
            "HYPRE_IJMatrixSetValues bulk");
    } else {
        size_t off = 0;
        for (HYPRE_Int i = 0; i < setN; ++i) {
            HYPRE_Int rowId = setRows[i];
            HYPRE_Int ncols = setNcols[i];
            hypre_check(HYPRE_IJMatrixSetValues(
                ijA,
                1,
                &ncols,
                &rowId,
                const_cast<HYPRE_Int*>(setCols + off),
                const_cast<HYPRE_Complex*>(setVals + off)),
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
    // HYPRE Krylov/BoomerAMG create/configure/setup.
    // This copied NSE branch supports GMRES/FlexGMRES for nonsymmetric PSPG/SUPG
    // saddle-point systems while the original Poisson PCG helper stays intact.
    // ----------------------------------------------------------------------
    auto tSetup0 = std::chrono::steady_clock::now();

    HYPRE_Solver krylov = nullptr;
    if (usePCG) {
        hypre_check(HYPRE_ParCSRPCGCreate(comm, &krylov), "HYPRE_ParCSRPCGCreate");
        hypre_check(HYPRE_PCGSetMaxIter(krylov, opt.maxit), "HYPRE_PCGSetMaxIter");
        hypre_check(HYPRE_PCGSetTol(krylov, opt.tol), "HYPRE_PCGSetTol");
        hypre_check(HYPRE_PCGSetTwoNorm(krylov, 1), "HYPRE_PCGSetTwoNorm");
        hypre_check(HYPRE_PCGSetPrintLevel(krylov, opt.hyprePrint), "HYPRE_PCGSetPrintLevel");
        hypre_check(HYPRE_PCGSetLogging(krylov, 1), "HYPRE_PCGSetLogging");
    } else if (useBiCGSTAB) {
        hypre_check(HYPRE_ParCSRBiCGSTABCreate(comm, &krylov), "HYPRE_ParCSRBiCGSTABCreate");
        hypre_check(HYPRE_ParCSRBiCGSTABSetTol(krylov, opt.tol), "HYPRE_ParCSRBiCGSTABSetTol");
        hypre_check(HYPRE_ParCSRBiCGSTABSetAbsoluteTol(krylov, 0.0), "HYPRE_ParCSRBiCGSTABSetAbsoluteTol");
        hypre_check(HYPRE_ParCSRBiCGSTABSetMaxIter(krylov, opt.maxit), "HYPRE_ParCSRBiCGSTABSetMaxIter");
        hypre_check(HYPRE_ParCSRBiCGSTABSetPrintLevel(krylov, opt.hyprePrint), "HYPRE_ParCSRBiCGSTABSetPrintLevel");
        hypre_check(HYPRE_ParCSRBiCGSTABSetLogging(krylov, 1), "HYPRE_ParCSRBiCGSTABSetLogging");
    } else if (useGMRES) {
        hypre_check(HYPRE_ParCSRGMRESCreate(comm, &krylov), "HYPRE_ParCSRGMRESCreate");
        hypre_check(HYPRE_ParCSRGMRESSetTol(krylov, opt.tol), "HYPRE_ParCSRGMRESSetTol");
        hypre_check(HYPRE_ParCSRGMRESSetAbsoluteTol(krylov, 0.0), "HYPRE_ParCSRGMRESSetAbsoluteTol");
        hypre_check(HYPRE_ParCSRGMRESSetMaxIter(krylov, opt.maxit), "HYPRE_ParCSRGMRESSetMaxIter");
        hypre_check(HYPRE_ParCSRGMRESSetKDim(krylov, std::max(10, gmresRestart)), "HYPRE_ParCSRGMRESSetKDim");
        hypre_check(HYPRE_ParCSRGMRESSetPrintLevel(krylov, opt.hyprePrint), "HYPRE_ParCSRGMRESSetPrintLevel");
        hypre_check(HYPRE_ParCSRGMRESSetLogging(krylov, 1), "HYPRE_ParCSRGMRESSetLogging");
    } else {
        hypre_check(HYPRE_ParCSRFlexGMRESCreate(comm, &krylov), "HYPRE_ParCSRFlexGMRESCreate");
        hypre_check(HYPRE_ParCSRFlexGMRESSetTol(krylov, opt.tol), "HYPRE_ParCSRFlexGMRESSetTol");
        hypre_check(HYPRE_ParCSRFlexGMRESSetAbsoluteTol(krylov, 0.0), "HYPRE_ParCSRFlexGMRESSetAbsoluteTol");
        hypre_check(HYPRE_ParCSRFlexGMRESSetMaxIter(krylov, opt.maxit), "HYPRE_ParCSRFlexGMRESSetMaxIter");
        hypre_check(HYPRE_ParCSRFlexGMRESSetKDim(krylov, std::max(10, gmresRestart)), "HYPRE_ParCSRFlexGMRESSetKDim");
        hypre_check(HYPRE_ParCSRFlexGMRESSetPrintLevel(krylov, opt.hyprePrint), "HYPRE_ParCSRFlexGMRESSetPrintLevel");
        hypre_check(HYPRE_ParCSRFlexGMRESSetLogging(krylov, 1), "HYPRE_ParCSRFlexGMRESSetLogging");
    }

    auto set_precond = [&](HYPRE_PtrToParSolverFcn solveFcn,
                           HYPRE_PtrToParSolverFcn setupFcn,
                           HYPRE_Solver pc) {
        if (usePCG) {
            hypre_check(HYPRE_PCGSetPrecond(krylov,
                                            (HYPRE_PtrToSolverFcn)solveFcn,
                                            (HYPRE_PtrToSolverFcn)setupFcn,
                                            pc),
                        "HYPRE_PCGSetPrecond");
        } else if (useBiCGSTAB) {
            hypre_check(HYPRE_ParCSRBiCGSTABSetPrecond(krylov, solveFcn, setupFcn, pc),
                        "HYPRE_ParCSRBiCGSTABSetPrecond");
        } else if (useGMRES) {
            hypre_check(HYPRE_ParCSRGMRESSetPrecond(krylov, solveFcn, setupFcn, pc),
                        "HYPRE_ParCSRGMRESSetPrecond");
        } else {
            hypre_check(HYPRE_ParCSRFlexGMRESSetPrecond(krylov, solveFcn, setupFcn, pc),
                        "HYPRE_ParCSRFlexGMRESSetPrecond");
        }
    };

    HYPRE_Solver precond = nullptr;
    const std::string pre = lower_copy(opt.precond);

    if (pre == "diagscale") {
        set_precond((HYPRE_PtrToParSolverFcn)HYPRE_ParCSRDiagScale,
                    (HYPRE_PtrToParSolverFcn)HYPRE_ParCSRDiagScaleSetup,
                    nullptr);
    } else if (pre == "amg" || pre == "boomeramg") {
        MemoirsHypreAmgReuseCache& amgReuseCache =
            memoirs_hypre_amg_reuse_cache_singleton();

        const bool canReuseAmg = amgReuseWithinStep;
        const bool reuseHit =
            canReuseAmg &&
            amgReuseCache.valid &&
            amgReuseCache.precond &&
            amgReuseCache.n == n &&
            amgReuseCache.key == amgReuseKey;

        if (reuseHit) {
            precond = amgReuseCache.precond;
            rep.amgReuseHit = 1;
            ++amgReuseCache.hits;

            set_precond((HYPRE_PtrToParSolverFcn)HYPRE_BoomerAMGSolve,
                        (HYPRE_PtrToParSolverFcn)memoirs_hypre_amg_reuse_noop_setup,
                        precond);
        } else {
            if (canReuseAmg && amgReuseCache.valid) {
                memoirs_hypre_amg_reuse_reset();
            }
            if (canReuseAmg) {
                rep.amgReuseBuild = 1;
            }

        hypre_check(HYPRE_BoomerAMGCreate(&precond), "HYPRE_BoomerAMGCreate");
        hypre_check(HYPRE_BoomerAMGSetPrintLevel(precond, opt.hyprePrint), "HYPRE_BoomerAMGSetPrintLevel");
        hypre_check(HYPRE_BoomerAMGSetCoarsenType(precond, amgCoarsenType), "HYPRE_BoomerAMGSetCoarsenType");
        hypre_check(HYPRE_BoomerAMGSetInterpType(precond, amgInterpType), "HYPRE_BoomerAMGSetInterpType");
        hypre_check(HYPRE_BoomerAMGSetRelaxType(precond, amgRelaxType), "HYPRE_BoomerAMGSetRelaxType");
        hypre_check(HYPRE_BoomerAMGSetNumSweeps(precond, amgNumSweeps), "HYPRE_BoomerAMGSetNumSweeps");
        hypre_check(HYPRE_BoomerAMGSetTol(precond, 0.0), "HYPRE_BoomerAMGSetTol");
        hypre_check(HYPRE_BoomerAMGSetMaxIter(precond, 1), "HYPRE_BoomerAMGSetMaxIter");
        hypre_check(HYPRE_BoomerAMGSetRelaxOrder(precond, 0), "HYPRE_BoomerAMGSetRelaxOrder");
        hypre_check(HYPRE_BoomerAMGSetPMaxElmts(precond, amgPmax), "HYPRE_BoomerAMGSetPMaxElmts");
        hypre_check(HYPRE_BoomerAMGSetKeepTranspose(precond, amgKeepTranspose), "HYPRE_BoomerAMGSetKeepTranspose");
        hypre_check(HYPRE_BoomerAMGSetTruncFactor(precond, amgTruncFactor), "HYPRE_BoomerAMGSetTruncFactor");
        hypre_check(HYPRE_BoomerAMGSetRAP2(precond, amgRAP2), "HYPRE_BoomerAMGSetRAP2");

        if (amgNumFunctions > 1) {
            hypre_check(HYPRE_BoomerAMGSetNumFunctions(precond, amgNumFunctions),
                        "HYPRE_BoomerAMGSetNumFunctions");
            if (amgNodal > 0) {
                hypre_check(HYPRE_BoomerAMGSetNodal(precond, amgNodal), "HYPRE_BoomerAMGSetNodal");
                hypre_check(HYPRE_BoomerAMGSetNodalLevels(precond, amgNodalLevels), "HYPRE_BoomerAMGSetNodalLevels");
                hypre_check(HYPRE_BoomerAMGSetKeepSameSign(precond, amgKeepSameSign), "HYPRE_BoomerAMGSetKeepSameSign");
                hypre_check(HYPRE_BoomerAMGSetNodalDiag(precond, amgNodalDiag), "HYPRE_BoomerAMGSetNodalDiag");
            }
        }

        if (amgAggLevels > 0) {
            hypre_check(HYPRE_BoomerAMGSetAggNumLevels(precond, amgAggLevels), "HYPRE_BoomerAMGSetAggNumLevels");
            hypre_check(HYPRE_BoomerAMGSetAggInterpType(precond, 4), "HYPRE_BoomerAMGSetAggInterpType");
        }
        if (amgStrongThreshold >= 0.0) {
            hypre_check(HYPRE_BoomerAMGSetStrongThreshold(precond, amgStrongThreshold), "HYPRE_BoomerAMGSetStrongThreshold");
        }
        set_precond((HYPRE_PtrToParSolverFcn)HYPRE_BoomerAMGSolve,
                    (HYPRE_PtrToParSolverFcn)HYPRE_BoomerAMGSetup,
                    precond);
        }
    } else if (pre == "none") {
        // no preconditioner
    } else {
        throw std::runtime_error("Unsupported -precond: " + opt.precond);
    }

    if (usePCG) {
        hypre_check(HYPRE_ParCSRPCGSetup(krylov, parA, parb, parx), "HYPRE_ParCSRPCGSetup");
    } else if (useBiCGSTAB) {
        hypre_check(HYPRE_ParCSRBiCGSTABSetup(krylov, parA, parb, parx), "HYPRE_ParCSRBiCGSTABSetup");
    } else if (useGMRES) {
        hypre_check(HYPRE_ParCSRGMRESSetup(krylov, parA, parb, parx), "HYPRE_ParCSRGMRESSetup");
    } else {
        hypre_check(HYPRE_ParCSRFlexGMRESSetup(krylov, parA, parb, parx), "HYPRE_ParCSRFlexGMRESSetup");
    }

    auto tSetup1 = std::chrono::steady_clock::now();

    if (amgReuseWithinStep &&
        (pre == "amg" || pre == "boomeramg") &&
        rep.amgReuseBuild &&
        precond &&
        ijA) {
        MemoirsHypreAmgReuseCache& c =
            memoirs_hypre_amg_reuse_cache_singleton();

        c.valid = true;
        c.precond = precond;
        c.ijAOwner = ijA;
        c.n = n;
        c.key = amgReuseKey;
        ++c.builds;

        // Ownership moved to the per-step cache.
        precond = nullptr;
        ijA = nullptr;
    }
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

        HYPRE_Int solveStatus = 0;
        if (usePCG) {
            solveStatus = HYPRE_ParCSRPCGSolve(krylov, parA, parb, parx);
        } else if (useBiCGSTAB) {
            solveStatus = HYPRE_ParCSRBiCGSTABSolve(krylov, parA, parb, parx);
        } else if (useGMRES) {
            solveStatus = HYPRE_ParCSRGMRESSolve(krylov, parA, parb, parx);
        } else {
            solveStatus = HYPRE_ParCSRFlexGMRESSolve(krylov, parA, parb, parx);
        }

        rep.hypreSolveStatus = int(solveStatus);
        if (solveStatus != 0 && rank == 0) {
            std::cerr << "WARNING: HYPRE Krylov solve returned status "
                      << int(solveStatus)
                      << "; continuing to query iterations/final residual.\n";
        }
    }

    auto tSolve1 = std::chrono::steady_clock::now();
    rep.hypreSolveOnlySeconds = elapsed(tSolve0, tSolve1);
    rep.hypreSolveOnlyAvgSeconds = rep.hypreSolveOnlySeconds / double(solveRepeats);
    rep.rhsUpdateSeconds = rhsUpdateAccum;
    rep.rhsUpdateAvgSeconds = rhsUpdateAccum / double(solveRepeats);

    HYPRE_Real finalRel = 0;
    if (usePCG) {
        hypre_check(HYPRE_PCGGetNumIterations(krylov, &rep.iterations), "HYPRE_PCGGetNumIterations");
        hypre_check(HYPRE_PCGGetFinalRelativeResidualNorm(krylov, &finalRel), "HYPRE_PCGGetFinalRelativeResidualNorm");
    } else if (useBiCGSTAB) {
        hypre_check(HYPRE_ParCSRBiCGSTABGetNumIterations(krylov, &rep.iterations), "HYPRE_ParCSRBiCGSTABGetNumIterations");
        hypre_check(HYPRE_ParCSRBiCGSTABGetFinalRelativeResidualNorm(krylov, &finalRel), "HYPRE_ParCSRBiCGSTABGetFinalRelativeResidualNorm");
    } else if (useGMRES) {
        hypre_check(HYPRE_ParCSRGMRESGetNumIterations(krylov, &rep.iterations), "HYPRE_ParCSRGMRESGetNumIterations");
        hypre_check(HYPRE_ParCSRGMRESGetFinalRelativeResidualNorm(krylov, &finalRel), "HYPRE_ParCSRGMRESGetFinalRelativeResidualNorm");
    } else {
        hypre_check(HYPRE_ParCSRFlexGMRESGetNumIterations(krylov, &rep.iterations), "HYPRE_ParCSRFlexGMRESGetNumIterations");
        hypre_check(HYPRE_ParCSRFlexGMRESGetFinalRelativeResidualNorm(krylov, &finalRel), "HYPRE_ParCSRFlexGMRESGetFinalRelativeResidualNorm");
    }
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

        if ((int)xOut.size() == (int)m.points.size()) {
            compute_nodal_error(m, mms, xOut, rep);

            ErrorNorms en = compute_error_norms(m, dm, mms, xOut);
            rep.L2Error = en.L2;
            rep.H1SemiError = en.H1Semi;
            rep.nodalErrorL2 = en.nodalL2;
            rep.nodalErrorMax = en.nodalMax;
        } else {
            // Vector/system solve: caller owns physical error norms.
            rep.L2Error = -1.0;
            rep.H1SemiError = -1.0;
            rep.nodalErrorL2 = -1.0;
            rep.nodalErrorMax = -1.0;
        }

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
    if (usePCG) {
        hypre_check(HYPRE_ParCSRPCGDestroy(krylov), "HYPRE_ParCSRPCGDestroy");
    } else if (useBiCGSTAB) {
        hypre_check(HYPRE_ParCSRBiCGSTABDestroy(krylov), "HYPRE_ParCSRBiCGSTABDestroy");
    } else if (useGMRES) {
        hypre_check(HYPRE_ParCSRGMRESDestroy(krylov), "HYPRE_ParCSRGMRESDestroy");
    } else {
        hypre_check(HYPRE_ParCSRFlexGMRESDestroy(krylov), "HYPRE_ParCSRFlexGMRESDestroy");
    }
    hypre_check(HYPRE_IJVectorDestroy(ijb), "HYPRE_IJVectorDestroy b");
    hypre_check(HYPRE_IJVectorDestroy(ijx), "HYPRE_IJVectorDestroy x");
    if (ijA) {
        hypre_check(HYPRE_IJMatrixDestroy(ijA), "HYPRE_IJMatrixDestroy A");
    }

    // Do not HYPRE_Finalize() here. Picard/NSE calls this helper repeatedly.

    (void)startedMPI;

    auto tDestroy1 = std::chrono::steady_clock::now();
    rep.hypreDestroyFinalizeSeconds = elapsed(tDestroy0, tDestroy1);

    return rep;
}

#else

static SolveReport solve_hypre_ij_krylov(
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
              << " numFunctions " << rep.amgNumFunctions
              << " nodal " << rep.amgNodal
              << " nodalLevels " << rep.amgNodalLevels
              << " keepSameSign " << rep.amgKeepSameSign
              << " nodalDiag " << rep.amgNodalDiag
              << "\n";
    std::cout << "ijSettings                = rowSizes " << rep.ijUseRowSizes
              << " bulkInsert " << rep.ijBulkInsert
              << " ijCachedPattern " << (memoirs_env_bool("MEMOIRS_Q1Q1_CACHE_IJ_PATTERN", true) ? 1 : 0)
              << " directIjValues " << (memoirs_env_bool("MEMOIRS_Q1Q1_DIRECT_IJ_VALUES", true) ? 1 : 0)
              << "\n";
    std::cout << "iterations                = " << rep.iterations << "\n";
    std::cout << "finalRelativeResidual     = " << std::setprecision(16) << rep.finalRelRes << "\n";
    std::cout << "hypreSolveStatus          = " << rep.hypreSolveStatus << "\n";
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
