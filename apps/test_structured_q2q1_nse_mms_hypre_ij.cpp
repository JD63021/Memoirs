// Memoirs v0 Q2/Q1 HYPRE-IJ NSE branch scaffold.
//
// Intentional first step of the Q1/Q1 -> Q2/Q1 refactor:
//   - preserve the Q1/Q1 production architecture direction;
//   - do not import the old scratch Q2/Q1 custom-GMRES solver behavior;
//   - establish structured Q2 velocity / Q1 pressure counting and cavity masks.
//
// Physics assembly is deliberately not enabled in this phase. The next patch
// should add the Q1/Q1-style fixed sparse pattern over 27 velocity and 8
// pressure local nodes, then CUDA qpoint assembly.

#include "memoirs/sections/00_common.hpp"
#include "memoirs/sections/01_options.hpp"
#include "memoirs/sections/02_polymesh.hpp"
#include "memoirs/sections/03_cell_topology.hpp"
#include "memoirs/sections/04_reference_elements.hpp"
#include "memoirs/sections/05_dof_map.hpp"  // lower_copy helper only in this scaffold
#include "memoirs/sections/06_sparse_rows.hpp"
#include "memoirs/sections/07_mms.hpp"
#include "memoirs/sections/08_laplacian_assembler.hpp"
#include "memoirs/sections/09_hypre_solver_krylov.hpp"
#include "memoirs/sections/16_q2q1_structured.hpp"
#include "memoirs/sections/17_q2q1_stokes_mms_host.hpp"
#include "memoirs/sections/18_q2q1_nse_cuda_audit.hpp"
#include "memoirs/diagnostics/GpuMemoryMonitor.hpp"

static void q2q1_fill_identity_values_and_rhs(
    const Q2Q1StructuredGrid& g,
    AssembledSystem& sys
) {
    const int nRows = sparse_nrows(sys.A);
    if ((long long)nRows != g.nRows) {
        throw std::runtime_error("q2q1 identity smoke: sparse row count does not match grid row count");
    }

    std::fill(sys.A.flatVals.begin(), sys.A.flatVals.end(), Real(0));
    sys.b.assign((std::size_t)nRows, Real(1));

    long long missingDiag = 0;
    for (int r = 0; r < nRows; ++r) {
        const int slot = sparse_lookup_flat_slot(sys.A, r, r);
        if (slot < 0) {
            ++missingDiag;
            continue;
        }
        sys.A.flatVals[(std::size_t)slot] = Real(1);
    }

    if (missingDiag != 0) {
        std::ostringstream os;
        os << "q2q1 identity smoke: missing diagonal slots = " << missingDiag;
        throw std::runtime_error(os.str());
    }
}

static void q2q1_print_identity_solution_check(const std::vector<Real>& x) {
    if (x.empty()) {
        std::cout << "q2q1HypreIdentitySolutionCheck = skipped_no_copyback\n";
        return;
    }

    double maxAbsErr = 0.0;
    double l2Err = 0.0;

    for (Real xr : x) {
        const double e = std::abs((double)xr - 1.0);
        maxAbsErr = std::max(maxAbsErr, e);
        l2Err += e * e;
    }

    l2Err = std::sqrt(l2Err);

    std::cout << "q2q1HypreIdentitySolutionCheck = 1\n";
    std::cout << "q2q1HypreIdentitySolutionL2Err = " << std::setprecision(16) << l2Err << "\n";
    std::cout << "q2q1HypreIdentitySolutionMaxErr = " << std::setprecision(16) << maxAbsErr << "\n";
}


static void q2q1_probe_first_and_last_cells(const Q2Q1StructuredGrid& g) {
    const auto v0 = q2q1_cell_velocity_nodes(g, 0, 0, 0);
    const auto p0 = q2q1_cell_pressure_nodes(g, 0, 0, 0);
    const auto v1 = q2q1_cell_velocity_nodes(g, g.N - 1, g.N - 1, g.N - 1);
    const auto p1 = q2q1_cell_pressure_nodes(g, g.N - 1, g.N - 1, g.N - 1);

    std::cout << "q2q1CellProbe firstCellVelocityNodes =";
    for (long long v : v0) std::cout << " " << v;
    std::cout << "\n";

    std::cout << "q2q1CellProbe firstCellPressureNodes =";
    for (long long p : p0) std::cout << " " << p;
    std::cout << "\n";

    std::cout << "q2q1CellProbe lastCellVelocityNodes  =";
    for (long long v : v1) std::cout << " " << v;
    std::cout << "\n";

    std::cout << "q2q1CellProbe lastCellPressureNodes  =";
    for (long long p : p1) std::cout << " " << p;
    std::cout << "\n";
}

int main(int argc, char** argv) {
    GpuMemoryMonitor gpuMemMon;
    gpuMemMon.start();

    try {
        Options opt = parse_cli(argc, argv);
        if (opt.polyMeshDir.empty()) {
            throw std::runtime_error("Use -polyMeshDir /path/to/case/constant/polyMesh");
        }

        const bool stageTimers = memoirs_stage_timers_enabled();
        auto tRead0 = std::chrono::steady_clock::now();
        if (stageTimers) std::cerr << "[stage] q2q1 read_polymesh begin\n" << std::flush;

        PolyMesh mesh = read_polymesh(opt.polyMeshDir);

        if (stageTimers) {
            std::cerr << "[stage] q2q1 read_polymesh done seconds=" << seconds_since(tRead0)
                      << " points=" << mesh.points.size()
                      << " faces=" << mesh.faces.size()
                      << " cells=" << mesh.cells.size() << "\n" << std::flush;
        }

        Q2Q1StructuredGrid g = q2q1_infer_structured_unit_cube_grid(mesh);
        q2q1_print_structured_banner(mesh, g);

        if (opt.probeMesh) probe_mesh(mesh);
        q2q1_probe_first_and_last_cells(g);

        const bool basisAudit = memoirs_env_bool("MEMOIRS_Q2Q1_BASIS_AUDIT", false);
        std::cout << "q2q1BasisAudit           = " << (basisAudit ? 1 : 0) << "\n";
        if (basisAudit) {
            Q2Q1BasisAuditReport brep = q2q1_run_basis_audit();
            q2q1_print_basis_audit_report(brep);
        }

        std::string problem = "mms";
        if (const char* e = std::getenv("MEMOIRS_Q2Q1_PROBLEM")) {
            problem = lower_copy(e);
        }

        std::cout << "q2q1ProblemMode           = " << problem << "\n";

        const bool buildPattern = memoirs_env_bool("MEMOIRS_Q2Q1_BUILD_PATTERN", false);
        std::cout << "q2q1BuildPattern          = " << (buildPattern ? 1 : 0) << "\n";

        if (buildPattern) {
            auto tPat0 = std::chrono::steady_clock::now();

            SparseRows A;
            const int pressurePinPNode = 0;
            q2q1_make_nse_picard_fixed_pattern(g, pressurePinPNode, A);

            const double patSeconds = seconds_since(tPat0);
            q2q1_print_pattern_summary(g, A, patSeconds);

            const int ci = (g.N >= 3) ? std::min(g.N - 2, std::max(1, g.N / 2)) : 0;
            const int cj = ci;
            const int ck = ci;
            const long long nSlotLookups =
                q2q1_probe_pattern_cell_slots(g, A, ci, cj, ck, pressurePinPNode);

            std::cout << "q2q1PatternProbeCell      = " << ci << " " << cj << " " << ck << "\n";
            std::cout << "q2q1PatternProbeLookups   = " << nSlotLookups << "\n";

            const bool buildSlotCache = memoirs_env_bool("MEMOIRS_Q2Q1_BUILD_SLOT_CACHE", false);
            std::cout << "q2q1BuildSlotCache        = " << (buildSlotCache ? 1 : 0) << "\n";

            if (buildSlotCache) {
                auto tSlot0 = std::chrono::steady_clock::now();

                Q2Q1CellSlotCache slotCache;
                q2q1_build_cell_slot_cache(g, A, pressurePinPNode, slotCache);

                const double slotSeconds = seconds_since(tSlot0);
                q2q1_print_cell_slot_cache_summary(g, slotCache, slotSeconds);
            }

            const bool assembleStokesMms = memoirs_env_bool("MEMOIRS_Q2Q1_ASSEMBLE_STOKES_MMS", false);
            std::cout << "q2q1AssembleStokesMms     = " << (assembleStokesMms ? 1 : 0) << "\n";

            if (assembleStokesMms) {
                AssembledSystem stokesSys;
                stokesSys.A = A;
                stokesSys.b.assign((std::size_t)g.nRows, Real(0));

                Q2Q1CellSlotCache stokesSlots;
                q2q1_build_cell_slot_cache(g, stokesSys.A, pressurePinPNode, stokesSlots);

                Q2Q1StokesMmsOptions sopt;
                sopt.nu = memoirs_env_double("MEMOIRS_Q2Q1_NU", memoirs_env_double("NU", 1.0));
                sopt.tau = memoirs_env_double("MEMOIRS_Q2Q1_TAU", -1.0);
                sopt.tauScale = memoirs_env_double("MEMOIRS_Q2Q1_TAU_SCALE", 1.0);
                sopt.tauC = memoirs_env_double("MEMOIRS_Q2Q1_TAU_C", 4.0);
                sopt.pressurePinPNode = pressurePinPNode;
                if (const char* rm = std::getenv("MEMOIRS_Q2Q1_STOKES_RHS_MODE")) {
                    sopt.rhsMode = lower_copy(rm);
                }

                const bool stokesCudaAssembly =
                    memoirs_env_bool("MEMOIRS_Q2Q1_STOKES_CUDA_ASSEMBLY", false);
                std::cout << "q2q1StokesCudaAssembly    = " << (stokesCudaAssembly ? 1 : 0) << "\n";

                Q2Q1StokesAssemblyReport srep;

                if (stokesCudaAssembly) {
                    auto tCudaAsm0 = std::chrono::steady_clock::now();

                    srep.rows = g.nRows;
                    srep.nnz = sparse_nnz_flat(stokesSys.A);
                    srep.nu = sopt.nu;
                    srep.rhsMode = sopt.rhsMode;
                    srep.h = std::min(g.hx, std::min(g.hy, g.hz));
                    srep.tau = q2q1_default_stokes_tau(g, sopt);

                    if (lower_copy(sopt.rhsMode) != "continuous") {
                        throw std::runtime_error("Q2/Q1 CUDA Stokes assembly currently supports continuous RHS mode only.");
                    }

                    Q2Q1CudaStokesFillReport crep =
                        q2q1_cuda_fill_stokes_pspg_continuous_rhs(
                            g,
                            stokesSlots,
                            stokesSys.A,
                            stokesSys.b,
                            sopt.nu,
                            srep.tau
                        );

                    q2q1_set_velocity_boundary_identities(g, stokesSys);
                    q2q1_set_pressure_pin_identity(g, sopt.pressurePinPNode, stokesSys);
                    q2q1_set_exact_strong_rhs(g, sopt, stokesSys);

                    srep.assemblySeconds = seconds_since(tCudaAsm0);
                    srep.continuousRhsSeconds = crep.totalSeconds;
                    srep.activeCellSlots = -1;

                    const std::vector<Real> xExact = q2q1_make_exact_vector(g);
                    q2q1_residual_norm_fixed(
                        stokesSys.A,
                        xExact,
                        stokesSys.b,
                        srep.exactResidualL2,
                        srep.exactResidualInf
                    );

                    double x2 = 0.0;
                    for (Real xv : xExact) x2 += (double)xv * (double)xv;
                    srep.exactVectorL2 = std::sqrt(x2);

                    std::cout << "q2q1CudaStokesFillDone    = 1\n";
                    std::cout << "q2q1CudaStokesRows        = " << crep.rows << "\n";
                    std::cout << "q2q1CudaStokesNnz         = " << crep.nnz << "\n";
                    std::cout << "q2q1CudaStokesCells       = " << crep.cells << "\n";
                    std::cout << "q2q1CudaStokesSlotCount   = " << crep.slotCount << "\n";
                    std::cout << "q2q1CudaStokesZeroSeconds = " << crep.zeroSeconds << "\n";
                    std::cout << "q2q1CudaStokesKernelSeconds = " << crep.cellKernelSeconds << "\n";
                    std::cout << "q2q1CudaStokesCopyBackSeconds = " << crep.copyBackSeconds << "\n";
                    std::cout << "q2q1CudaStokesTotalSeconds = " << crep.totalSeconds << "\n";
                } else {
                    srep =
                        q2q1_assemble_stokes_pspg_matrix_algebraic_mms_host(
                            g, stokesSlots, sopt, stokesSys
                        );
                }

                q2q1_print_stokes_assembly_report(srep);

                const bool stokesSolve = memoirs_env_bool("MEMOIRS_Q2Q1_STOKES_SOLVE", false);
                std::cout << "q2q1StokesMmsSolve        = " << (stokesSolve ? 1 : 0) << "\n";

                if (stokesSolve) {
                    auto tSolve0 = std::chrono::steady_clock::now();

                    LinearCgDofMap dmForHypreReport = build_linear_cg_dof_map(mesh, "cg_hex_q1");

                    std::vector<Real> sx;
                    SolveReport ssolve = solve_hypre_ij_krylov(
                        mesh,
                        dmForHypreReport,
                        stokesSys,
                        opt.mms,
                        opt,
                        sx
                    );

                    const double solveTotal = seconds_since(tSolve0);

                    print_solve_report(opt, dmForHypreReport, ssolve);
                    Q2Q1ExactCompareReport erep = q2q1_compare_solution_to_exact(g, sx);
                    q2q1_print_exact_compare_report(erep);

                    Q2Q1IntegratedErrorReport irep = q2q1_compute_integrated_error(g, sx);
                    q2q1_print_integrated_error_report(irep);

                    std::cout << "q2q1StokesMmsSolveDone    = 1\n";
                    std::cout << "q2q1StokesMmsSolveSeconds = " << solveTotal << "\n";
                    std::cout << "q2q1StokesMmsIterations   = " << ssolve.iterations << "\n";
                    std::cout << "q2q1StokesMmsFinalRel     = " << std::setprecision(16) << ssolve.finalRelRes << "\n";
                }
            }

            const bool hypreSmoke = memoirs_env_bool("MEMOIRS_Q2Q1_HYPRE_SMOKE", false);
            std::cout << "q2q1HypreSmoke            = " << (hypreSmoke ? 1 : 0) << "\n";

            if (hypreSmoke) {
                auto tHypre0 = std::chrono::steady_clock::now();

                AssembledSystem sys;
                sys.A = std::move(A);
                sys.nDirichlet = (int)(3LL * q2q1_count_boundary_nodes(g).velocityBoundaryNodes + 1LL);
                const bool cudaIdentityFill = memoirs_env_bool("MEMOIRS_Q2Q1_CUDA_IDENTITY_FILL", false);
                std::cout << "q2q1CudaIdentityFill      = " << (cudaIdentityFill ? 1 : 0) << "\n";

                if (cudaIdentityFill) {
                    Q2Q1CudaIdentityFillReport crep =
                        q2q1_cuda_fill_identity_values_rhs(g, sys.A, sys.b);

                    std::cout << "q2q1CudaIdentityFillDone  = 1\n";
                    std::cout << "q2q1CudaIdentityRows      = " << crep.rows << "\n";
                    std::cout << "q2q1CudaIdentityNnz       = " << crep.nnz << "\n";
                    std::cout << "q2q1CudaIdentityZeroSeconds = " << crep.zeroSeconds << "\n";
                    std::cout << "q2q1CudaIdentityDiagSeconds = " << crep.diagSeconds << "\n";
                    std::cout << "q2q1CudaIdentityCopyBackSeconds = " << crep.copyBackSeconds << "\n";
                    std::cout << "q2q1CudaIdentityTotalSeconds = " << crep.totalSeconds << "\n";
                } else {
                    q2q1_fill_identity_values_and_rhs(g, sys);
                }

                LinearCgDofMap dmForHypreReport = build_linear_cg_dof_map(mesh, "cg_hex_q1");

                std::vector<Real> x;
                SolveReport rep = solve_hypre_ij_krylov(
                    mesh,
                    dmForHypreReport,
                    sys,
                    opt.mms,
                    opt,
                    x
                );

                const double hypreSeconds = seconds_since(tHypre0);

                print_solve_report(opt, dmForHypreReport, rep);
                q2q1_print_identity_solution_check(x);

                std::cout << "q2q1HypreSmokeDone        = 1\n";
                std::cout << "q2q1HypreSmokeSeconds     = " << hypreSeconds << "\n";
                std::cout << "q2q1HypreSmokeRows        = " << g.nRows << "\n";
                std::cout << "q2q1HypreSmokeNnz         = " << sparse_nnz_flat(sys.A) << "\n";
                std::cout << "q2q1HypreSmokeIterations  = " << rep.iterations << "\n";
                std::cout << "q2q1HypreSmokeFinalRel    = " << std::setprecision(16) << rep.finalRelRes << "\n";
            }
        }

        std::cout << "q2q1ScaffoldOnly          = 1\n";
        std::cout << "q2q1NextPatch             = cuda_zero_fill_kernel_then_q2q1_basis_values\n";
        std::cout << "q2q1DesignRule            = emulate_q1q1_hypre_ij_first_do_not_import_old_q2q1_solver\n";

        gpuMemMon.stop();
        gpuMemMon.print(std::cout);
        return 0;
    } catch (const std::exception& e) {
        gpuMemMon.stop();
        gpuMemMon.print(std::cout);
        std::cerr << "FATAL: " << e.what() << "\n";
        return 1;
    }
}
