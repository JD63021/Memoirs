// Memoirs v0 NSE branch skeleton: Q1/Q1 HYPRE IJ/ParCSR path.
// Clean recovery app after broken Patch 2 paste.
// This is the known-good scalar Poisson smoke test using the copied Krylov helper.

#include "memoirs/sections/00_common.hpp"
#include "memoirs/sections/01_options.hpp"
#include "memoirs/sections/02_polymesh.hpp"
#include "memoirs/sections/03_cell_topology.hpp"
#include "memoirs/sections/04_reference_elements.hpp"
#include "memoirs/sections/05_dof_map.hpp"
#include "memoirs/sections/06_sparse_rows.hpp"
#include "memoirs/sections/07_mms.hpp"
#include "memoirs/sections/08_laplacian_assembler.hpp"
#include "memoirs/sections/09_hypre_solver_krylov.hpp"
#include "memoirs/sections/10_output_placeholder.hpp"
#include "memoirs/sections/11_q1q1_nse_utils.hpp"
#include "memoirs/sections/12_q1q1_alg_matrix.hpp"
#include "memoirs/sections/13_q1q1_continuous_stokes_rhs.hpp"
#include "memoirs/sections/14_q1q1_nse_picard.hpp"
#include "memoirs/sections/15_q1q1_nse_cuda_audit.hpp"
#include "memoirs/diagnostics/GpuMemoryMonitor.hpp"


static void print_q1q1_branch_banner(const PolyMesh& mesh, const LinearCgDofMap& dm) {
    const long long scalarNodes = (long long)dm.nDofs;
    std::cout << "--------------- q1q1 nse hypre branch ---------------\n";
    std::cout << "status                    = patch5A_q1q1_transient_picard_nse_pspg_supg_device_solve_ready\n";
    std::cout << "polyMesh points           = " << mesh.points.size() << "\n";
    std::cout << "polyMesh cells            = " << mesh.cells.size() << "\n";
    std::cout << "scalar q1 dofs            = " << scalarNodes << "\n";
    std::cout << "q1q1 rows                 = " << 4LL * scalarNodes << "\n";
    std::cout << "future ordering           = node-major [ux,uy,uz,p]\n";
    std::cout << "";
    std::cout << "requiredAMGEnv            = MEMOIRS_AMG_NUM_FUNCTIONS=4 MEMOIRS_AMG_NODAL=0\n";
}


int main(int argc, char** argv) {
    GpuMemoryMonitor gpuMemMon;
    gpuMemMon.start();

    try {
        Options opt = parse_cli(argc, argv);

        if (opt.polyMeshDir.empty()) {
            throw std::runtime_error("Use -polyMeshDir /path/to/case/constant/polyMesh");
        }

        if (opt.space == "auto") opt.space = "cg_hex_q1";
        if (opt.precond == "diagscale") opt.precond = "boomeramg";

        const bool stageTimers = memoirs_stage_timers_enabled();

        auto tRead0 = std::chrono::steady_clock::now();
        if (stageTimers) std::cerr << "[stage] read_polymesh begin\n" << std::flush;

        PolyMesh mesh = read_polymesh(opt.polyMeshDir);

        if (stageTimers) {
            std::cerr << "[stage] read_polymesh done seconds=" << seconds_since(tRead0)
                      << " points=" << mesh.points.size()
                      << " faces=" << mesh.faces.size()
                      << " cells=" << mesh.cells.size() << "\n" << std::flush;
        }

        LinearCgDofMap dm = build_linear_cg_dof_map(mesh, opt.space);

        print_q1q1_branch_banner(mesh, dm);
        q1q1_probe_exact_vector(mesh, dm);

        if (opt.probeMesh) probe_mesh(mesh);
        if (opt.probeDofs) probe_dofs(mesh, dm);

        auto tAsm0 = std::chrono::steady_clock::now();

        Q1Q1AlgInfo qinfo;
        std::string q1q1RhsMode = "continuous";
        if (const char* e = std::getenv("MEMOIRS_Q1Q1_RHS_MODE")) {
            q1q1RhsMode = lower_copy(e);
        }

        if (q1q1RhsMode == "picard" || q1q1RhsMode == "nse" || q1q1RhsMode == "nse_picard") {
            Q1Q1NsePicardOptions nopt;
            nopt.nu = memoirs_env_double("MEMOIRS_Q1Q1_NU", 1.0);
            nopt.dt = memoirs_env_double("MEMOIRS_Q1Q1_DT", 1.0e-3);
            nopt.advScale = memoirs_env_double("MEMOIRS_Q1Q1_ADV_SCALE", 1.0);

            // Match the working structured Q1/Q1 NSE app by default.
            nopt.tauScale = memoirs_env_double("MEMOIRS_Q1Q1_TAU_SCALE", 1.0);
            nopt.tau = memoirs_env_double("MEMOIRS_Q1Q1_TAU", -1.0);
            nopt.tauC = memoirs_env_double("MEMOIRS_Q1Q1_TAU_C", 4.0);
            nopt.tauCt = memoirs_env_double("MEMOIRS_Q1Q1_TAU_CT", 2.0);
            nopt.tauAdvScale = memoirs_env_double("MEMOIRS_Q1Q1_TAU_ADV_SCALE", 1.0);
            if (const char* e = std::getenv("MEMOIRS_Q1Q1_TAU_MODE")) {
                nopt.tauMode = lower_copy(e);
            } else {
                nopt.tauMode = "metric";
            }

            nopt.supg = memoirs_env_int("MEMOIRS_Q1Q1_SUPG", 1);
            nopt.supgTauScale = memoirs_env_double("MEMOIRS_Q1Q1_SUPG_TAU_SCALE", 1.0);
            nopt.gradDiv = 0; // grad-div/LSIC disabled for this performance branch
            nopt.gradDivScale = 0.0;
            nopt.gradDivCoeff = 0.0;

            nopt.pressurePinNode = memoirs_env_int("MEMOIRS_Q1Q1_PRESSURE_PIN", 0);

            const int maxPicard = memoirs_env_int("MEMOIRS_Q1Q1_MAX_PICARD", 10);
            const double picardTol = memoirs_env_double("MEMOIRS_Q1Q1_PICARD_TOL", 1.0e-7);
            const int diagEvery = memoirs_env_int("MEMOIRS_Q1Q1_DIAG_EVERY", 1);


            std::string q1q1Problem = "mms";
            if (const char* e = std::getenv("MEMOIRS_Q1Q1_PROBLEM")) {
                q1q1Problem = lower_copy(e);
            }

            if (q1q1Problem == "cavity" || q1q1Problem == "driven_cavity") {
                const int nSteps = memoirs_env_int("MEMOIRS_Q1Q1_CAVITY_STEPS", 100);
                const int vtuEvery = memoirs_env_int("MEMOIRS_Q1Q1_CAVITY_VTU_EVERY", 10);

                std::string cavityTimeScheme = "bdf1";
                if (const char* e = std::getenv("MEMOIRS_Q1Q1_CAVITY_TIME_SCHEME")) {
                    cavityTimeScheme = lower_copy(e);
                }
                const bool cavityUseBdf2Requested =
                    (cavityTimeScheme == "bdf2" || cavityTimeScheme == "bdf_2");

                const bool nonlinearResidualCheck =
                    memoirs_env_bool("MEMOIRS_Q1Q1_NONLINEAR_RESIDUAL_CHECK", false);
                const double nonlinearResidualTol =
                    memoirs_env_double("MEMOIRS_Q1Q1_NL_TOL", picardTol);
                const bool hypreAmgReuseWithinStep =
                    memoirs_env_bool("MEMOIRS_HYPRE_AMG_REUSE_WITHIN_STEP", false);

                const double lidUx = memoirs_env_double("MEMOIRS_Q1Q1_LID_UX", 1.0);
                const double lidUy = memoirs_env_double("MEMOIRS_Q1Q1_LID_UY", 0.0);
                const double lidUz = memoirs_env_double("MEMOIRS_Q1Q1_LID_UZ", 0.0);

                std::string vtuDir = "vtu/q1q1_cavity";
                if (const char* e = std::getenv("MEMOIRS_Q1Q1_CAVITY_VTU_DIR")) {
                    vtuDir = e;
                }

                const std::string mkdirCmd = "mkdir -p " + vtuDir;
                const int mkdirStatus = std::system(mkdirCmd.c_str());
                if (mkdirStatus != 0) {
                    throw std::runtime_error("Failed to create VTU directory: " + vtuDir);
                }

                std::vector<Real> oldState = q1q1_make_zero_beta(mesh);
                q1q1_apply_cavity_velocity_bc_to_state(mesh, dm, lidUx, lidUy, lidUz, oldState);

                std::vector<Real> olderState = oldState;

                {
                    std::ostringstream os;
                    os << vtuDir << "/q1q1_cavity_step_" << std::setw(6) << std::setfill('0') << 0 << ".vtu";
                    q1q1_write_vtu_solution(mesh, dm, oldState, os.str());
                    std::cout << "q1q1CavityVtuWritten     = " << os.str() << "\n";
                }

                std::cout << "--------------- q1q1 driven cavity BDF1 --------------\n";
                std::cout << "q1q1ProblemMode          = cavity\n";
                std::cout << "q1q1CavityScheme         = "
                          << (cavityUseBdf2Requested ? "BDF2_after_BDF1_startup" : "BDF1_pseudo_time_picard")
                          << "\n";
                std::cout << "q1q1CavitySteps          = " << nSteps << "\n";
                std::cout << "q1q1CavityDt             = " << std::setprecision(16) << nopt.dt << "\n";
                std::cout << "q1q1CavityNu             = " << std::setprecision(16) << nopt.nu << "\n";
                std::cout << "q1q1CavityLidVelocity    = "
                          << lidUx << " " << lidUy << " " << lidUz << "\n";
                std::cout << "q1q1CavityVtuEvery       = " << vtuEvery << "\n";
                std::cout << "q1q1CavityVtuDir         = " << vtuDir << "\n";
                std::cout << "q1q1CavityPressurePin    = " << nopt.pressurePinNode << "\n";
                std::cout << "q1q1NonlinearResidualCheck = " << (nonlinearResidualCheck ? 1 : 0) << "\n";
                std::cout << "q1q1NonlinearResidualTol = " << std::setprecision(16) << nonlinearResidualTol << "\n";
                std::cout << "q1q1HypreAmgReuseWithinStep = " << (hypreAmgReuseWithinStep ? 1 : 0) << "\n";
                std::cout << "q1q1HypreFvlikeReuseWithinStep = "
                          << (memoirs_env_bool("MEMOIRS_HYPRE_FVLIKE_REUSE_WITHIN_STEP", false) ? 1 : 0)
                          << "\n";
                std::cout << "-----------------------------------------------------\n";

                long long totalKrylov = 0;
                long long nonlinearResidualCallCount = 0;

                for (int step = 1; step <= nSteps; ++step) {
                    if (memoirs_env_bool("MEMOIRS_HYPRE_FVLIKE_REUSE_WITHIN_STEP", false)) {
                        memoirs_hypre_fvlike_reuse_reset();
                    }

                    if (hypreAmgReuseWithinStep) {
                        memoirs_hypre_amg_reuse_reset();
                    }

                    const double time = step * nopt.dt;

                    std::vector<Real> beta = oldState;
                    q1q1_apply_cavity_velocity_bc_to_state(mesh, dm, lidUx, lidUy, lidUz, beta);

                    std::vector<Real> x;
                    SolveReport stepRep;
                    bool stepConverged = false;
                    int stepConvergedIter = maxPicard;

                    for (int pic = 1; pic <= maxPicard; ++pic) {
                        auto tPicAsm0 = std::chrono::steady_clock::now();

                        Q1Q1AlgInfo info;
                        AssembledSystem sysPic;

                        if (memoirs_env_bool("MEMOIRS_Q1Q1_CUDA_ASSEMBLY", false)) {
                            sysPic = make_empty_q1q1_nse_picard_pspg_system(mesh, dm, beta, nopt, info);
                            const bool useBdf2ThisStep =
                                cavityUseBdf2Requested && (step >= 2);

                            q1q1_cuda_assemble_cavity_bdf_history(
                                mesh, dm, beta, oldState, olderState, nopt, info,
                                lidUx, lidUy, lidUz, useBdf2ThisStep, sysPic);
                        } else {
                            throw std::runtime_error(
                                "Driven cavity currently requires MEMOIRS_Q1Q1_CUDA_ASSEMBLY=1 "
                                "so it does not fall back to the slow host RHS path."
                            );
                        }

                        auto tPicAsm1 = std::chrono::steady_clock::now();

                        if (memoirs_env_bool("MEMOIRS_Q1Q1_VERBOSE_REPORTS", true)) {
                            q1q1_print_nse_picard_info(nopt, info, pic);
                        }
                        std::cout << "q1q1CavityRhsOverride    = CUDA_BDF1_old_velocity_plus_lid_dirichlet\n";
                        std::cout << "q1q1CavityStep           = " << step << "\n";
                        std::cout << "q1q1CavityTime           = " << std::setprecision(16) << time << "\n";

                        auto tPicSol0 = std::chrono::steady_clock::now();
                        SolveReport rep = solve_hypre_ij_krylov(mesh, dm, sysPic, opt.mms, opt, x);
                        auto tPicSol1 = std::chrono::steady_clock::now();

                        rep.assemblySeconds = std::chrono::duration<double>(tPicAsm1 - tPicAsm0).count();
                        rep.solveSeconds = std::chrono::duration<double>(tPicSol1 - tPicSol0).count();

                        const double relUpdate = q1q1_velocity_relative_update(beta, x);

                        double nlAbsL2 = -1.0;
                        double nlRelL2 = -1.0;
                        double nlRhsL2 = -1.0;
                        double nlAxL2 = -1.0;
                        double nlSeconds = 0.0;

                        if (nonlinearResidualCheck) {
                            const bool useBdf2ThisStep =
                                cavityUseBdf2Requested && (step >= 2);

                            std::vector<Real> xForResidual = x;
                            q1q1_apply_cavity_velocity_bc_to_state(
                                mesh, dm, lidUx, lidUy, lidUz, xForResidual);

                            ++nonlinearResidualCallCount;
                            const Q1Q1NonlinearResidualReport nlRep =
                                q1q1_cuda_cavity_nonlinear_residual(
                                    mesh, dm, xForResidual,
                                    oldState, olderState,
                                    nopt, info,
                                    lidUx, lidUy, lidUz,
                                    useBdf2ThisStep,
                                    sysPic);

                            nlAbsL2 = nlRep.absL2;
                            nlRelL2 = nlRep.relL2;
                            nlRhsL2 = nlRep.rhsL2;
                            nlAxL2 = nlRep.axL2;
                            nlSeconds = nlRep.seconds;
                        }

                        if (memoirs_env_bool("MEMOIRS_Q1Q1_VERBOSE_REPORTS", true)) {
                            print_solve_report(opt, dm, rep);
                        }

                        std::cout << "q1q1PicardSummary         = step " << step
                                  << " time " << std::setprecision(16) << time
                                  << " iter " << pic
                                  << " relUpdate " << relUpdate
                                  << " nonlinearResidualAbsL2 " << nlAbsL2
                                  << " nonlinearResidualRelL2 " << nlRelL2
                                  << " nonlinearResidualRhsL2 " << nlRhsL2
                                  << " nonlinearResidualAxL2 " << nlAxL2
                                  << " nonlinearResidualSeconds " << nlSeconds
                                  << " UFieldL2 " << -1.0
                                  << " PMeanShiftedL2 " << -1.0
                                  << " linIters " << rep.iterations
                                  << " linRel " << rep.finalRelRes
                                  << " hypreMatrixSeconds " << rep.hypreMatrixInsertSeconds
                                  << " hypreSetupSeconds " << rep.hypreSetupSeconds
                                  << " hypreSolveSeconds " << rep.hypreSolveOnlySeconds
                                  << " hypreGetSolutionSeconds " << rep.hypreGetSolutionSeconds
                                  << " amgReuseBuild " << rep.amgReuseBuild
                                  << " amgReuseHit " << rep.amgReuseHit
                                  << " fvReuseBuild " << rep.hypreFvReuseBuild
                                  << " fvReuseHit " << rep.hypreFvReuseHit
                                  << " fvReuseSetup " << rep.hypreFvReuseSetup
                                  << "\n";

                        totalKrylov += rep.iterations;
                        stepRep = rep;

                        q1q1_copy_velocity_to_beta(x, beta);
                        q1q1_apply_cavity_velocity_bc_to_state(mesh, dm, lidUx, lidUy, lidUz, beta);

                        const bool nonlinearStop =
                            nonlinearResidualCheck && (nlRelL2 >= 0.0) &&
                            (nlRelL2 < nonlinearResidualTol);

                        const bool updateStop =
                            (!nonlinearResidualCheck) && (relUpdate < picardTol);

                        if (nonlinearStop || updateStop) {
                            stepConverged = true;
                            stepConvergedIter = pic;
                            break;
                        }
                    }

                    if (x.empty()) {
                        throw std::runtime_error("Cavity step produced empty solution.");
                    }

                    olderState = oldState;
                    oldState = x;
                    q1q1_apply_cavity_velocity_bc_to_state(mesh, dm, lidUx, lidUy, lidUz, oldState);

                    std::cout << "q1q1CavityStepSummary    = step " << step
                              << " time " << std::setprecision(16) << time
                              << " converged " << (stepConverged ? 1 : 0)
                              << " convergedIter " << stepConvergedIter
                              << " lastLinIters " << stepRep.iterations
                              << " totalKrylov " << totalKrylov
                              << "\n";

                    if (vtuEvery > 0 && ((step % vtuEvery) == 0 || step == nSteps)) {
                        std::ostringstream os;
                        os << vtuDir << "/q1q1_cavity_step_"
                           << std::setw(6) << std::setfill('0') << step << ".vtu";
                        q1q1_write_vtu_solution(mesh, dm, oldState, os.str());
                        std::cout << "q1q1CavityVtuWritten     = " << os.str() << "\n";
                    }
                }

                if (memoirs_env_bool("MEMOIRS_HYPRE_FVLIKE_REUSE_WITHIN_STEP", false)) {
                    memoirs_hypre_fvlike_reuse_reset();
                }

                std::cout << "TOTAL_KRYLOV_ITERATIONS = " << totalKrylov << "\n";
                std::cout << "q1q1NonlinearResidualCallCount = " << nonlinearResidualCallCount << "\n";
                std::cout << "q1q1CavityCompleted      = 1\n";
                std::cout << "branchNote               = Q1/Q1 driven cavity BDF1 pseudo-time Picard, lid top z=max\n";

                gpuMemMon.stop();
                gpuMemMon.print(std::cout);
                return 0;
            }

            std::string betaInitial = "zero";
            if (const char* e = std::getenv("MEMOIRS_Q1Q1_BETA_INITIAL")) {
                betaInitial = lower_copy(e);
            }

            std::vector<Real> beta;
            if (betaInitial == "exact") {
                beta = q1q1_make_exact_beta(mesh);
            } else {
                beta = q1q1_make_zero_beta(mesh);
            }

            std::vector<Real> x;
            SolveReport lastRep;
            Q1Q1AlgInfo lastInfo;

            std::cout << "--------------- q1q1 NSE Picard loop ----------------\n";
            std::cout << "q1q1PicardMode            = transient_stationary_mms\n";
            std::cout << "q1q1PicardMax             = " << maxPicard << "\n";
            std::cout << "q1q1PicardTol             = " << std::setprecision(16) << picardTol << "\n";
            std::cout << "q1q1DiagEvery             = " << diagEvery << "\n";
            std::cout << "q1q1PicardCopybackNeeded  = 1  # host beta still required until device assembly lands\n";
            std::cout << "q1q1BetaInitial           = " << betaInitial << "\n";
            std::cout << "q1q1Supg                  = " << nopt.supg << "\n";
            std::cout << "-----------------------------------------------------\n";

            for (int pic = 1; pic <= maxPicard; ++pic) {
                auto tPicAsm0 = std::chrono::steady_clock::now();

                Q1Q1AlgInfo info;
                AssembledSystem sysPic;

                if (memoirs_env_bool("MEMOIRS_Q1Q1_CUDA_ASSEMBLY", false)) {
                    sysPic = make_empty_q1q1_nse_picard_pspg_system(mesh, dm, beta, nopt, info);
                    q1q1_cuda_audit_nse_assembly(mesh, dm, beta, nopt, info, sysPic);
                } else {
                    sysPic = assemble_q1q1_nse_picard_pspg(mesh, dm, beta, nopt, info);

                    if (memoirs_env_bool("MEMOIRS_Q1Q1_CUDA_AUDIT_ASSEMBLY", false)) {
                        q1q1_cuda_audit_nse_assembly(mesh, dm, beta, nopt, info, sysPic);
                    }
                }

                auto tPicAsm1 = std::chrono::steady_clock::now();

                if (memoirs_env_bool("MEMOIRS_Q1Q1_VERBOSE_REPORTS", true)) {
                            q1q1_print_nse_picard_info(nopt, info, pic);
                        }

                auto tPicSol0 = std::chrono::steady_clock::now();
                SolveReport rep = solve_hypre_ij_krylov(mesh, dm, sysPic, opt.mms, opt, x);
                auto tPicSol1 = std::chrono::steady_clock::now();

                rep.assemblySeconds = std::chrono::duration<double>(tPicAsm1 - tPicAsm0).count();
                rep.solveSeconds = std::chrono::duration<double>(tPicSol1 - tPicSol0).count();

                if (x.size() != beta.size()) {
                    std::cout << "q1q1PicardNoCopybackError = 1\n";
                    std::cout << "q1q1PicardNoCopybackReason= host Picard assembler still needs x copied back to update beta; keep MEMOIRS_COMPUTE_ERROR=1 until device assembly/device beta is implemented\n";
                    throw std::runtime_error("Q1/Q1 Picard currently requires host solution copyback for beta update. Use COMPUTE_ERROR=1 for now.");
                }

                const double relUpdate = q1q1_velocity_relative_update(beta, x);

                if (memoirs_env_bool("MEMOIRS_Q1Q1_VERBOSE_REPORTS", true)) {
                            print_solve_report(opt, dm, rep);
                        }

                const bool doDiag =
                    (diagEvery > 0 && ((pic % diagEvery) == 0)) ||
                    (relUpdate < picardTol) ||
                    (pic == maxPicard);

                double summaryUFieldL2 = -1.0;
                double summaryPMeanShiftedL2 = -1.0;

                if (doDiag) {
                    Q1Q1AlgError ne = q1q1_compute_alg_error(mesh, x);
                    q1q1_print_alg_error(ne);

                    Q1Q1FieldL2Error fe = q1q1_compute_field_l2_error(mesh, x);
                    q1q1_print_field_l2_error(fe);

                    summaryUFieldL2 = fe.uL2;
                    summaryPMeanShiftedL2 = fe.pMeanShiftedL2;
                } else {
                    std::cout << "q1q1DiagnosticsSkipped    = 1\n";
                }

                std::cout << "q1q1PicardSummary         = iter " << pic
                          << " relUpdate " << std::setprecision(16) << relUpdate
                          << " UFieldL2 " << summaryUFieldL2
                          << " PMeanShiftedL2 " << summaryPMeanShiftedL2
                          << " linIters " << rep.iterations
                          << " linRel " << rep.finalRelRes
                          << "\n";

                q1q1_copy_velocity_to_beta(x, beta);

                lastRep = rep;
                lastInfo = info;

                if (relUpdate < picardTol) {
                    std::cout << "q1q1PicardConverged       = 1\n";
                    std::cout << "q1q1PicardConvergedIter   = " << pic << "\n";
                    break;
                }

                if (pic == maxPicard) {
                    std::cout << "q1q1PicardConverged       = 0\n";
                    std::cout << "q1q1PicardConvergedIter   = " << pic << "\n";
                }
            }

            std::cout << "branchNote                = Patch4B Q1/Q1 transient Picard NSE PSPG + optional SUPG + optional grad-div/LSIC assembled IJ/ParCSR\n";

            gpuMemMon.stop();
            gpuMemMon.print(std::cout);
            return 0;
        }

        AssembledSystem sys;
        if (q1q1RhsMode == "algebraic") {
            sys = assemble_q1q1_stokes_pspg_algebraic(mesh, dm, qinfo);
        } else if (q1q1RhsMode == "continuous" || q1q1RhsMode == "stokes") {
            sys = assemble_q1q1_stokes_pspg_continuous(mesh, dm, qinfo);
        } else {
            throw std::runtime_error("Unsupported MEMOIRS_Q1Q1_RHS_MODE. Use continuous or algebraic.");
        }

        auto tAsm1 = std::chrono::steady_clock::now();

        q1q1_print_alg_info(qinfo);
        std::cout << "q1q1SelectedRhsMode       = " << q1q1RhsMode << "\n";

        std::vector<Real> x;

        auto tSol0 = std::chrono::steady_clock::now();
        SolveReport rep = solve_hypre_ij_krylov(mesh, dm, sys, opt.mms, opt, x);
        auto tSol1 = std::chrono::steady_clock::now();

        rep.assemblySeconds = std::chrono::duration<double>(tAsm1 - tAsm0).count();
        rep.solveSeconds = std::chrono::duration<double>(tSol1 - tSol0).count();

        if (memoirs_env_bool("MEMOIRS_Q1Q1_VERBOSE_REPORTS", true)) {
                            print_solve_report(opt, dm, rep);
                        }

        if ((int)x.size() == 4 * dm.nDofs) {
            Q1Q1AlgError qe = q1q1_compute_alg_error(mesh, x);
            q1q1_print_alg_error(qe);

            Q1Q1FieldL2Error fe = q1q1_compute_field_l2_error(mesh, x);
            q1q1_print_field_l2_error(fe);
        } else {
            std::cout << "q1q1AlgError              = skipped; set MEMOIRS_COMPUTE_ERROR=1 for solution copyback\n";
        }

        std::cout << "branchNote                = Patch3A real 4-field Q1/Q1 steady Stokes/PSPG continuous MMS RHS\n";

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
