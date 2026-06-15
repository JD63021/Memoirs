#include "memoirs/sections/00_common.hpp"
#include "memoirs/sections/01_options.hpp"
#include "memoirs/sections/02_polymesh.hpp"
#include "memoirs/sections/03_cell_topology.hpp"
#include "memoirs/sections/03a_topology_tables.hpp"
#include "memoirs/sections/04_reference_elements.hpp"
#include "memoirs/sections/04a_quadrature.hpp"
#include "memoirs/sections/04b_basis.hpp"
#include "memoirs/sections/05_dof_map.hpp"
#include "memoirs/sections/06_sparse_rows.hpp"
#include "memoirs/sections/07_mms.hpp"
#include "memoirs/sections/08a_boundary_conditions.hpp"
#include "memoirs/sections/08_laplacian_assembler.hpp"
#include "memoirs/sections/09_hypre_solver.hpp"
#include "memoirs/sections/09a_hypre_raw_solver.hpp"
#include "memoirs/sections/10_scalar_elliptic_spec.hpp"
#include "memoirs/sections/11_dg_tet_p1_sipg.hpp"
#include "memoirs/sections/11b_dg_tet_p2_sipg.hpp"
#include "memoirs/sections/11c_dg_tet_p2_modal_sipg.hpp"
#include "memoirs/diagnostics/GpuMemoryMonitor.hpp"

static double parse_sigma(int argc, char** argv, double defval) {
    double sigma = defval;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];

        if (a == "-penaltySigma" || a == "-sipgSigma" || a == "-sigma") {
            if (i + 1 >= argc) throw std::runtime_error("Missing value after " + a);
            sigma = std::stod(argv[++i]);
        }
    }

    return sigma;
}

int main(int argc, char** argv) {
    GpuMemoryMonitor gpuMemMon;

    try {
        Options opt = parse_cli(argc, argv);
        ScalarEllipticSpec scalarSpec = parse_scalar_elliptic_spec_from_cli(argc, argv, "sipg", 20.0);
        MemoirsQuadratureSpec quadSpec = parse_memoirs_quadrature_spec_from_cli(argc, argv);
        (void)quadSpec;

        DgSipgOptions dgopt;
        dgopt.penaltySigma = parse_sigma(
            argc,
            argv,
            memoirs_env_double("MEMOIRS_DG_PENALTY_SIGMA", 20.0)
        );

        gpuMemMon.start();

        std::cout << "Memoirs v0 :: DG tet P1/P2 SIPG Poisson MMS\n";
        std::cout << "precision         = " << kPrecisionName << "\n";
        std::cout << "polyMeshDir       = " << opt.polyMeshDir << "\n";
        std::cout << "space             = dg_tet_p1\n";
        std::cout << "mms               = " << opt.mms << "\n";
        std::cout << "penaltySigma      = " << dgopt.penaltySigma << "\n";
        std::cout << "scalarBcMode      = " << scalar_bc_mode_name(scalarSpec.boundary.mode) << "\n";
        std::cout << "sparseMode        = " << memoirs_sparse_mode() << "\n";
        print_memoirs_quadrature_orders(std::cout);

        if (opt.polyMeshDir.empty()) {
            throw std::runtime_error("Missing -polyMeshDir");
        }

        PolyMesh mesh = read_polymesh(opt.polyMeshDir);

        if (opt.probeMesh) {
            probe_mesh(mesh);
            probe_memoirs_topology_tables(mesh);
        }



        const bool useDgTetP2Modal = (
            opt.space == "dg_tet_p2_modal" ||
            opt.space == "dg_p2_modal" ||
            opt.space == "dg_tet_quadratic_modal"
        );

        if (useDgTetP2Modal) {
            DgTetP2ModalDofMap dm = build_dg_tet_p2_modal_dof_map(mesh);

            std::cout << "Memoirs v0 :: DG tet P2 modal SIPG Poisson MMS\n";
            std::cout << "precision         = " << kPrecisionName << "\n";
            std::cout << "polyMeshDir       = " << opt.polyMeshDir << "\n";
            std::cout << "space             = " << dm.resolvedSpace << "\n";
            std::cout << "mms               = " << opt.mms << "\n";
            std::cout << "penaltySigma      = " << dgopt.penaltySigma << "\n";
            std::cout << "scalarBcMode      = " << scalar_bc_mode_name(scalarSpec.boundary.mode) << "\n";
            std::cout << "sparseMode        = " << memoirs_sparse_mode() << "\n";
            print_memoirs_quadrature_orders(std::cout);

            if (opt.probeDofs) {
                std::cout << "---------------- DG modal dof probe -----------------\n";
                std::cout << "space             = " << dm.resolvedSpace << "\n";
                std::cout << "continuity        = DG\n";
                std::cout << "basis             = cell-local modal P2 tet\n";
                std::cout << "nCells            = " << dm.nCells << "\n";
                std::cout << "nDofs             = " << dm.nDofs << "\n";
                std::cout << "dofsPerCell       = 10\n";
                std::cout << "note              = coefficient fields are modal, not nodal values\n";
                std::cout << "-----------------------------------------------------\n";
            }

            if (opt.assemble || opt.solve) {
                if (scalarSpec.boundary.mode == ScalarBoundaryMode::Strong) {
                    throw std::runtime_error("dg_tet_p2_modal currently supports weak SIPG/Nitsche BCs only; use -bc sipg.");
                }
                if (!(scalarSpec.boundary.mode == ScalarBoundaryMode::WeakSipg ||
                      scalarSpec.boundary.mode == ScalarBoundaryMode::WeakNitsche)) {
                    throw std::runtime_error("DG tet P2 modal supports -bc sipg/nitsche.");
                }

                auto tAsm0 = std::chrono::steady_clock::now();
                AssembledSystem sys = assemble_dg_tet_p2_modal_sipg_poisson_mms(mesh, dm, opt.mms, dgopt, scalarSpec);
                auto tAsm1 = std::chrono::steady_clock::now();
                double assemblySeconds = std::chrono::duration<double>(tAsm1 - tAsm0).count();

                if (opt.diagLevel >= 1 || opt.assemble) {
                    probe_dg_tet_p2_modal_system(mesh, dm, sys, opt.diagLevel);
                }

                if (opt.solve) {
#if defined(MEMOIRS_USE_HYPRE)
                    std::vector<Real> x;
                    auto tSol0 = std::chrono::steady_clock::now();
                    SolveReport rep = solve_hypre_ij_pcg_raw(sys.A, sys.b, opt, x);
                    auto tErr0 = std::chrono::steady_clock::now();
                    ErrorNorms en = compute_dg_tet_p2_modal_error_norms(mesh, dm, opt.mms, x);
                    auto tErr1 = std::chrono::steady_clock::now();
                    auto tSol1 = std::chrono::steady_clock::now();

                    rep.assemblySeconds = assemblySeconds;
                    rep.solveSeconds = std::chrono::duration<double>(tSol1 - tSol0).count();
                    rep.errorNormSeconds = std::chrono::duration<double>(tErr1 - tErr0).count();
                    rep.nodalErrorL2 = en.nodalL2;
                    rep.nodalErrorMax = en.nodalMax;
                    rep.L2Error = en.L2;
                    rep.H1SemiError = en.H1Semi;

                    std::cout << "---------------- DG solve report -----------------\n";
                    std::cout << "space                  = " << dm.resolvedSpace << "\n";
                    std::cout << "nDofs                  = " << dm.nDofs << "\n";
                    std::cout << "penaltySigma           = " << dgopt.penaltySigma << "\n";
                    std::cout << "solver                 = " << opt.solver << "\n";
                    std::cout << "precond                = " << opt.precond << "\n";
                    std::cout << "iterations             = " << rep.iterations << "\n";
                    std::cout << "finalRelativeResidual  = " << std::scientific << rep.finalRelRes << std::defaultfloat << "\n";
                    std::cout << "coeffL2                = " << std::scientific << rep.nodalErrorL2 << "\n";
                    std::cout << "coeffMax               = " << rep.nodalErrorMax << "\n";
                    std::cout << "L2Error                = " << rep.L2Error << "\n";
                    std::cout << "H1SemiError            = " << rep.H1SemiError << std::defaultfloat << "\n";
                    std::cout << "assemblySeconds        = " << rep.assemblySeconds << "\n";
                    std::cout << "solveSeconds           = " << rep.solveSeconds << "\n";
                    std::cout << "hypreMatrixInsertSec   = " << rep.hypreMatrixInsertSeconds << "\n";
                    std::cout << "hypreMatrixMigrateSec  = " << rep.hypreMatrixMigrateSeconds << "\n";
                    std::cout << "hypreVectorInsertSec   = " << rep.hypreVectorInsertSeconds << "\n";
                    std::cout << "hypreVectorMigrateSec  = " << rep.hypreVectorMigrateSeconds << "\n";
                    std::cout << "hypreSetupSeconds      = " << rep.hypreSetupSeconds << "\n";
                    std::cout << "hypreSolveOnlySeconds  = " << rep.hypreSolveOnlySeconds << "\n";
                    std::cout << "hypreGetSolutionSec    = " << rep.hypreGetSolutionSeconds << "\n";
                    std::cout << "errorNormSeconds       = " << rep.errorNormSeconds << "\n";
                    std::cout << "hypreDestroyFinalSec   = " << rep.hypreDestroyFinalizeSeconds << "\n";
                    std::cout << "--------------------------------------------------\n";
#else
                    throw std::runtime_error("Built without MEMOIRS_USE_HYPRE=ON; cannot solve.");
#endif
                }
            }

            gpuMemMon.stop();
            gpuMemMon.print(std::cout);
            return 0;
        }

        const bool useDgTetP2 = (
            opt.space == "dg_tet_p2" ||
            opt.space == "dg_p2" ||
            opt.space == "dg_tet_quadratic"
        );

        if (useDgTetP2) {
            DgTetP2DofMap dm = build_dg_tet_p2_dof_map(mesh);

            std::cout << "Memoirs v0 :: DG tet P2 SIPG Poisson MMS\n";
            std::cout << "precision         = " << kPrecisionName << "\n";
            std::cout << "polyMeshDir       = " << opt.polyMeshDir << "\n";
            std::cout << "space             = " << dm.resolvedSpace << "\n";
            std::cout << "mms               = " << opt.mms << "\n";
            std::cout << "penaltySigma      = " << dgopt.penaltySigma << "\n";
            std::cout << "scalarBcMode      = " << scalar_bc_mode_name(scalarSpec.boundary.mode) << "\n";
            std::cout << "sparseMode        = " << memoirs_sparse_mode() << "\n";
            print_memoirs_quadrature_orders(std::cout);

            if (opt.probeDofs) {
                std::cout << "---------------- DG dof probe -----------------\n";
                std::cout << "space             = " << dm.resolvedSpace << "\n";
                std::cout << "continuity        = DG\n";
                std::cout << "basis             = cell-local nodal P2 tet\n";
                std::cout << "nCells            = " << dm.nCells << "\n";
                std::cout << "nDofs             = " << dm.nDofs << "\n";
                std::cout << "dofsPerCell       = 10\n";
                std::cout << "-----------------------------------------------\n";
            }

            if (opt.assemble || opt.solve) {
                auto tAsm0 = std::chrono::steady_clock::now();

                AssembledSystem sys;
                if (scalarSpec.boundary.mode == ScalarBoundaryMode::Strong) {
                    sys = assemble_dg_tet_p2_strong_poisson_mms(mesh, dm, opt.mms, dgopt, scalarSpec);
                } else if (scalarSpec.boundary.mode == ScalarBoundaryMode::WeakSipg ||
                           scalarSpec.boundary.mode == ScalarBoundaryMode::WeakNitsche) {
                    sys = assemble_dg_tet_p2_sipg_poisson_mms(mesh, dm, opt.mms, dgopt, scalarSpec);
                } else {
                    throw std::runtime_error("DG tet P2 supports -bc strong or -bc sipg.");
                }

                auto tAsm1 = std::chrono::steady_clock::now();
                double assemblySeconds = std::chrono::duration<double>(tAsm1 - tAsm0).count();

                if (opt.diagLevel >= 1 || opt.assemble) {
                    probe_dg_tet_p2_system(mesh, dm, sys, opt.diagLevel);
                }

                if (opt.solve) {
#if defined(MEMOIRS_USE_HYPRE)
                    std::vector<Real> x;
                    auto tSol0 = std::chrono::steady_clock::now();
                    SolveReport rep = solve_hypre_ij_pcg_raw(sys.A, sys.b, opt, x);
                    auto tErr0 = std::chrono::steady_clock::now();
                    ErrorNorms en = compute_dg_tet_p2_error_norms(mesh, dm, opt.mms, x);
                    auto tErr1 = std::chrono::steady_clock::now();
                    auto tSol1 = std::chrono::steady_clock::now();

                    rep.assemblySeconds = assemblySeconds;
                    rep.solveSeconds = std::chrono::duration<double>(tSol1 - tSol0).count();
                    rep.errorNormSeconds = std::chrono::duration<double>(tErr1 - tErr0).count();
                    rep.nodalErrorL2 = en.nodalL2;
                    rep.nodalErrorMax = en.nodalMax;
                    rep.L2Error = en.L2;
                    rep.H1SemiError = en.H1Semi;

                    std::cout << "---------------- DG solve report -----------------\n";
                    std::cout << "space                  = " << dm.resolvedSpace << "\n";
                    std::cout << "nDofs                  = " << dm.nDofs << "\n";
                    std::cout << "penaltySigma           = " << dgopt.penaltySigma << "\n";
                    std::cout << "solver                 = " << opt.solver << "\n";
                    std::cout << "precond                = " << opt.precond << "\n";
                    std::cout << "iterations             = " << rep.iterations << "\n";
                    std::cout << "finalRelativeResidual  = " << std::scientific << rep.finalRelRes << std::defaultfloat << "\n";
                    std::cout << "nodalErrorL2           = " << std::scientific << rep.nodalErrorL2 << "\n";
                    std::cout << "nodalErrorMax          = " << rep.nodalErrorMax << "\n";
                    std::cout << "L2Error                = " << rep.L2Error << "\n";
                    std::cout << "H1SemiError            = " << rep.H1SemiError << std::defaultfloat << "\n";
                    std::cout << "assemblySeconds        = " << rep.assemblySeconds << "\n";
                    std::cout << "solveSeconds           = " << rep.solveSeconds << "\n";
                    std::cout << "hypreMatrixInsertSec   = " << rep.hypreMatrixInsertSeconds << "\n";
                    std::cout << "hypreMatrixMigrateSec  = " << rep.hypreMatrixMigrateSeconds << "\n";
                    std::cout << "hypreVectorInsertSec   = " << rep.hypreVectorInsertSeconds << "\n";
                    std::cout << "hypreVectorMigrateSec  = " << rep.hypreVectorMigrateSeconds << "\n";
                    std::cout << "hypreSetupSeconds      = " << rep.hypreSetupSeconds << "\n";
                    std::cout << "hypreSolveOnlySeconds  = " << rep.hypreSolveOnlySeconds << "\n";
                    std::cout << "hypreGetSolutionSec    = " << rep.hypreGetSolutionSeconds << "\n";
                    std::cout << "errorNormSeconds       = " << rep.errorNormSeconds << "\n";
                    std::cout << "hypreDestroyFinalSec   = " << rep.hypreDestroyFinalizeSeconds << "\n";
                    std::cout << "--------------------------------------------------\n";
#else
                    throw std::runtime_error("Built without MEMOIRS_USE_HYPRE=ON; cannot solve.");
#endif
                }
            }

            gpuMemMon.stop();
            gpuMemMon.print(std::cout);
            return 0;
        }

        DgTetP1DofMap dm = build_dg_tet_p1_dof_map(mesh);

        if (opt.probeDofs) {
            std::cout << "---------------- DG dof probe -----------------\n";
            std::cout << "space             = " << dm.resolvedSpace << "\n";
            std::cout << "continuity        = DG\n";
            std::cout << "basis             = cell-local nodal P1 tet\n";
            std::cout << "nCells            = " << dm.nCells << "\n";
            std::cout << "nDofs             = " << dm.nDofs << "\n";
            std::cout << "dofsPerCell       = 4\n";
            std::cout << "-----------------------------------------------\n";
        }

        if (opt.assemble || opt.solve) {
            auto tAsm0 = std::chrono::steady_clock::now();

            AssembledSystem sys;
            if (scalarSpec.boundary.mode == ScalarBoundaryMode::Strong) {
                sys = assemble_dg_tet_p1_strong_poisson_mms(mesh, dm, opt.mms, dgopt);
            } else if (scalarSpec.boundary.mode == ScalarBoundaryMode::WeakSipg ||
                       scalarSpec.boundary.mode == ScalarBoundaryMode::WeakNitsche) {
                sys = assemble_dg_tet_p1_sipg_poisson_mms(mesh, dm, opt.mms, dgopt);
            } else {
                throw std::runtime_error("DG app supports -bc strong or -bc sipg.");
            }

            auto tAsm1 = std::chrono::steady_clock::now();
            double assemblySeconds = std::chrono::duration<double>(tAsm1 - tAsm0).count();

            if (opt.diagLevel >= 1 || opt.assemble) {
                probe_dg_tet_p1_system(mesh, dm, sys, opt.diagLevel);
            }

            if (opt.solve) {
#if defined(MEMOIRS_USE_HYPRE)
                std::vector<Real> x;

                auto tSol0 = std::chrono::steady_clock::now();

                SolveReport rep = solve_hypre_ij_pcg_raw(
                    sys.A,
                    sys.b,
                    opt,
                    x
                );

                auto tErr0 = std::chrono::steady_clock::now();
                ErrorNorms en = compute_dg_tet_p1_error_norms(mesh, dm, opt.mms, x);
                auto tErr1 = std::chrono::steady_clock::now();

                auto tSol1 = std::chrono::steady_clock::now();

                rep.assemblySeconds = assemblySeconds;
                rep.solveSeconds = std::chrono::duration<double>(tSol1 - tSol0).count();
                rep.errorNormSeconds = std::chrono::duration<double>(tErr1 - tErr0).count();
                rep.nodalErrorL2 = en.nodalL2;
                rep.nodalErrorMax = en.nodalMax;
                rep.L2Error = en.L2;
                rep.H1SemiError = en.H1Semi;

                std::cout << "---------------- DG solve report -----------------\n";
                std::cout << "space                  = " << dm.resolvedSpace << "\n";
                std::cout << "nDofs                  = " << dm.nDofs << "\n";
                std::cout << "penaltySigma           = " << dgopt.penaltySigma << "\n";
                std::cout << "solver                 = " << opt.solver << "\n";
                std::cout << "precond                = " << opt.precond << "\n";
                std::cout << "iterations             = " << rep.iterations << "\n";
                std::cout << "finalRelativeResidual  = " << std::scientific << rep.finalRelRes << std::defaultfloat << "\n";
                std::cout << "nodalErrorL2           = " << std::scientific << rep.nodalErrorL2 << "\n";
                std::cout << "nodalErrorMax          = " << rep.nodalErrorMax << "\n";
                std::cout << "L2Error                = " << rep.L2Error << "\n";
                std::cout << "H1SemiError            = " << rep.H1SemiError << std::defaultfloat << "\n";
                std::cout << "assemblySeconds        = " << rep.assemblySeconds << "\n";
                std::cout << "solveSeconds           = " << rep.solveSeconds << "\n";
                std::cout << "hypreMatrixInsertSec   = " << rep.hypreMatrixInsertSeconds << "\n";
                std::cout << "hypreMatrixMigrateSec  = " << rep.hypreMatrixMigrateSeconds << "\n";
                std::cout << "hypreVectorInsertSec   = " << rep.hypreVectorInsertSeconds << "\n";
                std::cout << "hypreVectorMigrateSec  = " << rep.hypreVectorMigrateSeconds << "\n";
                std::cout << "hypreSetupSeconds      = " << rep.hypreSetupSeconds << "\n";
                std::cout << "hypreSolveOnlySeconds  = " << rep.hypreSolveOnlySeconds << "\n";
                std::cout << "hypreGetSolutionSec    = " << rep.hypreGetSolutionSeconds << "\n";
                std::cout << "errorNormSeconds       = " << rep.errorNormSeconds << "\n";
                std::cout << "hypreDestroyFinalSec   = " << rep.hypreDestroyFinalizeSeconds << "\n";
                std::cout << "--------------------------------------------------\n";
#else
                throw std::runtime_error("Built without MEMOIRS_USE_HYPRE=ON; cannot solve.");
#endif
            }
        }

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
