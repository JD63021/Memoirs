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
#include "memoirs/sections/09b_hypre_gmres_raw.hpp"
#include "memoirs/sections/10_scalar_elliptic_spec.hpp"
#include "memoirs/sections/16_rt0_mixed_poisson.hpp"
#include "memoirs/diagnostics/GpuMemoryMonitor.hpp"

int main(int argc, char** argv) {
    GpuMemoryMonitor gpuMemMon;

    try {
        Options opt = parse_cli(argc, argv);

        ScalarEllipticSpec scalarSpec = parse_scalar_elliptic_spec_from_cli(
            argc,
            argv,
            "strong",
            0.0
        );
        (void)scalarSpec;

        MemoirsQuadratureSpec quadSpec = parse_memoirs_quadrature_spec_from_cli(argc, argv);
        (void)quadSpec;

        const std::string vtuFile = parse_rt0_vtu_file_from_cli(argc, argv);

        gpuMemMon.start();

        std::cout << "Memoirs v0 :: RT0 mixed H(div) Poisson MMS\n";
        std::cout << "precision         = " << kPrecisionName << "\n";
        std::cout << "polyMeshDir       = " << opt.polyMeshDir << "\n";
        std::cout << "requestedSpace    = " << opt.space << "\n";
        std::cout << "mms               = " << opt.mms << "\n";
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

        if (lower_copy(opt.precond) == "diagscale") {
            throw std::runtime_error(
                "RT0 mixed saddle matrix has a zero scalar block; "
                "-precond diagscale is invalid. Use -precond none for now."
            );
        }

        Rt0DofMap dm = build_rt0_dof_map(mesh);

        std::cout << "resolvedSpace     = " << dm.resolvedSpace << "\n";
        std::cout << "nFluxDofs         = " << dm.nFluxDofs << "\n";
        std::cout << "nScalarDofs       = " << dm.nScalarDofs << "\n";
        std::cout << "nTotalDofs        = " << dm.nTotalDofs << "\n";

        if (opt.probeDofs) {
            std::cout << "---------------- RT0 dof probe -----------------\n";
            std::cout << "space             = " << dm.resolvedSpace << "\n";
            std::cout << "fluxDofs          = one oriented normal flux per face\n";
            std::cout << "scalarDofs        = one cellwise scalar per cell\n";
            std::cout << "nFluxDofs         = " << dm.nFluxDofs << "\n";
            std::cout << "nScalarDofs       = " << dm.nScalarDofs << "\n";
            std::cout << "nTotalDofs        = " << dm.nTotalDofs << "\n";
            std::cout << "------------------------------------------------\n";
        }

        if (opt.assemble || opt.solve) {
            auto tAsm0 = std::chrono::steady_clock::now();

            AssembledSystem sys = assemble_rt0_mixed_poisson_mms(mesh, dm, opt.mms);

            auto tAsm1 = std::chrono::steady_clock::now();
            const double assemblySeconds = std::chrono::duration<double>(tAsm1 - tAsm0).count();

            if (opt.diagLevel >= 1 || opt.assemble) {
                probe_rt0_mixed_system(mesh, dm, sys, opt.diagLevel);
            }

            if (opt.solve) {
#if defined(MEMOIRS_USE_HYPRE)
                std::vector<Real> x;

                auto tSol0 = std::chrono::steady_clock::now();
                const std::string rt0SolveMode = memoirs_rt0_solve_mode();
                SolveReport rep;
                if (rt0SolveMode == "gmres") {
                    rep = solve_hypre_ij_gmres_raw(sys.A, sys.b, opt, x);
                } else if (rt0SolveMode == "lumped_schur") {
                    rep = solve_rt0_lumped_schur_approx(mesh, dm, sys, opt, x);
                } else if (rt0SolveMode == "block_schur_gmres") {
                    rep = solve_rt0_block_schur_fgmres(mesh, dm, sys, opt, x);
                } else {
                    throw std::runtime_error("Unsupported MEMOIRS_RT0_SOLVE_MODE: " + rt0SolveMode);
                }
                auto tErr0 = std::chrono::steady_clock::now();
                Rt0ErrorReport er = compute_rt0_mixed_errors(mesh, dm, opt.mms, x);

                if (!vtuFile.empty()) {
                    write_rt0_mixed_vtu(vtuFile, mesh, dm, opt.mms, x);
                }

                auto tErr1 = std::chrono::steady_clock::now();
                auto tSol1 = std::chrono::steady_clock::now();

                rep.assemblySeconds = assemblySeconds;
                rep.solveSeconds = std::chrono::duration<double>(tSol1 - tSol0).count();
                rep.errorNormSeconds = std::chrono::duration<double>(tErr1 - tErr0).count();

                std::cout << "---------------- RT0 mixed solve report -----------------\n";
                std::cout << "space                         = " << dm.resolvedSpace << "\n";
                std::cout << "nFluxDofs                     = " << dm.nFluxDofs << "\n";
                std::cout << "nScalarDofs                   = " << dm.nScalarDofs << "\n";
                std::cout << "nTotalDofs                    = " << dm.nTotalDofs << "\n";
                std::cout << "rt0SolveMode                 = " << rt0SolveMode << "\n";
                std::cout << "solver                        = "
                          << (rt0SolveMode == "gmres" ? "gmres" :
                              (rt0SolveMode == "lumped_schur" ? "pcg_on_lumped_schur" :
                               "host_fgmres_block_schur")) << "\n";
                std::cout << "precond                       = " << opt.precond << "\n";
                std::cout << "iterations                    = " << rep.iterations << "\n";
                std::cout << "finalRelativeResidual         = " << std::scientific << rep.finalRelRes << std::defaultfloat << "\n";
                std::cout << "scalarL2Error                 = " << std::scientific << er.scalarL2 << "\n";
                std::cout << "fluxL2Error                   = " << er.fluxL2 << "\n";
                std::cout << "divCellL2Error                = " << er.divCellL2 << "\n";
                std::cout << "divCellAbsMax                 = " << er.divCellAbsMax << "\n";
                std::cout << "cellConservationL2            = " << er.cellConservationL2 << "\n";
                std::cout << "cellConservationAbsMax        = " << er.cellConservationAbsMax << std::defaultfloat << "\n";
                if (!vtuFile.empty()) {
                    std::cout << "vtuFile                       = " << vtuFile << "\n";
                }
                std::cout << "assemblySeconds               = " << rep.assemblySeconds << "\n";
                std::cout << "solveSeconds                  = " << rep.solveSeconds << "\n";
                std::cout << "hypreMatrixInsertSec          = " << rep.hypreMatrixInsertSeconds << "\n";
                std::cout << "hypreMatrixMigrateSec         = " << rep.hypreMatrixMigrateSeconds << "\n";
                std::cout << "hypreVectorInsertSec          = " << rep.hypreVectorInsertSeconds << "\n";
                std::cout << "hypreVectorMigrateSec         = " << rep.hypreVectorMigrateSeconds << "\n";
                std::cout << "hypreSetupSeconds             = " << rep.hypreSetupSeconds << "\n";
                std::cout << "hypreSolveOnlySeconds         = " << rep.hypreSolveOnlySeconds << "\n";
                std::cout << "hypreGetSolutionSec           = " << rep.hypreGetSolutionSeconds << "\n";
                std::cout << "errorNormSeconds              = " << rep.errorNormSeconds << "\n";
                std::cout << "hypreDestroyFinalSec          = " << rep.hypreDestroyFinalizeSeconds << "\n";
                std::cout << "--------------------------------------------------------\n";
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
