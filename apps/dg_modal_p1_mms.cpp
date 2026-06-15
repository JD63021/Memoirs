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
#include "memoirs/sections/15_dg_modal_p1.hpp"
#include "memoirs/diagnostics/GpuMemoryMonitor.hpp"

int main(int argc, char** argv) {
    GpuMemoryMonitor gpuMemMon;

    try {
        Options opt = parse_cli(argc, argv);
        ScalarEllipticSpec scalarSpec = parse_scalar_elliptic_spec_from_cli(
            argc,
            argv,
            "sipg",
            10.0
        );
        MemoirsQuadratureSpec quadSpec = parse_memoirs_quadrature_spec_from_cli(argc, argv);
        (void)quadSpec;

        if (scalarSpec.boundary.mode == ScalarBoundaryMode::Strong) {
            throw std::runtime_error("Modal DG does not support -bc strong; use weak SIPG/Nitsche-style BC.");
        }

        DgModalSipgOptions dgopt;
        dgopt.penaltySigma = scalarSpec.boundary.penaltySigma;

        gpuMemMon.start();

        std::cout << "Memoirs v0 :: orthogonal modal DG P1 SIPG Poisson MMS\n";
        std::cout << "precision         = " << kPrecisionName << "\n";
        std::cout << "polyMeshDir       = " << opt.polyMeshDir << "\n";
        std::cout << "requestedSpace    = " << opt.space << "\n";
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

        DgModalP1DofMap dm = build_dg_modal_p1_dof_map(mesh);

        std::cout << "resolvedSpace     = " << dm.resolvedSpace << "\n";
        std::cout << "modesPerCell      = " << dm.modesPerCell << "\n";

        if (opt.probeDofs) {
            std::cout << "---------------- DG modal dof probe -----------------\n";
            std::cout << "space             = " << dm.resolvedSpace << "\n";
            std::cout << "continuity        = DG\n";
            std::cout << "basis             = cell-local L2-orthonormal modal P1\n";
            std::cout << "nCells            = " << dm.nCells << "\n";
            std::cout << "nDofs             = " << dm.nDofs << "\n";
            std::cout << "modesPerCell      = " << dm.modesPerCell << "\n";
            std::cout << "-----------------------------------------------------\n";
        }

        if (opt.assemble || opt.solve) {
            auto tAsm0 = std::chrono::steady_clock::now();

            AssembledSystem sys = assemble_dg_modal_p1_sipg_poisson_mms(
                mesh,
                dm,
                opt.mms,
                dgopt
            );

            auto tAsm1 = std::chrono::steady_clock::now();
            const double assemblySeconds = std::chrono::duration<double>(tAsm1 - tAsm0).count();

            if (opt.diagLevel >= 1 || opt.assemble) {
                probe_dg_modal_p1_system(mesh, dm, sys, opt.diagLevel);
            }

            if (opt.solve) {
#if defined(MEMOIRS_USE_HYPRE)
                std::vector<Real> x;

                auto tSol0 = std::chrono::steady_clock::now();

                SolveReport rep = solve_hypre_ij_pcg_raw(sys.A, sys.b, opt, x);

                auto tErr0 = std::chrono::steady_clock::now();
                ErrorNorms en = compute_dg_modal_p1_error_norms(mesh, dm, opt.mms, x);
                auto tErr1 = std::chrono::steady_clock::now();

                auto tSol1 = std::chrono::steady_clock::now();

                rep.assemblySeconds = assemblySeconds;
                rep.solveSeconds = std::chrono::duration<double>(tSol1 - tSol0).count();
                rep.errorNormSeconds = std::chrono::duration<double>(tErr1 - tErr0).count();
                rep.nodalErrorL2 = en.nodalL2;
                rep.nodalErrorMax = en.nodalMax;
                rep.L2Error = en.L2;
                rep.H1SemiError = en.H1Semi;

                std::cout << "---------------- DG modal solve report -----------------\n";
                std::cout << "space                  = " << dm.resolvedSpace << "\n";
                std::cout << "nDofs                  = " << dm.nDofs << "\n";
                std::cout << "modesPerCell           = " << dm.modesPerCell << "\n";
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
