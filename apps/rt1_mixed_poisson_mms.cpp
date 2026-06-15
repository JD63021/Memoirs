
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
#include "memoirs/sections/17_rt1_tet_dg1_mixed_poisson.hpp"
#include "memoirs/diagnostics/GpuMemoryMonitor.hpp"

int main(int argc, char** argv) {
    GpuMemoryMonitor gpuMemMon;
    try {
        Options opt = parse_cli(argc, argv);
        (void)parse_memoirs_quadrature_spec_from_cli(argc, argv);
        gpuMemMon.start();
        std::cout << "Memoirs v0 :: RT1/DG1 tet mixed Poisson MMS\n";
        std::cout << "precision         = " << kPrecisionName << "\n";
        std::cout << "polyMeshDir       = " << opt.polyMeshDir << "\n";
        std::cout << "requestedSpace    = " << opt.space << "\n";
        std::cout << "mms               = " << opt.mms << "\n";
        std::cout << "sparseMode        = " << memoirs_sparse_mode() << "\n";
        print_memoirs_quadrature_orders(std::cout);
        if (opt.polyMeshDir.empty()) throw std::runtime_error("Missing -polyMeshDir");
        PolyMesh mesh = read_polymesh(opt.polyMeshDir);
        if (opt.probeMesh) probe_mesh(mesh);
        Rt1DofMap dm = build_rt1_tet_dg1_dof_map(mesh);
        std::cout << "resolvedSpace     = " << dm.resolvedSpace << "\n";
        std::cout << "nFaceMomentDofs   = " << dm.nFaceMomentDofs << "\n";
        std::cout << "nInteriorFluxDofs = " << dm.nInteriorFluxDofs << "\n";
        std::cout << "nFluxDofs         = " << dm.nFluxDofs << "\n";
        std::cout << "nScalarDofs       = " << dm.nScalarDofs << "\n";
        std::cout << "nTotalDofs        = " << dm.nTotalDofs << "\n";
        if (opt.probeDofs) {
            std::cout << "---------------- RT1/DG1 dof probe ----------------\n";
            std::cout << "flux face dofs      = 3 P1 normal moments per mesh face\n";
            std::cout << "flux cell dofs      = 3 cell-interior vector moments per tet\n";
            std::cout << "scalar dofs         = 4 nodal DG1 dofs per tet\n";
            std::cout << "----------------------------------------------------\n";
        }
        if (opt.assemble || opt.solve) {
            auto tA0=std::chrono::steady_clock::now();
            AssembledSystem sys = assemble_rt1_tet_dg1_mixed_poisson_mms(mesh, dm, opt.mms);
            auto tA1=std::chrono::steady_clock::now();
            double assemblySeconds=std::chrono::duration<double>(tA1-tA0).count();
            if (opt.diagLevel>=1 || opt.assemble) probe_rt1_system(dm, sys);
            if (opt.solve) {
                std::vector<Real> x;
                SolveReport rep;
                const std::string pre=lower_copy(opt.precond);
                auto tS0=std::chrono::steady_clock::now();
                if (pre=="block_schur_amg" || pre=="global_schur" || pre=="global_schur_amg" || pre=="schur_amg" || pre=="rt0_style_schur") {
                    rep = solve_rt1_block_schur_amg_fgmres(sys.A, dm, sys.b, opt, x);
                } else if (pre=="cell_saddle" || pre=="cell_block" || pre=="local_saddle" || pre=="block_jacobi_saddle") {
                    rep = solve_rt1_cell_saddle_fgmres(mesh, sys.A, dm, sys.b, opt, x);
                } else if (pre=="block_schur" || pre=="schur" || pre=="block_schur_diag" || pre=="diag_schur") {
                    rep = solve_rt1_block_schur_diag_fgmres(sys.A, dm, sys.b, opt, x);
                } else {
#if defined(MEMOIRS_USE_HYPRE)
                    rep = solve_hypre_ij_gmres_raw(sys.A, sys.b, opt, x);
#else
                    throw std::runtime_error("Built without HYPRE; use -precond block_schur for host FGMRES only.");
#endif
                }
                auto tE0=std::chrono::steady_clock::now();
                Rt1FieldErrorReport er=compute_rt1_tet_dg1_errors(mesh, dm, opt.mms, x);
                auto tE1=std::chrono::steady_clock::now();
                auto tS1=std::chrono::steady_clock::now();
                rep.assemblySeconds=assemblySeconds;
                rep.solveSeconds=std::chrono::duration<double>(tS1-tS0).count();
                rep.errorNormSeconds=std::chrono::duration<double>(tE1-tE0).count();
                std::cout << "---------------- RT1/DG1 mixed solve report -----------------\n";
                std::cout << "space                         = " << dm.resolvedSpace << "\n";
                std::cout << "nFluxDofs                     = " << dm.nFluxDofs << "\n";
                std::cout << "nScalarDofs                   = " << dm.nScalarDofs << "\n";
                std::cout << "nTotalDofs                    = " << dm.nTotalDofs << "\n";
                std::cout << "solver                        = " << ((pre=="block_schur"||pre=="schur"||pre=="block_schur_diag"||pre=="diag_schur"||pre=="cell_saddle"||pre=="cell_block"||pre=="local_saddle"||pre=="block_jacobi_saddle"||pre=="block_schur_amg"||pre=="global_schur"||pre=="global_schur_amg"||pre=="schur_amg"||pre=="rt0_style_schur") ? "host_fgmres" : "hypre_gmres") << "\n";
                std::cout << "precond                       = " << opt.precond << "\n";
                std::cout << "iterations                    = " << rep.iterations << "\n";
                std::cout << "finalRelativeResidual         = " << std::scientific << rep.finalRelRes << std::defaultfloat << "\n";
                std::cout << "scalarL2Error                 = " << std::scientific << er.scalarL2 << "\n";
                std::cout << "fluxL2Error                   = " << er.fluxL2 << "\n";
                std::cout << "divCellL2Error                = " << er.divCellL2 << "\n";
                std::cout << "cellConservationL2            = " << er.cellConservationL2 << "\n";
                std::cout << "cellConservationAbsMax        = " << er.cellConservationAbsMax << std::defaultfloat << "\n";
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
                std::cout << "------------------------------------------------------------\n";
            }
        }
        gpuMemMon.stop(); gpuMemMon.print(std::cout); return 0;
    } catch (const std::exception& e) {
        gpuMemMon.stop(); gpuMemMon.print(std::cout); std::cerr << "FATAL: " << e.what() << "\n"; return 1;
    }
}
