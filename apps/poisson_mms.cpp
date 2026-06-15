// Memoirs v0 modular app: Poisson MMS driver.
// Sections are included in order to preserve frozen patch014 behavior.

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
#include "memoirs/sections/06a_rect_csr.hpp"
#include "memoirs/sections/06b_local_blocks.hpp"
#include "memoirs/sections/07_mms.hpp"
#include "memoirs/sections/08a_boundary_conditions.hpp"
#include "memoirs/sections/08_laplacian_assembler.hpp"
#include "memoirs/sections/10_scalar_elliptic_spec.hpp"
#include "memoirs/sections/12_cg_tet_p1_nitsche.hpp"
#include "memoirs/sections/14_cg_hex_q1_nitsche.hpp"
#include "memoirs/sections/09_hypre_solver.hpp"
#include "memoirs/sections/10_output_placeholder.hpp"
#include "memoirs/diagnostics/GpuMemoryMonitor.hpp"

// ============================================================================
// SECTION 11: main()
// ============================================================================

int main(int argc, char** argv) {
    GpuMemoryMonitor gpuMemMon;

    try {
        Options opt = parse_cli(argc, argv);
        ScalarEllipticSpec scalarSpec = parse_scalar_elliptic_spec_from_cli(argc, argv, "strong", 10.0);
        MemoirsQuadratureSpec quadSpec = parse_memoirs_quadrature_spec_from_cli(argc, argv);
        (void)quadSpec;
        gpuMemMon.start();

        std::cout << "Memoirs v0 :: one-shot Q1/P1 Poisson prototype\n";
        std::cout << "precision         = " << kPrecisionName << "\n";

        if (opt.polyMeshDir.empty()) {
            throw std::runtime_error("Missing -polyMeshDir");
        }

        std::cout << "polyMeshDir       = " << opt.polyMeshDir << "\n";

        std::cout << "space             = " << opt.space << "\n";
        std::cout << "scalarBcMode      = " << scalar_bc_mode_name(scalarSpec.boundary.mode) << "\n";
        std::cout << "effectiveOperatorBc= " << scalar_bc_mode_name(scalarSpec.boundary.mode) << "\n";
        std::cout << "penaltySigma      = " << scalarSpec.boundary.penaltySigma << "\n";
        std::cout << "kappa             = " << scalarSpec.diffusion.constant << "\n";

        const bool stageTimers = memoirs_stage_timers_enabled();

        auto tRead0 = std::chrono::steady_clock::now();
        if (stageTimers) {
            std::cerr << "[stage] read_polymesh begin\n" << std::flush;
        }
        PolyMesh mesh = read_polymesh(opt.polyMeshDir);
        if (stageTimers) {
            std::cerr << "[stage] read_polymesh done seconds=" << seconds_since(tRead0)
                      << " points=" << mesh.points.size()
                      << " faces=" << mesh.faces.size()
                      << " cells=" << mesh.cells.size()
                      << "\n" << std::flush;
        }

        if (opt.probeMesh) {
            probe_mesh(mesh);
            probe_memoirs_topology_tables(mesh);
        }

        if (opt.probeDofs) {
            LinearCgDofMap dm = build_linear_cg_dof_map(mesh, opt.space);
            probe_dofs(mesh, dm);
        }

        if (opt.assemble) {
            LinearCgDofMap dm = build_linear_cg_dof_map(mesh, opt.space);
            AssembledSystem sys;
            if (scalarSpec.boundary.mode == ScalarBoundaryMode::Strong) {
                sys = assemble_laplacian_dirichlet_mms(mesh, dm, opt.mms);
            } else if (scalarSpec.boundary.mode == ScalarBoundaryMode::WeakNitsche) {
                if (dm.resolvedSpace == "cg_tet_p1") {
                    sys = assemble_cg_tet_p1_nitsche_poisson_mms(mesh, dm, opt.mms, scalarSpec);
                } else if (dm.resolvedSpace == "cg_tet_p2") {
                    sys = assemble_cg_tet_p2_nitsche_poisson_mms(mesh, dm, opt.mms, scalarSpec);
                } else if (dm.resolvedSpace == "cg_hex_q1") {
                    sys = assemble_cg_hex_q1_nitsche_poisson_mms(mesh, dm, opt.mms, scalarSpec);
                } else {
                    throw std::runtime_error("CG Nitsche supports cg_tet_p1, cg_tet_p2, and cg_hex_q1.");
                }
            } else {
                throw std::runtime_error("CG Poisson app supports -bc strong or -bc nitsche.");
            }
            probe_assembled_system(mesh, dm, sys, opt.mms, opt.diagLevel);
        }

        if (opt.solve) {
            auto tDof0 = std::chrono::steady_clock::now();
            if (stageTimers) {
                std::cerr << "[stage] build_linear_cg_dof_map begin\n" << std::flush;
            }
            LinearCgDofMap dm = build_linear_cg_dof_map(mesh, opt.space);
            if (stageTimers) {
                std::cerr << "[stage] build_linear_cg_dof_map done seconds=" << seconds_since(tDof0)
                          << " nDofs=" << dm.nDofs
                          << " resolvedSpace=" << dm.resolvedSpace
                          << "\n" << std::flush;
            }

            auto tAsm0 = std::chrono::steady_clock::now();
            if (stageTimers) {
                std::cerr << "[stage] assemble_laplacian_dirichlet_mms begin\n" << std::flush;
            }
            AssembledSystem sys;
            if (scalarSpec.boundary.mode == ScalarBoundaryMode::Strong) {
                sys = assemble_laplacian_dirichlet_mms(mesh, dm, opt.mms);
            } else if (scalarSpec.boundary.mode == ScalarBoundaryMode::WeakNitsche) {
                if (dm.resolvedSpace == "cg_tet_p1") {
                    sys = assemble_cg_tet_p1_nitsche_poisson_mms(mesh, dm, opt.mms, scalarSpec);
                } else if (dm.resolvedSpace == "cg_tet_p2") {
                    sys = assemble_cg_tet_p2_nitsche_poisson_mms(mesh, dm, opt.mms, scalarSpec);
                } else if (dm.resolvedSpace == "cg_hex_q1") {
                    sys = assemble_cg_hex_q1_nitsche_poisson_mms(mesh, dm, opt.mms, scalarSpec);
                } else {
                    throw std::runtime_error("CG Nitsche supports cg_tet_p1, cg_tet_p2, and cg_hex_q1.");
                }
            } else {
                throw std::runtime_error("CG Poisson app supports -bc strong or -bc nitsche.");
            }
            auto tAsm1 = std::chrono::steady_clock::now();
            if (stageTimers) {
                std::cerr << "[stage] assemble_laplacian_dirichlet_mms done seconds="
                          << std::chrono::duration<double>(tAsm1 - tAsm0).count()
                          << " rows=" << sparse_nrows(sys.A)
                          << " nnz=" << sparse_nnz(sys.A)
                          << "\n" << std::flush;
            }

            if (opt.diagLevel >= 1) {
                auto tProbe0 = std::chrono::steady_clock::now();
                if (stageTimers) {
                    std::cerr << "[stage] probe_assembled_system begin\n" << std::flush;
                }
                probe_assembled_system(mesh, dm, sys, opt.mms, opt.diagLevel);
                if (stageTimers) {
                    std::cerr << "[stage] probe_assembled_system done seconds="
                              << seconds_since(tProbe0) << "\n" << std::flush;
                }
            }

            std::vector<Real> x;
            auto tSol0 = std::chrono::steady_clock::now();
            if (stageTimers) {
                std::cerr << "[stage] solve_hypre_ij_pcg begin\n" << std::flush;
            }
            SolveReport rep = solve_hypre_ij_pcg(mesh, dm, sys, opt.mms, opt, x);
            auto tSol1 = std::chrono::steady_clock::now();
            if (stageTimers) {
                std::cerr << "[stage] solve_hypre_ij_pcg done seconds="
                          << std::chrono::duration<double>(tSol1 - tSol0).count()
                          << "\n" << std::flush;
            }

            rep.assemblySeconds = std::chrono::duration<double>(tAsm1 - tAsm0).count();
            rep.solveSeconds = std::chrono::duration<double>(tSol1 - tSol0).count();

            print_solve_report(opt, dm, rep);
        }

        if (!opt.probeMesh && !opt.probeDofs && !opt.assemble && !opt.solve) {
            std::cout << "Nothing to do yet. Use -probeMesh 1, -probeDofs 1, -assemble 1, and/or -solve 1.\n";
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
