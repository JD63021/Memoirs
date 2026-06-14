#pragma once

// Auto-split from frozen patch014 one-shot source.
// This header is intentionally included in order by apps/poisson_mms.cpp.

// ============================================================================
// SECTION 8: Assembly and Dirichlet elimination
// ============================================================================

struct AssembledSystem {
    SparseRows A;
    std::vector<Real> b;
    int nDirichlet = 0;
};

static void assemble_hex_q1_cell(
    const PolyMesh& m,
    const Cell& c,
    const std::string& mms,
    SparseRows& A,
    std::vector<Real>& b
) {
    const auto hv = ordered_hex_vertices_axis_aligned(m, c);

    Real Ke[8][8] = {};
    Real Fe[8] = {};

    for (const auto& qpt : hex_q1_volume_quadrature_selected()) {
        const HexQ1BasisAtPoint basis = eval_hex_q1_basis_at(qpt.xi, qpt.eta, qpt.zeta);

        const Vec3 xq = map_hex_q1_to_physical(m, hv, basis.N);
        const Mat3 J = jacobian_hex_q1(m, hv, basis.dNref);
        const double detJ = det3(J);
        const double wdet = qpt.weight * std::abs(detJ);
        const Mat3 invJ = inv3(J);

        std::array<Vec3,8> grad;
        for (int a = 0; a < 8; ++a) {
            grad[a] = invJT_mul(invJ, basis.dNref[a]);
        }

        const double f = mms_rhs_value(xq, mms);

        for (int a = 0; a < 8; ++a) {
            Fe[a] += Real(basis.N[a] * f * wdet);
            for (int bb = 0; bb < 8; ++bb) {
                Ke[a][bb] += Real(dot3(grad[a], grad[bb]) * wdet);
            }
        }
    }

    for (int a = 0; a < 8; ++a) {
        int I = hv[a];
        b[I] += Fe[a];
        for (int bb = 0; bb < 8; ++bb) {
            int J = hv[bb];
            sparse_add(A, I, J, Ke[a][bb]);
        }
    }
}

static void assemble_tet_p1_cell(
    const PolyMesh& m,
    const Cell& c,
    const std::string& mms,
    SparseRows& A,
    std::vector<Real>& b
) {
    if (c.verts.size() != 4) throw std::runtime_error("Tet cell does not have 4 vertices.");

    std::array<int,4> tv = {c.verts[0], c.verts[1], c.verts[2], c.verts[3]};

    const Mat3 J = jacobian_tet_p1(m, tv);
    const double detJ = det3(J);
    const double volume = std::abs(detJ) / 6.0;
    const Mat3 invJ = inv3(J);

    const TetP1BasisAtPoint basis0 = eval_tet_p1_basis_at(0.25, 0.25, 0.25);

    std::array<Vec3,4> grad;
    for (int a = 0; a < 4; ++a) grad[a] = invJT_mul(invJ, basis0.dNref[a]);

    Real Ke[4][4] = {};
    Real Fe[4] = {};

    for (const auto& qpt : tet_p1_centroid_quadrature()) {
        const TetP1BasisAtPoint basis = eval_tet_p1_basis_at(qpt.xi, qpt.eta, qpt.zeta);
        const Vec3 xq = map_tet_p1_to_physical(m, tv, basis.N);
        const double f = mms_rhs_value(xq, mms);
        const double wdet = qpt.weight * std::abs(detJ);

        for (int a = 0; a < 4; ++a) {
            Fe[a] += Real(basis.N[a] * f * wdet);
        }
    }

    for (int a = 0; a < 4; ++a) {
        for (int bb = 0; bb < 4; ++bb) {
            Ke[a][bb] = Real(dot3(grad[a], grad[bb]) * volume);
        }
    }

    for (int a = 0; a < 4; ++a) {
        int I = tv[a];
        b[I] += Fe[a];
        for (int bb = 0; bb < 4; ++bb) {
            int Jg = tv[bb];
            sparse_add(A, I, Jg, Ke[a][bb]);
        }
    }
}

static void apply_dirichlet_all_boundary(
    const PolyMesh& m,
    const LinearCgDofMap& dm,
    const std::string& mms,
    SparseRows& A,
    std::vector<Real>& b
) {
    // Backward-compatible wrapper around the modular BC layer.  CG strong
    // Dirichlet is intentionally outside the volume operator so DG/SIPG can
    // later provide its own weak boundary integrator.
    apply_mms_strong_dirichlet_all_boundary(m, dm, mms, A, b);
}


struct HexQ1StructuredMetrics {
    double hx = 0.0;
    double hy = 0.0;
    double hz = 0.0;
};

static HexQ1StructuredMetrics hex_q1_structured_metrics(
    const PolyMesh& m,
    const std::array<int,8>& hv
) {
    double xmin =  std::numeric_limits<double>::infinity();
    double ymin =  std::numeric_limits<double>::infinity();
    double zmin =  std::numeric_limits<double>::infinity();
    double xmax = -std::numeric_limits<double>::infinity();
    double ymax = -std::numeric_limits<double>::infinity();
    double zmax = -std::numeric_limits<double>::infinity();

    for (int v : hv) {
        const Vec3& p = m.points[v];
        xmin = std::min(xmin, p.x); xmax = std::max(xmax, p.x);
        ymin = std::min(ymin, p.y); ymax = std::max(ymax, p.y);
        zmin = std::min(zmin, p.z); zmax = std::max(zmax, p.z);
    }

    HexQ1StructuredMetrics g;
    g.hx = xmax - xmin;
    g.hy = ymax - ymin;
    g.hz = zmax - zmin;

    if (!(g.hx > 0.0 && g.hy > 0.0 && g.hz > 0.0)) {
        throw std::runtime_error("structured_hex_q1_sumfact found a non-positive hex size.");
    }

    return g;
}

static void assert_structured_hex_q1_equisized(
    const PolyMesh& m,
    const std::vector<std::array<int,8>>& orderedHexes,
    const HexQ1StructuredMetrics& ref
) {
    const double scale = std::max({1.0, ref.hx, ref.hy, ref.hz});
    const double tol = 1.0e-10 * scale;

    for (std::size_t c = 0; c < orderedHexes.size(); ++c) {
        const HexQ1StructuredMetrics g = hex_q1_structured_metrics(m, orderedHexes[c]);

        if (std::abs(g.hx - ref.hx) > tol ||
            std::abs(g.hy - ref.hy) > tol ||
            std::abs(g.hz - ref.hz) > tol) {
            std::ostringstream oss;
            oss << "MEMOIRS_ASSEMBLY_MODE=structured_hex_q1_sumfact requires equisized "
                << "axis-aligned blockMesh cells. First cell h=("
                << ref.hx << "," << ref.hy << "," << ref.hz << ") but cell " << c
                << " has h=(" << g.hx << "," << g.hy << "," << g.hz << ").";
            throw std::runtime_error(oss.str());
        }
    }
}

static void build_hex_q1_sumfact_stiffness(
    const HexQ1StructuredMetrics& g,
    Real Ke[8][8]
) {
    // Q1 tensor-product stiffness on an affine cuboid with reference domain [-1,1]^3.
    //
    // 1D reference matrices:
    //   M_ab = int L_a L_b dr
    //   K_ab = int dL_a/dr dL_b/dr dr
    //
    // Physical mapping x = x0 + hx*(1+r)/2 gives coefficients:
    //   hy*hz/(2*hx), hx*hz/(2*hy), hx*hy/(2*hz).
    static const double M1[2][2] = {
        {2.0/3.0, 1.0/3.0},
        {1.0/3.0, 2.0/3.0}
    };

    static const double K1[2][2] = {
        { 0.5, -0.5},
        {-0.5,  0.5}
    };

    static const int lx[8] = {0,1,1,0,0,1,1,0};
    static const int ly[8] = {0,0,1,1,0,0,1,1};
    static const int lz[8] = {0,0,0,0,1,1,1,1};

    const double cx = g.hy * g.hz / (2.0 * g.hx);
    const double cy = g.hx * g.hz / (2.0 * g.hy);
    const double cz = g.hx * g.hy / (2.0 * g.hz);

    for (int a = 0; a < 8; ++a) {
        for (int b = 0; b < 8; ++b) {
            const double v =
                cx * K1[lx[a]][lx[b]] * M1[ly[a]][ly[b]] * M1[lz[a]][lz[b]] +
                cy * M1[lx[a]][lx[b]] * K1[ly[a]][ly[b]] * M1[lz[a]][lz[b]] +
                cz * M1[lx[a]][lx[b]] * M1[ly[a]][ly[b]] * K1[lz[a]][lz[b]];

            Ke[a][b] = Real(v);
        }
    }
}

static void assemble_hex_q1_sumfact_rhs(
    const PolyMesh& m,
    const std::array<int,8>& hv,
    const HexQ1StructuredMetrics& g,
    const std::string& mms,
    Real Fe[8]
) {
    // RHS uses the same 2x2x2 tensor-product Gauss quadrature as the generic path.
    // Written in tensor-product form so the p>1 extension can become true
    // sum-factorized source assembly.
    static const int lx[8] = {0,1,1,0,0,1,1,0};
    static const int ly[8] = {0,0,1,1,0,0,1,1};
    static const int lz[8] = {0,0,0,0,1,1,1,1};

    const double q = 1.0 / std::sqrt(3.0);
    const double qp[2] = {-q, q};
    const double detJ = g.hx * g.hy * g.hz / 8.0;

    for (int a = 0; a < 8; ++a) {
        Fe[a] = Real(0);
    }

    for (int ix = 0; ix < 2; ++ix) {
        const double Lx[2] = {0.5 * (1.0 - qp[ix]), 0.5 * (1.0 + qp[ix])};

        for (int iy = 0; iy < 2; ++iy) {
            const double Ly[2] = {0.5 * (1.0 - qp[iy]), 0.5 * (1.0 + qp[iy])};

            for (int iz = 0; iz < 2; ++iz) {
                const double Lz[2] = {0.5 * (1.0 - qp[iz]), 0.5 * (1.0 + qp[iz])};

                Vec3 xq;

                for (int a = 0; a < 8; ++a) {
                    const double Na = Lx[lx[a]] * Ly[ly[a]] * Lz[lz[a]];
                    const Vec3& X = m.points[hv[a]];

                    xq.x += Na * X.x;
                    xq.y += Na * X.y;
                    xq.z += Na * X.z;
                }

                const double f = mms_rhs_value(xq, mms);

                for (int a = 0; a < 8; ++a) {
                    const double Na = Lx[lx[a]] * Ly[ly[a]] * Lz[lz[a]];
                    Fe[a] += Real(Na * f * detJ);
                }
            }
        }
    }
}

static void assemble_laplacian_hex_q1_structured_sumfact(
    const PolyMesh& m,
    const std::string& mms,
    SparseRows& A,
    std::vector<Real>& b
) {
    if (m.cells.empty()) {
        return;
    }

    std::vector<std::array<int,8>> orderedHexes;
    orderedHexes.reserve(m.cells.size());

    for (const auto& c : m.cells) {
        orderedHexes.push_back(ordered_hex_vertices_axis_aligned(m, c));
    }

    const HexQ1StructuredMetrics ref = hex_q1_structured_metrics(m, orderedHexes.front());
    assert_structured_hex_q1_equisized(m, orderedHexes, ref);

    Real Ke[8][8] = {};
    build_hex_q1_sumfact_stiffness(ref, Ke);

    for (const auto& hv : orderedHexes) {
        Real Fe[8] = {};
        assemble_hex_q1_sumfact_rhs(m, hv, ref, mms, Fe);

        for (int a = 0; a < 8; ++a) {
            const int I = hv[a];
            b[I] += Fe[a];

            for (int bb = 0; bb < 8; ++bb) {
                sparse_add(A, I, hv[bb], Ke[a][bb]);
            }
        }
    }
}

static void assemble_cg_laplacian_volume_mms(
    const PolyMesh& m,
    const LinearCgDofMap& dm,
    const std::string& mms,
    SparseRows& A,
    std::vector<Real>& b
) {
    const std::string assemblyMode = memoirs_assembly_mode();

    if (dm.resolvedSpace == "cg_hex_q1") {
        if (assemblyMode == "structured_hex_q1_sumfact") {
            assemble_laplacian_hex_q1_structured_sumfact(m, mms, A, b);
        } else if (assemblyMode == "generic") {
            for (const auto& c : m.cells) {
                assemble_hex_q1_cell(m, c, mms, A, b);
            }
        } else {
            throw std::runtime_error("Unsupported assembly mode for cg_hex_q1: " + assemblyMode);
        }
    } else if (dm.resolvedSpace == "cg_tet_p1") {
        if (assemblyMode != "generic") {
            throw std::runtime_error("MEMOIRS_ASSEMBLY_MODE=" + assemblyMode +
                                     " is only implemented for cg_hex_q1 block meshes. Use generic for tets.");
        }

        for (const auto& c : m.cells) {
            assemble_tet_p1_cell(m, c, mms, A, b);
        }
    } else {
        throw std::runtime_error("Assembly unsupported for space: " + dm.resolvedSpace);
    }
}

static AssembledSystem assemble_laplacian_dirichlet_mms(
    const PolyMesh& m,
    const LinearCgDofMap& dm,
    const std::string& mms
) {
    AssembledSystem sys;
    sys.b.assign(dm.nDofs, Real(0));

    const std::string sparseMode = memoirs_sparse_mode();
    if (sparseMode == "fixed_csr") {
        sparse_init_scalar_cg_fixed_pattern(sys.A, m, dm);
    } else if (sparseMode == "legacy") {
        sys.A.rows.resize(dm.nDofs);
    } else {
        throw std::runtime_error("Unsupported sparse mode: " + sparseMode);
    }

    assemble_cg_laplacian_volume_mms(m, dm, mms, sys.A, sys.b);

    sys.nDirichlet = (int)dm.boundaryDofs.size();
    apply_dirichlet_all_boundary(m, dm, mms, sys.A, sys.b);

    return sys;
}

static void probe_assembled_system(
    const PolyMesh& m,
    const LinearCgDofMap& dm,
    const AssembledSystem& sys,
    const std::string& mms,
    int diagLevel
) {
    const int n = sparse_nrows(sys.A);

    double rhsL2 = 0.0;
    double rhsMax = 0.0;
    double minDiag = std::numeric_limits<double>::infinity();
    double maxDiag = 0.0;
    int missingDiag = 0;

    for (int i = 0; i < n; ++i) {
        rhsL2 += double(sys.b[i]) * double(sys.b[i]);
        rhsMax = std::max(rhsMax, std::abs(double(sys.b[i])));

        bool diagPresent = false;
        const double ad = sparse_diag_abs_or_zero(sys.A, i, diagPresent);
        if (!diagPresent) {
            missingDiag++;
        } else {
            minDiag = std::min(minDiag, ad);
            maxDiag = std::max(maxDiag, ad);
        }
    }
    rhsL2 = std::sqrt(rhsL2);

    std::cout << "------------- assembly probe ---------------\n";
    std::cout << "operator                  = laplacian(identity, u)\n";
    std::cout << "weak form                 = int grad(v).grad(u) dOmega\n";
    std::cout << "bc                        = Dirichlet MMS on all boundary dofs\n";
    std::cout << "mms                       = " << mms << "\n";
    std::cout << "space                     = " << dm.resolvedSpace << "\n";
    std::cout << "rows                      = " << n << "\n";
    std::cout << "nnz                       = " << sparse_nnz(sys.A) << "\n";
    std::cout << "nDirichlet                = " << sys.nDirichlet << "\n";
    std::cout << "rhsL2                     = " << std::setprecision(16) << rhsL2 << "\n";
    std::cout << "rhsMax                    = " << std::setprecision(16) << rhsMax << "\n";
    std::cout << "diagAbs min/max           = " << minDiag << " / " << maxDiag << "\n";
    std::cout << "missingDiag               = " << missingDiag << "\n";

    std::vector<Real> xExact(n, Real(0));
    for (int i = 0; i < n; ++i) {
        xExact[i] = Real(mms_exact_value(m.points[i], mms));
    }

    double resL2 = 0.0;
    double resMax = 0.0;
    for (int i = 0; i < n; ++i) {
        const double ri = double(sparse_matvec_row(sys.A, i, xExact) - sys.b[i]);
        resL2 += ri * ri;
        resMax = std::max(resMax, std::abs(ri));
    }
    resL2 = std::sqrt(resL2);

    std::cout << "exactNodalResidualL2      = " << std::setprecision(16) << resL2 << "\n";
    std::cout << "exactNodalResidualMax     = " << std::setprecision(16) << resMax << "\n";

    if (diagLevel >= 2) {
        const double symMax = sparse_symmetry_max_abs(sys.A);
        std::cout << "symmetryMaxAbs            = " << std::setprecision(16) << symMax << "\n";
    }

    std::cout << "hypre residency note      = use -hypreMemory host|device; assembly remains host-side in one-shot\n";
    std::cout << "--------------------------------------------\n";
}
