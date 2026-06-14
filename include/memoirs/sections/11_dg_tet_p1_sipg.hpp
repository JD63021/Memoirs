#pragma once

// DG tet P1 SIPG Poisson operator.
// This is intentionally a separate operator family from CG strong-Dirichlet Poisson.

struct DgTetP1DofMap {
    std::string resolvedSpace = "dg_tet_p1";
    int nCells = 0;
    int nDofs = 0;
};

struct DgTetP1CellGeom {
    std::array<int,4> verts{};
    std::array<Vec3,4> x{};
    std::array<Vec3,4> grad{};
    double volume = 0.0;
    Vec3 centroid{};
};

struct DgSipgOptions {
    double penaltySigma = 20.0;
};

static inline Vec3 dg_add(const Vec3& a, const Vec3& b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

static inline Vec3 dg_mul(double s, const Vec3& a) {
    return {s*a.x, s*a.y, s*a.z};
}

static inline double dg_norm(const Vec3& a) {
    return std::sqrt(dot3(a,a));
}

static DgTetP1DofMap build_dg_tet_p1_dof_map(const PolyMesh& m) {
    MeshCellCounts cc = count_cell_types(m);
    if (cc.nTet != (int)m.cells.size() || cc.nHex != 0 || cc.nOther != 0) {
        throw std::runtime_error("dg_tet_p1 requires a pure tet mesh.");
    }

    DgTetP1DofMap dm;
    dm.nCells = (int)m.cells.size();
    dm.nDofs = 4 * dm.nCells;
    return dm;
}

static void dg_tet_p1_orient_and_compute(DgTetP1CellGeom& g) {
    double vol6 = tet_signed_volume6(g.x[0], g.x[1], g.x[2], g.x[3]);

    if (vol6 < 0.0) {
        std::swap(g.verts[2], g.verts[3]);
        std::swap(g.x[2], g.x[3]);
        vol6 = -vol6;
    }

    if (!(vol6 > 0.0)) {
        throw std::runtime_error("DG tet has non-positive volume.");
    }

    Mat3 J;
    J.a[0][0] = g.x[1].x - g.x[0].x; J.a[1][0] = g.x[1].y - g.x[0].y; J.a[2][0] = g.x[1].z - g.x[0].z;
    J.a[0][1] = g.x[2].x - g.x[0].x; J.a[1][1] = g.x[2].y - g.x[0].y; J.a[2][1] = g.x[2].z - g.x[0].z;
    J.a[0][2] = g.x[3].x - g.x[0].x; J.a[1][2] = g.x[3].y - g.x[0].y; J.a[2][2] = g.x[3].z - g.x[0].z;

    Mat3 invJ = inv3(J);
    auto rg = tet_p1_grad_ref();

    for (int a = 0; a < 4; ++a) {
        g.grad[a] = invJT_mul(invJ, rg[a]);
    }

    g.volume = vol6 / 6.0;
    g.centroid = dg_mul(
        0.25,
        dg_add(dg_add(g.x[0], g.x[1]), dg_add(g.x[2], g.x[3]))
    );
}

static std::vector<DgTetP1CellGeom> build_dg_tet_p1_geoms(const PolyMesh& m) {
    std::vector<DgTetP1CellGeom> geoms(m.cells.size());

    for (int c = 0; c < (int)m.cells.size(); ++c) {
        if (m.cells[c].verts.size() != 4) {
            throw std::runtime_error("DG tet P1 found non-tet cell.");
        }

        DgTetP1CellGeom g;

        for (int a = 0; a < 4; ++a) {
            g.verts[a] = m.cells[c].verts[a];
            g.x[a] = m.points[g.verts[a]];
        }

        dg_tet_p1_orient_and_compute(g);
        geoms[c] = g;
    }

    return geoms;
}

static int dg_local_index_for_vertex(const DgTetP1CellGeom& K, int v) {
    for (int a = 0; a < 4; ++a) {
        if (K.verts[a] == v) return a;
    }
    return -1;
}

static std::array<double,4> dg_face_phi(
    const Face& f,
    const DgTetP1CellGeom& K,
    const std::array<double,3>& lam
) {
    std::array<double,4> phi = {0.0, 0.0, 0.0, 0.0};

    if (f.verts.size() != 3) {
        throw std::runtime_error("DG tet P1 expected triangular face.");
    }

    for (int q = 0; q < 3; ++q) {
        int li = dg_local_index_for_vertex(K, f.verts[q]);
        if (li < 0) {
            throw std::runtime_error("DG face vertex not found in tet.");
        }
        phi[li] = lam[q];
    }

    return phi;
}

static Vec3 dg_face_centroid(const PolyMesh& m, const Face& f) {
    Vec3 c;
    for (int v : f.verts) c = dg_add(c, m.points[v]);
    return dg_mul(1.0 / double(f.verts.size()), c);
}

static Vec3 dg_owner_outward_unit_normal(
    const PolyMesh& m,
    int faceId,
    const DgTetP1CellGeom& ownerGeom,
    double& area
) {
    const Face& f = m.faces[faceId];

    if (f.verts.size() != 3) {
        throw std::runtime_error("DG tet P1 expected triangular face.");
    }

    const Vec3& p0 = m.points[f.verts[0]];
    const Vec3& p1 = m.points[f.verts[1]];
    const Vec3& p2 = m.points[f.verts[2]];

    Vec3 n2 = cross3(p1 - p0, p2 - p0);
    double mag = dg_norm(n2);

    if (!(mag > 0.0)) {
        throw std::runtime_error("DG degenerate triangular face.");
    }

    area = 0.5 * mag;
    Vec3 n = dg_mul(1.0 / mag, n2);

    Vec3 fc = dg_face_centroid(m, f);
    if (dot3(n, fc - ownerGeom.centroid) < 0.0) {
        n = dg_mul(-1.0, n);
    }

    return n;
}

static std::vector<std::vector<int>> build_dg_tet_p1_sipg_pattern(
    const PolyMesh& m,
    const DgTetP1DofMap& dm
) {
    std::vector<std::vector<int>> rowCols(dm.nDofs);

    auto add_block = [&](int cRow, int cCol) {
        for (int i = 0; i < 4; ++i) {
            auto& cols = rowCols[4*cRow + i];
            for (int j = 0; j < 4; ++j) {
                cols.push_back(4*cCol + j);
            }
        }
    };

    for (int c = 0; c < dm.nCells; ++c) {
        add_block(c, c);
    }

    for (int f = 0; f < (int)m.neighbour.size(); ++f) {
        int P = m.owner[f];
        int N = m.neighbour[f];
        add_block(P, N);
        add_block(N, P);
    }

    return rowCols;
}

static AssembledSystem assemble_dg_tet_p1_sipg_poisson_mms(
    const PolyMesh& m,
    const DgTetP1DofMap& dm,
    const std::string& mms,
    const DgSipgOptions& dgopt,
    bool includeWeakBoundaryTerms = true
) {
    AssembledSystem sys;
    sys.b.assign(dm.nDofs, Real(0));

    const std::string sparseMode = memoirs_sparse_mode();
    if (sparseMode == "fixed_csr") {
        sparse_init_fixed_pattern(sys.A, build_dg_tet_p1_sipg_pattern(m, dm));
    } else if (sparseMode == "legacy") {
        sys.A.rows.resize(dm.nDofs);
    } else {
        throw std::runtime_error("Unsupported DG sparse mode: " + sparseMode);
    }

    const auto geom = build_dg_tet_p1_geoms(m);

    const std::array<std::array<double,4>,4> tetQ = {{
        {{0.5854101966249685, 0.1381966011250105, 0.1381966011250105, 0.1381966011250105}},
        {{0.1381966011250105, 0.5854101966249685, 0.1381966011250105, 0.1381966011250105}},
        {{0.1381966011250105, 0.1381966011250105, 0.5854101966249685, 0.1381966011250105}},
        {{0.1381966011250105, 0.1381966011250105, 0.1381966011250105, 0.5854101966249685}}
    }};

    for (int c = 0; c < dm.nCells; ++c) {
        const auto& K = geom[c];

        for (int i = 0; i < 4; ++i) {
            int row = 4*c + i;

            for (int j = 0; j < 4; ++j) {
                int col = 4*c + j;
                sparse_add(sys.A, row, col, Real(K.volume * dot3(K.grad[i], K.grad[j])));
            }
        }

        for (const auto& lam : tetQ) {
            Vec3 xq;
            for (int a = 0; a < 4; ++a) {
                xq = dg_add(xq, dg_mul(lam[a], K.x[a]));
            }

            double fval = mms_rhs_value(xq, mms);
            double w = K.volume / 4.0;

            for (int i = 0; i < 4; ++i) {
                sys.b[4*c + i] += Real(w * fval * lam[i]);
            }
        }
    }

    const std::array<std::array<double,3>,3> faceQ = {{
        {{2.0/3.0, 1.0/6.0, 1.0/6.0}},
        {{1.0/6.0, 2.0/3.0, 1.0/6.0}},
        {{1.0/6.0, 1.0/6.0, 2.0/3.0}}
    }};

    for (int f = 0; f < (int)m.faces.size(); ++f) {
        int P = m.owner[f];
        bool interior = (f < (int)m.neighbour.size());
        int N = interior ? m.neighbour[f] : -1;

        const auto& KP = geom[P];
        const auto* KN = interior ? &geom[N] : nullptr;

        double area = 0.0;
        Vec3 n = dg_owner_outward_unit_normal(m, f, KP, area);

        double hP = std::max(3.0 * KP.volume / area, 1.0e-300);
        double penalty = dgopt.penaltySigma / hP;

        if (interior) {
            double hN = std::max(3.0 * KN->volume / area, 1.0e-300);
            penalty = dgopt.penaltySigma * std::max(1.0 / hP, 1.0 / hN);
        }

        for (const auto& lam : faceQ) {
            double w = area / 3.0;
            auto phiP = dg_face_phi(m.faces[f], KP, lam);

            if (interior) {
                auto phiN = dg_face_phi(m.faces[f], *KN, lam);

                for (int ti = 0; ti < 8; ++ti) {
                    bool testOwner = (ti < 4);
                    int i = testOwner ? ti : ti - 4;
                    int rowCell = testOwner ? P : N;
                    int row = 4*rowCell + i;

                    double vJump = testOwner ? phiP[i] : -phiN[i];
                    const Vec3& gradV = testOwner ? KP.grad[i] : KN->grad[i];

                    for (int tj = 0; tj < 8; ++tj) {
                        bool trialOwner = (tj < 4);
                        int j = trialOwner ? tj : tj - 4;
                        int colCell = trialOwner ? P : N;
                        int col = 4*colCell + j;

                        double uJump = trialOwner ? phiP[j] : -phiN[j];
                        const Vec3& gradU = trialOwner ? KP.grad[j] : KN->grad[j];

                        double avgGradUdotN = 0.5 * dot3(gradU, n);
                        double avgGradVdotN = 0.5 * dot3(gradV, n);

                        double aij =
                            -avgGradUdotN * vJump
                            -avgGradVdotN * uJump
                            +penalty * uJump * vJump;

                        sparse_add(sys.A, row, col, Real(w * aij));
                    }
                }
            } else if (includeWeakBoundaryTerms) {
                Vec3 xq;
                for (int a = 0; a < 3; ++a) {
                    xq = dg_add(xq, dg_mul(lam[a], m.points[m.faces[f].verts[a]]));
                }

                double g = mms_exact_value(xq, mms);

                for (int i = 0; i < 4; ++i) {
                    int row = 4*P + i;
                    double v = phiP[i];

                    sys.b[row] += Real(w * (-dot3(KP.grad[i], n) * g + penalty * g * v));

                    for (int j = 0; j < 4; ++j) {
                        int col = 4*P + j;
                        double u = phiP[j];

                        double aij =
                            -dot3(KP.grad[j], n) * v
                            -dot3(KP.grad[i], n) * u
                            +penalty * u * v;

                        sparse_add(sys.A, row, col, Real(w * aij));
                    }
                }
            }
        }
    }

    return sys;
}

static ErrorNorms compute_dg_tet_p1_error_norms(
    const PolyMesh& m,
    const DgTetP1DofMap& dm,
    const std::string& mms,
    const std::vector<Real>& x
) {
    if ((int)x.size() != dm.nDofs) {
        throw std::runtime_error("DG error norm size mismatch.");
    }

    const auto geom = build_dg_tet_p1_geoms(m);

    double l2 = 0.0;
    double h1 = 0.0;
    double nodal2 = 0.0;
    double nodalMax = 0.0;

    for (int c = 0; c < dm.nCells; ++c) {
        const auto& K = geom[c];

        Vec3 guh;
        for (int a = 0; a < 4; ++a) {
            double ua = double(x[4*c + a]);
            guh = dg_add(guh, dg_mul(ua, K.grad[a]));

            double eNode = ua - mms_exact_value(K.x[a], mms);
            nodal2 += eNode * eNode;
            nodalMax = std::max(nodalMax, std::abs(eNode));
        }

        for (const auto& qpt : tet_p1_error_quadrature()) {
            auto basis = eval_tet_p1_basis_at(qpt.xi, qpt.eta, qpt.zeta);

            Vec3 xq;
            double uh = 0.0;

            for (int a = 0; a < 4; ++a) {
                xq = dg_add(xq, dg_mul(basis.N[a], K.x[a]));
                uh += basis.N[a] * double(x[4*c + a]);
            }

            double ue = mms_exact_value(xq, mms);
            Vec3 ge = mms_exact_grad(xq, mms);

            double eu = uh - ue;
            Vec3 eg = {guh.x - ge.x, guh.y - ge.y, guh.z - ge.z};

            double detJ = 6.0 * K.volume;
            l2 += qpt.weight * detJ * eu * eu;
            h1 += qpt.weight * detJ * dot3(eg, eg);
        }
    }

    ErrorNorms en;
    en.nodalL2 = std::sqrt(nodal2);
    en.nodalMax = nodalMax;
    en.L2 = std::sqrt(l2);
    en.H1Semi = std::sqrt(h1);
    return en;
}

static void probe_dg_tet_p1_system(
    const PolyMesh& m,
    const DgTetP1DofMap& dm,
    const AssembledSystem& sys,
    int diagLevel
) {
    std::cout << "---------------- DG tet P1 SIPG system probe -----------------\n";
    std::cout << "space                     = " << dm.resolvedSpace << "\n";
    std::cout << "continuity                = DG\n";
    std::cout << "basis                     = cell-local nodal P1 tet\n";
    std::cout << "nCells                    = " << m.cells.size() << "\n";
    std::cout << "nDofs                     = " << dm.nDofs << "\n";
    std::cout << "rows                      = " << sparse_nrows(sys.A) << "\n";
    std::cout << "nnz                       = " << sparse_nnz(sys.A) << "\n";
    std::cout << "sparseMode                = " << memoirs_sparse_mode() << "\n";

    int missingDiag = 0;
    double minDiag = std::numeric_limits<double>::infinity();
    double maxDiag = 0.0;

    for (int i = 0; i < dm.nDofs; ++i) {
        bool present = false;
        double d = sparse_diag_abs_or_zero(sys.A, i, present);
        if (!present) {
            missingDiag++;
        } else {
            minDiag = std::min(minDiag, d);
            maxDiag = std::max(maxDiag, d);
        }
    }

    if (!std::isfinite(minDiag)) minDiag = 0.0;

    std::cout << "missingDiag               = " << missingDiag << "\n";
    std::cout << "diagAbs min/max           = " << minDiag << " / " << maxDiag << "\n";

    if (diagLevel >= 2) {
        std::cout << "symmetryMaxAbs            = " << sparse_symmetry_max_abs(sys.A) << "\n";
    }

    std::cout << "--------------------------------------------------------------\n";
}

// ---------------------------------------------------------------------------
// Optional DG strong/clamped boundary trace.
// ---------------------------------------------------------------------------
//
// This is useful for testing and special cases.  For conservative DG diffusion,
// weak SIPG Dirichlet remains the default.

static StrongDirichletData build_dg_tet_p1_boundary_strong_mms(
    const PolyMesh& m,
    const DgTetP1DofMap& dm,
    const std::string& mms
) {
    (void)dm;

    // Important: DG assembly uses oriented local tet geometry.  Therefore the
    // strong-boundary dof picker must use the same oriented local vertex order,
    // not the raw m.cells[c].verts order.
    const std::vector<DgTetP1CellGeom> geoms = build_dg_tet_p1_geoms(m);

    std::vector<char> mark(4 * (int)m.cells.size(), 0);
    StrongDirichletData bc;

    for (int f = (int)m.neighbour.size(); f < (int)m.faces.size(); ++f) {
        const int c = m.owner[f];
        const DgTetP1CellGeom& K = geoms[c];

        for (int fv : m.faces[f].verts) {
            int local = -1;

            for (int a = 0; a < 4; ++a) {
                if (K.verts[a] == fv) {
                    local = a;
                    break;
                }
            }

            if (local < 0) {
                throw std::runtime_error("DG strong BC face vertex not found in oriented owner cell.");
            }

            const int dof = 4*c + local;

            if (!mark[dof]) {
                mark[dof] = 1;
                bc.dofs.push_back(dof);
                bc.values.push_back(Real(mms_exact_value(m.points[fv], mms)));
            }
        }
    }

    return bc;
}

static AssembledSystem assemble_dg_tet_p1_strong_poisson_mms(
    const PolyMesh& m,
    const DgTetP1DofMap& dm,
    const std::string& mms,
    const DgSipgOptions& dgopt
) {
    AssembledSystem sys = assemble_dg_tet_p1_sipg_poisson_mms(m, dm, mms, dgopt, false);

    StrongDirichletData bc = build_dg_tet_p1_boundary_strong_mms(m, dm, mms);
    apply_strong_dirichlet_by_dof_values(bc, sys.A, sys.b);

    return sys;
}
