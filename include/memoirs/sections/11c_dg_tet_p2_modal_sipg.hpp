#pragma once

// ============================================================================
// DG tet P2 modal modal SIPG Poisson operator.
// ============================================================================
// This is the nodal quadratic DG counterpart of 11_dg_tet_p1_sipg.hpp.
// Geometry is still affine P1 on straight-sided tet polyMeshes.  The local
// solution basis is P2 with local order:
//   0..3: vertices lambda0..lambda3
//   4: edge 0-1
//   5: edge 0-2
//   6: edge 0-3
//   7: edge 1-2
//   8: edge 1-3
//   9: edge 2-3
// All dofs are cell-local: global dof = 10*cell + local.
// ============================================================================

struct DgTetP2ModalDofMap {
    std::string resolvedSpace = "dg_tet_p2_modal";
    int nCells = 0;
    int nDofs = 0;
};

struct DgTetP2ModalCellGeom {
    std::array<int,4> verts{};
    std::array<Vec3,4> x{};
    Mat3 J{};
    Mat3 invJ{};
    double detJabs = 0.0;
    double volume = 0.0;
    Vec3 centroid{};
};

static DgTetP2ModalDofMap build_dg_tet_p2_modal_dof_map(const PolyMesh& m) {
    MeshCellCounts cc = count_cell_types(m);
    if (cc.nTet != (int)m.cells.size() || cc.nHex != 0 || cc.nOther != 0) {
        throw std::runtime_error("dg_tet_p2_modal requires a pure tet mesh.");
    }

    DgTetP2ModalDofMap dm;
    dm.nCells = (int)m.cells.size();
    dm.nDofs = 10 * dm.nCells;
    return dm;
}

static DgTetP2ModalCellGeom dg_tet_p2_modal_build_geom(const PolyMesh& m, const Cell& cell) {
    if (cell.verts.size() != 4) {
        throw std::runtime_error("DG tet P2 modal found non-tet cell.");
    }

    DgTetP2ModalCellGeom K;
    for (int a = 0; a < 4; ++a) {
        K.verts[a] = cell.verts[a];
        K.x[a] = m.points[K.verts[a]];
    }

    K.J = jacobian_tet_p1(m, K.verts);
    const double detJ = det3(K.J);
    K.detJabs = std::abs(detJ);
    if (!(K.detJabs > 0.0)) {
        throw std::runtime_error("DG tet P2 modal found degenerate tet.");
    }
    K.invJ = inv3(K.J);
    K.volume = K.detJabs / 6.0;
    K.centroid = dg_mul(
        0.25,
        dg_add(dg_add(K.x[0], K.x[1]), dg_add(K.x[2], K.x[3]))
    );
    return K;
}

static std::vector<DgTetP2ModalCellGeom> build_dg_tet_p2_modal_geoms(const PolyMesh& m) {
    std::vector<DgTetP2ModalCellGeom> geoms(m.cells.size());
    for (int c = 0; c < (int)m.cells.size(); ++c) {
        geoms[c] = dg_tet_p2_modal_build_geom(m, m.cells[c]);
    }
    return geoms;
}

static int dg_tet_p2_modal_local_index_for_vertex(const DgTetP2ModalCellGeom& K, int v) {
    for (int a = 0; a < 4; ++a) {
        if (K.verts[a] == v) return a;
    }
    return -1;
}

static TetP2BasisAtPoint dg_tet_p2_modal_face_basis(
    const Face& f,
    const DgTetP2ModalCellGeom& K,
    double lam0,
    double lam1,
    double lam2
) {
    if (f.verts.size() != 3) {
        throw std::runtime_error("DG tet P2 modal expected triangular face.");
    }

    std::array<double,4> L = {0.0, 0.0, 0.0, 0.0};
    const double lf[3] = {lam0, lam1, lam2};

    for (int q = 0; q < 3; ++q) {
        const int li = dg_tet_p2_modal_local_index_for_vertex(K, f.verts[q]);
        if (li < 0) {
            throw std::runtime_error("DG tet P2 modal face vertex not found in cell.");
        }
        L[li] = lf[q];
    }

    return eval_tet_p2_modal_basis_at(L[1], L[2], L[3]);
}

static Vec3 dg_tet_p2_modal_owner_outward_unit_normal(
    const PolyMesh& m,
    int faceId,
    const DgTetP2ModalCellGeom& ownerGeom,
    double& area
) {
    const Face& f = m.faces[faceId];
    if (f.verts.size() != 3) {
        throw std::runtime_error("DG tet P2 modal expected triangular face.");
    }

    const Vec3& p0 = m.points[f.verts[0]];
    const Vec3& p1 = m.points[f.verts[1]];
    const Vec3& p2 = m.points[f.verts[2]];

    Vec3 n2 = cross3(p1 - p0, p2 - p0);
    const double mag = dg_norm(n2);
    if (!(mag > 0.0)) {
        throw std::runtime_error("DG tet P2 modal degenerate triangular face.");
    }

    area = 0.5 * mag;
    Vec3 n = dg_mul(1.0 / mag, n2);

    Vec3 fc = dg_face_centroid(m, f);
    if (dot3(n, fc - ownerGeom.centroid) < 0.0) {
        n = dg_mul(-1.0, n);
    }

    return n;
}

static Vec3 dg_tet_p2_modal_node_coordinate(const DgTetP2ModalCellGeom& K, int a) {
    if (a < 4) return K.x[a];
    static const int e[6][2] = {
        {0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3}
    };
    const int i = e[a - 4][0];
    const int j = e[a - 4][1];
    return dg_mul(0.5, dg_add(K.x[i], K.x[j]));
}

static std::vector<std::vector<int>> build_dg_tet_p2_modal_sipg_pattern(
    const PolyMesh& m,
    const DgTetP2ModalDofMap& dm
) {
    std::vector<std::vector<int>> rowCols(dm.nDofs);

    auto add_block = [&](int cRow, int cCol) {
        for (int i = 0; i < 10; ++i) {
            auto& cols = rowCols[10*cRow + i];
            for (int j = 0; j < 10; ++j) {
                cols.push_back(10*cCol + j);
            }
        }
    };

    for (int c = 0; c < dm.nCells; ++c) {
        add_block(c, c);
    }

    for (int f = 0; f < (int)m.neighbour.size(); ++f) {
        const int P = m.owner[f];
        const int N = m.neighbour[f];
        add_block(P, N);
        add_block(N, P);
    }

    return rowCols;
}

static AssembledSystem assemble_dg_tet_p2_modal_sipg_poisson_mms(
    const PolyMesh& m,
    const DgTetP2ModalDofMap& dm,
    const std::string& mms,
    const DgSipgOptions& dgopt,
    const ScalarEllipticSpec& spec,
    bool includeWeakBoundaryTerms = true
) {
    AssembledSystem sys;
    sys.b.assign(dm.nDofs, Real(0));

    const std::string sparseMode = memoirs_sparse_mode();
    if (sparseMode == "fixed_csr") {
        sparse_init_fixed_pattern(sys.A, build_dg_tet_p2_modal_sipg_pattern(m, dm));
    } else if (sparseMode == "legacy") {
        sys.A.rows.resize(dm.nDofs);
    } else {
        throw std::runtime_error("Unsupported DG tet P2 modal sparse mode: " + sparseMode);
    }

    const auto geom = build_dg_tet_p2_modal_geoms(m);

    // Volume terms.
    for (int c = 0; c < dm.nCells; ++c) {
        const auto& K = geom[c];

        for (const auto& qpt : tet_volume_quadrature_selected()) {
            const TetP2BasisAtPoint basis = eval_tet_p2_modal_basis_at(qpt.xi, qpt.eta, qpt.zeta);
            const TetP1BasisAtPoint geomBasis = eval_tet_p1_basis_at(qpt.xi, qpt.eta, qpt.zeta);
            const Vec3 xq = map_tet_p1_to_physical(m, K.verts, geomBasis.N);

            const Real kappa = spec.diffusion.value_in_cell(c, xq);
            const double fval = mms_rhs_value(xq, mms);
            const double wdet = qpt.weight * K.detJabs;

            std::array<Vec3,10> grad{};
            for (int a = 0; a < 10; ++a) {
                grad[a] = invJT_mul(K.invJ, basis.dNref[a]);
            }

            for (int i = 0; i < 10; ++i) {
                const int row = 10*c + i;
                sys.b[row] += Real(wdet) * kappa * Real(fval * basis.N[i]);

                for (int j = 0; j < 10; ++j) {
                    const int col = 10*c + j;
                    sparse_add(sys.A, row, col, Real(wdet) * kappa * Real(dot3(grad[i], grad[j])));
                }
            }
        }
    }

    // SIPG interior and weak Dirichlet boundary terms.
    for (int f = 0; f < (int)m.faces.size(); ++f) {
        const int P = m.owner[f];
        const bool interior = (f < (int)m.neighbour.size());
        const int N = interior ? m.neighbour[f] : -1;

        const auto& KP = geom[P];
        const auto* KN = interior ? &geom[N] : nullptr;

        double area = 0.0;
        const Vec3 n = dg_tet_p2_modal_owner_outward_unit_normal(m, f, KP, area);

        const double hP = std::max(3.0 * KP.volume / area, 1.0e-300);
        Real kP = spec.diffusion.face_value(P, -1);
        Real kN = kP;
        double penalty = dgopt.penaltySigma * double(kP) / hP;

        if (interior) {
            const double hN = std::max(3.0 * KN->volume / area, 1.0e-300);
            kN = spec.diffusion.face_value(P, N);
            const double kmax = std::max(double(kP), double(kN));
            penalty = dgopt.penaltySigma * kmax * std::max(1.0 / hP, 1.0 / hN);
        }

        for (const auto& fq : triangle_face_quadrature_selected()) {
            const double lam0 = 1.0 - fq.r - fq.s;
            const double lam1 = fq.r;
            const double lam2 = fq.s;
            const double w = 2.0 * area * fq.weight;

            const TetP2BasisAtPoint phiP = dg_tet_p2_modal_face_basis(m.faces[f], KP, lam0, lam1, lam2);
            std::array<Vec3,10> gradP{};
            for (int a = 0; a < 10; ++a) {
                gradP[a] = invJT_mul(KP.invJ, phiP.dNref[a]);
            }

            if (interior) {
                const TetP2BasisAtPoint phiN = dg_tet_p2_modal_face_basis(m.faces[f], *KN, lam0, lam1, lam2);
                std::array<Vec3,10> gradN{};
                for (int a = 0; a < 10; ++a) {
                    gradN[a] = invJT_mul(KN->invJ, phiN.dNref[a]);
                }

                for (int ti = 0; ti < 20; ++ti) {
                    const bool testOwner = (ti < 10);
                    const int i = testOwner ? ti : ti - 10;
                    const int rowCell = testOwner ? P : N;
                    const int row = 10*rowCell + i;

                    const double vJump = testOwner ? phiP.N[i] : -phiN.N[i];
                    const Vec3& gradV = testOwner ? gradP[i] : gradN[i];
                    const double kV = testOwner ? double(kP) : double(kN);

                    for (int tj = 0; tj < 20; ++tj) {
                        const bool trialOwner = (tj < 10);
                        const int j = trialOwner ? tj : tj - 10;
                        const int colCell = trialOwner ? P : N;
                        const int col = 10*colCell + j;

                        const double uJump = trialOwner ? phiP.N[j] : -phiN.N[j];
                        const Vec3& gradU = trialOwner ? gradP[j] : gradN[j];
                        const double kU = trialOwner ? double(kP) : double(kN);

                        const double avgFluxU = 0.5 * kU * dot3(gradU, n);
                        const double avgFluxV = 0.5 * kV * dot3(gradV, n);

                        const double aij =
                            - avgFluxU * vJump
                            - avgFluxV * uJump
                            + penalty * uJump * vJump;

                        sparse_add(sys.A, row, col, Real(w * aij));
                    }
                }
            } else if (includeWeakBoundaryTerms) {
                Vec3 xq;
                xq = dg_add(xq, dg_mul(lam0, m.points[m.faces[f].verts[0]]));
                xq = dg_add(xq, dg_mul(lam1, m.points[m.faces[f].verts[1]]));
                xq = dg_add(xq, dg_mul(lam2, m.points[m.faces[f].verts[2]]));
                const double g = mms_exact_value(xq, mms);

                for (int i = 0; i < 10; ++i) {
                    const int row = 10*P + i;
                    const double v = phiP.N[i];
                    const double dn_v = dot3(gradP[i], n);

                    sys.b[row] += Real(w * (-double(kP) * dn_v * g + penalty * g * v));

                    for (int j = 0; j < 10; ++j) {
                        const int col = 10*P + j;
                        const double u = phiP.N[j];
                        const double dn_u = dot3(gradP[j], n);

                        const double aij =
                            - double(kP) * dn_u * v
                            - double(kP) * dn_v * u
                            + penalty * u * v;

                        sparse_add(sys.A, row, col, Real(w * aij));
                    }
                }
            }
        }
    }

    return sys;
}

static StrongDirichletData build_dg_tet_p2_modal_boundary_strong_mms(
    const PolyMesh&,
    const DgTetP2ModalDofMap&,
    const std::string&
) {
    throw std::runtime_error("dg_tet_p2_modal uses weak SIPG/Nitsche BCs; strong modal boundary elimination is not implemented.");
}

static AssembledSystem assemble_dg_tet_p2_modal_strong_poisson_mms(
    const PolyMesh& m,
    const DgTetP2ModalDofMap& dm,
    const std::string& mms,
    const DgSipgOptions& dgopt,
    const ScalarEllipticSpec& spec
) {
    AssembledSystem sys = assemble_dg_tet_p2_modal_sipg_poisson_mms(m, dm, mms, dgopt, spec, false);
    StrongDirichletData bc = build_dg_tet_p2_modal_boundary_strong_mms(m, dm, mms);
    apply_strong_dirichlet_by_dof_values(bc, sys.A, sys.b);
    return sys;
}

static ErrorNorms compute_dg_tet_p2_modal_error_norms(
    const PolyMesh& m,
    const DgTetP2ModalDofMap& dm,
    const std::string& mms,
    const std::vector<Real>& x
) {
    if ((int)x.size() != dm.nDofs) {
        throw std::runtime_error("DG tet P2 modal error norm size mismatch.");
    }

    const auto geom = build_dg_tet_p2_modal_geoms(m);

    double l2 = 0.0;
    double h1 = 0.0;
    double nodal2 = 0.0;
    double nodalMax = 0.0;

    for (int c = 0; c < dm.nCells; ++c) {
        const auto& K = geom[c];

        // Modal coefficients are not nodal values.  Reuse the existing
        // report fields as coefficient diagnostics only.
        for (int a = 0; a < 10; ++a) {
            const double ca = double(x[10*c + a]);
            nodal2 += ca * ca;
            nodalMax = std::max(nodalMax, std::abs(ca));
        }

        for (const auto& qpt : tet_volume_quadrature_selected_for_error()) {
            const TetP2BasisAtPoint basis = eval_tet_p2_modal_basis_at(qpt.xi, qpt.eta, qpt.zeta);
            const TetP1BasisAtPoint geomBasis = eval_tet_p1_basis_at(qpt.xi, qpt.eta, qpt.zeta);
            const Vec3 xq = map_tet_p1_to_physical(m, K.verts, geomBasis.N);

            std::array<Vec3,10> grad{};
            for (int a = 0; a < 10; ++a) {
                grad[a] = invJT_mul(K.invJ, basis.dNref[a]);
            }

            double uh = 0.0;
            Vec3 guh;
            for (int a = 0; a < 10; ++a) {
                const double ua = double(x[10*c + a]);
                uh += basis.N[a] * ua;
                guh = dg_add(guh, dg_mul(ua, grad[a]));
            }

            const double ue = mms_exact_value(xq, mms);
            const Vec3 ge = mms_exact_grad(xq, mms);
            const double eu = uh - ue;
            const Vec3 eg = {guh.x - ge.x, guh.y - ge.y, guh.z - ge.z};

            const double wdet = qpt.weight * K.detJabs;
            l2 += wdet * eu * eu;
            h1 += wdet * dot3(eg, eg);
        }
    }

    ErrorNorms en;
    en.nodalL2 = std::sqrt(nodal2);
    en.nodalMax = nodalMax;
    en.L2 = std::sqrt(l2);
    en.H1Semi = std::sqrt(h1);
    return en;
}

static void probe_dg_tet_p2_modal_system(
    const PolyMesh& m,
    const DgTetP2ModalDofMap& dm,
    const AssembledSystem& sys,
    int diagLevel
) {
    std::cout << "---------------- DG tet P2 modal modal SIPG system probe -----------------\n";
    std::cout << "space                     = " << dm.resolvedSpace << "\n";
    std::cout << "continuity                = DG\n";
    std::cout << "basis                     = cell-local modal P2 tet\n";
    std::cout << "nCells                    = " << m.cells.size() << "\n";
    std::cout << "nDofs                     = " << dm.nDofs << "\n";
    std::cout << "dofsPerCell               = 10\n";
    std::cout << "rows                      = " << sparse_nrows(sys.A) << "\n";
    std::cout << "nnz                       = " << sparse_nnz(sys.A) << "\n";
    std::cout << "sparseMode                = " << memoirs_sparse_mode() << "\n";

    int missingDiag = 0;
    double minDiag = std::numeric_limits<double>::infinity();
    double maxDiag = 0.0;
    for (int i = 0; i < dm.nDofs; ++i) {
        bool present = false;
        double d = sparse_diag_abs_or_zero(sys.A, i, present);
        if (!present) missingDiag++;
        else {
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
