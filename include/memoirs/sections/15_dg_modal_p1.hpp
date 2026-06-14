#pragma once

// ============================================================================
// SECTION 15: Orthogonal modal DG P1 scalar Poisson, tet + hex
// ============================================================================
//
// Cell-local modal P1 space:
//     span{1, x-cx, y-cy, z-cz}
//
// The 4 modes are L2-orthonormalized per cell by Gram-Schmidt using volume
// quadrature.  This gives a true local orthogonal modal basis while keeping
// gradients simple and constant.
//
// Supported now:
//   - pure tet mesh: 4 modes/cell
//   - pure hex mesh: 4 modes/cell
//   - weak SIPG Dirichlet MMS
//
// Strong/clamped BC is intentionally not provided for modal DG because modal
// dofs are not trace/nodal values.
// ============================================================================

enum class DgModalCellKind {
    Tet,
    Hex
};

struct DgModalP1DofMap {
    std::string resolvedSpace = "dg_modal_p1";
    DgModalCellKind kind = DgModalCellKind::Tet;
    int nCells = 0;
    int nDofs = 0;
    int modesPerCell = 4;
};

struct DgModalVolumeQ {
    Vec3 x{};
    double w = 0.0;
};

struct DgModalP1CellGeom {
    DgModalCellKind kind = DgModalCellKind::Tet;
    std::vector<int> verts;
    Vec3 centroid{};
    double volume = 0.0;

    // mode m is:
    //   coeff[m][0] + coeff[m][1]*(x-cx) + coeff[m][2]*(y-cy) + coeff[m][3]*(z-cz)
    std::array<std::array<double,4>,4> coeff{};
    std::array<Vec3,4> grad{};
};

struct DgModalSipgOptions {
    double penaltySigma = 10.0;
};

static inline Vec3 dgmodal_add(const Vec3& a, const Vec3& b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

static inline Vec3 dgmodal_mul(double s, const Vec3& a) {
    return {s*a.x, s*a.y, s*a.z};
}

static inline double dgmodal_norm(const Vec3& a) {
    return std::sqrt(dot3(a,a));
}

static DgModalP1DofMap build_dg_modal_p1_dof_map(const PolyMesh& m) {
    MeshCellCounts cc = count_cell_types(m);

    DgModalP1DofMap dm;
    dm.nCells = (int)m.cells.size();
    dm.nDofs = 4 * dm.nCells;

    if (cc.nTet == dm.nCells && cc.nHex == 0 && cc.nOther == 0) {
        dm.kind = DgModalCellKind::Tet;
        dm.resolvedSpace = "dg_tet_p1_modal";
    } else if (cc.nHex == dm.nCells && cc.nTet == 0 && cc.nOther == 0) {
        dm.kind = DgModalCellKind::Hex;
        dm.resolvedSpace = "dg_hex_p1_modal";
    } else {
        throw std::runtime_error("dg_modal_p1 requires a pure tet or pure hex mesh.");
    }

    return dm;
}

static std::vector<DgModalVolumeQ> dgmodal_tet_volume_quadrature(
    const PolyMesh& m,
    const std::vector<int>& verts,
    bool errorQuadrature
) {
    std::vector<DgModalVolumeQ> out;

    const Vec3& x0 = m.points[verts[0]];
    const Vec3& x1 = m.points[verts[1]];
    const Vec3& x2 = m.points[verts[2]];
    const Vec3& x3 = m.points[verts[3]];

    const double volume = std::abs(tet_signed_volume6(x0, x1, x2, x3)) / 6.0;
    const auto& refQs = errorQuadrature
        ? tet_volume_quadrature_selected_for_error()
        : tet_volume_quadrature_selected();

    for (const auto& tq : refQs) {
        const auto b = eval_tet_p1_basis_at(tq.xi, tq.eta, tq.zeta);

        DgModalVolumeQ q;
        q.x = dgmodal_add(
            dgmodal_add(dgmodal_mul(b.N[0], x0), dgmodal_mul(b.N[1], x1)),
            dgmodal_add(dgmodal_mul(b.N[2], x2), dgmodal_mul(b.N[3], x3))
        );

        // Reference tet weights integrate over volume 1/6, so scale by 6*|K|.
        q.w = tq.weight * 6.0 * volume;
        out.push_back(q);
    }

    return out;
}

static std::vector<DgModalVolumeQ> dgmodal_hex_volume_quadrature(
    const PolyMesh& m,
    const std::vector<int>& vertsVec,
    bool errorQuadrature
) {
    std::array<int,8> verts{};
    for (int i = 0; i < 8; ++i) verts[i] = vertsVec[i];

    std::vector<DgModalVolumeQ> out;

    const auto& qs = errorQuadrature
        ? hex_q1_error_quadrature_selected()
        : hex_q1_volume_quadrature_selected();

    for (const auto& hq : qs) {
        const auto basis = eval_hex_q1_basis_at(hq.xi, hq.eta, hq.zeta);
        const Vec3 xq = map_hex_q1_to_physical(m, verts, basis.N);
        const Mat3 J = jacobian_hex_q1(m, verts, basis.dNref);
        const double detJ = det3(J);

        if (!(detJ > 0.0)) {
            throw std::runtime_error("dg_modal_p1 hex non-positive detJ.");
        }

        DgModalVolumeQ q;
        q.x = xq;
        q.w = hq.weight * detJ;
        out.push_back(q);
    }

    return out;
}

static std::vector<DgModalVolumeQ> dgmodal_cell_volume_quadrature(
    const PolyMesh& m,
    const DgModalP1CellGeom& K,
    bool errorQuadrature
) {
    if (K.kind == DgModalCellKind::Tet) {
        return dgmodal_tet_volume_quadrature(m, K.verts, errorQuadrature);
    }

    return dgmodal_hex_volume_quadrature(m, K.verts, errorQuadrature);
}

static double dgmodal_eval_monomial(int k, const Vec3& x, const Vec3& c) {
    if (k == 0) return 1.0;
    if (k == 1) return x.x - c.x;
    if (k == 2) return x.y - c.y;
    return x.z - c.z;
}

static double dgmodal_eval_coeff(
    const std::array<double,4>& coeff,
    const Vec3& x,
    const Vec3& c
) {
    return coeff[0]
         + coeff[1] * (x.x - c.x)
         + coeff[2] * (x.y - c.y)
         + coeff[3] * (x.z - c.z);
}

static void dgmodal_orthonormalize_cell(
    const PolyMesh& m,
    DgModalP1CellGeom& K
) {
    auto qs = dgmodal_cell_volume_quadrature(m, K, false);

    K.volume = 0.0;
    K.centroid = Vec3{};

    for (const auto& q : qs) {
        K.volume += q.w;
        K.centroid = dgmodal_add(K.centroid, dgmodal_mul(q.w, q.x));
    }

    if (!(K.volume > 0.0)) {
        throw std::runtime_error("dg_modal_p1 non-positive cell volume.");
    }

    K.centroid = dgmodal_mul(1.0 / K.volume, K.centroid);

    for (int mMode = 0; mMode < 4; ++mMode) {
        std::array<double,4> c = {0.0, 0.0, 0.0, 0.0};
        c[mMode] = 1.0;

        for (int prev = 0; prev < mMode; ++prev) {
            double ip = 0.0;

            for (const auto& q : qs) {
                const double v = dgmodal_eval_coeff(c, q.x, K.centroid);
                const double p = dgmodal_eval_coeff(K.coeff[prev], q.x, K.centroid);
                ip += q.w * v * p;
            }

            for (int k = 0; k < 4; ++k) {
                c[k] -= ip * K.coeff[prev][k];
            }
        }

        double n2 = 0.0;
        for (const auto& q : qs) {
            const double v = dgmodal_eval_coeff(c, q.x, K.centroid);
            n2 += q.w * v * v;
        }

        if (!(n2 > 0.0)) {
            throw std::runtime_error("dg_modal_p1 Gram-Schmidt zero norm.");
        }

        const double invn = 1.0 / std::sqrt(n2);
        for (int k = 0; k < 4; ++k) {
            K.coeff[mMode][k] = invn * c[k];
        }

        K.grad[mMode] = {
            K.coeff[mMode][1],
            K.coeff[mMode][2],
            K.coeff[mMode][3]
        };
    }
}

static std::vector<DgModalP1CellGeom> build_dg_modal_p1_geoms(
    const PolyMesh& m,
    const DgModalP1DofMap& dm
) {
    std::vector<DgModalP1CellGeom> geoms(m.cells.size());

    for (int c = 0; c < (int)m.cells.size(); ++c) {
        DgModalP1CellGeom K;
        K.kind = dm.kind;

        if (dm.kind == DgModalCellKind::Tet) {
            if (m.cells[c].verts.size() != 4) {
                throw std::runtime_error("dg_modal_p1 expected tet cell.");
            }
            K.verts.assign(m.cells[c].verts.begin(), m.cells[c].verts.end());
        } else {
            if (m.cells[c].verts.size() != 8) {
                throw std::runtime_error("dg_modal_p1 expected hex cell.");
            }
            const auto ov = ordered_hex_vertices_axis_aligned(m, m.cells[c]);
            K.verts.assign(ov.begin(), ov.end());
        }

        dgmodal_orthonormalize_cell(m, K);
        geoms[c] = K;
    }

    return geoms;
}

static std::vector<std::vector<int>> build_dg_modal_p1_sipg_pattern(
    const PolyMesh& m,
    const DgModalP1DofMap& dm
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
        const int P = m.owner[f];
        const int N = m.neighbour[f];
        add_block(P, N);
        add_block(N, P);
    }

    return rowCols;
}

static Vec3 dgmodal_face_centroid(const PolyMesh& m, const Face& f) {
    Vec3 c;
    for (int v : f.verts) c = dgmodal_add(c, m.points[v]);
    return dgmodal_mul(1.0 / double(f.verts.size()), c);
}

static Vec3 dgmodal_owner_outward_unit_normal(
    const PolyMesh& m,
    int faceId,
    const DgModalP1CellGeom& ownerGeom,
    double& area
) {
    const Face& f = m.faces[faceId];

    if (f.verts.size() == 3) {
        const Vec3& p0 = m.points[f.verts[0]];
        const Vec3& p1 = m.points[f.verts[1]];
        const Vec3& p2 = m.points[f.verts[2]];

        Vec3 nA = dgmodal_mul(0.5, cross3(p1 - p0, p2 - p0));
        area = dgmodal_norm(nA);

        if (!(area > 0.0)) {
            throw std::runtime_error("dg_modal_p1 degenerate triangular face.");
        }

        Vec3 n = dgmodal_mul(1.0 / area, nA);
        const Vec3 fc = dgmodal_face_centroid(m, f);

        if (dot3(n, fc - ownerGeom.centroid) < 0.0) {
            n = dgmodal_mul(-1.0, n);
        }

        return n;
    }

    if (f.verts.size() == 4) {
        const Vec3& p0 = m.points[f.verts[0]];
        const Vec3& p1 = m.points[f.verts[1]];
        const Vec3& p2 = m.points[f.verts[2]];
        const Vec3& p3 = m.points[f.verts[3]];

        Vec3 nA1 = cross3(p1 - p0, p2 - p0);
        Vec3 nA2 = cross3(p2 - p0, p3 - p0);
        Vec3 nA = dgmodal_add(dgmodal_mul(0.5, nA1), dgmodal_mul(0.5, nA2));

        area = dgmodal_norm(nA);

        if (!(area > 0.0)) {
            throw std::runtime_error("dg_modal_p1 degenerate quadrilateral face.");
        }

        Vec3 n = dgmodal_mul(1.0 / area, nA);
        const Vec3 fc = dgmodal_face_centroid(m, f);

        if (dot3(n, fc - ownerGeom.centroid) < 0.0) {
            n = dgmodal_mul(-1.0, n);
        }

        return n;
    }

    throw std::runtime_error("dg_modal_p1 supports tri/quad faces only.");
}

struct DgModalFaceQ {
    Vec3 x{};
    double w = 0.0;
};

static std::vector<DgModalFaceQ> dgmodal_face_quadrature(
    const PolyMesh& m,
    const Face& f
) {
    std::vector<DgModalFaceQ> out;

    if (f.verts.size() == 3) {
        const Vec3& p0 = m.points[f.verts[0]];
        const Vec3& p1 = m.points[f.verts[1]];
        const Vec3& p2 = m.points[f.verts[2]];

        const Vec3 e1 = p1 - p0;
        const Vec3 e2 = p2 - p0;
        const double jac = dgmodal_norm(cross3(e1, e2));

        for (const auto& tq : triangle_face_quadrature_selected()) {
            DgModalFaceQ q;
            q.x = dgmodal_add(
                p0,
                dgmodal_add(dgmodal_mul(tq.r, e1), dgmodal_mul(tq.s, e2))
            );
            q.w = tq.weight * jac;
            out.push_back(q);
        }

        return out;
    }

    if (f.verts.size() == 4) {
        const Vec3& p0 = m.points[f.verts[0]];
        const Vec3& p1 = m.points[f.verts[1]];
        const Vec3& p2 = m.points[f.verts[2]];
        const Vec3& p3 = m.points[f.verts[3]];

        for (const auto& qq : quad_face_quadrature_selected()) {
            const double r = qq.r;
            const double s = qq.s;

            const double N0 = 0.25 * (1.0 - r) * (1.0 - s);
            const double N1 = 0.25 * (1.0 + r) * (1.0 - s);
            const double N2 = 0.25 * (1.0 + r) * (1.0 + s);
            const double N3 = 0.25 * (1.0 - r) * (1.0 + s);

            const Vec3 x = dgmodal_add(
                dgmodal_add(dgmodal_mul(N0, p0), dgmodal_mul(N1, p1)),
                dgmodal_add(dgmodal_mul(N2, p2), dgmodal_mul(N3, p3))
            );

            const Vec3 dxdr = dgmodal_add(
                dgmodal_add(dgmodal_mul(-0.25*(1.0-s), p0), dgmodal_mul( 0.25*(1.0-s), p1)),
                dgmodal_add(dgmodal_mul( 0.25*(1.0+s), p2), dgmodal_mul(-0.25*(1.0+s), p3))
            );

            const Vec3 dxds = dgmodal_add(
                dgmodal_add(dgmodal_mul(-0.25*(1.0-r), p0), dgmodal_mul(-0.25*(1.0+r), p1)),
                dgmodal_add(dgmodal_mul( 0.25*(1.0+r), p2), dgmodal_mul( 0.25*(1.0-r), p3))
            );

            DgModalFaceQ q;
            q.x = x;
            q.w = qq.weight * dgmodal_norm(cross3(dxdr, dxds));
            out.push_back(q);
        }

        return out;
    }

    throw std::runtime_error("dg_modal_p1 supports tri/quad face quadrature only.");
}

static std::array<double,4> dgmodal_eval_modes(
    const DgModalP1CellGeom& K,
    const Vec3& x
) {
    std::array<double,4> phi{};

    for (int m = 0; m < 4; ++m) {
        phi[m] = dgmodal_eval_coeff(K.coeff[m], x, K.centroid);
    }

    return phi;
}

static AssembledSystem assemble_dg_modal_p1_sipg_poisson_mms(
    const PolyMesh& m,
    const DgModalP1DofMap& dm,
    const std::string& mms,
    const DgModalSipgOptions& opt
) {
    AssembledSystem sys;
    sys.b.assign(dm.nDofs, Real(0));

    const std::string sparseMode = memoirs_sparse_mode();

    if (sparseMode == "fixed_csr") {
        sparse_init_fixed_pattern(sys.A, build_dg_modal_p1_sipg_pattern(m, dm));
    } else if (sparseMode == "legacy") {
        sys.A.rows.resize(dm.nDofs);
    } else {
        throw std::runtime_error("Unsupported dg_modal_p1 sparse mode: " + sparseMode);
    }

    const auto geoms = build_dg_modal_p1_geoms(m, dm);

    for (int c = 0; c < dm.nCells; ++c) {
        const auto& K = geoms[c];

        for (int i = 0; i < 4; ++i) {
            const int row = 4*c + i;

            for (int j = 0; j < 4; ++j) {
                const int col = 4*c + j;
                sparse_add(sys.A, row, col, Real(K.volume * dot3(K.grad[i], K.grad[j])));
            }
        }

        for (const auto& q : dgmodal_cell_volume_quadrature(m, K, false)) {
            const auto phi = dgmodal_eval_modes(K, q.x);
            const double fval = mms_rhs_value(q.x, mms);

            for (int i = 0; i < 4; ++i) {
                sys.b[4*c + i] += Real(q.w * fval * phi[i]);
            }
        }
    }

    for (int f = 0; f < (int)m.faces.size(); ++f) {
        const int P = m.owner[f];
        const bool interior = (f < (int)m.neighbour.size());
        const int N = interior ? m.neighbour[f] : -1;

        const auto& KP = geoms[P];
        const auto* KN = interior ? &geoms[N] : nullptr;

        double area = 0.0;
        const Vec3 n = dgmodal_owner_outward_unit_normal(m, f, KP, area);

        const double hP = std::max(KP.volume / area, 1.0e-300);
        double penalty = opt.penaltySigma / hP;

        if (interior) {
            const double hN = std::max(KN->volume / area, 1.0e-300);
            penalty = opt.penaltySigma * std::max(1.0 / hP, 1.0 / hN);
        }

        for (const auto& fq : dgmodal_face_quadrature(m, m.faces[f])) {
            const auto phiP = dgmodal_eval_modes(KP, fq.x);

            if (interior) {
                const auto phiN = dgmodal_eval_modes(*KN, fq.x);

                for (int ti = 0; ti < 8; ++ti) {
                    const bool testOwner = (ti < 4);
                    const int i = testOwner ? ti : ti - 4;
                    const int rowCell = testOwner ? P : N;
                    const int row = 4*rowCell + i;

                    const double vJump = testOwner ? phiP[i] : -phiN[i];
                    const Vec3& gradV = testOwner ? KP.grad[i] : KN->grad[i];

                    for (int tj = 0; tj < 8; ++tj) {
                        const bool trialOwner = (tj < 4);
                        const int j = trialOwner ? tj : tj - 4;
                        const int colCell = trialOwner ? P : N;
                        const int col = 4*colCell + j;

                        const double uJump = trialOwner ? phiP[j] : -phiN[j];
                        const Vec3& gradU = trialOwner ? KP.grad[j] : KN->grad[j];

                        const double avgGradUdotN = 0.5 * dot3(gradU, n);
                        const double avgGradVdotN = 0.5 * dot3(gradV, n);

                        const double aij =
                            - avgGradUdotN * vJump
                            - avgGradVdotN * uJump
                            + penalty * uJump * vJump;

                        sparse_add(sys.A, row, col, Real(fq.w * aij));
                    }
                }
            } else {
                const double g = mms_exact_value(fq.x, mms);

                for (int i = 0; i < 4; ++i) {
                    const int row = 4*P + i;
                    const double v = phiP[i];

                    sys.b[row] += Real(fq.w * (-dot3(KP.grad[i], n) * g + penalty * g * v));

                    for (int j = 0; j < 4; ++j) {
                        const int col = 4*P + j;
                        const double u = phiP[j];

                        const double aij =
                            - dot3(KP.grad[j], n) * v
                            - dot3(KP.grad[i], n) * u
                            + penalty * u * v;

                        sparse_add(sys.A, row, col, Real(fq.w * aij));
                    }
                }
            }
        }
    }

    return sys;
}

static ErrorNorms compute_dg_modal_p1_error_norms(
    const PolyMesh& m,
    const DgModalP1DofMap& dm,
    const std::string& mms,
    const std::vector<Real>& x
) {
    if ((int)x.size() != dm.nDofs) {
        throw std::runtime_error("dg_modal_p1 error norm size mismatch.");
    }

    const auto geoms = build_dg_modal_p1_geoms(m, dm);

    double l2 = 0.0;
    double h1 = 0.0;
    double nodal2 = 0.0;
    double nodalMax = 0.0;

    for (int c = 0; c < dm.nCells; ++c) {
        const auto& K = geoms[c];

        for (int v : K.verts) {
            const Vec3& xv = m.points[v];
            const auto phi = dgmodal_eval_modes(K, xv);

            double uh = 0.0;
            for (int a = 0; a < 4; ++a) {
                uh += double(x[4*c + a]) * phi[a];
            }

            const double e = uh - mms_exact_value(xv, mms);
            nodal2 += e * e;
            nodalMax = std::max(nodalMax, std::abs(e));
        }

        Vec3 guh;
        for (int a = 0; a < 4; ++a) {
            guh = dgmodal_add(guh, dgmodal_mul(double(x[4*c + a]), K.grad[a]));
        }

        for (const auto& q : dgmodal_cell_volume_quadrature(m, K, true)) {
            const auto phi = dgmodal_eval_modes(K, q.x);

            double uh = 0.0;
            for (int a = 0; a < 4; ++a) {
                uh += double(x[4*c + a]) * phi[a];
            }

            const double ue = mms_exact_value(q.x, mms);
            const Vec3 ge = mms_exact_grad(q.x, mms);
            const Vec3 eg = {guh.x - ge.x, guh.y - ge.y, guh.z - ge.z};
            const double eu = uh - ue;

            l2 += q.w * eu * eu;
            h1 += q.w * dot3(eg, eg);
        }
    }

    ErrorNorms en;
    en.nodalL2 = std::sqrt(nodal2);
    en.nodalMax = nodalMax;
    en.L2 = std::sqrt(l2);
    en.H1Semi = std::sqrt(h1);
    return en;
}

static void probe_dg_modal_p1_system(
    const PolyMesh& m,
    const DgModalP1DofMap& dm,
    const AssembledSystem& sys,
    int diagLevel
) {
    std::cout << "---------------- DG modal P1 SIPG system probe -----------------\n";
    std::cout << "space                     = " << dm.resolvedSpace << "\n";
    std::cout << "continuity                = DG\n";
    std::cout << "basis                     = cell-local L2-orthonormal modal P1\n";
    std::cout << "nCells                    = " << m.cells.size() << "\n";
    std::cout << "nDofs                     = " << dm.nDofs << "\n";
    std::cout << "modesPerCell              = " << dm.modesPerCell << "\n";
    std::cout << "rows                      = " << sparse_nrows(sys.A) << "\n";
    std::cout << "nnz                       = " << sparse_nnz(sys.A) << "\n";
    std::cout << "sparseMode                = " << memoirs_sparse_mode() << "\n";

    int missingDiag = 0;
    double minDiag = std::numeric_limits<double>::infinity();
    double maxDiag = 0.0;

    for (int i = 0; i < dm.nDofs; ++i) {
        bool present = false;
        const double d = sparse_diag_abs_or_zero(sys.A, i, present);

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

    std::cout << "---------------------------------------------------------------\n";
}
