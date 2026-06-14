#pragma once

// ============================================================================
// SECTION 13: DG hex Q1 nodal SIPG Poisson operator
// ============================================================================
//
// Cell-local nodal Q1 on axis-aligned/blockMesh-like hexes.
// Dofs are 8 per cell:
//
//   local order follows ordered_hex_vertices_axis_aligned():
//   0 (-,-,-), 1 (+,-,-), 2 (+,+,-), 3 (-,+,-),
//   4 (-,-,+), 5 (+,-,+), 6 (+,+,+), 7 (-,+,+)
//
// Supports:
//   - weak SIPG Dirichlet
//   - optional strong/clamped boundary trace
//
// This is the hex analogue of dg_tet_p1.
// ============================================================================

struct DgHexQ1DofMap {
    std::string resolvedSpace = "dg_hex_q1";
    int nCells = 0;
    int nDofs = 0;
};

struct DgHexQ1CellGeom {
    std::array<int,8> verts{};
    Vec3 centroid{};
    double volumeApprox = 0.0;
};

static inline Vec3 dghex_add(const Vec3& a, const Vec3& b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

static inline Vec3 dghex_mul(double s, const Vec3& a) {
    return {s*a.x, s*a.y, s*a.z};
}

static inline double dghex_norm(const Vec3& a) {
    return std::sqrt(dot3(a,a));
}

static DgHexQ1DofMap build_dg_hex_q1_dof_map(const PolyMesh& m) {
    MeshCellCounts cc = count_cell_types(m);

    if (cc.nHex != (int)m.cells.size() || cc.nTet != 0 || cc.nOther != 0) {
        throw std::runtime_error("dg_hex_q1 requires a pure hex mesh.");
    }

    DgHexQ1DofMap dm;
    dm.nCells = (int)m.cells.size();
    dm.nDofs = 8 * dm.nCells;
    return dm;
}

static std::vector<DgHexQ1CellGeom> build_dg_hex_q1_geoms(const PolyMesh& m) {
    std::vector<DgHexQ1CellGeom> geoms(m.cells.size());

    for (int c = 0; c < (int)m.cells.size(); ++c) {
        DgHexQ1CellGeom g;
        g.verts = ordered_hex_vertices_axis_aligned(m, m.cells[c]);

        for (int a = 0; a < 8; ++a) {
            g.centroid = dghex_add(g.centroid, m.points[g.verts[a]]);
        }
        g.centroid = dghex_mul(1.0 / 8.0, g.centroid);

        double xmin =  std::numeric_limits<double>::infinity();
        double ymin =  std::numeric_limits<double>::infinity();
        double zmin =  std::numeric_limits<double>::infinity();
        double xmax = -std::numeric_limits<double>::infinity();
        double ymax = -std::numeric_limits<double>::infinity();
        double zmax = -std::numeric_limits<double>::infinity();

        for (int a = 0; a < 8; ++a) {
            const Vec3& x = m.points[g.verts[a]];
            xmin = std::min(xmin, x.x); xmax = std::max(xmax, x.x);
            ymin = std::min(ymin, x.y); ymax = std::max(ymax, x.y);
            zmin = std::min(zmin, x.z); zmax = std::max(zmax, x.z);
        }

        g.volumeApprox = std::max(0.0, (xmax-xmin)*(ymax-ymin)*(zmax-zmin));
        geoms[c] = g;
    }

    return geoms;
}

static int dg_hex_local_index_for_vertex(const DgHexQ1CellGeom& K, int v) {
    for (int a = 0; a < 8; ++a) {
        if (K.verts[a] == v) return a;
    }
    return -1;
}

struct DgHexFaceInfo {
    int faceId = -1;
    std::array<int,4> localVerts{};
    int fixedAxis = 0;       // 0 xi, 1 eta, 2 zeta
    double fixedValue = 0.0; // -1 or +1
};

static DgHexFaceInfo dg_hex_identify_local_face(
    const Face& f,
    const DgHexQ1CellGeom& K
) {
    static const std::array<std::array<int,4>,6> faceLocals = {{
        {{0,3,7,4}}, // xi=-1
        {{1,2,6,5}}, // xi=+1
        {{0,1,5,4}}, // eta=-1
        {{3,2,6,7}}, // eta=+1
        {{0,1,2,3}}, // zeta=-1
        {{4,5,6,7}}  // zeta=+1
    }};

    static const int axis[6] = {0,0,1,1,2,2};
    static const double val[6] = {-1,1,-1,1,-1,1};

    std::array<int,4> faceLocal{};
    if (f.verts.size() != 4) {
        throw std::runtime_error("dg_hex_q1 expected quadrilateral face.");
    }

    for (int i = 0; i < 4; ++i) {
        int li = dg_hex_local_index_for_vertex(K, f.verts[i]);
        if (li < 0) {
            throw std::runtime_error("dg_hex_q1 face vertex not found in owner cell.");
        }
        faceLocal[i] = li;
    }

    auto sortedFace = faceLocal;
    std::sort(sortedFace.begin(), sortedFace.end());

    for (int fid = 0; fid < 6; ++fid) {
        auto cand = faceLocals[fid];
        std::sort(cand.begin(), cand.end());

        if (cand == sortedFace) {
            DgHexFaceInfo info;
            info.faceId = fid;
            info.localVerts = faceLocals[fid];
            info.fixedAxis = axis[fid];
            info.fixedValue = val[fid];
            return info;
        }
    }

    throw std::runtime_error("dg_hex_q1 could not identify local face.");
}

static VolumeQuadraturePoint dg_hex_face_to_volume_qp(
    const DgHexFaceInfo& info,
    double r,
    double s,
    double w
) {
    VolumeQuadraturePoint q;
    q.weight = w;

    if (info.fixedAxis == 0) {
        q.xi = info.fixedValue;
        q.eta = r;
        q.zeta = s;
    } else if (info.fixedAxis == 1) {
        q.xi = r;
        q.eta = info.fixedValue;
        q.zeta = s;
    } else {
        q.xi = r;
        q.eta = s;
        q.zeta = info.fixedValue;
    }

    return q;
}

static Vec3 dg_hex_face_centroid(const PolyMesh& m, const Face& f) {
    Vec3 c;
    for (int v : f.verts) c = dghex_add(c, m.points[v]);
    return dghex_mul(1.0 / double(f.verts.size()), c);
}

static Vec3 dg_hex_owner_outward_unit_normal(
    const PolyMesh& m,
    int faceId,
    const DgHexQ1CellGeom& ownerGeom,
    double& area
) {
    const Face& f = m.faces[faceId];

    if (f.verts.size() != 4) {
        throw std::runtime_error("dg_hex_q1 expected quadrilateral face.");
    }

    const Vec3& p0 = m.points[f.verts[0]];
    const Vec3& p1 = m.points[f.verts[1]];
    const Vec3& p2 = m.points[f.verts[2]];
    const Vec3& p3 = m.points[f.verts[3]];

    Vec3 nA1 = cross3(p1 - p0, p2 - p0);
    Vec3 nA2 = cross3(p2 - p0, p3 - p0);
    Vec3 nA = dghex_add(dghex_mul(0.5, nA1), dghex_mul(0.5, nA2));

    area = dghex_norm(nA);

    if (!(area > 0.0)) {
        throw std::runtime_error("dg_hex_q1 degenerate quadrilateral face.");
    }

    Vec3 n = dghex_mul(1.0 / area, nA);

    Vec3 fc = dg_hex_face_centroid(m, f);
    if (dot3(n, fc - ownerGeom.centroid) < 0.0) {
        n = dghex_mul(-1.0, n);
    }

    return n;
}

static std::vector<std::vector<int>> build_dg_hex_q1_sipg_pattern(
    const PolyMesh& m,
    const DgHexQ1DofMap& dm
) {
    std::vector<std::vector<int>> rowCols(dm.nDofs);

    auto add_block = [&](int cRow, int cCol) {
        for (int i = 0; i < 8; ++i) {
            auto& cols = rowCols[8*cRow + i];
            for (int j = 0; j < 8; ++j) {
                cols.push_back(8*cCol + j);
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

static AssembledSystem assemble_dg_hex_q1_sipg_poisson_mms(
    const PolyMesh& m,
    const DgHexQ1DofMap& dm,
    const std::string& mms,
    const DgSipgOptions& dgopt,
    bool includeWeakBoundaryTerms = true
) {
    AssembledSystem sys;
    sys.b.assign(dm.nDofs, Real(0));

    const std::string sparseMode = memoirs_sparse_mode();

    if (sparseMode == "fixed_csr") {
        sparse_init_fixed_pattern(sys.A, build_dg_hex_q1_sipg_pattern(m, dm));
    } else if (sparseMode == "legacy") {
        sys.A.rows.resize(dm.nDofs);
    } else {
        throw std::runtime_error("Unsupported dg_hex_q1 sparse mode: " + sparseMode);
    }

    const auto geoms = build_dg_hex_q1_geoms(m);

    for (int c = 0; c < dm.nCells; ++c) {
        const auto& K = geoms[c];

        for (const auto& q : hex_q1_volume_quadrature_selected()) {
            const auto basis = eval_hex_q1_basis_at(q.xi, q.eta, q.zeta);
            const Vec3 xq = map_hex_q1_to_physical(m, K.verts, basis.N);
            const Mat3 J = jacobian_hex_q1(m, K.verts, basis.dNref);
            const double detJ = det3(J);

            if (!(detJ > 0.0)) {
                throw std::runtime_error("dg_hex_q1 non-positive detJ.");
            }

            const Mat3 invJ = inv3(J);

            std::array<Vec3,8> grad{};
            for (int a = 0; a < 8; ++a) {
                grad[a] = invJT_mul(invJ, basis.dNref[a]);
            }

            const double w = q.weight * detJ;
            const double fval = mms_rhs_value(xq, mms);

            for (int i = 0; i < 8; ++i) {
                const int row = 8*c + i;
                sys.b[row] += Real(w * fval * basis.N[i]);

                for (int j = 0; j < 8; ++j) {
                    const int col = 8*c + j;
                    sparse_add(sys.A, row, col, Real(w * dot3(grad[i], grad[j])));
                }
            }
        }
    }

    for (int f = 0; f < (int)m.faces.size(); ++f) {
        const int P = m.owner[f];
        const bool interior = (f < (int)m.neighbour.size());
        const int N = interior ? m.neighbour[f] : -1;

        const auto& KP = geoms[P];
        const auto* KN = interior ? &geoms[N] : nullptr;

        const DgHexFaceInfo faceP = dg_hex_identify_local_face(m.faces[f], KP);
        DgHexFaceInfo faceN;
        if (interior) {
            faceN = dg_hex_identify_local_face(m.faces[f], *KN);
        }

        double area = 0.0;
        const Vec3 n = dg_hex_owner_outward_unit_normal(m, f, KP, area);

        const double hP = std::max(KP.volumeApprox / area, 1.0e-300);
        double penalty = dgopt.penaltySigma / hP;

        if (interior) {
            const double hN = std::max(KN->volumeApprox / area, 1.0e-300);
            penalty = dgopt.penaltySigma * std::max(1.0 / hP, 1.0 / hN);
        }

        for (const auto& fq : quad_face_quadrature_selected()) {
            const double w = area * 0.25 * fq.weight;

            const auto qP = dg_hex_face_to_volume_qp(faceP, fq.r, fq.s, fq.weight);
            const auto bP = eval_hex_q1_basis_at(qP.xi, qP.eta, qP.zeta);
            const Vec3 xqP = map_hex_q1_to_physical(m, KP.verts, bP.N);
            const Mat3 JP = jacobian_hex_q1(m, KP.verts, bP.dNref);
            const Mat3 invJP = inv3(JP);

            std::array<Vec3,8> gradP{};
            for (int a = 0; a < 8; ++a) {
                gradP[a] = invJT_mul(invJP, bP.dNref[a]);
            }

            if (interior) {
                const auto qN = dg_hex_face_to_volume_qp(faceN, fq.r, fq.s, fq.weight);
                const auto bN = eval_hex_q1_basis_at(qN.xi, qN.eta, qN.zeta);
                const Mat3 JN = jacobian_hex_q1(m, KN->verts, bN.dNref);
                const Mat3 invJN = inv3(JN);

                std::array<Vec3,8> gradN{};
                for (int a = 0; a < 8; ++a) {
                    gradN[a] = invJT_mul(invJN, bN.dNref[a]);
                }

                for (int ti = 0; ti < 16; ++ti) {
                    const bool testOwner = (ti < 8);
                    const int i = testOwner ? ti : ti - 8;
                    const int rowCell = testOwner ? P : N;
                    const int row = 8*rowCell + i;

                    const double vJump = testOwner ? bP.N[i] : -bN.N[i];
                    const Vec3& gradV = testOwner ? gradP[i] : gradN[i];

                    for (int tj = 0; tj < 16; ++tj) {
                        const bool trialOwner = (tj < 8);
                        const int j = trialOwner ? tj : tj - 8;
                        const int colCell = trialOwner ? P : N;
                        const int col = 8*colCell + j;

                        const double uJump = trialOwner ? bP.N[j] : -bN.N[j];
                        const Vec3& gradU = trialOwner ? gradP[j] : gradN[j];

                        const double avgGradUdotN = 0.5 * dot3(gradU, n);
                        const double avgGradVdotN = 0.5 * dot3(gradV, n);

                        const double aij =
                            - avgGradUdotN * vJump
                            - avgGradVdotN * uJump
                            + penalty * uJump * vJump;

                        sparse_add(sys.A, row, col, Real(w * aij));
                    }
                }
            } else if (includeWeakBoundaryTerms) {
                const double g = mms_exact_value(xqP, mms);

                for (int i = 0; i < 8; ++i) {
                    const int row = 8*P + i;
                    const double v = bP.N[i];

                    sys.b[row] += Real(w * (-dot3(gradP[i], n) * g + penalty * g * v));

                    for (int j = 0; j < 8; ++j) {
                        const int col = 8*P + j;
                        const double u = bP.N[j];

                        const double aij =
                            - dot3(gradP[j], n) * v
                            - dot3(gradP[i], n) * u
                            + penalty * u * v;

                        sparse_add(sys.A, row, col, Real(w * aij));
                    }
                }
            }
        }
    }

    return sys;
}

static StrongDirichletData build_dg_hex_q1_boundary_strong_mms(
    const PolyMesh& m,
    const DgHexQ1DofMap& dm,
    const std::string& mms
) {
    (void)dm;

    const auto geoms = build_dg_hex_q1_geoms(m);
    std::vector<char> mark(8 * (int)m.cells.size(), 0);
    StrongDirichletData bc;

    for (int f = (int)m.neighbour.size(); f < (int)m.faces.size(); ++f) {
        const int c = m.owner[f];
        const auto& K = geoms[c];

        for (int fv : m.faces[f].verts) {
            const int local = dg_hex_local_index_for_vertex(K, fv);
            if (local < 0) {
                throw std::runtime_error("dg_hex_q1 strong BC face vertex not found.");
            }

            const int dof = 8*c + local;

            if (!mark[dof]) {
                mark[dof] = 1;
                bc.dofs.push_back(dof);
                bc.values.push_back(Real(mms_exact_value(m.points[fv], mms)));
            }
        }
    }

    return bc;
}

static AssembledSystem assemble_dg_hex_q1_strong_poisson_mms(
    const PolyMesh& m,
    const DgHexQ1DofMap& dm,
    const std::string& mms,
    const DgSipgOptions& dgopt
) {
    AssembledSystem sys = assemble_dg_hex_q1_sipg_poisson_mms(
        m,
        dm,
        mms,
        dgopt,
        false
    );

    StrongDirichletData bc = build_dg_hex_q1_boundary_strong_mms(m, dm, mms);
    apply_strong_dirichlet_by_dof_values(bc, sys.A, sys.b);

    return sys;
}

static ErrorNorms compute_dg_hex_q1_error_norms(
    const PolyMesh& m,
    const DgHexQ1DofMap& dm,
    const std::string& mms,
    const std::vector<Real>& x
) {
    if ((int)x.size() != dm.nDofs) {
        throw std::runtime_error("dg_hex_q1 error norm size mismatch.");
    }

    const auto geoms = build_dg_hex_q1_geoms(m);

    double l2 = 0.0;
    double h1 = 0.0;
    double nodal2 = 0.0;
    double nodalMax = 0.0;

    for (int c = 0; c < dm.nCells; ++c) {
        const auto& K = geoms[c];

        for (int a = 0; a < 8; ++a) {
            const double ua = double(x[8*c + a]);
            const double ue = mms_exact_value(m.points[K.verts[a]], mms);
            const double e = ua - ue;
            nodal2 += e * e;
            nodalMax = std::max(nodalMax, std::abs(e));
        }

        for (const auto& q : hex_q1_error_quadrature_selected()) {
            const auto basis = eval_hex_q1_basis_at(q.xi, q.eta, q.zeta);
            const Vec3 xq = map_hex_q1_to_physical(m, K.verts, basis.N);
            const Mat3 J = jacobian_hex_q1(m, K.verts, basis.dNref);
            const double detJ = det3(J);
            const Mat3 invJ = inv3(J);

            std::array<Vec3,8> grad{};
            for (int a = 0; a < 8; ++a) {
                grad[a] = invJT_mul(invJ, basis.dNref[a]);
            }

            double uh = 0.0;
            Vec3 guh;

            for (int a = 0; a < 8; ++a) {
                const double ua = double(x[8*c + a]);
                uh += basis.N[a] * ua;
                guh = dghex_add(guh, dghex_mul(ua, grad[a]));
            }

            const double ue = mms_exact_value(xq, mms);
            const Vec3 ge = mms_exact_grad(xq, mms);

            const double eu = uh - ue;
            const Vec3 eg = {guh.x - ge.x, guh.y - ge.y, guh.z - ge.z};

            l2 += q.weight * detJ * eu * eu;
            h1 += q.weight * detJ * dot3(eg, eg);
        }
    }

    ErrorNorms en;
    en.nodalL2 = std::sqrt(nodal2);
    en.nodalMax = nodalMax;
    en.L2 = std::sqrt(l2);
    en.H1Semi = std::sqrt(h1);
    return en;
}

static void probe_dg_hex_q1_system(
    const PolyMesh& m,
    const DgHexQ1DofMap& dm,
    const AssembledSystem& sys,
    int diagLevel
) {
    std::cout << "---------------- DG hex Q1 SIPG system probe -----------------\n";
    std::cout << "space                     = " << dm.resolvedSpace << "\n";
    std::cout << "continuity                = DG\n";
    std::cout << "basis                     = cell-local nodal Q1 hex\n";
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

    std::cout << "--------------------------------------------------------------\n";
}
