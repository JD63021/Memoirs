#pragma once

// ============================================================================
// SECTION 14: CG hex Q1 scalar Poisson with weak/Nitsche Dirichlet BC
// ============================================================================
//
// CG Q1 volume diffusion with weak Dirichlet imposed on boundary faces:
//
//   a += - ∫Γ k grad(u).n v
//        - ∫Γ k grad(v).n u
//        + ∫Γ penalty k/h u v
//
//   b += - ∫Γ k grad(v).n g
//        + ∫Γ penalty k/h g v
//
// This completes the CG strong/weak BC pair for hex Q1.
// ============================================================================

struct CgHexQ1CellGeom {
    std::array<int,8> verts{};
    Vec3 centroid{};
    double volumeApprox = 0.0;
};

static inline Vec3 cghex_add(const Vec3& a, const Vec3& b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

static inline Vec3 cghex_mul(double s, const Vec3& a) {
    return {s*a.x, s*a.y, s*a.z};
}

static inline double cghex_norm(const Vec3& a) {
    return std::sqrt(dot3(a,a));
}

static std::vector<CgHexQ1CellGeom> build_cg_hex_q1_geoms(const PolyMesh& m) {
    std::vector<CgHexQ1CellGeom> geoms(m.cells.size());

    for (int c = 0; c < (int)m.cells.size(); ++c) {
        if (m.cells[c].verts.size() != 8) {
            throw std::runtime_error("CG hex Nitsche path requires hex cells.");
        }

        CgHexQ1CellGeom g;
        g.verts = ordered_hex_vertices_axis_aligned(m, m.cells[c]);

        for (int a = 0; a < 8; ++a) {
            g.centroid = cghex_add(g.centroid, m.points[g.verts[a]]);
        }
        g.centroid = cghex_mul(1.0 / 8.0, g.centroid);

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

static int cg_hex_local_index_for_vertex(const CgHexQ1CellGeom& K, int v) {
    for (int a = 0; a < 8; ++a) {
        if (K.verts[a] == v) return a;
    }
    return -1;
}

struct CgHexFaceInfo {
    int faceId = -1;
    std::array<int,4> localVerts{};
    int fixedAxis = 0;
    double fixedValue = 0.0;
};

static CgHexFaceInfo cg_hex_identify_local_face(
    const Face& f,
    const CgHexQ1CellGeom& K
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

    if (f.verts.size() != 4) {
        throw std::runtime_error("CG hex Nitsche expected quadrilateral face.");
    }

    std::array<int,4> faceLocal{};
    for (int i = 0; i < 4; ++i) {
        int li = cg_hex_local_index_for_vertex(K, f.verts[i]);
        if (li < 0) {
            throw std::runtime_error("CG hex Nitsche face vertex not found in owner cell.");
        }
        faceLocal[i] = li;
    }

    auto sortedFace = faceLocal;
    std::sort(sortedFace.begin(), sortedFace.end());

    for (int fid = 0; fid < 6; ++fid) {
        auto cand = faceLocals[fid];
        std::sort(cand.begin(), cand.end());

        if (cand == sortedFace) {
            CgHexFaceInfo info;
            info.faceId = fid;
            info.localVerts = faceLocals[fid];
            info.fixedAxis = axis[fid];
            info.fixedValue = val[fid];
            return info;
        }
    }

    throw std::runtime_error("CG hex Nitsche could not identify local face.");
}

static VolumeQuadraturePoint cg_hex_face_to_volume_qp(
    const CgHexFaceInfo& info,
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

static Vec3 cg_hex_face_centroid(const PolyMesh& m, const Face& f) {
    Vec3 c;
    for (int v : f.verts) c = cghex_add(c, m.points[v]);
    return cghex_mul(1.0 / double(f.verts.size()), c);
}

static Vec3 cg_hex_owner_outward_unit_normal(
    const PolyMesh& m,
    int faceId,
    const CgHexQ1CellGeom& ownerGeom,
    double& area
) {
    const Face& f = m.faces[faceId];

    if (f.verts.size() != 4) {
        throw std::runtime_error("CG hex Nitsche expected quadrilateral face.");
    }

    const Vec3& p0 = m.points[f.verts[0]];
    const Vec3& p1 = m.points[f.verts[1]];
    const Vec3& p2 = m.points[f.verts[2]];
    const Vec3& p3 = m.points[f.verts[3]];

    Vec3 nA1 = cross3(p1 - p0, p2 - p0);
    Vec3 nA2 = cross3(p2 - p0, p3 - p0);
    Vec3 nA = cghex_add(cghex_mul(0.5, nA1), cghex_mul(0.5, nA2));

    area = cghex_norm(nA);

    if (!(area > 0.0)) {
        throw std::runtime_error("CG hex Nitsche degenerate quadrilateral face.");
    }

    Vec3 n = cghex_mul(1.0 / area, nA);

    Vec3 fc = cg_hex_face_centroid(m, f);
    if (dot3(n, fc - ownerGeom.centroid) < 0.0) {
        n = cghex_mul(-1.0, n);
    }

    return n;
}

static AssembledSystem assemble_cg_hex_q1_nitsche_poisson_mms(
    const PolyMesh& m,
    const LinearCgDofMap& dm,
    const std::string& mms,
    const ScalarEllipticSpec& spec
) {
    if (dm.resolvedSpace != "cg_hex_q1") {
        throw std::runtime_error("CG hex Nitsche currently implemented only for cg_hex_q1.");
    }

    AssembledSystem sys;
    sys.b.assign(dm.nDofs, Real(0));

    const std::string sparseMode = memoirs_sparse_mode();
    if (sparseMode == "fixed_csr") {
        sparse_init_scalar_cg_fixed_pattern(sys.A, m, dm);
    } else if (sparseMode == "legacy") {
        sys.A.rows.resize(dm.nDofs);
    } else {
        throw std::runtime_error("Unsupported sparse mode in CG hex Nitsche.");
    }

    const auto geoms = build_cg_hex_q1_geoms(m);

    for (int c = 0; c < (int)m.cells.size(); ++c) {
        const auto& K = geoms[c];

        for (const auto& q : hex_q1_volume_quadrature_selected()) {
            const auto basis = eval_hex_q1_basis_at(q.xi, q.eta, q.zeta);
            const Vec3 xq = map_hex_q1_to_physical(m, K.verts, basis.N);
            const Mat3 J = jacobian_hex_q1(m, K.verts, basis.dNref);
            const double detJ = det3(J);

            if (!(detJ > 0.0)) {
                throw std::runtime_error("CG hex Nitsche non-positive detJ.");
            }

            const Mat3 invJ = inv3(J);

            std::array<Vec3,8> grad{};
            for (int a = 0; a < 8; ++a) {
                grad[a] = invJT_mul(invJ, basis.dNref[a]);
            }

            const Real kappa = spec.diffusion.value_in_cell(c, xq);
            const double w = q.weight * detJ;
            const double fval = mms_rhs_value(xq, mms);

            for (int i = 0; i < 8; ++i) {
                const int row = K.verts[i];

                sys.b[row] += Real(w) * kappa * Real(fval * basis.N[i]);

                for (int j = 0; j < 8; ++j) {
                    const int col = K.verts[j];
                    sparse_add(sys.A, row, col, Real(w) * kappa * Real(dot3(grad[i], grad[j])));
                }
            }
        }
    }

    for (int f = (int)m.neighbour.size(); f < (int)m.faces.size(); ++f) {
        const int P = m.owner[f];
        const auto& K = geoms[P];

        const CgHexFaceInfo faceInfo = cg_hex_identify_local_face(m.faces[f], K);

        double area = 0.0;
        const Vec3 n = cg_hex_owner_outward_unit_normal(m, f, K, area);
        const double hP = std::max(K.volumeApprox / area, 1.0e-300);

        for (const auto& fq : quad_face_quadrature_selected()) {
            const double w = area * 0.25 * fq.weight;

            const auto qv = cg_hex_face_to_volume_qp(faceInfo, fq.r, fq.s, fq.weight);
            const auto basis = eval_hex_q1_basis_at(qv.xi, qv.eta, qv.zeta);
            const Vec3 xq = map_hex_q1_to_physical(m, K.verts, basis.N);
            const Mat3 J = jacobian_hex_q1(m, K.verts, basis.dNref);
            const Mat3 invJ = inv3(J);

            std::array<Vec3,8> grad{};
            for (int a = 0; a < 8; ++a) {
                grad[a] = invJT_mul(invJ, basis.dNref[a]);
            }

            const Real kappaR = spec.diffusion.face_value(P, -1);
            const double kappa = double(kappaR);
            const double penalty = spec.boundary.penaltySigma * kappa / hP;
            const double g = mms_exact_value(xq, mms);

            for (int i = 0; i < 8; ++i) {
                const int row = K.verts[i];
                const double v = basis.N[i];

                sys.b[row] += Real(w * (-kappa * dot3(grad[i], n) * g + penalty * g * v));

                for (int j = 0; j < 8; ++j) {
                    const int col = K.verts[j];
                    const double u = basis.N[j];

                    const double aij =
                        - kappa * dot3(grad[j], n) * v
                        - kappa * dot3(grad[i], n) * u
                        + penalty * u * v;

                    sparse_add(sys.A, row, col, Real(w * aij));
                }
            }
        }
    }

    return sys;
}
