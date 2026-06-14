#pragma once

// ============================================================================
// SECTION 12: CG tet P1 scalar Poisson with weak/Nitsche Dirichlet BC
// ============================================================================
//
// This is the CG counterpart of the DG SIPG boundary treatment.
// It keeps the volume operator CG-continuous, but imposes boundary Dirichlet
// weakly:
//
//   a += - ∫Γ k grad(u).n v
//        - ∫Γ k grad(v).n u
//        + ∫Γ penalty k/h u v
//
//   b += - ∫Γ k grad(v).n g
//        + ∫Γ penalty k/h g v
//
// Tet P1 first; hex Nitsche can use the same spec later.
// ============================================================================

struct CgTetP1CellGeom {
    std::array<int,4> verts{};
    std::array<Vec3,4> x{};
    std::array<Vec3,4> grad{};
    double volume = 0.0;
    Vec3 centroid{};
};

static inline Vec3 cg_nitsche_add(const Vec3& a, const Vec3& b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

static inline Vec3 cg_nitsche_mul(double s, const Vec3& a) {
    return {s*a.x, s*a.y, s*a.z};
}

static inline double cg_nitsche_norm(const Vec3& a) {
    return std::sqrt(dot3(a,a));
}

static void cg_tet_p1_orient_and_compute(CgTetP1CellGeom& g) {
    double vol6 = tet_signed_volume6(g.x[0], g.x[1], g.x[2], g.x[3]);

    if (vol6 < 0.0) {
        std::swap(g.verts[2], g.verts[3]);
        std::swap(g.x[2], g.x[3]);
        vol6 = -vol6;
    }

    if (!(vol6 > 0.0)) {
        throw std::runtime_error("CG Nitsche tet has non-positive volume.");
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
    g.centroid = cg_nitsche_mul(
        0.25,
        cg_nitsche_add(cg_nitsche_add(g.x[0], g.x[1]), cg_nitsche_add(g.x[2], g.x[3]))
    );
}

static std::vector<CgTetP1CellGeom> build_cg_tet_p1_geoms(const PolyMesh& m) {
    std::vector<CgTetP1CellGeom> geoms(m.cells.size());

    for (int c = 0; c < (int)m.cells.size(); ++c) {
        if (m.cells[c].verts.size() != 4) {
            throw std::runtime_error("CG Nitsche path requires tet cells.");
        }

        CgTetP1CellGeom g;

        for (int a = 0; a < 4; ++a) {
            g.verts[a] = m.cells[c].verts[a];
            g.x[a] = m.points[g.verts[a]];
        }

        cg_tet_p1_orient_and_compute(g);
        geoms[c] = g;
    }

    return geoms;
}

static int cg_tet_local_index_for_vertex(const CgTetP1CellGeom& K, int v) {
    for (int a = 0; a < 4; ++a) {
        if (K.verts[a] == v) return a;
    }
    return -1;
}

static std::array<double,4> cg_tet_face_phi(
    const Face& f,
    const CgTetP1CellGeom& K,
    const std::array<double,3>& lam
) {
    std::array<double,4> phi = {0.0, 0.0, 0.0, 0.0};

    if (f.verts.size() != 3) {
        throw std::runtime_error("CG Nitsche expected triangular boundary face.");
    }

    for (int q = 0; q < 3; ++q) {
        int li = cg_tet_local_index_for_vertex(K, f.verts[q]);
        if (li < 0) {
            throw std::runtime_error("CG Nitsche face vertex not found in tet.");
        }
        phi[li] = lam[q];
    }

    return phi;
}

static Vec3 cg_tet_face_centroid(const PolyMesh& m, const Face& f) {
    Vec3 c;
    for (int v : f.verts) c = cg_nitsche_add(c, m.points[v]);
    return cg_nitsche_mul(1.0 / double(f.verts.size()), c);
}

static Vec3 cg_tet_owner_outward_unit_normal(
    const PolyMesh& m,
    int faceId,
    const CgTetP1CellGeom& ownerGeom,
    double& area
) {
    const Face& f = m.faces[faceId];

    if (f.verts.size() != 3) {
        throw std::runtime_error("CG Nitsche expected triangular boundary face.");
    }

    const Vec3& p0 = m.points[f.verts[0]];
    const Vec3& p1 = m.points[f.verts[1]];
    const Vec3& p2 = m.points[f.verts[2]];

    Vec3 n2 = cross3(p1 - p0, p2 - p0);
    double mag = cg_nitsche_norm(n2);

    if (!(mag > 0.0)) {
        throw std::runtime_error("CG Nitsche degenerate face.");
    }

    area = 0.5 * mag;
    Vec3 n = cg_nitsche_mul(1.0 / mag, n2);

    Vec3 fc = cg_tet_face_centroid(m, f);
    if (dot3(n, fc - ownerGeom.centroid) < 0.0) {
        n = cg_nitsche_mul(-1.0, n);
    }

    return n;
}

static AssembledSystem assemble_cg_tet_p1_nitsche_poisson_mms(
    const PolyMesh& m,
    const LinearCgDofMap& dm,
    const std::string& mms,
    const ScalarEllipticSpec& spec
) {
    if (dm.resolvedSpace != "cg_tet_p1") {
        throw std::runtime_error("CG Nitsche currently implemented only for cg_tet_p1.");
    }

    AssembledSystem sys;
    sys.b.assign(dm.nDofs, Real(0));

    const std::string sparseMode = memoirs_sparse_mode();
    if (sparseMode == "fixed_csr") {
        sparse_init_scalar_cg_fixed_pattern(sys.A, m, dm);
    } else if (sparseMode == "legacy") {
        sys.A.rows.resize(dm.nDofs);
    } else {
        throw std::runtime_error("Unsupported sparse mode in CG Nitsche.");
    }

    const auto geoms = build_cg_tet_p1_geoms(m);

    const std::array<std::array<double,4>,4> tetQ = {{
        {{0.5854101966249685, 0.1381966011250105, 0.1381966011250105, 0.1381966011250105}},
        {{0.1381966011250105, 0.5854101966249685, 0.1381966011250105, 0.1381966011250105}},
        {{0.1381966011250105, 0.1381966011250105, 0.5854101966249685, 0.1381966011250105}},
        {{0.1381966011250105, 0.1381966011250105, 0.1381966011250105, 0.5854101966249685}}
    }};

    for (int c = 0; c < (int)m.cells.size(); ++c) {
        const auto& K = geoms[c];

        Vec3 xc;
        for (int a = 0; a < 4; ++a) xc = cg_nitsche_add(xc, cg_nitsche_mul(0.25, K.x[a]));
        const Real kappaCell = spec.diffusion.value_in_cell(c, xc);

        for (int i = 0; i < 4; ++i) {
            int row = K.verts[i];

            for (int j = 0; j < 4; ++j) {
                int col = K.verts[j];
                sparse_add(sys.A, row, col, Real(K.volume) * kappaCell * Real(dot3(K.grad[i], K.grad[j])));
            }
        }

        for (const auto& lam : tetQ) {
            Vec3 xq;
            for (int a = 0; a < 4; ++a) xq = cg_nitsche_add(xq, cg_nitsche_mul(lam[a], K.x[a]));

            const Real kappa = spec.diffusion.value_in_cell(c, xq);
            const double fval = mms_rhs_value(xq, mms);
            const double w = K.volume / 4.0;

            for (int i = 0; i < 4; ++i) {
                sys.b[K.verts[i]] += Real(w) * kappa * Real(fval * lam[i]);
            }
        }
    }

    const std::array<std::array<double,3>,3> faceQ = {{
        {{2.0/3.0, 1.0/6.0, 1.0/6.0}},
        {{1.0/6.0, 2.0/3.0, 1.0/6.0}},
        {{1.0/6.0, 1.0/6.0, 2.0/3.0}}
    }};

    for (int f = (int)m.neighbour.size(); f < (int)m.faces.size(); ++f) {
        const int P = m.owner[f];
        const auto& K = geoms[P];

        double area = 0.0;
        Vec3 n = cg_tet_owner_outward_unit_normal(m, f, K, area);
        const double hP = std::max(3.0 * K.volume / area, 1.0e-300);

        for (const auto& lam : faceQ) {
            const double w = area / 3.0;
            const auto phi = cg_tet_face_phi(m.faces[f], K, lam);

            Vec3 xq;
            for (int a = 0; a < 3; ++a) {
                xq = cg_nitsche_add(xq, cg_nitsche_mul(lam[a], m.points[m.faces[f].verts[a]]));
            }

            const Real kappa = spec.diffusion.face_value(P, -1);
            const double penalty = spec.boundary.penaltySigma * double(kappa) / hP;
            const double g = mms_exact_value(xq, mms);

            for (int i = 0; i < 4; ++i) {
                int row = K.verts[i];
                const double v = phi[i];

                sys.b[row] += Real(w * (-double(kappa) * dot3(K.grad[i], n) * g + penalty * g * v));

                for (int j = 0; j < 4; ++j) {
                    int col = K.verts[j];
                    const double u = phi[j];

                    const double aij =
                        - double(kappa) * dot3(K.grad[j], n) * v
                        - double(kappa) * dot3(K.grad[i], n) * u
                        + penalty * u * v;

                    sparse_add(sys.A, row, col, Real(w * aij));
                }
            }
        }
    }

    return sys;
}
