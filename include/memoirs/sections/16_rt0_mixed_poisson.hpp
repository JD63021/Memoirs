#pragma once

// ============================================================================
// SECTION 16: RT0 mixed Poisson on polyMesh tet/hex cells
// ============================================================================
//
// Unknown ordering:
//   0 .. nFaces-1                    : one oriented normal flux dof per face
//   nFaces .. nFaces+nCells-1         : one scalar P0/Q0 dof per cell
//
// Global flux orientation:
//   q_f is the integrated normal flux through face f with normal outward from
//   owner cell. For neighbour cell, the local outward flux is -q_f.
//
// Mixed system with scalar Dirichlet u=g:
//
//   sigma = -grad u
//   div sigma = f
//
// Weak form:
//   (sigma,tau) - (u,div tau) = - <g,tau.n>
//   (div sigma,v)             = (f,v)
//
// Symmetric saddle matrix assembled as:
//
//   [ M  -B^T ] [q] = [r]
//   [-B    0  ] [u]   [-F]
//
// where B q = cell source integral.
// ============================================================================

enum class Rt0CellKind {
    Tet,
    Hex
};

struct Rt0DofMap {
    Rt0CellKind kind = Rt0CellKind::Hex;
    std::string resolvedSpace = "rt0_hex";
    int nFluxDofs = 0;
    int nScalarDofs = 0;
    int nTotalDofs = 0;
};

struct Rt0CellGeom {
    Rt0CellKind kind = Rt0CellKind::Hex;
    std::vector<int> verts;
    std::vector<int> faces;
    Vec3 centroid{};
    double volume = 0.0;

    // Hex geometry cache.
    double xmin = 0, xmax = 0;
    double ymin = 0, ymax = 0;
    double zmin = 0, zmax = 0;
};

struct Rt0ErrorReport {
    double scalarL2 = -1.0;
    double fluxL2 = -1.0;
    double divCellAbsMax = -1.0;
    double divCellL2 = -1.0;
    double cellConservationAbsMax = -1.0;
    double cellConservationL2 = -1.0;
};

static inline Vec3 rt0_add(const Vec3& a, const Vec3& b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

static inline Vec3 rt0_sub(const Vec3& a, const Vec3& b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

static inline Vec3 rt0_mul(double s, const Vec3& a) {
    return {s*a.x, s*a.y, s*a.z};
}

static inline double rt0_norm(const Vec3& a) {
    return std::sqrt(dot3(a,a));
}

static Rt0DofMap build_rt0_dof_map(const PolyMesh& m) {
    MeshCellCounts cc = count_cell_types(m);

    Rt0DofMap dm;
    dm.nFluxDofs = (int)m.faces.size();
    dm.nScalarDofs = (int)m.cells.size();
    dm.nTotalDofs = dm.nFluxDofs + dm.nScalarDofs;

    if (cc.nTet == (int)m.cells.size() && cc.nHex == 0 && cc.nOther == 0) {
        dm.kind = Rt0CellKind::Tet;
        dm.resolvedSpace = "rt0_tet_p0";
    } else if (cc.nHex == (int)m.cells.size() && cc.nTet == 0 && cc.nOther == 0) {
        dm.kind = Rt0CellKind::Hex;
        dm.resolvedSpace = "rt0_hex_q0";
    } else {
        throw std::runtime_error("RT0 mixed Poisson requires a pure tet or pure hex mesh.");
    }

    return dm;
}

static std::vector<std::vector<int>> build_cell_faces(const PolyMesh& m) {
    std::vector<std::vector<int>> cellFaces(m.cells.size());

    for (int f = 0; f < (int)m.faces.size(); ++f) {
        const int P = m.owner[f];
        cellFaces[P].push_back(f);

        if (f < (int)m.neighbour.size()) {
            const int N = m.neighbour[f];
            cellFaces[N].push_back(f);
        }
    }

    return cellFaces;
}

static double rt0_tet_volume_abs(const PolyMesh& m, const std::vector<int>& v) {
    return std::abs(tet_signed_volume6(
        m.points[v[0]], m.points[v[1]], m.points[v[2]], m.points[v[3]]
    )) / 6.0;
}

static std::vector<Rt0CellGeom> build_rt0_cell_geoms(
    const PolyMesh& m,
    const Rt0DofMap& dm
) {
    const auto cellFaces = build_cell_faces(m);
    std::vector<Rt0CellGeom> geoms(m.cells.size());

    for (int c = 0; c < (int)m.cells.size(); ++c) {
        Rt0CellGeom K;
        K.kind = dm.kind;

        if (dm.kind == Rt0CellKind::Tet) {
            if (m.cells[c].verts.size() != 4) {
                throw std::runtime_error("RT0 tet path found non-tet cell.");
            }

            K.verts.assign(m.cells[c].verts.begin(), m.cells[c].verts.end());
            K.volume = rt0_tet_volume_abs(m, K.verts);

            for (int v : K.verts) {
                K.centroid = rt0_add(K.centroid, m.points[v]);
            }
            K.centroid = rt0_mul(0.25, K.centroid);
        } else {
            if (m.cells[c].verts.size() != 8) {
                throw std::runtime_error("RT0 hex path found non-hex cell.");
            }

            const auto hv = ordered_hex_vertices_axis_aligned(m, m.cells[c]);
            K.verts.assign(hv.begin(), hv.end());

            K.xmin = K.ymin = K.zmin =  std::numeric_limits<double>::infinity();
            K.xmax = K.ymax = K.zmax = -std::numeric_limits<double>::infinity();

            for (int v : K.verts) {
                const Vec3& x = m.points[v];

                K.xmin = std::min(K.xmin, x.x);
                K.xmax = std::max(K.xmax, x.x);
                K.ymin = std::min(K.ymin, x.y);
                K.ymax = std::max(K.ymax, x.y);
                K.zmin = std::min(K.zmin, x.z);
                K.zmax = std::max(K.zmax, x.z);

                K.centroid = rt0_add(K.centroid, x);
            }

            K.centroid = rt0_mul(1.0 / 8.0, K.centroid);
            K.volume = (K.xmax - K.xmin) * (K.ymax - K.ymin) * (K.zmax - K.zmin);
        }

        if (!(K.volume > 0.0)) {
            throw std::runtime_error("RT0 found non-positive cell volume.");
        }

        K.faces = cellFaces[c];

        const int expectedFaces = (dm.kind == Rt0CellKind::Tet) ? 4 : 6;
        if ((int)K.faces.size() != expectedFaces) {
            throw std::runtime_error("RT0 cell face count mismatch.");
        }

        geoms[c] = K;
    }

    return geoms;
}

static int rt0_cell_face_sign(const PolyMesh& m, int c, int f) {
    if (m.owner[f] == c) return +1;
    if (f < (int)m.neighbour.size() && m.neighbour[f] == c) return -1;
    throw std::runtime_error("RT0 face not incident to cell.");
}

static int rt0_tet_local_face_index(
    const PolyMesh& m,
    const Rt0CellGeom& K,
    int faceId
) {
    const Face& f = m.faces[faceId];

    for (int missing = 0; missing < 4; ++missing) {
        bool allOtherPresent = true;

        for (int a = 0; a < 4; ++a) {
            if (a == missing) continue;

            bool found = false;
            for (int fv : f.verts) {
                if (fv == K.verts[a]) {
                    found = true;
                    break;
                }
            }

            if (!found) {
                allOtherPresent = false;
                break;
            }
        }

        if (allOtherPresent) return missing;
    }

    throw std::runtime_error("RT0 tet could not identify local face.");
}

static int rt0_hex_local_vertex_index(const Rt0CellGeom& K, int v) {
    for (int a = 0; a < 8; ++a) {
        if (K.verts[a] == v) return a;
    }
    return -1;
}

static int rt0_hex_local_face_index(
    const PolyMesh& m,
    const Rt0CellGeom& K,
    int faceId
) {
    static const std::array<std::array<int,4>,6> faceLocals = {{
        {{0,3,7,4}}, // x-
        {{1,2,6,5}}, // x+
        {{0,1,5,4}}, // y-
        {{3,2,6,7}}, // y+
        {{0,1,2,3}}, // z-
        {{4,5,6,7}}  // z+
    }};

    const Face& f = m.faces[faceId];

    if (f.verts.size() != 4) {
        throw std::runtime_error("RT0 hex expected quadrilateral face.");
    }

    std::array<int,4> local{};
    for (int i = 0; i < 4; ++i) {
        local[i] = rt0_hex_local_vertex_index(K, f.verts[i]);
        if (local[i] < 0) {
            throw std::runtime_error("RT0 hex face vertex not found in cell.");
        }
    }

    auto sortedLocal = local;
    std::sort(sortedLocal.begin(), sortedLocal.end());

    for (int fid = 0; fid < 6; ++fid) {
        auto cand = faceLocals[fid];
        std::sort(cand.begin(), cand.end());

        if (cand == sortedLocal) return fid;
    }

    throw std::runtime_error("RT0 hex could not identify local face.");
}

static int rt0_local_face_index(
    const PolyMesh& m,
    const Rt0CellGeom& K,
    int faceId
) {
    if (K.kind == Rt0CellKind::Tet) {
        return rt0_tet_local_face_index(m, K, faceId);
    }

    return rt0_hex_local_face_index(m, K, faceId);
}

static Vec3 rt0_eval_local_basis(
    const PolyMesh& m,
    const Rt0CellGeom& K,
    int localFace,
    const Vec3& x
) {
    if (K.kind == Rt0CellKind::Tet) {
        const Vec3& a = m.points[K.verts[localFace]];
        return rt0_mul(1.0 / (3.0 * K.volume), rt0_sub(x, a));
    }

    const double hx = K.xmax - K.xmin;
    const double hy = K.ymax - K.ymin;
    const double hz = K.zmax - K.zmin;

    const double Ax = hy * hz;
    const double Ay = hx * hz;
    const double Az = hx * hy;

    if (localFace == 0) {
        return {(x.x - K.xmax) / (hx * Ax), 0.0, 0.0};
    }
    if (localFace == 1) {
        return {(x.x - K.xmin) / (hx * Ax), 0.0, 0.0};
    }
    if (localFace == 2) {
        return {0.0, (x.y - K.ymax) / (hy * Ay), 0.0};
    }
    if (localFace == 3) {
        return {0.0, (x.y - K.ymin) / (hy * Ay), 0.0};
    }
    if (localFace == 4) {
        return {0.0, 0.0, (x.z - K.zmax) / (hz * Az)};
    }
    if (localFace == 5) {
        return {0.0, 0.0, (x.z - K.zmin) / (hz * Az)};
    }

    throw std::runtime_error("RT0 invalid local face index.");
}

struct Rt0VolQ {
    Vec3 x{};
    double w = 0.0;
};

static std::vector<Rt0VolQ> rt0_cell_volume_quadrature(
    const PolyMesh& m,
    const Rt0CellGeom& K,
    bool errorQuadrature
) {
    std::vector<Rt0VolQ> out;

    if (K.kind == Rt0CellKind::Tet) {
        const Vec3& x0 = m.points[K.verts[0]];
        const Vec3& x1 = m.points[K.verts[1]];
        const Vec3& x2 = m.points[K.verts[2]];
        const Vec3& x3 = m.points[K.verts[3]];

        const auto& qs = errorQuadrature
            ? tet_volume_quadrature_selected_for_error()
            : tet_volume_quadrature_selected();

        for (const auto& tq : qs) {
            const auto b = eval_tet_p1_basis_at(tq.xi, tq.eta, tq.zeta);

            Rt0VolQ q;
            q.x = rt0_add(
                rt0_add(rt0_mul(b.N[0], x0), rt0_mul(b.N[1], x1)),
                rt0_add(rt0_mul(b.N[2], x2), rt0_mul(b.N[3], x3))
            );
            q.w = tq.weight * 6.0 * K.volume;
            out.push_back(q);
        }

        return out;
    }

    std::array<int,8> hv{};
    for (int i = 0; i < 8; ++i) hv[i] = K.verts[i];

    const auto& qs = errorQuadrature
        ? hex_q1_error_quadrature_selected()
        : hex_q1_volume_quadrature_selected();

    for (const auto& hq : qs) {
        const auto basis = eval_hex_q1_basis_at(hq.xi, hq.eta, hq.zeta);
        const Vec3 xq = map_hex_q1_to_physical(m, hv, basis.N);
        const Mat3 J = jacobian_hex_q1(m, hv, basis.dNref);
        const double detJ = det3(J);

        if (!(detJ > 0.0)) {
            throw std::runtime_error("RT0 hex non-positive detJ.");
        }

        Rt0VolQ q;
        q.x = xq;
        q.w = hq.weight * detJ;
        out.push_back(q);
    }

    return out;
}

struct Rt0FaceQ {
    Vec3 x{};
    double w = 0.0;
};

static std::vector<Rt0FaceQ> rt0_face_quadrature(
    const PolyMesh& m,
    const Face& f
) {
    std::vector<Rt0FaceQ> out;

    if (f.verts.size() == 3) {
        const Vec3& p0 = m.points[f.verts[0]];
        const Vec3& p1 = m.points[f.verts[1]];
        const Vec3& p2 = m.points[f.verts[2]];

        const Vec3 e1 = p1 - p0;
        const Vec3 e2 = p2 - p0;
        const double jac = rt0_norm(cross3(e1, e2));

        for (const auto& tq : triangle_face_quadrature_selected()) {
            Rt0FaceQ q;
            q.x = rt0_add(p0, rt0_add(rt0_mul(tq.r, e1), rt0_mul(tq.s, e2)));
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

            const Vec3 x = rt0_add(
                rt0_add(rt0_mul(N0, p0), rt0_mul(N1, p1)),
                rt0_add(rt0_mul(N2, p2), rt0_mul(N3, p3))
            );

            const Vec3 dxdr = rt0_add(
                rt0_add(rt0_mul(-0.25*(1.0-s), p0), rt0_mul( 0.25*(1.0-s), p1)),
                rt0_add(rt0_mul( 0.25*(1.0+s), p2), rt0_mul(-0.25*(1.0+s), p3))
            );

            const Vec3 dxds = rt0_add(
                rt0_add(rt0_mul(-0.25*(1.0-r), p0), rt0_mul(-0.25*(1.0+r), p1)),
                rt0_add(rt0_mul( 0.25*(1.0+r), p2), rt0_mul( 0.25*(1.0-r), p3))
            );

            Rt0FaceQ q;
            q.x = x;
            q.w = qq.weight * rt0_norm(cross3(dxdr, dxds));
            out.push_back(q);
        }

        return out;
    }

    throw std::runtime_error("RT0 face quadrature supports tri/quad faces only.");
}

static std::vector<std::vector<int>> build_rt0_mixed_pattern(
    const PolyMesh& m,
    const Rt0DofMap& dm,
    const std::vector<Rt0CellGeom>& geoms
) {
    std::vector<std::vector<int>> rowCols(dm.nTotalDofs);

    for (int i = 0; i < dm.nTotalDofs; ++i) {
        rowCols[i].push_back(i);
    }

    for (int c = 0; c < dm.nScalarDofs; ++c) {
        const int prow = dm.nFluxDofs + c;
        rowCols[prow].push_back(prow);

        for (int f : geoms[c].faces) {
            rowCols[prow].push_back(f);
            rowCols[f].push_back(prow);
        }

        for (int fi : geoms[c].faces) {
            for (int fj : geoms[c].faces) {
                rowCols[fi].push_back(fj);
            }
        }
    }

    return rowCols;
}

static double rt0_boundary_g_face_average(
    const PolyMesh& m,
    int faceId,
    const std::string& mms
) {
    double integ = 0.0;
    double area = 0.0;

    for (const auto& q : rt0_face_quadrature(m, m.faces[faceId])) {
        integ += q.w * mms_exact_value(q.x, mms);
        area += q.w;
    }

    if (!(area > 0.0)) {
        throw std::runtime_error("RT0 boundary face has zero area.");
    }

    return integ / area;
}

static AssembledSystem assemble_rt0_mixed_poisson_mms(
    const PolyMesh& m,
    const Rt0DofMap& dm,
    const std::string& mms
) {
    AssembledSystem sys;
    sys.b.assign(dm.nTotalDofs, Real(0));

    const auto geoms = build_rt0_cell_geoms(m, dm);

    const std::string sparseMode = memoirs_sparse_mode();
    if (sparseMode == "fixed_csr") {
        sparse_init_fixed_pattern(sys.A, build_rt0_mixed_pattern(m, dm, geoms));
    } else if (sparseMode == "legacy") {
        sys.A.rows.resize(dm.nTotalDofs);
    } else {
        throw std::runtime_error("Unsupported sparse mode in RT0 mixed Poisson.");
    }

    // Flux mass and divergence/gradient coupling.
    for (int c = 0; c < dm.nScalarDofs; ++c) {
        const auto& K = geoms[c];
        const int pRow = dm.nFluxDofs + c;

        // Cell source integral F_c = ∫_K f dx.
        double F = 0.0;
        for (const auto& q : rt0_cell_volume_quadrature(m, K, false)) {
            F += q.w * mms_rhs_value(q.x, mms);
        }

        // Symmetric saddle uses -B q = -F.
        sys.b[pRow] += Real(-F);

        for (int a = 0; a < (int)K.faces.size(); ++a) {
            const int fa = K.faces[a];
            const int la = rt0_local_face_index(m, K, fa);
            const int sa = rt0_cell_face_sign(m, c, fa);

            // Coupling:
            // flux row: -B^T u
            // scalar row: -B q
            sparse_add(sys.A, fa, pRow, Real(-sa));
            sparse_add(sys.A, pRow, fa, Real(-sa));

            for (int b = 0; b < (int)K.faces.size(); ++b) {
                const int fb = K.faces[b];
                const int lb = rt0_local_face_index(m, K, fb);
                const int sb = rt0_cell_face_sign(m, c, fb);

                double mij = 0.0;
                for (const auto& q : rt0_cell_volume_quadrature(m, K, false)) {
                    const Vec3 phia = rt0_eval_local_basis(m, K, la, q.x);
                    const Vec3 phib = rt0_eval_local_basis(m, K, lb, q.x);
                    mij += q.w * dot3(phia, phib);
                }

                sparse_add(sys.A, fa, fb, Real(sa * sb * mij));
            }
        }
    }

    // Dirichlet scalar boundary contribution:
    // flux equation RHS = - <g, tau.n>; for RT0 face basis tau.n = 1/area.
    for (int f = (int)m.neighbour.size(); f < (int)m.faces.size(); ++f) {
        const double gAvg = rt0_boundary_g_face_average(m, f, mms);
        sys.b[f] += Real(-gAvg);
    }

    return sys;
}

static Vec3 rt0_reconstruct_flux_in_cell(
    const PolyMesh& m,
    const Rt0CellGeom& K,
    int c,
    const std::vector<Real>& x,
    const Vec3& xp
) {
    Vec3 sig{};

    for (int f : K.faces) {
        const int lf = rt0_local_face_index(m, K, f);
        const int s = rt0_cell_face_sign(m, c, f);
        const double qf = double(x[f]);
        const Vec3 phi = rt0_eval_local_basis(m, K, lf, xp);
        sig = rt0_add(sig, rt0_mul(s * qf, phi));
    }

    return sig;
}

static Rt0ErrorReport compute_rt0_mixed_errors(
    const PolyMesh& m,
    const Rt0DofMap& dm,
    const std::string& mms,
    const std::vector<Real>& x
) {
    if ((int)x.size() != dm.nTotalDofs) {
        throw std::runtime_error("RT0 error size mismatch.");
    }

    const auto geoms = build_rt0_cell_geoms(m, dm);

    Rt0ErrorReport er;

    double scalar2 = 0.0;
    double flux2 = 0.0;
    double divCell2 = 0.0;
    double divCellMax = 0.0;
    double cons2 = 0.0;
    double consMax = 0.0;

    for (int c = 0; c < dm.nScalarDofs; ++c) {
        const auto& K = geoms[c];
        const double uh = double(x[dm.nFluxDofs + c]);

        double sourceInt = 0.0;

        for (const auto& q : rt0_cell_volume_quadrature(m, K, true)) {
            const Vec3 sig = rt0_reconstruct_flux_in_cell(m, K, c, x, q.x);
            const Vec3 sigExact = rt0_mul(-1.0, mms_exact_grad(q.x, mms));

            const Vec3 es = {
                sig.x - sigExact.x,
                sig.y - sigExact.y,
                sig.z - sigExact.z
            };

            const double eu = uh - mms_exact_value(q.x, mms);
            scalar2 += q.w * eu * eu;
            flux2 += q.w * dot3(es, es);
            sourceInt += q.w * mms_rhs_value(q.x, mms);
        }

        double balance = 0.0;
        for (int f : K.faces) {
            const int s = rt0_cell_face_sign(m, c, f);
            balance += s * double(x[f]);
        }

        const double cons = balance - sourceInt;
        cons2 += cons * cons;
        consMax = std::max(consMax, std::abs(cons));

        const double divErr = balance / K.volume - sourceInt / K.volume;
        divCell2 += K.volume * divErr * divErr;
        divCellMax = std::max(divCellMax, std::abs(divErr));
    }

    er.scalarL2 = std::sqrt(scalar2);
    er.fluxL2 = std::sqrt(flux2);
    er.cellConservationL2 = std::sqrt(cons2);
    er.cellConservationAbsMax = consMax;
    er.divCellL2 = std::sqrt(divCell2);
    er.divCellAbsMax = divCellMax;

    return er;
}

static void probe_rt0_mixed_system(
    const PolyMesh& m,
    const Rt0DofMap& dm,
    const AssembledSystem& sys,
    int diagLevel
) {
    std::cout << "---------------- RT0 mixed Poisson system probe -----------------\n";
    std::cout << "space                     = " << dm.resolvedSpace << "\n";
    std::cout << "continuity                = H(div), one normal flux dof per face\n";
    std::cout << "scalarSpace               = cellwise P0/Q0\n";
    std::cout << "nCells                    = " << m.cells.size() << "\n";
    std::cout << "nFaces                    = " << m.faces.size() << "\n";
    std::cout << "nFluxDofs                 = " << dm.nFluxDofs << "\n";
    std::cout << "nScalarDofs               = " << dm.nScalarDofs << "\n";
    std::cout << "nTotalDofs                = " << dm.nTotalDofs << "\n";
    std::cout << "rows                      = " << sparse_nrows(sys.A) << "\n";
    std::cout << "nnz                       = " << sparse_nnz(sys.A) << "\n";
    std::cout << "sparseMode                = " << memoirs_sparse_mode() << "\n";

    int missingDiag = 0;
    double minDiag = std::numeric_limits<double>::infinity();
    double maxDiag = 0.0;

    for (int i = 0; i < dm.nTotalDofs; ++i) {
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


// ============================================================================
// RT0 visualization export
// ============================================================================
//
// ParaView/VTK does not natively store an RT0 finite-element basis here.
// We export reconstructed cell-centered quantities:
//   - rt0_scalar_u: P0 scalar unknown
//   - rt0_flux_centroid: RT0 reconstructed flux at cell centroid
//   - exact scalar/flux at centroid
//   - error and conservation diagnostics
// ============================================================================

#include <fstream>
#include <iomanip>

static void write_rt0_mixed_vtu(
    const std::string& filename,
    const PolyMesh& m,
    const Rt0DofMap& dm,
    const std::string& mms,
    const std::vector<Real>& x
) {
    if ((int)x.size() != dm.nTotalDofs) {
        throw std::runtime_error("RT0 VTU write size mismatch.");
    }

    const auto geoms = build_rt0_cell_geoms(m, dm);

    std::ofstream os(filename);
    if (!os) {
        throw std::runtime_error("Could not open RT0 VTU file for writing: " + filename);
    }

    os << std::setprecision(16);
    os << "<?xml version=\"1.0\"?>\n";
    os << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    os << "  <UnstructuredGrid>\n";
    os << "    <Piece NumberOfPoints=\"" << m.points.size()
       << "\" NumberOfCells=\"" << m.cells.size() << "\">\n";

    os << "      <Points>\n";
    os << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (const Vec3& pnt : m.points) {
        os << "          " << pnt.x << " " << pnt.y << " " << pnt.z << "\n";
    }
    os << "        </DataArray>\n";
    os << "      </Points>\n";

    os << "      <Cells>\n";

    os << "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n";
    for (const auto& K : geoms) {
        os << "          ";
        for (int v : K.verts) {
            os << v << " ";
        }
        os << "\n";
    }
    os << "        </DataArray>\n";

    os << "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n";
    long long off = 0;
    for (const auto& K : geoms) {
        off += (long long)K.verts.size();
        os << "          " << off << "\n";
    }
    os << "        </DataArray>\n";

    os << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    for (const auto& K : geoms) {
        const int vtkType = (K.kind == Rt0CellKind::Tet) ? 10 : 12;
        os << "          " << vtkType << "\n";
    }
    os << "        </DataArray>\n";

    os << "      </Cells>\n";

    os << "      <CellData Scalars=\"rt0_scalar_u\" Vectors=\"rt0_flux_centroid\">\n";

    auto write_scalar_array = [&](const std::string& name, const std::vector<double>& vals) {
        os << "        <DataArray type=\"Float64\" Name=\"" << name << "\" format=\"ascii\">\n";
        for (double v : vals) {
            os << "          " << v << "\n";
        }
        os << "        </DataArray>\n";
    };

    auto write_vector_array = [&](const std::string& name, const std::vector<Vec3>& vals) {
        os << "        <DataArray type=\"Float64\" Name=\"" << name
           << "\" NumberOfComponents=\"3\" format=\"ascii\">\n";
        for (const Vec3& v : vals) {
            os << "          " << v.x << " " << v.y << " " << v.z << "\n";
        }
        os << "        </DataArray>\n";
    };

    std::vector<double> uh(dm.nScalarDofs);
    std::vector<double> uExact(dm.nScalarDofs);
    std::vector<double> uErr(dm.nScalarDofs);
    std::vector<Vec3> sig(dm.nScalarDofs);
    std::vector<Vec3> sigExact(dm.nScalarDofs);
    std::vector<Vec3> sigErr(dm.nScalarDofs);
    std::vector<double> fluxErrMag(dm.nScalarDofs);
    std::vector<double> divCellErr(dm.nScalarDofs);
    std::vector<double> conservationResidual(dm.nScalarDofs);
    std::vector<double> sourceIntegral(dm.nScalarDofs);
    std::vector<double> fluxBalance(dm.nScalarDofs);

    for (int c = 0; c < dm.nScalarDofs; ++c) {
        const auto& K = geoms[c];
        const Vec3 xc = K.centroid;

        uh[c] = double(x[dm.nFluxDofs + c]);
        uExact[c] = mms_exact_value(xc, mms);
        uErr[c] = uh[c] - uExact[c];

        sig[c] = rt0_reconstruct_flux_in_cell(m, K, c, x, xc);
        sigExact[c] = rt0_mul(-1.0, mms_exact_grad(xc, mms));
        sigErr[c] = {
            sig[c].x - sigExact[c].x,
            sig[c].y - sigExact[c].y,
            sig[c].z - sigExact[c].z
        };
        fluxErrMag[c] = std::sqrt(dot3(sigErr[c], sigErr[c]));

        double src = 0.0;
        for (const auto& q : rt0_cell_volume_quadrature(m, K, true)) {
            src += q.w * mms_rhs_value(q.x, mms);
        }

        double bal = 0.0;
        for (int f : K.faces) {
            bal += rt0_cell_face_sign(m, c, f) * double(x[f]);
        }

        sourceIntegral[c] = src;
        fluxBalance[c] = bal;
        conservationResidual[c] = bal - src;
        divCellErr[c] = (bal - src) / K.volume;
    }

    write_scalar_array("rt0_scalar_u", uh);
    write_scalar_array("exact_scalar_centroid", uExact);
    write_scalar_array("scalar_error_centroid", uErr);

    write_vector_array("rt0_flux_centroid", sig);
    write_vector_array("exact_flux_centroid", sigExact);
    write_vector_array("flux_error_centroid", sigErr);

    write_scalar_array("flux_error_mag_centroid", fluxErrMag);
    write_scalar_array("div_cell_error", divCellErr);
    write_scalar_array("cell_conservation_residual", conservationResidual);
    write_scalar_array("source_integral", sourceIntegral);
    write_scalar_array("flux_balance", fluxBalance);

    os << "      </CellData>\n";

    os << "    </Piece>\n";
    os << "  </UnstructuredGrid>\n";
    os << "</VTKFile>\n";
}

static std::string parse_rt0_vtu_file_from_cli(int argc, char** argv) {
    for (int i = 1; i + 1 < argc; ++i) {
        const std::string a = argv[i];
        if (a == "-vtuFile" || a == "-outputVtu" || a == "-writeVtu") {
            return argv[i+1];
        }
    }
    return "";
}

