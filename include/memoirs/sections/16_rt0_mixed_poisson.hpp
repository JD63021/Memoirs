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
// Experimental RT0 lumped-Schur approximate solver
// ============================================================================
// This is an additive algorithmic probe, not a replacement for the verified
// monolithic GMRES path.
//
// Original conforming RT0 saddle system:
//
//   [ M  -C^T ] [q] = [r]
//   [-C   0  ] [u]   [-F]
//
// Exact elimination gives S = C M^{-1} C^T.  That exact Schur complement is
// not cheaply assembled because M^{-1} is global for conforming RT0 face DOFs.
// This probe uses the diagonal/lumped flux mass D instead:
//
//   S_lump = C D^{-1} C^T
//
// Then it solves S_lump u = F - C D^{-1} r with PCG+AMG/diagscale through the
// existing raw scalar HYPRE helper, and reconstructs q = D^{-1}(r + C^T u).
//
// Use it to answer: "does a Poisson-like Schur path make RT0 hex cheap?".
// It is also the preconditioner target for a later true outer FGMRES solve on
// the original saddle matrix.
// ============================================================================

static std::string rt0_env_string(const char* name, const std::string& defval) {
    const char* e = std::getenv(name);
    if (!e || !*e) return defval;
    return std::string(e);
}

static std::string memoirs_rt0_solve_mode() {
    std::string v = rt0_env_string("MEMOIRS_RT0_SOLVE_MODE", "gmres");
    for (char& c : v) c = char(std::tolower((unsigned char)c));
    if (v == "raw_gmres" || v == "monolithic_gmres") return "gmres";
    if (v == "schur" || v == "lumped" || v == "lumped_schur" || v == "diag_schur") return "lumped_schur";
    if (v == "block" || v == "block_schur" || v == "block_schur_gmres" ||
        v == "block_schur_fgmres" || v == "fgmres_schur") return "block_schur_gmres";
    return v;
}

#if defined(MEMOIRS_USE_HYPRE)
static SolveReport solve_rt0_lumped_schur_approx(
    const PolyMesh& m,
    const Rt0DofMap& dm,
    const AssembledSystem& sys,
    const Options& opt,
    std::vector<Real>& xOut
) {
    if ((int)sys.b.size() != dm.nTotalDofs) {
        throw std::runtime_error("RT0 lumped-Schur size mismatch in RHS.");
    }
    if (sparse_nrows(sys.A) != dm.nTotalDofs) {
        throw std::runtime_error("RT0 lumped-Schur size mismatch in matrix.");
    }

    const int nF = dm.nFluxDofs;
    const int nC = dm.nScalarDofs;

    const auto geoms = build_rt0_cell_geoms(m, dm);

    std::vector<double> dinv(nF, 0.0);
    double minDiag = std::numeric_limits<double>::infinity();
    double maxDiag = 0.0;
    int badDiag = 0;

    for (int f = 0; f < nF; ++f) {
        Real dR = Real(0);
        const bool ok = sparse_get_value(sys.A, f, f, dR);
        const double d = double(dR);
        if (!ok || !(d > 0.0) || !std::isfinite(d)) {
            ++badDiag;
            continue;
        }
        dinv[f] = 1.0 / d;
        minDiag = std::min(minDiag, d);
        maxDiag = std::max(maxDiag, d);
    }

    if (badDiag != 0) {
        throw std::runtime_error("RT0 lumped-Schur found non-positive flux mass diagonal entries.");
    }

    std::vector<std::vector<int>> rowCols(nC);
    for (int c = 0; c < nC; ++c) rowCols[c].push_back(c);

    auto add_graph_pair = [&](int c, int d) {
        if (c >= 0 && d >= 0) rowCols[c].push_back(d);
    };

    for (int f = 0; f < nF; ++f) {
        const int co = m.owner[f];
        add_graph_pair(co, co);
        if (f < (int)m.neighbour.size()) {
            const int cn = m.neighbour[f];
            add_graph_pair(co, cn);
            add_graph_pair(cn, co);
            add_graph_pair(cn, cn);
        }
    }

    SparseRows S;
    if (memoirs_sparse_mode() == "fixed_csr") {
        sparse_init_fixed_pattern(S, rowCols);
    } else {
        S.rows.resize(nC);
    }

    std::vector<Real> rhsS(nC, Real(0));

    // Assemble S_lump = C D^{-1} C^T face-by-face.
    for (int f = 0; f < nF; ++f) {
        const double w = dinv[f];
        const int co = m.owner[f];
        sparse_add(S, co, co, Real((+1.0) * w * (+1.0)));
        if (f < (int)m.neighbour.size()) {
            const int cn = m.neighbour[f];
            sparse_add(S, co, cn, Real((+1.0) * w * (-1.0)));
            sparse_add(S, cn, co, Real((-1.0) * w * (+1.0)));
            sparse_add(S, cn, cn, Real((-1.0) * w * (-1.0)));
        }
    }

    // rhs = F - C D^{-1} r, where the assembled scalar RHS stores -F.
    for (int c = 0; c < nC; ++c) {
        rhsS[c] = Real(-double(sys.b[nF + c]));
    }
    for (int c = 0; c < nC; ++c) {
        for (int f : geoms[c].faces) {
            const int s = rt0_cell_face_sign(m, c, f);
            rhsS[c] -= Real(double(s) * dinv[f] * double(sys.b[f]));
        }
    }

    Options sopt = opt;
    sopt.solver = "pcg";
    sopt.precond = rt0_env_string("MEMOIRS_RT0_SCHUR_PRECOND", "amg");
    sopt.tol = memoirs_env_double("MEMOIRS_RT0_SCHUR_TOL", opt.tol);
    sopt.maxit = memoirs_env_int("MEMOIRS_RT0_SCHUR_MAXIT", opt.maxit);

    std::vector<Real> u;
    auto t0 = std::chrono::steady_clock::now();
    SolveReport rep = solve_hypre_ij_pcg_raw(S, rhsS, sopt, u);
    auto t1 = std::chrono::steady_clock::now();

    if ((int)u.size() != nC) {
        throw std::runtime_error("RT0 lumped-Schur scalar solve returned wrong size.");
    }

    xOut.assign(dm.nTotalDofs, Real(0));

    // q = D^{-1}(r + C^T u).
    for (int f = 0; f < nF; ++f) {
        double val = double(sys.b[f]);
        const int co = m.owner[f];
        val += double(u[co]);
        if (f < (int)m.neighbour.size()) {
            const int cn = m.neighbour[f];
            val -= double(u[cn]);
        }
        xOut[f] = Real(dinv[f] * val);
    }
    for (int c = 0; c < nC; ++c) {
        xOut[nF + c] = u[c];
    }

    rep.solveSeconds = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "---------------- RT0 lumped Schur probe -----------------\n";
    std::cout << "schurApprox                  = C diag(M)^{-1} C^T\n";
    std::cout << "fluxRecovery                 = q = diag(M)^{-1}(r + C^T u)\n";
    std::cout << "schurPrecond                 = " << sopt.precond << "\n";
    std::cout << "schurIterations              = " << rep.iterations << "\n";
    std::cout << "schurFinalRelativeResidual   = " << std::scientific << rep.finalRelRes << std::defaultfloat << "\n";
    std::cout << "fluxMassDiagMin              = " << std::scientific << minDiag << "\n";
    std::cout << "fluxMassDiagMax              = " << maxDiag << std::defaultfloat << "\n";
    std::cout << "---------------------------------------------------------\n";

    return rep;
}


// ============================================================================
// Experimental RT0 block-Schur preconditioned host FGMRES
// ============================================================================
// This mode solves the ORIGINAL assembled RT0 saddle system using a small host
// restarted FGMRES.  The lumped Schur operator is used only as a preconditioner:
//
//   A = [ M  -C^T ]
//       [-C   0   ]
//
// Given a residual (rq, ru), apply the approximate block solve:
//
//   S_lump zu = -ru - C D^{-1} rq
//   zq        =  D^{-1} (rq + C^T zu)
//
// where D=diag(M), S_lump=C D^{-1} C^T.  This keeps the final Krylov solution
// tied to the exact saddle matrix while testing whether the Poisson-like Schur
// path removes the grid-dependent GMRES growth.
// ============================================================================

struct Rt0LumpedSchurData {
    SparseRows S;
    std::vector<double> dinv;
    std::vector<Rt0CellGeom> geoms;
    int nF = 0;
    int nC = 0;
    double minDiag = 0.0;
    double maxDiag = 0.0;
};

struct Rt0BlockSchurStats {
    int schurApplications = 0;
    int schurIterationsTotal = 0;
    int schurIterationsMax = 0;
    double schurFinalRelLast = 0.0;
    int schurSetupApplications = 0;
    int schurReuseSetup = 0;
    int schurReuseEvery = 0;
    double hypreMatrixInsertSeconds = 0.0;
    double hypreMatrixMigrateSeconds = 0.0;
    double hypreVectorInsertSeconds = 0.0;
    double hypreVectorMigrateSeconds = 0.0;
    double hypreSetupSeconds = 0.0;
    double hypreSolveOnlySeconds = 0.0;
    double hypreGetSolutionSeconds = 0.0;
    double hypreDestroyFinalizeSeconds = 0.0;
};

static double rt0_vec_dot_real(const std::vector<Real>& a, const std::vector<Real>& b) {
    if (a.size() != b.size()) throw std::runtime_error("rt0_vec_dot size mismatch");
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) s += double(a[i]) * double(b[i]);
    return s;
}

static double rt0_vec_norm_real(const std::vector<Real>& a) {
    return std::sqrt(std::max(0.0, rt0_vec_dot_real(a, a)));
}

static std::vector<Real> rt0_sparse_matvec_all(
    const SparseRows& A,
    const std::vector<Real>& x
) {
    const int n = sparse_nrows(A);
    if ((int)x.size() != n) throw std::runtime_error("rt0_sparse_matvec_all size mismatch");
    std::vector<Real> y(n, Real(0));
    for (int i = 0; i < n; ++i) y[i] = sparse_matvec_row(A, i, x);
    return y;
}

static Rt0LumpedSchurData rt0_build_lumped_schur_data(
    const PolyMesh& m,
    const Rt0DofMap& dm,
    const AssembledSystem& sys
) {
    if ((int)sys.b.size() != dm.nTotalDofs) {
        throw std::runtime_error("RT0 block-Schur size mismatch in RHS.");
    }
    if (sparse_nrows(sys.A) != dm.nTotalDofs) {
        throw std::runtime_error("RT0 block-Schur size mismatch in matrix.");
    }

    Rt0LumpedSchurData d;
    d.nF = dm.nFluxDofs;
    d.nC = dm.nScalarDofs;
    d.geoms = build_rt0_cell_geoms(m, dm);
    d.dinv.assign(d.nF, 0.0);

    d.minDiag = std::numeric_limits<double>::infinity();
    d.maxDiag = 0.0;
    int badDiag = 0;

    for (int f = 0; f < d.nF; ++f) {
        Real diagR = Real(0);
        const bool ok = sparse_get_value(sys.A, f, f, diagR);
        const double diag = double(diagR);
        if (!ok || !(diag > 0.0) || !std::isfinite(diag)) {
            ++badDiag;
            continue;
        }
        d.dinv[f] = 1.0 / diag;
        d.minDiag = std::min(d.minDiag, diag);
        d.maxDiag = std::max(d.maxDiag, diag);
    }

    if (badDiag != 0) {
        throw std::runtime_error("RT0 block-Schur found non-positive flux mass diagonal entries.");
    }

    std::vector<std::vector<int>> rowCols(d.nC);
    for (int c = 0; c < d.nC; ++c) rowCols[c].push_back(c);

    auto add_graph_pair = [&](int c, int e) {
        if (c >= 0 && e >= 0) rowCols[c].push_back(e);
    };

    for (int f = 0; f < d.nF; ++f) {
        const int co = m.owner[f];
        add_graph_pair(co, co);
        if (f < (int)m.neighbour.size()) {
            const int cn = m.neighbour[f];
            add_graph_pair(co, cn);
            add_graph_pair(cn, co);
            add_graph_pair(cn, cn);
        }
    }

    if (memoirs_sparse_mode() == "fixed_csr") {
        sparse_init_fixed_pattern(d.S, rowCols);
    } else {
        d.S.rows.resize(d.nC);
    }

    // S_lump = C D^{-1} C^T.
    for (int f = 0; f < d.nF; ++f) {
        const double w = d.dinv[f];
        const int co = m.owner[f];
        sparse_add(d.S, co, co, Real(w));
        if (f < (int)m.neighbour.size()) {
            const int cn = m.neighbour[f];
            sparse_add(d.S, co, cn, Real(-w));
            sparse_add(d.S, cn, co, Real(-w));
            sparse_add(d.S, cn, cn, Real(w));
        }
    }

    return d;
}

static std::vector<double> rt0_solve_small_normal_eq(
    const std::vector<std::vector<double>>& H,
    double beta,
    int k
) {
    // Least squares min || beta e0 - H_k y ||, where H has rows k+1 and cols k.
    std::vector<std::vector<double>> A(k, std::vector<double>(k, 0.0));
    std::vector<double> g(k, 0.0);

    for (int i = 0; i < k; ++i) {
        g[i] = H[0][i] * beta;
        for (int j = 0; j < k; ++j) {
            double s = 0.0;
            for (int r = 0; r <= k; ++r) s += H[r][i] * H[r][j];
            A[i][j] = s;
        }
    }

    for (int col = 0; col < k; ++col) {
        int piv = col;
        double best = std::abs(A[col][col]);
        for (int r = col + 1; r < k; ++r) {
            const double ar = std::abs(A[r][col]);
            if (ar > best) { best = ar; piv = r; }
        }
        if (best < 1e-300) continue;
        if (piv != col) {
            std::swap(A[piv], A[col]);
            std::swap(g[piv], g[col]);
        }
        const double diag = A[col][col];
        for (int r = col + 1; r < k; ++r) {
            const double m = A[r][col] / diag;
            if (m == 0.0) continue;
            for (int c = col; c < k; ++c) A[r][c] -= m * A[col][c];
            g[r] -= m * g[col];
        }
    }

    std::vector<double> y(k, 0.0);
    for (int i = k - 1; i >= 0; --i) {
        double s = g[i];
        for (int j = i + 1; j < k; ++j) s -= A[i][j] * y[j];
        if (std::abs(A[i][i]) > 1e-300) y[i] = s / A[i][i];
    }
    return y;
}

static double rt0_fgmres_small_residual_norm(
    const std::vector<std::vector<double>>& H,
    const std::vector<double>& y,
    double beta,
    int k
) {
    double r2 = 0.0;
    for (int row = 0; row <= k; ++row) {
        double s = (row == 0 ? beta : 0.0);
        for (int col = 0; col < k; ++col) s -= H[row][col] * y[col];
        r2 += s * s;
    }
    return std::sqrt(std::max(0.0, r2));
}


// Reusable HYPRE PCG+AMG object for repeated scalar Schur applications.
// This is intentionally local to the RT0 experimental path: the outer FGMRES
// remains host-side, but the expensive AMG hierarchy for S_lump is built once
// and reused for many preconditioner applications.
struct Rt0ReusableSchurPcg {
#if defined(MEMOIRS_USE_HYPRE)
    bool active = false;
    bool useDevice = false;
    HYPRE_Int n = 0;
    MPI_Comm comm = MPI_COMM_WORLD;

    HYPRE_IJMatrix ijA = nullptr;
    HYPRE_ParCSRMatrix parA = nullptr;
    HYPRE_IJVector ijb = nullptr;
    HYPRE_IJVector ijx = nullptr;
    HYPRE_ParVector parb = nullptr;
    HYPRE_ParVector parx = nullptr;
    HYPRE_Solver pcg = nullptr;
    HYPRE_Solver precond = nullptr;

    std::vector<HYPRE_Int> rows;
    std::vector<HYPRE_Int> ncols;
    std::vector<HYPRE_Int> cols;
    std::vector<HYPRE_Complex> vals;
    std::vector<HYPRE_Complex> bvals;
    std::vector<HYPRE_Complex> xvals;

    static double elapsed(const std::chrono::steady_clock::time_point& a,
                          const std::chrono::steady_clock::time_point& b) {
        return std::chrono::duration<double>(b - a).count();
    }

    void setup(const SparseRows& A, const Options& opt, Rt0BlockSchurStats& stats) {
        if (active) return;

        const int nHost = sparse_nrows(A);
        if (nHost <= 0) throw std::runtime_error("RT0 reusable Schur PCG received empty matrix.");

        int size = 1;
        MPI_Comm_size(comm, &size);
        if (size != 1) {
            throw std::runtime_error("RT0 reusable Schur PCG currently supports one MPI rank only.");
        }

        const HYPRE_MemoryLocation requestedMem = parse_hypre_memory_location(opt.hypreMemory);
        useDevice = (requestedMem == HYPRE_MEMORY_DEVICE);

        hypre_check(HYPRE_Init(), "HYPRE_Init rt0 reusable Schur");
        if (useDevice) {
            hypre_check(HYPRE_SetMemoryLocation(HYPRE_MEMORY_DEVICE), "HYPRE_SetMemoryLocation DEVICE rt0 reusable Schur");
            hypre_check(HYPRE_SetExecutionPolicy(HYPRE_EXEC_DEVICE), "HYPRE_SetExecutionPolicy DEVICE rt0 reusable Schur");
        } else {
            hypre_check(HYPRE_SetMemoryLocation(HYPRE_MEMORY_HOST), "HYPRE_SetMemoryLocation HOST rt0 reusable Schur");
            hypre_check(HYPRE_SetExecutionPolicy(HYPRE_EXEC_HOST), "HYPRE_SetExecutionPolicy HOST rt0 reusable Schur");
        }

        n = (HYPRE_Int)nHost;
        const HYPRE_BigInt ilower = 0;
        const HYPRE_BigInt iupper = n - 1;

        auto tMat0 = std::chrono::steady_clock::now();

        hypre_check(HYPRE_IJMatrixCreate(comm, ilower, iupper, ilower, iupper, &ijA), "HYPRE_IJMatrixCreate rt0 reusable Schur");
        hypre_check(HYPRE_IJMatrixSetObjectType(ijA, HYPRE_PARCSR), "HYPRE_IJMatrixSetObjectType rt0 reusable Schur");

        rows.resize(n);
        ncols.resize(n);
        cols.clear();
        vals.clear();
        cols.reserve(sparse_nnz(A));
        vals.reserve(sparse_nnz(A));

        for (HYPRE_Int i = 0; i < n; ++i) {
            rows[i] = i;
            if (A.fixedPattern) {
                const auto& c = sparse_cols_row(A, i);
                ncols[i] = (HYPRE_Int)c.size();
                for (int k = 0; k < (int)c.size(); ++k) {
                    cols.push_back((HYPRE_Int)c[k]);
                    vals.push_back((HYPRE_Complex)sparse_value_at(A, i, k));
                }
            } else {
                ncols[i] = (HYPRE_Int)A.rows[i].size();
                for (const auto& kv : A.rows[i]) {
                    cols.push_back((HYPRE_Int)kv.first);
                    vals.push_back((HYPRE_Complex)kv.second);
                }
            }
        }

        hypre_check(HYPRE_IJMatrixSetRowSizes(ijA, ncols.data()), "HYPRE_IJMatrixSetRowSizes rt0 reusable Schur");
        hypre_initialize_ij_matrix_for_host_insertion(ijA);
        hypre_check(HYPRE_IJMatrixSetValues(ijA, n, ncols.data(), rows.data(), cols.data(), vals.data()),
                    "HYPRE_IJMatrixSetValues rt0 reusable Schur");
        hypre_check(HYPRE_IJMatrixAssemble(ijA), "HYPRE_IJMatrixAssemble rt0 reusable Schur");

        auto tMat1 = std::chrono::steady_clock::now();
        stats.hypreMatrixInsertSeconds += elapsed(tMat0, tMat1);

        auto tMig0 = std::chrono::steady_clock::now();
        if (useDevice) {
            hypre_check(HYPRE_IJMatrixMigrate(ijA, HYPRE_MEMORY_DEVICE), "HYPRE_IJMatrixMigrate rt0 reusable Schur");
        }
        hypre_check(HYPRE_IJMatrixGetObject(ijA, (void**)&parA), "HYPRE_IJMatrixGetObject rt0 reusable Schur");
        auto tMig1 = std::chrono::steady_clock::now();
        stats.hypreMatrixMigrateSeconds += elapsed(tMig0, tMig1);

        bvals.assign(n, HYPRE_Complex(0));
        xvals.assign(n, HYPRE_Complex(0));

        auto tVec0 = std::chrono::steady_clock::now();
        hypre_check(HYPRE_IJVectorCreate(comm, ilower, iupper, &ijb), "HYPRE_IJVectorCreate b rt0 reusable Schur");
        hypre_check(HYPRE_IJVectorSetObjectType(ijb, HYPRE_PARCSR), "HYPRE_IJVectorSetObjectType b rt0 reusable Schur");
        hypre_initialize_ij_vector_for_host_insertion(ijb);
        hypre_check(HYPRE_IJVectorCreate(comm, ilower, iupper, &ijx), "HYPRE_IJVectorCreate x rt0 reusable Schur");
        hypre_check(HYPRE_IJVectorSetObjectType(ijx, HYPRE_PARCSR), "HYPRE_IJVectorSetObjectType x rt0 reusable Schur");
        hypre_initialize_ij_vector_for_host_insertion(ijx);
        hypre_check(HYPRE_IJVectorSetValues(ijb, n, rows.data(), bvals.data()), "HYPRE_IJVectorSetValues b setup rt0 reusable Schur");
        hypre_check(HYPRE_IJVectorSetValues(ijx, n, rows.data(), xvals.data()), "HYPRE_IJVectorSetValues x setup rt0 reusable Schur");
        hypre_check(HYPRE_IJVectorAssemble(ijb), "HYPRE_IJVectorAssemble b setup rt0 reusable Schur");
        hypre_check(HYPRE_IJVectorAssemble(ijx), "HYPRE_IJVectorAssemble x setup rt0 reusable Schur");
        auto tVec1 = std::chrono::steady_clock::now();
        stats.hypreVectorInsertSeconds += elapsed(tVec0, tVec1);

        auto tVecMig0 = std::chrono::steady_clock::now();
        if (useDevice) {
            hypre_check(HYPRE_IJVectorMigrate(ijb, HYPRE_MEMORY_DEVICE), "HYPRE_IJVectorMigrate b setup rt0 reusable Schur");
            hypre_check(HYPRE_IJVectorMigrate(ijx, HYPRE_MEMORY_DEVICE), "HYPRE_IJVectorMigrate x setup rt0 reusable Schur");
        }
        hypre_check(HYPRE_IJVectorGetObject(ijb, (void**)&parb), "HYPRE_IJVectorGetObject b rt0 reusable Schur");
        hypre_check(HYPRE_IJVectorGetObject(ijx, (void**)&parx), "HYPRE_IJVectorGetObject x rt0 reusable Schur");
        auto tVecMig1 = std::chrono::steady_clock::now();
        stats.hypreVectorMigrateSeconds += elapsed(tVecMig0, tVecMig1);

        auto tSetup0 = std::chrono::steady_clock::now();
        hypre_check(HYPRE_ParCSRPCGCreate(comm, &pcg), "HYPRE_ParCSRPCGCreate rt0 reusable Schur");
        hypre_check(HYPRE_PCGSetMaxIter(pcg, opt.maxit), "HYPRE_PCGSetMaxIter rt0 reusable Schur");
        hypre_check(HYPRE_PCGSetTol(pcg, opt.tol), "HYPRE_PCGSetTol rt0 reusable Schur");
        hypre_check(HYPRE_PCGSetTwoNorm(pcg, 1), "HYPRE_PCGSetTwoNorm rt0 reusable Schur");
        hypre_check(HYPRE_PCGSetPrintLevel(pcg, opt.hyprePrint), "HYPRE_PCGSetPrintLevel rt0 reusable Schur");
        hypre_check(HYPRE_PCGSetLogging(pcg, 1), "HYPRE_PCGSetLogging rt0 reusable Schur");

        const std::string pre = lower_copy(opt.precond);
        if (pre == "diagscale") {
            hypre_check(HYPRE_PCGSetPrecond(
                pcg,
                (HYPRE_PtrToSolverFcn)HYPRE_ParCSRDiagScale,
                (HYPRE_PtrToSolverFcn)HYPRE_ParCSRDiagScaleSetup,
                nullptr),
                "HYPRE_PCGSetPrecond diag rt0 reusable Schur");
        } else if (pre == "amg" || pre == "boomeramg") {
            hypre_check(HYPRE_BoomerAMGCreate(&precond), "HYPRE_BoomerAMGCreate rt0 reusable Schur");
            hypre_check(HYPRE_BoomerAMGSetPrintLevel(precond, opt.hyprePrint), "HYPRE_BoomerAMGSetPrintLevel rt0 reusable Schur");
            hypre_check(HYPRE_BoomerAMGSetCoarsenType(precond, memoirs_env_int("MEMOIRS_AMG_COARSEN", 8)), "HYPRE_BoomerAMGSetCoarsenType rt0 reusable Schur");
            hypre_check(HYPRE_BoomerAMGSetInterpType(precond, memoirs_env_int("MEMOIRS_AMG_INTERP", 6)), "HYPRE_BoomerAMGSetInterpType rt0 reusable Schur");
            hypre_check(HYPRE_BoomerAMGSetRelaxType(precond, memoirs_env_int("MEMOIRS_AMG_RELAX", 18)), "HYPRE_BoomerAMGSetRelaxType rt0 reusable Schur");
            hypre_check(HYPRE_BoomerAMGSetNumSweeps(precond, memoirs_env_int("MEMOIRS_AMG_SWEEPS", 1)), "HYPRE_BoomerAMGSetNumSweeps rt0 reusable Schur");
            hypre_check(HYPRE_BoomerAMGSetTol(precond, 0.0), "HYPRE_BoomerAMGSetTol rt0 reusable Schur");
            hypre_check(HYPRE_BoomerAMGSetMaxIter(precond, 1), "HYPRE_BoomerAMGSetMaxIter rt0 reusable Schur");
            hypre_check(HYPRE_BoomerAMGSetRelaxOrder(precond, 0), "HYPRE_BoomerAMGSetRelaxOrder rt0 reusable Schur");
            hypre_check(HYPRE_BoomerAMGSetPMaxElmts(precond, memoirs_env_int("MEMOIRS_AMG_PMAX", 4)), "HYPRE_BoomerAMGSetPMaxElmts rt0 reusable Schur");
            hypre_check(HYPRE_BoomerAMGSetKeepTranspose(precond, memoirs_env_int("MEMOIRS_AMG_KEEP_TRANSPOSE", 1)), "HYPRE_BoomerAMGSetKeepTranspose rt0 reusable Schur");
            hypre_check(HYPRE_BoomerAMGSetTruncFactor(precond, memoirs_env_double("MEMOIRS_AMG_TRUNC", 0.0)), "HYPRE_BoomerAMGSetTruncFactor rt0 reusable Schur");
            hypre_check(HYPRE_BoomerAMGSetRAP2(precond, memoirs_env_int("MEMOIRS_AMG_RAP2", 0)), "HYPRE_BoomerAMGSetRAP2 rt0 reusable Schur");

            const int agg = memoirs_env_int("MEMOIRS_AMG_AGG_LEVELS", 0);
            if (agg > 0) hypre_check(HYPRE_BoomerAMGSetAggNumLevels(precond, agg), "HYPRE_BoomerAMGSetAggNumLevels rt0 reusable Schur");
            const double strong = memoirs_env_double("MEMOIRS_AMG_STRONG", -1.0);
            if (strong >= 0.0) hypre_check(HYPRE_BoomerAMGSetStrongThreshold(precond, strong), "HYPRE_BoomerAMGSetStrongThreshold rt0 reusable Schur");

            hypre_check(HYPRE_PCGSetPrecond(
                pcg,
                (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSolve,
                (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSetup,
                precond),
                "HYPRE_PCGSetPrecond AMG rt0 reusable Schur");
        } else if (pre == "none") {
            // no preconditioner
        } else {
            throw std::runtime_error("Unsupported MEMOIRS_RT0_SCHUR_PRECOND in reusable Schur PCG: " + opt.precond);
        }

        hypre_check(HYPRE_ParCSRPCGSetup(pcg, parA, parb, parx), "HYPRE_ParCSRPCGSetup rt0 reusable Schur");
        auto tSetup1 = std::chrono::steady_clock::now();
        stats.hypreSetupSeconds += elapsed(tSetup0, tSetup1);
        stats.schurSetupApplications += 1;
        active = true;
    }

    SolveReport solve(const std::vector<Real>& b, std::vector<Real>& xOut) {
        if (!active) throw std::runtime_error("RT0 reusable Schur PCG solve called before setup.");
        if ((int)b.size() != (int)n) throw std::runtime_error("RT0 reusable Schur PCG RHS size mismatch.");

        SolveReport rep;
        rep.iterations = 0;
        rep.finalRelRes = -1.0;

        for (HYPRE_Int i = 0; i < n; ++i) bvals[i] = (HYPRE_Complex)b[i];

        auto tVec0 = std::chrono::steady_clock::now();
        hypre_check(HYPRE_IJVectorSetValues(ijb, n, rows.data(), bvals.data()), "HYPRE_IJVectorSetValues b rt0 reusable Schur solve");
        hypre_check(HYPRE_IJVectorAssemble(ijb), "HYPRE_IJVectorAssemble b rt0 reusable Schur solve");
        auto tVec1 = std::chrono::steady_clock::now();
        rep.hypreVectorInsertSeconds = elapsed(tVec0, tVec1);

        auto tSolve0 = std::chrono::steady_clock::now();
        hypre_check(HYPRE_ParVectorSetConstantValues(parx, HYPRE_Complex(0)), "HYPRE_ParVectorSetConstantValues x rt0 reusable Schur solve");
        hypre_check(HYPRE_ParCSRPCGSolve(pcg, parA, parb, parx), "HYPRE_ParCSRPCGSolve rt0 reusable Schur solve");
        auto tSolve1 = std::chrono::steady_clock::now();
        rep.hypreSolveOnlySeconds = elapsed(tSolve0, tSolve1);
        rep.hypreSolveOnlyAvgSeconds = rep.hypreSolveOnlySeconds;

        hypre_check(HYPRE_PCGGetNumIterations(pcg, &rep.iterations), "HYPRE_PCGGetNumIterations rt0 reusable Schur solve");
        HYPRE_Real finalRel = 0.0;
        hypre_check(HYPRE_PCGGetFinalRelativeResidualNorm(pcg, &finalRel), "HYPRE_PCGGetFinalRelativeResidualNorm rt0 reusable Schur solve");
        rep.finalRelRes = (double)finalRel;

        auto tGet0 = std::chrono::steady_clock::now();
        if (useDevice) {
            hypre_check(HYPRE_IJVectorMigrate(ijx, HYPRE_MEMORY_HOST), "HYPRE_IJVectorMigrate x HOST rt0 reusable Schur solve");
        }
        std::fill(xvals.begin(), xvals.end(), HYPRE_Complex(0));
        hypre_check(HYPRE_IJVectorGetValues(ijx, n, rows.data(), xvals.data()), "HYPRE_IJVectorGetValues x rt0 reusable Schur solve");
        xOut.assign((int)n, Real(0));
        for (HYPRE_Int i = 0; i < n; ++i) xOut[i] = Real(xvals[i]);
        if (useDevice) {
            hypre_check(HYPRE_IJVectorMigrate(ijx, HYPRE_MEMORY_DEVICE), "HYPRE_IJVectorMigrate x DEVICE rt0 reusable Schur solve");
        }
        auto tGet1 = std::chrono::steady_clock::now();
        rep.hypreGetSolutionSeconds = elapsed(tGet0, tGet1);
        return rep;
    }

    void destroy(Rt0BlockSchurStats& stats) {
        if (!active) return;
        auto t0 = std::chrono::steady_clock::now();
        if (precond) hypre_check(HYPRE_BoomerAMGDestroy(precond), "HYPRE_BoomerAMGDestroy rt0 reusable Schur");
        if (pcg) hypre_check(HYPRE_ParCSRPCGDestroy(pcg), "HYPRE_ParCSRPCGDestroy rt0 reusable Schur");
        if (ijx) hypre_check(HYPRE_IJVectorDestroy(ijx), "HYPRE_IJVectorDestroy x rt0 reusable Schur");
        if (ijb) hypre_check(HYPRE_IJVectorDestroy(ijb), "HYPRE_IJVectorDestroy b rt0 reusable Schur");
        if (ijA) hypre_check(HYPRE_IJMatrixDestroy(ijA), "HYPRE_IJMatrixDestroy A rt0 reusable Schur");
        hypre_check(HYPRE_Finalize(), "HYPRE_Finalize rt0 reusable Schur");
        auto t1 = std::chrono::steady_clock::now();
        stats.hypreDestroyFinalizeSeconds += elapsed(t0, t1);
        active = false;
        ijA = nullptr; parA = nullptr; ijb = nullptr; ijx = nullptr; parb = nullptr; parx = nullptr; pcg = nullptr; precond = nullptr;
    }
#endif
};

static std::vector<Real> rt0_apply_lumped_schur_preconditioner(
    const PolyMesh& m,
    const Rt0LumpedSchurData& data,
    const std::vector<Real>& r,
    const Options& opt,
    Rt0BlockSchurStats& stats,
    Rt0ReusableSchurPcg* reusableSchur = nullptr
) {
    const int nF = data.nF;
    const int nC = data.nC;
    if ((int)r.size() != nF + nC) {
        throw std::runtime_error("RT0 block-Schur preconditioner residual size mismatch.");
    }

    std::vector<Real> rhsS(nC, Real(0));

    // Solve S_lump zu = -ru - C D^{-1} rq.
    for (int c = 0; c < nC; ++c) {
        rhsS[c] = Real(-double(r[nF + c]));
    }
    for (int c = 0; c < nC; ++c) {
        for (int f : data.geoms[c].faces) {
            const int s = rt0_cell_face_sign(m, c, f);
            rhsS[c] -= Real(double(s) * data.dinv[f] * double(r[f]));
        }
    }

    Options sopt = opt;
    sopt.solver = "pcg";
    sopt.precond = rt0_env_string("MEMOIRS_RT0_SCHUR_PRECOND", "amg");
    sopt.tol = memoirs_env_double("MEMOIRS_RT0_SCHUR_TOL", 1e-4);
    sopt.maxit = memoirs_env_int("MEMOIRS_RT0_SCHUR_MAXIT", 100);

    std::vector<Real> zu;
    SolveReport srep;
    if (reusableSchur) {
        srep = reusableSchur->solve(rhsS, zu);
    } else {
        srep = solve_hypre_ij_pcg_raw(data.S, rhsS, sopt, zu);
    }
    if ((int)zu.size() != nC) {
        throw std::runtime_error("RT0 block-Schur scalar preconditioner solve returned wrong size.");
    }

    stats.schurApplications += 1;
    stats.schurIterationsTotal += srep.iterations;
    stats.schurIterationsMax = std::max(stats.schurIterationsMax, srep.iterations);
    stats.schurFinalRelLast = srep.finalRelRes;
    stats.hypreMatrixInsertSeconds += srep.hypreMatrixInsertSeconds;
    stats.hypreMatrixMigrateSeconds += srep.hypreMatrixMigrateSeconds;
    stats.hypreVectorInsertSeconds += srep.hypreVectorInsertSeconds;
    stats.hypreVectorMigrateSeconds += srep.hypreVectorMigrateSeconds;
    stats.hypreSetupSeconds += srep.hypreSetupSeconds;
    stats.hypreSolveOnlySeconds += srep.hypreSolveOnlySeconds;
    stats.hypreGetSolutionSeconds += srep.hypreGetSolutionSeconds;
    stats.hypreDestroyFinalizeSeconds += srep.hypreDestroyFinalizeSeconds;

    std::vector<Real> z(nF + nC, Real(0));

    // zq = D^{-1}(rq + C^T zu).
    for (int f = 0; f < nF; ++f) {
        double val = double(r[f]);
        const int co = m.owner[f];
        val += double(zu[co]);
        if (f < (int)m.neighbour.size()) {
            const int cn = m.neighbour[f];
            val -= double(zu[cn]);
        }
        z[f] = Real(data.dinv[f] * val);
    }
    for (int c = 0; c < nC; ++c) z[nF + c] = zu[c];

    return z;
}

static SolveReport solve_rt0_block_schur_fgmres(
    const PolyMesh& m,
    const Rt0DofMap& dm,
    const AssembledSystem& sys,
    const Options& opt,
    std::vector<Real>& xOut
) {
    const int n = dm.nTotalDofs;
    if ((int)sys.b.size() != n || sparse_nrows(sys.A) != n) {
        throw std::runtime_error("RT0 block-Schur FGMRES size mismatch.");
    }

    int mpiWasInitialized = 0;
    MPI_Initialized(&mpiWasInitialized);
    bool startedMPI = false;
    if (!mpiWasInitialized) {
        int argc = 0;
        char** argv = nullptr;
        MPI_Init(&argc, &argv);
        startedMPI = true;
    }

    auto t0 = std::chrono::steady_clock::now();

    const Rt0LumpedSchurData data = rt0_build_lumped_schur_data(m, dm, sys);
    Rt0BlockSchurStats stats;

    const bool reuseSchurSetup = memoirs_env_int("MEMOIRS_RT0_SCHUR_REUSE_SETUP", 1) != 0;
    const int schurReuseEvery = std::max(1, memoirs_env_int("MEMOIRS_RT0_SCHUR_REUSE_EVERY", 500));
    stats.schurReuseSetup = reuseSchurSetup ? 1 : 0;
    stats.schurReuseEvery = schurReuseEvery;

    Options reusableSchurOpt = opt;
    reusableSchurOpt.solver = "pcg";
    reusableSchurOpt.precond = rt0_env_string("MEMOIRS_RT0_SCHUR_PRECOND", "amg");
    reusableSchurOpt.tol = memoirs_env_double("MEMOIRS_RT0_SCHUR_TOL", 1e-4);
    reusableSchurOpt.maxit = memoirs_env_int("MEMOIRS_RT0_SCHUR_MAXIT", 100);

    Rt0ReusableSchurPcg reusableSchur;
    if (reuseSchurSetup) {
        reusableSchur.setup(data.S, reusableSchurOpt, stats);
    }

    const int restart = std::max(2, memoirs_env_int("MEMOIRS_RT0_OUTER_RESTART", 40));
    const int maxit = memoirs_env_int("MEMOIRS_RT0_OUTER_MAXIT", opt.maxit);
    const double tol = memoirs_env_double("MEMOIRS_RT0_OUTER_TOL", opt.tol);

    xOut.assign(n, Real(0));
    const double bNorm = std::max(1e-300, rt0_vec_norm_real(sys.b));

    SolveReport rep;
    rep.iterations = 0;
    rep.finalRelRes = 1.0;

    std::vector<Real> Ax = rt0_sparse_matvec_all(sys.A, xOut);
    std::vector<Real> r(n, Real(0));
    for (int i = 0; i < n; ++i) r[i] = sys.b[i] - Ax[i];

    double beta = rt0_vec_norm_real(r);
    rep.finalRelRes = beta / bNorm;

    if (rep.finalRelRes <= tol) {
        std::cout << "---------------- RT0 block Schur FGMRES probe -----------\n";
        std::cout << "outerSolver                   = host_restarted_fgmres\n";
        std::cout << "outerIterations               = 0\n";
        std::cout << "outerFinalRelativeResidual    = " << std::scientific << rep.finalRelRes << std::defaultfloat << "\n";
        std::cout << "---------------------------------------------------------\n";
        if (reuseSchurSetup) reusableSchur.destroy(stats);
        if (startedMPI) MPI_Finalize();
        return rep;
    }

    while (rep.iterations < maxit) {
        beta = rt0_vec_norm_real(r);
        if (beta / bNorm <= tol) break;

        std::vector<std::vector<Real>> V(restart + 1, std::vector<Real>(n, Real(0)));
        std::vector<std::vector<Real>> Z(restart, std::vector<Real>());
        std::vector<std::vector<double>> H(restart + 1, std::vector<double>(restart, 0.0));

        for (int i = 0; i < n; ++i) V[0][i] = Real(double(r[i]) / beta);

        int used = 0;
        std::vector<double> bestY;

        for (int j = 0; j < restart && rep.iterations < maxit; ++j) {
            if (reuseSchurSetup && stats.schurApplications > 0 && (stats.schurApplications % schurReuseEvery) == 0) {
                reusableSchur.destroy(stats);
                reusableSchur.setup(data.S, reusableSchurOpt, stats);
            }
            Z[j] = rt0_apply_lumped_schur_preconditioner(
                m, data, V[j], opt, stats,
                reuseSchurSetup ? &reusableSchur : nullptr
            );
            std::vector<Real> w = rt0_sparse_matvec_all(sys.A, Z[j]);

            for (int i = 0; i <= j; ++i) {
                H[i][j] = rt0_vec_dot_real(w, V[i]);
                for (int k = 0; k < n; ++k) w[k] -= Real(H[i][j] * double(V[i][k]));
            }

            // One classical re-orthogonalization pass helps when the saddle operator is ill-scaled.
            for (int i = 0; i <= j; ++i) {
                const double hij2 = rt0_vec_dot_real(w, V[i]);
                H[i][j] += hij2;
                for (int k = 0; k < n; ++k) w[k] -= Real(hij2 * double(V[i][k]));
            }

            H[j + 1][j] = rt0_vec_norm_real(w);
            if (H[j + 1][j] > 1e-300 && j + 1 < restart + 1) {
                for (int k = 0; k < n; ++k) V[j + 1][k] = Real(double(w[k]) / H[j + 1][j]);
            }

            used = j + 1;
            bestY = rt0_solve_small_normal_eq(H, beta, used);
            const double smallRes = rt0_fgmres_small_residual_norm(H, bestY, beta, used);
            rep.iterations += 1;
            rep.finalRelRes = smallRes / bNorm;

            if (rep.finalRelRes <= tol || H[j + 1][j] <= 1e-300) {
                break;
            }
        }

        if (used <= 0) break;
        if ((int)bestY.size() != used) bestY = rt0_solve_small_normal_eq(H, beta, used);

        for (int j = 0; j < used; ++j) {
            for (int i = 0; i < n; ++i) {
                xOut[i] += Real(bestY[j] * double(Z[j][i]));
            }
        }

        Ax = rt0_sparse_matvec_all(sys.A, xOut);
        for (int i = 0; i < n; ++i) r[i] = sys.b[i] - Ax[i];
        rep.finalRelRes = rt0_vec_norm_real(r) / bNorm;

        if (rep.finalRelRes <= tol) break;
    }

    if (reuseSchurSetup) reusableSchur.destroy(stats);

    auto t1 = std::chrono::steady_clock::now();
    rep.solveSeconds = std::chrono::duration<double>(t1 - t0).count();

    rep.hypreMatrixInsertSeconds = stats.hypreMatrixInsertSeconds;
    rep.hypreMatrixMigrateSeconds = stats.hypreMatrixMigrateSeconds;
    rep.hypreVectorInsertSeconds = stats.hypreVectorInsertSeconds;
    rep.hypreVectorMigrateSeconds = stats.hypreVectorMigrateSeconds;
    rep.hypreSetupSeconds = stats.hypreSetupSeconds;
    rep.hypreSolveOnlySeconds = stats.hypreSolveOnlySeconds;
    rep.hypreGetSolutionSeconds = stats.hypreGetSolutionSeconds;
    rep.hypreDestroyFinalizeSeconds = stats.hypreDestroyFinalizeSeconds;

    const double avgSchurIts = stats.schurApplications > 0
        ? double(stats.schurIterationsTotal) / double(stats.schurApplications)
        : 0.0;

    std::cout << "---------------- RT0 block Schur FGMRES probe -----------\n";
    std::cout << "outerSolver                   = host_restarted_fgmres\n";
    std::cout << "outerOperator                 = original_rt0_saddle_matrix\n";
    std::cout << "preconditioner                = block_triangular_Dinv_plus_lumped_schur\n";
    std::cout << "schurApprox                   = C diag(M)^{-1} C^T\n";
    std::cout << "schurPrecond                  = " << rt0_env_string("MEMOIRS_RT0_SCHUR_PRECOND", "amg") << "\n";
    std::cout << "schurReuseSetup              = " << stats.schurReuseSetup << "\n";
    std::cout << "schurReuseEvery              = " << stats.schurReuseEvery << "\n";
    std::cout << "schurSetups                  = " << stats.schurSetupApplications << "\n";
    std::cout << "amgCoarsen                   = " << memoirs_env_int("MEMOIRS_AMG_COARSEN", 8) << "\n";
    std::cout << "amgInterp                    = " << memoirs_env_int("MEMOIRS_AMG_INTERP", 6) << "\n";
    std::cout << "amgRelax                    = " << memoirs_env_int("MEMOIRS_AMG_RELAX", 18) << "\n";
    std::cout << "amgSweeps                   = " << memoirs_env_int("MEMOIRS_AMG_SWEEPS", 1) << "\n";
    std::cout << "outerRestart                  = " << restart << "\n";
    std::cout << "outerIterations               = " << rep.iterations << "\n";
    std::cout << "outerFinalRelativeResidual    = " << std::scientific << rep.finalRelRes << std::defaultfloat << "\n";
    std::cout << "schurApplications             = " << stats.schurApplications << "\n";
    std::cout << "schurIterationsTotal          = " << stats.schurIterationsTotal << "\n";
    std::cout << "schurIterationsAvg            = " << avgSchurIts << "\n";
    std::cout << "schurIterationsMax            = " << stats.schurIterationsMax << "\n";
    std::cout << "schurFinalRelLast             = " << std::scientific << stats.schurFinalRelLast << std::defaultfloat << "\n";
    std::cout << "fluxMassDiagMin               = " << std::scientific << data.minDiag << "\n";
    std::cout << "fluxMassDiagMax               = " << data.maxDiag << std::defaultfloat << "\n";
    std::cout << "---------------------------------------------------------\n";

    if (startedMPI) MPI_Finalize();
    return rep;
}

#endif

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

