
#pragma once
#include <functional>

// ============================================================================
// SECTION 17: Experimental RT1(tet) / DG1 mixed Poisson MMS
// ============================================================================
// Bring-up scope:
//   * pure tetrahedral polyMesh only
//   * RT1 flux space via local 15x15 moment inversion on each affine tet
//   * nodal DG1 scalar, 4 cell-local scalar dofs per cell
//   * symmetric mixed saddle assembly:
//         [ M  -B^T ] [sigma] = [r]
//         [-B    0  ] [u    ]   [-f]
//   * host-side experimental FGMRES with diagonal block-Schur preconditioner
//     for first correctness tests, plus exact sparse saddle matvec.
//
// Notes:
//   This is deliberately not optimized.  The RT1 basis is generated from a
//   physical polynomial spanning set and local DOF matrix inversion so the first
//   patch is auditable.  Once convergence/orientation is verified, the solver
//   and basis can be specialized.
// ============================================================================

struct Rt1DofMap {
    int nFaceMomentDofs = 0;
    int nInteriorFluxDofs = 0;
    int nFluxDofs = 0;
    int nScalarDofs = 0;
    int nTotalDofs = 0;
    std::string resolvedSpace = "rt1_tet_dg1";
};

struct Rt1CellGeom {
    std::array<int,4> verts{{-1,-1,-1,-1}};
    std::vector<int> faces;
    Vec3 centroid{};
    double volume = 0.0;
    Mat3 J{};
    Mat3 invJ{};
    double detJ = 0.0;
};

struct Rt1FieldErrorReport {
    double scalarL2 = -1.0;
    double fluxL2 = -1.0;
    double divCellL2 = -1.0;
    double divCellAbsMax = -1.0;
    double cellConservationL2 = -1.0;
    double cellConservationAbsMax = -1.0;
};

static inline Vec3 rt1_add(const Vec3& a, const Vec3& b) { return {a.x+b.x,a.y+b.y,a.z+b.z}; }
static inline Vec3 rt1_sub(const Vec3& a, const Vec3& b) { return {a.x-b.x,a.y-b.y,a.z-b.z}; }
static inline Vec3 rt1_mul(double s, const Vec3& a) { return {s*a.x,s*a.y,s*a.z}; }
static inline double rt1_norm(const Vec3& a) { return std::sqrt(dot3(a,a)); }

static Rt1DofMap build_rt1_tet_dg1_dof_map(const PolyMesh& m) {
    MeshCellCounts cc = count_cell_types(m);
    if (!(cc.nTet == (int)m.cells.size() && cc.nHex == 0 && cc.nOther == 0)) {
        throw std::runtime_error("RT1/DG1 phase6 requires a pure tetrahedral mesh.");
    }
    Rt1DofMap dm;
    dm.nFaceMomentDofs = 3 * (int)m.faces.size();
    dm.nInteriorFluxDofs = 3 * (int)m.cells.size();
    dm.nFluxDofs = dm.nFaceMomentDofs + dm.nInteriorFluxDofs;
    dm.nScalarDofs = 4 * (int)m.cells.size();
    dm.nTotalDofs = dm.nFluxDofs + dm.nScalarDofs;
    return dm;
}

static std::vector<std::vector<int>> rt1_build_cell_faces(const PolyMesh& m) {
    std::vector<std::vector<int>> cellFaces(m.cells.size());
    for (int f=0; f<(int)m.faces.size(); ++f) {
        cellFaces[m.owner[f]].push_back(f);
        if (f < (int)m.neighbour.size()) cellFaces[m.neighbour[f]].push_back(f);
    }
    return cellFaces;
}

static Vec3 rt1_cell_centroid(const PolyMesh& m, const Cell& c) {
    Vec3 x{};
    for (int v: c.verts) x = rt1_add(x, m.points[v]);
    return rt1_mul(1.0/(double)c.verts.size(), x);
}

static std::vector<Rt1CellGeom> build_rt1_cell_geoms(const PolyMesh& m) {
    auto cf = rt1_build_cell_faces(m);
    std::vector<Rt1CellGeom> out(m.cells.size());
    for (int c=0; c<(int)m.cells.size(); ++c) {
        if (m.cells[c].verts.size()!=4) throw std::runtime_error("RT1 found non-tet cell.");
        Rt1CellGeom K;
        for (int a=0;a<4;++a) K.verts[a] = m.cells[c].verts[a];
        K.faces = cf[c];
        if ((int)K.faces.size()!=4) throw std::runtime_error("RT1 tet cell face count mismatch.");
        K.centroid = rt1_cell_centroid(m, m.cells[c]);
        K.J = jacobian_tet_p1(m, K.verts);
        K.detJ = det3(K.J);
        if (std::abs(K.detJ) < 1e-300) throw std::runtime_error("RT1 singular tet Jacobian.");
        K.invJ = inv3(K.J);
        K.volume = std::abs(K.detJ) / 6.0;
        out[c]=K;
    }
    return out;
}

static Vec3 rt1_matvec(const Mat3& A, const Vec3& x) {
    return {
        A.a[0][0]*x.x + A.a[0][1]*x.y + A.a[0][2]*x.z,
        A.a[1][0]*x.x + A.a[1][1]*x.y + A.a[1][2]*x.z,
        A.a[2][0]*x.x + A.a[2][1]*x.y + A.a[2][2]*x.z
    };
}

static Vec3 rt1_ref_coords(const Rt1CellGeom& K, const Vec3& x, const PolyMesh& m) {
    const Vec3 x0 = m.points[K.verts[0]];
    return rt1_matvec(K.invJ, rt1_sub(x, x0));
}

static Vec3 rt1_face_centroid(const PolyMesh& m, int f) {
    Vec3 x{};
    for (int v: m.faces[f].verts) x = rt1_add(x, m.points[v]);
    return rt1_mul(1.0/(double)m.faces[f].verts.size(), x);
}

static Vec3 rt1_owner_outward_unit_normal(const PolyMesh& m, int f, const std::vector<Rt1CellGeom>& geoms, double& area) {
    const Face& F = m.faces[f];
    if (F.verts.size()!=3) throw std::runtime_error("RT1 tet path found non-tri face.");
    const Vec3& p0=m.points[F.verts[0]];
    const Vec3& p1=m.points[F.verts[1]];
    const Vec3& p2=m.points[F.verts[2]];
    Vec3 nA = rt1_mul(0.5, cross3(p1-p0, p2-p0));
    area = rt1_norm(nA);
    if (!(area>0.0)) throw std::runtime_error("RT1 degenerate face.");
    Vec3 n = rt1_mul(1.0/area, nA);
    const Vec3 fc = rt1_face_centroid(m, f);
    const Rt1CellGeom& Ko = geoms[m.owner[f]];
    if (dot3(n, rt1_sub(fc, Ko.centroid)) < 0.0) n = rt1_mul(-1.0, n);
    return n;
}

struct Rt1RawBasisAtPoint {
    std::array<Vec3,15> v{};
    std::array<double,15> div{};
    std::array<double,4> P{};
};

static Rt1RawBasisAtPoint rt1_eval_raw_basis(const PolyMesh& m, const Rt1CellGeom& K, double r, double s, double t) {
    Rt1RawBasisAtPoint B;
    const auto N = eval_tet_p1_basis_at(r,s,t);
    Vec3 x = map_tet_p1_to_physical(m, K.verts, N.N);
    Vec3 xrel = rt1_sub(x, m.points[K.verts[0]]);

    B.P[0]=1.0; B.P[1]=r; B.P[2]=s; B.P[3]=t;
    std::array<Vec3,4> gp{};
    gp[0] = {0,0,0};
    gp[1] = invJT_mul(K.invJ, {1,0,0});
    gp[2] = invJT_mul(K.invJ, {0,1,0});
    gp[3] = invJT_mul(K.invJ, {0,0,1});

    for (int k=0;k<4;++k) {
        B.v[k]   = {B.P[k],0,0}; B.div[k]   = gp[k].x;
        B.v[4+k] = {0,B.P[k],0}; B.div[4+k] = gp[k].y;
        B.v[8+k] = {0,0,B.P[k]}; B.div[8+k] = gp[k].z;
    }
    for (int q=0;q<3;++q) {
        const int k = q+1;
        B.v[12+q] = rt1_mul(B.P[k], xrel);
        B.div[12+q] = 3.0*B.P[k] + dot3(xrel, gp[k]);
    }
    return B;
}

static bool rt1_solve_dense_15(double A[15][15], double b[15], double x[15]) {
    double aug[15][16]{};
    for (int i=0;i<15;++i) { for (int j=0;j<15;++j) aug[i][j]=A[i][j]; aug[i][15]=b[i]; }
    for (int k=0;k<15;++k) {
        int piv=k; double best=std::abs(aug[k][k]);
        for (int i=k+1;i<15;++i) if (std::abs(aug[i][k])>best) { best=std::abs(aug[i][k]); piv=i; }
        if (!(best>1e-30) || !std::isfinite(best)) return false;
        if (piv!=k) for (int j=k;j<16;++j) std::swap(aug[piv][j], aug[k][j]);
        double d=aug[k][k];
        for (int j=k;j<16;++j) aug[k][j]/=d;
        for (int i=0;i<15;++i) if (i!=k) {
            double a=aug[i][k];
            if (a!=0.0) for (int j=k;j<16;++j) aug[i][j]-=a*aug[k][j];
        }
    }
    for (int i=0;i<15;++i) x[i]=aug[i][15];
    return true;
}

static std::array<double,15> rt1_compute_basis_coeff_column(
    const double D[15][15], int col
) {
    double A[15][15]; double rhs[15]{}; double sol[15]{};
    for (int i=0;i<15;++i) for (int j=0;j<15;++j) A[i][j]=D[i][j];
    rhs[col]=1.0;
    if (!rt1_solve_dense_15(A, rhs, sol)) throw std::runtime_error("RT1 local moment matrix singular.");
    std::array<double,15> out{};
    for (int i=0;i<15;++i) out[i]=sol[i];
    return out;
}

struct Rt1LocalBasis {
    std::array<std::array<double,15>,15> coeff{}; // basis a = sum raw j coeff[a][j]
    std::array<int,15> globalFlux{};
    double minMomentDiag = 0.0;
};

static Rt1LocalBasis rt1_build_local_basis(
    const PolyMesh& m, const Rt1DofMap& dm, int c, const Rt1CellGeom& K,
    const std::vector<Rt1CellGeom>& geoms
) {
    Rt1LocalBasis LB;
    for (int i=0;i<15;++i) LB.globalFlux[i] = -1;

    double D[15][15]{};
    int row=0;
    for (int lf=0; lf<4; ++lf) {
        const int f = K.faces[lf];
        double area=0.0;
        const Vec3 nG = rt1_owner_outward_unit_normal(m, f, geoms, area);
        const Face& F = m.faces[f];
        const Vec3& p0=m.points[F.verts[0]];
        const Vec3& p1=m.points[F.verts[1]];
        const Vec3& p2=m.points[F.verts[2]];
        const Vec3 e1=p1-p0, e2=p2-p0;
        const double jac = rt1_norm(cross3(e1,e2));
        for (int q=0;q<3;++q) LB.globalFlux[row+q] = 3*f + q;
        for (const auto& tq: triangle_face_quadrature_selected()) {
            const double l0=1.0-tq.r-tq.s, l1=tq.r, l2=tq.s;
            const double lm[3] = {l0,l1,l2};
            Vec3 x = rt1_add(p0, rt1_add(rt1_mul(tq.r,e1), rt1_mul(tq.s,e2)));
            Vec3 rr = rt1_ref_coords(K, x, m);
            auto RB = rt1_eval_raw_basis(m, K, rr.x, rr.y, rr.z);
            const double w = tq.weight * jac;
            for (int j=0;j<15;++j) {
                const double vn = dot3(RB.v[j], nG);
                for (int q=0;q<3;++q) D[row+q][j] += w * vn * lm[q];
            }
        }
        row += 3;
    }
    for (int k=0;k<3;++k) {
        LB.globalFlux[12+k] = dm.nFaceMomentDofs + 3*c + k;
    }
    for (const auto& q: tet_volume_quadrature_selected()) {
        auto RB = rt1_eval_raw_basis(m, K, q.xi, q.eta, q.zeta);
        const double w = q.weight * std::abs(K.detJ);
        for (int j=0;j<15;++j) {
            D[12][j] += w * RB.v[j].x;
            D[13][j] += w * RB.v[j].y;
            D[14][j] += w * RB.v[j].z;
        }
    }
    for (int a=0;a<15;++a) LB.coeff[a] = rt1_compute_basis_coeff_column(D, a);
    return LB;
}

static Vec3 rt1_eval_basis_vec(const Rt1LocalBasis& LB, int a, const Rt1RawBasisAtPoint& RB) {
    Vec3 v{};
    for (int j=0;j<15;++j) v = rt1_add(v, rt1_mul(LB.coeff[a][j], RB.v[j]));
    return v;
}
static double rt1_eval_basis_div(const Rt1LocalBasis& LB, int a, const Rt1RawBasisAtPoint& RB) {
    double d=0.0; for (int j=0;j<15;++j) d += LB.coeff[a][j] * RB.div[j]; return d;
}

static std::vector<std::vector<int>> build_rt1_saddle_row_columns(
    const PolyMesh& m, const Rt1DofMap& dm, const std::vector<Rt1CellGeom>& geoms
) {
    std::vector<std::vector<int>> cols(dm.nTotalDofs);
    for (int i=0;i<dm.nTotalDofs;++i) cols[i].push_back(i);
    for (int c=0;c<(int)m.cells.size();++c) {
        const Rt1CellGeom& K=geoms[c];
        std::vector<int> fl;
        for (int f: K.faces) for (int q=0;q<3;++q) fl.push_back(3*f+q);
        for (int k=0;k<3;++k) fl.push_back(dm.nFaceMomentDofs+3*c+k);
        std::vector<int> sc;
        for (int i=0;i<4;++i) sc.push_back(dm.nFluxDofs + 4*c + i);
        for (int I: fl) {
            for (int J: fl) cols[I].push_back(J);
            for (int J: sc) cols[I].push_back(J);
        }
        for (int I: sc) for (int J: fl) cols[I].push_back(J);
    }
    return cols;
}

static AssembledSystem assemble_rt1_tet_dg1_mixed_poisson_mms(
    const PolyMesh& m, const Rt1DofMap& dm, const std::string& mms
) {
    auto geoms = build_rt1_cell_geoms(m);
    AssembledSystem sys;
    sys.b.assign(dm.nTotalDofs, Real(0));
    if (memoirs_sparse_mode()=="fixed_csr") sparse_init_fixed_pattern(sys.A, build_rt1_saddle_row_columns(m, dm, geoms));
    else sys.A.rows.resize(dm.nTotalDofs);

    for (int c=0;c<(int)m.cells.size();++c) {
        const Rt1CellGeom& K=geoms[c];
        Rt1LocalBasis LB = rt1_build_local_basis(m, dm, c, K, geoms);
        double M[15][15]{}; double B[4][15]{}; double F[4]{}; double R[15]{};
        for (const auto& q: tet_volume_quadrature_selected()) {
            auto P = eval_tet_p1_basis_at(q.xi,q.eta,q.zeta);
            auto RB = rt1_eval_raw_basis(m, K, q.xi,q.eta,q.zeta);
            Vec3 x = map_tet_p1_to_physical(m, K.verts, P.N);
            double w = q.weight * std::abs(K.detJ);
            for (int a=0;a<15;++a) {
                Vec3 va=rt1_eval_basis_vec(LB,a,RB);
                double da=rt1_eval_basis_div(LB,a,RB);
                for (int b=0;b<15;++b) {
                    Vec3 vb=rt1_eval_basis_vec(LB,b,RB);
                    M[a][b] += w * dot3(va,vb);
                }
                for (int i=0;i<4;++i) B[i][a] += w * P.N[i] * da;
            }
            double fval = mms_rhs_value(x,mms);
            for (int i=0;i<4;++i) F[i] += w * fval * P.N[i];
        }
        // Dirichlet scalar boundary term: (sigma,tau)-(u,div tau)=-<g,tau.n_cell>.
        for (int lf=0; lf<4; ++lf) {
            const int f = K.faces[lf];
            if (f < (int)m.neighbour.size()) continue;
            double area=0.0;
            Vec3 n = rt1_owner_outward_unit_normal(m, f, geoms, area); // owner outward; boundary face owner is this cell
            const Face& Fce=m.faces[f];
            const Vec3& p0=m.points[Fce.verts[0]]; const Vec3& p1=m.points[Fce.verts[1]]; const Vec3& p2=m.points[Fce.verts[2]];
            Vec3 e1=p1-p0,e2=p2-p0; double jac=rt1_norm(cross3(e1,e2));
            for (const auto& tq: triangle_face_quadrature_selected()) {
                Vec3 x=rt1_add(p0, rt1_add(rt1_mul(tq.r,e1), rt1_mul(tq.s,e2)));
                Vec3 rr=rt1_ref_coords(K,x,m);
                auto RB=rt1_eval_raw_basis(m,K,rr.x,rr.y,rr.z);
                double w=tq.weight*jac;
                double g=mms_exact_value(x,mms);
                for (int a=0;a<15;++a) {
                    Vec3 va=rt1_eval_basis_vec(LB,a,RB);
                    R[a] += -w * g * dot3(va,n);
                }
            }
        }
        for (int a=0;a<15;++a) {
            int I=LB.globalFlux[a];
            sys.b[I] += Real(R[a]);
            for (int b=0;b<15;++b) sparse_add(sys.A, I, LB.globalFlux[b], Real(M[a][b]));
            for (int i=0;i<4;++i) {
                int S=dm.nFluxDofs + 4*c + i;
                sparse_add(sys.A, I, S, Real(-B[i][a]));
                sparse_add(sys.A, S, I, Real(-B[i][a]));
            }
        }
        for (int i=0;i<4;++i) sys.b[dm.nFluxDofs+4*c+i] += Real(-F[i]);
    }
    return sys;
}

static double rt1_dot_real(const std::vector<Real>& a, const std::vector<Real>& b) {
    double s=0.0; for (std::size_t i=0;i<a.size();++i) s += double(a[i])*double(b[i]); return s;
}
static double rt1_norm_real(const std::vector<Real>& a) { return std::sqrt(std::max(0.0, rt1_dot_real(a,a))); }
static std::vector<Real> rt1_matvec_sparse(const SparseRows& A, const std::vector<Real>& x) {
    int n=sparse_nrows(A); std::vector<Real> y(n,Real(0)); for (int i=0;i<n;++i) y[i]=sparse_matvec_row(A,i,x); return y;
}

static SolveReport solve_rt1_block_schur_diag_fgmres(
    const SparseRows& A, const Rt1DofMap& dm, const std::vector<Real>& b,
    const Options& opt, std::vector<Real>& xOut
) {
    SolveReport rep;
    const int n=dm.nTotalDofs, nF=dm.nFluxDofs, nS=dm.nScalarDofs;
    if (sparse_nrows(A)!=n || (int)b.size()!=n) throw std::runtime_error("RT1 FGMRES size mismatch.");
    std::vector<double> dinv(nF,1.0), sdiag(nS,0.0);
    for (int i=0;i<nF;++i) {
        Real d=Real(0); bool ok=false;
        if (A.fixedPattern) { const auto& sr=sparse_slot_row(A,i); auto it=sr.find(i); if (it!=sr.end()) { d=sparse_value_at(A,i,it->second); ok=true; }}
        else { auto it=A.rows[i].find(i); if (it!=A.rows[i].end()) { d=it->second; ok=true; }}
        if (!ok || !(double(d)>0.0)) throw std::runtime_error("RT1 FGMRES found bad flux mass diagonal.");
        dinv[i]=1.0/double(d);
    }
    // Schur diagonal from scalar rows: S ~= B D^-1 B^T.
    for (int si=0; si<nS; ++si) {
        int row=nF+si;
        double sd=0.0;
        if (A.fixedPattern) {
            const auto& c=sparse_cols_row(A,row);
            for (int k=0;k<(int)c.size();++k) if (c[k]<nF) { double Bij=-double(sparse_value_at(A,row,k)); sd += Bij*Bij*dinv[c[k]]; }
        } else {
            for (auto& kv:A.rows[row]) if (kv.first<nF) { double Bij=-double(kv.second); sd += Bij*Bij*dinv[kv.first]; }
        }
        sdiag[si] = (sd>1e-300 && std::isfinite(sd)) ? sd : 1.0;
    }
    auto apply_prec = [&](const std::vector<Real>& r) {
        std::vector<Real> z(n,Real(0));
        std::vector<double> tmp(nS,0.0);
        for (int si=0; si<nS; ++si) tmp[si] = -double(r[nF+si]);
        // tmp += -B D^-1 rq.  Scalar row stores -B.
        for (int si=0; si<nS; ++si) {
            int row=nF+si;
            if (A.fixedPattern) {
                const auto& c=sparse_cols_row(A,row);
                for (int k=0;k<(int)c.size();++k) if (c[k]<nF) {
                    double minusB = double(sparse_value_at(A,row,k));
                    double B = -minusB;
                    tmp[si] -= B * dinv[c[k]] * double(r[c[k]]);
                }
            } else {
                for (auto& kv:A.rows[row]) if (kv.first<nF) {
                    double B = -double(kv.second);
                    tmp[si] -= B * dinv[kv.first] * double(r[kv.first]);
                }
            }
        }
        for (int si=0; si<nS; ++si) z[nF+si] = Real(tmp[si]/sdiag[si]);
        for (int fi=0; fi<nF; ++fi) z[fi] = Real(dinv[fi]*double(r[fi]));
        // zq += D^-1 B^T zu. Flux row scalar columns store -B^T.
        for (int fi=0; fi<nF; ++fi) {
            double acc=double(r[fi]);
            if (A.fixedPattern) {
                const auto& c=sparse_cols_row(A,fi);
                for (int k=0;k<(int)c.size();++k) if (c[k]>=nF) {
                    double minusBT = double(sparse_value_at(A,fi,k));
                    double B = -minusBT;
                    acc += B * double(z[c[k]]);
                }
            } else {
                for (auto& kv:A.rows[fi]) if (kv.first>=nF) {
                    double B = -double(kv.second); acc += B*double(z[kv.first]);
                }
            }
            z[fi]=Real(dinv[fi]*acc);
        }
        return z;
    };
    const int restart = memoirs_env_int("MEMOIRS_RT1_OUTER_RESTART", 40);
    const int maxit = opt.maxit;
    const double tol=opt.tol;
    xOut.assign(n,Real(0));
    auto t0=std::chrono::steady_clock::now();
    std::vector<Real> Ax=rt1_matvec_sparse(A,xOut), r(n);
    for (int i=0;i<n;++i) r[i]=b[i]-Ax[i];
    const double bnorm=std::max(1e-300, rt1_norm_real(b));
    double rel=rt1_norm_real(r)/bnorm;
    int itTotal=0;
    std::vector<std::vector<Real>> V(restart+1), Z(restart);
    std::vector<std::vector<double>> H(restart+1, std::vector<double>(restart,0.0));
    std::vector<double> cs(restart,0.0), sn(restart,0.0), g(restart+1,0.0);
    while (itTotal<maxit && rel>tol) {
        double beta=rt1_norm_real(r);
        if (!(beta>0.0)) break;
        V[0].assign(n,Real(0)); for (int i=0;i<n;++i) V[0][i]=Real(double(r[i])/beta);
        for (auto& row:H) std::fill(row.begin(), row.end(), 0.0);
        std::fill(g.begin(), g.end(), 0.0); g[0]=beta;
        int m=0;
        for (; m<restart && itTotal<maxit; ++m,++itTotal) {
            Z[m]=apply_prec(V[m]);
            std::vector<Real> w=rt1_matvec_sparse(A,Z[m]);
            for (int j=0;j<=m;++j) {
                H[j][m]=rt1_dot_real(w,V[j]);
                for (int i=0;i<n;++i) w[i] -= Real(H[j][m]) * V[j][i];
            }
            H[m+1][m]=rt1_norm_real(w);
            V[m+1].assign(n,Real(0));
            if (H[m+1][m]>0.0) for (int i=0;i<n;++i) V[m+1][i]=Real(double(w[i])/H[m+1][m]);
            for (int j=0;j<m;++j) {
                double h0=H[j][m], h1=H[j+1][m];
                H[j][m]=cs[j]*h0 + sn[j]*h1;
                H[j+1][m]=-sn[j]*h0 + cs[j]*h1;
            }
            double rho=std::hypot(H[m][m], H[m+1][m]);
            if (!(rho>0.0)) rho=1.0;
            cs[m]=H[m][m]/rho; sn[m]=H[m+1][m]/rho;
            H[m][m]=rho; H[m+1][m]=0.0;
            double g0=g[m], g1=g[m+1];
            g[m]=cs[m]*g0 + sn[m]*g1;
            g[m+1]=-sn[m]*g0 + cs[m]*g1;
            rel=std::abs(g[m+1])/bnorm;
            if (rel<=tol) { ++m; break; }
        }
        std::vector<double> y(m,0.0);
        for (int i=m-1;i>=0;--i) {
            double s=g[i]; for (int j=i+1;j<m;++j) s-=H[i][j]*y[j];
            y[i]=s/H[i][i];
        }
        for (int j=0;j<m;++j) for (int i=0;i<n;++i) xOut[i] += Real(y[j])*Z[j][i];
        Ax=rt1_matvec_sparse(A,xOut);
        for (int i=0;i<n;++i) r[i]=b[i]-Ax[i];
        rel=rt1_norm_real(r)/bnorm;
    }
    auto t1=std::chrono::steady_clock::now();
    rep.iterations=itTotal;
    rep.finalRelRes=rel;
    rep.hypreSolveOnlySeconds=std::chrono::duration<double>(t1-t0).count();
    rep.solveSeconds=rep.hypreSolveOnlySeconds;
    return rep;
}



// --------------------------------------------------------------------------
// Stronger experimental RT1 preconditioner: additive cell-saddle block inverse.
//
// The earlier diagonal Schur preconditioner is intentionally very crude and can
// fail to move the residual for RT1/DG1.  This preconditioner builds one dense
// 19x19 block per tet using the 15 local RT1 flux dofs and 4 local DG1 scalar
// dofs, inverts that local saddle block, and applies the inverses additively
// with overlap weighting on shared face-moment dofs.  It is a correctness and
// orientation bring-up preconditioner, not a final production solver.
// --------------------------------------------------------------------------

static double rt1_sparse_get_value_or_zero(const SparseRows& A, int row, int col) {
    if (A.fixedPattern) {
        const auto& sr = sparse_slot_row(A, row);
        auto it = sr.find(col);
        if (it == sr.end()) return 0.0;
        return double(sparse_value_at(A, row, it->second));
    }
    auto it = A.rows[row].find(col);
    if (it == A.rows[row].end()) return 0.0;
    return double(it->second);
}

static bool rt1_invert_dense_n(std::vector<double> A, int n, std::vector<double>& inv) {
    inv.assign(n*n, 0.0);
    std::vector<double> aug(n*(2*n), 0.0);
    for (int i=0;i<n;++i) {
        for (int j=0;j<n;++j) aug[i*(2*n)+j] = A[i*n+j];
        aug[i*(2*n)+(n+i)] = 1.0;
    }
    for (int k=0;k<n;++k) {
        int piv=k; double best=std::abs(aug[k*(2*n)+k]);
        for (int i=k+1;i<n;++i) {
            double v=std::abs(aug[i*(2*n)+k]);
            if (v>best) { best=v; piv=i; }
        }
        if (!(best>1e-28) || !std::isfinite(best)) return false;
        if (piv!=k) {
            for (int j=0;j<2*n;++j) std::swap(aug[piv*(2*n)+j], aug[k*(2*n)+j]);
        }
        double d=aug[k*(2*n)+k];
        for (int j=0;j<2*n;++j) aug[k*(2*n)+j] /= d;
        for (int i=0;i<n;++i) if (i!=k) {
            double a=aug[i*(2*n)+k];
            if (a!=0.0) for (int j=0;j<2*n;++j) aug[i*(2*n)+j] -= a*aug[k*(2*n)+j];
        }
    }
    for (int i=0;i<n;++i) for (int j=0;j<n;++j) inv[i*n+j]=aug[i*(2*n)+(n+j)];
    return true;
}

struct Rt1CellSaddleBlockInverse {
    std::array<int,19> ids{};
    std::array<double,19*19> inv{};
};

static std::vector<Rt1CellSaddleBlockInverse> build_rt1_cell_saddle_block_inverse_cache(
    const PolyMesh& m,
    const SparseRows& A,
    const Rt1DofMap& dm,
    std::vector<double>& overlapWeight,
    double& minPivotProxy,
    double& maxAbsInvEntry
) {
    auto geoms = build_rt1_cell_geoms(m);
    std::vector<Rt1CellSaddleBlockInverse> blocks(m.cells.size());
    overlapWeight.assign(dm.nTotalDofs, 0.0);
    minPivotProxy = 1e300;
    maxAbsInvEntry = 0.0;
    for (int c=0;c<(int)m.cells.size();++c) {
        Rt1LocalBasis LB = rt1_build_local_basis(m, dm, c, geoms[c], geoms);
        Rt1CellSaddleBlockInverse blk;
        for (int a=0;a<15;++a) blk.ids[a]=LB.globalFlux[a];
        for (int i=0;i<4;++i) blk.ids[15+i]=dm.nFluxDofs + 4*c + i;
        std::vector<double> local(19*19, 0.0), inv;
        for (int i=0;i<19;++i) for (int j=0;j<19;++j) {
            local[i*19+j] = rt1_sparse_get_value_or_zero(A, blk.ids[i], blk.ids[j]);
        }
        double diagMin=1e300;
        for (int i=0;i<15;++i) diagMin = std::min(diagMin, std::abs(local[i*19+i]));
        minPivotProxy = std::min(minPivotProxy, diagMin);
        if (!rt1_invert_dense_n(local, 19, inv)) {
            throw std::runtime_error("RT1 cell-saddle block preconditioner found singular 19x19 local block.");
        }
        for (int k=0;k<19*19;++k) {
            blk.inv[k]=inv[k];
            maxAbsInvEntry = std::max(maxAbsInvEntry, std::abs(inv[k]));
        }
        for (int i=0;i<19;++i) overlapWeight[blk.ids[i]] += 1.0;
        blocks[c]=blk;
    }
    for (double& w: overlapWeight) if (!(w>0.0)) w=1.0;
    return blocks;
}

static SolveReport solve_rt1_fgmres_with_preconditioner(
    const SparseRows& A,
    int n,
    const std::vector<Real>& b,
    const Options& opt,
    std::vector<Real>& xOut,
    const std::function<std::vector<Real>(const std::vector<Real>&)>& apply_prec
) {
    SolveReport rep;
    if (sparse_nrows(A)!=n || (int)b.size()!=n) throw std::runtime_error("RT1 generic FGMRES size mismatch.");
    const int restart = memoirs_env_int("MEMOIRS_RT1_OUTER_RESTART", 40);
    const int maxit = opt.maxit;
    const double tol = opt.tol;
    xOut.assign(n, Real(0));
    auto t0=std::chrono::steady_clock::now();
    std::vector<Real> Ax=rt1_matvec_sparse(A,xOut), r(n);
    for (int i=0;i<n;++i) r[i]=b[i]-Ax[i];
    const double bnorm=std::max(1e-300, rt1_norm_real(b));
    double rel=rt1_norm_real(r)/bnorm;
    int itTotal=0;
    std::vector<std::vector<Real>> V(restart+1), Z(restart);
    std::vector<std::vector<double>> H(restart+1, std::vector<double>(restart,0.0));
    std::vector<double> cs(restart,0.0), sn(restart,0.0), g(restart+1,0.0);
    while (itTotal<maxit && rel>tol) {
        double beta=rt1_norm_real(r);
        if (!(beta>0.0)) break;
        V[0].assign(n,Real(0)); for (int i=0;i<n;++i) V[0][i]=Real(double(r[i])/beta);
        for (auto& row:H) std::fill(row.begin(), row.end(), 0.0);
        std::fill(g.begin(), g.end(), 0.0); g[0]=beta;
        int m=0;
        for (; m<restart && itTotal<maxit; ++m,++itTotal) {
            Z[m]=apply_prec(V[m]);
            std::vector<Real> w=rt1_matvec_sparse(A,Z[m]);
            for (int j=0;j<=m;++j) {
                H[j][m]=rt1_dot_real(w,V[j]);
                for (int i=0;i<n;++i) w[i] -= Real(H[j][m]) * V[j][i];
            }
            H[m+1][m]=rt1_norm_real(w);
            V[m+1].assign(n,Real(0));
            if (H[m+1][m]>0.0) for (int i=0;i<n;++i) V[m+1][i]=Real(double(w[i])/H[m+1][m]);
            for (int j=0;j<m;++j) {
                double h0=H[j][m], h1=H[j+1][m];
                H[j][m]=cs[j]*h0 + sn[j]*h1;
                H[j+1][m]=-sn[j]*h0 + cs[j]*h1;
            }
            double rho=std::hypot(H[m][m], H[m+1][m]);
            if (!(rho>0.0)) rho=1.0;
            cs[m]=H[m][m]/rho; sn[m]=H[m+1][m]/rho;
            H[m][m]=rho; H[m+1][m]=0.0;
            double g0=g[m], g1=g[m+1];
            g[m]=cs[m]*g0 + sn[m]*g1;
            g[m+1]=-sn[m]*g0 + cs[m]*g1;
            rel=std::abs(g[m+1])/bnorm;
            if (rel<=tol) { ++m; break; }
        }
        std::vector<double> y(m,0.0);
        for (int i=m-1;i>=0;--i) {
            double s=g[i]; for (int j=i+1;j<m;++j) s-=H[i][j]*y[j];
            y[i]=s/H[i][i];
        }
        for (int j=0;j<m;++j) for (int i=0;i<n;++i) xOut[i] += Real(y[j])*Z[j][i];
        Ax=rt1_matvec_sparse(A,xOut);
        for (int i=0;i<n;++i) r[i]=b[i]-Ax[i];
        rel=rt1_norm_real(r)/bnorm;
        if (memoirs_env_int("MEMOIRS_RT1_FGMRES_PRINT_RESTARTS",0)!=0) {
            std::cout << "rt1FgmresRestart          = it " << itTotal << " rel " << std::scientific << rel << std::defaultfloat << "\n";
        }
    }
    auto t1=std::chrono::steady_clock::now();
    rep.iterations=itTotal;
    rep.finalRelRes=rel;
    rep.hypreSolveOnlySeconds=std::chrono::duration<double>(t1-t0).count();
    rep.solveSeconds=rep.hypreSolveOnlySeconds;
    return rep;
}

static SolveReport solve_rt1_cell_saddle_fgmres(
    const PolyMesh& m,
    const SparseRows& A,
    const Rt1DofMap& dm,
    const std::vector<Real>& b,
    const Options& opt,
    std::vector<Real>& xOut
) {
    std::vector<double> weight;
    double minPivotProxy=0.0, maxAbsInvEntry=0.0;
    auto blocks = build_rt1_cell_saddle_block_inverse_cache(m, A, dm, weight, minPivotProxy, maxAbsInvEntry);
    if (opt.diagLevel>=1) {
        std::cout << "rt1CellSaddleBlocks        = " << blocks.size() << "\n";
        std::cout << "rt1CellSaddleMinDiagProxy  = " << std::scientific << minPivotProxy << "\n";
        std::cout << "rt1CellSaddleMaxAbsInv     = " << maxAbsInvEntry << std::defaultfloat << "\n";
    }
    auto apply_prec = [&](const std::vector<Real>& r) {
        std::vector<Real> z(dm.nTotalDofs, Real(0));
        for (const auto& blk: blocks) {
            double rr[19]{};
            for (int i=0;i<19;++i) rr[i]=double(r[blk.ids[i]]);
            for (int i=0;i<19;++i) {
                double zi=0.0;
                for (int j=0;j<19;++j) zi += blk.inv[i*19+j] * rr[j];
                z[blk.ids[i]] += Real(zi);
            }
        }
        for (int i=0;i<dm.nTotalDofs;++i) z[i] = Real(double(z[i]) / weight[i]);
        return z;
    };
    return solve_rt1_fgmres_with_preconditioner(A, dm.nTotalDofs, b, opt, xOut, apply_prec);
}



// --------------------------------------------------------------------------
// RT1/DG1 global Schur AMG preconditioner, analogous to the validated RT0 path.
//
// Outer Krylov operator: exact assembled RT1/DG1 saddle matrix
//        A = [ M  -B^T ]
//            [ -B   0  ]
//
// Preconditioner: block triangular cheap-flux-inverse + global scalar Schur
//        D = diag(M)
//        S = B D^{-1} B^T        on the DG1 scalar space
//        S z_u = -r_u - B D^{-1} r_q
//        z_q   = D^{-1}(r_q + B^T z_u)
//
// This is the direct RT1 analog of the RT0 block-Schur path that reduced the
// RT0 outer iterations to ~O(10).  The first version intentionally uses the
// diagonal flux-mass inverse D^{-1}; later this can be replaced by element-block
// M^{-1} or hybridized static condensation.
// --------------------------------------------------------------------------

struct Rt1GlobalSchurData {
    SparseRows S;
    std::vector<double> dinv;
    std::vector<std::vector<std::pair<int,double>>> scalarB; // per scalar row: (flux dof, B_si_f)
    std::vector<std::vector<std::pair<int,double>>> fluxBT;  // per flux row:   (scalar dof, B_si_f)
    int nF = 0;
    int nS = 0;
    double fluxMassDiagMin = 0.0;
    double fluxMassDiagMax = 0.0;
    long long schurNnz = 0;
};


static std::string memoirs_rt1_env_string(const char* name, const char* defval) {
    const char* v = std::getenv(name);
    if (!v || !*v) return std::string(defval);
    return std::string(v);
}

struct Rt1GlobalSchurStats {
    int schurApplications = 0;
    int schurIterationsTotal = 0;
    int schurIterationsMax = 0;
    double schurFinalRelLast = 0.0;
    double hypreMatrixInsertSeconds = 0.0;
    double hypreMatrixMigrateSeconds = 0.0;
    double hypreVectorInsertSeconds = 0.0;
    double hypreVectorMigrateSeconds = 0.0;
    double hypreSetupSeconds = 0.0;
    double hypreSolveOnlySeconds = 0.0;
    double hypreGetSolutionSeconds = 0.0;
    double hypreDestroyFinalizeSeconds = 0.0;
};

static Rt1GlobalSchurData rt1_build_global_diag_schur_data(
    const SparseRows& A,
    const Rt1DofMap& dm
) {
    Rt1GlobalSchurData d;
    d.nF = dm.nFluxDofs;
    d.nS = dm.nScalarDofs;
    const int n = dm.nTotalDofs;
    if (sparse_nrows(A) != n) throw std::runtime_error("RT1 global Schur matrix size mismatch.");

    d.dinv.assign(d.nF, 0.0);
    d.scalarB.assign(d.nS, {});
    d.fluxBT.assign(d.nF, {});
    d.fluxMassDiagMin = std::numeric_limits<double>::infinity();
    d.fluxMassDiagMax = 0.0;

    for (int f=0; f<d.nF; ++f) {
        const double diag = rt1_sparse_get_value_or_zero(A, f, f);
        if (!(diag > 0.0) || !std::isfinite(diag)) {
            throw std::runtime_error("RT1 global Schur found non-positive/bad flux mass diagonal.");
        }
        d.dinv[f] = 1.0 / diag;
        d.fluxMassDiagMin = std::min(d.fluxMassDiagMin, diag);
        d.fluxMassDiagMax = std::max(d.fluxMassDiagMax, diag);
    }

    // Extract B from scalar rows.  Scalar rows store -B in the full saddle matrix.
    for (int si=0; si<d.nS; ++si) {
        const int row = d.nF + si;
        if (A.fixedPattern) {
            const auto& cols = sparse_cols_row(A, row);
            for (int k=0;k<(int)cols.size();++k) {
                const int col = cols[k];
                if (col < d.nF) {
                    const double B = -double(sparse_value_at(A, row, k));
                    if (B != 0.0) {
                        d.scalarB[si].push_back({col, B});
                        d.fluxBT[col].push_back({si, B});
                    }
                }
            }
        } else {
            for (const auto& kv : A.rows[row]) {
                const int col = kv.first;
                if (col < d.nF) {
                    const double B = -double(kv.second);
                    if (B != 0.0) {
                        d.scalarB[si].push_back({col, B});
                        d.fluxBT[col].push_back({si, B});
                    }
                }
            }
        }
    }

    std::vector<std::vector<int>> rowCols(d.nS);
    for (int si=0; si<d.nS; ++si) rowCols[si].push_back(si);
    for (int f=0; f<d.nF; ++f) {
        const auto& ss = d.fluxBT[f];
        for (const auto& a : ss) for (const auto& b : ss) rowCols[a.first].push_back(b.first);
    }
    if (memoirs_sparse_mode() == "fixed_csr") sparse_init_fixed_pattern(d.S, rowCols);
    else d.S.rows.resize(d.nS);

    for (int f=0; f<d.nF; ++f) {
        const double w = d.dinv[f];
        const auto& ss = d.fluxBT[f];
        for (const auto& a : ss) {
            const int si = a.first;
            const double Bi = a.second;
            for (const auto& b : ss) {
                const int sj = b.first;
                const double Bj = b.second;
                sparse_add(d.S, si, sj, Real(Bi * w * Bj));
            }
        }
    }
    d.schurNnz = 0;
    for (int si=0; si<d.nS; ++si) {
        if (d.S.fixedPattern) {
            d.schurNnz += (long long)(sparse_row_end(d.S, si) - sparse_row_start(d.S, si));
        } else {
            d.schurNnz += (long long)d.S.rows[si].size();
        }
    }
    return d;
}


// Reusable HYPRE Schur PCG/AMG object for RT1 global-Schur preconditioning.
// Builds S and its AMG hierarchy once, then only updates the Schur RHS on each
// FGMRES preconditioner application.  This is the RT1 analog of the RT0 Schur
// setup-reuse path.
struct Rt1ReusableSchurSolver {
    bool ready = false;
    bool useDevice = false;
    int nHost = 0;
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
    std::vector<HYPRE_Complex> bvals;
    std::vector<HYPRE_Complex> xvals;
    Options sopt;

    static double elapsed_seconds(
        const std::chrono::steady_clock::time_point& a,
        const std::chrono::steady_clock::time_point& b
    ) {
        return std::chrono::duration<double>(b-a).count();
    }

    void setup(const SparseRows& S, const Options& opt, Rt1GlobalSchurStats& stats) {
        if (ready) return;
        nHost = sparse_nrows(S);
        if (nHost <= 0) throw std::runtime_error("RT1 reusable Schur solver got empty Schur matrix.");

        sopt = opt;
        sopt.solver = "pcg";
        sopt.precond = memoirs_rt1_env_string("MEMOIRS_RT1_SCHUR_PRECOND", "amg");
        sopt.tol = memoirs_env_double("MEMOIRS_RT1_SCHUR_TOL", 1e-4);
        sopt.maxit = memoirs_env_int("MEMOIRS_RT1_SCHUR_MAXIT", 100);

        const HYPRE_MemoryLocation requestedMem = parse_hypre_memory_location(opt.hypreMemory);
        useDevice = (requestedMem == HYPRE_MEMORY_DEVICE);

        hypre_check(HYPRE_Init(), "HYPRE_Init RT1 reusable Schur");
        if (useDevice) {
            hypre_check(HYPRE_SetMemoryLocation(HYPRE_MEMORY_DEVICE), "HYPRE_SetMemoryLocation DEVICE RT1 reusable Schur");
            hypre_check(HYPRE_SetExecutionPolicy(HYPRE_EXEC_DEVICE), "HYPRE_SetExecutionPolicy DEVICE RT1 reusable Schur");
        } else {
            hypre_check(HYPRE_SetMemoryLocation(HYPRE_MEMORY_HOST), "HYPRE_SetMemoryLocation HOST RT1 reusable Schur");
            hypre_check(HYPRE_SetExecutionPolicy(HYPRE_EXEC_HOST), "HYPRE_SetExecutionPolicy HOST RT1 reusable Schur");
        }

        const HYPRE_Int n = (HYPRE_Int)nHost;
        const HYPRE_BigInt ilower = 0;
        const HYPRE_BigInt iupper = n - 1;

        rows.resize(n);
        ncols.resize(n);
        std::vector<HYPRE_Int> cols;
        std::vector<HYPRE_Complex> vals;
        cols.reserve(sparse_nnz(S));
        vals.reserve(sparse_nnz(S));

        auto tMat0 = std::chrono::steady_clock::now();

        hypre_check(HYPRE_IJMatrixCreate(comm, ilower, iupper, ilower, iupper, &ijA), "HYPRE_IJMatrixCreate RT1 reusable Schur");
        hypre_check(HYPRE_IJMatrixSetObjectType(ijA, HYPRE_PARCSR), "HYPRE_IJMatrixSetObjectType RT1 reusable Schur");

        for (HYPRE_Int i=0; i<n; ++i) {
            rows[i] = i;
            if (S.fixedPattern) {
                const auto& c = sparse_cols_row(S, i);
                ncols[i] = (HYPRE_Int)c.size();
                for (int k=0; k<(int)c.size(); ++k) {
                    cols.push_back((HYPRE_Int)c[k]);
                    vals.push_back((HYPRE_Complex)sparse_value_at(S, i, k));
                }
            } else {
                ncols[i] = (HYPRE_Int)S.rows[i].size();
                for (const auto& kv : S.rows[i]) {
                    cols.push_back((HYPRE_Int)kv.first);
                    vals.push_back((HYPRE_Complex)kv.second);
                }
            }
        }

        hypre_check(HYPRE_IJMatrixSetRowSizes(ijA, ncols.data()), "HYPRE_IJMatrixSetRowSizes RT1 reusable Schur");
        hypre_initialize_ij_matrix_for_host_insertion(ijA);
        hypre_check(HYPRE_IJMatrixSetValues(ijA, n, ncols.data(), rows.data(), cols.data(), vals.data()), "HYPRE_IJMatrixSetValues RT1 reusable Schur");
        hypre_check(HYPRE_IJMatrixAssemble(ijA), "HYPRE_IJMatrixAssemble RT1 reusable Schur");

        auto tMat1 = std::chrono::steady_clock::now();
        stats.hypreMatrixInsertSeconds += elapsed_seconds(tMat0, tMat1);

        auto tMatMig0 = std::chrono::steady_clock::now();
        if (useDevice) hypre_check(HYPRE_IJMatrixMigrate(ijA, HYPRE_MEMORY_DEVICE), "HYPRE_IJMatrixMigrate RT1 reusable Schur");
        hypre_check(HYPRE_IJMatrixGetObject(ijA, (void**)&parA), "HYPRE_IJMatrixGetObject RT1 reusable Schur");
        auto tMatMig1 = std::chrono::steady_clock::now();
        stats.hypreMatrixMigrateSeconds += elapsed_seconds(tMatMig0, tMatMig1);

        bvals.assign(n, HYPRE_Complex(0));
        xvals.assign(n, HYPRE_Complex(0));

        auto tVec0 = std::chrono::steady_clock::now();
        hypre_check(HYPRE_IJVectorCreate(comm, ilower, iupper, &ijb), "HYPRE_IJVectorCreate b RT1 reusable Schur");
        hypre_check(HYPRE_IJVectorSetObjectType(ijb, HYPRE_PARCSR), "HYPRE_IJVectorSetObjectType b RT1 reusable Schur");
        hypre_initialize_ij_vector_for_host_insertion(ijb);
        hypre_check(HYPRE_IJVectorCreate(comm, ilower, iupper, &ijx), "HYPRE_IJVectorCreate x RT1 reusable Schur");
        hypre_check(HYPRE_IJVectorSetObjectType(ijx, HYPRE_PARCSR), "HYPRE_IJVectorSetObjectType x RT1 reusable Schur");
        hypre_initialize_ij_vector_for_host_insertion(ijx);
        hypre_check(HYPRE_IJVectorSetValues(ijb, n, rows.data(), bvals.data()), "HYPRE_IJVectorSetValues b init RT1 reusable Schur");
        hypre_check(HYPRE_IJVectorSetValues(ijx, n, rows.data(), xvals.data()), "HYPRE_IJVectorSetValues x init RT1 reusable Schur");
        hypre_check(HYPRE_IJVectorAssemble(ijb), "HYPRE_IJVectorAssemble b init RT1 reusable Schur");
        hypre_check(HYPRE_IJVectorAssemble(ijx), "HYPRE_IJVectorAssemble x init RT1 reusable Schur");
        auto tVec1 = std::chrono::steady_clock::now();
        stats.hypreVectorInsertSeconds += elapsed_seconds(tVec0, tVec1);

        auto tVecMig0 = std::chrono::steady_clock::now();
        if (useDevice) {
            hypre_check(HYPRE_IJVectorMigrate(ijb, HYPRE_MEMORY_DEVICE), "HYPRE_IJVectorMigrate b init RT1 reusable Schur");
            hypre_check(HYPRE_IJVectorMigrate(ijx, HYPRE_MEMORY_DEVICE), "HYPRE_IJVectorMigrate x init RT1 reusable Schur");
        }
        hypre_check(HYPRE_IJVectorGetObject(ijb, (void**)&parb), "HYPRE_IJVectorGetObject b init RT1 reusable Schur");
        hypre_check(HYPRE_IJVectorGetObject(ijx, (void**)&parx), "HYPRE_IJVectorGetObject x init RT1 reusable Schur");
        auto tVecMig1 = std::chrono::steady_clock::now();
        stats.hypreVectorMigrateSeconds += elapsed_seconds(tVecMig0, tVecMig1);

        auto tSetup0 = std::chrono::steady_clock::now();
        hypre_check(HYPRE_ParCSRPCGCreate(comm, &pcg), "HYPRE_ParCSRPCGCreate RT1 reusable Schur");
        hypre_check(HYPRE_PCGSetMaxIter(pcg, sopt.maxit), "HYPRE_PCGSetMaxIter RT1 reusable Schur");
        hypre_check(HYPRE_PCGSetTol(pcg, sopt.tol), "HYPRE_PCGSetTol RT1 reusable Schur");
        hypre_check(HYPRE_PCGSetTwoNorm(pcg, 1), "HYPRE_PCGSetTwoNorm RT1 reusable Schur");
        hypre_check(HYPRE_PCGSetPrintLevel(pcg, sopt.hyprePrint), "HYPRE_PCGSetPrintLevel RT1 reusable Schur");
        hypre_check(HYPRE_PCGSetLogging(pcg, 1), "HYPRE_PCGSetLogging RT1 reusable Schur");

        const std::string pre = lower_copy(sopt.precond);
        if (pre == "diagscale") {
            hypre_check(
                HYPRE_PCGSetPrecond(
                    pcg,
                    (HYPRE_PtrToSolverFcn)HYPRE_ParCSRDiagScale,
                    (HYPRE_PtrToSolverFcn)HYPRE_ParCSRDiagScaleSetup,
                    nullptr
                ),
                "HYPRE_PCGSetPrecond diag RT1 reusable Schur"
            );
        } else if (pre == "amg" || pre == "boomeramg") {
            hypre_check(HYPRE_BoomerAMGCreate(&precond), "HYPRE_BoomerAMGCreate RT1 reusable Schur");
            hypre_check(HYPRE_BoomerAMGSetPrintLevel(precond, sopt.hyprePrint), "HYPRE_BoomerAMGSetPrintLevel RT1 reusable Schur");
            hypre_check(HYPRE_BoomerAMGSetCoarsenType(precond, memoirs_env_int("MEMOIRS_AMG_COARSEN", 8)), "HYPRE_BoomerAMGSetCoarsenType RT1 reusable Schur");
            hypre_check(HYPRE_BoomerAMGSetInterpType(precond, memoirs_env_int("MEMOIRS_AMG_INTERP", 6)), "HYPRE_BoomerAMGSetInterpType RT1 reusable Schur");
            hypre_check(HYPRE_BoomerAMGSetRelaxType(precond, memoirs_env_int("MEMOIRS_AMG_RELAX", 18)), "HYPRE_BoomerAMGSetRelaxType RT1 reusable Schur");
            hypre_check(HYPRE_BoomerAMGSetNumSweeps(precond, memoirs_env_int("MEMOIRS_AMG_SWEEPS", 1)), "HYPRE_BoomerAMGSetNumSweeps RT1 reusable Schur");
            hypre_check(HYPRE_BoomerAMGSetTol(precond, 0.0), "HYPRE_BoomerAMGSetTol RT1 reusable Schur");
            hypre_check(HYPRE_BoomerAMGSetMaxIter(precond, 1), "HYPRE_BoomerAMGSetMaxIter RT1 reusable Schur");
            hypre_check(HYPRE_BoomerAMGSetRelaxOrder(precond, 0), "HYPRE_BoomerAMGSetRelaxOrder RT1 reusable Schur");
            hypre_check(HYPRE_BoomerAMGSetPMaxElmts(precond, memoirs_env_int("MEMOIRS_AMG_PMAX", 4)), "HYPRE_BoomerAMGSetPMaxElmts RT1 reusable Schur");
            hypre_check(HYPRE_BoomerAMGSetKeepTranspose(precond, memoirs_env_int("MEMOIRS_AMG_KEEP_TRANSPOSE", 1)), "HYPRE_BoomerAMGSetKeepTranspose RT1 reusable Schur");
            hypre_check(HYPRE_BoomerAMGSetTruncFactor(precond, memoirs_env_double("MEMOIRS_AMG_TRUNC", 0.0)), "HYPRE_BoomerAMGSetTruncFactor RT1 reusable Schur");
            hypre_check(HYPRE_BoomerAMGSetRAP2(precond, memoirs_env_int("MEMOIRS_AMG_RAP2", 0)), "HYPRE_BoomerAMGSetRAP2 RT1 reusable Schur");
            const int agg = memoirs_env_int("MEMOIRS_AMG_AGG_LEVELS", 0);
            if (agg > 0) hypre_check(HYPRE_BoomerAMGSetAggNumLevels(precond, agg), "HYPRE_BoomerAMGSetAggNumLevels RT1 reusable Schur");
            const double strong = memoirs_env_double("MEMOIRS_AMG_STRONG", -1.0);
            if (strong >= 0.0) hypre_check(HYPRE_BoomerAMGSetStrongThreshold(precond, strong), "HYPRE_BoomerAMGSetStrongThreshold RT1 reusable Schur");
            hypre_check(
                HYPRE_PCGSetPrecond(
                    pcg,
                    (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSolve,
                    (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSetup,
                    precond
                ),
                "HYPRE_PCGSetPrecond AMG RT1 reusable Schur"
            );
        } else if (pre == "none") {
            // no preconditioner
        } else {
            throw std::runtime_error("Unsupported MEMOIRS_RT1_SCHUR_PRECOND in reusable Schur solver: " + sopt.precond);
        }

        hypre_check(HYPRE_ParCSRPCGSetup(pcg, parA, parb, parx), "HYPRE_ParCSRPCGSetup RT1 reusable Schur");
        auto tSetup1 = std::chrono::steady_clock::now();
        stats.hypreSetupSeconds += elapsed_seconds(tSetup0, tSetup1);
        ready = true;
    }

    void solve(const std::vector<Real>& rhs, std::vector<Real>& xOut, Rt1GlobalSchurStats& stats) {
        if (!ready) throw std::runtime_error("RT1 reusable Schur solver used before setup.");
        if ((int)rhs.size() != nHost) throw std::runtime_error("RT1 reusable Schur RHS size mismatch.");
        const HYPRE_Int n = (HYPRE_Int)nHost;

        auto tVec0 = std::chrono::steady_clock::now();
        for (HYPRE_Int i=0; i<n; ++i) {
            bvals[i] = (HYPRE_Complex)rhs[i];
            xvals[i] = HYPRE_Complex(0);
        }
        if (useDevice) {
            hypre_check(HYPRE_IJVectorMigrate(ijb, HYPRE_MEMORY_HOST), "HYPRE_IJVectorMigrate b HOST RT1 reusable Schur update");
            hypre_check(HYPRE_IJVectorMigrate(ijx, HYPRE_MEMORY_HOST), "HYPRE_IJVectorMigrate x HOST RT1 reusable Schur update");
        }
        hypre_check(HYPRE_IJVectorSetValues(ijb, n, rows.data(), bvals.data()), "HYPRE_IJVectorSetValues b RT1 reusable Schur update");
        hypre_check(HYPRE_IJVectorSetValues(ijx, n, rows.data(), xvals.data()), "HYPRE_IJVectorSetValues x RT1 reusable Schur update");
        hypre_check(HYPRE_IJVectorAssemble(ijb), "HYPRE_IJVectorAssemble b RT1 reusable Schur update");
        hypre_check(HYPRE_IJVectorAssemble(ijx), "HYPRE_IJVectorAssemble x RT1 reusable Schur update");
        auto tVec1 = std::chrono::steady_clock::now();
        stats.hypreVectorInsertSeconds += elapsed_seconds(tVec0, tVec1);

        auto tVecMig0 = std::chrono::steady_clock::now();
        if (useDevice) {
            hypre_check(HYPRE_IJVectorMigrate(ijb, HYPRE_MEMORY_DEVICE), "HYPRE_IJVectorMigrate b DEVICE RT1 reusable Schur update");
            hypre_check(HYPRE_IJVectorMigrate(ijx, HYPRE_MEMORY_DEVICE), "HYPRE_IJVectorMigrate x DEVICE RT1 reusable Schur update");
        }
        hypre_check(HYPRE_IJVectorGetObject(ijb, (void**)&parb), "HYPRE_IJVectorGetObject b RT1 reusable Schur update");
        hypre_check(HYPRE_IJVectorGetObject(ijx, (void**)&parx), "HYPRE_IJVectorGetObject x RT1 reusable Schur update");
        auto tVecMig1 = std::chrono::steady_clock::now();
        stats.hypreVectorMigrateSeconds += elapsed_seconds(tVecMig0, tVecMig1);

        auto tSolve0 = std::chrono::steady_clock::now();
        hypre_check(HYPRE_ParVectorSetConstantValues(parx, HYPRE_Complex(0)), "HYPRE_ParVectorSetConstantValues RT1 reusable Schur");
        hypre_check(HYPRE_ParCSRPCGSolve(pcg, parA, parb, parx), "HYPRE_ParCSRPCGSolve RT1 reusable Schur");
        auto tSolve1 = std::chrono::steady_clock::now();
        stats.hypreSolveOnlySeconds += elapsed_seconds(tSolve0, tSolve1);

        int its = 0;
        hypre_check(HYPRE_PCGGetNumIterations(pcg, &its), "HYPRE_PCGGetNumIterations RT1 reusable Schur");
        HYPRE_Real finalRel = 0.0;
        hypre_check(HYPRE_PCGGetFinalRelativeResidualNorm(pcg, &finalRel), "HYPRE_PCGGetFinalRelativeResidualNorm RT1 reusable Schur");
        stats.schurApplications += 1;
        stats.schurIterationsTotal += its;
        stats.schurIterationsMax = std::max(stats.schurIterationsMax, its);
        stats.schurFinalRelLast = (double)finalRel;

        auto tGet0 = std::chrono::steady_clock::now();
        if (useDevice) hypre_check(HYPRE_IJVectorMigrate(ijx, HYPRE_MEMORY_HOST), "HYPRE_IJVectorMigrate x HOST RT1 reusable Schur get");
        std::fill(xvals.begin(), xvals.end(), HYPRE_Complex(0));
        hypre_check(HYPRE_IJVectorGetValues(ijx, n, rows.data(), xvals.data()), "HYPRE_IJVectorGetValues x RT1 reusable Schur");
        xOut.assign(nHost, Real(0));
        for (HYPRE_Int i=0; i<n; ++i) xOut[i] = Real(xvals[i]);
        auto tGet1 = std::chrono::steady_clock::now();
        stats.hypreGetSolutionSeconds += elapsed_seconds(tGet0, tGet1);
    }

    void destroy(Rt1GlobalSchurStats& stats) {
        if (!ready) return;
        auto t0 = std::chrono::steady_clock::now();
        if (precond) { hypre_check(HYPRE_BoomerAMGDestroy(precond), "HYPRE_BoomerAMGDestroy RT1 reusable Schur"); precond = nullptr; }
        if (pcg) { hypre_check(HYPRE_ParCSRPCGDestroy(pcg), "HYPRE_ParCSRPCGDestroy RT1 reusable Schur"); pcg = nullptr; }
        if (ijx) { hypre_check(HYPRE_IJVectorDestroy(ijx), "HYPRE_IJVectorDestroy x RT1 reusable Schur"); ijx = nullptr; }
        if (ijb) { hypre_check(HYPRE_IJVectorDestroy(ijb), "HYPRE_IJVectorDestroy b RT1 reusable Schur"); ijb = nullptr; }
        if (ijA) { hypre_check(HYPRE_IJMatrixDestroy(ijA), "HYPRE_IJMatrixDestroy A RT1 reusable Schur"); ijA = nullptr; }
        hypre_check(HYPRE_Finalize(), "HYPRE_Finalize RT1 reusable Schur");
        auto t1 = std::chrono::steady_clock::now();
        stats.hypreDestroyFinalizeSeconds += elapsed_seconds(t0, t1);
        ready = false;
    }
};

static std::vector<Real> rt1_apply_global_schur_amg_reuse_preconditioner(
    const Rt1GlobalSchurData& data,
    Rt1ReusableSchurSolver& schurSolver,
    const std::vector<Real>& r,
    Rt1GlobalSchurStats& stats
) {
    const int nF = data.nF;
    const int nS = data.nS;
    if ((int)r.size() != nF + nS) throw std::runtime_error("RT1 global Schur residual size mismatch.");

    std::vector<Real> rhsS(nS, Real(0));
    for (int si=0; si<nS; ++si) {
        double v = -double(r[nF + si]);
        for (const auto& fb : data.scalarB[si]) {
            const int f = fb.first;
            const double B = fb.second;
            v -= B * data.dinv[f] * double(r[f]);
        }
        rhsS[si] = Real(v);
    }

    std::vector<Real> zu;
    schurSolver.solve(rhsS, zu, stats);
    if ((int)zu.size() != nS) throw std::runtime_error("RT1 reusable global Schur solve returned wrong scalar size.");

    std::vector<Real> z(nF+nS, Real(0));
    for (int si=0; si<nS; ++si) z[nF+si] = zu[si];

    for (int f=0; f<nF; ++f) {
        double v = double(r[f]);
        for (const auto& sb : data.fluxBT[f]) {
            const int si = sb.first;
            const double B = sb.second;
            v += B * double(zu[si]);
        }
        z[f] = Real(data.dinv[f] * v);
    }
    return z;
}

static std::vector<Real> rt1_apply_global_schur_amg_preconditioner(
    const Rt1GlobalSchurData& data,
    const std::vector<Real>& r,
    const Options& opt,
    Rt1GlobalSchurStats& stats
) {
    const int nF = data.nF;
    const int nS = data.nS;
    if ((int)r.size() != nF + nS) throw std::runtime_error("RT1 global Schur residual size mismatch.");

    // rhsS = -r_u - B D^{-1} r_q.
    std::vector<Real> rhsS(nS, Real(0));
    for (int si=0; si<nS; ++si) {
        double v = -double(r[nF + si]);
        for (const auto& fb : data.scalarB[si]) {
            const int f = fb.first;
            const double B = fb.second;
            v -= B * data.dinv[f] * double(r[f]);
        }
        rhsS[si] = Real(v);
    }

    Options sopt = opt;
    sopt.solver = "pcg";
    sopt.precond = memoirs_rt1_env_string("MEMOIRS_RT1_SCHUR_PRECOND", "amg");
    sopt.tol = memoirs_env_double("MEMOIRS_RT1_SCHUR_TOL", 1e-4);
    sopt.maxit = memoirs_env_int("MEMOIRS_RT1_SCHUR_MAXIT", 100);

    std::vector<Real> zu;
    SolveReport srep = solve_hypre_ij_pcg_raw(data.S, rhsS, sopt, zu);
    if ((int)zu.size() != nS) throw std::runtime_error("RT1 global Schur solve returned wrong scalar size.");

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

    std::vector<Real> z(nF+nS, Real(0));
    for (int si=0; si<nS; ++si) z[nF+si] = zu[si];

    // z_q = D^{-1}(r_q + B^T z_u).
    for (int f=0; f<nF; ++f) {
        double v = double(r[f]);
        for (const auto& sb : data.fluxBT[f]) {
            const int si = sb.first;
            const double B = sb.second;
            v += B * double(zu[si]);
        }
        z[f] = Real(data.dinv[f] * v);
    }
    return z;
}

static SolveReport solve_rt1_block_schur_amg_fgmres(
    const SparseRows& A,
    const Rt1DofMap& dm,
    const std::vector<Real>& b,
    const Options& opt,
    std::vector<Real>& xOut
) {
    int mpiWasInitialized = 0;
    MPI_Initialized(&mpiWasInitialized);
    bool startedMPI = false;
    if (!mpiWasInitialized) {
        int argc=0; char** argv=nullptr;
        MPI_Init(&argc, &argv);
        startedMPI = true;
    }

    auto t0 = std::chrono::steady_clock::now();
    Rt1GlobalSchurData data = rt1_build_global_diag_schur_data(A, dm);
    Rt1GlobalSchurStats stats;

    if (opt.diagLevel >= 1) {
        std::cout << "---------------- RT1 global Schur probe ----------------\n";
        std::cout << "schurApprox                   = B diag(M)^{-1} B^T\n";
        std::cout << "schurRows                     = " << data.nS << "\n";
        std::cout << "schurNnz                      = " << data.schurNnz << "\n";
        std::cout << "schurPrecond                  = " << memoirs_rt1_env_string("MEMOIRS_RT1_SCHUR_PRECOND", "amg") << "\n";
        std::cout << "schurTol                      = " << std::scientific << memoirs_env_double("MEMOIRS_RT1_SCHUR_TOL", 1e-4) << std::defaultfloat << "\n";
        std::cout << "schurMaxit                    = " << memoirs_env_int("MEMOIRS_RT1_SCHUR_MAXIT", 100) << "\n";
        std::cout << "fluxMassDiagMin               = " << std::scientific << data.fluxMassDiagMin << "\n";
        std::cout << "fluxMassDiagMax               = " << data.fluxMassDiagMax << std::defaultfloat << "\n";
        std::cout << "--------------------------------------------------------\n";
    }

    Rt1ReusableSchurSolver schurSolver;
    schurSolver.setup(data.S, opt, stats);

    auto apply_prec = [&](const std::vector<Real>& r) {
        return rt1_apply_global_schur_amg_reuse_preconditioner(data, schurSolver, r, stats);
    };

    SolveReport rep = solve_rt1_fgmres_with_preconditioner(A, dm.nTotalDofs, b, opt, xOut, apply_prec);
    schurSolver.destroy(stats);
    auto t1 = std::chrono::steady_clock::now();
    rep.solveSeconds = std::chrono::duration<double>(t1-t0).count();
    rep.hypreMatrixInsertSeconds = stats.hypreMatrixInsertSeconds;
    rep.hypreMatrixMigrateSeconds = stats.hypreMatrixMigrateSeconds;
    rep.hypreVectorInsertSeconds = stats.hypreVectorInsertSeconds;
    rep.hypreVectorMigrateSeconds = stats.hypreVectorMigrateSeconds;
    rep.hypreSetupSeconds = stats.hypreSetupSeconds;
    rep.hypreSolveOnlySeconds = stats.hypreSolveOnlySeconds;
    rep.hypreGetSolutionSeconds = stats.hypreGetSolutionSeconds;
    rep.hypreDestroyFinalizeSeconds = stats.hypreDestroyFinalizeSeconds;

    const double avg = stats.schurApplications > 0 ? double(stats.schurIterationsTotal)/double(stats.schurApplications) : 0.0;
    std::cout << "---------------- RT1 global Schur FGMRES probe -----------\n";
    std::cout << "outerSolver                   = host_restarted_fgmres\n";
    std::cout << "outerOperator                 = original_rt1_dg1_saddle_matrix\n";
    std::cout << "preconditioner                = block_triangular_Dinv_plus_global_schur\n";
    std::cout << "schurApprox                   = B diag(M)^{-1} B^T\n";
    std::cout << "schurPrecond                  = " << memoirs_rt1_env_string("MEMOIRS_RT1_SCHUR_PRECOND", "amg") << "\n";
    std::cout << "outerIterations               = " << rep.iterations << "\n";
    std::cout << "outerFinalRelativeResidual    = " << std::scientific << rep.finalRelRes << std::defaultfloat << "\n";
    std::cout << "schurReuseSetup               = 1\n";
    std::cout << "schurApplications             = " << stats.schurApplications << "\n";
    std::cout << "schurIterationsTotal          = " << stats.schurIterationsTotal << "\n";
    std::cout << "schurIterationsAvg            = " << avg << "\n";
    std::cout << "schurIterationsMax            = " << stats.schurIterationsMax << "\n";
    std::cout << "schurFinalRelLast             = " << std::scientific << stats.schurFinalRelLast << std::defaultfloat << "\n";
    std::cout << "fluxMassDiagMin               = " << std::scientific << data.fluxMassDiagMin << "\n";
    std::cout << "fluxMassDiagMax               = " << data.fluxMassDiagMax << std::defaultfloat << "\n";
    std::cout << "----------------------------------------------------------\n";

    if (startedMPI) MPI_Finalize();
    return rep;
}

static Rt1FieldErrorReport compute_rt1_tet_dg1_errors(
    const PolyMesh& m, const Rt1DofMap& dm, const std::string& mms,
    const std::vector<Real>& x
) {
    auto geoms=build_rt1_cell_geoms(m);
    Rt1FieldErrorReport er;
    double eU2=0,eQ2=0,eDiv2=0,cons2=0, maxDiv=0, maxCons=0;
    for (int c=0;c<(int)m.cells.size();++c) {
        const Rt1CellGeom& K=geoms[c];
        Rt1LocalBasis LB=rt1_build_local_basis(m,dm,c,K,geoms);
        double sourceInt=0, divInt=0;
        for (const auto& q: tet_volume_quadrature_selected_for_error()) {
            auto P=eval_tet_p1_basis_at(q.xi,q.eta,q.zeta);
            auto RB=rt1_eval_raw_basis(m,K,q.xi,q.eta,q.zeta);
            Vec3 xp=map_tet_p1_to_physical(m,K.verts,P.N);
            double w=q.weight*std::abs(K.detJ);
            double uh=0.0;
            for (int i=0;i<4;++i) uh += double(x[dm.nFluxDofs+4*c+i])*P.N[i];
            Vec3 qh{}; double divh=0.0;
            for (int a=0;a<15;++a) {
                double qa=double(x[LB.globalFlux[a]]);
                qh=rt1_add(qh, rt1_mul(qa, rt1_eval_basis_vec(LB,a,RB)));
                divh += qa * rt1_eval_basis_div(LB,a,RB);
            }
            double ue=mms_exact_value(xp,mms);
            Vec3 qe=rt1_mul(-1.0,mms_exact_grad(xp,mms));
            double eu=uh-ue;
            Vec3 eq=rt1_sub(qh,qe);
            eU2 += w*eu*eu;
            eQ2 += w*dot3(eq,eq);
            double src=mms_rhs_value(xp,mms);
            double ed=divh-src;
            eDiv2 += w*ed*ed;
            sourceInt += w*src;
            divInt += w*divh;
        }
        double cons=divInt-sourceInt;
        cons2 += cons*cons;
        maxCons=std::max(maxCons,std::abs(cons));
    }
    er.scalarL2=std::sqrt(eU2); er.fluxL2=std::sqrt(eQ2); er.divCellL2=std::sqrt(eDiv2);
    er.divCellAbsMax=maxDiv; er.cellConservationL2=std::sqrt(cons2); er.cellConservationAbsMax=maxCons;
    return er;
}

static void probe_rt1_system(const Rt1DofMap& dm, const AssembledSystem& sys) {
    std::cout << "---------------- RT1/DG1 system probe ----------------\n";
    std::cout << "nFluxDofs                 = " << dm.nFluxDofs << "\n";
    std::cout << "nScalarDofs               = " << dm.nScalarDofs << "\n";
    std::cout << "nTotalDofs                = " << dm.nTotalDofs << "\n";
    std::cout << "nnz                       = " << sparse_nnz(sys.A) << "\n";
    std::cout << "------------------------------------------------------\n";
}
