#pragma once

struct Q1Q1AlgInfo {
    int nNodes = 0;
    int nRows = 0;
    int nBoundaryNodes = 0;
    int nStrongVelocityRows = 0;
    int pressurePinNode = 0;
    std::size_t nnz = 0;
    double nu = 1.0;
    double tau = 0.0;
    double hmin = 0.0;
};

struct Q1Q1AlgError {
    double uL2 = -1.0;
    double uMax = -1.0;
    double pL2 = -1.0;
    double pMax = -1.0;
    double pMeanShift = 0.0;
    double pMeanShiftedL2 = -1.0;
};

static inline double q1q1_cell_hmin(const PolyMesh& mesh, const Cell& c) {
    const auto hv = ordered_hex_vertices_axis_aligned(mesh, c);

    double xmin =  1e300, ymin =  1e300, zmin =  1e300;
    double xmax = -1e300, ymax = -1e300, zmax = -1e300;

    for (int a = 0; a < 8; ++a) {
        const Vec3& X = mesh.points[hv[a]];
        xmin = std::min(xmin, X.x); xmax = std::max(xmax, X.x);
        ymin = std::min(ymin, X.y); ymax = std::max(ymax, X.y);
        zmin = std::min(zmin, X.z); zmax = std::max(zmax, X.z);
    }

    return std::min({xmax - xmin, ymax - ymin, zmax - zmin});
}

static inline double q1q1_global_hmin(const PolyMesh& mesh) {
    double h = 1e300;
    for (const auto& c : mesh.cells) h = std::min(h, q1q1_cell_hmin(mesh, c));
    if (!(h > 0.0) || !std::isfinite(h)) throw std::runtime_error("Bad q1q1 hmin.");
    return h;
}

static inline void q1q1_identity_row(SparseRows& A, int row) {
    sparse_set_row_identity(A, row);
}

static inline std::vector<Real> q1q1_sparse_matvec(const SparseRows& A, const std::vector<Real>& x) {
    const int n = A.fixedPattern ? (int)A.cols.size() : (int)A.rows.size();
    std::vector<Real> y(n, Real(0));
    for (int i = 0; i < n; ++i) y[i] = sparse_matvec_row(A, i, x);
    return y;
}

static inline AssembledSystem assemble_q1q1_stokes_pspg_algebraic(
    const PolyMesh& mesh,
    const LinearCgDofMap& dm,
    Q1Q1AlgInfo& info
) {
    if (dm.resolvedSpace != "cg_hex_q1") {
        throw std::runtime_error("Q1/Q1 algebraic matrix currently requires -space cg_hex_q1.");
    }
    if (dm.nDofs != (int)mesh.points.size()) {
        throw std::runtime_error("Q1/Q1 node-major assembly expects scalar Q1 DOF = OpenFOAM point index.");
    }

    const int nNodes = dm.nDofs;
    const int nRows = 4 * nNodes;

    info.nNodes = nNodes;
    info.nRows = nRows;
    info.nBoundaryNodes = dm.nBoundaryDofs;
    info.nStrongVelocityRows = 3 * dm.nBoundaryDofs;
    info.pressurePinNode = memoirs_env_int("MEMOIRS_Q1Q1_PRESSURE_PIN", 0);
    info.nu = memoirs_env_double("MEMOIRS_Q1Q1_NU", 1.0);
    info.hmin = q1q1_global_hmin(mesh);

    const double tauUser = memoirs_env_double("MEMOIRS_Q1Q1_TAU", -1.0);
    const double tauScale = memoirs_env_double("MEMOIRS_Q1Q1_TAU_SCALE", 0.25);
    info.tau = (tauUser > 0.0) ? tauUser : tauScale * info.hmin * info.hmin / info.nu;

    if (!(info.nu > 0.0)) throw std::runtime_error("MEMOIRS_Q1Q1_NU must be positive.");
    if (!(info.tau > 0.0)) throw std::runtime_error("Q1/Q1 PSPG tau must be positive.");
    if (info.pressurePinNode < 0 || info.pressurePinNode >= nNodes) {
        throw std::runtime_error("Bad MEMOIRS_Q1Q1_PRESSURE_PIN.");
    }

    AssembledSystem sys;
    sys.A.rows.resize(nRows);
    sys.b.assign(nRows, Real(0));
    sys.nDirichlet = info.nStrongVelocityRows + 1;

    const double q = 1.0 / std::sqrt(3.0);
    const double qp[2] = {-q, q};

    for (const auto& c : mesh.cells) {
        const auto hv = ordered_hex_vertices_axis_aligned(mesh, c);

        Real K[8][8] = {};
        Real Gx[8][8] = {};
        Real Gy[8][8] = {};
        Real Gz[8][8] = {};

        for (int iq = 0; iq < 2; ++iq) {
            for (int jq = 0; jq < 2; ++jq) {
                for (int kq = 0; kq < 2; ++kq) {
                    const double xi = qp[iq];
                    const double eta = qp[jq];
                    const double zeta = qp[kq];

                    std::array<double,8> N;
                    std::array<Vec3,8> dNref;
                    hex_q1_basis(xi, eta, zeta, N, dNref);

                    Mat3 J;
                    for (int a = 0; a < 8; ++a) {
                        const Vec3& X = mesh.points[hv[a]];

                        J.a[0][0] += dNref[a].x * X.x;
                        J.a[1][0] += dNref[a].x * X.y;
                        J.a[2][0] += dNref[a].x * X.z;

                        J.a[0][1] += dNref[a].y * X.x;
                        J.a[1][1] += dNref[a].y * X.y;
                        J.a[2][1] += dNref[a].y * X.z;

                        J.a[0][2] += dNref[a].z * X.x;
                        J.a[1][2] += dNref[a].z * X.y;
                        J.a[2][2] += dNref[a].z * X.z;
                    }

                    const double detJ = std::abs(det3(J));
                    const Mat3 invJ = inv3(J);

                    std::array<Vec3,8> grad;
                    for (int a = 0; a < 8; ++a) grad[a] = invJT_mul(invJ, dNref[a]);

                    for (int a = 0; a < 8; ++a) {
                        for (int b = 0; b < 8; ++b) {
                            K[a][b] += Real(dot3(grad[a], grad[b]) * detJ);

                            // G_comp(q_a, u_b) = integral N_a * dN_b/dcomp.
                            Gx[a][b] += Real(N[a] * grad[b].x * detJ);
                            Gy[a][b] += Real(N[a] * grad[b].y * detJ);
                            Gz[a][b] += Real(N[a] * grad[b].z * detJ);
                        }
                    }
                }
            }
        }

        for (int a = 0; a < 8; ++a) {
            const int na = hv[a];

            for (int b = 0; b < 8; ++b) {
                const int nb = hv[b];

                // Velocity diffusion block: nu (grad v, grad u).
                for (int comp = 0; comp < 3; ++comp) {
                    sparse_add(sys.A, q1q1_row(na, comp), q1q1_row(nb, comp),
                               Real(info.nu) * K[a][b]);
                }

                // Momentum pressure block: -(p, div v).
                sparse_add(sys.A, q1q1_row(na, 0), q1q1_row(nb, 3), -Gx[b][a]);
                sparse_add(sys.A, q1q1_row(na, 1), q1q1_row(nb, 3), -Gy[b][a]);
                sparse_add(sys.A, q1q1_row(na, 2), q1q1_row(nb, 3), -Gz[b][a]);

                // Continuity block: (q, div u).
                sparse_add(sys.A, q1q1_row(na, 3), q1q1_row(nb, 0), Gx[a][b]);
                sparse_add(sys.A, q1q1_row(na, 3), q1q1_row(nb, 1), Gy[a][b]);
                sparse_add(sys.A, q1q1_row(na, 3), q1q1_row(nb, 2), Gz[a][b]);

                // PSPG pressure-pressure block: tau (grad q, grad p).
                sparse_add(sys.A, q1q1_row(na, 3), q1q1_row(nb, 3),
                           Real(info.tau) * K[a][b]);
            }
        }
    }

    // Strong homogeneous velocity on all boundary nodes.
    // This MMS velocity is exactly zero on all unit-cube walls.
    for (int node : dm.boundaryDofs) {
        q1q1_identity_row(sys.A, q1q1_row(node, 0));
        q1q1_identity_row(sys.A, q1q1_row(node, 1));
        q1q1_identity_row(sys.A, q1q1_row(node, 2));
    }

    // Single pressure pin. The chosen pressure has p(0,0,0)=0.
    q1q1_identity_row(sys.A, q1q1_row(info.pressurePinNode, 3));

    // Algebraic MMS: exact vector is the exact solution by construction.
    const std::vector<Real> xExact = q1q1_make_exact_node_major_vector(mesh);
    sys.b = q1q1_sparse_matvec(sys.A, xExact);

    info.nnz = sparse_nnz(sys.A);
    return sys;
}

static inline Q1Q1AlgError q1q1_compute_alg_error(const PolyMesh& mesh, const std::vector<Real>& x) {
    const int nNodes = (int)mesh.points.size();
    if ((int)x.size() != 4 * nNodes) throw std::runtime_error("Q1/Q1 error expects 4*nNodes vector.");

    Q1Q1AlgError e;
    double u2 = 0.0, p2 = 0.0, pShift = 0.0;

    for (int node = 0; node < nNodes; ++node) {
        const Q1Q1ExactAtNode ex = q1q1_exact_mms_at(mesh.points[node]);

        const double dux = double(x[q1q1_row(node, 0)]) - ex.ux;
        const double duy = double(x[q1q1_row(node, 1)]) - ex.uy;
        const double duz = double(x[q1q1_row(node, 2)]) - ex.uz;
        const double dp  = double(x[q1q1_row(node, 3)]) - ex.p;

        const double un = std::sqrt(dux*dux + duy*duy + duz*duz);

        u2 += un * un;
        p2 += dp * dp;
        pShift += dp;

        e.uMax = std::max(e.uMax, un);
        e.pMax = std::max(e.pMax, std::abs(dp));
    }

    pShift /= double(nNodes);

    double ps2 = 0.0;
    for (int node = 0; node < nNodes; ++node) {
        const Q1Q1ExactAtNode ex = q1q1_exact_mms_at(mesh.points[node]);
        const double dp = double(x[q1q1_row(node, 3)]) - ex.p;
        const double dps = dp - pShift;
        ps2 += dps * dps;
    }

    e.uL2 = std::sqrt(u2);
    e.pL2 = std::sqrt(p2);
    e.pMeanShift = pShift;
    e.pMeanShiftedL2 = std::sqrt(ps2);
    return e;
}

static inline void q1q1_print_alg_info(const Q1Q1AlgInfo& info) {
    std::cout << "--------------- q1q1 alg matrix info ----------------\n";
    std::cout << "q1q1Operator              = stokes_pspg_algebraic_rhs\n";
    std::cout << "q1q1Rows                  = " << info.nRows << "\n";
    std::cout << "q1q1Nodes                 = " << info.nNodes << "\n";
    std::cout << "q1q1BoundaryNodes         = " << info.nBoundaryNodes << "\n";
    std::cout << "q1q1StrongVelocityRows    = " << info.nStrongVelocityRows << "\n";
    std::cout << "q1q1PressurePinNode       = " << info.pressurePinNode << "\n";
    std::cout << "q1q1Nnz                   = " << info.nnz << "\n";
    std::cout << "q1q1Nu                    = " << std::setprecision(16) << info.nu << "\n";
    std::cout << "q1q1Hmin                  = " << std::setprecision(16) << info.hmin << "\n";
    std::cout << "q1q1Tau                   = " << std::setprecision(16) << info.tau << "\n";
    std::cout << "q1q1RhsMode               = algebraic_A_times_exact\n";
    std::cout << "-----------------------------------------------------\n";
}

static inline void q1q1_print_alg_error(const Q1Q1AlgError& e) {
    std::cout << "--------------- q1q1 alg nodal error ----------------\n";
    std::cout << "q1q1UNodalL2              = " << std::setprecision(16) << e.uL2 << "\n";
    std::cout << "q1q1UNodalMax             = " << std::setprecision(16) << e.uMax << "\n";
    std::cout << "q1q1PNodalL2              = " << std::setprecision(16) << e.pL2 << "\n";
    std::cout << "q1q1PNodalMax             = " << std::setprecision(16) << e.pMax << "\n";
    std::cout << "q1q1PMeanShift            = " << std::setprecision(16) << e.pMeanShift << "\n";
    std::cout << "q1q1PMeanShiftedNodalL2   = " << std::setprecision(16) << e.pMeanShiftedL2 << "\n";
    std::cout << "-----------------------------------------------------\n";
}
