#pragma once

// ============================================================================
// SECTION 14: Q1/Q1 transient Picard NSE MMS, PSPG + optional SUPG, grad-div/LSIC disabled
// ============================================================================
//
// Linearized transient equation per Picard iteration:
//
//   (u/dt) + beta.grad(u) - nu Lap(u) + grad(p)
//     = f_nse + u_old/dt
//
//   div(u) = 0
//
// where the manufactured stationary solution has u_old = u_exact, and
//
//   f_nse = advScale * (u_exact.grad)u_exact - nu Lap(u_exact) + grad(p_exact)
//
// PSPG pressure row:
//
//   (q, div u)
//   + tau (grad q, u/dt)
//   + tau (grad q, beta.grad u)
//   + tau (grad q, grad p)
//   = tau (grad q, f_nse + u_old/dt)
//
// This patch includes optional SUPG and optional grad-div/LSIC stabilization.
// ============================================================================

struct Q1Q1NsePicardOptions {
    double nu = 1.0;
    double dt = 1.0e-3;
    double advScale = 1.0;

    // Match the working structured Q1/Q1 NSE defaults.
    double tau = -1.0;
    double tauScale = 1.0;
    std::string tauMode = "metric";
    double tauC = 4.0;      // diffusion constant in metric tau
    double tauCt = 2.0;     // transient constant: tau ~ dt/tauCt when time term dominates
    double tauAdvScale = 1.0;

    int supg = 1;
    double supgTauScale = 1.0;

    // Grad-div / LSIC term:
    //   gamma (div v, div u)
    // Default coefficient when enabled:
    //   gamma = gradDivScale / nu
    // Override with gradDivCoeff > 0 if desired.
    int gradDiv = 0;
    double gradDivScale = 1.0;
    double gradDivCoeff = -1.0;

    int pressurePinNode = 0;
};

struct Q1Q1VelocityGradientExact {
    double dux_dx = 0.0;
    double dux_dy = 0.0;
    double dux_dz = 0.0;

    double duy_dx = 0.0;
    double duy_dy = 0.0;
    double duy_dz = 0.0;
};

static inline Q1Q1VelocityGradientExact q1q1_exact_velocity_gradient_at(const Vec3& X) {
    const double pi = kPi;

    const double sx = std::sin(pi * X.x);
    const double sy = std::sin(pi * X.y);
    const double sz = std::sin(pi * X.z);

    const double cx = std::cos(pi * X.x);
    const double cy = std::cos(pi * X.y);

    const double s2x = std::sin(2.0 * pi * X.x);
    const double s2y = std::sin(2.0 * pi * X.y);
    const double s2z = std::sin(2.0 * pi * X.z);

    const double c2x = std::cos(2.0 * pi * X.x);
    const double c2y = std::cos(2.0 * pi * X.y);

    const double A = sx * sx;
    const double B = sy * cy;
    const double C = sz * sz;

    const double D = sx * cx;
    const double E = sy * sy;

    Q1Q1VelocityGradientExact g;

    // ux =  2*pi * sx^2 * sy*cy * sz^2
    g.dux_dx =  2.0 * pi * pi * s2x * B * C;
    g.dux_dy =  2.0 * pi * pi * A * c2y * C;
    g.dux_dz =  2.0 * pi * pi * A * B * s2z;

    // uy = -2*pi * sx*cx * sy^2 * sz^2
    g.duy_dx = -2.0 * pi * pi * c2x * E * C;
    g.duy_dy = -2.0 * pi * pi * D * s2y * C;
    g.duy_dz = -2.0 * pi * pi * D * E * s2z;

    return g;
}

static inline Q1Q1ForceAtPoint q1q1_nse_force_at(const Vec3& X, double nu, double advScale) {
    Q1Q1ForceAtPoint f = q1q1_stokes_force_at(X, nu);

    const Q1Q1ExactAtNode u = q1q1_exact_mms_at(X);
    const Q1Q1VelocityGradientExact g = q1q1_exact_velocity_gradient_at(X);

    const double convx = u.ux * g.dux_dx + u.uy * g.dux_dy + u.uz * g.dux_dz;
    const double convy = u.ux * g.duy_dx + u.uy * g.duy_dy + u.uz * g.duy_dz;
    const double convz = 0.0;

    f.fx += advScale * convx;
    f.fy += advScale * convy;
    f.fz += advScale * convz;

    return f;
}

static inline void q1q1_eval_beta_at_qp(
    const std::array<int,8>& hv,
    const std::array<double,8>& N,
    const std::vector<Real>& beta,
    double& bx,
    double& by,
    double& bz
) {
    bx = 0.0;
    by = 0.0;
    bz = 0.0;

    for (int a = 0; a < 8; ++a) {
        const int node = hv[a];
        bx += N[a] * double(beta[q1q1_row(node, 0)]);
        by += N[a] * double(beta[q1q1_row(node, 1)]);
        bz += N[a] * double(beta[q1q1_row(node, 2)]);
    }
}

static inline std::vector<Real> q1q1_make_zero_beta(const PolyMesh& mesh) {
    return std::vector<Real>(4 * mesh.points.size(), Real(0));
}

static inline std::vector<Real> q1q1_make_exact_beta(const PolyMesh& mesh) {
    return q1q1_make_exact_node_major_vector(mesh);
}

static inline double q1q1_velocity_relative_update(
    const std::vector<Real>& oldBeta,
    const std::vector<Real>& x
) {
    if (oldBeta.size() != x.size()) return std::numeric_limits<double>::infinity();

    double num2 = 0.0;
    double den2 = 0.0;

    const int nNodes = (int)x.size() / 4;
    for (int node = 0; node < nNodes; ++node) {
        for (int c = 0; c < 3; ++c) {
            const double d = double(x[q1q1_row(node, c)] - oldBeta[q1q1_row(node, c)]);
            const double v = double(x[q1q1_row(node, c)]);
            num2 += d * d;
            den2 += v * v;
        }
    }

    return std::sqrt(num2 / std::max(den2, 1.0e-300));
}

static inline void q1q1_copy_velocity_to_beta(
    const std::vector<Real>& x,
    std::vector<Real>& beta
) {
    if (beta.size() != x.size()) beta.assign(x.size(), Real(0));

    const int nNodes = (int)x.size() / 4;
    for (int node = 0; node < nNodes; ++node) {
        beta[q1q1_row(node, 0)] = x[q1q1_row(node, 0)];
        beta[q1q1_row(node, 1)] = x[q1q1_row(node, 1)];
        beta[q1q1_row(node, 2)] = x[q1q1_row(node, 2)];
        beta[q1q1_row(node, 3)] = Real(0);
    }
}


static inline double q1q1_beta_max_speed(const std::vector<Real>& beta) {
    double umax = 0.0;
    const int nNodes = (int)beta.size() / 4;
    for (int node = 0; node < nNodes; ++node) {
        const double ux = double(beta[q1q1_row(node, 0)]);
        const double uy = double(beta[q1q1_row(node, 1)]);
        const double uz = double(beta[q1q1_row(node, 2)]);
        umax = std::max(umax, std::sqrt(ux*ux + uy*uy + uz*uz));
    }
    return umax;
}

static inline double q1q1_compute_tau_structured_match(
    const PolyMesh& mesh,
    const std::vector<Real>& beta,
    const Q1Q1NsePicardOptions& opt,
    double hmin
) {
    if (opt.tau > 0.0) return opt.tau;

    const double h = hmin;
    const double pi = kPi;
    const double volume = h * h * h;
    const double hEq = std::pow(6.0 * volume / pi, 1.0 / 3.0);

    if (opt.tauMode == "simple") {
        return opt.tauScale * h * h / opt.nu;
    }

    if (opt.tauMode == "sphere" || opt.tauMode == "spherec") {
        return opt.tauScale * hEq * hEq / (opt.tauC * opt.nu);
    }

    const double gx = 2.0 / h;
    const double gy = 2.0 / h;
    const double gz = 2.0 / h;
    const double metric2 = gx*gx + gy*gy + gz*gz;
    const double metric4 = gx*gx*gx*gx + gy*gy*gy*gy + gz*gz*gz*gz;

    if (opt.tauMode == "metric") {
        // Old diffusion-only metric tau retained for A/B testing.
        return opt.tauScale / ((opt.tauC * opt.nu) * std::sqrt(metric4));
    }

    if (opt.tauMode == "tezduyar" ||
        opt.tauMode == "shakib" ||
        opt.tauMode == "transient" ||
        opt.tauMode == "metric_transient") {
        // Tezduyar/Shakib-style transient SUPG/PSPG scale:
        // tau = tauScale / sqrt( (Ct/dt)^2 + beta.G.beta + (Cnu*nu)^2 G:G )
        // For these unit-cube blockMeshes we approximate G by diag((2/h)^2).
        const double umax = q1q1_beta_max_speed(beta);
        const double timeTerm = opt.tauCt / opt.dt;
        const double advTerm = opt.tauAdvScale * umax * std::sqrt(metric2 / 3.0);
        const double diffTerm = (opt.tauC * opt.nu) * std::sqrt(metric4);
        const double denom2 = timeTerm*timeTerm + advTerm*advTerm + diffTerm*diffTerm;
        return opt.tauScale / std::sqrt(std::max(denom2, 1.0e-300));
    }

    return opt.tauScale * h * h / opt.nu;
}


static inline void q1q1_make_nse_picard_fixed_pattern(
    const PolyMesh& mesh,
    const LinearCgDofMap& dm,
    int pressurePinNode,
    SparseRows& A
) {
    const int nNodes = dm.nDofs;
    const int nRows = 4 * nNodes;

    static std::shared_ptr<const SparseFixedPattern> cachedPattern;
    static int cachedNRows = -1;
    static int cachedNCells = -1;
    static int cachedPressurePinNode = -999;

    if (cachedPattern &&
        cachedNRows == nRows &&
        cachedNCells == (int)mesh.cells.size() &&
        cachedPressurePinNode == pressurePinNode) {
        sparse_init_from_fixed_pattern(A, cachedPattern);
        return;
    }

    std::vector<unsigned char> isBnd(nNodes, 0);
    for (int node : dm.boundaryDofs) {
        if (node >= 0 && node < nNodes) {
            isBnd[node] = 1;
        }
    }

    std::vector<std::vector<int>> rowCols(nRows);

    auto add = [&](int r, int c) {
        if (r < 0 || r >= nRows || c < 0 || c >= nRows) {
            throw std::runtime_error("q1q1 fixed pattern row/col out of range");
        }
        rowCols[r].push_back(c);
    };

    // Identity rows for strong velocity BCs and pressure pin.
    for (int node : dm.boundaryDofs) {
        add(q1q1_row(node, 0), q1q1_row(node, 0));
        add(q1q1_row(node, 1), q1q1_row(node, 1));
        add(q1q1_row(node, 2), q1q1_row(node, 2));
    }
    add(q1q1_row(pressurePinNode, 3), q1q1_row(pressurePinNode, 3));

    // Full Q1/Q1 element graph for active rows/columns.
    // Pattern includes grad-div cross velocity coupling unconditionally so
    // GRAD_DIV can be toggled without rebuilding the graph.
    for (int cellI = 0; cellI < (int)mesh.cells.size(); ++cellI) {
        const auto& c = mesh.cells[cellI];
        const auto hv = ordered_hex_vertices_axis_aligned(mesh, c);

        for (int a = 0; a < 8; ++a) {
            const int na = hv[a];

            for (int b = 0; b < 8; ++b) {
                const int nb = hv[b];

                // Momentum velocity rows.
                if (!isBnd[na]) {
                    // Velocity-velocity block.
                    // Grad-div/LSIC is disabled in this branch, so the
                    // velocity graph is component-diagonal:
                    //   ux-ux, uy-uy, uz-uz.
                    if (!isBnd[nb]) {
                        add(q1q1_row(na, 0), q1q1_row(nb, 0));
                        add(q1q1_row(na, 1), q1q1_row(nb, 1));
                        add(q1q1_row(na, 2), q1q1_row(nb, 2));
                    }

                    // Momentum-pressure block.
                    if (nb != pressurePinNode) {
                        add(q1q1_row(na, 0), q1q1_row(nb, 3));
                        add(q1q1_row(na, 1), q1q1_row(nb, 3));
                        add(q1q1_row(na, 2), q1q1_row(nb, 3));
                    }
                }

                // Pressure row: continuity + PSPG.
                if (na != pressurePinNode) {
                    if (!isBnd[nb]) {
                        add(q1q1_row(na, 3), q1q1_row(nb, 0));
                        add(q1q1_row(na, 3), q1q1_row(nb, 1));
                        add(q1q1_row(na, 3), q1q1_row(nb, 2));
                    }

                    if (nb != pressurePinNode) {
                        add(q1q1_row(na, 3), q1q1_row(nb, 3));
                    }
                }
            }
        }
    }

    cachedPattern = sparse_make_fixed_pattern(std::move(rowCols));
    cachedNRows = nRows;
    cachedNCells = (int)mesh.cells.size();
    cachedPressurePinNode = pressurePinNode;

    sparse_init_from_fixed_pattern(A, cachedPattern);
}


struct Q1Q1NseCellSlots {
    int vv[8][8][3]; // row-local slot: momentum velocity row -> same-component velocity col
    int vp[8][8][3]; // row-local slot: momentum velocity row -> pressure col
    int pv[8][8][3]; // row-local slot: pressure row -> velocity col
    int pp[8][8];    // row-local slot: pressure row -> pressure col

    int vvFlat[8][8][3]; // global flat value slot
    int vpFlat[8][8][3];
    int pvFlat[8][8][3];
    int ppFlat[8][8];
};

static inline void q1q1_init_cell_slots(Q1Q1NseCellSlots& cs) {
    for (int a = 0; a < 8; ++a) {
        for (int b = 0; b < 8; ++b) {
            cs.pp[a][b] = -1;
            cs.ppFlat[a][b] = -1;
            for (int c = 0; c < 3; ++c) {
                cs.vv[a][b][c] = -1;
                cs.vp[a][b][c] = -1;
                cs.pv[a][b][c] = -1;
                cs.vvFlat[a][b][c] = -1;
                cs.vpFlat[a][b][c] = -1;
                cs.pvFlat[a][b][c] = -1;
            }
        }
    }
}

static inline const std::vector<Q1Q1NseCellSlots>& q1q1_get_nse_cell_slots_cached(
    const PolyMesh& mesh,
    const LinearCgDofMap& dm,
    int pressurePinNode,
    const SparseRows& A
) {
    static std::vector<Q1Q1NseCellSlots> cached;
    static int cachedNCells = -1;
    static int cachedNNodes = -1;
    static int cachedPressurePinNode = -999;
    static const void* cachedPatternPtr = nullptr;

    const void* patternPtr = A.pattern ? (const void*)A.pattern.get() : (const void*)nullptr;

    if (!cached.empty() &&
        cachedNCells == (int)mesh.cells.size() &&
        cachedNNodes == dm.nDofs &&
        cachedPressurePinNode == pressurePinNode &&
        cachedPatternPtr == patternPtr) {
        return cached;
    }

    const int nNodes = dm.nDofs;
    std::vector<unsigned char> isBnd(nNodes, 0);
    for (int node : dm.boundaryDofs) {
        if (node >= 0 && node < nNodes) isBnd[node] = 1;
    }

    cached.assign(mesh.cells.size(), Q1Q1NseCellSlots{});

    for (int cellI = 0; cellI < (int)mesh.cells.size(); ++cellI) {
        auto& cs = cached[cellI];
        q1q1_init_cell_slots(cs);

        const auto hv = ordered_hex_vertices_axis_aligned(mesh, mesh.cells[cellI]);

        for (int a = 0; a < 8; ++a) {
            const int na = hv[a];

            for (int b = 0; b < 8; ++b) {
                const int nb = hv[b];

                if (!isBnd[na] && !isBnd[nb]) {
                    cs.vv[a][b][0] = sparse_lookup_slot(A, q1q1_row(na, 0), q1q1_row(nb, 0));
                    cs.vvFlat[a][b][0] = sparse_lookup_flat_slot(A, q1q1_row(na, 0), q1q1_row(nb, 0));
                    cs.vv[a][b][1] = sparse_lookup_slot(A, q1q1_row(na, 1), q1q1_row(nb, 1));
                    cs.vvFlat[a][b][1] = sparse_lookup_flat_slot(A, q1q1_row(na, 1), q1q1_row(nb, 1));
                    cs.vv[a][b][2] = sparse_lookup_slot(A, q1q1_row(na, 2), q1q1_row(nb, 2));
                    cs.vvFlat[a][b][2] = sparse_lookup_flat_slot(A, q1q1_row(na, 2), q1q1_row(nb, 2));
                }

                if (!isBnd[na] && nb != pressurePinNode) {
                    cs.vp[a][b][0] = sparse_lookup_slot(A, q1q1_row(na, 0), q1q1_row(nb, 3));
                    cs.vpFlat[a][b][0] = sparse_lookup_flat_slot(A, q1q1_row(na, 0), q1q1_row(nb, 3));
                    cs.vp[a][b][1] = sparse_lookup_slot(A, q1q1_row(na, 1), q1q1_row(nb, 3));
                    cs.vpFlat[a][b][1] = sparse_lookup_flat_slot(A, q1q1_row(na, 1), q1q1_row(nb, 3));
                    cs.vp[a][b][2] = sparse_lookup_slot(A, q1q1_row(na, 2), q1q1_row(nb, 3));
                    cs.vpFlat[a][b][2] = sparse_lookup_flat_slot(A, q1q1_row(na, 2), q1q1_row(nb, 3));
                }

                if (na != pressurePinNode && !isBnd[nb]) {
                    cs.pv[a][b][0] = sparse_lookup_slot(A, q1q1_row(na, 3), q1q1_row(nb, 0));
                    cs.pvFlat[a][b][0] = sparse_lookup_flat_slot(A, q1q1_row(na, 3), q1q1_row(nb, 0));
                    cs.pv[a][b][1] = sparse_lookup_slot(A, q1q1_row(na, 3), q1q1_row(nb, 1));
                    cs.pvFlat[a][b][1] = sparse_lookup_flat_slot(A, q1q1_row(na, 3), q1q1_row(nb, 1));
                    cs.pv[a][b][2] = sparse_lookup_slot(A, q1q1_row(na, 3), q1q1_row(nb, 2));
                    cs.pvFlat[a][b][2] = sparse_lookup_flat_slot(A, q1q1_row(na, 3), q1q1_row(nb, 2));
                }

                if (na != pressurePinNode && nb != pressurePinNode) {
                    cs.pp[a][b] = sparse_lookup_slot(A, q1q1_row(na, 3), q1q1_row(nb, 3));
                    cs.ppFlat[a][b] = sparse_lookup_flat_slot(A, q1q1_row(na, 3), q1q1_row(nb, 3));
                }
            }
        }
    }

    cachedNCells = (int)mesh.cells.size();
    cachedNNodes = dm.nDofs;
    cachedPressurePinNode = pressurePinNode;
    cachedPatternPtr = patternPtr;

    return cached;
}


static inline AssembledSystem make_empty_q1q1_nse_picard_pspg_system(
    const PolyMesh& mesh,
    const LinearCgDofMap& dm,
    const std::vector<Real>& beta,
    const Q1Q1NsePicardOptions& opt,
    Q1Q1AlgInfo& info
) {
    if (dm.resolvedSpace != "cg_hex_q1") {
        throw std::runtime_error("Q1/Q1 NSE Picard currently requires -space cg_hex_q1.");
    }
    if (dm.nDofs != (int)mesh.points.size()) {
        throw std::runtime_error("Q1/Q1 NSE Picard expects scalar Q1 DOF = OpenFOAM point index.");
    }

    const int nNodes = dm.nDofs;
    const int nRows = 4 * nNodes;

    if ((int)beta.size() != nRows) {
        throw std::runtime_error("Q1/Q1 NSE Picard beta vector must have 4*nnode entries.");
    }

    if (!(opt.nu > 0.0)) throw std::runtime_error("Q1/Q1 NSE Picard nu must be positive.");
    if (!(opt.dt > 0.0)) throw std::runtime_error("Q1/Q1 NSE Picard dt must be positive.");

    info.nNodes = nNodes;
    info.nRows = nRows;
    info.nBoundaryNodes = dm.nBoundaryDofs;
    info.nStrongVelocityRows = 3 * dm.nBoundaryDofs;
    info.pressurePinNode = opt.pressurePinNode;
    info.nu = opt.nu;
    info.hmin = q1q1_global_hmin(mesh);
    info.tau = q1q1_compute_tau_structured_match(mesh, beta, opt, info.hmin);

    if (info.pressurePinNode < 0 || info.pressurePinNode >= nNodes) {
        throw std::runtime_error("Bad Q1/Q1 pressure pin node.");
    }

    if (!memoirs_env_bool("MEMOIRS_Q1Q1_FIXED_PATTERN", true)) {
        throw std::runtime_error("CUDA Q1/Q1 assembly requires MEMOIRS_Q1Q1_FIXED_PATTERN=1.");
    }

    AssembledSystem sys;
    q1q1_make_nse_picard_fixed_pattern(mesh, dm, info.pressurePinNode, sys.A);
    sys.b.assign(nRows, Real(0));
    sys.nDirichlet = info.nStrongVelocityRows + 1;

    // Force slot cache construction. CUDA uses the flat slot IDs.
    (void)q1q1_get_nse_cell_slots_cached(mesh, dm, info.pressurePinNode, sys.A);

    info.nnz = sparse_nnz(sys.A);
    return sys;
}


static inline void q1q1_gauss_rule_1d(
    int qOrder,
    std::vector<double>& qp,
    std::vector<double>& qw
) {
    qp.clear();
    qw.clear();

    if (qOrder == 1) {
        qp = {0.0};
        qw = {2.0};
    } else if (qOrder == 2) {
        const double a = 0.57735026918962576451;
        qp = {-a, a};
        qw = {1.0, 1.0};
    } else if (qOrder == 3) {
        const double a = 0.77459666924148337704;
        qp = {-a, 0.0, a};
        qw = {0.55555555555555555556,
              0.88888888888888888889,
              0.55555555555555555556};
    } else if (qOrder == 4) {
        qp = {-0.86113631159405257522,
              -0.33998104358485626480,
               0.33998104358485626480,
               0.86113631159405257522};
        qw = {0.34785484513745385737,
              0.65214515486254614263,
              0.65214515486254614263,
              0.34785484513745385737};
    } else {
        throw std::runtime_error("MEMOIRS_Q1Q1_ASSEMBLY_QUAD must be 1, 2, 3, or 4.");
    }
}

static inline int q1q1_assembly_quad_order_from_env() {
    int qOrder = 2; // exact for Q1/Q1 PSPG-only affine hex matrix terms
    if (const char* e = std::getenv("MEMOIRS_Q1Q1_ASSEMBLY_QUAD")) {
        qOrder = std::atoi(e);
    }
    if (qOrder < 1 || qOrder > 4) {
        throw std::runtime_error("Bad MEMOIRS_Q1Q1_ASSEMBLY_QUAD. Use 1, 2, 3, or 4.");
    }
    return qOrder;
}


static inline AssembledSystem assemble_q1q1_nse_picard_pspg(
    const PolyMesh& mesh,
    const LinearCgDofMap& dm,
    const std::vector<Real>& beta,
    const Q1Q1NsePicardOptions& opt,
    Q1Q1AlgInfo& info
) {
    if (dm.resolvedSpace != "cg_hex_q1") {
        throw std::runtime_error("Q1/Q1 NSE Picard currently requires -space cg_hex_q1.");
    }
    if (dm.nDofs != (int)mesh.points.size()) {
        throw std::runtime_error("Q1/Q1 NSE Picard expects scalar Q1 DOF = OpenFOAM point index.");
    }

    const int nNodes = dm.nDofs;
    const int nRows = 4 * nNodes;

    if ((int)beta.size() != nRows) {
        throw std::runtime_error("Q1/Q1 NSE Picard beta vector must have 4*nnode entries.");
    }

    if (!(opt.nu > 0.0)) throw std::runtime_error("Q1/Q1 NSE Picard nu must be positive.");
    if (!(opt.dt > 0.0)) throw std::runtime_error("Q1/Q1 NSE Picard dt must be positive.");

    info.nNodes = nNodes;
    info.nRows = nRows;
    info.nBoundaryNodes = dm.nBoundaryDofs;
    info.nStrongVelocityRows = 3 * dm.nBoundaryDofs;
    info.pressurePinNode = opt.pressurePinNode;
    info.nu = opt.nu;
    info.hmin = q1q1_global_hmin(mesh);
    info.tau = q1q1_compute_tau_structured_match(mesh, beta, opt, info.hmin);

    if (info.pressurePinNode < 0 || info.pressurePinNode >= nNodes) {
        throw std::runtime_error("Bad Q1/Q1 pressure pin node.");
    }

    const double massCoeff = 1.0 / opt.dt;
    const double tauSupg = (opt.supg != 0) ? info.tau * opt.supgTauScale : 0.0;
    const double gradDivCoeff = 0.0; // grad-div/LSIC disabled for this performance branch;

    std::vector<unsigned char> isBnd(nNodes, 0);
    for (int node : dm.boundaryDofs) {
        if (node >= 0 && node < nNodes) isBnd[node] = 1;
    }

    const std::vector<Real> oldExactFe = q1q1_make_exact_node_major_vector(mesh);

    AssembledSystem sys;
    if (memoirs_env_bool("MEMOIRS_Q1Q1_FIXED_PATTERN", false)) {
        q1q1_make_nse_picard_fixed_pattern(mesh, dm, info.pressurePinNode, sys.A);
    } else {
        sys.A.rows.resize(nRows);
    }
    sys.b.assign(nRows, Real(0));
    sys.nDirichlet = info.nStrongVelocityRows + 1;

    const bool useSlotAssembly =
        sys.A.fixedPattern && memoirs_env_bool("MEMOIRS_Q1Q1_SLOT_ASSEMBLY", true);

    const bool useFlatSlotAssembly =
        useSlotAssembly && memoirs_env_bool("MEMOIRS_Q1Q1_FLAT_SLOT_ASSEMBLY", true);

    const std::vector<Q1Q1NseCellSlots>* cellSlotsPtr = nullptr;
    if (useSlotAssembly) {
        cellSlotsPtr = &q1q1_get_nse_cell_slots_cached(
            mesh, dm, info.pressurePinNode, sys.A
        );
    }

    auto addA = [&](int row, int col, int slot, int flatSlot, Real val) {
        if (useFlatSlotAssembly) {
            sparse_add_flat_slot(sys.A, flatSlot, val);
        } else if (useSlotAssembly) {
            sparse_add_slot(sys.A, row, slot, val);
        } else {
            sparse_add(sys.A, row, col, val);
        }
    };

    const int qOrder = q1q1_assembly_quad_order_from_env();
    std::vector<double> qp;
    std::vector<double> qw;
    q1q1_gauss_rule_1d(qOrder, qp, qw);

    for (int cellI = 0; cellI < (int)mesh.cells.size(); ++cellI) {
        const auto& c = mesh.cells[cellI];
        const auto hv = ordered_hex_vertices_axis_aligned(mesh, c);
        const Q1Q1NseCellSlots* cs = useSlotAssembly ? &((*cellSlotsPtr)[cellI]) : nullptr;

        for (int iq = 0; iq < qOrder; ++iq) {
            for (int jq = 0; jq < qOrder; ++jq) {
                for (int kq = 0; kq < qOrder; ++kq) {
                    const double xi = qp[iq];
                    const double eta = qp[jq];
                    const double zeta = qp[kq];
                    const double w = qw[iq] * qw[jq] * qw[kq];

                    std::array<double,8> N;
                    std::array<Vec3,8> dNref;
                    hex_q1_basis(xi, eta, zeta, N, dNref);

                    Vec3 xq;
                    Mat3 J;

                    for (int a = 0; a < 8; ++a) {
                        const Vec3& X = mesh.points[hv[a]];

                        xq.x += N[a] * X.x;
                        xq.y += N[a] * X.y;
                        xq.z += N[a] * X.z;

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

                    const double dV = std::abs(det3(J)) * w;
                    const Mat3 invJ = inv3(J);

                    std::array<Vec3,8> grad;
                    for (int a = 0; a < 8; ++a) {
                        grad[a] = invJT_mul(invJ, dNref[a]);
                    }

                    // Structured-match beta and old velocity interpolation:
                    // boundary velocity trial DOFs are excluded from element residual.
                    double bx = 0.0, by = 0.0, bz = 0.0;
                    double oldHx = 0.0, oldHy = 0.0, oldHz = 0.0;

                    for (int b = 0; b < 8; ++b) {
                        const int nb = hv[b];

                        bx += N[b] * double(beta[q1q1_row(nb, 0)]);
                        by += N[b] * double(beta[q1q1_row(nb, 1)]);
                        bz += N[b] * double(beta[q1q1_row(nb, 2)]);

                        if (isBnd[nb]) continue;

                        oldHx += N[b] * double(oldExactFe[q1q1_row(nb, 0)]);
                        oldHy += N[b] * double(oldExactFe[q1q1_row(nb, 1)]);
                        oldHz += N[b] * double(oldExactFe[q1q1_row(nb, 2)]);
                    }

                    double fx, fy, fz;
                    {
                        const Q1Q1ForceAtPoint f = q1q1_nse_force_at(xq, opt.nu, opt.advScale);
                        fx = f.fx + massCoeff * oldHx;
                        fy = f.fy + massCoeff * oldHy;
                        fz = f.fz + massCoeff * oldHz;
                    }

                    // RHS: exactly same structure as kernel_assemble_rhs_q1q1_nse_elements.
                    for (int a = 0; a < 8; ++a) {
                        const int na = hv[a];

                        const double streamNa =
                            opt.advScale * (bx * grad[a].x + by * grad[a].y + bz * grad[a].z);
                        const double test = N[a] + tauSupg * streamNa;

                        if (!isBnd[na]) {
                            sys.b[q1q1_row(na, 0)] += Real(test * fx * dV);
                            sys.b[q1q1_row(na, 1)] += Real(test * fy * dV);
                            sys.b[q1q1_row(na, 2)] += Real(test * fz * dV);
                        }

                        if (na != info.pressurePinNode) {
                            const double rp =
                                info.tau * (grad[a].x * fx + grad[a].y * fy + grad[a].z * fz);
                            sys.b[q1q1_row(na, 3)] += Real(rp * dV);
                        }
                    }

                    // Matrix: residual-linearization form matching kernel_apply_q1q1_nse_elements.
                    for (int a = 0; a < 8; ++a) {
                        const int na = hv[a];

                        const double streamNa =
                            opt.advScale * (bx * grad[a].x + by * grad[a].y + bz * grad[a].z);

                        for (int b = 0; b < 8; ++b) {
                            const int nb = hv[b];

                            const double stiff = dot3(grad[a], grad[b]) * dV;
                            const double betaGradNb =
                                opt.advScale * (bx * grad[b].x +
                                                by * grad[b].y +
                                                bz * grad[b].z);

                            const double residualVelTrial =
                                massCoeff * N[b] + betaGradNb;

                            // Velocity momentum rows.
                            if (!isBnd[na]) {
                                if (!isBnd[nb]) {
                                    const double baseVel =
                                        N[a] * residualVelTrial * dV
                                      + opt.nu * stiff
                                      + tauSupg * streamNa * residualVelTrial * dV;

                                    addA(q1q1_row(na, 0), q1q1_row(nb, 0), cs ? cs->vv[a][b][0] : -1, cs ? cs->vvFlat[a][b][0] : -1, Real(baseVel));
                                    addA(q1q1_row(na, 1), q1q1_row(nb, 1), cs ? cs->vv[a][b][1] : -1, cs ? cs->vvFlat[a][b][1] : -1, Real(baseVel));
                                    addA(q1q1_row(na, 2), q1q1_row(nb, 2), cs ? cs->vv[a][b][2] : -1, cs ? cs->vvFlat[a][b][2] : -1, Real(baseVel));

                                    // Grad-div / LSIC:
                                    //   gamma (div v, div u)
                                    // For vector basis v = N_a e_i and u = N_b e_j:
                                    //   div(v) = dN_a/dx_i
                                    //   div(u) = dN_b/dx_j
                                    if (false) { // grad-div/LSIC disabled for this performance branch
                                        const double gd = gradDivCoeff * dV;

                                        sparse_add(sys.A, q1q1_row(na, 0), q1q1_row(nb, 0),
                                                   Real(gd * grad[a].x * grad[b].x));
                                        sparse_add(sys.A, q1q1_row(na, 0), q1q1_row(nb, 1),
                                                   Real(gd * grad[a].x * grad[b].y));
                                        sparse_add(sys.A, q1q1_row(na, 0), q1q1_row(nb, 2),
                                                   Real(gd * grad[a].x * grad[b].z));

                                        sparse_add(sys.A, q1q1_row(na, 1), q1q1_row(nb, 0),
                                                   Real(gd * grad[a].y * grad[b].x));
                                        sparse_add(sys.A, q1q1_row(na, 1), q1q1_row(nb, 1),
                                                   Real(gd * grad[a].y * grad[b].y));
                                        sparse_add(sys.A, q1q1_row(na, 1), q1q1_row(nb, 2),
                                                   Real(gd * grad[a].y * grad[b].z));

                                        sparse_add(sys.A, q1q1_row(na, 2), q1q1_row(nb, 0),
                                                   Real(gd * grad[a].z * grad[b].x));
                                        sparse_add(sys.A, q1q1_row(na, 2), q1q1_row(nb, 1),
                                                   Real(gd * grad[a].z * grad[b].y));
                                        sparse_add(sys.A, q1q1_row(na, 2), q1q1_row(nb, 2),
                                                   Real(gd * grad[a].z * grad[b].z));
                                    }
                                }

                                if (nb != info.pressurePinNode) {
                                    // -p grad(v) + SUPG * stream(v) * grad(p)
                                    addA(q1q1_row(na, 0), q1q1_row(nb, 3), cs ? cs->vp[a][b][0] : -1, cs ? cs->vpFlat[a][b][0] : -1,
                                         Real((-N[b] * grad[a].x
                                               + tauSupg * streamNa * grad[b].x) * dV));
                                    addA(q1q1_row(na, 1), q1q1_row(nb, 3), cs ? cs->vp[a][b][1] : -1, cs ? cs->vpFlat[a][b][1] : -1,
                                         Real((-N[b] * grad[a].y
                                               + tauSupg * streamNa * grad[b].y) * dV));
                                    addA(q1q1_row(na, 2), q1q1_row(nb, 3), cs ? cs->vp[a][b][2] : -1, cs ? cs->vpFlat[a][b][2] : -1,
                                         Real((-N[b] * grad[a].z
                                               + tauSupg * streamNa * grad[b].z) * dV));
                                }
                            }

                            // Pressure row: continuity + PSPG residual.
                            if (na != info.pressurePinNode) {
                                if (!isBnd[nb]) {
                                    addA(q1q1_row(na, 3), q1q1_row(nb, 0), cs ? cs->pv[a][b][0] : -1, cs ? cs->pvFlat[a][b][0] : -1,
                                         Real((N[a] * grad[b].x
                                               + info.tau * grad[a].x * residualVelTrial) * dV));
                                    addA(q1q1_row(na, 3), q1q1_row(nb, 1), cs ? cs->pv[a][b][1] : -1, cs ? cs->pvFlat[a][b][1] : -1,
                                         Real((N[a] * grad[b].y
                                               + info.tau * grad[a].y * residualVelTrial) * dV));
                                    addA(q1q1_row(na, 3), q1q1_row(nb, 2), cs ? cs->pv[a][b][2] : -1, cs ? cs->pvFlat[a][b][2] : -1,
                                         Real((N[a] * grad[b].z
                                               + info.tau * grad[a].z * residualVelTrial) * dV));
                                }

                                if (nb != info.pressurePinNode) {
                                    addA(q1q1_row(na, 3), q1q1_row(nb, 3), cs ? cs->pp[a][b] : -1, cs ? cs->ppFlat[a][b] : -1,
                                         Real(info.tau * stiff));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Strong boundary velocity rows.
    for (int node : dm.boundaryDofs) {
        const Q1Q1ExactAtNode ex = q1q1_exact_mms_at(mesh.points[node]);

        q1q1_identity_row(sys.A, q1q1_row(node, 0));
        q1q1_identity_row(sys.A, q1q1_row(node, 1));
        q1q1_identity_row(sys.A, q1q1_row(node, 2));

        sys.b[q1q1_row(node, 0)] = Real(ex.ux);
        sys.b[q1q1_row(node, 1)] = Real(ex.uy);
        sys.b[q1q1_row(node, 2)] = Real(ex.uz);
    }

    // Pressure pin.
    const Q1Q1ExactAtNode pex = q1q1_exact_mms_at(mesh.points[info.pressurePinNode]);
    q1q1_identity_row(sys.A, q1q1_row(info.pressurePinNode, 3));
    sys.b[q1q1_row(info.pressurePinNode, 3)] = Real(pex.p);

    info.nnz = sparse_nnz(sys.A);
    return sys;
}


static inline Vec3 q1q1_cavity_lid_velocity_at_node(
    const PolyMesh& mesh,
    int node,
    double lidUx,
    double lidUy,
    double lidUz
) {
    struct BoundsCache {
        const Vec3* pointsPtr = nullptr;
        int nPoints = 0;
        double xmin = 0.0, ymin = 0.0, zmin = 0.0;
        double xmax = 0.0, ymax = 0.0, zmax = 0.0;
        double tol = 0.0;
    };

    static BoundsCache cache;

    const Vec3* ptr = mesh.points.empty() ? nullptr : mesh.points.data();
    const int npts = (int)mesh.points.size();

    if (cache.pointsPtr != ptr || cache.nPoints != npts) {
        cache.pointsPtr = ptr;
        cache.nPoints = npts;

        cache.xmin =  std::numeric_limits<double>::infinity();
        cache.ymin =  std::numeric_limits<double>::infinity();
        cache.zmin =  std::numeric_limits<double>::infinity();
        cache.xmax = -std::numeric_limits<double>::infinity();
        cache.ymax = -std::numeric_limits<double>::infinity();
        cache.zmax = -std::numeric_limits<double>::infinity();

        for (const Vec3& p : mesh.points) {
            cache.xmin = std::min(cache.xmin, p.x);
            cache.ymin = std::min(cache.ymin, p.y);
            cache.zmin = std::min(cache.zmin, p.z);
            cache.xmax = std::max(cache.xmax, p.x);
            cache.ymax = std::max(cache.ymax, p.y);
            cache.zmax = std::max(cache.zmax, p.z);
        }

        const double L = std::max({
            1.0,
            cache.xmax - cache.xmin,
            cache.ymax - cache.ymin,
            cache.zmax - cache.zmin
        });
        cache.tol = std::max(1.0e-12, 1.0e-10 * L);
    }

    const Vec3& X = mesh.points[node];

    const bool onTop = std::abs(X.z - cache.zmax) <= cache.tol;

    // Deliberate driven-cavity convention:
    // the top rim/corners are wall no-slip, not moving-lid.
    // This is equivalent to applying lid BC first and then wall BCs
    // overriding the lid at the rim.
    const bool onSideWall =
        std::abs(X.x - cache.xmin) <= cache.tol ||
        std::abs(X.x - cache.xmax) <= cache.tol ||
        std::abs(X.y - cache.ymin) <= cache.tol ||
        std::abs(X.y - cache.ymax) <= cache.tol;

    if (onTop && !onSideWall) {
        return {lidUx, lidUy, lidUz};
    }

    return {0.0, 0.0, 0.0};
}

static inline void q1q1_apply_cavity_velocity_bc_to_state(
    const PolyMesh& mesh,
    const LinearCgDofMap& dm,
    double lidUx,
    double lidUy,
    double lidUz,
    std::vector<Real>& x
) {
    if ((int)x.size() != 4 * dm.nDofs) x.assign(4 * dm.nDofs, Real(0));

    for (int node : dm.boundaryDofs) {
        const Vec3 g = q1q1_cavity_lid_velocity_at_node(mesh, node, lidUx, lidUy, lidUz);
        x[q1q1_row(node,0)] = Real(g.x);
        x[q1q1_row(node,1)] = Real(g.y);
        x[q1q1_row(node,2)] = Real(g.z);
    }
}

static inline void q1q1_overwrite_cavity_bdf1_rhs(
    const PolyMesh& mesh,
    const LinearCgDofMap& dm,
    const std::vector<Real>& beta,
    const std::vector<Real>& oldState,
    const Q1Q1NsePicardOptions& opt,
    const Q1Q1AlgInfo& info,
    double lidUx,
    double lidUy,
    double lidUz,
    AssembledSystem& sys
) {
    (void)mesh;
    (void)dm;
    (void)beta;
    (void)oldState;
    (void)opt;
    (void)info;
    (void)lidUx;
    (void)lidUy;
    (void)lidUz;
    (void)sys;

    throw std::runtime_error(
        "Host cavity RHS override is disabled because it destroys performance. "
        "Driven cavity must use MEMOIRS_Q1Q1_CUDA_ASSEMBLY=1 and "
        "q1q1_cuda_assemble_cavity_bdf1()."
    );
}

static inline void q1q1_print_nse_picard_info(
    const Q1Q1NsePicardOptions& opt,
    const Q1Q1AlgInfo& info,
    int picardIter
) {
    std::cout << "--------------- q1q1 NSE Picard matrix --------------\n";
    std::cout << "q1q1Operator              = transient_picard_nse_pspg_supg_graddiv\n";
    std::cout << "q1q1PicardIter            = " << picardIter << "\n";
    std::cout << "q1q1Rows                  = " << info.nRows << "\n";
    std::cout << "q1q1Nodes                 = " << info.nNodes << "\n";
    std::cout << "q1q1Nnz                   = " << info.nnz << "\n";
    std::cout << "q1q1FixedPattern          = " << (memoirs_env_bool("MEMOIRS_Q1Q1_FIXED_PATTERN", false) ? 1 : 0) << "\n";
    std::cout << "q1q1SlotAssembly          = " << (memoirs_env_bool("MEMOIRS_Q1Q1_SLOT_ASSEMBLY", true) ? 1 : 0) << "\n";
    std::cout << "q1q1FlatSlotAssembly      = " << (memoirs_env_bool("MEMOIRS_Q1Q1_FLAT_SLOT_ASSEMBLY", true) ? 1 : 0) << "\n";
    std::cout << "q1q1Nu                    = " << std::setprecision(16) << opt.nu << "\n";
    std::cout << "q1q1Dt                    = " << std::setprecision(16) << opt.dt << "\n";
    std::cout << "q1q1InvDt                 = " << std::setprecision(16) << (1.0 / opt.dt) << "\n";
    std::cout << "q1q1TauMode               = " << opt.tauMode << "\n";
    std::cout << "q1q1TauCt                 = " << std::setprecision(16) << opt.tauCt << "\n";
    std::cout << "q1q1TauAdvScale           = " << std::setprecision(16) << opt.tauAdvScale << "\n";
    std::cout << "q1q1TauOverDt             = " << std::setprecision(16) << (info.tau / opt.dt) << "\n";
    std::cout << "q1q1TimeTerm              = momentum_mass_over_dt_plus_pspg_gradq_mass_over_dt\n";
    std::cout << "q1q1OldTimeRhs            = Q1_interpolated_exact_old_velocity\n";
    std::cout << "q1q1AdvScale              = " << std::setprecision(16) << opt.advScale << "\n";
    std::cout << "q1q1Tau                   = " << std::setprecision(16) << info.tau << "\n";
    std::cout << "q1q1AssemblyQuad          = " << q1q1_assembly_quad_order_from_env()
              << "x" << q1q1_assembly_quad_order_from_env()
              << "x" << q1q1_assembly_quad_order_from_env() << "\n";
    std::cout << "q1q1Supg                  = " << opt.supg << "\n";
    std::cout << "q1q1SupgTauScale          = " << std::setprecision(16) << opt.supgTauScale << "\n";
    std::cout << "q1q1GradDiv               = " << opt.gradDiv << "\n";
    std::cout << "q1q1GradDivScale          = " << std::setprecision(16) << opt.gradDivScale << "\n";
    std::cout << "q1q1GradDivCoeff          = " << std::setprecision(16)
              << ((opt.gradDiv != 0)
                    ? ((opt.gradDivCoeff > 0.0) ? opt.gradDivCoeff
                                                : opt.gradDivScale / std::max(opt.nu, 1.0e-300))
                    : 0.0)
              << "\n";
    std::cout << "-----------------------------------------------------\n";
}
