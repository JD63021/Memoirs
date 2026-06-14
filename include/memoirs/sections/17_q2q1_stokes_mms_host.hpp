#pragma once

// ============================================================================
// SECTION 17: Q2/Q1 host Stokes+PSPG MMS matrix scaffold
// ============================================================================
//
// This is the first real Q2/Q1 operator assembly, but still a controlled
// scaffold:
//
//   - structured affine unit-cube hexes only;
//   - Q2 velocity, Q1 pressure;
//   - steady Stokes matrix:
//        velocity rows: nu (grad v, grad u) - (div v, p)
//        pressure rows: (q, div u)
//                     + tau (grad q, -nu lap(u) + grad p)
//   - Q2 strong viscous PSPG term is included through lap(Q2);
//   - RHS mode in this phase is algebraic: b = A * x_exact.
//     That verifies the matrix/slots/solver before continuous trigonometric RHS.
// ============================================================================

struct Q2Q1MmsAtPoint {
    double ux = 0.0;
    double uy = 0.0;
    double uz = 0.0;
    double p  = 0.0;
};

struct Q2Q1StokesMmsOptions {
    double nu = 1.0;
    double tau = -1.0;
    double tauScale = 1.0;
    double tauC = 4.0;
    int pressurePinPNode = 0;
    std::string rhsMode = "algebraic"; // algebraic or continuous
};

struct Q2Q1StokesAssemblyReport {
    long long rows = 0;
    long long nnz = 0;
    long long activeCellSlots = 0;
    double nu = 1.0;
    double tau = 0.0;
    double h = 0.0;
    std::string rhsMode = "algebraic";
    double assemblySeconds = -1.0;
    double algebraicRhsSeconds = -1.0;
    double continuousRhsSeconds = -1.0;
    double exactResidualL2 = -1.0;
    double exactResidualInf = -1.0;
    double exactVectorL2 = -1.0;
};

struct Q2Q1ExactCompareReport {
    double absL2 = -1.0;
    double relL2 = -1.0;
    double maxAbs = -1.0;
    double uAbsL2 = -1.0;
    double pAbsL2 = -1.0;
};

static inline Q2Q1MmsAtPoint q2q1_exact_mms_at(const Vec3& X) {
    const double pi = kPi;

    const double sx = std::sin(pi * X.x);
    const double sy = std::sin(pi * X.y);
    const double sz = std::sin(pi * X.z);

    const double cx = std::cos(pi * X.x);
    const double cy = std::cos(pi * X.y);

    Q2Q1MmsAtPoint e;

    // Same divergence-free sine MMS as the Q1/Q1 branch.
    e.ux =  2.0 * pi * sx * sx * sy * cy * sz * sz;
    e.uy = -2.0 * pi * sx * cx * sy * sy * sz * sz;
    e.uz =  0.0;

    // p(0,0,0)=0, convenient pressure pin.
    e.p = cx * sy * sz;

    return e;
}

static inline Vec3 q2q1_stokes_mms_force_at(const Vec3& X, double nu) {
    const double pi = kPi;

    const double sx = std::sin(pi * X.x);
    const double sy = std::sin(pi * X.y);
    const double sz = std::sin(pi * X.z);

    const double cx = std::cos(pi * X.x);
    const double cy = std::cos(pi * X.y);
    const double cz = std::cos(pi * X.z);

    const double c2x = std::cos(2.0 * pi * X.x);
    const double c2y = std::cos(2.0 * pi * X.y);
    const double c2z = std::cos(2.0 * pi * X.z);

    const double A = sx * sx;
    const double B = sy * cy;
    const double C = sz * sz;
    const double D = sx * cx;
    const double E = sy * sy;

    // ux = 2*pi*A*B*C
    // uy = -2*pi*D*E*C
    const double lapUx =
        4.0 * pi * pi * pi * B * (c2x * C - 2.0 * A * C + A * c2z);

    const double lapUy =
        4.0 * pi * pi * pi * D * (2.0 * E * C - c2y * C - E * c2z);

    // p = cos(pi*x)*sin(pi*y)*sin(pi*z)
    const double dpdx = -pi * sx * sy * sz;
    const double dpdy =  pi * cx * cy * sz;
    const double dpdz =  pi * cx * sy * cz;

    Vec3 f;
    f.x = -nu * lapUx + dpdx;
    f.y = -nu * lapUy + dpdy;
    f.z = dpdz;
    return f;
}

static inline Vec3 q2q1_velocity_node_coord(const Q2Q1StructuredGrid& g, int I, int J, int K) {
    return {
        g.xmin + (g.xmax - g.xmin) * ((double)I / (double)(g.nv1 - 1)),
        g.ymin + (g.ymax - g.ymin) * ((double)J / (double)(g.nv1 - 1)),
        g.zmin + (g.zmax - g.zmin) * ((double)K / (double)(g.nv1 - 1))
    };
}

static inline Vec3 q2q1_pressure_node_coord(const Q2Q1StructuredGrid& g, int i, int j, int k) {
    return {
        g.xmin + (g.xmax - g.xmin) * ((double)i / (double)(g.np1 - 1)),
        g.ymin + (g.ymax - g.ymin) * ((double)j / (double)(g.np1 - 1)),
        g.zmin + (g.zmax - g.zmin) * ((double)k / (double)(g.np1 - 1))
    };
}

static inline std::vector<Real> q2q1_make_exact_vector(const Q2Q1StructuredGrid& g) {
    std::vector<Real> x((std::size_t)g.nRows, Real(0));

    for (int K = 0; K < g.nv1; ++K) {
        for (int J = 0; J < g.nv1; ++J) {
            for (int I = 0; I < g.nv1; ++I) {
                const long long v = q2q1_vnode_id(g, I, J, K);
                const Vec3 X = q2q1_velocity_node_coord(g, I, J, K);
                const Q2Q1MmsAtPoint e = q2q1_exact_mms_at(X);

                x[(std::size_t)q2q1_ux_row(g, v)] = Real(e.ux);
                x[(std::size_t)q2q1_uy_row(g, v)] = Real(e.uy);
                x[(std::size_t)q2q1_uz_row(g, v)] = Real(e.uz);
            }
        }
    }

    for (int k = 0; k < g.np1; ++k) {
        for (int j = 0; j < g.np1; ++j) {
            for (int i = 0; i < g.np1; ++i) {
                const long long pnode = q2q1_pnode_id(g, i, j, k);
                const Vec3 X = q2q1_pressure_node_coord(g, i, j, k);
                const Q2Q1MmsAtPoint e = q2q1_exact_mms_at(X);

                x[(std::size_t)q2q1_p_row(g, pnode)] = Real(e.p);
            }
        }
    }

    return x;
}

static inline double q2q1_default_stokes_tau(
    const Q2Q1StructuredGrid& g,
    const Q2Q1StokesMmsOptions& opt
) {
    if (opt.tau > 0.0) return opt.tau;

    const double h = std::min(g.hx, std::min(g.hy, g.hz));
    return opt.tauScale * h * h / (opt.tauC * opt.nu);
}

static inline void q2q1_apply_sparse_fixed(
    const SparseRows& A,
    const std::vector<Real>& x,
    std::vector<Real>& y
) {
    const int nRows = sparse_nrows(A);
    if ((int)x.size() != nRows) throw std::runtime_error("q2q1_apply_sparse_fixed x size mismatch.");

    y.assign((std::size_t)nRows, Real(0));

    for (int r = 0; r < nRows; ++r) {
        double s = 0.0;
        const int b = sparse_row_start(A, r);
        const int e = sparse_row_end(A, r);
        const auto& cols = sparse_cols_row(A, r);

        for (int kk = b; kk < e; ++kk) {
            const int local = kk - b;
            const int c = cols[(std::size_t)local];
            s += (double)A.flatVals[(std::size_t)kk] * (double)x[(std::size_t)c];
        }

        y[(std::size_t)r] = Real(s);
    }
}

static inline void q2q1_residual_norm_fixed(
    const SparseRows& A,
    const std::vector<Real>& x,
    const std::vector<Real>& b,
    double& l2,
    double& infNorm
) {
    std::vector<Real> Ax;
    q2q1_apply_sparse_fixed(A, x, Ax);

    l2 = 0.0;
    infNorm = 0.0;

    for (std::size_t i = 0; i < Ax.size(); ++i) {
        const double r = (double)Ax[i] - (double)b[i];
        l2 += r * r;
        infNorm = std::max(infNorm, std::abs(r));
    }

    l2 = std::sqrt(l2);
}

static inline Q2Q1ExactCompareReport q2q1_compare_solution_to_exact(
    const Q2Q1StructuredGrid& g,
    const std::vector<Real>& x
) {
    Q2Q1ExactCompareReport rep;
    if ((long long)x.size() != g.nRows) return rep;

    const std::vector<Real> xe = q2q1_make_exact_vector(g);

    double e2 = 0.0;
    double n2 = 0.0;
    double u2 = 0.0;
    double p2 = 0.0;
    double maxe = 0.0;

    for (long long r = 0; r < g.nRows; ++r) {
        const double d = (double)x[(std::size_t)r] - (double)xe[(std::size_t)r];
        const double v = (double)xe[(std::size_t)r];

        e2 += d * d;
        n2 += v * v;
        maxe = std::max(maxe, std::abs(d));

        if (r < 3LL * g.nVelocityNodes) u2 += d * d;
        else p2 += d * d;
    }

    rep.absL2 = std::sqrt(e2);
    rep.relL2 = std::sqrt(e2 / std::max(n2, 1.0e-300));
    rep.maxAbs = maxe;
    rep.uAbsL2 = std::sqrt(u2);
    rep.pAbsL2 = std::sqrt(p2);
    return rep;
}

static inline bool q2q1_is_cavity_moving_lid_node(
    const Q2Q1StructuredGrid& g,
    int I,
    int J,
    int K
) {
    return (K == g.nv1 - 1) &&
           (I > 0 && I < g.nv1 - 1) &&
           (J > 0 && J < g.nv1 - 1);
}

static inline void q2q1_cavity_velocity_value(
    const Q2Q1StructuredGrid& g,
    int I,
    int J,
    int K,
    double& ux,
    double& uy,
    double& uz
) {
    ux = 0.0;
    uy = 0.0;
    uz = 0.0;

    if (q2q1_is_cavity_moving_lid_node(g, I, J, K)) {
        ux = 1.0;
    }
}

static inline void q2q1_set_cavity_strong_rhs(
    const Q2Q1StructuredGrid& g,
    const Q2Q1StokesMmsOptions& opt,
    AssembledSystem& sys
) {
    sys.b.assign((std::size_t)g.nRows, Real(0));

    for (int K = 0; K < g.nv1; ++K) {
        for (int J = 0; J < g.nv1; ++J) {
            for (int I = 0; I < g.nv1; ++I) {
                if (!q2q1_is_velocity_boundary_node(g, I, J, K)) continue;

                const long long v = q2q1_vnode_id(g, I, J, K);

                double ux, uy, uz;
                q2q1_cavity_velocity_value(g, I, J, K, ux, uy, uz);

                sys.b[(std::size_t)q2q1_ux_row(g, v)] = Real(ux);
                sys.b[(std::size_t)q2q1_uy_row(g, v)] = Real(uy);
                sys.b[(std::size_t)q2q1_uz_row(g, v)] = Real(uz);
            }
        }
    }

    // Pressure pin value.
    sys.b[(std::size_t)q2q1_p_row(g, opt.pressurePinPNode)] = Real(0);
}

struct Q2Q1CavitySolutionSummary {
    int movingLidNodes = 0;
    int noSlipNodes = 0;
    double boundaryMaxError = -1.0;
    double maxSpeed = -1.0;
    double maxAbsUx = -1.0;
    double maxAbsUy = -1.0;
    double maxAbsUz = -1.0;
    double pMin = 0.0;
    double pMax = 0.0;
    double pMean = 0.0;
};

static inline Q2Q1CavitySolutionSummary q2q1_compute_cavity_solution_summary(
    const Q2Q1StructuredGrid& g,
    const std::vector<Real>& x
) {
    Q2Q1CavitySolutionSummary r;
    if ((long long)x.size() != g.nRows) return r;

    double bcMax = 0.0;
    int lid = 0;
    int noslip = 0;

    double maxSpeed = 0.0;
    double maxUx = 0.0;
    double maxUy = 0.0;
    double maxUz = 0.0;

    for (int K = 0; K < g.nv1; ++K) {
        for (int J = 0; J < g.nv1; ++J) {
            for (int I = 0; I < g.nv1; ++I) {
                const long long v = q2q1_vnode_id(g, I, J, K);

                const double ux = (double)x[(std::size_t)q2q1_ux_row(g, v)];
                const double uy = (double)x[(std::size_t)q2q1_uy_row(g, v)];
                const double uz = (double)x[(std::size_t)q2q1_uz_row(g, v)];

                maxUx = std::max(maxUx, std::abs(ux));
                maxUy = std::max(maxUy, std::abs(uy));
                maxUz = std::max(maxUz, std::abs(uz));
                maxSpeed = std::max(maxSpeed, std::sqrt(ux * ux + uy * uy + uz * uz));

                if (q2q1_is_velocity_boundary_node(g, I, J, K)) {
                    double ex, ey, ez;
                    q2q1_cavity_velocity_value(g, I, J, K, ex, ey, ez);

                    const double e = std::sqrt(
                        (ux - ex) * (ux - ex) +
                        (uy - ey) * (uy - ey) +
                        (uz - ez) * (uz - ez)
                    );
                    bcMax = std::max(bcMax, e);

                    if (q2q1_is_cavity_moving_lid_node(g, I, J, K)) {
                        ++lid;
                    } else {
                        ++noslip;
                    }
                }
            }
        }
    }

    double pMin = 0.0;
    double pMax = 0.0;
    double pSum = 0.0;
    for (long long p = 0; p < g.nPressureNodes; ++p) {
        const double pv = (double)x[(std::size_t)q2q1_p_row(g, p)];
        if (p == 0) {
            pMin = pv;
            pMax = pv;
        } else {
            pMin = std::min(pMin, pv);
            pMax = std::max(pMax, pv);
        }
        pSum += pv;
    }

    r.movingLidNodes = lid;
    r.noSlipNodes = noslip;
    r.boundaryMaxError = bcMax;
    r.maxSpeed = maxSpeed;
    r.maxAbsUx = maxUx;
    r.maxAbsUy = maxUy;
    r.maxAbsUz = maxUz;
    r.pMin = pMin;
    r.pMax = pMax;
    r.pMean = pSum / std::max<long long>(g.nPressureNodes, 1);

    return r;
}

static inline void q2q1_print_cavity_solution_summary(const Q2Q1CavitySolutionSummary& r) {
    std::cout << "q2q1CavitySummaryDone     = 1\n";
    std::cout << "q2q1CavityMovingLidNodes  = " << r.movingLidNodes << "\n";
    std::cout << "q2q1CavityNoSlipNodes     = " << r.noSlipNodes << "\n";
    std::cout << "q2q1CavityBoundaryMaxError = " << std::setprecision(16) << r.boundaryMaxError << "\n";
    std::cout << "q2q1CavityMaxSpeed        = " << std::setprecision(16) << r.maxSpeed << "\n";
    std::cout << "q2q1CavityMaxAbsUx        = " << std::setprecision(16) << r.maxAbsUx << "\n";
    std::cout << "q2q1CavityMaxAbsUy        = " << std::setprecision(16) << r.maxAbsUy << "\n";
    std::cout << "q2q1CavityMaxAbsUz        = " << std::setprecision(16) << r.maxAbsUz << "\n";
    std::cout << "q2q1CavityPressureMin     = " << std::setprecision(16) << r.pMin << "\n";
    std::cout << "q2q1CavityPressureMax     = " << std::setprecision(16) << r.pMax << "\n";
    std::cout << "q2q1CavityPressureMean    = " << std::setprecision(16) << r.pMean << "\n";
}


static inline void q2q1_set_exact_strong_rhs(
    const Q2Q1StructuredGrid& g,
    const Q2Q1StokesMmsOptions& opt,
    AssembledSystem& sys
) {
    for (int K = 0; K < g.nv1; ++K) {
        for (int J = 0; J < g.nv1; ++J) {
            for (int I = 0; I < g.nv1; ++I) {
                if (!q2q1_is_velocity_boundary_node(g, I, J, K)) continue;

                const long long v = q2q1_vnode_id(g, I, J, K);
                const Vec3 X = q2q1_velocity_node_coord(g, I, J, K);
                const Q2Q1MmsAtPoint e = q2q1_exact_mms_at(X);

                sys.b[(std::size_t)q2q1_ux_row(g, v)] = Real(e.ux);
                sys.b[(std::size_t)q2q1_uy_row(g, v)] = Real(e.uy);
                sys.b[(std::size_t)q2q1_uz_row(g, v)] = Real(e.uz);
            }
        }
    }

    const int pp = opt.pressurePinPNode;
    int pk = pp / (g.np1 * g.np1);
    int rem = pp - pk * g.np1 * g.np1;
    int pj = rem / g.np1;
    int pi = rem - pj * g.np1;

    const Vec3 Xp = q2q1_pressure_node_coord(g, pi, pj, pk);
    const Q2Q1MmsAtPoint ep = q2q1_exact_mms_at(Xp);
    sys.b[(std::size_t)q2q1_p_row(g, pp)] = Real(ep.p);
}

static inline void q2q1_assemble_stokes_continuous_rhs_host(
    const Q2Q1StructuredGrid& g,
    const Q2Q1StokesMmsOptions& opt,
    double tau,
    AssembledSystem& sys
) {
    sys.b.assign((std::size_t)g.nRows, Real(0));

    std::array<double,4> qp;
    std::array<double,4> qw;
    q2q1_gauss4_1d(qp, qw);

    const double invHx = 2.0 / g.hx;
    const double invHy = 2.0 / g.hy;
    const double invHz = 2.0 / g.hz;
    const double detJ = g.hx * g.hy * g.hz / 8.0;

    auto addB = [&](long long row, double val) {
        if (row >= 0 && row < g.nRows) {
            sys.b[(std::size_t)row] += Real(val);
        }
    };

    for (int k = 0; k < g.N; ++k) {
        for (int j = 0; j < g.N; ++j) {
            for (int i = 0; i < g.N; ++i) {
                const auto vnodes = q2q1_cell_velocity_nodes(g, i, j, k);
                const auto pnodes = q2q1_cell_pressure_nodes(g, i, j, k);

                for (int kq = 0; kq < 4; ++kq) {
                    for (int jq = 0; jq < 4; ++jq) {
                        for (int iq = 0; iq < 4; ++iq) {
                            const double xi = qp[iq];
                            const double eta = qp[jq];
                            const double zeta = qp[kq];
                            const double dV = qw[iq] * qw[jq] * qw[kq] * detJ;

                            const Vec3 Xq{
                                g.xmin + ((double)i + 0.5 * (xi   + 1.0)) * g.hx,
                                g.ymin + ((double)j + 0.5 * (eta  + 1.0)) * g.hy,
                                g.zmin + ((double)k + 0.5 * (zeta + 1.0)) * g.hz
                            };

                            const Vec3 f = q2q1_stokes_mms_force_at(Xq, opt.nu);

                            std::array<double,27> Nv;
                            std::array<Vec3,27> dNvRef;
                            std::array<Vec3,27> ddNvRef;
                            q2q1_hex_q2_basis_ref(xi, eta, zeta, Nv, dNvRef, ddNvRef);

                            std::array<double,8> Np;
                            std::array<Vec3,8> dNpRef;
                            q2q1_hex_q1_pressure_basis_ref(xi, eta, zeta, Np, dNpRef);

                            std::array<Vec3,8> gradP;
                            for (int a = 0; a < 8; ++a) {
                                gradP[a].x = dNpRef[a].x * invHx;
                                gradP[a].y = dNpRef[a].y * invHy;
                                gradP[a].z = dNpRef[a].z * invHz;
                            }

                            // Momentum RHS: (v, f)
                            for (int a = 0; a < 27; ++a) {
                                const long long va = vnodes[a];
                                addB(q2q1_ux_row(g, va), Nv[a] * f.x * dV);
                                addB(q2q1_uy_row(g, va), Nv[a] * f.y * dV);
                                addB(q2q1_uz_row(g, va), Nv[a] * f.z * dV);
                            }

                            // PSPG RHS: tau (grad q, f)
                            for (int ap = 0; ap < 8; ++ap) {
                                const long long pa = pnodes[ap];
                                const double rhsP =
                                    tau * (gradP[ap].x * f.x +
                                           gradP[ap].y * f.y +
                                           gradP[ap].z * f.z) * dV;
                                addB(q2q1_p_row(g, pa), rhsP);
                            }
                        }
                    }
                }
            }
        }
    }

    q2q1_set_exact_strong_rhs(g, opt, sys);
}


static inline void q2q1_set_velocity_boundary_identities(
    const Q2Q1StructuredGrid& g,
    AssembledSystem& sys
) {
    for (int K = 0; K < g.nv1; ++K) {
        for (int J = 0; J < g.nv1; ++J) {
            for (int I = 0; I < g.nv1; ++I) {
                if (!q2q1_is_velocity_boundary_node(g, I, J, K)) continue;

                const long long v = q2q1_vnode_id(g, I, J, K);

                sparse_set_row_identity(sys.A, (int)q2q1_ux_row(g, v));
                sparse_set_row_identity(sys.A, (int)q2q1_uy_row(g, v));
                sparse_set_row_identity(sys.A, (int)q2q1_uz_row(g, v));
            }
        }
    }
}

static inline void q2q1_set_pressure_pin_identity(
    const Q2Q1StructuredGrid& g,
    int pressurePinPNode,
    AssembledSystem& sys
) {
    sparse_set_row_identity(sys.A, (int)q2q1_p_row(g, pressurePinPNode));
}

static inline Q2Q1StokesAssemblyReport q2q1_assemble_stokes_pspg_matrix_algebraic_mms_host(
    const Q2Q1StructuredGrid& g,
    const Q2Q1CellSlotCache& slots,
    const Q2Q1StokesMmsOptions& opt,
    AssembledSystem& sys
) {
    if (!sys.A.fixedPattern) throw std::runtime_error("Q2/Q1 Stokes assembly requires fixed-pattern SparseRows.");
    if (slots.nCells != g.nCells) throw std::runtime_error("Q2/Q1 Stokes assembly slot cache cell mismatch.");
    if (!(opt.nu > 0.0)) throw std::runtime_error("Q2/Q1 Stokes assembly requires nu > 0.");

    auto tAsm0 = std::chrono::steady_clock::now();

    Q2Q1StokesAssemblyReport rep;
    rep.rows = g.nRows;
    rep.nnz = sparse_nnz_flat(sys.A);
    rep.nu = opt.nu;
    rep.rhsMode = opt.rhsMode;
    rep.h = std::min(g.hx, std::min(g.hy, g.hz));
    rep.tau = q2q1_default_stokes_tau(g, opt);

    sparse_zero_values(sys.A);
    sys.b.assign((std::size_t)g.nRows, Real(0));
    sys.nDirichlet = (int)(3LL * q2q1_count_boundary_nodes(g).velocityBoundaryNodes + 1LL);

    std::array<double,4> qp;
    std::array<double,4> qw;
    q2q1_gauss4_1d(qp, qw);

    const double invHx = 2.0 / g.hx;
    const double invHy = 2.0 / g.hy;
    const double invHz = 2.0 / g.hz;

    const double lapScaleX = invHx * invHx;
    const double lapScaleY = invHy * invHy;
    const double lapScaleZ = invHz * invHz;

    const double detJ = g.hx * g.hy * g.hz / 8.0;

    auto addFlat = [&](long long cell, int localSlot, double val) {
        const int fs = slots.at(cell, localSlot);
        if (fs >= 0) {
            sparse_add_flat_slot(sys.A, fs, Real(val));
            ++rep.activeCellSlots;
        }
    };

    for (int k = 0; k < g.N; ++k) {
        for (int j = 0; j < g.N; ++j) {
            for (int i = 0; i < g.N; ++i) {
                const long long cell = q2q1_cell_linear_id(g, i, j, k);

                for (int kq = 0; kq < 4; ++kq) {
                    for (int jq = 0; jq < 4; ++jq) {
                        for (int iq = 0; iq < 4; ++iq) {
                            const double xi = qp[iq];
                            const double eta = qp[jq];
                            const double zeta = qp[kq];
                            const double dV = qw[iq] * qw[jq] * qw[kq] * detJ;

                            std::array<double,27> Nv;
                            std::array<Vec3,27> dNvRef;
                            std::array<Vec3,27> ddNvRef;
                            q2q1_hex_q2_basis_ref(xi, eta, zeta, Nv, dNvRef, ddNvRef);

                            std::array<double,8> Np;
                            std::array<Vec3,8> dNpRef;
                            q2q1_hex_q1_pressure_basis_ref(xi, eta, zeta, Np, dNpRef);

                            std::array<Vec3,27> gradV;
                            std::array<double,27> lapV;
                            for (int a = 0; a < 27; ++a) {
                                gradV[a].x = dNvRef[a].x * invHx;
                                gradV[a].y = dNvRef[a].y * invHy;
                                gradV[a].z = dNvRef[a].z * invHz;

                                lapV[a] =
                                    ddNvRef[a].x * lapScaleX +
                                    ddNvRef[a].y * lapScaleY +
                                    ddNvRef[a].z * lapScaleZ;
                            }

                            std::array<Vec3,8> gradP;
                            for (int a = 0; a < 8; ++a) {
                                gradP[a].x = dNpRef[a].x * invHx;
                                gradP[a].y = dNpRef[a].y * invHy;
                                gradP[a].z = dNpRef[a].z * invHz;
                            }

                            // Velocity rows: component-diagonal diffusion + pressure gradient.
                            for (int a = 0; a < 27; ++a) {
                                for (int b = 0; b < 27; ++b) {
                                    const double vv = opt.nu * dot3(gradV[a], gradV[b]) * dV;

                                    addFlat(cell, Q2Q1CellSlotCache::idx_vv(0, a, b), vv);
                                    addFlat(cell, Q2Q1CellSlotCache::idx_vv(1, a, b), vv);
                                    addFlat(cell, Q2Q1CellSlotCache::idx_vv(2, a, b), vv);
                                }

                                for (int bp = 0; bp < 8; ++bp) {
                                    addFlat(cell, Q2Q1CellSlotCache::idx_vp(0, a, bp),
                                            (-Np[bp] * gradV[a].x) * dV);
                                    addFlat(cell, Q2Q1CellSlotCache::idx_vp(1, a, bp),
                                            (-Np[bp] * gradV[a].y) * dV);
                                    addFlat(cell, Q2Q1CellSlotCache::idx_vp(2, a, bp),
                                            (-Np[bp] * gradV[a].z) * dV);
                                }
                            }

                            // Pressure rows: continuity + PSPG strong Stokes residual.
                            for (int ap = 0; ap < 8; ++ap) {
                                for (int b = 0; b < 27; ++b) {
                                    addFlat(cell, Q2Q1CellSlotCache::idx_pv(ap, 0, b),
                                            (Np[ap] * gradV[b].x
                                             + rep.tau * gradP[ap].x * (-opt.nu * lapV[b])) * dV);
                                    addFlat(cell, Q2Q1CellSlotCache::idx_pv(ap, 1, b),
                                            (Np[ap] * gradV[b].y
                                             + rep.tau * gradP[ap].y * (-opt.nu * lapV[b])) * dV);
                                    addFlat(cell, Q2Q1CellSlotCache::idx_pv(ap, 2, b),
                                            (Np[ap] * gradV[b].z
                                             + rep.tau * gradP[ap].z * (-opt.nu * lapV[b])) * dV);
                                }

                                for (int bp = 0; bp < 8; ++bp) {
                                    addFlat(cell, Q2Q1CellSlotCache::idx_pp(ap, bp),
                                            rep.tau * dot3(gradP[ap], gradP[bp]) * dV);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    q2q1_set_velocity_boundary_identities(g, sys);
    q2q1_set_pressure_pin_identity(g, opt.pressurePinPNode, sys);

    rep.assemblySeconds = seconds_since(tAsm0);

    auto tRhs0 = std::chrono::steady_clock::now();
    const std::vector<Real> xExact = q2q1_make_exact_vector(g);

    const std::string rhsModeLower = lower_copy(opt.rhsMode);
    if (rhsModeLower == "continuous") {
        q2q1_assemble_stokes_continuous_rhs_host(g, opt, rep.tau, sys);
        rep.continuousRhsSeconds = seconds_since(tRhs0);
    } else if (rhsModeLower == "cavity") {
        q2q1_set_cavity_strong_rhs(g, opt, sys);
        rep.continuousRhsSeconds = seconds_since(tRhs0);
    } else {
        q2q1_apply_sparse_fixed(sys.A, xExact, sys.b);
        rep.algebraicRhsSeconds = seconds_since(tRhs0);
    }

    q2q1_residual_norm_fixed(sys.A, xExact, sys.b, rep.exactResidualL2, rep.exactResidualInf);

    double x2 = 0.0;
    for (Real v : xExact) x2 += (double)v * (double)v;
    rep.exactVectorL2 = std::sqrt(x2);

    return rep;
}

static inline void q2q1_print_stokes_assembly_report(const Q2Q1StokesAssemblyReport& r) {
    std::cout << "q2q1StokesMmsAssembled    = 1\n";
    std::cout << "q2q1StokesMmsRows         = " << r.rows << "\n";
    std::cout << "q2q1StokesMmsNnz          = " << r.nnz << "\n";
    std::cout << "q2q1StokesMmsNu           = " << std::setprecision(16) << r.nu << "\n";
    std::cout << "q2q1StokesMmsRhsMode      = " << r.rhsMode << "\n";
    std::cout << "q2q1StokesMmsTau          = " << std::setprecision(16) << r.tau << "\n";
    std::cout << "q2q1StokesMmsH            = " << std::setprecision(16) << r.h << "\n";
    std::cout << "q2q1StokesMmsActiveCellSlotAdds = " << r.activeCellSlots << "\n";
    std::cout << "q2q1StokesMmsAssemblySeconds = " << r.assemblySeconds << "\n";
    std::cout << "q2q1StokesMmsAlgebraicRhsSeconds = " << r.algebraicRhsSeconds << "\n";
    std::cout << "q2q1StokesMmsContinuousRhsSeconds = " << r.continuousRhsSeconds << "\n";
    std::cout << "q2q1StokesMmsExactResidualL2 = " << std::setprecision(16) << r.exactResidualL2 << "\n";
    std::cout << "q2q1StokesMmsExactResidualInf = " << std::setprecision(16) << r.exactResidualInf << "\n";
    std::cout << "q2q1StokesMmsExactVectorL2 = " << std::setprecision(16) << r.exactVectorL2 << "\n";
}

struct Q2Q1IntegratedErrorReport {
    double l2U = -1.0;
    double relL2U = -1.0;
    double l2P = -1.0;
    double relL2P = -1.0;
    double maxAbsU = -1.0;
    double maxAbsP = -1.0;
};

static inline Q2Q1IntegratedErrorReport q2q1_compute_integrated_error(
    const Q2Q1StructuredGrid& g,
    const std::vector<Real>& x
) {
    Q2Q1IntegratedErrorReport rep;
    if ((long long)x.size() != g.nRows) return rep;

    std::array<double,4> qp;
    std::array<double,4> qw;
    q2q1_gauss4_1d(qp, qw);

    const double detJ = g.hx * g.hy * g.hz / 8.0;

    double eU2 = 0.0;
    double nU2 = 0.0;
    double eP2 = 0.0;
    double nP2 = 0.0;
    double maxU = 0.0;
    double maxP = 0.0;

    for (int k = 0; k < g.N; ++k) {
        for (int j = 0; j < g.N; ++j) {
            for (int i = 0; i < g.N; ++i) {
                const auto vnodes = q2q1_cell_velocity_nodes(g, i, j, k);
                const auto pnodes = q2q1_cell_pressure_nodes(g, i, j, k);

                for (int kq = 0; kq < 4; ++kq) {
                    for (int jq = 0; jq < 4; ++jq) {
                        for (int iq = 0; iq < 4; ++iq) {
                            const double xi = qp[iq];
                            const double eta = qp[jq];
                            const double zeta = qp[kq];
                            const double dV = qw[iq] * qw[jq] * qw[kq] * detJ;

                            const Vec3 Xq{
                                g.xmin + ((double)i + 0.5 * (xi   + 1.0)) * g.hx,
                                g.ymin + ((double)j + 0.5 * (eta  + 1.0)) * g.hy,
                                g.zmin + ((double)k + 0.5 * (zeta + 1.0)) * g.hz
                            };

                            const Q2Q1MmsAtPoint ex = q2q1_exact_mms_at(Xq);

                            std::array<double,27> Nv;
                            std::array<Vec3,27> dNv;
                            std::array<Vec3,27> ddNv;
                            q2q1_hex_q2_basis_ref(xi, eta, zeta, Nv, dNv, ddNv);

                            std::array<double,8> Np;
                            std::array<Vec3,8> dNp;
                            q2q1_hex_q1_pressure_basis_ref(xi, eta, zeta, Np, dNp);

                            double uhx = 0.0;
                            double uhy = 0.0;
                            double uhz = 0.0;
                            for (int a = 0; a < 27; ++a) {
                                const long long v = vnodes[a];
                                uhx += Nv[a] * (double)x[(std::size_t)q2q1_ux_row(g, v)];
                                uhy += Nv[a] * (double)x[(std::size_t)q2q1_uy_row(g, v)];
                                uhz += Nv[a] * (double)x[(std::size_t)q2q1_uz_row(g, v)];
                            }

                            double ph = 0.0;
                            for (int a = 0; a < 8; ++a) {
                                const long long p = pnodes[a];
                                ph += Np[a] * (double)x[(std::size_t)q2q1_p_row(g, p)];
                            }

                            const double dux = uhx - ex.ux;
                            const double duy = uhy - ex.uy;
                            const double duz = uhz - ex.uz;
                            const double dp = ph - ex.p;

                            const double eu = dux * dux + duy * duy + duz * duz;
                            const double nu = ex.ux * ex.ux + ex.uy * ex.uy + ex.uz * ex.uz;

                            eU2 += eu * dV;
                            nU2 += nu * dV;
                            eP2 += dp * dp * dV;
                            nP2 += ex.p * ex.p * dV;

                            maxU = std::max(maxU, std::sqrt(eu));
                            maxP = std::max(maxP, std::abs(dp));
                        }
                    }
                }
            }
        }
    }

    rep.l2U = std::sqrt(eU2);
    rep.relL2U = std::sqrt(eU2 / std::max(nU2, 1.0e-300));
    rep.l2P = std::sqrt(eP2);
    rep.relL2P = std::sqrt(eP2 / std::max(nP2, 1.0e-300));
    rep.maxAbsU = maxU;
    rep.maxAbsP = maxP;

    return rep;
}

static inline void q2q1_print_integrated_error_report(const Q2Q1IntegratedErrorReport& r) {
    std::cout << "q2q1IntegratedErrorDone   = 1\n";
    std::cout << "q2q1IntegratedL2U         = " << std::setprecision(16) << r.l2U << "\n";
    std::cout << "q2q1IntegratedRelL2U      = " << std::setprecision(16) << r.relL2U << "\n";
    std::cout << "q2q1IntegratedL2P         = " << std::setprecision(16) << r.l2P << "\n";
    std::cout << "q2q1IntegratedRelL2P      = " << std::setprecision(16) << r.relL2P << "\n";
    std::cout << "q2q1IntegratedMaxAbsU     = " << std::setprecision(16) << r.maxAbsU << "\n";
    std::cout << "q2q1IntegratedMaxAbsP     = " << std::setprecision(16) << r.maxAbsP << "\n";
}


static inline void q2q1_print_exact_compare_report(const Q2Q1ExactCompareReport& r) {
    std::cout << "q2q1ExactCompareDone      = 1\n";
    std::cout << "q2q1ExactCompareAbsL2     = " << std::setprecision(16) << r.absL2 << "\n";
    std::cout << "q2q1ExactCompareRelL2     = " << std::setprecision(16) << r.relL2 << "\n";
    std::cout << "q2q1ExactCompareMaxAbs    = " << std::setprecision(16) << r.maxAbs << "\n";
    std::cout << "q2q1ExactCompareUAbsL2    = " << std::setprecision(16) << r.uAbsL2 << "\n";
    std::cout << "q2q1ExactComparePAbsL2    = " << std::setprecision(16) << r.pAbsL2 << "\n";
}
