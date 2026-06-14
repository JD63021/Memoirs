#pragma once

// ============================================================================
// SECTION 13: Continuous steady Stokes/PSPG MMS RHS for assembled Q1/Q1
// ============================================================================
//
// PDE target:
//   -nu Lap(u) + grad(p) = f
//    div(u) = 0
//
// Weak rows:
//   velocity:
//      nu (grad v, grad u) - (div v, p) = (v, f)
//
//   pressure/PSPG:
//      (q, div u) + tau (grad q, grad p) = tau (grad q, f)
//
// For Q1 on affine hexes, the strong Laplacian of the discrete velocity basis
// is zero inside each element, so the PSPG LHS currently contains no velocity
// diffusion-residual block. The exact MMS forcing f still contains -nu Lap(u_exact).
// ============================================================================

struct Q1Q1ForceAtPoint {
    double fx = 0.0;
    double fy = 0.0;
    double fz = 0.0;
};

struct Q1Q1FieldL2Error {
    double uL2 = -1.0;
    double pL2 = -1.0;
    double pMeanShift = 0.0;
    double pMeanShiftedL2 = -1.0;
    double volume = 0.0;
};

static inline Q1Q1ForceAtPoint q1q1_stokes_force_at(const Vec3& X, double nu) {
    const double pi = kPi;

    const double sx = std::sin(pi * X.x);
    const double sy = std::sin(pi * X.y);
    const double sz = std::sin(pi * X.z);

    const double cx = std::cos(pi * X.x);
    const double cy = std::cos(pi * X.y);
    const double cz = std::cos(pi * X.z);

    const double s2x = std::sin(2.0 * pi * X.x);
    const double s2y = std::sin(2.0 * pi * X.y);
    const double c2x = std::cos(2.0 * pi * X.x);
    const double c2y = std::cos(2.0 * pi * X.y);
    const double c2z = std::cos(2.0 * pi * X.z);

    const double A  = sx * sx;
    const double B  = sy * cy;
    const double C  = sz * sz;

    const double D  = sx * cx;
    const double E  = sy * sy;

    const double Add = 2.0 * pi * pi * c2x;
    const double Bdd = -2.0 * pi * pi * s2y;
    const double Cdd = 2.0 * pi * pi * c2z;

    const double Ddd = -2.0 * pi * pi * s2x;
    const double Edd = 2.0 * pi * pi * c2y;

    const double lapUx =  2.0 * pi * (Add * B * C + A * Bdd * C + A * B * Cdd);
    const double lapUy = -2.0 * pi * (Ddd * E * C + D * Edd * C + D * E * Cdd);
    const double lapUz =  0.0;

    const double dpdx = -pi * sx * sy * sz;
    const double dpdy =  pi * cx * cy * sz;
    const double dpdz =  pi * cx * sy * cz;

    Q1Q1ForceAtPoint f;
    f.fx = -nu * lapUx + dpdx;
    f.fy = -nu * lapUy + dpdy;
    f.fz = -nu * lapUz + dpdz;
    return f;
}

static inline void q1q1_zero_rhs_and_add_continuous_stokes_rhs(
    const PolyMesh& mesh,
    const LinearCgDofMap& dm,
    const Q1Q1AlgInfo& info,
    AssembledSystem& sys
) {
    const int nNodes = dm.nDofs;
    const int nRows = 4 * nNodes;

    if ((int)sys.b.size() != nRows) {
        throw std::runtime_error("Q1/Q1 continuous RHS expected 4*nnode RHS.");
    }

    sys.b.assign(nRows, Real(0));

    // 4-point Gauss-Legendre on [-1,1]. Use this for the sine MMS forcing;
    // 2x2x2 was exact for polynomial Q1 matrix pieces but under-integrated
    // the continuous trigonometric RHS.
    const double qp[4] = {
        -0.86113631159405257522,
        -0.33998104358485626480,
         0.33998104358485626480,
         0.86113631159405257522
    };
    const double qw[4] = {
         0.34785484513745385737,
         0.65214515486254614263,
         0.65214515486254614263,
         0.34785484513745385737
    };

    for (const auto& c : mesh.cells) {
        const auto hv = ordered_hex_vertices_axis_aligned(mesh, c);

        for (int iq = 0; iq < 4; ++iq) {
            for (int jq = 0; jq < 4; ++jq) {
                for (int kq = 0; kq < 4; ++kq) {
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

                    const double detJ = std::abs(det3(J)) * w;
                    const Mat3 invJ = inv3(J);

                    std::array<Vec3,8> grad;
                    for (int a = 0; a < 8; ++a) grad[a] = invJT_mul(invJ, dNref[a]);

                    const Q1Q1ForceAtPoint f = q1q1_stokes_force_at(xq, info.nu);

                    for (int a = 0; a < 8; ++a) {
                        const int na = hv[a];

                        sys.b[q1q1_row(na, 0)] += Real(N[a] * f.fx * detJ);
                        sys.b[q1q1_row(na, 1)] += Real(N[a] * f.fy * detJ);
                        sys.b[q1q1_row(na, 2)] += Real(N[a] * f.fz * detJ);

                        const double pspgRhs =
                            info.tau * (grad[a].x * f.fx +
                                        grad[a].y * f.fy +
                                        grad[a].z * f.fz) * detJ;

                        sys.b[q1q1_row(na, 3)] += Real(pspgRhs);
                    }
                }
            }
        }
    }

    // Strong velocity rows. Exact wall velocity is zero for this MMS, but use
    // the exact evaluator rather than hard-coding zero.
    for (int node : dm.boundaryDofs) {
        const Q1Q1ExactAtNode ex = q1q1_exact_mms_at(mesh.points[node]);

        sys.b[q1q1_row(node, 0)] = Real(ex.ux);
        sys.b[q1q1_row(node, 1)] = Real(ex.uy);
        sys.b[q1q1_row(node, 2)] = Real(ex.uz);
    }

    // Pressure pin.
    const Q1Q1ExactAtNode pex = q1q1_exact_mms_at(mesh.points[info.pressurePinNode]);
    sys.b[q1q1_row(info.pressurePinNode, 3)] = Real(pex.p);
}

static inline AssembledSystem assemble_q1q1_stokes_pspg_continuous(
    const PolyMesh& mesh,
    const LinearCgDofMap& dm,
    Q1Q1AlgInfo& info
) {
    AssembledSystem sys = assemble_q1q1_stokes_pspg_algebraic(mesh, dm, info);
    q1q1_zero_rhs_and_add_continuous_stokes_rhs(mesh, dm, info, sys);
    return sys;
}

static inline void q1q1_eval_uh_ph(
    const PolyMesh& mesh,
    const std::array<int,8>& hv,
    const std::array<double,8>& N,
    const std::vector<Real>& x,
    double& uhx,
    double& uhy,
    double& uhz,
    double& ph
) {
    uhx = 0.0;
    uhy = 0.0;
    uhz = 0.0;
    ph = 0.0;

    for (int a = 0; a < 8; ++a) {
        const int node = hv[a];
        uhx += N[a] * double(x[q1q1_row(node, 0)]);
        uhy += N[a] * double(x[q1q1_row(node, 1)]);
        uhz += N[a] * double(x[q1q1_row(node, 2)]);
        ph  += N[a] * double(x[q1q1_row(node, 3)]);
    }
}

static inline Q1Q1FieldL2Error q1q1_compute_field_l2_error(
    const PolyMesh& mesh,
    const std::vector<Real>& x
) {
    const int nNodes = (int)mesh.points.size();
    if ((int)x.size() != 4 * nNodes) {
        throw std::runtime_error("Q1/Q1 field L2 error expected 4*nnode vector.");
    }

    // 4-point Gauss-Legendre for integrated field L2 errors.
    const double qp[4] = {
        -0.86113631159405257522,
        -0.33998104358485626480,
         0.33998104358485626480,
         0.86113631159405257522
    };
    const double qw[4] = {
         0.34785484513745385737,
         0.65214515486254614263,
         0.65214515486254614263,
         0.34785484513745385737
    };

    Q1Q1FieldL2Error e;

    double uErr2 = 0.0;
    double pErr2 = 0.0;
    double pDiffInt = 0.0;
    double vol = 0.0;

    for (const auto& c : mesh.cells) {
        const auto hv = ordered_hex_vertices_axis_aligned(mesh, c);

        for (int iq = 0; iq < 4; ++iq) {
            for (int jq = 0; jq < 4; ++jq) {
                for (int kq = 0; kq < 4; ++kq) {
                    const double xi = qp[iq];
                    const double eta = qp[jq];
                    const double zeta = qp[kq];
                    const double w = qw[iq] * qw[jq] * qw[kq];

                    std::array<double,8> N;
                    std::array<Vec3,8> dNref;
                    hex_q1_basis(xi, eta, zeta, N, dNref);

                    Vec3 xq;
                    Mat3 J;

                    for (int aa = 0; aa < 8; ++aa) {
                        const Vec3& X = mesh.points[hv[aa]];

                        xq.x += N[aa] * X.x;
                        xq.y += N[aa] * X.y;
                        xq.z += N[aa] * X.z;

                        J.a[0][0] += dNref[aa].x * X.x;
                        J.a[1][0] += dNref[aa].x * X.y;
                        J.a[2][0] += dNref[aa].x * X.z;

                        J.a[0][1] += dNref[aa].y * X.x;
                        J.a[1][1] += dNref[aa].y * X.y;
                        J.a[2][1] += dNref[aa].y * X.z;

                        J.a[0][2] += dNref[aa].z * X.x;
                        J.a[1][2] += dNref[aa].z * X.y;
                        J.a[2][2] += dNref[aa].z * X.z;
                    }

                    const double dV = std::abs(det3(J)) * w;

                    double uhx, uhy, uhz, ph;
                    q1q1_eval_uh_ph(mesh, hv, N, x, uhx, uhy, uhz, ph);

                    const Q1Q1ExactAtNode ex = q1q1_exact_mms_at(xq);

                    const double dux = uhx - ex.ux;
                    const double duy = uhy - ex.uy;
                    const double duz = uhz - ex.uz;
                    const double dp = ph - ex.p;

                    uErr2 += (dux*dux + duy*duy + duz*duz) * dV;
                    pErr2 += dp * dp * dV;
                    pDiffInt += dp * dV;
                    vol += dV;
                }
            }
        }
    }

    const double pShift = pDiffInt / vol;

    double pShiftedErr2 = 0.0;

    for (const auto& c : mesh.cells) {
        const auto hv = ordered_hex_vertices_axis_aligned(mesh, c);

        for (int iq = 0; iq < 4; ++iq) {
            for (int jq = 0; jq < 4; ++jq) {
                for (int kq = 0; kq < 4; ++kq) {
                    const double xi = qp[iq];
                    const double eta = qp[jq];
                    const double zeta = qp[kq];
                    const double w = qw[iq] * qw[jq] * qw[kq];

                    std::array<double,8> N;
                    std::array<Vec3,8> dNref;
                    hex_q1_basis(xi, eta, zeta, N, dNref);

                    Vec3 xq;
                    Mat3 J;

                    for (int aa = 0; aa < 8; ++aa) {
                        const Vec3& X = mesh.points[hv[aa]];

                        xq.x += N[aa] * X.x;
                        xq.y += N[aa] * X.y;
                        xq.z += N[aa] * X.z;

                        J.a[0][0] += dNref[aa].x * X.x;
                        J.a[1][0] += dNref[aa].x * X.y;
                        J.a[2][0] += dNref[aa].x * X.z;

                        J.a[0][1] += dNref[aa].y * X.x;
                        J.a[1][1] += dNref[aa].y * X.y;
                        J.a[2][1] += dNref[aa].y * X.z;

                        J.a[0][2] += dNref[aa].z * X.x;
                        J.a[1][2] += dNref[aa].z * X.y;
                        J.a[2][2] += dNref[aa].z * X.z;
                    }

                    const double dV = std::abs(det3(J)) * w;

                    double uhx, uhy, uhz, ph;
                    q1q1_eval_uh_ph(mesh, hv, N, x, uhx, uhy, uhz, ph);

                    const Q1Q1ExactAtNode ex = q1q1_exact_mms_at(xq);
                    const double dpShifted = (ph - ex.p) - pShift;

                    pShiftedErr2 += dpShifted * dpShifted * dV;
                }
            }
        }
    }

    e.uL2 = std::sqrt(uErr2);
    e.pL2 = std::sqrt(pErr2);
    e.pMeanShift = pShift;
    e.pMeanShiftedL2 = std::sqrt(pShiftedErr2);
    e.volume = vol;
    return e;
}

static inline void q1q1_print_field_l2_error(const Q1Q1FieldL2Error& e) {
    std::cout << "--------------- q1q1 field L2 error -----------------\n";
    std::cout << "q1q1UFieldL2              = " << std::setprecision(16) << e.uL2 << "\n";
    std::cout << "q1q1PFieldL2              = " << std::setprecision(16) << e.pL2 << "\n";
    std::cout << "q1q1PFieldMeanShift       = " << std::setprecision(16) << e.pMeanShift << "\n";
    std::cout << "q1q1PFieldMeanShiftedL2   = " << std::setprecision(16) << e.pMeanShiftedL2 << "\n";
    std::cout << "q1q1FieldErrorQuadrature  = 4x4x4\n";
    std::cout << "q1q1ErrorVolume           = " << std::setprecision(16) << e.volume << "\n";
    std::cout << "-----------------------------------------------------\n";
}
