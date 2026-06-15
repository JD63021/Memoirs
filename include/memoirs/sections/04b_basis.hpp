#pragma once

// ============================================================================
// SECTION 4b: Basis evaluators
// ============================================================================
// Thin abstraction layer over the frozen Q1/P1 basis routines.  This is the
// place where the next spaces should be added:
//   - dg_tet_p1_modal / dg_tet_p1_nodal
//   - dg_hex_q1_modal / dg_hex_q1_nodal
//   - rt0_tet / rt0_hex mixed bases
// ============================================================================

struct HexQ1BasisAtPoint {
    std::array<double,8> N{};
    std::array<Vec3,8> dNref{};
};

struct TetP1BasisAtPoint {
    std::array<double,4> N{};
    std::array<Vec3,4> dNref{};
};


struct TetP2BasisAtPoint {
    // Local order:
    //   0..3: vertices lambda0..lambda3
    //   4: edge 0-1
    //   5: edge 0-2
    //   6: edge 0-3
    //   7: edge 1-2
    //   8: edge 1-3
    //   9: edge 2-3
    std::array<double,10> N{};
    std::array<Vec3,10> dNref{};
};

static HexQ1BasisAtPoint eval_hex_q1_basis_at(double xi, double eta, double zeta) {
    HexQ1BasisAtPoint b;
    hex_q1_basis(xi, eta, zeta, b.N, b.dNref);
    return b;
}

static TetP1BasisAtPoint eval_tet_p1_basis_at(double r, double s, double t) {
    TetP1BasisAtPoint b;
    b.N[0] = 1.0 - r - s - t;
    b.N[1] = r;
    b.N[2] = s;
    b.N[3] = t;
    const auto& g = tet_p1_grad_ref();
    for (int a = 0; a < 4; ++a) b.dNref[a] = g[a];
    return b;
}

static Vec3 map_hex_q1_to_physical(
    const PolyMesh& m,
    const std::array<int,8>& hv,
    const std::array<double,8>& N
) {
    Vec3 x;
    for (int a = 0; a < 8; ++a) {
        const Vec3& X = m.points[hv[a]];
        x.x += N[a] * X.x;
        x.y += N[a] * X.y;
        x.z += N[a] * X.z;
    }
    return x;
}

static Mat3 jacobian_hex_q1(
    const PolyMesh& m,
    const std::array<int,8>& hv,
    const std::array<Vec3,8>& dNref
) {
    Mat3 J;
    for (int a = 0; a < 8; ++a) {
        const Vec3& X = m.points[hv[a]];
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
    return J;
}


static TetP2BasisAtPoint eval_tet_p2_basis_at(double r, double s, double t) {
    TetP2BasisAtPoint b;

    const double l0 = 1.0 - r - s - t;
    const double l1 = r;
    const double l2 = s;
    const double l3 = t;

    const double L[4] = {l0, l1, l2, l3};
    const Vec3 gL[4] = {
        {-1.0, -1.0, -1.0},
        { 1.0,  0.0,  0.0},
        { 0.0,  1.0,  0.0},
        { 0.0,  0.0,  1.0}
    };

    for (int a = 0; a < 4; ++a) {
        b.N[a] = L[a] * (2.0 * L[a] - 1.0);
        const double ca = (4.0 * L[a] - 1.0);
        b.dNref[a] = {ca * gL[a].x, ca * gL[a].y, ca * gL[a].z};
    }

    static const int e[6][2] = {
        {0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3}
    };
    for (int k = 0; k < 6; ++k) {
        const int i = e[k][0];
        const int j = e[k][1];
        const int a = 4 + k;
        b.N[a] = 4.0 * L[i] * L[j];
        b.dNref[a] = {
            4.0 * (L[i] * gL[j].x + L[j] * gL[i].x),
            4.0 * (L[i] * gL[j].y + L[j] * gL[i].y),
            4.0 * (L[i] * gL[j].z + L[j] * gL[i].z)
        };
    }

    return b;
}



// Cell-local modal/monomial P2 basis on the reference tetrahedron.
// Coordinates are (r,s,t) = (lambda1, lambda2, lambda3), lambda0=1-r-s-t.
// This spans the same P2 space as the nodal basis but the coefficients are
// cell-local polynomial/modal coefficients, not nodal values.
static TetP2BasisAtPoint eval_tet_p2_modal_basis_at(double r, double s, double t) {
    TetP2BasisAtPoint b;

    b.N[0] = 1.0;
    b.N[1] = r;
    b.N[2] = s;
    b.N[3] = t;
    b.N[4] = r * s;
    b.N[5] = r * t;
    b.N[6] = s * t;
    b.N[7] = r * r;
    b.N[8] = s * s;
    b.N[9] = t * t;

    b.dNref[0] = {0.0, 0.0, 0.0};
    b.dNref[1] = {1.0, 0.0, 0.0};
    b.dNref[2] = {0.0, 1.0, 0.0};
    b.dNref[3] = {0.0, 0.0, 1.0};
    b.dNref[4] = {s,   r,   0.0};
    b.dNref[5] = {t,   0.0, r};
    b.dNref[6] = {0.0, t,   s};
    b.dNref[7] = {2.0*r, 0.0,   0.0};
    b.dNref[8] = {0.0,   2.0*s, 0.0};
    b.dNref[9] = {0.0,   0.0,   2.0*t};

    return b;
}

static Vec3 map_tet_p1_to_physical(
    const PolyMesh& m,
    const std::array<int,4>& tv,
    const std::array<double,4>& N
) {
    Vec3 x;
    for (int a = 0; a < 4; ++a) {
        const Vec3& X = m.points[tv[a]];
        x.x += N[a] * X.x;
        x.y += N[a] * X.y;
        x.z += N[a] * X.z;
    }
    return x;
}

static Mat3 jacobian_tet_p1(
    const PolyMesh& m,
    const std::array<int,4>& tv
) {
    const Vec3& X0 = m.points[tv[0]];
    const Vec3& X1 = m.points[tv[1]];
    const Vec3& X2 = m.points[tv[2]];
    const Vec3& X3 = m.points[tv[3]];

    Mat3 J;
    J.a[0][0] = X1.x - X0.x; J.a[1][0] = X1.y - X0.y; J.a[2][0] = X1.z - X0.z;
    J.a[0][1] = X2.x - X0.x; J.a[1][1] = X2.y - X0.y; J.a[2][1] = X2.z - X0.z;
    J.a[0][2] = X3.x - X0.x; J.a[1][2] = X3.y - X0.y; J.a[2][2] = X3.z - X0.z;
    return J;
}
