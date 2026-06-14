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
