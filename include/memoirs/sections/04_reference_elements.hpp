#pragma once

// Auto-split from frozen patch014 one-shot source.
// This header is intentionally included in order by apps/poisson_mms.cpp.

// ============================================================================
// SECTION 4: Reference elements Q1 hex / P1 tet
// ============================================================================
//
// Current scope:
//   - Hex Q1 nodal Lagrange on [-1,1]^3.
//   - Tet P1 nodal Lagrange on reference tet.
//   - Generic non-sumfactorized assembly.
//   - Hex vertex ordering assumes axis-aligned blockMesh cells for this one-shot.
//     This is enough for the first frozen Q1 blockMesh workflow.
// ============================================================================

static constexpr double kPi = 3.141592653589793238462643383279502884;

struct Mat3 {
    double a[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
};

static double det3(const Mat3& M) {
    return
        M.a[0][0]*(M.a[1][1]*M.a[2][2] - M.a[1][2]*M.a[2][1])
      - M.a[0][1]*(M.a[1][0]*M.a[2][2] - M.a[1][2]*M.a[2][0])
      + M.a[0][2]*(M.a[1][0]*M.a[2][1] - M.a[1][1]*M.a[2][0]);
}

static Mat3 inv3(const Mat3& M) {
    const double d = det3(M);
    if (std::abs(d) < 1e-300) {
        throw std::runtime_error("Singular 3x3 Jacobian.");
    }

    Mat3 B;
    B.a[0][0] =  (M.a[1][1]*M.a[2][2] - M.a[1][2]*M.a[2][1]) / d;
    B.a[0][1] = -(M.a[0][1]*M.a[2][2] - M.a[0][2]*M.a[2][1]) / d;
    B.a[0][2] =  (M.a[0][1]*M.a[1][2] - M.a[0][2]*M.a[1][1]) / d;

    B.a[1][0] = -(M.a[1][0]*M.a[2][2] - M.a[1][2]*M.a[2][0]) / d;
    B.a[1][1] =  (M.a[0][0]*M.a[2][2] - M.a[0][2]*M.a[2][0]) / d;
    B.a[1][2] = -(M.a[0][0]*M.a[1][2] - M.a[0][2]*M.a[1][0]) / d;

    B.a[2][0] =  (M.a[1][0]*M.a[2][1] - M.a[1][1]*M.a[2][0]) / d;
    B.a[2][1] = -(M.a[0][0]*M.a[2][1] - M.a[0][1]*M.a[2][0]) / d;
    B.a[2][2] =  (M.a[0][0]*M.a[1][1] - M.a[0][1]*M.a[1][0]) / d;
    return B;
}

static Vec3 invJT_mul(const Mat3& invJ, const Vec3& gref) {
    // grad_x = J^{-T} grad_ref.
    return {
        invJ.a[0][0]*gref.x + invJ.a[1][0]*gref.y + invJ.a[2][0]*gref.z,
        invJ.a[0][1]*gref.x + invJ.a[1][1]*gref.y + invJ.a[2][1]*gref.z,
        invJ.a[0][2]*gref.x + invJ.a[1][2]*gref.y + invJ.a[2][2]*gref.z
    };
}

static std::array<int,8> ordered_hex_vertices_axis_aligned(const PolyMesh& m, const Cell& c) {
    if (c.verts.size() != 8) throw std::runtime_error("Hex cell does not have 8 vertices.");

    Vec3 cen;
    for (int v : c.verts) {
        cen.x += m.points[v].x;
        cen.y += m.points[v].y;
        cen.z += m.points[v].z;
    }
    cen.x /= 8.0; cen.y /= 8.0; cen.z /= 8.0;

    // Desired local order:
    // 0 (-,-,-), 1 (+,-,-), 2 (+,+,-), 3 (-,+,-),
    // 4 (-,-,+), 5 (+,-,+), 6 (+,+,+), 7 (-,+,+)
    int bitToLocal[8];
    bitToLocal[0] = 0;
    bitToLocal[1] = 1;
    bitToLocal[3] = 2;
    bitToLocal[2] = 3;
    bitToLocal[4] = 4;
    bitToLocal[5] = 5;
    bitToLocal[7] = 6;
    bitToLocal[6] = 7;

    std::array<int,8> hv;
    hv.fill(-1);

    for (int v : c.verts) {
        const auto& p = m.points[v];
        int bx = (p.x > cen.x) ? 1 : 0;
        int by = (p.y > cen.y) ? 1 : 0;
        int bz = (p.z > cen.z) ? 1 : 0;
        int bits = bx + 2*by + 4*bz;
        int loc = bitToLocal[bits];
        if (hv[loc] != -1) {
            throw std::runtime_error("Failed axis-aligned hex ordering: duplicate corner.");
        }
        hv[loc] = v;
    }

    for (int i = 0; i < 8; ++i) {
        if (hv[i] < 0) throw std::runtime_error("Failed axis-aligned hex ordering: missing corner.");
    }

    return hv;
}

static void hex_q1_basis(
    double xi, double eta, double zeta,
    std::array<double,8>& N,
    std::array<Vec3,8>& dN
) {
    static const int sx[8] = {-1,  1,  1, -1, -1,  1,  1, -1};
    static const int sy[8] = {-1, -1,  1,  1, -1, -1,  1,  1};
    static const int sz[8] = {-1, -1, -1, -1,  1,  1,  1,  1};

    for (int a = 0; a < 8; ++a) {
        const double X = 1.0 + sx[a]*xi;
        const double Y = 1.0 + sy[a]*eta;
        const double Z = 1.0 + sz[a]*zeta;

        N[a] = 0.125 * X * Y * Z;
        dN[a].x = 0.125 * sx[a] * Y * Z;
        dN[a].y = 0.125 * X * sy[a] * Z;
        dN[a].z = 0.125 * X * Y * sz[a];
    }
}

static const std::array<Vec3,4>& tet_p1_grad_ref() {
    static const std::array<Vec3,4> g = {{
        {-1.0, -1.0, -1.0},
        { 1.0,  0.0,  0.0},
        { 0.0,  1.0,  0.0},
        { 0.0,  0.0,  1.0}
    }};
    return g;
}
