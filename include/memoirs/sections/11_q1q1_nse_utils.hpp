#pragma once

// ============================================================================
// SECTION 11A: Q1/Q1 node-major utilities for the experimental NSE IJ branch
// ============================================================================
//
// This header is intentionally tiny. It only introduces:
//   - node-major [ux,uy,uz,p] row numbering
//   - the divergence-free sine MMS used for Q1/Q1 Stokes/NSE checks
//   - exact 4-field vector construction
//   - a small probe print
//
// No matrix assembly is added in this patch.
// ============================================================================

struct Q1Q1ExactAtNode {
    double ux = 0.0;
    double uy = 0.0;
    double uz = 0.0;
    double p  = 0.0;
};

static inline int q1q1_row(int node, int field) {
    return 4 * node + field;
}

static inline const char* q1q1_field_name(int field) {
    switch (field) {
        case 0: return "ux";
        case 1: return "uy";
        case 2: return "uz";
        case 3: return "p";
        default: return "?";
    }
}

static inline Q1Q1ExactAtNode q1q1_exact_mms_at(const Vec3& X) {
    const double pi = kPi;

    const double sx = std::sin(pi * X.x);
    const double sy = std::sin(pi * X.y);
    const double sz = std::sin(pi * X.z);

    const double cx = std::cos(pi * X.x);
    const double cy = std::cos(pi * X.y);

    Q1Q1ExactAtNode e;

    // Divergence-free smooth velocity, zero on all unit-cube walls.
    e.ux =  2.0 * pi * sx * sx * sy * cy * sz * sz;
    e.uy = -2.0 * pi * sx * cx * sy * sy * sz * sz;
    e.uz =  0.0;

    // Smooth pressure. p(0,0,0)=0, so node 0 is a convenient pressure pin.
    e.p = cx * sy * sz;

    return e;
}

static inline std::vector<Real> q1q1_make_exact_node_major_vector(const PolyMesh& mesh) {
    const int nNodes = (int)mesh.points.size();
    std::vector<Real> x(4 * nNodes, Real(0));

    for (int node = 0; node < nNodes; ++node) {
        const Q1Q1ExactAtNode e = q1q1_exact_mms_at(mesh.points[node]);

        x[q1q1_row(node, 0)] = Real(e.ux);
        x[q1q1_row(node, 1)] = Real(e.uy);
        x[q1q1_row(node, 2)] = Real(e.uz);
        x[q1q1_row(node, 3)] = Real(e.p);
    }

    return x;
}

static inline void q1q1_probe_exact_vector(const PolyMesh& mesh, const LinearCgDofMap& dm) {
    const int nNodes = (int)mesh.points.size();
    const int nRows = 4 * nNodes;

    if (dm.nDofs != nNodes) {
        std::cout << "q1q1ProbeWarning          = dm.nDofs != mesh.points.size(); Q1 node-major assumes point DOFs\n";
    }

    const std::vector<Real> x = q1q1_make_exact_node_major_vector(mesh);

    double u2 = 0.0;
    double p2 = 0.0;
    double umax = 0.0;
    double pmax = 0.0;

    for (int node = 0; node < nNodes; ++node) {
        const double ux = double(x[q1q1_row(node, 0)]);
        const double uy = double(x[q1q1_row(node, 1)]);
        const double uz = double(x[q1q1_row(node, 2)]);
        const double p  = double(x[q1q1_row(node, 3)]);

        const double un = std::sqrt(ux*ux + uy*uy + uz*uz);

        u2 += ux*ux + uy*uy + uz*uz;
        p2 += p*p;
        umax = std::max(umax, un);
        pmax = std::max(pmax, std::abs(p));
    }

    std::cout << "--------------- q1q1 exact-vector probe -------------\n";
    std::cout << "q1q1ProbeStatus           = patch2A_node_major_exact_vector_only\n";
    std::cout << "q1q1Nodes                 = " << nNodes << "\n";
    std::cout << "q1q1Rows                  = " << nRows << "\n";
    std::cout << "q1q1Ordering              = node-major [ux,uy,uz,p]\n";
    std::cout << "q1q1Node0Rows             = "
              << q1q1_row(0,0) << " "
              << q1q1_row(0,1) << " "
              << q1q1_row(0,2) << " "
              << q1q1_row(0,3) << "\n";
    if (nNodes > 1) {
        std::cout << "q1q1Node1Rows             = "
                  << q1q1_row(1,0) << " "
                  << q1q1_row(1,1) << " "
                  << q1q1_row(1,2) << " "
                  << q1q1_row(1,3) << "\n";
    }
    std::cout << "q1q1ExactVectorSize       = " << x.size() << "\n";
    std::cout << "q1q1ExactUNodalNorm       = " << std::setprecision(16) << std::sqrt(u2) << "\n";
    std::cout << "q1q1ExactPNodalNorm       = " << std::setprecision(16) << std::sqrt(p2) << "\n";
    std::cout << "q1q1ExactUNodalMax        = " << std::setprecision(16) << umax << "\n";
    std::cout << "q1q1ExactPNodalMax        = " << std::setprecision(16) << pmax << "\n";
    std::cout << "q1q1PressurePinNode0Value = " << std::setprecision(16)
              << double(x[q1q1_row(0,3)]) << "\n";
    std::cout << "-----------------------------------------------------\n";
}
