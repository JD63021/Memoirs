#pragma once

// ============================================================================
// SECTION 4a: Quadrature rules
// ============================================================================
// Small host-side quadrature helpers used by the generic CG Poisson path today
// and by the future DG/SIPG path.  The intent is to remove hard-coded Gauss
// loops from operators and diagnostics without changing any numerical results.
// ============================================================================

struct VolumeQuadraturePoint {
    double xi = 0.0;
    double eta = 0.0;
    double zeta = 0.0;
    double weight = 0.0;
};

struct FaceQuadraturePoint {
    double r = 0.0;
    double s = 0.0;
    double weight = 0.0;
};

static const std::vector<VolumeQuadraturePoint>& hex_q1_volume_quadrature_2x2x2() {
    static const std::vector<VolumeQuadraturePoint> q = [] {
        const double a = 1.0 / std::sqrt(3.0);
        const double p[2] = {-a, a};
        std::vector<VolumeQuadraturePoint> out;
        out.reserve(8);
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                for (int k = 0; k < 2; ++k) {
                    out.push_back({p[i], p[j], p[k], 1.0});
                }
            }
        }
        return out;
    }();
    return q;
}

static const std::vector<VolumeQuadraturePoint>& hex_q1_error_quadrature_3x3x3() {
    static const std::vector<VolumeQuadraturePoint> q = [] {
        const double a = std::sqrt(3.0 / 5.0);
        const double p[3] = {-a, 0.0, a};
        const double w[3] = {5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0};
        std::vector<VolumeQuadraturePoint> out;
        out.reserve(27);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                for (int k = 0; k < 3; ++k) {
                    out.push_back({p[i], p[j], p[k], w[i] * w[j] * w[k]});
                }
            }
        }
        return out;
    }();
    return q;
}

static const std::vector<VolumeQuadraturePoint>& tet_p1_centroid_quadrature() {
    // Reference tet: r>=0, s>=0, t>=0, r+s+t<=1.  Reference volume = 1/6.
    // The physical volume integral is sum_q weight * |detJ| * f(x_q).
    static const std::vector<VolumeQuadraturePoint> q = {
        {0.25, 0.25, 0.25, 1.0 / 6.0}
    };
    return q;
}

static const std::vector<VolumeQuadraturePoint>& tet_p1_error_quadrature() {
    // Frozen MMS diagnostic rule: symmetric four-point tetra quadrature,
    // degree 2.  Reference volume = 1/6, so each reference weight is 1/24.
    static const std::vector<VolumeQuadraturePoint> q = [] {
        const double aa = 0.5854101966249685;
        const double bb = 0.1381966011250105;
        return std::vector<VolumeQuadraturePoint>{
            // Stored as (r,s,t) = (lambda1, lambda2, lambda3).
            {bb, bb, bb, 1.0 / 24.0},
            {aa, bb, bb, 1.0 / 24.0},
            {bb, aa, bb, 1.0 / 24.0},
            {bb, bb, aa, 1.0 / 24.0}
        };
    }();
    return q;
}

static const std::vector<FaceQuadraturePoint>& triangle_face_quadrature_1point() {
    // Unit reference triangle area = 1/2.
    static const std::vector<FaceQuadraturePoint> q = {
        {1.0 / 3.0, 1.0 / 3.0, 0.5}
    };
    return q;
}

static const std::vector<FaceQuadraturePoint>& quad_face_quadrature_2x2() {
    static const std::vector<FaceQuadraturePoint> q = [] {
        const double a = 1.0 / std::sqrt(3.0);
        const double p[2] = {-a, a};
        std::vector<FaceQuadraturePoint> out;
        out.reserve(4);
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                out.push_back({p[i], p[j], 1.0});
            }
        }
        return out;
    }();
    return q;
}


// ============================================================================
// SECTION 04A EXT: generalized selectable quadrature
// ============================================================================
//
// User-facing knobs:
//   -quadOrder         2|3|4    sets volume, face, and error quadrature
//   -volumeQuadOrder  2|3|4
//   -faceQuadOrder    2|3|4
//   -errorQuadOrder   2|3|4
//
// Env fallbacks:
//   MEMOIRS_QUAD_ORDER
//   MEMOIRS_VOLUME_QUAD_ORDER
//   MEMOIRS_FACE_QUAD_ORDER
//   MEMOIRS_ERROR_QUAD_ORDER
//
// Tensor hex/quad exactness:
//   n=2: exact degree 3 per coordinate
//   n=3: exact degree 5 per coordinate
//   n=4: exact degree 7 per coordinate
//
// Tet/tri use Duffy tensor-product constructions with the same n knob.
// ============================================================================

struct MemoirsQuadratureSpec {
    int volumeOrder = 2;
    int faceOrder = 2;
    int errorOrder = 3;
};

static int memoirs_clamp_quad_order(int q) {
    if (q < 2) return 2;
    if (q > 4) return 4;
    return q;
}

static int& memoirs_volume_quadrature_order_ref() {
    static int q = 2;
    return q;
}

static int& memoirs_face_quadrature_order_ref() {
    static int q = 2;
    return q;
}

static int& memoirs_error_quadrature_order_ref() {
    static int q = 3;
    return q;
}

static int memoirs_volume_quadrature_order() {
    return memoirs_volume_quadrature_order_ref();
}

static int memoirs_face_quadrature_order() {
    return memoirs_face_quadrature_order_ref();
}

static int memoirs_error_quadrature_order() {
    return memoirs_error_quadrature_order_ref();
}

static void set_memoirs_quadrature_orders(int volumeOrder, int faceOrder, int errorOrder) {
    memoirs_volume_quadrature_order_ref() = memoirs_clamp_quad_order(volumeOrder);
    memoirs_face_quadrature_order_ref() = memoirs_clamp_quad_order(faceOrder);
    memoirs_error_quadrature_order_ref() = memoirs_clamp_quad_order(errorOrder);
}

static int memoirs_parse_int_env_or(const char* name, int fallback) {
    const char* v = std::getenv(name);
    if (!v || !*v) return fallback;
    return std::atoi(v);
}

static MemoirsQuadratureSpec parse_memoirs_quadrature_spec_from_cli(int argc, char** argv) {
    MemoirsQuadratureSpec q;

    const int allEnv = memoirs_parse_int_env_or("MEMOIRS_QUAD_ORDER", -1);
    if (allEnv >= 0) {
        q.volumeOrder = allEnv;
        q.faceOrder = allEnv;
        q.errorOrder = allEnv;
    }

    q.volumeOrder = memoirs_parse_int_env_or("MEMOIRS_VOLUME_QUAD_ORDER", q.volumeOrder);
    q.faceOrder = memoirs_parse_int_env_or("MEMOIRS_FACE_QUAD_ORDER", q.faceOrder);
    q.errorOrder = memoirs_parse_int_env_or("MEMOIRS_ERROR_QUAD_ORDER", q.errorOrder);

    for (int i = 1; i + 1 < argc; ++i) {
        const std::string a = argv[i];

        if (a == "-quadOrder" || a == "-quadratureOrder" || a == "-gaussOrder") {
            const int v = std::atoi(argv[++i]);
            q.volumeOrder = v;
            q.faceOrder = v;
            q.errorOrder = v;
        } else if (a == "-volumeQuadOrder" || a == "-volQuadOrder") {
            q.volumeOrder = std::atoi(argv[++i]);
        } else if (a == "-faceQuadOrder") {
            q.faceOrder = std::atoi(argv[++i]);
        } else if (a == "-errorQuadOrder") {
            q.errorOrder = std::atoi(argv[++i]);
        }
    }

    q.volumeOrder = memoirs_clamp_quad_order(q.volumeOrder);
    q.faceOrder = memoirs_clamp_quad_order(q.faceOrder);
    q.errorOrder = memoirs_clamp_quad_order(q.errorOrder);

    set_memoirs_quadrature_orders(q.volumeOrder, q.faceOrder, q.errorOrder);
    return q;
}

struct MemoirsGaussPoint1D {
    double x = 0.0;
    double w = 0.0;
};

static std::vector<MemoirsGaussPoint1D> memoirs_gauss_legendre_1d(int n) {
    n = memoirs_clamp_quad_order(n);

    if (n == 2) {
        const double a = 1.0 / std::sqrt(3.0);
        return {{-a, 1.0}, {a, 1.0}};
    }

    if (n == 3) {
        const double a = std::sqrt(3.0 / 5.0);
        return {{-a, 5.0/9.0}, {0.0, 8.0/9.0}, {a, 5.0/9.0}};
    }

    const double a = std::sqrt((3.0 - 2.0*std::sqrt(6.0/5.0)) / 7.0);
    const double b = std::sqrt((3.0 + 2.0*std::sqrt(6.0/5.0)) / 7.0);
    const double wa = (18.0 + std::sqrt(30.0)) / 36.0;
    const double wb = (18.0 - std::sqrt(30.0)) / 36.0;
    return {{-b, wb}, {-a, wa}, {a, wa}, {b, wb}};
}

static std::vector<MemoirsGaussPoint1D> memoirs_gauss_legendre_01(int n) {
    auto r = memoirs_gauss_legendre_1d(n);
    for (auto& p : r) {
        p.x = 0.5 * (p.x + 1.0);
        p.w = 0.5 * p.w;
    }
    return r;
}

static std::vector<VolumeQuadraturePoint> make_hex_q1_volume_quadrature_tensor(int n) {
    const auto g = memoirs_gauss_legendre_1d(n);
    std::vector<VolumeQuadraturePoint> out;
    out.reserve(g.size() * g.size() * g.size());

    for (const auto& a : g) {
        for (const auto& b : g) {
            for (const auto& c : g) {
                VolumeQuadraturePoint q;
                q.xi = a.x;
                q.eta = b.x;
                q.zeta = c.x;
                q.weight = a.w * b.w * c.w;
                out.push_back(q);
            }
        }
    }

    return out;
}

static const std::vector<VolumeQuadraturePoint>& hex_q1_volume_quadrature_by_order(int n) {
    static const std::vector<VolumeQuadraturePoint> q2 = make_hex_q1_volume_quadrature_tensor(2);
    static const std::vector<VolumeQuadraturePoint> q3 = make_hex_q1_volume_quadrature_tensor(3);
    static const std::vector<VolumeQuadraturePoint> q4 = make_hex_q1_volume_quadrature_tensor(4);

    n = memoirs_clamp_quad_order(n);
    if (n == 2) return q2;
    if (n == 3) return q3;
    return q4;
}

static const std::vector<VolumeQuadraturePoint>& hex_q1_volume_quadrature_selected() {
    return hex_q1_volume_quadrature_by_order(memoirs_volume_quadrature_order());
}

static const std::vector<VolumeQuadraturePoint>& hex_q1_error_quadrature_selected() {
    return hex_q1_volume_quadrature_by_order(memoirs_error_quadrature_order());
}

static std::vector<FaceQuadraturePoint> make_quad_face_quadrature_tensor(int n) {
    const auto g = memoirs_gauss_legendre_1d(n);
    std::vector<FaceQuadraturePoint> out;
    out.reserve(g.size() * g.size());

    for (const auto& a : g) {
        for (const auto& b : g) {
            FaceQuadraturePoint q;
            q.r = a.x;
            q.s = b.x;
            q.weight = a.w * b.w;
            out.push_back(q);
        }
    }

    return out;
}

static const std::vector<FaceQuadraturePoint>& quad_face_quadrature_by_order(int n) {
    static const std::vector<FaceQuadraturePoint> q2 = make_quad_face_quadrature_tensor(2);
    static const std::vector<FaceQuadraturePoint> q3 = make_quad_face_quadrature_tensor(3);
    static const std::vector<FaceQuadraturePoint> q4 = make_quad_face_quadrature_tensor(4);

    n = memoirs_clamp_quad_order(n);
    if (n == 2) return q2;
    if (n == 3) return q3;
    return q4;
}

static const std::vector<FaceQuadraturePoint>& quad_face_quadrature_selected() {
    return quad_face_quadrature_by_order(memoirs_face_quadrature_order());
}

static std::vector<VolumeQuadraturePoint> make_tet_volume_quadrature_duffy(int n) {
    const auto g = memoirs_gauss_legendre_01(n);
    std::vector<VolumeQuadraturePoint> out;
    out.reserve(g.size() * g.size() * g.size());

    for (const auto& a : g) {
        for (const auto& b : g) {
            for (const auto& c : g) {
                const double xi   = a.x * (1.0 - b.x) * (1.0 - c.x);
                const double eta  = b.x * (1.0 - c.x);
                const double zeta = c.x;
                const double jac  = (1.0 - b.x) * (1.0 - c.x) * (1.0 - c.x);

                VolumeQuadraturePoint q;
                q.xi = xi;
                q.eta = eta;
                q.zeta = zeta;
                q.weight = a.w * b.w * c.w * jac;
                out.push_back(q);
            }
        }
    }

    return out;
}

static const std::vector<VolumeQuadraturePoint>& tet_volume_quadrature_by_order(int n) {
    static const std::vector<VolumeQuadraturePoint> q2 = make_tet_volume_quadrature_duffy(2);
    static const std::vector<VolumeQuadraturePoint> q3 = make_tet_volume_quadrature_duffy(3);
    static const std::vector<VolumeQuadraturePoint> q4 = make_tet_volume_quadrature_duffy(4);

    n = memoirs_clamp_quad_order(n);
    if (n == 2) return q2;
    if (n == 3) return q3;
    return q4;
}

static const std::vector<VolumeQuadraturePoint>& tet_volume_quadrature_selected() {
    return tet_volume_quadrature_by_order(memoirs_volume_quadrature_order());
}

static const std::vector<VolumeQuadraturePoint>& tet_volume_quadrature_selected_for_error() {
    return tet_volume_quadrature_by_order(memoirs_error_quadrature_order());
}

static std::vector<FaceQuadraturePoint> make_triangle_face_quadrature_duffy(int n) {
    const auto g = memoirs_gauss_legendre_01(n);
    std::vector<FaceQuadraturePoint> out;
    out.reserve(g.size() * g.size());

    for (const auto& a : g) {
        for (const auto& b : g) {
            const double r = a.x * (1.0 - b.x);
            const double s = b.x;
            const double jac = 1.0 - b.x;

            FaceQuadraturePoint q;
            q.r = r;
            q.s = s;
            q.weight = a.w * b.w * jac;
            out.push_back(q);
        }
    }

    return out;
}

static const std::vector<FaceQuadraturePoint>& triangle_face_quadrature_by_order(int n) {
    static const std::vector<FaceQuadraturePoint> q2 = make_triangle_face_quadrature_duffy(2);
    static const std::vector<FaceQuadraturePoint> q3 = make_triangle_face_quadrature_duffy(3);
    static const std::vector<FaceQuadraturePoint> q4 = make_triangle_face_quadrature_duffy(4);

    n = memoirs_clamp_quad_order(n);
    if (n == 2) return q2;
    if (n == 3) return q3;
    return q4;
}

static const std::vector<FaceQuadraturePoint>& triangle_face_quadrature_selected() {
    return triangle_face_quadrature_by_order(memoirs_face_quadrature_order());
}

static void print_memoirs_quadrature_orders(std::ostream& os) {
    os << "volumeQuadOrder   = " << memoirs_volume_quadrature_order() << "\n";
    os << "faceQuadOrder     = " << memoirs_face_quadrature_order() << "\n";
    os << "errorQuadOrder    = " << memoirs_error_quadrature_order() << "\n";
}

