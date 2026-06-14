#pragma once

// Auto-split from frozen patch014 one-shot source.
// This header is intentionally included in order by apps/poisson_mms.cpp.

// ============================================================================
// SECTION 7: MMS source/exact solution
// ============================================================================

static double mms_exact_value(const Vec3& p, const std::string& mms) {
    std::string m = lower_copy(mms);
    if (m == "linear") {
        return 1.0 + p.x + 2.0*p.y + 3.0*p.z;
    }
    if (m == "sin") {
        return std::sin(kPi*p.x) * std::sin(kPi*p.y) * std::sin(kPi*p.z);
    }
    throw std::runtime_error("Unsupported -mms: " + mms);
}

static double mms_rhs_value(const Vec3& p, const std::string& mms) {
    std::string m = lower_copy(mms);
    if (m == "linear") {
        return 0.0; // -Delta(linear) = 0
    }
    if (m == "sin") {
        const double u = std::sin(kPi*p.x) * std::sin(kPi*p.y) * std::sin(kPi*p.z);
        return 3.0 * kPi * kPi * u; // -Delta u = 3*pi^2*u
    }
    throw std::runtime_error("Unsupported -mms: " + mms);
}

static Vec3 mms_exact_grad(const Vec3& p, const std::string& mms) {
    std::string m = lower_copy(mms);
    if (m == "linear") {
        return {1.0, 2.0, 3.0};
    }
    if (m == "sin") {
        return {
            kPi * std::cos(kPi*p.x) * std::sin(kPi*p.y) * std::sin(kPi*p.z),
            kPi * std::sin(kPi*p.x) * std::cos(kPi*p.y) * std::sin(kPi*p.z),
            kPi * std::sin(kPi*p.x) * std::sin(kPi*p.y) * std::cos(kPi*p.z)
        };
    }
    throw std::runtime_error("Unsupported -mms: " + mms);
}

struct ErrorNorms {
    double L2 = 0.0;
    double H1Semi = 0.0;
    double nodalL2 = 0.0;
    double nodalMax = 0.0;
};
