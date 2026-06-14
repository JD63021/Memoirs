#pragma once

// ============================================================================
// SECTION 10: Scalar elliptic coefficient and boundary-condition specification
// ============================================================================
//
// This layer keeps operator choices out of the app:
//
//   coefficient:
//      constant, later cellwise / quadrature callback
//
//   boundary:
//      strong          CG strong Dirichlet or DG clamped trace
//      nitsche/weak    CG weak Dirichlet
//      sipg/weak       DG weak Dirichlet
//
// Current implementation is intentionally small:
//   - coefficient = constant or cellwise vector container
//   - MMS boundary values
//   - tet paths first
// ============================================================================

enum class ScalarCoeffMode {
    Constant,
    Cellwise
};

struct ScalarDiffusionCoefficient {
    ScalarCoeffMode mode = ScalarCoeffMode::Constant;
    Real constant = Real(1);
    std::vector<Real> cellValue;

    static ScalarDiffusionCoefficient constant_coeff(double kappa) {
        ScalarDiffusionCoefficient c;
        c.mode = ScalarCoeffMode::Constant;
        c.constant = Real(kappa);
        return c;
    }

    static ScalarDiffusionCoefficient cellwise_coeff(std::vector<Real> values) {
        ScalarDiffusionCoefficient c;
        c.mode = ScalarCoeffMode::Cellwise;
        c.cellValue = std::move(values);
        return c;
    }

    Real value_in_cell(int cell, const Vec3&) const {
        if (mode == ScalarCoeffMode::Constant) return constant;

        if (cell < 0 || cell >= (int)cellValue.size()) {
            throw std::runtime_error("ScalarDiffusionCoefficient cell index out of range.");
        }

        return cellValue[cell];
    }

    Real face_value(int owner, int neighbour) const {
        if (mode == ScalarCoeffMode::Constant) return constant;

        const Real ko = cellValue.at(owner);
        if (neighbour < 0) return ko;

        const Real kn = cellValue.at(neighbour);

        // Harmonic average is safer for discontinuous diffusion.
        const Real denom = ko + kn;
        if (std::abs(double(denom)) <= 1.0e-300) return Real(0);

        return Real(2) * ko * kn / denom;
    }
};

enum class ScalarBoundaryMode {
    Strong,
    WeakNitsche,
    WeakSipg
};

struct ScalarBoundarySpec {
    ScalarBoundaryMode mode = ScalarBoundaryMode::Strong;
    std::string valueMode = "mms";
    double penaltySigma = 10.0;
};

struct ScalarEllipticSpec {
    ScalarDiffusionCoefficient diffusion = ScalarDiffusionCoefficient::constant_coeff(1.0);
    ScalarBoundarySpec boundary;
    std::string sourceMode = "mms";
};

static inline std::string scalar_bc_mode_name(ScalarBoundaryMode m) {
    switch (m) {
        case ScalarBoundaryMode::Strong: return "strong";
        case ScalarBoundaryMode::WeakNitsche: return "weak_nitsche";
        case ScalarBoundaryMode::WeakSipg: return "weak_sipg";
    }
    return "unknown";
}

static inline ScalarBoundaryMode parse_scalar_bc_mode(const std::string& sIn) {
    std::string s = lower_copy(sIn);

    if (s == "strong" || s == "clamp" || s == "clamped") {
        return ScalarBoundaryMode::Strong;
    }

    if (s == "weak" || s == "nitsche" || s == "weak_nitsche" || s == "cg_weak") {
        return ScalarBoundaryMode::WeakNitsche;
    }

    if (s == "sipg" || s == "weak_sipg" || s == "dg_weak") {
        return ScalarBoundaryMode::WeakSipg;
    }

    throw std::runtime_error("Unknown scalar BC mode: " + sIn);
}

static inline ScalarEllipticSpec parse_scalar_elliptic_spec_from_cli(
    int argc,
    char** argv,
    const std::string& defaultBc,
    double defaultPenalty
) {
    ScalarEllipticSpec spec;
    spec.diffusion = ScalarDiffusionCoefficient::constant_coeff(1.0);
    spec.boundary.mode = parse_scalar_bc_mode(defaultBc);
    spec.boundary.penaltySigma = defaultPenalty;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];

        auto need = [&](const std::string& key) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("Missing value after " + key);
            return std::string(argv[++i]);
        };

        if (a == "-bc" || a == "-bcMode" || a == "-boundaryMode") {
            spec.boundary.mode = parse_scalar_bc_mode(need(a));
        } else if (a == "-penaltySigma" || a == "-sipgSigma" || a == "-nitscheSigma" || a == "-sigma") {
            spec.boundary.penaltySigma = std::stod(need(a));
        } else if (a == "-kappa" || a == "-diffusion" || a == "-viscosity" || a == "-nu") {
            spec.diffusion = ScalarDiffusionCoefficient::constant_coeff(std::stod(need(a)));
        }
    }

    return spec;
}
