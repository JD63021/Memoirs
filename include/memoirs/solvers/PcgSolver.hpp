#pragma once

#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace memoirs {
namespace solvers {

template <class Real>
struct PcgReport {
    int iterations = 0;
    Real initialResidual = Real(0);
    Real finalResidual = Real(0);
    Real finalRelativeResidual = Real(0);
    int converged = 0;
    int breakdown = 0;
};

template <class Real>
static Real dot(const std::vector<Real>& a, const std::vector<Real>& b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("dot size mismatch");
    }

    Real s = Real(0);
    for (std::size_t i = 0; i < a.size(); ++i) {
        s += a[i] * b[i];
    }
    return s;
}

template <class Real>
static Real norm2(const std::vector<Real>& a) {
    return std::sqrt(dot(a, a));
}

template <class Real>
static void axpy(Real alpha, const std::vector<Real>& x, std::vector<Real>& y) {
    if (x.size() != y.size()) {
        throw std::runtime_error("axpy size mismatch");
    }

    for (std::size_t i = 0; i < x.size(); ++i) {
        y[i] += alpha * x[i];
    }
}

template <class Real, class Operator, class Preconditioner>
PcgReport<Real> pcg_solve(
    const Operator& A,
    Preconditioner& M,
    const std::vector<Real>& b,
    std::vector<Real>& x,
    Real tol,
    int maxit,
    int printEvery = 0
) {
    const std::size_t n = b.size();

    if (x.size() != n) {
        x.assign(n, Real(0));
    }

    std::vector<Real> Axv;
    std::vector<Real> r(n, Real(0));
    std::vector<Real> z(n, Real(0));
    std::vector<Real> p(n, Real(0));
    std::vector<Real> Ap(n, Real(0));

    A.apply(x, Axv);

    for (std::size_t i = 0; i < n; ++i) {
        r[i] = b[i] - Axv[i];
    }

    const Real bnorm = std::max(norm2(b), Real(1));
    const Real r0 = norm2(r);

    PcgReport<Real> rep;
    rep.initialResidual = r0;
    rep.finalResidual = r0;
    rep.finalRelativeResidual = r0 / bnorm;

    if (rep.finalRelativeResidual <= tol) {
        rep.converged = 1;
        return rep;
    }

    M.apply(r, z);
    p = z;

    Real rz = dot(r, z);

    if (!(std::abs(rz) > Real(0))) {
        rep.breakdown = 1;
        return rep;
    }

    for (int it = 1; it <= maxit; ++it) {
        A.apply(p, Ap);

        const Real pAp = dot(p, Ap);
        if (!(pAp > Real(0))) {
            rep.iterations = it - 1;
            rep.breakdown = 1;
            return rep;
        }

        const Real alpha = rz / pAp;

        for (std::size_t i = 0; i < n; ++i) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }

        const Real rnorm = norm2(r);
        rep.iterations = it;
        rep.finalResidual = rnorm;
        rep.finalRelativeResidual = rnorm / bnorm;

        if (printEvery > 0 && (it == 1 || it % printEvery == 0)) {
            std::cout << "pcg it=" << it
                      << " rel=" << double(rep.finalRelativeResidual)
                      << "\n";
        }

        if (rep.finalRelativeResidual <= tol) {
            rep.converged = 1;
            return rep;
        }

        M.apply(r, z);

        const Real rzNew = dot(r, z);
        if (!(std::abs(rzNew) > Real(0))) {
            rep.breakdown = 1;
            return rep;
        }

        const Real beta = rzNew / rz;

        for (std::size_t i = 0; i < n; ++i) {
            p[i] = z[i] + beta * p[i];
        }

        rz = rzNew;
    }

    return rep;
}

} // namespace solvers
} // namespace memoirs
