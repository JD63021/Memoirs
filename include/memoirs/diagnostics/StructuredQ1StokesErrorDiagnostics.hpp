#pragma once

#include "memoirs/structured/StructuredGrid3D.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

namespace memoirs {
namespace diagnostics {

template <class Real>
struct StructuredQ1StokesErrors {
    double velocityL2 = 0.0;
    double pressureL2 = 0.0;
    double pressureL2MeanShifted = 0.0;
    double pressureMeanShift = 0.0;
};

inline void stokes_exact_mms(double x,
                             double y,
                             double z,
                             double& u,
                             double& v,
                             double& w,
                             double& p) {
    const double pi = std::acos(-1.0);

    const double sx = std::sin(pi * x);
    const double sy = std::sin(pi * y);
    const double sz = std::sin(pi * z);

    const double cx = std::cos(pi * x);
    const double cy = std::cos(pi * y);

    u =  2.0 * pi * sx * sx * sy * cy * sz * sz;
    v = -2.0 * pi * sx * cx * sy * sy * sz * sz;
    w =  0.0;
    p =  cx * sy * sz;
}

template <class Real>
StructuredQ1StokesErrors<Real> structured_q1_l2_error_stokes_mms(
    const memoirs::structured::StructuredGrid3D& g,
    const std::vector<Real>& x
) {
    const int n = g.n_nodes();

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

    const double jac = g.hx() * g.hy() * g.hz() / 8.0;

    double int_pdiff = 0.0;
    double volume = 0.0;

    // First pass: pressure mean shift, matching the MATLAB diagnostic style.
    for (int ez = 0; ez < g.nz - 1; ++ez) {
        for (int ey = 0; ey < g.ny - 1; ++ey) {
            for (int ex = 0; ex < g.nx - 1; ++ex) {
                double pn[8];

                for (int a = 0; a < 8; ++a) {
                    const int ix = a & 1;
                    const int iy = (a >> 1) & 1;
                    const int iz = (a >> 2) & 1;
                    const int q = g.id(ex + ix, ey + iy, ez + iz);
                    pn[a] = double(x[static_cast<std::size_t>(3 * n + q)]);
                }

                for (int iq = 0; iq < 4; ++iq) {
                    const double xi = qp[iq];
                    const double wx = qw[iq];
                    const double Nx[2] = {0.5 * (1.0 - xi), 0.5 * (1.0 + xi)};
                    const double xx = (double(ex) + 0.5 * (1.0 + xi)) * g.hx();

                    for (int jq = 0; jq < 4; ++jq) {
                        const double eta = qp[jq];
                        const double wy = qw[jq];
                        const double Ny[2] = {0.5 * (1.0 - eta), 0.5 * (1.0 + eta)};
                        const double yy = (double(ey) + 0.5 * (1.0 + eta)) * g.hy();

                        for (int kq = 0; kq < 4; ++kq) {
                            const double zeta = qp[kq];
                            const double wz = qw[kq];
                            const double Nz[2] = {0.5 * (1.0 - zeta), 0.5 * (1.0 + zeta)};
                            const double zz = (double(ez) + 0.5 * (1.0 + zeta)) * g.hz();

                            double ph = 0.0;
                            for (int a = 0; a < 8; ++a) {
                                const int ix = a & 1;
                                const int iy = (a >> 1) & 1;
                                const int iz = (a >> 2) & 1;
                                const double Na = Nx[ix] * Ny[iy] * Nz[iz];
                                ph += Na * pn[a];
                            }

                            double ue, ve, we, pe;
                            stokes_exact_mms(xx, yy, zz, ue, ve, we, pe);

                            const double wq = wx * wy * wz * jac;
                            int_pdiff += wq * (ph - pe);
                            volume += wq;
                        }
                    }
                }
            }
        }
    }

    const double p_shift = int_pdiff / volume;

    double ev2 = 0.0;
    double ep2 = 0.0;
    double ep2_shifted = 0.0;

    for (int ez = 0; ez < g.nz - 1; ++ez) {
        for (int ey = 0; ey < g.ny - 1; ++ey) {
            for (int ex = 0; ex < g.nx - 1; ++ex) {
                double un[8], vn[8], wn[8], pn[8];

                for (int a = 0; a < 8; ++a) {
                    const int ix = a & 1;
                    const int iy = (a >> 1) & 1;
                    const int iz = (a >> 2) & 1;
                    const int q = g.id(ex + ix, ey + iy, ez + iz);

                    un[a] = double(x[static_cast<std::size_t>(q)]);
                    vn[a] = double(x[static_cast<std::size_t>(n + q)]);
                    wn[a] = double(x[static_cast<std::size_t>(2 * n + q)]);
                    pn[a] = double(x[static_cast<std::size_t>(3 * n + q)]);
                }

                for (int iq = 0; iq < 4; ++iq) {
                    const double xi = qp[iq];
                    const double wx = qw[iq];
                    const double Nx[2] = {0.5 * (1.0 - xi), 0.5 * (1.0 + xi)};
                    const double xx = (double(ex) + 0.5 * (1.0 + xi)) * g.hx();

                    for (int jq = 0; jq < 4; ++jq) {
                        const double eta = qp[jq];
                        const double wy = qw[jq];
                        const double Ny[2] = {0.5 * (1.0 - eta), 0.5 * (1.0 + eta)};
                        const double yy = (double(ey) + 0.5 * (1.0 + eta)) * g.hy();

                        for (int kq = 0; kq < 4; ++kq) {
                            const double zeta = qp[kq];
                            const double wz = qw[kq];
                            const double Nz[2] = {0.5 * (1.0 - zeta), 0.5 * (1.0 + zeta)};
                            const double zz = (double(ez) + 0.5 * (1.0 + zeta)) * g.hz();

                            double uh = 0.0;
                            double vh = 0.0;
                            double wh = 0.0;
                            double ph = 0.0;

                            for (int a = 0; a < 8; ++a) {
                                const int ix = a & 1;
                                const int iy = (a >> 1) & 1;
                                const int iz = (a >> 2) & 1;
                                const double Na = Nx[ix] * Ny[iy] * Nz[iz];

                                uh += Na * un[a];
                                vh += Na * vn[a];
                                wh += Na * wn[a];
                                ph += Na * pn[a];
                            }

                            double ue, ve, we, pe;
                            stokes_exact_mms(xx, yy, zz, ue, ve, we, pe);

                            const double du = uh - ue;
                            const double dv = vh - ve;
                            const double dw = wh - we;
                            const double dp = ph - pe;
                            const double dps = (ph - p_shift) - pe;

                            const double wq = wx * wy * wz * jac;
                            ev2 += wq * (du * du + dv * dv + dw * dw);
                            ep2 += wq * dp * dp;
                            ep2_shifted += wq * dps * dps;
                        }
                    }
                }
            }
        }
    }

    StructuredQ1StokesErrors<Real> e;
    e.velocityL2 = std::sqrt(ev2);
    e.pressureL2 = std::sqrt(ep2);
    e.pressureL2MeanShifted = std::sqrt(ep2_shifted);
    e.pressureMeanShift = p_shift;
    return e;
}

} // namespace diagnostics
} // namespace memoirs
