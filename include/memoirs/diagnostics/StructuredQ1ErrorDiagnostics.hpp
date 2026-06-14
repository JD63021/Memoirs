#pragma once

#include "memoirs/structured/StructuredGrid3D.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

namespace memoirs {
namespace diagnostics {

template <class Real>
double structured_q1_l2_error_transport_sin(
    const memoirs::structured::StructuredGrid3D& g,
    const std::vector<Real>& uh
) {
    const double pi = std::acos(-1.0);

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

    double err2 = 0.0;

    for (int ez = 0; ez < g.nz - 1; ++ez) {
        for (int ey = 0; ey < g.ny - 1; ++ey) {
            for (int ex = 0; ex < g.nx - 1; ++ex) {
                double uNode[8];

                for (int a = 0; a < 8; ++a) {
                    const int ix = a & 1;
                    const int iy = (a >> 1) & 1;
                    const int iz = (a >> 2) & 1;

                    const int q = g.id(ex + ix, ey + iy, ez + iz);
                    uNode[a] = double(uh[static_cast<std::size_t>(q)]);
                }

                for (int iq = 0; iq < 4; ++iq) {
                    const double xi = qp[iq];
                    const double wx = qw[iq];

                    const double Nx[2] = {
                        0.5 * (1.0 - xi),
                        0.5 * (1.0 + xi)
                    };

                    const double x =
                        (double(ex) + 0.5 * (1.0 + xi)) * g.hx();

                    for (int jq = 0; jq < 4; ++jq) {
                        const double eta = qp[jq];
                        const double wy = qw[jq];

                        const double Ny[2] = {
                            0.5 * (1.0 - eta),
                            0.5 * (1.0 + eta)
                        };

                        const double y =
                            (double(ey) + 0.5 * (1.0 + eta)) * g.hy();

                        for (int kq = 0; kq < 4; ++kq) {
                            const double zeta = qp[kq];
                            const double wz = qw[kq];

                            const double Nz[2] = {
                                0.5 * (1.0 - zeta),
                                0.5 * (1.0 + zeta)
                            };

                            const double z =
                                (double(ez) + 0.5 * (1.0 + zeta)) * g.hz();

                            double uhq = 0.0;

                            for (int a = 0; a < 8; ++a) {
                                const int ix = a & 1;
                                const int iy = (a >> 1) & 1;
                                const int iz = (a >> 2) & 1;

                                const double Na = Nx[ix] * Ny[iy] * Nz[iz];
                                uhq += Na * uNode[a];
                            }

                            const double uex =
                                std::sin(pi * x) *
                                std::sin(pi * y) *
                                std::sin(pi * z);

                            const double e = uhq - uex;
                            err2 += wx * wy * wz * jac * e * e;
                        }
                    }
                }
            }
        }
    }

    return std::sqrt(err2);
}

} // namespace diagnostics
} // namespace memoirs
