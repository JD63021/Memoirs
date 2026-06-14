#pragma once

#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>

namespace memoirs {
namespace structured {

struct StructuredGrid3D {
    int nx = 0; // nodes in x
    int ny = 0; // nodes in y
    int nz = 0; // nodes in z

    double lx = 1.0;
    double ly = 1.0;
    double lz = 1.0;

    StructuredGrid3D() = default;

    StructuredGrid3D(int nx_, int ny_, int nz_,
                     double lx_ = 1.0, double ly_ = 1.0, double lz_ = 1.0)
        : nx(nx_), ny(ny_), nz(nz_), lx(lx_), ly(ly_), lz(lz_) {
        validate();
    }

    void validate() const {
        if (nx < 2 || ny < 2 || nz < 2) {
            throw std::runtime_error("StructuredGrid3D requires at least 2 nodes per direction");
        }
        if (!(lx > 0.0 && ly > 0.0 && lz > 0.0)) {
            throw std::runtime_error("StructuredGrid3D requires positive lengths");
        }
    }

    int ex() const { return nx - 1; }
    int ey() const { return ny - 1; }
    int ez() const { return nz - 1; }

    double hx() const { return lx / double(nx - 1); }
    double hy() const { return ly / double(ny - 1); }
    double hz() const { return lz / double(nz - 1); }

    int n_nodes() const { return nx * ny * nz; }

    std::size_t size() const {
        return static_cast<std::size_t>(n_nodes());
    }

    int id(int i, int j, int k) const {
        return (k * ny + j) * nx + i;
    }

    bool is_boundary(int i, int j, int k) const {
        return i == 0 || j == 0 || k == 0 ||
               i == nx - 1 || j == ny - 1 || k == nz - 1;
    }

    bool is_boundary_id(int q) const {
        const int i = q % nx;
        const int tmp = q / nx;
        const int j = tmp % ny;
        const int k = tmp / ny;
        return is_boundary(i, j, k);
    }

    double x(int i) const { return hx() * double(i); }
    double y(int j) const { return hy() * double(j); }
    double z(int k) const { return hz() * double(k); }

    bool can_coarsen_by_2() const {
        return ((nx - 1) % 2 == 0) &&
               ((ny - 1) % 2 == 0) &&
               ((nz - 1) % 2 == 0) &&
               nx > 3 && ny > 3 && nz > 3;
    }

    StructuredGrid3D coarsened_by_2() const {
        if (!can_coarsen_by_2()) {
            std::ostringstream oss;
            oss << "Cannot coarsen grid by 2: nodes="
                << nx << "x" << ny << "x" << nz;
            throw std::runtime_error(oss.str());
        }

        return StructuredGrid3D((nx - 1) / 2 + 1,
                                (ny - 1) / 2 + 1,
                                (nz - 1) / 2 + 1,
                                lx, ly, lz);
    }

    std::string label() const {
        std::ostringstream oss;
        oss << nx << "x" << ny << "x" << nz
            << " nodes, elements "
            << ex() << "x" << ey() << "x" << ez();
        return oss.str();
    }
};

} // namespace structured
} // namespace memoirs
