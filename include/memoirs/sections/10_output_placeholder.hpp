#pragma once

// ============================================================================
// SECTION 10: Error diagnostics / VTU
// ============================================================================

static inline void q1q1_write_vtu_solution(
    const PolyMesh& mesh,
    const LinearCgDofMap& dm,
    const std::vector<Real>& x,
    const std::string& path
) {
    if ((int)x.size() != 4 * dm.nDofs) {
        throw std::runtime_error("q1q1_write_vtu_solution requires node-major [ux,uy,uz,p] vector.");
    }

    auto row4 = [](int node, int field) -> int {
        return 4 * node + field;
    };

    std::ofstream f(path);
    if (!f) throw std::runtime_error("Could not open VTU output: " + path);

    const int nPoints = (int)mesh.points.size();
    const int nCells = (int)mesh.cells.size();

    f << "<?xml version=\"1.0\"?>\n";
    f << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    f << "<UnstructuredGrid>\n";
    f << "<Piece NumberOfPoints=\"" << nPoints << "\" NumberOfCells=\"" << nCells << "\">\n";

    f << "<PointData Vectors=\"U\" Scalars=\"p\">\n";

    f << "<DataArray type=\"Float64\" Name=\"U\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    f << std::setprecision(16);
    for (int i = 0; i < nPoints; ++i) {
        f << double(x[row4(i,0)]) << " "
          << double(x[row4(i,1)]) << " "
          << double(x[row4(i,2)]) << "\n";
    }
    f << "</DataArray>\n";

    f << "<DataArray type=\"Float64\" Name=\"p\" NumberOfComponents=\"1\" format=\"ascii\">\n";
    for (int i = 0; i < nPoints; ++i) {
        f << double(x[row4(i,3)]) << "\n";
    }
    f << "</DataArray>\n";

    f << "</PointData>\n";

    f << "<Points>\n";
    f << "<DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (const Vec3& p : mesh.points) {
        f << p.x << " " << p.y << " " << p.z << "\n";
    }
    f << "</DataArray>\n";
    f << "</Points>\n";

    f << "<Cells>\n";

    f << "<DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
    for (const Cell& c : mesh.cells) {
        const auto hv = ordered_hex_vertices_axis_aligned(mesh, c);
        for (int a = 0; a < 8; ++a) f << hv[a] << (a == 7 ? '\n' : ' ');
    }
    f << "</DataArray>\n";

    f << "<DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
    for (int c = 0; c < nCells; ++c) f << 8 * (c + 1) << "\n";
    f << "</DataArray>\n";

    f << "<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    for (int c = 0; c < nCells; ++c) f << 12 << "\n"; // VTK_HEXAHEDRON
    f << "</DataArray>\n";

    f << "</Cells>\n";
    f << "</Piece>\n";
    f << "</UnstructuredGrid>\n";
    f << "</VTKFile>\n";
}
