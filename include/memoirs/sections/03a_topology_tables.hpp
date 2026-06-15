#pragma once

// ============================================================================
// SECTION 03A: Global topology tables for higher-order FE spaces
// ============================================================================
// Purpose:
//   - Provide canonical global edge ids for CG P2 / Q2 edge dofs.
//   - Provide canonical global face ids for Q2 face-center dofs and H(div)
//     moment orientation work.
//   - Keep this purely topological and additive; current low-order apps do not
//     depend on these tables for assembly yet.
//
// Supported mesh families today:
//   - all-tet OpenFOAM polyMesh: triangular faces, 4 vertices/cell
//   - blockMesh-style hex polyMesh: quad faces, 8 vertices/cell
// ============================================================================

#include <array>
#include <map>
#include <set>
#include <sstream>
#include <vector>

struct MemoirsEdgeKey {
    int a = -1;
    int b = -1;

    MemoirsEdgeKey() = default;
    MemoirsEdgeKey(int i, int j) {
        if (i <= j) { a = i; b = j; }
        else        { a = j; b = i; }
    }

    bool operator<(const MemoirsEdgeKey& o) const {
        if (a != o.a) return a < o.a;
        return b < o.b;
    }
};

struct MemoirsTopologyTables {
    std::vector<std::array<int,2>> edgeVerts;
    std::vector<std::vector<int>> faceEdges;
    std::vector<std::vector<int>> cellEdges;
    std::vector<std::vector<int>> cellFaces;

    std::map<MemoirsEdgeKey,int> edgeId;
};

static int memoirs_topology_get_or_add_edge(
    MemoirsTopologyTables& topo,
    int va,
    int vb
) {
    MemoirsEdgeKey key(va, vb);
    auto it = topo.edgeId.find(key);
    if (it != topo.edgeId.end()) return it->second;

    const int id = (int)topo.edgeVerts.size();
    topo.edgeId[key] = id;
    topo.edgeVerts.push_back({key.a, key.b});
    return id;
}

static MemoirsTopologyTables build_memoirs_topology_tables(const PolyMesh& m) {
    MemoirsTopologyTables topo;
    topo.faceEdges.resize(m.faces.size());
    topo.cellEdges.resize(m.cells.size());
    topo.cellFaces.resize(m.cells.size());

    // Build global edge ids by walking the cyclic vertices of every face.
    // This is robust for triangular tet faces and quadrilateral hex faces.
    for (int f = 0; f < (int)m.faces.size(); ++f) {
        const auto& fv = m.faces[f].verts;
        if (fv.size() < 2) {
            throw std::runtime_error("Topology table builder found a face with <2 vertices.");
        }

        auto& fe = topo.faceEdges[f];
        fe.reserve(fv.size());
        for (int i = 0; i < (int)fv.size(); ++i) {
            const int a = fv[i];
            const int b = fv[(i + 1) % (int)fv.size()];
            fe.push_back(memoirs_topology_get_or_add_edge(topo, a, b));
        }
    }

    // Cell -> faces and cell -> unique edges.  OpenFOAM gives cell faces in
    // owner/neighbour form; the PolyMesh reader has already reconstructed this.
    for (int c = 0; c < (int)m.cells.size(); ++c) {
        topo.cellFaces[c] = m.cells[c].faces;
        std::set<int> uniqueEdges;
        for (int f : m.cells[c].faces) {
            if (f < 0 || f >= (int)topo.faceEdges.size()) {
                throw std::runtime_error("Topology table builder found a cell face id out of range.");
            }
            for (int e : topo.faceEdges[f]) uniqueEdges.insert(e);
        }
        topo.cellEdges[c].assign(uniqueEdges.begin(), uniqueEdges.end());
    }

    return topo;
}

static Vec3 memoirs_edge_midpoint(const PolyMesh& m, const std::array<int,2>& ev) {
    const Vec3& a = m.points[ev[0]];
    const Vec3& b = m.points[ev[1]];
    return {0.5*(a.x + b.x), 0.5*(a.y + b.y), 0.5*(a.z + b.z)};
}

static void probe_memoirs_topology_tables(const PolyMesh& m) {
    const auto topo = build_memoirs_topology_tables(m);

    std::map<int,int> faceEdgeHist;
    for (const auto& fe : topo.faceEdges) faceEdgeHist[(int)fe.size()]++;

    std::map<int,int> cellEdgeHist;
    for (const auto& ce : topo.cellEdges) cellEdgeHist[(int)ce.size()]++;

    std::map<int,int> cellFaceHist;
    for (const auto& cf : topo.cellFaces) cellFaceHist[(int)cf.size()]++;

    int nBoundaryFaces = (int)m.faces.size() - (int)m.neighbour.size();
    int nInteriorFaces = (int)m.neighbour.size();

    int nBadTetEdges = 0;
    int nBadHexEdges = 0;
    for (int c = 0; c < (int)m.cells.size(); ++c) {
        const int nv = (int)m.cells[c].verts.size();
        const int nf = (int)m.cells[c].faces.size();
        const int ne = (int)topo.cellEdges[c].size();
        if (nv == 4 && nf == 4 && ne != 6) ++nBadTetEdges;
        if (nv == 8 && nf == 6 && ne != 12) ++nBadHexEdges;
    }

    double minLen = std::numeric_limits<double>::infinity();
    double maxLen = 0.0;
    double sumLen = 0.0;
    for (const auto& ev : topo.edgeVerts) {
        const Vec3 d = m.points[ev[1]] - m.points[ev[0]];
        const double L = std::sqrt(dot3(d, d));
        minLen = std::min(minLen, L);
        maxLen = std::max(maxLen, L);
        sumLen += L;
    }
    const double meanLen = topo.edgeVerts.empty() ? 0.0 : sumLen / (double)topo.edgeVerts.size();

    std::cout << "------------- topology probe ---------------\n";
    std::cout << "nTopologyEdges             = " << topo.edgeVerts.size() << "\n";
    std::cout << "nTopologyFaces             = " << m.faces.size() << "\n";
    std::cout << "nInteriorFaces             = " << nInteriorFaces << "\n";
    std::cout << "nBoundaryFaces             = " << nBoundaryFaces << "\n";

    std::cout << "face edge histogram:\n";
    for (auto [k,v] : faceEdgeHist) {
        std::cout << "  nEdgesPerFace=" << k << " count=" << v << "\n";
    }

    std::cout << "cell edge histogram:\n";
    for (auto [k,v] : cellEdgeHist) {
        std::cout << "  nEdgesPerCell=" << k << " count=" << v << "\n";
    }

    std::cout << "cell face histogram:\n";
    for (auto [k,v] : cellFaceHist) {
        std::cout << "  nFacesPerCell=" << k << " count=" << v << "\n";
    }

    std::cout << "badTetEdgeCount            = " << nBadTetEdges << "\n";
    std::cout << "badHexEdgeCount            = " << nBadHexEdges << "\n";
    std::cout << "edgeLength min/mean/max    = "
              << std::setprecision(16) << minLen << " / " << meanLen << " / " << maxLen << "\n";
    std::cout << "topologyStatus             = "
              << ((nBadTetEdges == 0 && nBadHexEdges == 0) ? "ok" : "check") << "\n";
    std::cout << "--------------------------------------------\n";
}
