#pragma once

// Auto-split from frozen patch014 one-shot source.
// This header is intentionally included in order by apps/poisson_mms.cpp.

// ============================================================================
// SECTION 5: CG dof map
// ============================================================================
//
// Current scope:
//   - CG linear spaces only.
//   - Hex blockMesh Q1 and tet mesh P1.
//   - For both, global dof id = polyMesh point id.
//
// Later modular extraction:
//   this section becomes fem/dof_map_cg.{h,cpp}
// ============================================================================

struct MeshCellCounts {
    int nHex = 0;
    int nTet = 0;
    int nOther = 0;
};

static MeshCellCounts count_cell_types(const PolyMesh& m) {
    MeshCellCounts cc;
    for (const auto& c : m.cells) {
        const int nf = (int)c.faces.size();
        const int nv = (int)c.verts.size();
        if (nf == 6 && nv == 8) cc.nHex++;
        else if (nf == 4 && nv == 4) cc.nTet++;
        else cc.nOther++;
    }
    return cc;
}

struct LinearCgDofMap {
    std::string resolvedSpace;
    std::string cellFamily;
    int nDofs = 0;
    int nBoundaryDofs = 0;
    int nInteriorDofs = 0;

    // For CG linear space, pointToDof[p] = p.
    std::vector<int> pointToDof;

    // For now, cellDofs are the cell vertex point ids.
    // Note: sorted vertex ids are enough for dof-count diagnostics.
    // Later assembly will introduce ordered cell vertices per reference element.
    std::vector<std::vector<int>> cellDofs;

    std::set<int> boundaryDofs;
    std::map<std::string, std::set<int>> patchDofs;

    // Coordinates for all scalar CG dofs. For linear spaces this is a copy
    // of polyMesh points. For cg_tet_p2, edge-midpoint dofs are appended.
    std::vector<Vec3> dofCoordinates;
};

static std::string lower_copy(std::string s) {
    for (char& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}


static Vec3 cg_dof_coordinate(const PolyMesh& m, const LinearCgDofMap& dm, int dof) {
    if (!dm.dofCoordinates.empty()) {
        if (dof < 0 || dof >= (int)dm.dofCoordinates.size()) {
            throw std::runtime_error("CG dof coordinate index out of range.");
        }
        return dm.dofCoordinates[dof];
    }
    if (dof < 0 || dof >= (int)m.points.size()) {
        throw std::runtime_error("CG linear dof coordinate index out of range.");
    }
    return m.points[dof];
}

static const int MEMOIRS_TET_P2_EDGE_VERTS[6][2] = {
    {0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3}
};

static int memoirs_find_edge_id_or_throw(
    const MemoirsTopologyTables& topo,
    int va,
    int vb
) {
    auto it = topo.edgeId.find(MemoirsEdgeKey(va, vb));
    if (it == topo.edgeId.end()) {
        throw std::runtime_error("Could not find canonical edge id for tet P2 dof map.");
    }
    return it->second;
}

static LinearCgDofMap build_linear_cg_dof_map(const PolyMesh& m, const std::string& requestedSpace) {
    MeshCellCounts cc = count_cell_types(m);
    const bool allHex = (cc.nHex == (int)m.cells.size());
    const bool allTet = (cc.nTet == (int)m.cells.size());

    std::string req = lower_copy(requestedSpace);

    LinearCgDofMap dm;
    bool useTetP2 = false;

    if (req == "auto" || req == "cg_linear" || req == "cg_q1" || req == "cg_p1") {
        if (allHex) {
            dm.resolvedSpace = "cg_hex_q1";
            dm.cellFamily = "hex";
        } else if (allTet) {
            dm.resolvedSpace = "cg_tet_p1";
            dm.cellFamily = "tet";
        } else {
            throw std::runtime_error("Cannot auto-resolve CG linear space on mixed/unsupported mesh.");
        }
    } else if (req == "cg_hex_q1") {
        if (!allHex) throw std::runtime_error("-space cg_hex_q1 requested, but mesh is not all hex.");
        dm.resolvedSpace = "cg_hex_q1";
        dm.cellFamily = "hex";
    } else if (req == "cg_tet_p1") {
        if (!allTet) throw std::runtime_error("-space cg_tet_p1 requested, but mesh is not all tet.");
        dm.resolvedSpace = "cg_tet_p1";
        dm.cellFamily = "tet";
    } else if (req == "cg_tet_p2" || req == "cg_p2" || req == "cg_tet_quadratic") {
        if (!allTet) throw std::runtime_error("-space cg_tet_p2 requested, but mesh is not all tet.");
        dm.resolvedSpace = "cg_tet_p2";
        dm.cellFamily = "tet";
        useTetP2 = true;
    } else {
        throw std::runtime_error("Unsupported -space for CG Poisson app: " + requestedSpace);
    }

    dm.pointToDof.resize(m.points.size());
    for (int p = 0; p < (int)m.points.size(); ++p) dm.pointToDof[p] = p;

    if (!useTetP2) {
        dm.nDofs = (int)m.points.size();
        dm.dofCoordinates = m.points;

        dm.cellDofs.resize(m.cells.size());
        for (int c = 0; c < (int)m.cells.size(); ++c) {
            dm.cellDofs[c] = m.cells[c].verts;
            std::sort(dm.cellDofs[c].begin(), dm.cellDofs[c].end());
        }

        for (const auto& patch : m.patches) {
            std::set<int> pdofs;
            for (int lf = 0; lf < patch.nFaces; ++lf) {
                int f = patch.startFace + lf;
                if (f < 0 || f >= (int)m.faces.size()) {
                    throw std::runtime_error("Patch face index out of range for patch " + patch.name);
                }
                for (int v : m.faces[f].verts) {
                    if (v < 0 || v >= (int)m.points.size()) {
                        throw std::runtime_error("Face vertex index out of range.");
                    }
                    pdofs.insert(v);
                    dm.boundaryDofs.insert(v);
                }
            }
            dm.patchDofs[patch.name] = std::move(pdofs);
        }
    } else {
        const MemoirsTopologyTables topo = build_memoirs_topology_tables(m);
        const int nVertexDofs = (int)m.points.size();
        const int nEdgeDofs = (int)topo.edgeVerts.size();
        dm.nDofs = nVertexDofs + nEdgeDofs;

        dm.dofCoordinates.resize(dm.nDofs);
        for (int p = 0; p < nVertexDofs; ++p) dm.dofCoordinates[p] = m.points[p];
        for (int e = 0; e < nEdgeDofs; ++e) {
            dm.dofCoordinates[nVertexDofs + e] = memoirs_edge_midpoint(m, topo.edgeVerts[e]);
        }

        dm.cellDofs.resize(m.cells.size());
        for (int c = 0; c < (int)m.cells.size(); ++c) {
            const auto& cell = m.cells[c];
            if (cell.verts.size() != 4) throw std::runtime_error("cg_tet_p2 dof map found non-tet cell.");
            dm.cellDofs[c].resize(10);
            for (int a = 0; a < 4; ++a) dm.cellDofs[c][a] = cell.verts[a];
            for (int e = 0; e < 6; ++e) {
                const int la = MEMOIRS_TET_P2_EDGE_VERTS[e][0];
                const int lb = MEMOIRS_TET_P2_EDGE_VERTS[e][1];
                const int edgeId = memoirs_find_edge_id_or_throw(topo, cell.verts[la], cell.verts[lb]);
                dm.cellDofs[c][4 + e] = nVertexDofs + edgeId;
            }
        }

        for (const auto& patch : m.patches) {
            std::set<int> pdofs;
            for (int lf = 0; lf < patch.nFaces; ++lf) {
                const int f = patch.startFace + lf;
                if (f < 0 || f >= (int)m.faces.size()) {
                    throw std::runtime_error("Patch face index out of range for patch " + patch.name);
                }
                const auto& fv = m.faces[f].verts;
                for (int v : fv) {
                    pdofs.insert(v);
                    dm.boundaryDofs.insert(v);
                }
                for (int i = 0; i < (int)fv.size(); ++i) {
                    const int va = fv[i];
                    const int vb = fv[(i + 1) % (int)fv.size()];
                    const int edgeId = memoirs_find_edge_id_or_throw(topo, va, vb);
                    const int dof = nVertexDofs + edgeId;
                    pdofs.insert(dof);
                    dm.boundaryDofs.insert(dof);
                }
            }
            dm.patchDofs[patch.name] = std::move(pdofs);
        }
    }

    dm.nBoundaryDofs = (int)dm.boundaryDofs.size();
    dm.nInteriorDofs = dm.nDofs - dm.nBoundaryDofs;

    return dm;
}

static void probe_dofs(const PolyMesh& m, const LinearCgDofMap& dm) {
    std::map<int,int> localDofCountHist;
    for (const auto& cd : dm.cellDofs) {
        localDofCountHist[(int)cd.size()]++;
    }

    std::cout << "---------------- dof probe -----------------\n";
    std::cout << "requested/resolved space = " << dm.resolvedSpace << "\n";
    std::cout << "cellFamily               = " << dm.cellFamily << "\n";
    std::cout << "continuity               = CG\n";
    if (dm.resolvedSpace == "cg_tet_p2") {
        std::cout << "basis                    = nodal Lagrange quadratic\n";
        std::cout << "global dof rule          = polyMesh point ids + canonical edge midpoint ids\n";
    } else {
        std::cout << "basis                    = nodal Lagrange linear\n";
        std::cout << "global dof rule          = polyMesh point id\n";
    }
    std::cout << "nCells                   = " << m.cells.size() << "\n";
    std::cout << "nPoints                  = " << m.points.size() << "\n";
    std::cout << "nDofs                    = " << dm.nDofs << "\n";
    std::cout << "nBoundaryDofs            = " << dm.nBoundaryDofs << "\n";
    std::cout << "nInteriorDofs            = " << dm.nInteriorDofs << "\n";

    std::cout << "local dofs per cell histogram:\n";
    for (auto [k,v] : localDofCountHist) {
        std::cout << "  nLocalDofs=" << k << " count=" << v << "\n";
    }

    std::cout << "patch dof counts:\n";
    for (const auto& kv : dm.patchDofs) {
        std::cout << "  " << kv.first << " uniqueDofs=" << kv.second.size() << "\n";
    }

    if (dm.resolvedSpace == "cg_hex_q1" && m.points.size() == 4913) {
        std::cout << "expected hex16 CG Q1 dofs        = 4913\n";
        std::cout << "expected hex16 boundary dofs     = 1538\n";
        std::cout << "expected hex16 interior dofs     = 3375\n";
    }

    std::cout << "--------------------------------------------\n";
}
