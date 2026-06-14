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
};

static std::string lower_copy(std::string s) {
    for (char& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

static LinearCgDofMap build_linear_cg_dof_map(const PolyMesh& m, const std::string& requestedSpace) {
    MeshCellCounts cc = count_cell_types(m);
    const bool allHex = (cc.nHex == (int)m.cells.size());
    const bool allTet = (cc.nTet == (int)m.cells.size());

    std::string req = lower_copy(requestedSpace);

    LinearCgDofMap dm;

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
    } else {
        throw std::runtime_error("Unsupported -space for Patch 002: " + requestedSpace);
    }

    dm.nDofs = (int)m.points.size();
    dm.pointToDof.resize(m.points.size());
    for (int p = 0; p < (int)m.points.size(); ++p) dm.pointToDof[p] = p;

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
    std::cout << "basis                    = nodal Lagrange linear\n";
    std::cout << "global dof rule          = polyMesh point id\n";
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
