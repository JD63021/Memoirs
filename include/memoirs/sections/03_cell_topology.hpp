#pragma once

// Auto-split from frozen patch014 one-shot source.
// This header is intentionally included in order by apps/poisson_mms.cpp.

// ============================================================================
// SECTION 3: Cell reconstruction / classification
// ============================================================================

static double approximate_cell_volume(const PolyMesh& m, const Cell& c) {
    // Robust enough for diagnostics on tet/hex unit-cube meshes.
    // Use vertex bbox as a positive scale; exact volume comes later during FE mapping.
    double xmin =  std::numeric_limits<double>::infinity();
    double ymin =  std::numeric_limits<double>::infinity();
    double zmin =  std::numeric_limits<double>::infinity();
    double xmax = -std::numeric_limits<double>::infinity();
    double ymax = -std::numeric_limits<double>::infinity();
    double zmax = -std::numeric_limits<double>::infinity();

    for (int v : c.verts) {
        const auto& p = m.points[v];
        xmin = std::min(xmin, p.x); xmax = std::max(xmax, p.x);
        ymin = std::min(ymin, p.y); ymax = std::max(ymax, p.y);
        zmin = std::min(zmin, p.z); zmax = std::max(zmax, p.z);
    }
    return std::max(0.0, (xmax-xmin)*(ymax-ymin)*(zmax-zmin));
}

static void probe_mesh(const PolyMesh& m) {
    std::map<int,int> faceVertHist;
    for (const auto& f : m.faces) faceVertHist[(int)f.verts.size()]++;

    std::map<std::string,int> cellHist;
    int nHex = 0;
    int nTet = 0;
    int nOther = 0;

    double minVolBox = std::numeric_limits<double>::infinity();
    double maxVolBox = 0.0;

    for (const auto& c : m.cells) {
        const int nf = (int)c.faces.size();
        const int nv = (int)c.verts.size();

        if (nf == 6 && nv == 8) {
            nHex++;
            cellHist["hex_like"]++;
        } else if (nf == 4 && nv == 4) {
            nTet++;
            cellHist["tet_like"]++;
        } else {
            nOther++;
            cellHist["other"]++;
        }

        double vb = approximate_cell_volume(m, c);
        minVolBox = std::min(minVolBox, vb);
        maxVolBox = std::max(maxVolBox, vb);
    }

    std::cout << "---------------- mesh probe ----------------\n";
    std::cout << "nPoints           = " << m.points.size() << "\n";
    std::cout << "nFaces            = " << m.faces.size() << "\n";
    std::cout << "nOwner            = " << m.owner.size() << "\n";
    std::cout << "nNeighbour        = " << m.neighbour.size() << "\n";
    std::cout << "nBoundaryFaces    = " << (m.faces.size() - m.neighbour.size()) << "\n";
    std::cout << "nCells            = " << m.cells.size() << "\n";
    std::cout << "nBoundaryPatches  = " << m.patches.size() << "\n";

    std::cout << "face vertex histogram:\n";
    for (auto [k,v] : faceVertHist) {
        std::cout << "  nVertsPerFace=" << k << " count=" << v << "\n";
    }

    std::cout << "cell histogram:\n";
    for (auto& kv : cellHist) {
        std::cout << "  " << kv.first << " = " << kv.second << "\n";
    }

    std::cout << "allHex            = " << ((nHex == (int)m.cells.size()) ? "yes" : "no") << "\n";
    std::cout << "allTet            = " << ((nTet == (int)m.cells.size()) ? "yes" : "no") << "\n";
    std::cout << "otherCells        = " << nOther << "\n";
    std::cout << "bboxVol min/max   = " << minVolBox << " / " << maxVolBox << "\n";

    std::cout << "patches:\n";
    for (const auto& p : m.patches) {
        std::cout << "  " << p.name
                  << " type=" << p.type
                  << " nFaces=" << p.nFaces
                  << " startFace=" << p.startFace
                  << "\n";
    }
    std::cout << "--------------------------------------------\n";
}
