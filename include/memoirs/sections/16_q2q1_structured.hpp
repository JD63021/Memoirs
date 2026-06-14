#pragma once

// ============================================================================
// SECTION 16: Structured Q2/Q1 bookkeeping for the HYPRE-IJ NSE refactor
// ============================================================================
//
// This header is intentionally narrow. It does NOT import the old scratch
// Q2/Q1 solver logic. It only provides structured unit-cube bookkeeping that
// the new Q2/Q1 HYPRE-IJ path needs before assembly is ported from Q1/Q1:
//
//   velocity space: continuous Q2 on a logical (2N+1)^3 grid
//   pressure space: continuous Q1 on a logical (N+1)^3 grid
//
// The production solver architecture should continue to follow the working
// Q1/Q1 HYPRE-IJ cavity branch.
// ============================================================================

struct Q2Q1StructuredGrid {
    int N = 0;
    int nv1 = 0;
    int np1 = 0;

    long long nVelocityNodes = 0;
    long long nPressureNodes = 0;
    long long nRows = 0;
    long long nCells = 0;

    double xmin = 0.0, xmax = 1.0;
    double ymin = 0.0, ymax = 1.0;
    double zmin = 0.0, zmax = 1.0;
    double hx = 0.0, hy = 0.0, hz = 0.0;
};

static long long q2q1_i64_cube(long long n) {
    return n * n * n;
}

static long long q2q1_vnode_id(const Q2Q1StructuredGrid& g, int I, int J, int K) {
    return (long long)I + (long long)g.nv1 * ((long long)J + (long long)g.nv1 * (long long)K);
}

static long long q2q1_pnode_id(const Q2Q1StructuredGrid& g, int i, int j, int k) {
    return (long long)i + (long long)g.np1 * ((long long)j + (long long)g.np1 * (long long)k);
}

static long long q2q1_ux_row(const Q2Q1StructuredGrid&, long long v) { return v; }
static long long q2q1_uy_row(const Q2Q1StructuredGrid& g, long long v) { return g.nVelocityNodes + v; }
static long long q2q1_uz_row(const Q2Q1StructuredGrid& g, long long v) { return 2LL * g.nVelocityNodes + v; }
static long long q2q1_p_row (const Q2Q1StructuredGrid& g, long long p) { return 3LL * g.nVelocityNodes + p; }

static Q2Q1StructuredGrid q2q1_infer_structured_unit_cube_grid(const PolyMesh& mesh) {
    Q2Q1StructuredGrid g;
    if (mesh.points.empty()) throw std::runtime_error("Q2/Q1 grid inference: mesh has no points.");
    if (mesh.cells.empty()) throw std::runtime_error("Q2/Q1 grid inference: mesh has no cells.");

    g.xmin = g.ymin = g.zmin =  std::numeric_limits<double>::infinity();
    g.xmax = g.ymax = g.zmax = -std::numeric_limits<double>::infinity();

    for (const Vec3& p : mesh.points) {
        g.xmin = std::min(g.xmin, p.x); g.xmax = std::max(g.xmax, p.x);
        g.ymin = std::min(g.ymin, p.y); g.ymax = std::max(g.ymax, p.y);
        g.zmin = std::min(g.zmin, p.z); g.zmax = std::max(g.zmax, p.z);
    }

    const long long nCells = (long long)mesh.cells.size();
    const int N = (int)std::llround(std::cbrt((double)nCells));

    if (N <= 0 || (long long)N * (long long)N * (long long)N != nCells) {
        std::ostringstream os;
        os << "Q2/Q1 first refactor assumes structured cube cell count N^3; got cells=" << nCells;
        throw std::runtime_error(os.str());
    }

    const long long expectedQ1Points = q2q1_i64_cube((long long)N + 1LL);
    if ((long long)mesh.points.size() != expectedQ1Points) {
        std::ostringstream os;
        os << "Q2/Q1 first refactor assumes blockMesh points=(N+1)^3; got points="
           << mesh.points.size() << " expected=" << expectedQ1Points << " for N=" << N;
        throw std::runtime_error(os.str());
    }

    for (const Cell& c : mesh.cells) {
        if (c.verts.size() != 8) {
            throw std::runtime_error("Q2/Q1 first refactor only supports all-hex structured blockMesh cells.");
        }
    }

    g.N = N;
    g.nv1 = 2 * N + 1;
    g.np1 = N + 1;
    g.nVelocityNodes = q2q1_i64_cube(g.nv1);
    g.nPressureNodes = q2q1_i64_cube(g.np1);
    g.nRows = 3LL * g.nVelocityNodes + g.nPressureNodes;
    g.nCells = nCells;
    g.hx = (g.xmax - g.xmin) / (double)N;
    g.hy = (g.ymax - g.ymin) / (double)N;
    g.hz = (g.zmax - g.zmin) / (double)N;

    return g;
}

static std::array<long long,27> q2q1_cell_velocity_nodes(const Q2Q1StructuredGrid& g, int i, int j, int k) {
    std::array<long long,27> ids{};
    int a = 0;
    for (int kk = 0; kk < 3; ++kk) {
        for (int jj = 0; jj < 3; ++jj) {
            for (int ii = 0; ii < 3; ++ii) {
                ids[a++] = q2q1_vnode_id(g, 2*i + ii, 2*j + jj, 2*k + kk);
            }
        }
    }
    return ids;
}

static std::array<long long,8> q2q1_cell_pressure_nodes(const Q2Q1StructuredGrid& g, int i, int j, int k) {
    return {{
        q2q1_pnode_id(g, i,   j,   k),
        q2q1_pnode_id(g, i+1, j,   k),
        q2q1_pnode_id(g, i+1, j+1, k),
        q2q1_pnode_id(g, i,   j+1, k),
        q2q1_pnode_id(g, i,   j,   k+1),
        q2q1_pnode_id(g, i+1, j,   k+1),
        q2q1_pnode_id(g, i+1, j+1, k+1),
        q2q1_pnode_id(g, i,   j+1, k+1)
    }};
}

static bool q2q1_is_velocity_boundary_node(const Q2Q1StructuredGrid& g, int I, int J, int K) {
    return I == 0 || J == 0 || K == 0 || I == g.nv1 - 1 || J == g.nv1 - 1 || K == g.nv1 - 1;
}

static bool q2q1_is_cavity_moving_lid_velocity_node(const Q2Q1StructuredGrid& g, int I, int J, int K) {
    return K == g.nv1 - 1 && I > 0 && I < g.nv1 - 1 && J > 0 && J < g.nv1 - 1;
}

struct Q2Q1BoundaryCounts {
    long long velocityBoundaryNodes = 0;
    long long cavityMovingLidVelocityNodes = 0;
    long long cavityNoSlipVelocityNodes = 0;
};

static Q2Q1BoundaryCounts q2q1_count_boundary_nodes(const Q2Q1StructuredGrid& g) {
    Q2Q1BoundaryCounts c;
    for (int K = 0; K < g.nv1; ++K) {
        for (int J = 0; J < g.nv1; ++J) {
            for (int I = 0; I < g.nv1; ++I) {
                if (!q2q1_is_velocity_boundary_node(g, I, J, K)) continue;
                ++c.velocityBoundaryNodes;
                if (q2q1_is_cavity_moving_lid_velocity_node(g, I, J, K)) {
                    ++c.cavityMovingLidVelocityNodes;
                } else {
                    ++c.cavityNoSlipVelocityNodes;
                }
            }
        }
    }
    return c;
}

static void q2q1_print_structured_banner(const PolyMesh& mesh, const Q2Q1StructuredGrid& g) {
    const Q2Q1BoundaryCounts bc = q2q1_count_boundary_nodes(g);

    std::cout << "--------------- q2q1 nse hypre branch ---------------\n";
    std::cout << "status                    = q2q1_hypre_ij_scaffold_phase0_grid_and_bc_masks\n";
    std::cout << "polyMesh points           = " << mesh.points.size() << "\n";
    std::cout << "polyMesh cells            = " << mesh.cells.size() << "\n";
    std::cout << "structured N              = " << g.N << "\n";
    std::cout << "q2 velocity grid          = " << g.nv1 << "x" << g.nv1 << "x" << g.nv1 << "\n";
    std::cout << "q1 pressure grid          = " << g.np1 << "x" << g.np1 << "x" << g.np1 << "\n";
    std::cout << "q2 velocity nodes         = " << g.nVelocityNodes << "\n";
    std::cout << "q1 pressure nodes         = " << g.nPressureNodes << "\n";
    std::cout << "q2q1 rows                 = " << g.nRows << "\n";
    std::cout << "ordering                  = block-major [ux(all q2), uy(all q2), uz(all q2), p(all q1)]\n";
    std::cout << "cell local velocity dofs  = 27\n";
    std::cout << "cell local pressure dofs  = 8\n";
    std::cout << "cavity velocity boundary nodes = " << bc.velocityBoundaryNodes << "\n";
    std::cout << "cavity moving lid q2 nodes     = " << bc.cavityMovingLidVelocityNodes << "\n";
    std::cout << "cavity no-slip q2 nodes        = " << bc.cavityNoSlipVelocityNodes << "\n";
    std::cout << "expected moving lid q2 nodes   = " << (long long)(g.nv1 - 2) * (long long)(g.nv1 - 2) << "\n";
    std::cout << "pressure pin row default       = " << q2q1_p_row(g, 0) << "\n";
    std::cout << "requiredAMGEnv            = MEMOIRS_AMG_NUM_FUNCTIONS=4 MEMOIRS_AMG_NODAL=0\n";
    std::cout << "-----------------------------------------------------\n";
}

// ============================================================================
// Q2/Q1 fixed sparse pattern scaffold
// ============================================================================
// This is still a pattern-only phase. It intentionally mirrors the verified
// Q1/Q1 fixed-pattern logic:
//
//   - strong velocity boundary rows are identity rows;
//   - one pressure pin row is an identity row;
//   - active momentum rows include same-component velocity columns and pressure
//     columns;
//   - active pressure rows include all velocity-component columns and pressure
//     PSPG columns.
//
// No numerical element values are assembled here yet.
// ============================================================================

struct Q2Q1PatternStats {
    long long rows = 0;
    long long nnz = 0;
    int minRowNnz = 0;
    int maxRowNnz = 0;
    double avgRowNnz = 0.0;

    long long uxNnz = 0;
    long long uyNnz = 0;
    long long uzNnz = 0;
    long long pNnz = 0;

    long long identityRows = 0;
    long long velocityStrongRows = 0;
    long long pressurePinRows = 0;
};

static inline Q2Q1PatternStats q2q1_analyze_fixed_pattern(
    const Q2Q1StructuredGrid& g,
    const SparseRows& A
) {
    Q2Q1PatternStats st;
    st.rows = sparse_nrows(A);
    st.nnz = sparse_nnz_flat(A);
    st.minRowNnz = (st.rows > 0) ? std::numeric_limits<int>::max() : 0;
    st.maxRowNnz = 0;

    for (int r = 0; r < (int)st.rows; ++r) {
        const int rowNnz = sparse_row_end(A, r) - sparse_row_start(A, r);
        st.minRowNnz = std::min(st.minRowNnz, rowNnz);
        st.maxRowNnz = std::max(st.maxRowNnz, rowNnz);

        if (rowNnz == 1) {
            const auto& cols = sparse_cols_row(A, r);
            if (!cols.empty() && cols[0] == r) ++st.identityRows;
        }

        if (r < (int)g.nVelocityNodes) {
            st.uxNnz += rowNnz;
        } else if (r < (int)(2LL * g.nVelocityNodes)) {
            st.uyNnz += rowNnz;
        } else if (r < (int)(3LL * g.nVelocityNodes)) {
            st.uzNnz += rowNnz;
        } else {
            st.pNnz += rowNnz;
        }
    }

    st.avgRowNnz = (st.rows > 0) ? (double)st.nnz / (double)st.rows : 0.0;
    st.velocityStrongRows = 3LL * q2q1_count_boundary_nodes(g).velocityBoundaryNodes;
    st.pressurePinRows = 1;
    if (st.minRowNnz == std::numeric_limits<int>::max()) st.minRowNnz = 0;
    return st;
}

static inline void q2q1_make_nse_picard_fixed_pattern(
    const Q2Q1StructuredGrid& g,
    int pressurePinPNode,
    SparseRows& A
) {
    if (pressurePinPNode < 0 || pressurePinPNode >= g.nPressureNodes) {
        throw std::runtime_error("q2q1 fixed pattern pressure pin out of range");
    }
    if (g.nRows > (long long)std::numeric_limits<int>::max()) {
        throw std::runtime_error("q2q1 fixed pattern currently requires int-sized row ids");
    }

    static std::shared_ptr<const SparseFixedPattern> cachedPattern;
    static int cachedN = -1;
    static int cachedPressurePinPNode = -999;

    if (cachedPattern && cachedN == g.N && cachedPressurePinPNode == pressurePinPNode) {
        sparse_init_from_fixed_pattern(A, cachedPattern);
        return;
    }

    const int nRows = (int)g.nRows;
    std::vector<unsigned char> isVelBoundary((std::size_t)g.nVelocityNodes, 0);

    for (int K = 0; K < g.nv1; ++K) {
        for (int J = 0; J < g.nv1; ++J) {
            for (int I = 0; I < g.nv1; ++I) {
                const long long v = q2q1_vnode_id(g, I, J, K);
                if (q2q1_is_velocity_boundary_node(g, I, J, K)) {
                    isVelBoundary[(std::size_t)v] = 1;
                }
            }
        }
    }

    std::vector<std::vector<int>> rowCols((std::size_t)nRows);

    auto add = [&](long long rLL, long long cLL) {
        if (rLL < 0 || rLL >= g.nRows || cLL < 0 || cLL >= g.nRows) {
            std::ostringstream os;
            os << "q2q1 fixed pattern row/col out of range row=" << rLL << " col=" << cLL;
            throw std::runtime_error(os.str());
        }
        rowCols[(std::size_t)rLL].push_back((int)cLL);
    };

    // Strong velocity boundary rows.
    for (int K = 0; K < g.nv1; ++K) {
        for (int J = 0; J < g.nv1; ++J) {
            for (int I = 0; I < g.nv1; ++I) {
                if (!q2q1_is_velocity_boundary_node(g, I, J, K)) continue;

                const long long v = q2q1_vnode_id(g, I, J, K);
                add(q2q1_ux_row(g, v), q2q1_ux_row(g, v));
                add(q2q1_uy_row(g, v), q2q1_uy_row(g, v));
                add(q2q1_uz_row(g, v), q2q1_uz_row(g, v));
            }
        }
    }

    // Pressure pin row.
    add(q2q1_p_row(g, pressurePinPNode), q2q1_p_row(g, pressurePinPNode));

    // Element graph. This follows the Q1/Q1 branch's current component-diagonal
    // momentum graph. PSPG pressure rows couple to all velocity components and
    // to pressure.
    for (int k = 0; k < g.N; ++k) {
        for (int j = 0; j < g.N; ++j) {
            for (int i = 0; i < g.N; ++i) {
                const auto vnodes = q2q1_cell_velocity_nodes(g, i, j, k);
                const auto pnodes = q2q1_cell_pressure_nodes(g, i, j, k);

                // Momentum rows.
                for (int a = 0; a < 27; ++a) {
                    const long long va = vnodes[a];

                    if (!isVelBoundary[(std::size_t)va]) {
                        // Velocity-velocity, component diagonal.
                        for (int b = 0; b < 27; ++b) {
                            const long long vb = vnodes[b];
                            if (!isVelBoundary[(std::size_t)vb]) {
                                add(q2q1_ux_row(g, va), q2q1_ux_row(g, vb));
                                add(q2q1_uy_row(g, va), q2q1_uy_row(g, vb));
                                add(q2q1_uz_row(g, va), q2q1_uz_row(g, vb));
                            }
                        }

                        // Momentum-pressure.
                        for (int bp = 0; bp < 8; ++bp) {
                            const long long pb = pnodes[bp];
                            if (pb != pressurePinPNode) {
                                add(q2q1_ux_row(g, va), q2q1_p_row(g, pb));
                                add(q2q1_uy_row(g, va), q2q1_p_row(g, pb));
                                add(q2q1_uz_row(g, va), q2q1_p_row(g, pb));
                            }
                        }
                    }
                }

                // Pressure rows: continuity + PSPG.
                for (int ap = 0; ap < 8; ++ap) {
                    const long long pa = pnodes[ap];
                    if (pa == pressurePinPNode) continue;

                    // Pressure-velocity.
                    for (int b = 0; b < 27; ++b) {
                        const long long vb = vnodes[b];
                        if (!isVelBoundary[(std::size_t)vb]) {
                            add(q2q1_p_row(g, pa), q2q1_ux_row(g, vb));
                            add(q2q1_p_row(g, pa), q2q1_uy_row(g, vb));
                            add(q2q1_p_row(g, pa), q2q1_uz_row(g, vb));
                        }
                    }

                    // Pressure-pressure PSPG block.
                    for (int bp = 0; bp < 8; ++bp) {
                        const long long pb = pnodes[bp];
                        if (pb != pressurePinPNode) {
                            add(q2q1_p_row(g, pa), q2q1_p_row(g, pb));
                        }
                    }
                }
            }
        }
    }

    cachedPattern = sparse_make_fixed_pattern(std::move(rowCols));
    cachedN = g.N;
    cachedPressurePinPNode = pressurePinPNode;

    sparse_init_from_fixed_pattern(A, cachedPattern);
}

static inline long long q2q1_probe_pattern_cell_slots(
    const Q2Q1StructuredGrid& g,
    const SparseRows& A,
    int ci,
    int cj,
    int ck,
    int pressurePinPNode
) {
    const auto vnodes = q2q1_cell_velocity_nodes(g, ci, cj, ck);
    const auto pnodes = q2q1_cell_pressure_nodes(g, ci, cj, ck);
    long long nLookups = 0;

    auto is_boundary_v = [&](long long v) -> bool {
        const int K = (int)(v / ((long long)g.nv1 * (long long)g.nv1));
        const long long rem = v - (long long)K * (long long)g.nv1 * (long long)g.nv1;
        const int J = (int)(rem / (long long)g.nv1);
        const int I = (int)(rem - (long long)J * (long long)g.nv1);
        return q2q1_is_velocity_boundary_node(g, I, J, K);
    };

    for (int a = 0; a < 27; ++a) {
        const long long va = vnodes[a];

        if (!is_boundary_v(va)) {
            for (int b = 0; b < 27; ++b) {
                const long long vb = vnodes[b];

                if (!is_boundary_v(vb)) {
                    (void)sparse_lookup_flat_slot(A, (int)q2q1_ux_row(g, va), (int)q2q1_ux_row(g, vb)); ++nLookups;
                    (void)sparse_lookup_flat_slot(A, (int)q2q1_uy_row(g, va), (int)q2q1_uy_row(g, vb)); ++nLookups;
                    (void)sparse_lookup_flat_slot(A, (int)q2q1_uz_row(g, va), (int)q2q1_uz_row(g, vb)); ++nLookups;
                }
            }

            for (int bp = 0; bp < 8; ++bp) {
                const long long pb = pnodes[bp];

                if (pb != pressurePinPNode) {
                    (void)sparse_lookup_flat_slot(A, (int)q2q1_ux_row(g, va), (int)q2q1_p_row(g, pb)); ++nLookups;
                    (void)sparse_lookup_flat_slot(A, (int)q2q1_uy_row(g, va), (int)q2q1_p_row(g, pb)); ++nLookups;
                    (void)sparse_lookup_flat_slot(A, (int)q2q1_uz_row(g, va), (int)q2q1_p_row(g, pb)); ++nLookups;
                }
            }
        }
    }

    for (int ap = 0; ap < 8; ++ap) {
        const long long pa = pnodes[ap];
        if (pa == pressurePinPNode) continue;

        for (int b = 0; b < 27; ++b) {
            const long long vb = vnodes[b];

            if (!is_boundary_v(vb)) {
                (void)sparse_lookup_flat_slot(A, (int)q2q1_p_row(g, pa), (int)q2q1_ux_row(g, vb)); ++nLookups;
                (void)sparse_lookup_flat_slot(A, (int)q2q1_p_row(g, pa), (int)q2q1_uy_row(g, vb)); ++nLookups;
                (void)sparse_lookup_flat_slot(A, (int)q2q1_p_row(g, pa), (int)q2q1_uz_row(g, vb)); ++nLookups;
            }
        }

        for (int bp = 0; bp < 8; ++bp) {
            const long long pb = pnodes[bp];

            if (pb != pressurePinPNode) {
                (void)sparse_lookup_flat_slot(A, (int)q2q1_p_row(g, pa), (int)q2q1_p_row(g, pb)); ++nLookups;
            }
        }
    }

    return nLookups;
}

static inline void q2q1_print_pattern_summary(
    const Q2Q1StructuredGrid& g,
    const SparseRows& A,
    double seconds
) {
    const Q2Q1PatternStats st = q2q1_analyze_fixed_pattern(g, A);

    std::cout << "q2q1PatternBuilt          = 1\n";
    std::cout << "q2q1PatternSeconds        = " << seconds << "\n";
    std::cout << "q2q1PatternRows           = " << st.rows << "\n";
    std::cout << "q2q1PatternNnz            = " << st.nnz << "\n";
    std::cout << "q2q1PatternAvgRowNnz      = " << st.avgRowNnz << "\n";
    std::cout << "q2q1PatternMinRowNnz      = " << st.minRowNnz << "\n";
    std::cout << "q2q1PatternMaxRowNnz      = " << st.maxRowNnz << "\n";
    std::cout << "q2q1PatternUxNnz          = " << st.uxNnz << "\n";
    std::cout << "q2q1PatternUyNnz          = " << st.uyNnz << "\n";
    std::cout << "q2q1PatternUzNnz          = " << st.uzNnz << "\n";
    std::cout << "q2q1PatternPNnz           = " << st.pNnz << "\n";
    std::cout << "q2q1PatternIdentityRows   = " << st.identityRows << "\n";
    std::cout << "q2q1PatternVelocityStrongRows = " << st.velocityStrongRows << "\n";
    std::cout << "q2q1PatternPressurePinRows    = " << st.pressurePinRows << "\n";
}

// ============================================================================
// Q2/Q1 per-cell flat-slot cache
// ============================================================================
// Still no physics assembly. This cache maps each local Q2/Q1 element coupling
// to a flat sparse-values slot. Later CUDA assembly should write directly to
// these slots, preserving the Q1/Q1 fixed-pattern / flat-slot strategy.
//
// Local slot layout per cell:
//
//   vv: comp=0..2, a=0..26, b=0..26     count 3*27*27 = 2187
//   vp: comp=0..2, a=0..26, bp=0..7     count 3*27*8  = 648
//   pv: ap=0..7, comp=0..2, b=0..26     count 8*3*27  = 648
//   pp: ap=0..7, bp=0..7                count 8*8     = 64
//
// Total full active-cell local slots = 3547.
// Boundary/pressure-pin suppressed entries are stored as -1.
// ============================================================================

struct Q2Q1CellSlotCache {
    static constexpr int vvOffset = 0;
    static constexpr int vvCount  = 3 * 27 * 27;

    static constexpr int vpOffset = vvOffset + vvCount;
    static constexpr int vpCount  = 3 * 27 * 8;

    static constexpr int pvOffset = vpOffset + vpCount;
    static constexpr int pvCount  = 8 * 3 * 27;

    static constexpr int ppOffset = pvOffset + pvCount;
    static constexpr int ppCount  = 8 * 8;

    static constexpr int slotsPerCell = ppOffset + ppCount; // 3547

    int N = 0;
    long long nCells = 0;
    std::vector<int> slot; // length nCells * slotsPerCell, -1 for inactive/suppressed entries

    static constexpr int idx_vv(int comp, int a, int b) {
        return vvOffset + comp * 27 * 27 + a * 27 + b;
    }

    static constexpr int idx_vp(int comp, int a, int bp) {
        return vpOffset + comp * 27 * 8 + a * 8 + bp;
    }

    static constexpr int idx_pv(int ap, int comp, int b) {
        return pvOffset + ap * 3 * 27 + comp * 27 + b;
    }

    static constexpr int idx_pp(int ap, int bp) {
        return ppOffset + ap * 8 + bp;
    }

    int& at(long long cell, int localSlot) {
        return slot[(std::size_t)cell * (std::size_t)slotsPerCell + (std::size_t)localSlot];
    }

    int at(long long cell, int localSlot) const {
        return slot[(std::size_t)cell * (std::size_t)slotsPerCell + (std::size_t)localSlot];
    }
};

struct Q2Q1CellSlotCacheStats {
    long long nCells = 0;
    int slotsPerCell = 0;
    long long totalLocalSlots = 0;
    long long activeSlots = 0;
    long long inactiveSlots = 0;
    int minActiveSlotsPerCell = 0;
    int maxActiveSlotsPerCell = 0;
    double avgActiveSlotsPerCell = 0.0;
    int interiorProbeActiveSlots = 0;
    int cornerProbeActiveSlots = 0;
};

static inline bool q2q1_decode_velocity_node_ijk(
    const Q2Q1StructuredGrid& g,
    long long v,
    int& I,
    int& J,
    int& K
) {
    if (v < 0 || v >= g.nVelocityNodes) return false;
    K = (int)(v / ((long long)g.nv1 * (long long)g.nv1));
    const long long rem = v - (long long)K * (long long)g.nv1 * (long long)g.nv1;
    J = (int)(rem / (long long)g.nv1);
    I = (int)(rem - (long long)J * (long long)g.nv1);
    return true;
}

static inline bool q2q1_is_boundary_vnode_id(const Q2Q1StructuredGrid& g, long long v) {
    int I = 0, J = 0, K = 0;
    if (!q2q1_decode_velocity_node_ijk(g, v, I, J, K)) return true;
    return q2q1_is_velocity_boundary_node(g, I, J, K);
}

static inline long long q2q1_cell_linear_id(const Q2Q1StructuredGrid& g, int i, int j, int k) {
    return (long long)i + (long long)g.N * ((long long)j + (long long)g.N * (long long)k);
}

static inline void q2q1_build_cell_slot_cache(
    const Q2Q1StructuredGrid& g,
    const SparseRows& A,
    int pressurePinPNode,
    Q2Q1CellSlotCache& cache
) {
    if (pressurePinPNode < 0 || pressurePinPNode >= g.nPressureNodes) {
        throw std::runtime_error("q2q1 cell slot cache pressure pin out of range");
    }

    cache.N = g.N;
    cache.nCells = g.nCells;
    cache.slot.assign(
        (std::size_t)g.nCells * (std::size_t)Q2Q1CellSlotCache::slotsPerCell,
        -1
    );

    auto store_slot = [&](long long cell, int localSlot, long long row, long long col) {
        if (row < 0 || row >= g.nRows || col < 0 || col >= g.nRows) {
            std::ostringstream os;
            os << "q2q1 cell slot cache row/col out of range row=" << row << " col=" << col;
            throw std::runtime_error(os.str());
        }
        const int flatSlot = sparse_lookup_flat_slot(A, (int)row, (int)col);
        if (flatSlot < 0) {
            std::ostringstream os;
            os << "q2q1 cell slot cache missing flat slot row=" << row << " col=" << col;
            throw std::runtime_error(os.str());
        }
        cache.at(cell, localSlot) = flatSlot;
    };

    for (int k = 0; k < g.N; ++k) {
        for (int j = 0; j < g.N; ++j) {
            for (int i = 0; i < g.N; ++i) {
                const long long cell = q2q1_cell_linear_id(g, i, j, k);
                const auto vnodes = q2q1_cell_velocity_nodes(g, i, j, k);
                const auto pnodes = q2q1_cell_pressure_nodes(g, i, j, k);

                // vv and vp for active momentum rows.
                for (int a = 0; a < 27; ++a) {
                    const long long va = vnodes[a];
                    if (q2q1_is_boundary_vnode_id(g, va)) continue;

                    for (int b = 0; b < 27; ++b) {
                        const long long vb = vnodes[b];
                        if (q2q1_is_boundary_vnode_id(g, vb)) continue;

                        store_slot(cell, Q2Q1CellSlotCache::idx_vv(0, a, b), q2q1_ux_row(g, va), q2q1_ux_row(g, vb));
                        store_slot(cell, Q2Q1CellSlotCache::idx_vv(1, a, b), q2q1_uy_row(g, va), q2q1_uy_row(g, vb));
                        store_slot(cell, Q2Q1CellSlotCache::idx_vv(2, a, b), q2q1_uz_row(g, va), q2q1_uz_row(g, vb));
                    }

                    for (int bp = 0; bp < 8; ++bp) {
                        const long long pb = pnodes[bp];
                        if (pb == pressurePinPNode) continue;

                        store_slot(cell, Q2Q1CellSlotCache::idx_vp(0, a, bp), q2q1_ux_row(g, va), q2q1_p_row(g, pb));
                        store_slot(cell, Q2Q1CellSlotCache::idx_vp(1, a, bp), q2q1_uy_row(g, va), q2q1_p_row(g, pb));
                        store_slot(cell, Q2Q1CellSlotCache::idx_vp(2, a, bp), q2q1_uz_row(g, va), q2q1_p_row(g, pb));
                    }
                }

                // pv and pp for active pressure rows.
                for (int ap = 0; ap < 8; ++ap) {
                    const long long pa = pnodes[ap];
                    if (pa == pressurePinPNode) continue;

                    for (int b = 0; b < 27; ++b) {
                        const long long vb = vnodes[b];
                        if (q2q1_is_boundary_vnode_id(g, vb)) continue;

                        store_slot(cell, Q2Q1CellSlotCache::idx_pv(ap, 0, b), q2q1_p_row(g, pa), q2q1_ux_row(g, vb));
                        store_slot(cell, Q2Q1CellSlotCache::idx_pv(ap, 1, b), q2q1_p_row(g, pa), q2q1_uy_row(g, vb));
                        store_slot(cell, Q2Q1CellSlotCache::idx_pv(ap, 2, b), q2q1_p_row(g, pa), q2q1_uz_row(g, vb));
                    }

                    for (int bp = 0; bp < 8; ++bp) {
                        const long long pb = pnodes[bp];
                        if (pb == pressurePinPNode) continue;

                        store_slot(cell, Q2Q1CellSlotCache::idx_pp(ap, bp), q2q1_p_row(g, pa), q2q1_p_row(g, pb));
                    }
                }
            }
        }
    }
}

static inline int q2q1_count_active_slots_for_cell(
    const Q2Q1CellSlotCache& cache,
    long long cell
) {
    int n = 0;
    for (int q = 0; q < Q2Q1CellSlotCache::slotsPerCell; ++q) {
        if (cache.at(cell, q) >= 0) ++n;
    }
    return n;
}

static inline Q2Q1CellSlotCacheStats q2q1_analyze_cell_slot_cache(
    const Q2Q1StructuredGrid& g,
    const Q2Q1CellSlotCache& cache
) {
    Q2Q1CellSlotCacheStats st;
    st.nCells = cache.nCells;
    st.slotsPerCell = Q2Q1CellSlotCache::slotsPerCell;
    st.totalLocalSlots = cache.nCells * (long long)Q2Q1CellSlotCache::slotsPerCell;
    st.minActiveSlotsPerCell = Q2Q1CellSlotCache::slotsPerCell;
    st.maxActiveSlotsPerCell = 0;

    for (long long cell = 0; cell < cache.nCells; ++cell) {
        const int active = q2q1_count_active_slots_for_cell(cache, cell);
        st.activeSlots += active;
        st.minActiveSlotsPerCell = std::min(st.minActiveSlotsPerCell, active);
        st.maxActiveSlotsPerCell = std::max(st.maxActiveSlotsPerCell, active);
    }

    st.inactiveSlots = st.totalLocalSlots - st.activeSlots;
    st.avgActiveSlotsPerCell = (cache.nCells > 0) ? (double)st.activeSlots / (double)cache.nCells : 0.0;

    const long long cornerCell = q2q1_cell_linear_id(g, 0, 0, 0);
    const int ci = (g.N >= 3) ? std::min(g.N - 2, std::max(1, g.N / 2)) : 0;
    const long long interiorCell = q2q1_cell_linear_id(g, ci, ci, ci);

    st.cornerProbeActiveSlots = q2q1_count_active_slots_for_cell(cache, cornerCell);
    st.interiorProbeActiveSlots = q2q1_count_active_slots_for_cell(cache, interiorCell);

    if (cache.nCells == 0) st.minActiveSlotsPerCell = 0;
    return st;
}

static inline void q2q1_print_cell_slot_cache_summary(
    const Q2Q1StructuredGrid& g,
    const Q2Q1CellSlotCache& cache,
    double seconds
) {
    const Q2Q1CellSlotCacheStats st = q2q1_analyze_cell_slot_cache(g, cache);

    std::cout << "q2q1CellSlotCacheBuilt    = 1\n";
    std::cout << "q2q1CellSlotCacheSeconds  = " << seconds << "\n";
    std::cout << "q2q1CellSlotCacheCells    = " << st.nCells << "\n";
    std::cout << "q2q1CellSlotCacheSlotsPerCell = " << st.slotsPerCell << "\n";
    std::cout << "q2q1CellSlotCacheTotalLocalSlots = " << st.totalLocalSlots << "\n";
    std::cout << "q2q1CellSlotCacheActiveSlots     = " << st.activeSlots << "\n";
    std::cout << "q2q1CellSlotCacheInactiveSlots   = " << st.inactiveSlots << "\n";
    std::cout << "q2q1CellSlotCacheAvgActivePerCell = " << st.avgActiveSlotsPerCell << "\n";
    std::cout << "q2q1CellSlotCacheMinActivePerCell = " << st.minActiveSlotsPerCell << "\n";
    std::cout << "q2q1CellSlotCacheMaxActivePerCell = " << st.maxActiveSlotsPerCell << "\n";
    std::cout << "q2q1CellSlotCacheCornerActive     = " << st.cornerProbeActiveSlots << "\n";
    std::cout << "q2q1CellSlotCacheInteriorActive   = " << st.interiorProbeActiveSlots << "\n";
    std::cout << "q2q1CellSlotCacheExpectedFullInterior = " << Q2Q1CellSlotCache::slotsPerCell << "\n";
}

// ============================================================================
// Q2 velocity / Q1 pressure reference basis audit
// ============================================================================
// This is a host-side mathematical scaffold only. It fixes the local ordering
// used by the coming CUDA assembly:
//
//   Q2 velocity local id:
//     a = ii + 3*(jj + 3*kk), ii,jj,kk = 0,1,2
//     reference nodes = {-1,0,1}^3
//
//   Q1 pressure local id:
//     same order as q2q1_cell_pressure_nodes:
//       0 (-,-,-), 1 (+,-,-), 2 (+,+,-), 3 (-,+,-),
//       4 (-,-,+), 5 (+,-,+), 6 (+,+,+), 7 (-,+,+)
//
// For Q2 we provide values, first derivatives, and pure second derivatives
// d2/dxi2, d2/deta2, d2/dzeta2. Cross-derivatives are not needed for the
// axis-aligned Laplacian used in the first structured-cube path.
// ============================================================================

struct Q2Q1BasisAuditReport {
    double q2MaxPartitionErr = 0.0;
    double q2MaxGradSumErr = 0.0;
    double q2MaxSecondSumErr = 0.0;
    double q2MaxNodalDiagErr = 0.0;
    double q2MaxNodalOffDiag = 0.0;

    double q1MaxPartitionErr = 0.0;
    double q1MaxGradSumErr = 0.0;
    double q1MaxNodalDiagErr = 0.0;
    double q1MaxNodalOffDiag = 0.0;

    double quadWeightSum = 0.0;
    double quadWeightVolumeErr = 0.0;

    int quadraturePoints = 0;
};

static inline void q2q1_q2_1d(
    double x,
    double L[3],
    double dL[3],
    double ddL[3]
) {
    // Lagrange nodes: -1, 0, +1
    L[0] = 0.5 * x * (x - 1.0);
    L[1] = 1.0 - x * x;
    L[2] = 0.5 * x * (x + 1.0);

    dL[0] = x - 0.5;
    dL[1] = -2.0 * x;
    dL[2] = x + 0.5;

    ddL[0] = 1.0;
    ddL[1] = -2.0;
    ddL[2] = 1.0;
}

static inline void q2q1_q1_1d(
    double x,
    double L[2],
    double dL[2]
) {
    L[0] = 0.5 * (1.0 - x);
    L[1] = 0.5 * (1.0 + x);

    dL[0] = -0.5;
    dL[1] =  0.5;
}

static inline void q2q1_hex_q2_basis_ref(
    double xi,
    double eta,
    double zeta,
    std::array<double,27>& N,
    std::array<Vec3,27>& dN,
    std::array<Vec3,27>& ddN
) {
    double Lx[3], Ly[3], Lz[3];
    double dLx[3], dLy[3], dLz[3];
    double ddLx[3], ddLy[3], ddLz[3];

    q2q1_q2_1d(xi,   Lx, dLx, ddLx);
    q2q1_q2_1d(eta,  Ly, dLy, ddLy);
    q2q1_q2_1d(zeta, Lz, dLz, ddLz);

    int a = 0;
    for (int kk = 0; kk < 3; ++kk) {
        for (int jj = 0; jj < 3; ++jj) {
            for (int ii = 0; ii < 3; ++ii) {
                N[a] = Lx[ii] * Ly[jj] * Lz[kk];

                dN[a].x = dLx[ii] * Ly[jj]  * Lz[kk];
                dN[a].y = Lx[ii]  * dLy[jj] * Lz[kk];
                dN[a].z = Lx[ii]  * Ly[jj]  * dLz[kk];

                ddN[a].x = ddLx[ii] * Ly[jj]   * Lz[kk];   // d2/dxi2
                ddN[a].y = Lx[ii]   * ddLy[jj] * Lz[kk];   // d2/deta2
                ddN[a].z = Lx[ii]   * Ly[jj]   * ddLz[kk]; // d2/dzeta2

                ++a;
            }
        }
    }
}

static inline void q2q1_hex_q1_pressure_basis_ref(
    double xi,
    double eta,
    double zeta,
    std::array<double,8>& N,
    std::array<Vec3,8>& dN
) {
    static const int sx[8] = {-1,  1,  1, -1, -1,  1,  1, -1};
    static const int sy[8] = {-1, -1,  1,  1, -1, -1,  1,  1};
    static const int sz[8] = {-1, -1, -1, -1,  1,  1,  1,  1};

    for (int a = 0; a < 8; ++a) {
        const double X = 1.0 + sx[a] * xi;
        const double Y = 1.0 + sy[a] * eta;
        const double Z = 1.0 + sz[a] * zeta;

        N[a] = 0.125 * X * Y * Z;
        dN[a].x = 0.125 * sx[a] * Y * Z;
        dN[a].y = 0.125 * X * sy[a] * Z;
        dN[a].z = 0.125 * X * Y * sz[a];
    }
}

static inline void q2q1_gauss4_1d(std::array<double,4>& x, std::array<double,4>& w) {
    const double a = std::sqrt((3.0 - 2.0 * std::sqrt(6.0 / 5.0)) / 7.0);
    const double b = std::sqrt((3.0 + 2.0 * std::sqrt(6.0 / 5.0)) / 7.0);

    x = {{-b, -a, a, b}};

    const double wa = (18.0 + std::sqrt(30.0)) / 36.0;
    const double wb = (18.0 - std::sqrt(30.0)) / 36.0;
    w = {{wb, wa, wa, wb}};
}

static inline Q2Q1BasisAuditReport q2q1_run_basis_audit() {
    Q2Q1BasisAuditReport r;

    std::array<double,4> gx, gw;
    q2q1_gauss4_1d(gx, gw);

    for (int k = 0; k < 4; ++k) {
        for (int j = 0; j < 4; ++j) {
            for (int i = 0; i < 4; ++i) {
                const double xi = gx[i];
                const double eta = gx[j];
                const double zeta = gx[k];
                const double wt = gw[i] * gw[j] * gw[k];

                r.quadWeightSum += wt;
                ++r.quadraturePoints;

                std::array<double,27> N2;
                std::array<Vec3,27> dN2;
                std::array<Vec3,27> ddN2;
                q2q1_hex_q2_basis_ref(xi, eta, zeta, N2, dN2, ddN2);

                double sumN2 = 0.0;
                Vec3 sumD2;
                Vec3 sumDD2;

                for (int a = 0; a < 27; ++a) {
                    sumN2 += N2[a];
                    sumD2.x += dN2[a].x; sumD2.y += dN2[a].y; sumD2.z += dN2[a].z;
                    sumDD2.x += ddN2[a].x; sumDD2.y += ddN2[a].y; sumDD2.z += ddN2[a].z;
                }

                r.q2MaxPartitionErr = std::max(r.q2MaxPartitionErr, std::abs(sumN2 - 1.0));
                r.q2MaxGradSumErr = std::max(r.q2MaxGradSumErr, std::abs(sumD2.x));
                r.q2MaxGradSumErr = std::max(r.q2MaxGradSumErr, std::abs(sumD2.y));
                r.q2MaxGradSumErr = std::max(r.q2MaxGradSumErr, std::abs(sumD2.z));
                r.q2MaxSecondSumErr = std::max(r.q2MaxSecondSumErr, std::abs(sumDD2.x));
                r.q2MaxSecondSumErr = std::max(r.q2MaxSecondSumErr, std::abs(sumDD2.y));
                r.q2MaxSecondSumErr = std::max(r.q2MaxSecondSumErr, std::abs(sumDD2.z));

                std::array<double,8> N1;
                std::array<Vec3,8> dN1;
                q2q1_hex_q1_pressure_basis_ref(xi, eta, zeta, N1, dN1);

                double sumN1 = 0.0;
                Vec3 sumD1;

                for (int a = 0; a < 8; ++a) {
                    sumN1 += N1[a];
                    sumD1.x += dN1[a].x; sumD1.y += dN1[a].y; sumD1.z += dN1[a].z;
                }

                r.q1MaxPartitionErr = std::max(r.q1MaxPartitionErr, std::abs(sumN1 - 1.0));
                r.q1MaxGradSumErr = std::max(r.q1MaxGradSumErr, std::abs(sumD1.x));
                r.q1MaxGradSumErr = std::max(r.q1MaxGradSumErr, std::abs(sumD1.y));
                r.q1MaxGradSumErr = std::max(r.q1MaxGradSumErr, std::abs(sumD1.z));
            }
        }
    }

    r.quadWeightVolumeErr = std::abs(r.quadWeightSum - 8.0);

    // Q2 nodality.
    const double q2nodes[3] = {-1.0, 0.0, 1.0};
    for (int kk = 0; kk < 3; ++kk) {
        for (int jj = 0; jj < 3; ++jj) {
            for (int ii = 0; ii < 3; ++ii) {
                const int node = ii + 3 * (jj + 3 * kk);

                std::array<double,27> N2;
                std::array<Vec3,27> dN2;
                std::array<Vec3,27> ddN2;
                q2q1_hex_q2_basis_ref(q2nodes[ii], q2nodes[jj], q2nodes[kk], N2, dN2, ddN2);

                for (int a = 0; a < 27; ++a) {
                    if (a == node) {
                        r.q2MaxNodalDiagErr = std::max(r.q2MaxNodalDiagErr, std::abs(N2[a] - 1.0));
                    } else {
                        r.q2MaxNodalOffDiag = std::max(r.q2MaxNodalOffDiag, std::abs(N2[a]));
                    }
                }
            }
        }
    }

    // Q1 nodality using local order 0,1,2,3,4,5,6,7 as above.
    static const double q1x[8] = {-1,  1,  1, -1, -1,  1,  1, -1};
    static const double q1y[8] = {-1, -1,  1,  1, -1, -1,  1,  1};
    static const double q1z[8] = {-1, -1, -1, -1,  1,  1,  1,  1};

    for (int node = 0; node < 8; ++node) {
        std::array<double,8> N1;
        std::array<Vec3,8> dN1;
        q2q1_hex_q1_pressure_basis_ref(q1x[node], q1y[node], q1z[node], N1, dN1);

        for (int a = 0; a < 8; ++a) {
            if (a == node) {
                r.q1MaxNodalDiagErr = std::max(r.q1MaxNodalDiagErr, std::abs(N1[a] - 1.0));
            } else {
                r.q1MaxNodalOffDiag = std::max(r.q1MaxNodalOffDiag, std::abs(N1[a]));
            }
        }
    }

    return r;
}

static inline void q2q1_print_basis_audit_report(const Q2Q1BasisAuditReport& r) {
    std::cout << "q2q1BasisAuditDone        = 1\n";
    std::cout << "q2q1BasisAuditQuadraturePoints = " << r.quadraturePoints << "\n";
    std::cout << "q2q1BasisAuditQuadWeightSum = " << std::setprecision(16) << r.quadWeightSum << "\n";
    std::cout << "q2q1BasisAuditQuadVolumeErr = " << std::setprecision(16) << r.quadWeightVolumeErr << "\n";
    std::cout << "q2q1BasisAuditQ2PartitionErr = " << std::setprecision(16) << r.q2MaxPartitionErr << "\n";
    std::cout << "q2q1BasisAuditQ2GradSumErr = " << std::setprecision(16) << r.q2MaxGradSumErr << "\n";
    std::cout << "q2q1BasisAuditQ2SecondSumErr = " << std::setprecision(16) << r.q2MaxSecondSumErr << "\n";
    std::cout << "q2q1BasisAuditQ2NodalDiagErr = " << std::setprecision(16) << r.q2MaxNodalDiagErr << "\n";
    std::cout << "q2q1BasisAuditQ2NodalOffDiag = " << std::setprecision(16) << r.q2MaxNodalOffDiag << "\n";
    std::cout << "q2q1BasisAuditQ1PartitionErr = " << std::setprecision(16) << r.q1MaxPartitionErr << "\n";
    std::cout << "q2q1BasisAuditQ1GradSumErr = " << std::setprecision(16) << r.q1MaxGradSumErr << "\n";
    std::cout << "q2q1BasisAuditQ1NodalDiagErr = " << std::setprecision(16) << r.q1MaxNodalDiagErr << "\n";
    std::cout << "q2q1BasisAuditQ1NodalOffDiag = " << std::setprecision(16) << r.q1MaxNodalOffDiag << "\n";
}

