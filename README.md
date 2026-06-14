# Memoirs

Memoirs is a modular FEM + CFD research codebase aimed at building accurate, extensible finite-element workflows for GPU-oriented simulation.

It is a memorabilia of FEM and CFD methods: a place where different discretizations, solver strategies, verification tests, and GPU workflows can be developed, compared, and preserved.

## Project aim

The long-term goal is to build a modular finite-element and CFD framework that can support:

- Continuous Galerkin finite elements
- Discontinuous Galerkin finite elements
- Mixed and H(div)-conforming methods
- Hex and tet meshes
- Nodal and modal bases
- Single and double precision builds
- HYPRE IJ/ParCSR sparse solver paths
- Structured matrix-free and geometric multigrid paths
- GPU-resident operators and solvers
- Scalar Poisson and diffusion
- Scalar transport
- Stokes and Navier-Stokes workflows
- PSPG, SUPG, VMS, and related stabilized formulations

The emphasis is on correctness, modularity, and numerical traceability first, with GPU efficiency developed progressively as methods become verified.

## Current status

The current code includes scalar Poisson MMS paths for several CG/DG combinations on unit-cube meshes, including:

- CG strong Dirichlet paths
- CG Nitsche weak boundary paths
- DG SIPG weak boundary paths
- DG strong/clamped boundary paths
- Hex Q1 and tet P1 variants
- Modal DG P1 variants
- Generalized quadrature controls

A mixed H(div) Poisson vertical slice has also been added:

- RT0 mixed Poisson on hex meshes
- Cellwise P0/Q0 scalar space
- One normal flux degree of freedom per mesh face
- Locally conservative H(div)-conforming flux
- Double-precision hex MMS verification
- VTU export of cellwise scalar and reconstructed cell-centered flux

## RT0 mixed Poisson note

The RT0 mixed formulation solves:

    sigma = -grad(u)
    div(sigma) = f

The numerical flux is H(div)-conforming: its normal component is single-valued across faces. The divergence/source equation is enforced cell by cell through the discontinuous scalar test space.

For the Poisson MMS case, this means:

    div(sigma_h) = Pi_0 f

where Pi_0 f is the cell-average projection of the source. The method is locally conservative to solver tolerance.

## Example build

Typical double-precision HYPRE configure:

    export CUDACXX=/usr/local/cuda-12.2/bin/nvcc
    export CMAKE_CUDA_COMPILER=/usr/local/cuda-12.2/bin/nvcc
    export CMAKE_CUDA_ARCHITECTURES=86
    export MEMOIRS_BUILD_EXPERIMENTS=OFF
    export MEMOIRS_SPARSE_MODE=fixed_csr

    cmake -S . -B build_dp_hypre \
      -DCMAKE_BUILD_TYPE=Release \
      -DMEMOIRS_PRECISION=double \
      -DMEMOIRS_USE_HYPRE=ON \
      -DMEMOIRS_BUILD_EXPERIMENTS=OFF \
      -DHYPRE_ROOT=/home/jd/opt/hypre-cuda-double-clean \
      -DCMAKE_CUDA_ARCHITECTURES=86

Build the RT0 mixed Poisson target:

    cmake --build build_dp_hypre --target memoirs_rt0_mixed_poisson_mms -j 8

Example RT0 hex MMS run:

    export LD_LIBRARY_PATH=/home/jd/opt/hypre-cuda-double-clean/lib:${LD_LIBRARY_PATH:-}
    export MEMOIRS_SPARSE_MODE=fixed_csr
    export MEMOIRS_QUAD_ORDER=4
    export MEMOIRS_GMRES_KDIM=200

    ./build_dp_hypre/memoirs_rt0_mixed_poisson_mms \
      -polyMeshDir /home/jd/Desktop/meshes/unitcube/blockmesh/16cube/constant/polyMesh \
      -space rt0 \
      -mms sin \
      -solve 1 \
      -solver gmres \
      -precond none \
      -hypreMemory host \
      -tol 1e-10 \
      -maxit 10000 \
      -diagLevel 0 \
      -hyprePrint 0 \
      -vtuFile vtu/rt0_hex16_mms.vtu

## License

This project is licensed under the Apache License 2.0.
