# OpenFOAM Foundation extensions

Custom libraries and solver modules for atmospheric boundary layer simulations
with turbulence-driven adaptive mesh refinement (AMR). The work combines the
`kOmegaSSTSAS` turbulence model with refinement indicators and continuity
diagnostics to study refinement bursts and regions that remain refined.

The code targets Foundation OpenFOAM-dev after the OpenFOAM 14 release. The
directory layout follows OpenFOAM.

## Structure

```text
OpenFOAM-foundation/
├── Allwmake                       # Build the custom libraries and solver module.
├── applications/                  # Custom solver modules and their diagnostics.
├── src/                           # Custom OpenFOAM libraries.
│   ├── MomentumTransportModels/   # Modified kOmegaSSTSAS turbulence model.
│   ├── atmosphericModels/         # Atmospheric boundary conditions and porosity model.
│   ├── finiteVolume/              # Modified time-varying boundary conditions.
│   ├── functionObjects/           # AMR indicators and simulation function objects.
│   ├── fvMeshTopoChangers/        # Custom adaptive mesh refinement library.
│   └── sampling/                  # Modified sampling library.
└── tutorials/                     # Original test cases and supporting data.
    ├── array_buildings/orig/
    └── single_building/orig/
```

## Build and use

Load the matching OpenFOAM environment, then run `./Allwmake` from this directory.
See the [solver module notes](applications/modules/incompressibleFluid/README.md)
for continuity diagnostics before AMR and around `correctPhi`.

## Ongoing development

- Simulation automation, convergence monitoring, and averaging control.
- Voronoi grid generation with `foamyHexMesh`.
- Single-building meshing with `snappyHexMesh` and wall layers.
- Grid convergence, experimental validation, and steady-to-unsteady analysis.
