The module inherits the Foundation OpenFOAM-dev incompressible solver and adds
continuity diagnostics around mesh-change flux correction. Load the matching
OpenFOAM environment before compiling with `wmake libso` in this directory.
The library is `libmyincompressibleFluidSolver.so` in `$FOAM_USER_LIBBIN`.

Copy `examples/continuityDiagnostics` and `examples/continuityBeforeAMR` into the
case's `system` directory. Include the latter inside `controlDict`'s `functions`:

```foam
functions
{
    #include "continuityBeforeAMR"
}
```

Set the shared inverse-time `threshold` in `system/continuityDiagnostics`.
The coded object reports before the next mesh update. Set `writeResidual true`
to also write `continuityResidual` at ordinary case write times. These baseline
fields belong to normal case output; they are not independent mesh snapshots.

Merge this entry into the case's `system/configDict`:

```foam
DebugSwitches
{
    correctPhi 2;
}
```

Launch from the case directory with the custom library on the search path:

```bash
foamRun -solver myincompressibleFluid > log.foamRun 2>&1
```

Debug 0 is off. Debug 1 logs statistics. Debug 2 logs the same statistics and
writes the residual field immediately before and after `fv::correctPhi`.
These captures follow flux reconstruction/boundary correction and precede
relative-frame conversion. They run only on topology-changing correction calls;
repeated correctors produce separate pairs. This switch does not control the
baseline coded object or the existing `PIMPLE.correctPhi` setting.

Every stage logs one line:

```text
continuity stage=beforeCorrectPhi cellsAbove=... maxAbs=... maxLocation=(...) volAverageAbs=...
```

For `R = div(phi)`, `cellsAbove` counts cells with `mag(R) > threshold`;
`maxAbs` is the maximum magnitude and `maxLocation` its cell centre;
`volAverageAbs` is `sum(V*mag(R))/sum(V)` over the entire mesh. Residuals and the
threshold have units s^-1; locations have units m. All reductions are global
across the participating ranks. Tied maxima may select different valid cells
under different decompositions. No timestep multiplier is applied.

Level 2 writes independent cases under:

```text
postProcessing/continuity/<mesh>/<run>/<pair>/beforeCorrectPhi/
postProcessing/continuity/<mesh>/<run>/<pair>/afterCorrectPhi/
```

Each contains a copied `constant/polyMesh`, zones, `system/controlDict`, and a
capture-time `continuityResidual` field. Parallel output retains the processor
patch connectivity and rank-local mesh ordering. It does not copy potentially
stale reconstruction addressing from the live mesh; regenerate reconstruction
addressing with native mesh reconstruction tools if needed. Existing file
handlers are retained and flushed before completion. `continuity.foam` is
created only after all ranks successfully write, and can be opened by a native
OpenFOAM reader. Internal residual values are authoritative; non-constraint
boundary values use zero-gradient extrapolation. Constraint/coupled fields use
the matching patch type. Snapshot meshes are owned independently; live time,
mesh instances and global stream precision are not modified.

The supported scope is stationary, source-free incompressible volumetric flux,
without MRF. Level 2 supports conformal meshes with patch/wall, empty, wedge,
symmetry/symmetryPlane, cyclic, processor and processorCyclic patches; other
patch types are rejected. Disk cost is one complete mesh and field per capture.
Runs receive separate output directories, so restarts do not overwrite previous
captures. No production cases or installed upstream libraries are modified.
