# Atmospheric boundary-layer omega inlet for OpenFOAM-dev

This library adds `atmosphericBoundaryLayerTurbulentOmega` to the Foundation
development version. It uses the same
`constant/atmosphericBoundaryLayerProperties` dictionary as the upstream
`atmosphericBoundaryLayerVelocity`,
`atmosphericBoundaryLayerTurbulentKineticEnergy`, and
`atmosphericBoundaryLayerTurbulentEpsilon` conditions.

For the neutral log layer, the inlet value is

\[
\omega = \frac{u^*}{\kappa\sqrt{C_\mu}(z-z_{Ground}+z_0)}.
\]

The implementation evaluates the upstream k and epsilon profiles and uses
`omega = epsilon/(Cmu*k)`. This is algebraically identical to the equation
above and follows the upstream spatial `z0` and `zGround` functions. A zero
velocity profile yields zero omega. If the upstream below-ground offset is
enabled, its `kLower` and `epsilonLower` values determine omega there.

Source OpenFOAM-dev's `etc/bashrc`, then run `wmake libso` here. In the
`0/omega` field, use dimensions `[0 0 -1 0 0 0 0]` and, for an inlet patch:

```foam
inlet
{
    type atmosphericBoundaryLayerTurbulentOmega;
    libs ("libmyatmosphericModels.so");
}
```

Define `flowDir`, `zDir`, `Uref`, `Zref`, `z0`, and `zGround` in
`constant/atmosphericBoundaryLayerProperties` as for the upstream conditions.
`Make/files` builds only the omega condition. Building to the default user
library directory replaces the former `libmyatmosphericModels.so`; the old
`atmBoundaryLayerInlet*` patch types are not registered by this build.
