# NOTES

-`CaseA.xls` --> `expData.csv` : removed duplicates from experimental data


## Inlet boundary data

Source: `CaseA(1_1_2).xls`, sheet `Inflow`, rows 3 to 26.
Column B contains `z/b`. Column C contains `U` in m/s.
Column G contains `k` in m2/s2.
Use `b = 0.08 m`. Retain the workbook precision before conversion.
The 24 measured heights extend from 0.005 m to 0.6 m.
The workbook `k` equals `(sigma_u^2 + sigma_v^2 + sigma_w^2)/2`.

The points cover the inlet of `single-building/grid-generation`:
`x = -0.80 m`, `y = -0.80 m` to `0.80 m`, and `z = 0` to `1.28 m`.
These limits match the current mesh and its active `system/userDict`.
The building height is `2 * 0.08 = 0.16 m`.
The domain height is `8 * 0.16 = 1.28 m`.
Update the boundary points when these grid limits change.
Do not scale the experimental heights when the domain changes.

Each of the 93 heights has two identical field values, one at each side.
The points alternate between the two sides. The first three are non-collinear.
The inlet fields in `0.orig` use `planarInterpolation`.
Set `perturb 0` to retain the exact point coordinates during interpolation.
This gives linear interpolation in height for each field.

Add `U = 0` and `k = 0` at the ground.
Between the ground and the first measurement, interpolate linearly.
Above 0.6 m, retain the original upper source samples for `U`, `k`, and `omega`.
These added values are not experimental measurements.
The original `k` extension decreases through approximately
`k = 0.07 m2/s2` at `z = 0.74 m`, to `k = 0.05 m2/s2` at `z = 0.9 m`.
The source samples retain the original height spacing and stored precision.
The linear mapper connects these samples.
Above 0.9 m, hold all three fields at their original 0.9 m values:
`U = 6.751 m/s`, `k = 0.05 m2/s2`, and `omega = 1.10636393 1/s`.
The added top points at 1.28 m use those same values.
The ground values and upper extension are modeling assumptions.

### Modeled omega

The workbook has no measured `omega` or dissipation rate.
At measured source heights from 0.005 m to 0.6 m, calculate:

```text
omega = sqrt(k) / (Cmu^0.25 * kappa * z)
Cmu = 0.09
kappa = 0.41
```

Here `z` is in m, `k` is in m2/s2, and `omega` is in 1/s.
At the ground, copy `omega` from the first measured height, 0.005 m.
This gives a finite ground value and avoids division by zero.
Above 0.6 m, retain the original upper `omega` samples.
Above 0.9 m, hold `omega = 1.10636393 1/s` to the top.
Do not recalculate the formula in this constant upper segment.
Between source heights, the mapper interpolates `omega` linearly.
It does not recalculate the formula at each inlet face.

### Startup ramp

The experimental profile has no prescribed time variation.
The two time directories define a numerical startup ramp.
At time `0`, multiply all full-profile `U`, `k`, and `omega` values by `0.1`.
At time `0.8 s`, use the full values.
The boundary condition interpolates each field linearly in time.
After `0.8 s`, it holds the full values.
