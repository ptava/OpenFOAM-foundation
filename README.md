## OpenFOAM function objects
---

This repository contains a collection of additional OpenFOAM foundation function objects.

---

### 1. refinementInfo
This function object writes any user-specified `uniformDimensionedScalarField` found in the registry at each `execute()` call.

> [!NOTE]
> - created to write run-time into a file all info available associated with mesh refinement process;
>
> - it needs modified `refiner` (see [modified dynamicFvMesh](https://github.com/ptava/fvMeshTopoChangers.git)) because is designed to access stored values in the registry such as `nTotToRefine`, `nTotRefined`, `lowerLimit`, `upperLimit` etc.

---

### 2. sasRefineIndicator
This function object generates a `volScalarField` that drives adaptive mesh refinement.

> [!NOTE]
> - computations are restricted to `cellZone` defined by the user, allowing to focus refinement on specific regions of the domain;
> - it is tailored for Scale-Adaptive Simulation (SAS) modelling, where mesh refinement should focus on regions of interest defined bu the turbulence length scales;
> - requires length scale `Lvk` and its terms `C1` and `C2` (where $Lvk=max(C1,C2)$): available options to store these fields are in a modified `kOmegaSSTSAS` turbulence model (see [modified kOmegaSSTSAS](https://github.com/ptava/kOmegaSSTSAS.git)).
> - To visualise and compare transfer functions, run from the repository root (supporting `basic` `constant` `oddScaler` `safeScaler` `gaussSinkCore` `gaussSinkPeriphery` `all`):

```bash
python field/sasRefineIndicator/new_transfer_functions.py all
```


#### Available user options (refinement criteria):

Refinement indicator based on the **von Kármán length scale** ($L_{vk}$) and is
designed to serve as an adaptive refinement indicator for a Scale Adaptive
(SA) turbulente model.

Where:
- $L_{vk} = \max(c_1, c_2)$
- $c_1$ is the physical turbulence length scale
- $c_2$ is the grid-related high wave number damper (linked to grid size)

The porpose is to adaptively refine mesh in regions where vortices are being
detected by a scale-resolving simulation, refining in vortices-associated
regions and coarsening outside them.

Options are provided to enphasise different regions of the flow/vortices:

##### 1. **focusRegion = "core"**

Focuses refinement targeting the core of turbulent eddies detected by Scale-Adaptive Simulation (SAS) models. It aims to refine the mesh where the "grid filter" ($c_2$) is larger than the "physical length scale" ($c_1$), indicating that the current mesh is too coarse to resolve the turbulent structures present.

Available transfer functions:

###### a) **markCoreConstant**:

Apply constant value to all cells where $d = c2 - c1 > 0$

```math
    f(d) = \text{const}
```


###### b) **markCoreOddScaler**:

Apply an odd, monotonic, sign-preserving function, symmetric around
origin, normalised by its maximum value:

<table align="center">
  <tr>
    <td align="center">
      <img src="imgs/tf_2.png" width="320" />
    </td>
    <td align="center">
      <img src="imgs/tf_3.png" width="320" />
    </td>
  </tr>
  <tr>
    <td align="center" colspan="2">
      <em>Transfer function representation with changing $w$ (left) and $\sigma$ (right)</em>
    </td>
  </tr>
</table>


```math
  f(d) = \overline{d} \cdot \left( 1 - \exp\left( -\frac{\overline{d}^2}{2 \sigma^2} \right) \right)
```


> [!NOTE]
> - As $|d| \to \infty$, the exponential vanishes, and $f(d) \sim c \cdot d$
> - $\lim_{x \to a} f(x) = x^3$ - function behaves cubically near 0 (useful for smoothly marking the interface region)
> - $\sigma \in (0.0, 1.0]$ controls the width of the transition region leading to steep decrease of $f(d)$ near $d^{max}$ for increasing $\sigma$ values
> - $\overline{d} = d + w \cdot d^{max}$, with $w \geq 0$ - shifts the function to the left including in the refinement region also cells with small negative $d$ values
> - output function normalised by its maximum value

<p align="center">
    <img src="imgs/os_1_0_18931.png" width="50%" height="50%">
    <br>
    <em>Example with $\sigma \eq 1$ and $w \eq 0$</em>
</p>
<p align="center">
    <img src="imgs/os_1_0p3_22534.png" width="50%" height="50%">
    <br>
    <em>Example with $\sigma \eq 1$ and $w \eq 0.3$</em>
</p>
<p align="center">
    <img src="imgs/os_1_0p5_24348.png" width="50%" height="50%">
    <br>
    <em>Example with $\sigma \eq 1$ and $w \eq 0.5$</em>
</p>
<p align="center">
    <img src="imgs/os_1_1_28106.png" width="50%" height="50%">
    <br>
    <em>Example with $\sigma \eq 1$ and $w \eq 1$</em>
</p>

###### c) **markCoreSafeScaler**:

`safeScaler` adds damping to the positive branch of `oddScaler`.
`weight1` shifts the input, and `sigma` controls the base width for every alpha.

| `alpha` | Continuous curve behavior |
|---|---|
| `0` | Exactly `oddScaler`; also recovered as alpha approaches zero |
| Between `0` and `1` | Positive endpoint, below the curve maximum |
| `1` | Zero endpoint |
| Greater than `1` | Negative endpoint |

All finite, non-negative alpha values are allowed. The default is `0.6`.
For alpha greater than zero, the definition is:

```math
u=d+w_1d^{max},\qquad U=(1+w_1)d^{max},\qquad
B(u)=u\left(1-\exp\left[-\frac{u^2}{2\sigma^2}\right]\right),\qquad
b=\frac{B(u)}{B(U)}
```

```math
F(b)=\begin{cases}
b, & b\leq0,\\
b\left(1-cb^{2/\alpha}\right), & b>0.
\end{cases}
\qquad
\frac{F(1)}{\max_{0\leq b\leq1}F(b)}=1-\alpha
```

The calculated coefficient `c` sets the continuous endpoint-to-maximum ratio.
The code stores `endpoint = 1-c` and uses an equivalent form of the equation.
This preserves precision near alpha = 1, where `c` is close to 1.
The output field is then divided by its global positive cell maximum, giving a maximum of 1.
A field with no positive maximum after damping cannot be normalized.

Output values vary with the cell inputs used for normalization.
If cells miss the continuous peak, the final endpoint ratio can differ from `1-alpha`.
The same divisor scales negative values, which can fall below -1.

To compare alpha values, run from the repository root:

```bash
python field/sasRefineIndicator/plot_transfer_functions.py safeScaler \
  --weight1 1 --sigma 0.05 --alpha 0 \
  --compare alpha=1 --compare alpha=1.5 --d-min -6
```

###### d) **markCoreGaussSink**:

Gaussian-like function peaking where nLvk = Lvk / c2 = 1, and decays for
increasing values of nLvk:

<table align="center">
  <tr>
    <td align="center">
      <img src="imgs/tf_0.png" width="320" />
    </td>
    <td align="center">
      <img src="imgs/tf_1.png" width="320" />
    </td>
  </tr>
  <tr>
    <td align="center" colspan="2">
      <em>Transfer function representation with changing $w$ (left) and $\sigma$ (right)</em>
    </td>
  </tr>
</table>

```math
    f(nLvk) = \exp\left( -\frac{1}{2} \left( \frac{nLvk - 1}{\sigma} \right)^2 \right) + w \cdot (nLvk - 1)^2
```


<p align="center">
    <img src="imgs/gs_0p05_1_20037.png" width="50%" height="50%">
    <br>
    <em>Example with $\sigma \eq 0.05$ and $w \eq 1$</em>
</p>
<p align="center">
    <img src="imgs/gs_0p5_1_24103.png" width="50%" height="50%">
    <br>
    <em>Example with $\sigma \eq 0.5$ and $w \eq 1$</em>
</p>
<p align="center">
    <img src="imgs/gs_0p05_1_span.png" width="50%" height="50%">
    <br>
    <em>Example with $\sigma \eq 0.05$ and $w \eq 1$</em>
</p>

<p align="center">
    <img src="imgs/gs_0p5_10_span.png" width="50%" height="50%">
    <br>
    <em>Example with $\sigma \eq 0.05$ and $w \eq 10$</em>
</p>


> [!NOTE]
> - $w \geq 0$
> - $w$ controls values outside the focus region
> - by construction, $L_{vk} = max(c_1, c_2)$ therefore all cells where $c_1 \leq c_2$ (i.e., $d \geq 0$) are going to have $nLvk eq 1$

##### 2. **focusRegion = "periphery"**

Focuses refinement in the periphery of vortex structures rather than
their core.

Available transfer functions:

###### e) **markPeripheryGaussSink**:

Gaussian-like function peaking at a user-defined reference value
of Lvk (e.g., a free-stream value). Same as 'c' but different nLvk
definition:


```math
    f(nLvk) = w_1 \cdot \exp\left( -\frac{1}{2} \left( \frac{nLvk - 1}{\sigma} \right)^2 \right) - w_2 \cdot (nLvk - 1)^2
```

