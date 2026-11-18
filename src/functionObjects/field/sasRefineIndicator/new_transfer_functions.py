"""Plot the transfer functions implemented by sasRefineIndicator."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import is_color_like, to_rgba
from matplotlib.patches import Rectangle


SMALL = 1.0e-15
VSMALL = np.finfo(float).tiny
COMPARISON_PRIMARY_ALPHA = 0.28
COMPARISON_SECONDARY_ALPHA = 0.12
GAUSS_SINK_FUNCTIONS = ("gaussSinkCore", "gaussSinkPeriphery")
TRANSFER_FUNCTIONS = (
    "basic",
    "constant",
    "oddScaler",
    "safeScaler",
    *GAUSS_SINK_FUNCTIONS,
)
COMPARISON_PARAMETERS = ("weight1", "weight2", "sigma", "alpha")
FUNCTION_PARAMETERS = {
    "basic": (),
    "constant": ("weight2",),
    "oddScaler": ("weight1", "sigma"),
    "safeScaler": ("weight1", "sigma", "alpha"),
    "gaussSinkCore": ("weight2", "sigma"),
    "gaussSinkPeriphery": ("weight2", "sigma"),
}


def _as_values(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _normalize_by_max(values: np.ndarray) -> np.ndarray:
    """Match the function object's whole-field ``G /= maxG`` operation."""

    result = np.asarray(values, dtype=float).copy()
    maximum = float(np.max(result))
    if maximum > SMALL:
        result /= maximum
    return result


def basic_transfer(values: np.ndarray) -> np.ma.MaskedArray:
    """Return ``d`` where ``d > 0`` and mask OpenFOAM's ``-GREAT`` region."""

    d = _as_values(values)
    return np.ma.masked_where(d <= 0.0, d)


def constant_transfer(
    values: np.ndarray,
    weight2: float,
) -> np.ma.MaskedArray:
    """Return ``weight2`` where ``d > 0`` and mask the inactive region."""

    d = _as_values(values)
    result = np.full_like(d, weight2)
    return np.ma.masked_where(d <= 0.0, result)


def odd_scaler_transfer(
    values: np.ndarray,
    weight1: float,
    sigma: float,
) -> np.ndarray | np.ma.MaskedArray:
    """Apply ``markCoreOddScaler`` to representative ``d`` values."""

    d = _as_values(values)
    d_max = float(np.max(d))
    if d_max <= SMALL:
        return np.ma.masked_all_like(d)

    shifted = d + weight1 * d_max
    result = shifted * (-np.expm1(-(shifted * shifted) / (2.0 * sigma * sigma)))
    return _normalize_by_max(result)


def _log_endpoint_ratio_magnitude(
    log_end: float, p: float, negative_end: bool,
) -> float:
    log_damping = (
        math.log1p(math.exp(log_end))
        if negative_end else math.log(-math.expm1(log_end))
    )
    return log_end + (log_damping + math.log1p(p)) / p + math.log1p(1.0 / p)


def _damping_endpoint(alpha: float) -> float:
    """Return 1-c, retaining precision near alpha=1."""

    if alpha == 1.0:
        return 0.0
    p = 2.0 / alpha
    negative_end = alpha > 1.0
    log_target = math.log(alpha - 1.0) if negative_end else math.log1p(-alpha)
    high = 0.0 if negative_end else -math.log1p(1.0 / p)
    while (
        negative_end
        and _log_endpoint_ratio_magnitude(high, p, negative_end) < log_target
    ):
        high += 1.0
    low = high - 1.0
    while _log_endpoint_ratio_magnitude(low, p, negative_end) > log_target:
        low -= max(1.0, abs(low))
    for _ in range(100):
        mid = 0.5 * (low + high)
        if _log_endpoint_ratio_magnitude(mid, p, negative_end) < log_target:
            low = mid
        else:
            high = mid
    log_end = 0.5 * (low + high)
    error = _log_endpoint_ratio_magnitude(log_end, p, negative_end) - log_target
    if not math.isfinite(error) or abs(error) > math.sqrt(SMALL):
        raise ValueError(f"cannot resolve safeScaler damping for alpha={alpha}")
    endpoint = math.exp(log_end)
    return -endpoint if negative_end else endpoint


def safe_scaler_transfer(
    values: np.ndarray,
    weight1: float,
    sigma: float,
    alpha: float,
) -> np.ndarray | np.ma.MaskedArray:
    """Apply localised damping, then normalize by the sampled maximum.

    alpha sets endpoint/continuous maximum = 1-alpha. The final endpoint
    ratio can differ when the samples miss the continuous curve peak.
    """

    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be finite and non-negative")
    if not math.isfinite(sigma) or sigma <= SMALL:
        raise ValueError("sigma must be finite and greater than zero")
    if not math.isfinite(weight1) or weight1 < 0.0:
        raise ValueError("weight1 must be finite and non-negative")
    if 1.0 - alpha == 1.0:
        return odd_scaler_transfer(values, weight1, sigma)

    d = _as_values(values)
    if d.size == 0 or not np.all(np.isfinite(d)):
        raise ValueError("safeScaler requires a non-empty finite input field")
    d_max = float(np.max(d))
    if d_max <= SMALL:
        return np.ma.masked_all_like(d)

    shift = weight1 * d_max
    u_max = d_max + shift
    inv_two_sigma2 = 0.5 / (sigma * sigma)
    end_factor = -math.expm1(-u_max * u_max * inv_two_sigma2)
    if not math.isfinite(u_max) or not end_factor > 0.0:
        raise ValueError("cannot scale the oddScaler endpoint")

    u = d + shift
    result = (u / u_max) * (-np.expm1(-u * u * inv_two_sigma2)) / end_factor
    result[d == d_max] = 1.0
    positive = result > 0.0
    b = result[positive]
    log_power = 2.0 * np.log(np.minimum(b, 1.0)) / alpha
    power = np.exp(log_power)
    # endpoint = 1-c. This form preserves precision near alpha=1
    # when evaluating b*(1-c*b^p), with p=2/alpha.
    result[positive] *= -np.expm1(log_power) + _damping_endpoint(alpha) * power

    maximum = float(np.max(result))
    if not math.isfinite(maximum) or maximum <= VSMALL:
        raise ValueError("safeScaler has no positive maximum after damping")
    return result / maximum


def _gauss_sink_transfer(
    normalized_lvk: np.ndarray,
    weight2: float,
    sigma: float,
) -> np.ndarray:
    """Apply the Gaussian sink to an OpenFOAM-normalized coordinate."""

    values = _as_values(normalized_lvk)
    offset = values - 1.0
    return np.exp(-(offset * offset) / (2.0 * sigma * sigma)) - weight2 * offset * offset


def gauss_sink_core_transfer(
    c1_over_c2: np.ndarray,
    weight2: float,
    sigma: float,
) -> np.ndarray:
    """Apply the core sink using ``Lvk/C2=max(C1/C2, 1)``."""

    normalized_lvk = np.maximum(_as_values(c1_over_c2), 1.0)
    return _gauss_sink_transfer(normalized_lvk, weight2, sigma)


def gauss_sink_periphery_transfer(
    lvk_over_lvk_ref: np.ndarray,
    weight2: float,
    sigma: float,
) -> np.ndarray:
    """Apply the periphery sink using unrestricted ``Lvk/LvkRef``."""

    return _gauss_sink_transfer(lvk_over_lvk_ref, weight2, sigma)


class _HelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    """Preserve explanatory sections while displaying option defaults."""

    def _get_help_string(self, action: argparse.Action) -> str:
        if action.default is None:
            return action.help
        return super()._get_help_string(action)


def _comparison_specification(specification: str) -> dict[str, float]:
    """Parse one comma-separated comparison parameter override."""

    overrides: dict[str, float] = {}
    for assignment in specification.split(","):
        name, separator, raw_value = assignment.partition("=")
        if not separator or not name or not raw_value:
            raise argparse.ArgumentTypeError(
                "comparison values must use PARAM=VALUE[,PARAM=VALUE...]"
            )
        if name not in COMPARISON_PARAMETERS:
            choices = ", ".join(COMPARISON_PARAMETERS)
            raise argparse.ArgumentTypeError(
                f"unknown comparison parameter '{name}'; choose from {choices}"
            )
        if name in overrides:
            raise argparse.ArgumentTypeError(
                f"comparison parameter '{name}' must not be repeated"
            )
        try:
            overrides[name] = float(raw_value)
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                f"comparison parameter '{name}' must be a number"
            ) from error
    return overrides


def _effective_parameter_values(
    function_name: str,
    parameters: dict[str, float],
) -> tuple[float, ...]:
    """Return parameters that affect the selected curve."""

    names = FUNCTION_PARAMETERS[function_name]
    return tuple(parameters[name] for name in names)


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="""Create representative sasRefineIndicator transfer-function plots.

Each requested function is drawn in a separate figure. Without --output-dir,
all figures are opened interactively; otherwise, one named PNG is saved per
function.""",
        epilog="""TRANSFER FUNCTIONS
  basic (core)
    G=d when d>0; otherwise G=-GREAT.

  constant (core)
    G=weight2 when d>0; otherwise G=-GREAT.

  oddScaler (core)
    u=d+weight1*dMax
    G=u*(1-exp(-u^2/(2*sigma^2)))

  safeScaler (core)
    At alpha=0, this is oddScaler with the user-provided sigma.
    For alpha>0, b=B(u)/B(uMax), with B the oddScaler base, and
    G=b*(1-c*b^(2/alpha)) only where b>0. The coefficient c sets
    endpoint/continuous maximum=1-alpha. The negative branch is undamped
    before whole-curve normalization. As alpha approaches zero, this
    recovers oddScaler. sigma is active for every alpha.
    Finite alpha>=0 is allowed; alpha>1 gives a negative endpoint.
    At alpha=1, the endpoint is zero.
    Final output values vary with the samples used for normalization.
    The endpoint ratio can differ from 1-alpha if samples miss the curve peak.
    A field with no positive maximum after damping cannot be normalized.

  gaussSinkCore (core)
    nLvk=max(C1/C2,1)
    G=exp(-(nLvk-1)^2/(2*sigma^2))-weight2*(nLvk-1)^2

  gaussSinkPeriphery (periphery)
    nLvk=Lvk/LvkRef
    G=exp(-(nLvk-1)^2/(2*sigma^2))-weight2*(nLvk-1)^2

  oddScaler and safeScaler divide the entire curve, including negative values,
  by max(G). oddScaler normalizes when max(G)>SMALL; safeScaler with alpha>0
  requires a positive maximum after damping. basic, constant, gaussSinkCore, and
  gaussSinkPeriphery are not normalized. all plots all six functions in the
  order above.

PARAMETER OWNERSHIP
  weight1 affects oddScaler and safeScaler.
  weight2 affects constant, gaussSinkCore, and gaussSinkPeriphery.
  sigma affects oddScaler, gaussSinkCore, and gaussSinkPeriphery. For
  safeScaler it controls the base width for every non-negative alpha.
  alpha affects safeScaler only; endpoint/continuous maximum=1-alpha.

COMPARISON CURVES
  --compare requires exactly one transfer function and may be repeated. Each
  PARAM=VALUE list inherits the baseline command-line parameters and overrides
  only the values it names. All curves use the Matplotlib color cycle and
  continuous lines. Comparison fills use each curve's color: refinement uses
  the stronger primary opacity and unrefinement uses a more transparent version.
  --primary-alpha and --secondary-alpha are applied directly without curve-count
  scaling, while neutral gray legend entries use the selected primary and
  secondary opacities to describe the two candidate regions. Fills with the
  largest signed area are drawn first so the
  positive and negative sides receive the opposite nesting order. Use
  --no-area-fill to suppress all colored areas and their two legend entries.
  Endpoint markers remain visible and share the legend entry
  "current min/max of processed value".

  Comparable parameters are weight2 for constant; weight1 and sigma for
  oddScaler; weight1, sigma, and alpha for safeScaler; and weight2 and sigma for
  gaussSinkCore and gaussSinkPeriphery. basic has no configurable
  transfer-function parameter to compare.

OPENFOAM COORDINATES AND PLOTTING
  d is C2-C1. In OpenFOAM, dMax is the parallel maximum of C2-C1 over the
  selected cellZone. The plot uses --d-max as that representative dMax.
  gaussSinkCore plots C1/C2 and applies the model relation
  Lvk/C2=max(C1/C2,1), exposing the G=1 plateau where C1/C2<=1.
  gaussSinkPeriphery plots Lvk/LvkRef without that lower bound. Core
  transferFunction selection does not apply to the periphery: periphery always
  uses gaussSinkPeriphery.
  basic and constant assign -GREAT where d <= 0; the plots leave that inactive
  interval blank rather than drawing the sentinel as a finite value.
  Dashed endpoint markers run from G=0 to the corresponding finite curve value;
  markers with masked endpoint values are omitted. In single-curve plots,
  positive and negative areas use --positive-color and --negative-color, with
  --primary-alpha and --secondary-alpha controlling their respective opacities.
  Comparison plots instead use each curve's color and the neutral legend
  entries described above. Use --legend-outside to place the legend to the
  right of the plot instead of inside the axes.

EXAMPLES
  python3 %(prog)s oddScaler --weight1 0.3 --sigma 0.1
  python3 %(prog)s oddScaler safeScaler --d-min -4 --d-max 4 --alpha 0.7
  python3 %(prog)s safeScaler --compare alpha=0.3 --compare alpha=1,sigma=0.1
  python3 %(prog)s safeScaler --alpha 0 --compare alpha=1 --compare alpha=1.5 --d-min -6
  python3 %(prog)s oddScaler --compare weight1=0.3 --primary-alpha 0.15 --secondary-alpha 0.05
  python3 %(prog)s gaussSinkCore --nlvk-min 0 --nlvk-max 2 --weight2 1
  python3 %(prog)s gaussSinkPeriphery --nlvk-min 0 --nlvk-max 2 --weight2 1
  python3 %(prog)s all --output-dir plots""",
        formatter_class=_HelpFormatter,
    )
    parser.add_argument(
        "functions",
        nargs="+",
        choices=(*TRANSFER_FUNCTIONS, "all"),
        metavar="FUNCTION",
        help=(
            "one or more of basic, constant, oddScaler, safeScaler, "
            "gaussSinkCore, gaussSinkPeriphery, or all"
        ),
    )

    ranges = parser.add_argument_group("plot ranges")
    ranges.add_argument(
        "--d-min",
        type=float,
        default=-3.0,
        metavar="VALUE",
        help="minimum plotted d = C2-C1",
    )
    ranges.add_argument(
        "--d-max",
        type=float,
        default=3.0,
        metavar="VALUE",
        help="maximum plotted d and representative field-derived dMax",
    )
    ranges.add_argument(
        "--nlvk-min",
        type=float,
        default=0.0,
        metavar="VALUE",
        help="minimum C1/C2 or Lvk/LvkRef coordinate for a gauss-sink plot",
    )
    ranges.add_argument(
        "--nlvk-max",
        type=float,
        default=2.0,
        metavar="VALUE",
        help="maximum C1/C2 or Lvk/LvkRef coordinate for a gauss-sink plot",
    )

    parameters = parser.add_argument_group("OpenFOAM transfer-function parameters")
    parameters.add_argument(
        "--weight1",
        type=float,
        default=0.0,
        metavar="VALUE",
        help="non-negative shift weight for oddScaler and safeScaler",
    )
    parameters.add_argument(
        "--weight2",
        type=float,
        default=1.0,
        metavar="VALUE",
        help="non-negative level/sink weight for constant and both gauss sinks",
    )
    parameters.add_argument(
        "--sigma",
        type=float,
        default=0.05,
        metavar="VALUE",
        help="positive width for oddScaler, both gauss sinks, and safeScaler at every alpha",
    )
    parameters.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        metavar="VALUE",
        help="non-negative safeScaler damping; endpoint/curve maximum=1-alpha; alpha>1 gives a negative endpoint",
    )
    parameters.add_argument(
        "--compare",
        action="append",
        type=_comparison_specification,
        dest="comparisons",
        default=None,
        metavar="PARAM=VALUE[,PARAM=VALUE...]",
        help="add a comparison curve by overriding selected parameters; repeatable",
    )

    output = parser.add_argument_group("sampling and output")
    output.add_argument(
        "--points",
        type=int,
        default=1000,
        metavar="COUNT",
        help="number of samples in each figure; must be at least 2",
    )
    output.add_argument(
        "--positive-color",
        default="lightsteelblue",
        metavar="COLOR",
        help="fill color where the indicator is positive",
    )
    output.add_argument(
        "--negative-color",
        default="lightgray",
        metavar="COLOR",
        help="fill color where the indicator is negative",
    )
    output.add_argument(
        "--primary-alpha",
        type=float,
        default=COMPARISON_PRIMARY_ALPHA,
        metavar="VALUE",
        help=(
            "refinement-fill opacity for single and comparison plots in the "
            "range (0, 1]"
        ),
    )
    output.add_argument(
        "--secondary-alpha",
        type=float,
        default=COMPARISON_SECONDARY_ALPHA,
        metavar="VALUE",
        help=(
            "unrefinement-fill opacity for single and comparison plots; must "
            "be non-negative and smaller than --primary-alpha"
        ),
    )
    output.add_argument(
        "--no-area-fill",
        action="store_true",
        help="suppress all colored areas and their fill legend entries",
    )
    output.add_argument(
        "--legend-outside",
        action="store_true",
        help="place the legend to the right of the plot",
    )
    output.add_argument(
        "--output-dir",
        type=Path,
        metavar="DIRECTORY",
        help="save one PNG per function; omit to open figures interactively",
    )
    return parser


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse and validate the plotting command line."""

    parser = _argument_parser()
    args = parser.parse_args(arguments)
    args.comparisons = tuple(args.comparisons or ())

    if "all" in args.functions:
        if len(args.functions) != 1:
            parser.error("'all' cannot be combined with individual function names")
        args.functions = TRANSFER_FUNCTIONS
    else:
        args.functions = tuple(args.functions)

    if len(set(args.functions)) != len(args.functions):
        parser.error("function names must not be repeated")

    numeric_options = {
        "--d-min": args.d_min,
        "--d-max": args.d_max,
        "--nlvk-min": args.nlvk_min,
        "--nlvk-max": args.nlvk_max,
        "--weight1": args.weight1,
        "--weight2": args.weight2,
        "--sigma": args.sigma,
        "--alpha": args.alpha,
        "--primary-alpha": args.primary_alpha,
        "--secondary-alpha": args.secondary_alpha,
    }
    non_finite = [name for name, value in numeric_options.items() if not math.isfinite(value)]
    if non_finite:
        parser.error(f"{', '.join(non_finite)} must be finite")

    if args.d_min >= args.d_max:
        parser.error("--d-min must be smaller than --d-max")
    if args.nlvk_min >= args.nlvk_max:
        parser.error("--nlvk-min must be smaller than --nlvk-max")
    if args.weight1 < 0.0 or args.weight2 < 0.0:
        parser.error("--weight1 and --weight2 must be non-negative")
    if args.sigma <= SMALL:
        parser.error("--sigma must be greater than zero")
    if args.alpha < 0.0:
        parser.error("--alpha must be non-negative")
    if not 0.0 < args.primary_alpha <= 1.0:
        parser.error("--primary-alpha must be in the range (0, 1]")
    if not 0.0 <= args.secondary_alpha < args.primary_alpha:
        parser.error(
            "--secondary-alpha must be in the range [0, --primary-alpha)"
        )
    if args.points < 2:
        parser.error("--points must be at least 2")
    if not is_color_like(args.positive_color):
        parser.error("--positive-color must be a valid Matplotlib color")
    if not is_color_like(args.negative_color):
        parser.error("--negative-color must be a valid Matplotlib color")

    if args.comparisons:
        if len(args.functions) != 1:
            parser.error("--compare requires exactly one transfer function")

        function_name = args.functions[0]
        allowed_parameters = FUNCTION_PARAMETERS[function_name]
        if not allowed_parameters:
            parser.error(f"{function_name} has no parameters to compare")

        baseline = {
            name: getattr(args, name)
            for name in COMPARISON_PARAMETERS
        }
        seen_parameter_sets = {
            _effective_parameter_values(function_name, baseline)
        }
        for overrides in args.comparisons:
            irrelevant = set(overrides).difference(allowed_parameters)
            if irrelevant:
                names = ", ".join(sorted(irrelevant))
                parser.error(f"{names} does not affect {function_name}")

            comparison = {**baseline, **overrides}
            if not all(math.isfinite(value) for value in comparison.values()):
                parser.error("comparison parameters must be finite")
            if comparison["weight1"] < 0.0 or comparison["weight2"] < 0.0:
                parser.error("comparison weights must be non-negative")
            if comparison["sigma"] <= SMALL:
                parser.error("comparison sigma must be greater than zero")
            if comparison["alpha"] < 0.0:
                parser.error("comparison alpha must be non-negative")
            effective_values = _effective_parameter_values(
                function_name,
                comparison,
            )
            if effective_values in seen_parameter_sets:
                parser.error("comparison curves must be distinct from earlier curves")
            seen_parameter_sets.add(effective_values)
    return args


def _style_axes(
    axis: plt.Axes,
    title: str,
    x_label: str,
    legend_outside: bool,
) -> None:
    axis.set_title(f"sasRefineIndicator: {title}", fontweight="bold")
    axis.set_xlabel(x_label, fontweight="bold")
    axis.set_ylabel("sasRefineIndicator", fontweight="bold")
    axis.axhline(0.0, color="black", linewidth=0.7)
    axis.grid(which="major", color="lightgray")
    axis.grid(which="minor", color="lightgray", linestyle="--", alpha=0.5)
    axis.minorticks_on()
    if legend_outside:
        axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    else:
        axis.legend(loc="best")


def _decorate_indicator(
    axis: plt.Axes,
    coordinates: np.ndarray,
    indicator: np.ndarray | np.ma.MaskedArray,
    positive_color: str,
    negative_color: str,
    primary_alpha: float,
    secondary_alpha: float,
    fill_areas: bool,
) -> None:
    """Add signed area fills and explicit domain-limit markers."""

    masked_indicator = np.ma.asarray(indicator)
    values = masked_indicator.filled(np.nan)
    active = ~np.ma.getmaskarray(masked_indicator) & np.isfinite(values)
    positive = active & (values > 0.0)
    negative = active & (values < 0.0)
    if fill_areas and np.any(positive):
        axis.fill_between(
            coordinates,
            0.0,
            values,
            where=positive,
            interpolate=True,
            color=positive_color,
            alpha=primary_alpha,
            zorder=1,
            label="_nolegend_",
        )
        _add_area_legend_entry(
            axis,
            "candidate cells for refinement",
            positive_color,
            primary_alpha,
            "black",
        )
    if fill_areas and np.any(negative):
        axis.fill_between(
            coordinates,
            0.0,
            values,
            where=negative,
            interpolate=True,
            color=negative_color,
            alpha=secondary_alpha,
            zorder=1,
            label="_nolegend_",
        )
        _add_area_legend_entry(
            axis,
            "candidate cells for unrefinement",
            negative_color,
            secondary_alpha,
            "black",
        )
    endpoint_indices = (0, -1)
    legend_entry_available = True
    for index in endpoint_indices:
        if not active[index]:
            continue
        label = (
            "current min/max of processed value"
            if legend_entry_available
            else "_nolegend_"
        )
        axis.plot(
            [coordinates[index], coordinates[index]],
            [0.0, values[index]],
            color="black",
            linewidth=0.9,
            linestyle="--",
            alpha=0.65,
            label=label,
        )
        legend_entry_available = False


def _signed_area(
    coordinates: np.ndarray,
    indicator: np.ndarray | np.ma.MaskedArray,
    positive: bool,
) -> float:
    """Return the integrated magnitude on one side of zero."""

    masked_indicator = np.ma.asarray(indicator)
    values = masked_indicator.filled(np.nan)
    active = ~np.ma.getmaskarray(masked_indicator) & np.isfinite(values)
    selected = active & ((values > 0.0) if positive else (values < 0.0))
    magnitudes = np.where(selected, np.abs(values), 0.0)
    widths = np.diff(coordinates)
    return float(np.sum(0.5 * (magnitudes[:-1] + magnitudes[1:]) * widths))


def _fill_comparison_indicator_side(
    axis: plt.Axes,
    coordinates: np.ndarray,
    indicator: np.ndarray | np.ma.MaskedArray,
    color: str,
    positive: bool,
    alpha: float,
) -> None:
    """Fill one signed side of a comparison curve."""

    masked_indicator = np.ma.asarray(indicator)
    values = masked_indicator.filled(np.nan)
    active = ~np.ma.getmaskarray(masked_indicator) & np.isfinite(values)
    selected = active & ((values > 0.0) if positive else (values < 0.0))
    if not np.any(selected):
        return
    axis.fill_between(
        coordinates,
        0.0,
        values,
        where=selected,
        interpolate=True,
        color=color,
        alpha=alpha,
        zorder=0.5,
        label="_nolegend_",
    )


def _fill_comparison_indicators(
    axis: plt.Axes,
    coordinates: np.ndarray,
    curves: list[tuple[str, np.ndarray | np.ma.MaskedArray]],
    primary_alpha: float,
    secondary_alpha: float,
) -> None:
    """Draw larger signed areas first so nested regions remain distinguishable."""

    for positive, alpha in (
        (True, primary_alpha),
        (False, secondary_alpha),
    ):
        ordered_curves = sorted(
            curves,
            key=lambda curve: _signed_area(coordinates, curve[1], positive),
            reverse=True,
        )
        for color, indicator in ordered_curves:
            _fill_comparison_indicator_side(
                axis,
                coordinates,
                indicator,
                color,
                positive,
                alpha,
            )


def _add_area_legend_entry(
    axis: plt.Axes,
    label: str,
    face_color: str,
    face_alpha: float,
    edge_color: str,
) -> None:
    """Add an area proxy with independent face transparency and opaque edge."""

    axis.add_artist(
        Rectangle(
            (0.0, 0.0),
            0.0,
            0.0,
            facecolor=to_rgba(face_color, alpha=face_alpha),
            edgecolor=edge_color,
            label=label,
        )
    )


def _add_comparison_fill_legend_entries(
    axis: plt.Axes,
    primary_alpha: float,
    secondary_alpha: float,
) -> None:
    """Add neutral proxies for signed comparison fills."""

    for alpha, label in (
        (primary_alpha, "candidate cells for refinement"),
        (secondary_alpha, "candidate cells for unrefinement"),
    ):
        _add_area_legend_entry(
            axis,
            label,
            "gray",
            alpha,
            "dimgray",
        )


def _format_parameter(value: float) -> str:
    """Format a float compactly without merging distinct values."""

    text = repr(float(value))
    return text[:-2] if text.endswith(".0") else text


def _curve_values_and_label(
    function_name: str,
    coordinates: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray | np.ma.MaskedArray, str]:
    """Evaluate one curve and build its parameter label."""

    if function_name == "basic":
        return basic_transfer(coordinates), r"$f(d)=d$ for $d>0$"
    if function_name == "constant":
        return (
            constant_transfer(coordinates, args.weight2),
            rf"$w_2={_format_parameter(args.weight2)}$ for $d>0$",
        )
    if function_name == "oddScaler":
        return (
            odd_scaler_transfer(coordinates, args.weight1, args.sigma),
            (
                rf"$w_1={_format_parameter(args.weight1)}$, "
                rf"$\sigma={_format_parameter(args.sigma)}$"
            ),
        )
    if function_name == "safeScaler":
        indicator = safe_scaler_transfer(
            coordinates,
            args.weight1,
            args.sigma,
            args.alpha,
        )
        label = (
            rf"$w_1={_format_parameter(args.weight1)}$, "
            rf"$\alpha={_format_parameter(args.alpha)}$, "
            rf"$\sigma={_format_parameter(args.sigma)}$"
        )
        return indicator, label
    if function_name == "gaussSinkCore":
        return (
            gauss_sink_core_transfer(coordinates, args.weight2, args.sigma),
            (
                rf"$w_2={_format_parameter(args.weight2)}$, "
                rf"$\sigma={_format_parameter(args.sigma)}$"
            ),
        )
    if function_name == "gaussSinkPeriphery":
        return (
            gauss_sink_periphery_transfer(coordinates, args.weight2, args.sigma),
            (
                rf"$w_2={_format_parameter(args.weight2)}$, "
                rf"$\sigma={_format_parameter(args.sigma)}$"
            ),
        )
    raise ValueError(f"unknown transfer function: {function_name}")


def _greatest_absolute_curve_indicator(
    indicators: list[np.ndarray | np.ma.MaskedArray],
) -> np.ma.MaskedArray:
    """Return the finite, unmasked value with greatest magnitude per coordinate."""

    stacked = np.ma.vstack(
        [np.ma.masked_invalid(np.ma.asarray(indicator)) for indicator in indicators]
    )
    greatest_indices = np.ma.argmax(np.ma.abs(stacked), axis=0)
    coordinate_indices = np.arange(stacked.shape[1])
    return stacked[greatest_indices, coordinate_indices]


def _mark_curve_maximum(
    axis: plt.Axes,
    coordinates: np.ndarray,
    plotted_values: np.ndarray | np.ma.MaskedArray,
    color: str,
) -> None:
    """Mark the highest finite plotted sample without adding a legend entry."""

    values = np.ma.asarray(plotted_values).filled(np.nan)
    finite_indices = np.flatnonzero(np.isfinite(values))
    if finite_indices.size == 0:
        return

    maximum_index = finite_indices[np.argmax(values[finite_indices])]
    axis.plot(
        [coordinates[maximum_index]],
        [values[maximum_index]],
        color=color,
        marker="o",
        linestyle="None",
        label="_nolegend_",
        zorder=3,
    )


def create_figure(function_name: str, args: argparse.Namespace) -> plt.Figure:
    """Create one independent figure for a selected transfer function."""

    figure, axis = plt.subplots(figsize=(9, 5.5))

    if function_name in GAUSS_SINK_FUNCTIONS:
        coordinates = np.linspace(args.nlvk_min, args.nlvk_max, args.points)
        x_label = (
            r"$C_1/C_2$ (core; $L_{vk}/C_2=\max(C_1/C_2,1)$)"
            if function_name == "gaussSinkCore"
            else r"$L_{vk}/L_{vk,\mathrm{ref}}$ (periphery)"
        )
        reference_coordinate = 1.0
    else:
        coordinates = np.linspace(args.d_min, args.d_max, args.points)
        x_label = r"$d = C_2 - C_1$"
        reference_coordinate = 0.0

    indicator, label = _curve_values_and_label(function_name, coordinates, args)
    plotted_indicator = indicator
    baseline_style = (
        {"linestyle": "-"} if args.comparisons else {"color": "black"}
    )
    baseline_line, = axis.plot(
        coordinates,
        plotted_indicator,
        label=label,
        **baseline_style,
    )
    comparison_curves = [(baseline_line.get_color(), plotted_indicator)]
    marker_indicators = [indicator]

    for overrides in args.comparisons:
        comparison_args = argparse.Namespace(**{**vars(args), **overrides})
        comparison_indicator, comparison_label = _curve_values_and_label(
            function_name,
            coordinates,
            comparison_args,
        )
        plotted_comparison = comparison_indicator
        comparison_line, = axis.plot(
            coordinates,
            plotted_comparison,
            linestyle="-",
            label=comparison_label,
        )
        comparison_curves.append((comparison_line.get_color(), plotted_comparison))
        marker_indicators.append(comparison_indicator)

    if args.comparisons and not args.no_area_fill:
        _fill_comparison_indicators(
            axis,
            coordinates,
            comparison_curves,
            args.primary_alpha,
            args.secondary_alpha,
        )
        _add_comparison_fill_legend_entries(
            axis,
            args.primary_alpha,
            args.secondary_alpha,
        )

    if function_name == "safeScaler" and args.comparisons:
        for color, plotted_values in comparison_curves:
            _mark_curve_maximum(axis, coordinates, plotted_values, color)
        axis.plot(
            [],
            [],
            color="black",
            marker="o",
            linestyle="None",
            label="transfer function maximum value",
        )

    marker_indicator = (
        _greatest_absolute_curve_indicator(marker_indicators)
        if args.comparisons
        else indicator
    )
    _decorate_indicator(
        axis,
        coordinates,
        marker_indicator,
        args.positive_color,
        args.negative_color,
        args.primary_alpha,
        args.secondary_alpha,
        fill_areas=not bool(args.comparisons) and not args.no_area_fill,
    )
    axis.axvline(
        reference_coordinate,
        color="black",
        linewidth=0.7,
        linestyle=":",
    )
    if function_name in {"basic", "constant"}:
        axis.set_xlim(coordinates[0], coordinates[-1])
    _style_axes(
        axis,
        function_name,
        x_label,
        args.legend_outside,
    )

    figure.tight_layout()
    return figure


def main(arguments: Sequence[str] | None = None) -> list[Path]:
    """Create requested figures and return paths of any saved PNG files."""

    args = parse_arguments(arguments)
    output_paths: list[Path] = []
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    for function_name in args.functions:
        figure = create_figure(function_name, args)
        if args.output_dir is not None:
            output_path = args.output_dir / f"{function_name}.png"
            save_options = {"bbox_inches": "tight"} if args.legend_outside else {}
            figure.savefig(output_path, dpi=200, **save_options)
            plt.close(figure)
            output_paths.append(output_path)

    if args.output_dir is None:
        plt.show()
    return output_paths


if __name__ == "__main__":
    main()
