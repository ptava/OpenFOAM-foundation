import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import new_transfer_functions as transfer_plots


class TransferFunctionApiTests(unittest.TestCase):
    def test_exposes_every_openfoam_transfer_function(self):
        expected = {
            "basic_transfer",
            "constant_transfer",
            "odd_scaler_transfer",
            "safe_scaler_transfer",
            "gauss_sink_core_transfer",
            "gauss_sink_periphery_transfer",
        }

        missing = sorted(name for name in expected if not hasattr(transfer_plots, name))

        self.assertEqual([], missing)

    def test_basic_masks_nonpositive_values(self):
        result = transfer_plots.basic_transfer(np.array([-1.0, 0.0, 2.0]))

        np.testing.assert_array_equal([True, True, False], result.mask)
        self.assertEqual(2.0, result[2])

    def test_constant_uses_weight2_and_masks_nonpositive_values(self):
        result = transfer_plots.constant_transfer(
            np.array([-1.0, 0.0, 2.0]),
            weight2=2.5,
        )

        np.testing.assert_array_equal([True, True, False], result.mask)
        self.assertEqual(2.5, result[2])

    def test_odd_scaler_matches_openfoam_shift_and_whole_curve_normalization(self):
        result = transfer_plots.odd_scaler_transfer(
            np.array([-1.0, 0.0, 2.0]),
            weight1=0.0,
            sigma=1.0,
        )

        np.testing.assert_allclose(
            [-0.22752711696, 0.0, 1.0],
            result,
            rtol=1e-7,
        )

    def test_safe_scaler_normalizes_negative_and_positive_values_together(self):
        result = transfer_plots.safe_scaler_transfer(
            np.array([-3.0, 3.0]),
            weight1=0.3,
            sigma=0.05,
            alpha=0.6,
        )

        np.testing.assert_allclose([-3.64095130, 1.0], result, rtol=1e-6)

    def test_safe_scaler_alpha_one_delegates_to_odd_scaler_with_user_sigma(self):
        result = transfer_plots.safe_scaler_transfer(
            np.array([-1.0, 1.0]),
            weight1=0.0,
            sigma=1.0,
            alpha=1.0,
        )

        np.testing.assert_allclose([-1.0, 1.0], result, rtol=1e-12)

    def test_gauss_sink_periphery_matches_normalized_length_scale_formula(self):
        result = transfer_plots.gauss_sink_periphery_transfer(
            np.array([0.0, 1.0, 2.0]),
            weight2=0.5,
            sigma=1.0,
        )

        np.testing.assert_allclose(
            [0.1065306597, 1.0, 0.1065306597],
            result,
            rtol=1e-9,
        )

    def test_gauss_sink_core_clamps_c1_over_c2_to_plateau_below_one(self):
        result = transfer_plots.gauss_sink_core_transfer(
            np.array([0.0, 0.5, 1.0, 2.0]),
            weight2=0.5,
            sigma=1.0,
        )

        np.testing.assert_allclose(
            [1.0, 1.0, 1.0, 0.1065306597],
            result,
            rtol=1e-9,
        )


class CommandLineTests(unittest.TestCase):
    def test_accepts_distinct_core_and_periphery_gauss_sink_selectors(self):
        args = transfer_plots.parse_arguments(
            ["gaussSinkCore", "gaussSinkPeriphery"]
        )

        self.assertEqual(
            ("gaussSinkCore", "gaussSinkPeriphery"),
            args.functions,
        )

    def test_rejects_removed_ambiguous_gauss_sink_selector(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            transfer_plots.parse_arguments(["gaussSink"])

    def test_comparisons_default_to_empty(self):
        args = transfer_plots.parse_arguments(["oddScaler"])

        self.assertEqual((), getattr(args, "comparisons", None))

    def test_accepts_repeated_comparison_parameter_overrides(self):
        try:
            args = transfer_plots.parse_arguments(
                [
                    "safeScaler",
                    "--weight1",
                    "0.1",
                    "--alpha",
                    "0.6",
                    "--compare",
                    "alpha=0.3",
                    "--compare",
                    "alpha=1,sigma=0.1",
                ]
            )
        except SystemExit as error:
            self.fail(f"comparison arguments were rejected with exit code {error.code}")

        self.assertEqual(
            ({"alpha": 0.3}, {"alpha": 1.0, "sigma": 0.1}),
            args.comparisons,
        )

    def test_rejects_ambiguous_or_invalid_comparison_requests(self):
        invalid_argument_sets = (
            ["oddScaler", "safeScaler", "--compare", "weight1=0.2"],
            ["all", "--compare", "weight1=0.2"],
            ["basic", "--compare", "weight1=0.2"],
            ["constant", "--compare", "weight1=0.2"],
            ["safeScaler", "--compare", "alpha"],
            ["safeScaler", "--compare", "unknown=1"],
            ["safeScaler", "--compare", "alpha=0.3,alpha=0.4"],
            ["safeScaler", "--compare", "alpha=text"],
            ["safeScaler", "--compare", "alpha=nan"],
            ["oddScaler", "--compare", "weight1=-0.1"],
            ["gaussSinkCore", "--compare", "weight2=-0.1"],
            ["oddScaler", "--compare", "sigma=0"],
            ["safeScaler", "--compare", "alpha=0"],
            ["safeScaler", "--compare", "alpha=1.1"],
            ["oddScaler", "--compare", "weight1=0"],
            ["safeScaler", "--compare", "sigma=0.1"],
            ["safeScaler", "--compare", "weight1=0.2,sigma=0.1"],
            [
                "safeScaler",
                "--compare",
                "alpha=0.3",
                "--compare",
                "alpha=0.3,sigma=0.1",
            ],
        )

        for arguments in invalid_argument_sets:
            with (
                self.subTest(arguments=arguments),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                transfer_plots.parse_arguments(arguments)

    def test_legend_outside_is_opt_in(self):
        inside_args = transfer_plots.parse_arguments(["basic"])

        self.assertIs(False, getattr(inside_args, "legend_outside", None))

        try:
            outside_args = transfer_plots.parse_arguments(["basic", "--legend-outside"])
        except SystemExit as error:
            self.fail(f"--legend-outside was rejected with exit code {error.code}")
        self.assertTrue(outside_args.legend_outside)

    def test_fill_colors_default_to_blue_and_grey(self):
        args = transfer_plots.parse_arguments(["basic"])

        self.assertEqual(
            ("blue", "grey"),
            (
                getattr(args, "positive_color", None),
                getattr(args, "negative_color", None),
            ),
        )

    def test_accepts_matplotlib_color_names_and_hex_values(self):
        args = transfer_plots.parse_arguments(
            [
                "gaussSinkPeriphery",
                "--positive-color",
                "navy",
                "--negative-color",
                "#808080",
            ]
        )

        self.assertEqual(("navy", "#808080"), (args.positive_color, args.negative_color))

    def test_accepts_comparison_primary_and_secondary_alpha_values(self):
        args = transfer_plots.parse_arguments(
            [
                "oddScaler",
                "--compare",
                "weight1=0.3",
                "--primary-alpha",
                "0.22",
                "--secondary-alpha",
                "0.07",
            ]
        )

        self.assertEqual((0.22, 0.07), (args.primary_alpha, args.secondary_alpha))

    def test_area_fill_suppression_is_opt_in(self):
        default_args = transfer_plots.parse_arguments(["oddScaler"])
        disabled_args = transfer_plots.parse_arguments(
            ["oddScaler", "--no-area-fill"]
        )

        self.assertFalse(default_args.no_area_fill)
        self.assertTrue(disabled_args.no_area_fill)

    def test_rejects_invalid_fill_colors(self):
        for option in ("--positive-color", "--negative-color"):
            with (
                self.subTest(option=option),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                transfer_plots.parse_arguments(["basic", option, "not-a-color"])

    def test_help_explains_function_semantics_parameter_ownership_and_defaults(self):
        environment = os.environ.copy()
        environment.update(MPLCONFIGDIR="/tmp", PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run(
            [sys.executable, str(Path(__file__).with_name("new_transfer_functions.py")), "--help"],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )

        help_text = result.stdout
        normalized_help = " ".join(help_text.split())
        self.assertEqual(0, result.returncode, result.stderr)
        for required_text in (
            "TRANSFER FUNCTIONS",
            "basic (core)",
            "G=d when d>0; otherwise G=-GREAT",
            "constant (core)",
            "G=weight2 when d>0; otherwise G=-GREAT",
            "oddScaler (core)",
            "u=d+weight1*dMax",
            "G=u*(1-exp(-u^2/(2*sigma^2)))",
            "safeScaler (core)",
            "G=G/(1+kStar*u^2) only where u>0",
            "gaussSinkCore (core)",
            "nLvk=max(C1/C2,1)",
            "gaussSinkPeriphery (periphery)",
            "G=exp(-(nLvk-1)^2/(2*sigma^2))-weight2*(nLvk-1)^2",
            "entire curve, including negative values, by max(G)",
            "oddScaler and safeScaler",
            "parallel maximum of C2-C1 over the selected cellZone",
            "Lvk/C2=max(C1/C2,1)",
            "Lvk/LvkRef",
            "periphery always uses gaussSinkPeriphery",
            "plots leave that inactive interval blank",
            "EXAMPLES",
            "python3 new_transfer_functions.py oddScaler",
            "omit to open figures interactively",
            "--legend-outside",
            "place the legend to the right of the plot",
            "COMPARISON CURVES",
            "--compare alpha=0.3",
            "requires exactly one transfer function",
            "All curves use the Matplotlib color cycle and continuous lines",
            "Comparison fills use each curve's color",
            "--primary-alpha",
            "--secondary-alpha",
            "refinement-fill opacity for single and comparison plots",
            "unrefinement-fill opacity for single and comparison plots",
            "applied directly without curve-count scaling",
            "--no-area-fill",
            "suppress all colored areas",
            "neutral gray legend entries use the selected primary and secondary opacities",
            "current min/max of processed value",
        ):
            with self.subTest(required_text=required_text):
                self.assertIn(required_text, normalized_help)
        for default in ("-3.0", "3.0", "0.0", "2.0", "1.0", "0.05", "0.6", "1000"):
            with self.subTest(default=default):
                self.assertIn(f"default: {default}", normalized_help)
        self.assertNotIn("default: None", normalized_help)

    def test_accepts_individual_names_and_plot_parameters(self):
        args = transfer_plots.parse_arguments(
            [
                "oddScaler",
                "gaussSinkCore",
                "gaussSinkPeriphery",
                "--d-min",
                "-2",
                "--d-max",
                "5",
                "--nlvk-min",
                "0.5",
                "--nlvk-max",
                "3",
                "--weight1",
                "2",
                "--weight2",
                "4",
                "--sigma",
                "0.2",
                "--alpha",
                "0.7",
                "--points",
                "50",
            ]
        )

        self.assertEqual(
            ("oddScaler", "gaussSinkCore", "gaussSinkPeriphery"),
            args.functions,
        )
        self.assertEqual((-2.0, 5.0), (args.d_min, args.d_max))
        self.assertEqual((0.5, 3.0), (args.nlvk_min, args.nlvk_max))
        self.assertEqual((2.0, 4.0), (args.weight1, args.weight2))
        self.assertEqual((0.2, 0.7, 50), (args.sigma, args.alpha, args.points))

    def test_all_expands_to_every_function_in_openfoam_order(self):
        args = transfer_plots.parse_arguments(["all"])

        self.assertEqual(
            (
                "basic",
                "constant",
                "oddScaler",
                "safeScaler",
                "gaussSinkCore",
                "gaussSinkPeriphery",
            ),
            args.functions,
        )

    def test_rejects_all_mixed_with_individual_names(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            transfer_plots.parse_arguments(["all", "basic"])

    def test_rejects_duplicate_function_names(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            transfer_plots.parse_arguments(["basic", "basic"])

    def test_rejects_invalid_plot_parameters(self):
        invalid_argument_sets = (
            ["basic", "--d-min", "1", "--d-max", "1"],
            ["gaussSinkCore", "--nlvk-min", "2", "--nlvk-max", "1"],
            ["oddScaler", "--weight1", "-0.1"],
            ["constant", "--weight2", "-0.1"],
            ["gaussSinkPeriphery", "--sigma", "0"],
            ["safeScaler", "--alpha", "0"],
            ["safeScaler", "--alpha", "1.1"],
            ["basic", "--points", "1"],
            ["basic", "--d-min", "nan"],
            ["basic", "--d-max", "inf"],
            ["gaussSinkCore", "--nlvk-min", "-inf"],
            ["gaussSinkPeriphery", "--nlvk-max", "nan"],
            ["oddScaler", "--weight1", "nan"],
            ["constant", "--weight2", "inf"],
            ["gaussSinkCore", "--sigma", "nan"],
            ["safeScaler", "--alpha", "nan"],
            ["oddScaler", "--primary-alpha", "-0.1"],
            ["oddScaler", "--primary-alpha", "1.1"],
            ["oddScaler", "--secondary-alpha", "-0.1"],
            ["oddScaler", "--secondary-alpha", "1.1"],
            [
                "oddScaler",
                "--primary-alpha",
                "0.2",
                "--secondary-alpha",
                "0.2",
            ],
            [
                "oddScaler",
                "--primary-alpha",
                "0.2",
                "--secondary-alpha",
                "0.3",
            ],
        )

        for arguments in invalid_argument_sets:
            with (
                self.subTest(arguments=arguments),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                transfer_plots.parse_arguments(arguments)

    def test_output_directory_receives_one_figure_per_selected_function(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory) / "plots"

            paths = transfer_plots.main(
                [
                    "basic",
                    "safeScaler",
                    "gaussSinkCore",
                    "gaussSinkPeriphery",
                    "--points",
                    "25",
                    "--output-dir",
                    str(output_directory),
                ]
            )

            self.assertEqual(
                [
                    output_directory / "basic.png",
                    output_directory / "safeScaler.png",
                    output_directory / "gaussSinkCore.png",
                    output_directory / "gaussSinkPeriphery.png",
                ],
                paths,
            )
            self.assertTrue(all(path.stat().st_size > 0 for path in paths))

    def test_outside_legend_is_present_and_unclipped_in_saved_png(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory) / "plots"
            args = transfer_plots.parse_arguments(["oddScaler", "--legend-outside"])
            figure = transfer_plots.create_figure("oddScaler", args)
            figure.set_dpi(200)
            figure.canvas.draw()
            tight_bounds = figure.get_tightbbox(figure.canvas.get_renderer())
            padding = matplotlib.rcParams["savefig.pad_inches"]
            expected_width = round((tight_bounds.width + 2.0 * padding) * 200)
            expected_height = round((tight_bounds.height + 2.0 * padding) * 200)
            matplotlib.pyplot.close(figure)

            [output_path] = transfer_plots.main(
                [
                    "oddScaler",
                    "--legend-outside",
                    "--output-dir",
                    str(output_directory),
                ]
            )

            image = matplotlib.pyplot.imread(output_path)
            rgb = image[..., :3]
            background = rgb[0, 0]
            non_background = np.any(np.abs(rgb - background) > 0.05, axis=2)
            rows, columns = np.nonzero(non_background)

            self.assertAlmostEqual(expected_width, image.shape[1], delta=2)
            self.assertAlmostEqual(expected_height, image.shape[0], delta=2)
            self.assertGreater(columns.max(), 0.8 * image.shape[1])
            self.assertLess(columns.max(), image.shape[1] - 1)
            self.assertLess(rows.max(), image.shape[0] - 1)


class PlotAppearanceTests(unittest.TestCase):
    def test_core_gauss_sink_plot_has_plateau_for_c1_over_c2_below_one(self):
        args = transfer_plots.parse_arguments(
            [
                "gaussSinkCore",
                "--nlvk-min",
                "0",
                "--nlvk-max",
                "2",
                "--weight2",
                "0.5",
                "--sigma",
                "1",
                "--points",
                "5",
            ]
        )

        figure = transfer_plots.create_figure("gaussSinkCore", args)
        curve = figure.axes[0].lines[0]

        np.testing.assert_allclose([0.0, 0.5, 1.0, 1.5, 2.0], curve.get_xdata())
        np.testing.assert_allclose(
            [1.0, 1.0, 1.0, 0.7574969026, 0.1065306597],
            curve.get_ydata(),
            rtol=1e-9,
        )
        matplotlib.pyplot.close(figure)

    def test_periphery_gauss_sink_plot_retains_two_sided_curve(self):
        args = transfer_plots.parse_arguments(
            [
                "gaussSinkPeriphery",
                "--nlvk-min",
                "0",
                "--nlvk-max",
                "2",
                "--weight2",
                "0.5",
                "--sigma",
                "1",
                "--points",
                "3",
            ]
        )

        figure = transfer_plots.create_figure("gaussSinkPeriphery", args)
        curve = figure.axes[0].lines[0]

        np.testing.assert_allclose([0.0, 1.0, 2.0], curve.get_xdata())
        np.testing.assert_allclose(
            [0.1065306597, 1.0, 0.1065306597],
            curve.get_ydata(),
            rtol=1e-9,
        )
        matplotlib.pyplot.close(figure)

    def test_single_curve_retains_configured_linestyle(self):
        args = transfer_plots.parse_arguments(["oddScaler"])

        with matplotlib.rc_context({"lines.linestyle": "--"}):
            figure = transfer_plots.create_figure("oddScaler", args)

        self.assertEqual("--", figure.axes[0].lines[0].get_linestyle())
        matplotlib.pyplot.close(figure)

    def test_comparison_uses_solid_curves_with_matching_signed_area_fills(self):
        args = transfer_plots.parse_arguments(
            [
                "safeScaler",
                "--weight1",
                "0.1",
                "--alpha",
                "0.6",
                "--points",
                "101",
                "--primary-alpha",
                "0.22",
                "--secondary-alpha",
                "0.07",
                "--compare",
                "alpha=0.3",
                "--compare",
                "alpha=1,sigma=0.1",
            ]
        )
        with matplotlib.rc_context({"lines.linestyle": "--"}):
            figure = transfer_plots.create_figure("safeScaler", args)
        axis = figure.axes[0]
        expected_labels = (
            r"$w_1=0.1$, $\alpha=0.6$",
            r"$w_1=0.1$, $\alpha=0.3$",
            r"$w_1=0.1$, $\alpha=1$, user $\sigma$ active",
        )
        curve_lines = [
            line for line in axis.lines if line.get_label() in expected_labels
        ]

        self.assertEqual(expected_labels, tuple(line.get_label() for line in curve_lines))
        self.assertEqual(3, len({line.get_color() for line in curve_lines}))
        self.assertNotIn("black", {line.get_color() for line in curve_lines})
        self.assertEqual({"-"}, {line.get_linestyle() for line in curve_lines})

        d = np.asarray(curve_lines[0].get_xdata())
        np.testing.assert_allclose(
            transfer_plots.safe_scaler_transfer(d, 0.1, 0.05, 0.3),
            curve_lines[1].get_ydata(),
        )
        np.testing.assert_allclose(
            transfer_plots.safe_scaler_transfer(d, 0.1, 0.1, 1.0),
            curve_lines[2].get_ydata(),
        )
        self.assertEqual(6, len(axis.collections))
        curve_rgbs = {
            matplotlib.colors.to_rgba(curve.get_color())[:3]
            for curve in curve_lines
        }
        positive_fills = axis.collections[:3]
        negative_fills = axis.collections[3:]
        self.assertEqual(
            curve_rgbs,
            {tuple(fill.get_facecolor()[0][:3]) for fill in positive_fills},
        )
        self.assertEqual(
            curve_rgbs,
            {tuple(fill.get_facecolor()[0][:3]) for fill in negative_fills},
        )
        for fill in positive_fills:
            self.assertAlmostEqual(0.22, fill.get_facecolor()[0][3])
            self.assertLess(fill.get_zorder(), curve_lines[0].get_zorder())
            self.assertEqual("_nolegend_", fill.get_label())
        for fill in negative_fills:
            self.assertAlmostEqual(0.07, fill.get_facecolor()[0][3])
            self.assertLess(fill.get_zorder(), curve_lines[0].get_zorder())
            self.assertEqual("_nolegend_", fill.get_label())
        boundary_markers = [
            line
            for line in axis.lines
            if line.get_linestyle() == "--"
            and np.allclose(line.get_xdata(), line.get_xdata()[0])
        ]
        self.assertEqual(2, len(boundary_markers))
        self.assertEqual(
            1,
            axis.get_legend_handles_labels()[1].count(
                "current min/max of processed value"
            ),
        )
        matplotlib.pyplot.close(figure)

    def test_comparison_uses_opposite_area_order_for_positive_and_negative_values(self):
        args = transfer_plots.parse_arguments(
            [
                "oddScaler",
                "--d-min",
                "-1",
                "--d-max",
                "1",
                "--sigma",
                "1",
                "--compare",
                "weight1=0.3",
                "--compare",
                "weight1=0.5",
            ]
        )

        figure = transfer_plots.create_figure("oddScaler", args)
        axis = figure.axes[0]
        curve_colors = [line.get_color() for line in axis.lines[:3]]
        fill_colors = [
            matplotlib.colors.to_hex(collection.get_facecolor()[0])
            for collection in axis.collections
        ]
        expected_positive = [
            matplotlib.colors.to_hex(color)
            for color in reversed(curve_colors)
        ]
        expected_negative = [
            matplotlib.colors.to_hex(color)
            for color in curve_colors
        ]

        self.assertEqual(expected_positive + expected_negative, fill_colors)
        matplotlib.pyplot.close(figure)

    def test_no_area_fill_suppresses_fills_and_their_legend_entries(self):
        argument_sets = (
            ["oddScaler", "--no-area-fill"],
            [
                "oddScaler",
                "--compare",
                "weight1=0.3",
                "--no-area-fill",
            ],
        )

        for arguments in argument_sets:
            with self.subTest(arguments=arguments):
                args = transfer_plots.parse_arguments(arguments)
                figure = transfer_plots.create_figure("oddScaler", args)
                axis = figure.axes[0]

                self.assertEqual(0, len(axis.collections))
                self.assertFalse(
                    any(
                        label.startswith("candidate cells for")
                        for label in axis.get_legend_handles_labels()[1]
                    )
                )
                matplotlib.pyplot.close(figure)

    def test_comparison_fill_opacity_does_not_change_with_curve_count(self):
        two_curve_args = transfer_plots.parse_arguments(
            [
                "oddScaler",
                "--primary-alpha",
                "0.22",
                "--secondary-alpha",
                "0.07",
                "--compare",
                "weight1=0.2",
            ]
        )
        three_curve_args = transfer_plots.parse_arguments(
            [
                "oddScaler",
                "--primary-alpha",
                "0.22",
                "--secondary-alpha",
                "0.07",
                "--compare",
                "weight1=0.2",
                "--compare",
                "weight1=0.4",
            ]
        )

        two_curve_figure = transfer_plots.create_figure("oddScaler", two_curve_args)
        three_curve_figure = transfer_plots.create_figure(
            "oddScaler",
            three_curve_args,
        )
        two_curve_alpha = two_curve_figure.axes[0].collections[0].get_facecolor()[0][3]
        three_curve_alpha = (
            three_curve_figure.axes[0].collections[0].get_facecolor()[0][3]
        )

        self.assertAlmostEqual(0.22, two_curve_alpha)
        self.assertAlmostEqual(0.22, three_curve_alpha)
        matplotlib.pyplot.close(two_curve_figure)
        matplotlib.pyplot.close(three_curve_figure)

    def test_single_curve_areas_and_legend_use_selected_transparencies(self):
        args = transfer_plots.parse_arguments(
            [
                "oddScaler",
                "--positive-color",
                "navy",
                "--negative-color",
                "gold",
                "--primary-alpha",
                "0.52",
                "--secondary-alpha",
                "0.17",
            ]
        )

        figure = transfer_plots.create_figure("oddScaler", args)
        axis = figure.axes[0]
        np.testing.assert_allclose(
            [
                matplotlib.colors.to_rgba("navy", alpha=0.52),
                matplotlib.colors.to_rgba("gold", alpha=0.17),
            ],
            [collection.get_facecolor()[0] for collection in axis.collections],
        )
        handles, labels = axis.get_legend_handles_labels()
        signed_entries = {
            label: handle
            for handle, label in zip(handles, labels)
            if label.startswith("candidate cells for")
        }

        np.testing.assert_allclose(
            matplotlib.colors.to_rgba("navy", alpha=0.52),
            signed_entries["candidate cells for refinement"].get_facecolor(),
        )
        np.testing.assert_allclose(
            matplotlib.colors.to_rgba("gold", alpha=0.17),
            signed_entries["candidate cells for unrefinement"].get_facecolor(),
        )
        for handle in signed_entries.values():
            np.testing.assert_allclose(
                matplotlib.colors.to_rgba("black"),
                handle.get_edgecolor(),
            )
        matplotlib.pyplot.close(figure)

    def test_comparison_legend_uses_neutral_color_at_selected_transparencies(self):
        args = transfer_plots.parse_arguments(
            [
                "oddScaler",
                "--compare",
                "weight1=0.2",
                "--primary-alpha",
                "0.22",
                "--secondary-alpha",
                "0.07",
            ]
        )

        figure = transfer_plots.create_figure("oddScaler", args)
        handles, labels = figure.axes[0].get_legend_handles_labels()
        signed_entries = {
            label: handle
            for handle, label in zip(handles, labels)
            if label.startswith("candidate cells for")
        }

        self.assertEqual(
            {
                "candidate cells for refinement",
                "candidate cells for unrefinement",
            },
            set(signed_entries),
        )
        np.testing.assert_allclose(
            matplotlib.colors.to_rgba("gray", alpha=0.22),
            signed_entries["candidate cells for refinement"].get_facecolor(),
        )
        np.testing.assert_allclose(
            matplotlib.colors.to_rgba("gray", alpha=0.07),
            signed_entries["candidate cells for unrefinement"].get_facecolor(),
        )
        for handle in signed_entries.values():
            np.testing.assert_allclose(
                matplotlib.colors.to_rgba("dimgray"),
                handle.get_edgecolor(),
            )
        matplotlib.pyplot.close(figure)

    def test_safe_scaler_comparison_marks_each_curve_maximum_in_its_color(self):
        args = transfer_plots.parse_arguments(
            [
                "safeScaler",
                "--weight1",
                "0.1",
                "--alpha",
                "0.6",
                "--points",
                "101",
                "--compare",
                "alpha=0.3",
                "--compare",
                "alpha=1,sigma=0.1",
            ]
        )

        figure = transfer_plots.create_figure("safeScaler", args)
        axis = figure.axes[0]
        curve_lines = [
            line
            for line in axis.lines
            if line.get_label().startswith(r"$w_1=")
        ]
        maximum_markers = [
            line
            for line in axis.lines
            if line.get_marker() == "o" and len(line.get_xdata()) == 1
        ]

        self.assertEqual(3, len(curve_lines))
        self.assertEqual(3, len(maximum_markers))
        for curve, marker in zip(curve_lines, maximum_markers):
            curve_x = np.asarray(curve.get_xdata(), dtype=float)
            curve_y = np.asarray(curve.get_ydata(), dtype=float)
            maximum_index = int(np.nanargmax(curve_y))
            np.testing.assert_allclose(marker.get_xdata(), [curve_x[maximum_index]])
            np.testing.assert_allclose(marker.get_ydata(), [curve_y[maximum_index]])
            self.assertAlmostEqual(1.0, float(marker.get_ydata()[0]))
            self.assertEqual(curve.get_color(), marker.get_color())
            self.assertEqual("_nolegend_", marker.get_label())

        matplotlib.pyplot.close(figure)

    def test_safe_scaler_comparison_legend_explains_maximum_dots(self):
        args = transfer_plots.parse_arguments(
            ["safeScaler", "--compare", "alpha=0.3"]
        )

        figure = transfer_plots.create_figure("safeScaler", args)
        handles, labels = figure.axes[0].get_legend_handles_labels()
        maximum_handles = [
            handle
            for handle, label in zip(handles, labels)
            if label == "transfer function maximum value"
        ]

        self.assertEqual(1, len(maximum_handles))
        self.assertEqual("o", maximum_handles[0].get_marker())
        self.assertEqual("black", maximum_handles[0].get_color())
        self.assertEqual("None", maximum_handles[0].get_linestyle())
        matplotlib.pyplot.close(figure)

    def test_maximum_markers_are_limited_to_safe_scaler_comparisons(self):
        args = transfer_plots.parse_arguments(
            ["oddScaler", "--compare", "weight1=0.2"]
        )

        figure = transfer_plots.create_figure("oddScaler", args)

        self.assertFalse(
            any(line.get_marker() == "o" for line in figure.axes[0].lines)
        )
        matplotlib.pyplot.close(figure)

    def test_each_parameterized_function_supports_a_comparison_curve(self):
        cases = (
            ("constant", "weight2=2", r"$w_2=2$ for $d>0$"),
            ("oddScaler", "weight1=0.2", r"$w_1=0.2$, $\sigma=0.05$"),
            (
                "safeScaler",
                "alpha=0.3",
                r"$w_1=0$, $\alpha=0.3$",
            ),
            ("gaussSinkCore", "weight2=0.5", r"$w_2=0.5$, $\sigma=0.05$"),
            (
                "gaussSinkPeriphery",
                "weight2=0.5",
                r"$w_2=0.5$, $\sigma=0.05$",
            ),
        )

        for function_name, comparison, expected_label in cases:
            with self.subTest(function_name=function_name):
                args = transfer_plots.parse_arguments(
                    [function_name, "--compare", comparison]
                )
                figure = transfer_plots.create_figure(function_name, args)

                self.assertIn(
                    expected_label,
                    [line.get_label() for line in figure.axes[0].lines],
                )
                matplotlib.pyplot.close(figure)

    def test_close_comparison_values_receive_distinct_legend_labels(self):
        cases = (
            (
                "constant",
                "weight2=1.0000001",
                r"$w_2=1.0000001$ for $d>0$",
            ),
            (
                "oddScaler",
                "sigma=0.05000001",
                r"$w_1=0$, $\sigma=0.05000001$",
            ),
            (
                "safeScaler",
                "alpha=0.6000001",
                r"$w_1=0$, $\alpha=0.6000001$",
            ),
            (
                "gaussSinkPeriphery",
                "sigma=0.05000001",
                r"$w_2=1$, $\sigma=0.05000001$",
            ),
        )

        for function_name, comparison, expected_label in cases:
            with self.subTest(function_name=function_name):
                args = transfer_plots.parse_arguments(
                    [function_name, "--compare", comparison]
                )
                figure = transfer_plots.create_figure(function_name, args)
                parameter_labels = [
                    line.get_label()
                    for line in figure.axes[0].lines[:2]
                ]

                self.assertEqual(2, len(set(parameter_labels)))
                self.assertIn(expected_label, parameter_labels)
                matplotlib.pyplot.close(figure)

    def test_plot_labels_use_mathematical_notation(self):
        cases = (
            ("basic", [], r"$f(d)=d$ for $d>0$"),
            ("constant", [], r"$w_2=1$ for $d>0$"),
            ("oddScaler", [], r"$w_1=0$, $\sigma=0.05$"),
            (
                "safeScaler",
                [],
                r"$w_1=0$, $\alpha=0.6$",
            ),
            (
                "safeScaler",
                ["--alpha", "1"],
                r"$w_1=0$, $\alpha=1$, user $\sigma$ active",
            ),
            ("gaussSinkCore", [], r"$w_2=1$, $\sigma=0.05$"),
            ("gaussSinkPeriphery", [], r"$w_2=1$, $\sigma=0.05$"),
        )

        for function_name, extra_arguments, expected_curve_label in cases:
            with self.subTest(function_name=function_name, arguments=extra_arguments):
                args = transfer_plots.parse_arguments(
                    [function_name, *extra_arguments]
                )
                figure = transfer_plots.create_figure(function_name, args)
                figure.canvas.draw()
                axis = figure.axes[0]
                legend_labels = axis.get_legend_handles_labels()[1]

                self.assertEqual(expected_curve_label, axis.lines[0].get_label())
                if function_name == "gaussSinkCore":
                    self.assertEqual(
                        r"$C_1/C_2$ (core; $L_{vk}/C_2=\max(C_1/C_2,1)$)",
                        axis.get_xlabel(),
                    )
                elif function_name == "gaussSinkPeriphery":
                    self.assertEqual(
                        r"$L_{vk}/L_{vk,\mathrm{ref}}$ (periphery)",
                        axis.get_xlabel(),
                    )
                else:
                    self.assertEqual(r"$d = C_2 - C_1$", axis.get_xlabel())
                self.assertEqual(
                    1,
                    legend_labels.count("current min/max of processed value"),
                )
                matplotlib.pyplot.close(figure)

    def test_legend_outside_places_legend_beyond_the_axes_right_edge(self):
        args = transfer_plots.parse_arguments(["oddScaler"])
        args.legend_outside = True
        figure = transfer_plots.create_figure("oddScaler", args)
        figure.canvas.draw()
        axis = figure.axes[0]
        legend = axis.get_legend()

        self.assertIsNotNone(legend)
        self.assertGreaterEqual(
            legend.get_window_extent().x0,
            axis.get_window_extent().x1,
        )
        legend_bounds = legend.get_window_extent()
        axis_bounds = axis.get_window_extent()
        self.assertAlmostEqual(
            (legend_bounds.y0 + legend_bounds.y1) / 2.0,
            (axis_bounds.y0 + axis_bounds.y1) / 2.0,
            delta=1.0,
        )
        matplotlib.pyplot.close(figure)

    def test_finite_domain_markers_end_at_their_curve_values(self):
        for function_name in transfer_plots.TRANSFER_FUNCTIONS:
            with self.subTest(function_name=function_name):
                args = transfer_plots.parse_arguments(
                    [
                        function_name,
                        "--d-min",
                        "-2",
                        "--d-max",
                        "3",
                        "--nlvk-min",
                        "0.25",
                        "--nlvk-max",
                        "1.75",
                        "--points",
                        "51",
                    ]
                )
                figure = transfer_plots.create_figure(function_name, args)
                axis = figure.axes[0]
                dashed_verticals = [
                    line
                    for line in axis.lines
                    if line.get_linestyle() == "--"
                    and np.allclose(line.get_xdata(), line.get_xdata()[0])
                ]
                expected_limits = (
                    (0.25, 1.75)
                    if function_name in {
                        "gaussSinkCore",
                        "gaussSinkPeriphery",
                    }
                    else ((3.0,) if function_name in {"basic", "constant"} else (-2.0, 3.0))
                )
                expected_labels = (
                    ("current min/max of processed value", "_nolegend_")
                    if len(expected_limits) == 2
                    else ("current min/max of processed value",)
                )
                curve = axis.lines[0]
                curve_x = np.asarray(curve.get_xdata())
                curve_y = np.ma.asarray(curve.get_ydata())
                curve_endpoint_by_x = {
                    float(curve_x[index]): float(curve_y[index])
                    for index in (0, -1)
                    if not np.ma.is_masked(curve_y[index])
                }

                self.assertEqual(len(expected_limits), len(dashed_verticals))
                self.assertEqual(
                    expected_limits,
                    tuple(float(line.get_xdata()[0]) for line in dashed_verticals),
                )
                self.assertEqual(expected_labels, tuple(line.get_label() for line in dashed_verticals))
                for line in dashed_verticals:
                    x_value = float(line.get_xdata()[0])
                    np.testing.assert_allclose(
                        [0.0, curve_endpoint_by_x[x_value]],
                        line.get_ydata(),
                    )
                matplotlib.pyplot.close(figure)

    def test_comparison_domain_markers_end_at_greatest_absolute_endpoint_value(self):
        args = transfer_plots.parse_arguments(
            [
                "oddScaler",
                "--d-min",
                "-1",
                "--d-max",
                "1",
                "--weight1",
                "0.5",
                "--sigma",
                "1",
                "--compare",
                "weight1=0.3",
                "--compare",
                "weight1=0",
                "--no-area-fill",
            ]
        )

        figure = transfer_plots.create_figure("oddScaler", args)
        dashed_verticals = [
            line
            for line in figure.axes[0].lines
            if line.get_linestyle() == "--"
            and np.allclose(line.get_xdata(), line.get_xdata()[0])
        ]

        self.assertEqual(2, len(dashed_verticals))
        self.assertEqual(
            (-1.0, 1.0),
            tuple(float(marker.get_xdata()[0]) for marker in dashed_verticals),
        )
        np.testing.assert_allclose([0.0, -1.0], dashed_verticals[0].get_ydata())
        np.testing.assert_allclose([0.0, 1.0], dashed_verticals[1].get_ydata())
        matplotlib.pyplot.close(figure)

    def test_positive_and_negative_areas_use_selected_translucent_colors(self):
        args = transfer_plots.parse_arguments(
            [
                "oddScaler",
                "--d-min",
                "-2",
                "--d-max",
                "2",
                "--positive-color",
                "navy",
                "--negative-color",
                "#808080",
                "--primary-alpha",
                "0.2",
                "--secondary-alpha",
                "0.1",
                "--points",
                "101",
            ]
        )

        figure = transfer_plots.create_figure("oddScaler", args)
        face_colors = [tuple(collection.get_facecolor()[0]) for collection in figure.axes[0].collections]
        fill_labels = [collection.get_label() for collection in figure.axes[0].collections]

        np.testing.assert_allclose(
            [
                matplotlib.colors.to_rgba("navy", alpha=0.2),
                matplotlib.colors.to_rgba("#808080", alpha=0.1),
            ],
            face_colors,
        )
        self.assertEqual(["_nolegend_", "_nolegend_"], fill_labels)
        matplotlib.pyplot.close(figure)

    def test_curves_are_black_and_basic_constant_are_masked_before_activation(self):
        for function_name in transfer_plots.TRANSFER_FUNCTIONS:
            with self.subTest(function_name=function_name):
                args = transfer_plots.parse_arguments(
                    [function_name, "--d-min", "-2", "--d-max", "2", "--points", "101"]
                )
                figure = transfer_plots.create_figure(function_name, args)
                axis = figure.axes[0]
                curve = axis.lines[0]

                self.assertEqual("black", curve.get_color())
                if function_name in {"basic", "constant"}:
                    x_values = np.asarray(curve.get_xdata())
                    y_values = np.ma.asarray(curve.get_ydata())
                    mask = np.ma.getmaskarray(y_values)
                    np.testing.assert_array_equal(x_values <= 0.0, mask)
                    legend_labels = axis.get_legend_handles_labels()[1]
                    self.assertFalse(any("inactive" in label for label in legend_labels))
                matplotlib.pyplot.close(figure)

    def test_basic_constant_keep_requested_inactive_domain_visible(self):
        for function_name in ("basic", "constant"):
            with self.subTest(function_name=function_name):
                args = transfer_plots.parse_arguments(
                    [function_name, "--d-min", "-2", "--d-max", "2", "--points", "101"]
                )
                figure = transfer_plots.create_figure(function_name, args)

                np.testing.assert_allclose((-2.0, 2.0), figure.axes[0].get_xlim())
                matplotlib.pyplot.close(figure)

    def test_masked_great_region_does_not_create_a_negative_fill(self):
        for function_name in ("basic", "constant"):
            with self.subTest(function_name=function_name):
                args = transfer_plots.parse_arguments(
                    [
                        function_name,
                        "--d-min",
                        "-2",
                        "--d-max",
                        "2",
                        "--positive-color",
                        "blue",
                        "--negative-color",
                        "red",
                        "--primary-alpha",
                        "0.2",
                    ]
                )
                figure = transfer_plots.create_figure(function_name, args)
                face_colors = [
                    tuple(collection.get_facecolor()[0])
                    for collection in figure.axes[0].collections
                ]

                self.assertEqual(1, len(face_colors))
                np.testing.assert_allclose(
                    matplotlib.colors.to_rgba("blue", alpha=0.2),
                    face_colors[0],
                )
                matplotlib.pyplot.close(figure)


if __name__ == "__main__":
    unittest.main()
