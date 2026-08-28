"""Vector PDF export for the numerical-studies, early-exercise-policy and paired
validation figures.

Kaleido (Plotly's static-image renderer) bundles a full browser runtime, which is
heavy and can be unreliable on constrained hosted environments. These figures are
built entirely from small scalar summaries -- the same summaries already written to
the CSV exports -- so they are regenerated here as pure-Python vector graphics with
Matplotlib (an existing project dependency) instead of rasterising or converting the
interactive Plotly figures.
"""

from __future__ import annotations

import io
from collections.abc import Iterable

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages

from option_pricing_app.domain import PricingResult, PutContract
from option_pricing_app.early_exercise import EarlyExercisePolicyRow
from option_pricing_app.numerical_studies import ExperimentPoint, NumericalStudiesResult
from option_pricing_app.policy_validation import PairedValidationStudyResult

_BLUE = "#5B9BD5"
_GREEN = "#2CA02C"
_RED = "#FF3B30"
_YAXIS_LABEL = "Out-of-sample estimated American put value"


def _draw_ordered_study(ax: Axes, points: Iterable[ExperimentPoint], title: str, xaxis_label: str) -> None:
    """Individual replications, mean line and empirical quantile band vs an ordered setting."""
    points = list(points)
    x_values = [point.setting_value for point in points]
    means = [point.mean_estimate for point in points]
    lower = [point.lower_empirical_quantile for point in points]
    upper = [point.upper_empirical_quantile for point in points]
    ax.fill_between(
        x_values, lower, upper, color=_BLUE, alpha=0.18,
        label="95% empirical replication interval",
    )
    for point in points:
        ax.scatter(
            [point.setting_value] * len(point.runs),
            [run.estimated_price for run in point.runs],
            s=14, color=_BLUE, alpha=0.35, linewidths=0,
        )
    ax.plot(x_values, means, color=_BLUE, marker="o", linewidth=2, label="Mean across replications")
    ax.set_title(title)
    ax.set_xlabel(xaxis_label)
    ax.set_ylabel(_YAXIS_LABEL)
    ax.grid(alpha=0.25)


def _draw_categorical_study(ax: Axes, points: Iterable[ExperimentPoint], title: str, xaxis_label: str) -> None:
    """Independent per-category points and a mean marker with a vertical error bar.

    Basis specifications are categorical and unordered, so unlike the ordered studies
    this deliberately avoids a connecting line or a shaded band between categories.
    """
    points = list(points)
    positions = list(range(len(points)))
    means = [point.mean_estimate for point in points]
    lower_errors = [point.mean_estimate - point.lower_empirical_quantile for point in points]
    upper_errors = [point.upper_empirical_quantile - point.mean_estimate for point in points]
    for position, point in zip(positions, points, strict=True):
        ax.scatter(
            [position] * len(point.runs),
            [run.estimated_price for run in point.runs],
            s=14, color=_BLUE, alpha=0.35, linewidths=0,
        )
    ax.errorbar(
        positions, means, yerr=[lower_errors, upper_errors],
        fmt="D", color=_BLUE, ecolor=_BLUE, elinewidth=2, capsize=4, markersize=7,
        label="Mean, 95% empirical replication interval",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels([point.setting_label for point in points], rotation=20, ha="right")
    ax.set_title(title)
    ax.set_xlabel(xaxis_label)
    ax.set_ylabel(_YAXIS_LABEL)
    ax.grid(alpha=0.25)


def _draw_binomial_reference(ax: Axes, reference: float) -> None:
    ax.axhline(
        reference, color=_GREEN, linestyle=":", linewidth=2,
        label="CRR binomial reference (American)",
    )


def _draw_bias_rmse_table(ax: Axes, points: Iterable[ExperimentPoint], title: str) -> None:
    """Tabulate each setting's bias and RMSE against the CRR binomial reference."""
    rows = [
        [
            point.setting_label,
            f"{point.mean_estimate:.4f}",
            f"{point.bias_vs_binomial:+.4f}",
            f"{point.rmse_vs_binomial:.4f}",
            str(len(point.runs)),
        ]
        for point in points
    ]
    _draw_generic_table(
        ax,
        ["Setting", "Mean estimate", "Bias vs CRR", "RMSE vs CRR", "Replications"],
        rows,
        title,
    )


def _draw_stopping_frequency(
    ax: Axes,
    time_grid: np.ndarray,
    exercise_percentages: np.ndarray,
    out_of_the_money_percentage: float,
) -> None:
    """Bar chart of the share of independent validation paths exercised at each date.

    Takes explicit arrays -- mirroring ``charts.exercise_figure`` -- so this can be
    fed the out-of-sample stopping statistics from an ``IndependentPolicyEvaluation``.
    """
    early_exercise_percentage = float(np.sum(exercise_percentages[:-1]))
    width = 0.8 * float(np.min(np.diff(time_grid))) if time_grid.size > 1 else 0.01
    ax.bar(time_grid, exercise_percentages, width=width, color=_BLUE)
    ax.text(
        0.99, 0.96,
        f"Exercised before maturity: {early_exercise_percentage:.1f}% of paths\n"
        f"Out-of-the-money at maturity: {out_of_the_money_percentage:.1f}% of paths",
        transform=ax.transAxes, ha="right", va="top", fontsize=8,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "lightgray"},
    )
    ax.set_title("LSMC stopping policy by exercise time (out-of-sample)")
    ax.set_xlabel("Exercise time (years, final point is maturity)")
    ax.set_ylabel("Percentage of independent validation paths")
    ax.grid(alpha=0.25)


def _draw_exercise_boundary(ax: Axes, result: PricingResult, contract: PutContract) -> None:
    """Estimated exercise boundary through time, shaded exercise/continuation regions."""
    times = np.asarray(result.time_grid[1:-1], dtype=float)
    boundaries = np.full(times.shape, np.nan, dtype=float)
    for diagnostic in result.continuation_diagnostics:
        grid_index = diagnostic.timestep - 1
        if 0 <= grid_index < boundaries.size and np.isfinite(diagnostic.boundary):
            boundaries[grid_index] = diagnostic.boundary
    finite = boundaries[np.isfinite(boundaries)]
    if finite.size:
        floor = max(0.0, float(np.min(finite)) * 0.95)
        ceiling = max(contract.strike, float(np.max(finite))) * 1.02
        ax.fill_between(times, floor, boundaries, color=_RED, alpha=0.08, label="Exercise region (lower S)")
        ax.fill_between(times, boundaries, ceiling, color=_BLUE, alpha=0.10, label="Continuation region (higher S)")
        ax.plot(times, boundaries, color=_BLUE, marker="o", markersize=3, linewidth=2, label="Estimated exercise boundary")
        ax.set_ylim(floor, ceiling)
    ax.axhline(contract.strike, color=_RED, linestyle="--", linewidth=2, label=f"K = {contract.strike:,.2f}")
    ax.set_title("Estimated American put early exercise boundary (in-sample fitted policy)")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Critical underlying price, S*")
    ax.grid(alpha=0.25)
    if times.size:
        # Matplotlib's autoscale drops the leading run of NaN boundaries (where
        # continuation dominates and no boundary is fitted) instead of showing it as
        # blank space, which makes the chart look like a boundary was constructed for
        # the entire horizon. Pin the x-axis to the full evaluated horizon so that
        # early blank region is visible, matching the interactive Plotly chart.
        ax.set_xlim(float(times[0]), float(times[-1]))


def _draw_generic_table(ax: Axes, headers: list[str], rows: list[list[str]], title: str) -> None:
    """Draw a table sized to its content so long labels cannot overlap other columns.

    ``ax.table`` defaults to equal-width, centre-aligned columns; a label column much
    longer than the others (e.g. a method name) then overflows symmetrically past its
    own cell and can visually collide with the neighbouring column's values. Sizing
    columns to content and left-aligning the label column avoids that.
    """
    ax.axis("off")
    ax.set_title(title)
    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(col=list(range(len(headers))))
    for row_index in range(len(rows) + 1):  # +1 for the header row
        table[row_index, 0].set_text_props(ha="left")
        table[row_index, 0].PAD = 0.02
    table.scale(1, 1.6)


def early_exercise_policy_pdf(
    result: PricingResult,
    contract: PutContract,
    policy_rows: tuple[EarlyExercisePolicyRow, ...],
    time_grid: np.ndarray,
    exercise_percentages: np.ndarray,
    out_of_the_money_percentage: float,
) -> bytes:
    """Render the early-exercise-policy figures and summary table as a vector PDF.

    ``time_grid``/``exercise_percentages``/``out_of_the_money_percentage`` are the
    out-of-sample stopping statistics (see ``charts.exercise_figure``), not derived
    from ``result`` directly.
    """
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        _draw_stopping_frequency(ax, time_grid, exercise_percentages, out_of_the_money_percentage)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        if result.continuation_diagnostics:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            _draw_exercise_boundary(ax, result, contract)
            ax.legend(loc="best", fontsize=8)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(9.5, 3.5))
        rows = [
            [
                row.method,
                f"{row.american_value:.4f}"
                + (f" ({row.american_standard_error:.4f})" if row.american_standard_error is not None else ""),
                f"{row.european_value:.4f}",
                f"{row.early_exercise_value:.4f}",
            ]
            for row in policy_rows
        ]
        _draw_generic_table(
            ax,
            ["Method", "American (s.e.)", "European", "Early exercise value"],
            rows,
            "Estimated value from early exercise",
        )
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    return buffer.getvalue()


def numerical_studies_pdf(studies: NumericalStudiesResult) -> bytes:
    """Render the numerical-studies figures as a multi-page vector PDF."""
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        _draw_ordered_study(
            ax,
            studies.path_count.points,
            "Path-count convergence study (out-of-sample)",
            "Number of training paths used to fit LSMC",
        )
        if studies.path_count.binomial_reference is not None:
            _draw_binomial_reference(ax, studies.path_count.binomial_reference)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        if studies.path_count.binomial_reference is not None:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            _draw_bias_rmse_table(
                ax, studies.path_count.points, "Path-count study: bias and RMSE vs CRR binomial reference"
            )
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        if studies.exercise_grid is not None:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            _draw_ordered_study(
                ax,
                studies.exercise_grid.points,
                "Exercise-grid convergence study (out-of-sample)",
                "Number of exercise dates",
            )
            if studies.exercise_grid.binomial_reference is not None:
                _draw_binomial_reference(ax, studies.exercise_grid.binomial_reference)
            ax.legend(loc="best", fontsize=8)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            if studies.exercise_grid.binomial_reference is not None:
                fig, ax = plt.subplots(figsize=(8, 4.5))
                _draw_bias_rmse_table(
                    ax,
                    studies.exercise_grid.points,
                    "Exercise-grid study: bias and RMSE vs CRR binomial reference",
                )
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        _draw_categorical_study(
            ax,
            studies.basis_sensitivity.points,
            "LSMC basis-function sensitivity (out-of-sample)",
            "Basis specification",
        )
        if studies.basis_sensitivity.binomial_reference is not None:
            _draw_binomial_reference(ax, studies.basis_sensitivity.binomial_reference)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        if studies.basis_sensitivity.binomial_reference is not None:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            _draw_bias_rmse_table(
                ax,
                studies.basis_sensitivity.points,
                "Basis-sensitivity study: bias and RMSE vs CRR binomial reference",
            )
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    return buffer.getvalue()


def _draw_paired_replication(ax: Axes, study: PairedValidationStudyResult) -> None:
    """Same-sample vs independent estimate per replication, joined by a thin line.

    The line connects two results from the same training/valuation seed pair, not
    an ordered x-axis -- mirrors ``charts.paired_replication_figure``.
    """
    replications = [row.replication for row in study.replications]
    same_sample = [row.same_sample_estimate for row in study.replications]
    independent = [row.independent_estimate for row in study.replications]
    for replication, same, indep in zip(replications, same_sample, independent, strict=True):
        ax.plot(
            [replication, replication], [same, indep],
            color=_BLUE, alpha=0.5, linewidth=1, zorder=1,
        )
    ax.scatter(
        replications, same_sample, color=_RED, s=45, zorder=2,
        label="Same-sample estimate",
    )
    ax.scatter(
        replications, independent, color=_BLUE, marker="D", s=45, zorder=2,
        label="Independent (out-of-sample) estimate",
    )
    ax.axhline(
        study.matched_grid_crr, color=_GREEN, linestyle=":", linewidth=2,
        label="Matched-grid CRR reference",
    )
    ax.set_title("Paired replication validation: same-sample vs independent policy value")
    ax.set_xlabel("Replication")
    ax.set_ylabel("Estimated American put value")
    ax.set_xticks(replications)
    ax.grid(alpha=0.25)


def paired_validation_pdf(study: PairedValidationStudyResult) -> bytes:
    """Render the paired-replication validation chart and summary table as a vector PDF."""
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        _draw_paired_replication(ax, study)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        rows = [
            ["Continuous-exercise CRR", f"{study.continuous_crr:.4f}"],
            ["Matched-grid CRR", f"{study.matched_grid_crr:.4f}"],
            ["Mean same-sample LSMC", f"{study.mean_same_sample_estimate:.4f}"],
            ["Same-sample empirical SD", f"{study.same_sample_empirical_sd:.4f}"],
            ["Mean independent policy value", f"{study.mean_independent_estimate:.4f}"],
            ["Independent empirical SD", f"{study.independent_empirical_sd:.4f}"],
            ["Mean paired gap", f"{study.mean_paired_difference:.4f}"],
            [
                "Independent mean error against matched CRR",
                f"{study.independent_mean_error_vs_matched_crr:+.4f}",
            ],
            [
                "Independent RMSE against matched CRR",
                f"{study.independent_rmse_vs_matched_crr:.4f}",
            ],
        ]
        _draw_generic_table(
            ax, ["Validation measure", "Value"], rows, "Paired replication validation summary"
        )
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    return buffer.getvalue()
