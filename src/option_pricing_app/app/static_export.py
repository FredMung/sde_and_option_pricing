"""Vector PDF export for the numerical-studies figures.

Kaleido (Plotly's static-image renderer) bundles a full browser runtime, which is
heavy and can be unreliable on constrained hosted environments. The numerical-studies
figures are built entirely from small scalar summaries -- the same summaries already
written to the CSV export -- so they are regenerated here as pure-Python vector
graphics with Matplotlib (an existing project dependency) instead of rasterising or
converting the interactive Plotly figures.
"""

from __future__ import annotations

import io
from collections.abc import Iterable

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages

from option_pricing_app.numerical_studies import ExperimentPoint, NumericalStudiesResult

_BLUE = "#5B9BD5"
_ORANGE = "#ff7f0e"
_YAXIS_LABEL = "Estimated American put value"


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


def numerical_studies_pdf(studies: NumericalStudiesResult) -> bytes:
    """Render the numerical-studies figures as a multi-page vector PDF."""
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        _draw_ordered_study(
            ax,
            studies.path_count.points,
            "Path-count convergence study",
            "Number of paths used to fit and value LSMC",
        )
        if studies.path_count.european_reference is not None:
            ax.axhline(
                studies.path_count.european_reference,
                color=_ORANGE, linestyle="--", linewidth=2,
                label="European Black–Scholes value",
            )
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        if studies.exercise_grid is not None:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            _draw_ordered_study(
                ax,
                studies.exercise_grid.points,
                "Exercise-grid convergence study",
                "Number of exercise dates",
            )
            ax.legend(loc="best", fontsize=8)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        _draw_categorical_study(
            ax,
            studies.basis_sensitivity.points,
            "LSMC basis-function sensitivity",
            "Basis specification",
        )
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    return buffer.getvalue()
