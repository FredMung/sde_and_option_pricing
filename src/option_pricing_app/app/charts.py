"""Interactive presentation figures; numerical modules do not import Plotly."""

import numpy as np
import plotly.graph_objects as go

from option_pricing_app.domain import (
    ExerciseStyle,
    LSMCContinuationDiagnostic,
    PricingResult,
    PutContract,
)
from option_pricing_app.numerical_studies import (
    BasisSensitivityStudyResult,
    ExerciseGridStudyResult,
    ExperimentPoint,
    PathCountStudyResult,
)

PRIMARY_BLUE = "#5B9BD5"
BLUE_PATH = "rgba(91, 155, 213, 0.28)"
BLUE_BAND = "rgba(91, 155, 213, 0.18)"
REFERENCE_RED = "#FF3B30"
CONTINUATION_ORANGE = "#ff7f0e"
STRIKE_PURPLE = "#7a3e9d"
BINOMIAL_GREEN = "#2CA02C"
EXERCISE_FILL = "rgba(255, 59, 48, 0.08)"
CONTINUATION_FILL = "rgba(91, 155, 213, 0.10)"


def paths_figure(result: PricingResult, contract: PutContract) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=result.time_grid,
            y=result.path_quantile_95,
            mode="lines",
            line={"width": 0},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=result.time_grid,
            y=result.path_quantile_05,
            customdata=result.path_quantile_95,
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor=BLUE_BAND,
            name="5–95% path band",
            hovertemplate=(
                "t=%{x:.3f} years<br>5th percentile=%{y:.2f}"
                "<br>95th percentile=%{customdata:.2f}<extra></extra>"
            ),
        )
    )
    for number, path in enumerate(result.displayed_paths.T, start=1):
        figure.add_trace(
            go.Scatter(
                x=result.time_grid,
                y=path,
                mode="lines",
                line={"width": 1, "color": BLUE_PATH},
                name=f"Path {number}",
                showlegend=False,
                hoverinfo="skip",
            )
        )
    figure.add_trace(
        go.Scatter(
            x=result.time_grid,
            y=result.risk_neutral_expected_path,
            mode="lines",
            line={"width": 3, "dash": "dash", "color": CONTINUATION_ORANGE},
            name="Risk-neutral expected price",
            hovertemplate="t=%{x:.3f} years<br>E[S]=%{y:.2f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=result.time_grid,
            y=np.full_like(result.time_grid, contract.strike),
            mode="lines",
            line={"width": 2, "color": STRIKE_PURPLE},
            name=f"Put strike K = {contract.strike:,.2f}",
            hovertemplate=f"Put strike K={contract.strike:,.2f}<extra></extra>",
        )
    )
    figure.update_layout(
        title=f"Simulated price paths and distribution envelope — {result.model_name}",
        xaxis_title="Time (years)",
        yaxis_title="Underlying price",
        template="plotly_white",
        height=470,
        hovermode="x unified",
    )
    return figure


def variance_paths_figure(result: PricingResult) -> go.Figure:
    """Plot Heston instantaneous variance paths and their distribution envelope."""
    if (
        result.displayed_variance_paths is None
        or result.variance_quantile_05 is None
        or result.variance_quantile_95 is None
        or result.expected_variance_path is None
        or result.long_run_variance is None
    ):
        raise ValueError("Variance paths are only available for the Heston model")

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=result.time_grid,
            y=result.variance_quantile_95,
            mode="lines",
            line={"width": 0},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=result.time_grid,
            y=result.variance_quantile_05,
            customdata=result.variance_quantile_95,
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor="rgba(42, 122, 102, 0.15)",
            name="5–95% variance band",
            hovertemplate=(
                "t=%{x:.3f} years<br>5th percentile=%{y:.6f}"
                "<br>95th percentile=%{customdata:.6f}<extra></extra>"
            ),
        )
    )
    for path in result.displayed_variance_paths.T:
        figure.add_trace(
            go.Scatter(
                x=result.time_grid,
                y=path,
                mode="lines",
                line={"width": 1, "color": "rgba(42, 122, 102, 0.20)"},
                showlegend=False,
                hoverinfo="skip",
            )
        )
    figure.add_trace(
        go.Scatter(
            x=result.time_grid,
            y=result.expected_variance_path,
            mode="lines",
            line={"width": 3, "dash": "dash", "color": CONTINUATION_ORANGE},
            name="Expected variance",
            hovertemplate="t=%{x:.3f} years<br>E[v]=%{y:.6f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=result.time_grid,
            y=np.full_like(result.time_grid, result.long_run_variance),
            mode="lines",
            line={"width": 2, "color": STRIKE_PURPLE},
            name=f"Long-run variance θ = {result.long_run_variance:.6f}",
            hovertemplate=(
                f"Long-run variance θ={result.long_run_variance:.6f}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title="Heston instantaneous variance paths",
        xaxis_title="Time (years)",
        yaxis_title="Variance, v (volatility²)",
        template="plotly_white",
        height=470,
        hovermode="x unified",
    )
    return figure


def terminal_figure(result: PricingResult, contract: PutContract) -> go.Figure:
    figure = go.Figure(
        go.Histogram(x=result.terminal_prices, nbinsx=50, marker_color=PRIMARY_BLUE)
    )
    figure.add_vline(
        x=contract.strike,
        line_dash="dash",
        line_color=REFERENCE_RED,
        line_width=3,
        annotation_text=f"K = {contract.strike:,.2f}",
    )
    figure.update_layout(
        title="Terminal-price distribution",
        xaxis_title="Terminal underlying price",
        yaxis_title="Simulated paths",
        template="plotly_white",
        height=410,
    )
    return figure


def payoff_figure(result: PricingResult, contract: PutContract) -> go.Figure:
    upper = max(contract.strike * 2, float(np.quantile(result.terminal_prices, 0.99)) * 1.1)
    prices = np.linspace(0.0, upper, 300)
    payoffs = np.maximum(contract.strike - prices, 0.0)
    figure = go.Figure(
        go.Scatter(
            x=prices,
            y=payoffs,
            mode="lines",
            line={"width": 3, "color": PRIMARY_BLUE},
        )
    )
    figure.add_vline(
        x=contract.strike,
        line_dash="dash",
        line_color=REFERENCE_RED,
        line_width=3,
        annotation_text=f"K = {contract.strike:,.2f}",
    )
    figure.update_layout(
        title="Put payoff at exercise",
        xaxis_title="Underlying price",
        yaxis_title="max(K − S, 0)",
        template="plotly_white",
        height=410,
    )
    return figure


def cash_flow_figure(result: PricingResult) -> go.Figure:
    """Plot pathwise realised option cash flows after discounting to time zero."""
    cash_flows = result.discounted_realised_cash_flows
    zero_percentage = 100.0 * float(np.mean(np.isclose(cash_flows, 0.0, atol=1e-12)))
    figure = go.Figure(
        go.Histogram(
            x=cash_flows,
            nbinsx=50,
            histnorm="percent",
            marker_color=PRIMARY_BLUE,
            hovertemplate="Cash flow=%{x:.4f}<br>Paths=%{y:.2f}%<extra></extra>",
        )
    )
    figure.add_vline(
        x=result.price,
        line_dash="dash",
        line_color=REFERENCE_RED,
        line_width=3,
        annotation_text=f"Mean = {result.price:,.4f}",
    )
    figure.add_annotation(
        x=0.99,
        y=0.96,
        xref="paper",
        yref="paper",
        xanchor="right",
        yanchor="top",
        text=f"Zero cash flow: {zero_percentage:.1f}% of paths",
        showarrow=False,
    )
    figure.update_layout(
        title="Distribution of discounted realised option cash flows",
        xaxis_title="Realised cash flow discounted to time zero",
        yaxis_title="Percentage of simulated paths",
        yaxis={"ticksuffix": "%"},
        template="plotly_white",
        height=430,
        bargap=0.04,
    )
    return figure


def convergence_figure(result: PricingResult) -> go.Figure:
    """Plot a cumulative diagnostic from one completed simulation."""
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=result.convergence_path_counts,
            y=result.convergence_upper,
            mode="lines",
            line={"width": 0},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=result.convergence_path_counts,
            y=result.convergence_lower,
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor=BLUE_BAND,
            name="Approximate 95% interval",
            hovertemplate="Paths=%{x:,}<br>Lower=%{y:.4f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=result.convergence_path_counts,
            y=result.convergence_estimates,
            mode="lines",
            line={"width": 2, "color": PRIMARY_BLUE},
            name="Cumulative estimate",
            hovertemplate="Paths=%{x:,}<br>Estimate=%{y:.4f}<extra></extra>",
        )
    )
    if (
        result.exercise_style is ExerciseStyle.EUROPEAN
        and result.european_exact_price is not None
    ):
        figure.add_hline(
            y=result.european_exact_price,
            line_dash="dash",
            line_color=REFERENCE_RED,
            line_width=3,
            annotation_text="Black–Scholes exact",
        )
    figure.update_layout(
        title="Single-run cumulative estimate",
        xaxis_title="Number of paths included",
        yaxis_title="Estimated option value",
        template="plotly_white",
        height=430,
        hovermode="x unified",
    )
    return figure


def _repeated_study_figure(
    points: tuple[ExperimentPoint, ...],
    title: str,
    xaxis_title: str,
) -> go.Figure:
    """Plot repeated full-pricing estimates and empirical replication quantiles.

    Only appropriate for settings with a genuine continuous ordering (path count,
    exercise-grid size); the connecting line and band imply interpolation between
    x-values, which is not valid for categorical settings such as basis choice.
    """
    x_values = [point.setting_value for point in points]
    lower = np.asarray([point.lower_empirical_quantile for point in points])
    upper = np.asarray([point.upper_empirical_quantile for point in points])
    means = np.asarray([point.mean_estimate for point in points])
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=x_values,
            y=upper,
            mode="lines",
            line={"width": 0},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=x_values,
            y=lower,
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor=BLUE_BAND,
            name="95% empirical replication interval",
            customdata=upper,
            hovertemplate=(
                "Setting=%{x}<br>2.5%=%{y:.4f}<br>97.5%=%{customdata:.4f}"
                "<extra></extra>"
            ),
        )
    )
    run_x = []
    run_y = []
    run_seed = []
    for x_value, point in zip(x_values, points, strict=True):
        for run in point.runs:
            run_x.append(x_value)
            run_y.append(run.estimated_price)
            run_seed.append(run.seed)
    figure.add_trace(
        go.Scatter(
            x=run_x,
            y=run_y,
            mode="markers",
            marker={"size": 6, "color": PRIMARY_BLUE, "opacity": 0.28},
            name="Individual replication",
            customdata=run_seed,
            hovertemplate=(
                "Setting=%{x}<br>Estimate=%{y:.4f}<br>Seed=%{customdata}"
                "<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=x_values,
            y=means,
            mode="lines+markers",
            line={"width": 3, "color": PRIMARY_BLUE},
            marker={"size": 8, "color": PRIMARY_BLUE},
            name="Mean across replications",
            hovertemplate="Setting=%{x}<br>Mean=%{y:.4f}<extra></extra>",
        )
    )
    figure.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title="Estimated American put value",
        template="plotly_white",
        height=430,
        hovermode="x unified",
    )
    return figure


def path_count_study_figure(study: PathCountStudyResult) -> go.Figure:
    """Plot true LSMC path-count convergence from complete repeated reruns."""
    figure = _repeated_study_figure(
        study.points,
        "Path-count convergence study",
        "Number of paths used to fit and value LSMC",
    )
    final = study.points[-1]
    figure.add_trace(
        go.Scatter(
            x=[final.setting_value],
            y=[final.mean_estimate],
            mode="markers",
            marker={"size": 13, "symbol": "diamond", "color": REFERENCE_RED},
            name="Selected path count",
            hovertemplate="Selected paths=%{x:,}<br>Mean=%{y:.4f}<extra></extra>",
        )
    )
    if study.binomial_reference is not None:
        figure.add_hline(
            y=study.binomial_reference,
            line_dash="dot",
            line_color=BINOMIAL_GREEN,
            line_width=3,
            annotation_text="CRR binomial reference (American)",
            annotation_position="bottom right",
        )
    return figure


def exercise_grid_study_figure(study: ExerciseGridStudyResult) -> go.Figure:
    """Plot repeated American estimates as the permitted exercise grid is refined."""
    figure = _repeated_study_figure(
        study.points,
        "Exercise-grid convergence study",
        "Number of exercise dates",
    )
    if study.binomial_reference is not None:
        figure.add_hline(
            y=study.binomial_reference,
            line_dash="dot",
            line_color=BINOMIAL_GREEN,
            line_width=3,
            annotation_text="CRR binomial reference (American)",
            annotation_position="bottom right",
        )
    return figure


def basis_sensitivity_study_figure(study: BasisSensitivityStudyResult) -> go.Figure:
    """Plot repeated American estimates across documented LSMC basis choices.

    Basis specifications are categorical and unordered, so unlike the path-count and
    exercise-grid studies this draws independent points per basis rather than a
    connected line or shaded band, which would falsely imply a continuous ordering.
    """
    labels = [point.setting_label for point in study.points]
    means = np.asarray([point.mean_estimate for point in study.points])
    lower = np.asarray([point.lower_empirical_quantile for point in study.points])
    upper = np.asarray([point.upper_empirical_quantile for point in study.points])
    figure = go.Figure()
    run_x = []
    run_y = []
    run_seed = []
    for label, point in zip(labels, study.points, strict=True):
        for run in point.runs:
            run_x.append(label)
            run_y.append(run.estimated_price)
            run_seed.append(run.seed)
    figure.add_trace(
        go.Scatter(
            x=run_x,
            y=run_y,
            mode="markers",
            marker={"size": 6, "color": PRIMARY_BLUE, "opacity": 0.28},
            name="Individual replication",
            customdata=run_seed,
            hovertemplate=(
                "Basis=%{x}<br>Estimate=%{y:.4f}<br>Seed=%{customdata}<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=labels,
            y=means,
            mode="markers",
            marker={
                "size": 12,
                "color": PRIMARY_BLUE,
                "symbol": "diamond",
                "line": {"width": 1, "color": "white"},
            },
            error_y={
                "type": "data",
                "symmetric": False,
                "array": upper - means,
                "arrayminus": means - lower,
                "color": PRIMARY_BLUE,
                "thickness": 2,
                "width": 8,
            },
            name="Mean, 95% empirical replication interval",
            hovertemplate=(
                "Basis=%{x}<br>Mean=%{y:.4f}<br>2.5%=%{customdata[0]:.4f}<br>"
                "97.5%=%{customdata[1]:.4f}<extra></extra>"
            ),
            customdata=np.stack([lower, upper], axis=1),
        )
    )
    if study.binomial_reference is not None:
        figure.add_hline(
            y=study.binomial_reference,
            line_dash="dot",
            line_color=BINOMIAL_GREEN,
            line_width=3,
            annotation_text="CRR binomial reference (American)",
            annotation_position="bottom right",
        )
    figure.update_layout(
        title="LSMC basis-function sensitivity",
        xaxis_title="Basis specification",
        xaxis={"type": "category"},
        yaxis_title="Estimated American put value",
        template="plotly_white",
        height=430,
    )
    return figure


def exercise_figure(result: PricingResult) -> go.Figure:
    """Plot the share of all paths whose selected stopping time is each time point."""
    if result.exercise_percentages is None:
        raise ValueError("Exercise frequencies are only available for American options")
    early_exercise_percentage = float(np.sum(result.exercise_percentages[:-1]))
    bar_width = (
        0.8 * float(np.min(np.diff(result.time_grid)))
        if result.time_grid.size > 1
        else None
    )
    figure = go.Figure(
        go.Bar(
            x=result.time_grid,
            y=result.exercise_percentages,
            width=bar_width,
            marker_color=PRIMARY_BLUE,
            hovertemplate="t=%{x:.3f} years<br>Paths exercised=%{y:.2f}%<extra></extra>",
        )
    )
    figure.add_annotation(
        x=0.99,
        y=0.96,
        xref="paper",
        yref="paper",
        xanchor="right",
        yanchor="top",
        text=f"Exercised before maturity: {early_exercise_percentage:.1f}% of paths",
        showarrow=False,
    )
    figure.update_layout(
        title="LSMC stopping policy by exercise time",
        xaxis_title="Exercise time (years, final point is maturity)",
        yaxis_title="Percentage of all paths",
        yaxis={"ticksuffix": "%"},
        template="plotly_white",
        height=430,
    )
    return figure


def exercise_vs_continuation_figure(
    diagnostic: LSMCContinuationDiagnostic,
    contract: PutContract,
) -> go.Figure:
    """Compare immediate exercise with one stored cross-sectional LSMC fit.

    The curve uses the exact regression coefficients, column scaling and basis terms
    fitted to the in-the-money paths during backward induction. It is not a realised
    cash flow from any individual simulated path.
    """
    variance_note = ""
    if diagnostic.representative_variance is not None:
        variance_note = (
            f" | conditional on median ITM variance "
            f"v={diagnostic.representative_variance:.6f}"
        )
    subtitle = (
        f"t = {diagnostic.time:.3f} years | "
        f"{diagnostic.time_to_maturity:.3f} years to maturity | "
        f"{diagnostic.itm_path_count:,} ITM paths{variance_note}"
    )
    figure = go.Figure()
    if np.isfinite(diagnostic.boundary):
        figure.add_vrect(
            x0=float(diagnostic.price_grid[0]),
            x1=diagnostic.boundary,
            fillcolor=EXERCISE_FILL,
            line_width=0,
            annotation_text="Exercise region",
            annotation_position="top left",
            layer="below",
        )
        figure.add_vrect(
            x0=diagnostic.boundary,
            x1=float(diagnostic.price_grid[-1]),
            fillcolor=CONTINUATION_FILL,
            line_width=0,
            annotation_text="Continuation region",
            annotation_position="top right",
            layer="below",
        )
    figure.add_trace(
        go.Scatter(
            x=diagnostic.price_grid,
            y=diagnostic.immediate_payoff,
            mode="lines",
            line={"width": 3, "color": PRIMARY_BLUE},
            name="Immediate exercise payoff, H(S)",
            hovertemplate="S=%{x:.2f}<br>H(S)=%{y:.4f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=diagnostic.price_grid,
            y=diagnostic.continuation_value,
            mode="lines",
            line={"width": 3, "color": CONTINUATION_ORANGE},
            name="Estimated continuation value, Ĉ(S)",
            hovertemplate="S=%{x:.2f}<br>Ĉ(S)=%{y:.4f}<extra></extra>",
        )
    )
    if np.isfinite(diagnostic.boundary):
        figure.add_vline(
            x=diagnostic.boundary,
            line_dash="dash",
            line_color=REFERENCE_RED,
            line_width=3,
            annotation_text=f"Threshold ≈ {diagnostic.boundary:,.2f}",
        )
    else:
        reason = (
            "Insufficient ITM paths for a supported boundary"
            if diagnostic.itm_path_count < diagnostic.minimum_required_paths
            else "No single exercise-to-continuation crossing within the ITM support"
        )
        figure.add_annotation(
            x=0.5,
            y=0.96,
            xref="paper",
            yref="paper",
            text=reason,
            showarrow=False,
        )
    figure.update_layout(
        title=f"Exercise Payoff vs Estimated Continuation Value<br><sup>{subtitle}</sup>",
        xaxis_title="Underlying price, Sₜ",
        yaxis_title="Value at selected exercise time",
        template="plotly_white",
        height=430,
        hovermode="x unified",
    )
    return figure


def exercise_boundary_figure(result: PricingResult, contract: PutContract) -> go.Figure:
    """Plot accepted regression-based American put boundaries through time."""
    if result.continuation_diagnostics is None:
        raise ValueError("Exercise boundaries are only available for American options")

    accepted = [
        diagnostic
        for diagnostic in result.continuation_diagnostics
        if np.isfinite(diagnostic.boundary)
    ]
    figure = go.Figure()
    if accepted:
        times = np.asarray(result.time_grid[1:-1], dtype=float)
        boundaries = np.full(times.shape, np.nan, dtype=float)
        itm_counts = np.full(times.shape, np.nan, dtype=float)
        for diagnostic in result.continuation_diagnostics:
            grid_index = diagnostic.timestep - 1
            if 0 <= grid_index < boundaries.size:
                itm_counts[grid_index] = diagnostic.itm_path_count
                if np.isfinite(diagnostic.boundary):
                    boundaries[grid_index] = diagnostic.boundary
        finite_boundaries = boundaries[np.isfinite(boundaries)]
        floor = max(0.0, float(np.min(finite_boundaries)) * 0.95)
        ceiling = max(contract.strike, float(np.max(finite_boundaries))) * 1.02
        figure.add_trace(
            go.Scatter(
                x=times,
                y=np.full_like(times, floor),
                mode="lines",
                line={"width": 0},
                hoverinfo="skip",
                showlegend=False,
            )
        )
        figure.add_trace(
            go.Scatter(
                x=times,
                y=boundaries,
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor=EXERCISE_FILL,
                name="Exercise region (lower S)",
                hoverinfo="skip",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=times,
                y=np.full_like(times, ceiling),
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor=CONTINUATION_FILL,
                name="Continuation region (higher S)",
                hoverinfo="skip",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=times,
                y=boundaries,
                mode="lines+markers",
                connectgaps=False,
                line={"width": 3, "color": PRIMARY_BLUE},
                marker={"size": 5, "color": PRIMARY_BLUE},
                name="Estimated exercise boundary",
                customdata=itm_counts,
                hovertemplate=(
                    "t=%{x:.3f} years<br>Boundary=%{y:.2f}"
                    "<br>ITM paths=%{customdata:,}<extra></extra>"
                ),
            )
        )
        figure.update_yaxes(range=[floor, ceiling])
    else:
        figure.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            text="No single supported regression crossing was found on the observed ITM support.",
            showarrow=False,
        )

    figure.add_hline(
        y=contract.strike,
        line_dash="dash",
        line_color=REFERENCE_RED,
        line_width=3,
        annotation_text=f"K = {contract.strike:,.2f}",
    )
    variance_note = ""
    if any(
        diagnostic.representative_variance is not None
        for diagnostic in result.continuation_diagnostics
    ):
        variance_note = "<br><sup>Conditional on median ITM variance at each timestep</sup>"
    figure.update_layout(
        title=f"Estimated American Put Early Exercise Boundary{variance_note}",
        xaxis_title="Time (years)",
        yaxis_title="Critical underlying price, Sₜ*",
        template="plotly_white",
        height=430,
        hovermode="x unified",
    )
    return figure
