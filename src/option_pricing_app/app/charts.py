"""Interactive presentation figures; numerical modules do not import Plotly."""

import numpy as np
import plotly.graph_objects as go

from option_pricing_app.domain import (
    ExerciseStyle,
    LSMCContinuationDiagnostic,
    PricingResult,
    PutContract,
)

PRIMARY_BLUE = "#5B9BD5"
BLUE_PATH = "rgba(91, 155, 213, 0.28)"
BLUE_BAND = "rgba(91, 155, 213, 0.18)"
REFERENCE_RED = "#FF3B30"
CONTINUATION_ORANGE = "#ff7f0e"
STRIKE_PURPLE = "#7a3e9d"
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
    """Plot the cumulative Monte Carlo estimate and its approximate 95% interval."""
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
        title="Monte Carlo convergence by simulated path",
        xaxis_title="Number of paths included",
        yaxis_title="Estimated option value",
        template="plotly_white",
        height=430,
        hovermode="x unified",
    )
    return figure


def exercise_figure(result: PricingResult) -> go.Figure:
    """Plot the share of all paths whose selected stopping time is each step."""
    if result.exercise_percentages is None:
        raise ValueError("Exercise frequencies are only available for American options")
    early_exercise_percentage = float(np.sum(result.exercise_percentages[:-1]))
    figure = go.Figure(
        go.Bar(
            x=np.arange(result.exercise_percentages.size),
            y=result.exercise_percentages,
            marker_color=PRIMARY_BLUE,
            hovertemplate="Step=%{x}<br>Paths exercised=%{y:.2f}%<extra></extra>",
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
        title="LSMC stopping policy by exercise step",
        xaxis_title="Exercise step (final step is maturity)",
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
            "Insufficient ITM paths for a reliable boundary"
            if diagnostic.itm_path_count < diagnostic.minimum_required_paths
            else "No reliable exercise-to-continuation crossing within the ITM support"
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
    """Plot reliable regression-based American put boundaries through time."""
    if result.continuation_diagnostics is None:
        raise ValueError("Exercise boundaries are only available for American options")

    reliable = [
        diagnostic
        for diagnostic in result.continuation_diagnostics
        if np.isfinite(diagnostic.boundary)
    ]
    figure = go.Figure()
    if reliable:
        times = np.array([diagnostic.time for diagnostic in reliable])
        boundaries = np.array([diagnostic.boundary for diagnostic in reliable])
        floor = max(0.0, float(np.min(boundaries)) * 0.95)
        ceiling = max(contract.strike, float(np.max(boundaries))) * 1.02
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
                connectgaps=True,
                line={"width": 3, "color": PRIMARY_BLUE},
                marker={"size": 5, "color": PRIMARY_BLUE},
                name="Estimated exercise boundary",
                customdata=np.array([diagnostic.itm_path_count for diagnostic in reliable]),
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
            text="No reliable regression crossing was found on the observed ITM support.",
            showarrow=False,
        )

    figure.add_hline(
        y=contract.strike,
        line_color=STRIKE_PURPLE,
        line_width=2,
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
