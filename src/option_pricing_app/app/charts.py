"""Interactive presentation figures; numerical modules do not import Plotly."""

import numpy as np
import plotly.graph_objects as go

from option_pricing_app.domain import PricingResult, PutContract


def paths_figure(result: PricingResult) -> go.Figure:
    figure = go.Figure()
    for number, path in enumerate(result.displayed_paths.T, start=1):
        figure.add_trace(
            go.Scatter(
                x=result.time_grid,
                y=path,
                mode="lines",
                line={"width": 1},
                opacity=0.55,
                name=f"Path {number}",
                showlegend=False,
                hovertemplate="t=%{x:.3f} years<br>S=%{y:.2f}<extra></extra>",
            )
        )
    figure.update_layout(
        title=f"Simulated price paths — {result.model_name}",
        xaxis_title="Time (years)",
        yaxis_title="Underlying price",
        template="plotly_white",
        height=470,
    )
    return figure


def terminal_figure(result: PricingResult, contract: PutContract) -> go.Figure:
    figure = go.Figure(
        go.Histogram(x=result.terminal_prices, nbinsx=50, marker_color="#245b82")
    )
    figure.add_vline(
        x=contract.strike,
        line_dash="dash",
        line_color="#a43d3d",
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
        go.Scatter(x=prices, y=payoffs, mode="lines", line={"width": 3, "color": "#245b82"})
    )
    figure.add_vline(
        x=contract.strike,
        line_dash="dash",
        line_color="#a43d3d",
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
            fillcolor="rgba(36, 91, 130, 0.18)",
            name="Approximate 95% interval",
            hovertemplate="Paths=%{x:,}<br>Lower=%{y:.4f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=result.convergence_path_counts,
            y=result.convergence_estimates,
            mode="lines",
            line={"width": 2, "color": "#245b82"},
            name="Cumulative estimate",
            hovertemplate="Paths=%{x:,}<br>Estimate=%{y:.4f}<extra></extra>",
        )
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
    figure = go.Figure(
        go.Bar(
            x=np.arange(result.exercise_percentages.size),
            y=result.exercise_percentages,
            marker_color="#245b82",
            hovertemplate="Step=%{x}<br>Paths exercised=%{y:.2f}%<extra></extra>",
        )
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
