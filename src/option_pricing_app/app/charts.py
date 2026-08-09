"""Interactive presentation figures; numerical modules do not import Plotly."""

import numpy as np
import plotly.graph_objects as go

from option_pricing_app.domain import ExerciseStyle, PricingResult, PutContract


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
            fillcolor="rgba(36, 91, 130, 0.14)",
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
                line={"width": 1, "color": "rgba(36, 91, 130, 0.22)"},
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
            line={"width": 3, "dash": "dash", "color": "#ff7f0e"},
            name="Risk-neutral expected price",
            hovertemplate="t=%{x:.3f} years<br>E[S]=%{y:.2f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=result.time_grid,
            y=np.full_like(result.time_grid, contract.strike),
            mode="lines",
            line={"width": 2, "color": "#7a3e9d"},
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


def cash_flow_figure(result: PricingResult) -> go.Figure:
    """Plot pathwise realised option cash flows after discounting to time zero."""
    cash_flows = result.discounted_realised_cash_flows
    zero_percentage = 100.0 * float(np.mean(np.isclose(cash_flows, 0.0, atol=1e-12)))
    figure = go.Figure(
        go.Histogram(
            x=cash_flows,
            nbinsx=50,
            histnorm="percent",
            marker_color="#245b82",
            hovertemplate="Cash flow=%{x:.4f}<br>Paths=%{y:.2f}%<extra></extra>",
        )
    )
    figure.add_vline(
        x=result.price,
        line_dash="dash",
        line_color="#a43d3d",
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
    if result.exercise_style is ExerciseStyle.EUROPEAN:
        figure.add_hline(
            y=result.european_exact_price,
            line_dash="dash",
            line_color="#a43d3d",
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
            marker_color="#245b82",
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
