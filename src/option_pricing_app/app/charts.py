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
from option_pricing_app.policy_validation import PairedValidationStudyResult

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
        annotation_text=f"Same-sample mean = {result.price:,.4f}",
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
            name="Approximate 95% interval (same-sample)",
            hovertemplate="Paths=%{x:,}<br>Lower=%{y:.4f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=result.convergence_path_counts,
            y=result.convergence_estimates,
            mode="lines",
            line={"width": 2, "color": PRIMARY_BLUE},
            name="Cumulative estimate (same-sample)",
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
    """Plot repeated out-of-sample policy estimates and empirical replication quantiles.

    Each point is a frozen LSMC policy fitted on training paths and evaluated,
    without refitting, on independently simulated validation paths (see
    ``lsmc_policy.py``) -- not a same-sample price. Only appropriate for settings
    with a genuine continuous ordering (path count, exercise-grid size); the
    connecting line and band imply interpolation between x-values, which is not
    valid for categorical settings such as basis choice.
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
            run_seed.append(run.training_seed)
    figure.add_trace(
        go.Scatter(
            x=run_x,
            y=run_y,
            mode="markers",
            marker={"size": 6, "color": PRIMARY_BLUE, "opacity": 0.28},
            name="Individual replication (out-of-sample)",
            customdata=run_seed,
            hovertemplate=(
                "Setting=%{x}<br>Out-of-sample estimate=%{y:.4f}<br>"
                "Training seed=%{customdata}<extra></extra>"
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
            name="Mean out-of-sample estimate",
            hovertemplate="Setting=%{x}<br>Mean=%{y:.4f}<extra></extra>",
        )
    )
    figure.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title="Out-of-sample estimated American put value",
        template="plotly_white",
        height=430,
        hovermode="x unified",
    )
    return figure


def path_count_study_figure(study: PathCountStudyResult) -> go.Figure:
    """Plot out-of-sample LSMC path-count convergence from complete repeated reruns.

    Tests whether increasing the number of training paths improves the fitted
    policy's performance, isolated from validation noise by holding the independent
    evaluation set fixed at the currently selected path count across all settings.
    """
    figure = _repeated_study_figure(
        study.points,
        "Path-count convergence study (out-of-sample)",
        "Number of training paths used to fit LSMC",
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
    """Plot repeated out-of-sample American estimates as the exercise grid is refined.

    Fitting and evaluating out-of-sample here prevents same-sample reuse from being
    mistaken for a genuine exercise-grid effect.
    """
    figure = _repeated_study_figure(
        study.points,
        "Exercise-grid convergence study (out-of-sample)",
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
    """Plot repeated out-of-sample American estimates across LSMC basis choices.

    Each point is a frozen LSMC policy fitted on training paths and evaluated,
    without refitting, on independently simulated validation paths (see
    ``lsmc_policy.py``) -- particularly important here, since a richer basis can fit
    the training paths well without improving the policy's out-of-sample performance.
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
            run_seed.append(run.training_seed)
    figure.add_trace(
        go.Scatter(
            x=run_x,
            y=run_y,
            mode="markers",
            marker={"size": 6, "color": PRIMARY_BLUE, "opacity": 0.28},
            name="Individual replication (out-of-sample)",
            customdata=run_seed,
            hovertemplate=(
                "Basis=%{x}<br>Out-of-sample estimate=%{y:.4f}<br>"
                "Training seed=%{customdata}<extra></extra>"
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
            name="Mean out-of-sample estimate, 95% empirical replication interval",
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
        title="LSMC basis-function sensitivity (out-of-sample)",
        xaxis_title="Basis specification",
        xaxis={"type": "category"},
        yaxis_title="Out-of-sample estimated American put value",
        template="plotly_white",
        height=430,
    )
    return figure


def paired_replication_figure(study: PairedValidationStudyResult) -> go.Figure:
    """Plot same-sample vs independent policy value for each paired replication.

    One marker per replication for the same-sample estimate (fit and evaluated on
    the same training paths) and one for the independent estimate (that frozen
    policy evaluated on separately simulated valuation paths), joined by a thin
    line. The line is meaningful here -- unlike in the path-count/exercise-grid
    charts -- because it connects two results computed from the same training and
    valuation seed pair, not because replication number has a continuous ordering.
    The matched-exercise-grid CRR reference is drawn as a horizontal line; the
    continuous-exercise CRR value is reported in the summary table instead of a
    second line, to keep the chart from getting crowded.
    """
    replications = [row.replication for row in study.replications]
    same_sample = [row.same_sample_estimate for row in study.replications]
    independent = [row.independent_estimate for row in study.replications]
    training_seeds = [row.training_seed for row in study.replications]
    valuation_seeds = [row.valuation_seed for row in study.replications]

    figure = go.Figure()
    for replication, same, indep in zip(replications, same_sample, independent, strict=True):
        figure.add_trace(
            go.Scatter(
                x=[replication, replication],
                y=[same, indep],
                mode="lines",
                line={"width": 1, "color": "rgba(91, 155, 213, 0.5)"},
                showlegend=False,
                hoverinfo="skip",
            )
        )
    figure.add_trace(
        go.Scatter(
            x=replications,
            y=same_sample,
            mode="markers",
            marker={"size": 9, "color": REFERENCE_RED, "symbol": "circle"},
            name="Same-sample estimate",
            customdata=training_seeds,
            hovertemplate=(
                "Replication=%{x}<br>Same-sample=%{y:.4f}<br>Training seed=%{customdata}"
                "<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=replications,
            y=independent,
            mode="markers",
            marker={"size": 9, "color": PRIMARY_BLUE, "symbol": "diamond"},
            name="Independent (out-of-sample) estimate",
            customdata=valuation_seeds,
            hovertemplate=(
                "Replication=%{x}<br>Independent=%{y:.4f}<br>Valuation seed=%{customdata}"
                "<extra></extra>"
            ),
        )
    )
    figure.add_hline(
        y=study.matched_grid_crr,
        line_dash="dot",
        line_color=BINOMIAL_GREEN,
        line_width=3,
        annotation_text="Matched-grid CRR reference",
        annotation_position="bottom right",
    )
    figure.update_layout(
        title="Paired replication validation: same-sample vs independent policy value",
        xaxis_title="Replication",
        xaxis={"dtick": 1},
        yaxis_title="Estimated American put value",
        template="plotly_white",
        height=440,
        hovermode="closest",
    )
    return figure


def exercise_figure(
    time_grid: np.ndarray,
    exercise_percentages: np.ndarray,
    out_of_the_money_percentage: float,
) -> go.Figure:
    """Plot the share of independent validation paths exercised at each time point.

    Takes explicit arrays rather than a ``PricingResult`` so the same chart can be
    fed the out-of-sample stopping statistics from an ``IndependentPolicyEvaluation``
    (see ``lsmc_policy.py``) -- how often the frozen fitted policy exercises when
    applied to new paths, not how often it exercised on its own training data.
    """
    early_exercise_percentage = float(np.sum(exercise_percentages[:-1]))
    bar_width = (
        0.8 * float(np.min(np.diff(time_grid))) if time_grid.size > 1 else None
    )
    figure = go.Figure(
        go.Bar(
            x=time_grid,
            y=exercise_percentages,
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
        text=(
            f"Exercised before maturity: {early_exercise_percentage:.1f}% of paths"
            f"<br>Out-of-the-money at maturity: {out_of_the_money_percentage:.1f}% of paths"
        ),
        showarrow=False,
    )
    figure.update_layout(
        title="LSMC stopping policy by exercise time (out-of-sample)",
        xaxis_title="Exercise time (years, final point is maturity)",
        yaxis_title="Percentage of independent validation paths",
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
