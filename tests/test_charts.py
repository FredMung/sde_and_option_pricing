import numpy as np
import pytest

from option_pricing_app import (
    ExerciseStyle,
    MarketInputs,
    PutContract,
    SimulationConfig,
    price_put,
)
from option_pricing_app.app.charts import (
    BINOMIAL_GREEN,
    PRIMARY_BLUE,
    REFERENCE_RED,
    exercise_boundary_figure,
    exercise_figure,
    exercise_vs_continuation_figure,
    paired_replication_figure,
    path_count_study_figure,
)
from option_pricing_app.numerical_studies import run_path_count_study
from option_pricing_app.policy_validation import run_paired_validation_study


@pytest.fixture
def american_result():
    contract = PutContract(100, 1)
    result = price_put(
        contract,
        MarketInputs(100, 0.2, 0.05),
        SimulationConfig(500, 25, 42, 20),
        ExerciseStyle.AMERICAN,
    )
    return contract, result


def test_exercise_vs_continuation_uses_existing_plotly_conventions(american_result):
    contract, result = american_result
    diagnostic = min(
        result.continuation_diagnostics, key=lambda item: abs(item.time - 0.5)
    )
    figure = exercise_vs_continuation_figure(diagnostic, contract)
    assert figure.layout.height == 430
    assert "Exercise Payoff vs Estimated Continuation Value" in figure.layout.title.text
    assert [trace.name for trace in figure.data] == [
        "Immediate exercise payoff, H(S)",
        "Estimated continuation value, Ĉ(S)",
    ]
    assert figure.data[0].line.color == PRIMARY_BLUE
    assert np.array_equal(figure.data[0].x, diagnostic.price_grid)
    assert np.array_equal(figure.data[1].y, diagnostic.continuation_value)
    if np.isfinite(diagnostic.boundary):
        assert any(shape.line.color == REFERENCE_RED for shape in figure.layout.shapes)


def test_exercise_boundary_uses_regression_crossings_and_strike_reference(american_result):
    contract, result = american_result
    figure = exercise_boundary_figure(result, contract)
    assert figure.layout.height == 430
    assert figure.layout.title.text == "Estimated American Put Early Exercise Boundary"
    boundary_trace = next(
        trace for trace in figure.data if trace.name == "Estimated exercise boundary"
    )
    expected = np.array(
        [
            next(
                (
                    diagnostic.boundary
                    for diagnostic in result.continuation_diagnostics
                    if diagnostic.timestep == timestep
                    and np.isfinite(diagnostic.boundary)
                ),
                np.nan,
            )
            for timestep in range(1, result.time_grid.size - 1)
        ]
    )
    np.testing.assert_allclose(boundary_trace.y, expected, equal_nan=True)
    assert boundary_trace.line.color == PRIMARY_BLUE
    assert boundary_trace.connectgaps is False
    assert any(shape.line.color == REFERENCE_RED for shape in figure.layout.shapes)


def test_exercise_boundary_rejects_european_result():
    contract = PutContract(100, 1)
    result = price_put(
        contract,
        MarketInputs(100, 0.2, 0.05),
        SimulationConfig(100, 10, 42, 20),
        ExerciseStyle.EUROPEAN,
    )
    with pytest.raises(ValueError, match="only available for American"):
        exercise_boundary_figure(result, contract)


def test_cumulative_chart_is_not_labelled_as_path_count_convergence(american_result):
    _contract, result = american_result
    from option_pricing_app.app.charts import convergence_figure

    figure = convergence_figure(result)
    assert figure.layout.title.text == "Single-run cumulative estimate"
    assert "convergence" not in figure.layout.title.text.lower()


def test_path_count_study_chart_uses_empirical_replication_label():
    def fake_fit_policy(contract, market, config, model, heston_inputs):
        return object()

    def fake_generate_paths(contract, market, config, model, heston_inputs):
        return config, None

    def fake_evaluate(policy, paths, variance_paths):
        config = paths
        return type(
            "Evaluation", (), {"mean_estimate": config.n_paths / 100, "standard_error": 0.1}
        )()

    study = run_path_count_study(
        PutContract(100, 1),
        MarketInputs(100, 0.2, 0.05),
        SimulationConfig(250, 5, 42, 1),
        "GBM",
        None,
        42,
        3,
        fake_fit_policy,
        fake_generate_paths,
        fake_evaluate,
    )
    figure = path_count_study_figure(study)
    assert figure.layout.title.text == "Path-count convergence study (out-of-sample)"
    assert figure.layout.height == 430
    assert any(
        trace.name == "95% empirical replication interval" for trace in figure.data
    )
    assert any(trace.name == "Selected path count" for trace in figure.data)


def test_exercise_figure_reports_out_of_sample_stopping_statistics():
    time_grid = np.linspace(0.0, 1.0, 6)
    exercise_percentages = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 50.0])
    figure = exercise_figure(time_grid, exercise_percentages, 12.5)
    assert figure.layout.title.text == "LSMC stopping policy by exercise time (out-of-sample)"
    assert "Out-of-the-money at maturity: 12.5% of paths" in figure.layout.annotations[0].text
    assert np.array_equal(figure.data[0].y, exercise_percentages)


def test_paired_replication_figure_has_two_markers_per_replication_and_matched_grid_line():
    contract = PutContract(100, 1)
    market = MarketInputs(100, 0.2, 0.05)
    config = SimulationConfig(200, 8, 42, 1)
    study = run_paired_validation_study(contract, market, config, base_seed=123, replications=3)
    figure = paired_replication_figure(study)
    same_sample_trace = next(trace for trace in figure.data if trace.name == "Same-sample estimate")
    independent_trace = next(
        trace for trace in figure.data if trace.name == "Independent (out-of-sample) estimate"
    )
    assert len(same_sample_trace.x) == 3
    assert len(independent_trace.x) == 3
    assert any(shape.line.color == BINOMIAL_GREEN for shape in figure.layout.shapes)
