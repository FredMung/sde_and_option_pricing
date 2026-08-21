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
    PRIMARY_BLUE,
    REFERENCE_RED,
    STRIKE_PURPLE,
    exercise_boundary_figure,
    exercise_vs_continuation_figure,
)


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
            diagnostic.boundary
            for diagnostic in result.continuation_diagnostics
            if np.isfinite(diagnostic.boundary)
        ]
    )
    np.testing.assert_allclose(boundary_trace.y, expected)
    assert boundary_trace.line.color == PRIMARY_BLUE
    assert any(shape.line.color == STRIKE_PURPLE for shape in figure.layout.shapes)


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
