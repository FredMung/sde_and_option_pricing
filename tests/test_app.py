from pathlib import Path

from streamlit.testing.v1 import AppTest

from option_pricing_app.app.dashboard import (
    DISCLAIMER,
    MANUAL_MATURITY,
    determine_manual_mode,
    is_plausible_implied_volatility,
)


def test_manual_mode_rule():
    assert determine_manual_mode("Manual price", None, None)
    assert determine_manual_mode("Live ticker", MANUAL_MATURITY, None)
    assert determine_manual_mode("Live ticker", "2030-01-01", "network error")
    assert not determine_manual_mode("Live ticker", "2030-01-01", None)


def test_implied_volatility_data_quality_range():
    assert not is_plausible_implied_volatility(0.0078)
    assert is_plausible_implied_volatility(0.20)
    assert not is_plausible_implied_volatility(3.01)


def test_app_starts_and_generates_manual_result():
    path = Path(__file__).parents[1] / "streamlit_app.py"
    app = AppTest.from_file(str(path))
    app.session_state["spot_source"] = "Manual price"
    app.session_state["_manual_mode"] = False
    app.session_state["manual_volatility"] = 499.0
    app.session_state["manual_rate"] = -88.0
    app.run(timeout=30)
    assert not app.exception
    assert any(DISCLAIMER in warning.value for warning in app.warning)
    assert next(
        widget.value for widget in app.number_input if widget.label == "Volatility (%)"
    ) == 20.0
    assert next(
        widget.value for widget in app.number_input if widget.label == "Risk-free rate (%)"
    ) == 4.0
    next(widget for widget in app.selectbox if widget.label == "Exercise style").select(
        "European"
    )
    for widget in app.number_input:
        if widget.label == "Number of paths":
            widget.set_value(100)
        elif widget.label == "Number of time steps":
            widget.set_value(10)
    next(button for button in app.button if button.label == "Generate simulation").click()
    app.run(timeout=30)
    assert not app.exception
    assert [metric.label for metric in app.metric] == [
        "Market put value",
        "Manually input volatility",
        "Manually input risk-free rate",
        "Estimated European put value",
        "Exact European put value",
        "Monte Carlo standard error",
        "Approximate 95% interval",
    ]
    assert next(metric for metric in app.metric if metric.label == "Market put value").delta == ""

    next(widget for widget in app.selectbox if widget.label == "Exercise style").select(
        "American"
    )
    next(button for button in app.button if button.label == "Generate simulation").click()
    app.run(timeout=30)
    assert not app.exception
    assert [metric.label for metric in app.metric] == [
        "Market put value",
        "Manually input volatility",
        "Manually input risk-free rate",
        "Estimated American put value",
        "Estimated European put value",
        "Monte Carlo standard error",
        "Approximate 95% interval",
    ]
    assert (
        next(
            metric
            for metric in app.metric
            if metric.label == "Estimated European put value"
        ).delta
        == ""
    )
    assert not any(
        widget.label == "Exercise timestep for continuation diagnostic"
        for widget in app.selectbox
    )

    next(widget for widget in app.selectbox if widget.label == "Asset-price model").select(
        "Heston"
    )
    app.run(timeout=30)
    assert not app.exception
    assert any(
        widget.label == "Initial volatility, √v₀ (%)" for widget in app.number_input
    )
    next(button for button in app.button if button.label == "Generate simulation").click()
    app.run(timeout=30)
    assert not app.exception
    result, _inputs = app.session_state["pricing_output"]
    assert result.model_name == "Heston"
    assert result.displayed_variance_paths is not None
