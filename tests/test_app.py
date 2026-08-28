from pathlib import Path

import pytest
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
        "Spot price",
        "Strike",
        "Maturity",
        "Market put value",
        "Manually input volatility",
        "Manually input risk-free rate",
        "Estimated European put value",
        "Exact European put value",
        "Monte Carlo standard error",
        "Approximate 95% interval (European put value)",
    ]
    assert next(metric for metric in app.metric if metric.label == "Market put value").delta == ""

    next(widget for widget in app.selectbox if widget.label == "Exercise style").select(
        "American"
    )
    next(button for button in app.button if button.label == "Generate simulation").click()
    app.run(timeout=30)
    assert not app.exception
    assert [metric.label for metric in app.metric] == [
        "Spot price",
        "Strike",
        "Maturity",
        "Market put value",
        "Manually input volatility",
        "Manually input risk-free rate",
        "Estimated American put value (out-of-sample)",
        "Estimated European put value",
        "Early-exercise premium (American − European)",
        "Monte Carlo standard error (out-of-sample)",
        "Approximate 95% interval (out-of-sample American put value)",
    ]
    assert (
        next(
            metric
            for metric in app.metric
            if metric.label == "Estimated European put value"
        ).delta
        == ""
    )
    premium_metric = next(
        metric
        for metric in app.metric
        if metric.label == "Early-exercise premium (American − European)"
    )
    assert premium_metric.delta.endswith("approx. SE")

    # The headline out-of-sample estimate and the Early-Exercise Policy table's LSMC
    # row are computed from the same shared independent evaluation, so they must
    # report the identical number.
    headline_metric = next(
        metric
        for metric in app.metric
        if metric.label == "Estimated American put value (out-of-sample)"
    )
    headline_value = float(headline_metric.value.replace(",", ""))
    policy_table = next(
        df.value
        for df in app.dataframe
        if {"Method", "American", "Early exercise value"} <= set(df.value.columns)
    )
    lsmc_row = policy_table[policy_table["Method"].str.contains("out-of-sample")]
    assert not lsmc_row.empty
    # headline_value is parsed from a 4-decimal display string, so allow rounding.
    assert float(lsmc_row["American"].iloc[0]) == pytest.approx(headline_value, abs=1e-4)
    assert not any(
        widget.label == "Exercise timestep for continuation diagnostic"
        for widget in app.selectbox
    )
    # Numerical studies now run automatically alongside the main simulation, with no
    # separate trigger button.
    assert not any(button.label == "Run numerical studies" for button in app.button)
    assert any(
        widget.label == "Number of replications" for widget in app.number_input
    )
    assert "numerical_studies_output" in app.session_state
    assert len(app.session_state["numerical_studies_output"].path_count.derived_seeds) == 20
    next(
        widget
        for widget in app.number_input
        if widget.label == "Number of replications"
    ).set_value(3)
    app.run(timeout=60)
    assert not app.exception
    studies = app.session_state["numerical_studies_output"]
    assert len(studies.path_count.derived_seeds) == 3
    assert studies.exercise_grid is not None

    # Numerical validation: paired same-sample vs independent replication study.
    next(
        widget
        for widget in app.number_input
        if widget.label == "Number of paired replications"
    ).set_value(3)
    app.run(timeout=30)
    assert not app.exception
    validation_table = next(
        df.value
        for df in app.dataframe
        if "Validation measure" in df.value.columns
    )
    assert list(validation_table["Validation measure"]) == [
        "Continuous-exercise CRR",
        "Matched-grid CRR",
        "Mean same-sample LSMC",
        "Same-sample empirical SD",
        "Mean independent policy value",
        "Independent empirical SD",
        "Mean paired gap",
        "Independent mean error against matched CRR",
        "Independent RMSE against matched CRR",
    ]
    download_labels = {button.label for button in app.get("download_button")}
    assert "Download numerical validation (CSV)" in download_labels
    assert "Download numerical validation charts (PDF)" in download_labels

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
    assert result.fitted_policy is not None
    assert any(
        "Numerical validation is available for American exercise under the GBM"
        in info.value
        for info in app.info
    )
