from pathlib import Path

from streamlit.testing.v1 import AppTest

from option_pricing_app.app.dashboard import DISCLAIMER, MANUAL_MATURITY, determine_manual_mode


def test_manual_mode_rule():
    assert determine_manual_mode("Manual price", None, None)
    assert determine_manual_mode("Live ticker", MANUAL_MATURITY, None)
    assert determine_manual_mode("Live ticker", "2030-01-01", "network error")
    assert not determine_manual_mode("Live ticker", "2030-01-01", None)


def test_app_starts_and_generates_manual_result():
    path = Path(__file__).parents[1] / "streamlit_app.py"
    app = AppTest.from_file(str(path)).run(timeout=30)
    assert not app.exception
    assert any(DISCLAIMER in warning.value for warning in app.warning)
    for widget in app.number_input:
        if widget.label == "Number of paths":
            widget.set_value(100)
        elif widget.label == "Number of time steps":
            widget.set_value(10)
    next(button for button in app.button if button.label == "Generate simulation").click()
    app.run(timeout=30)
    assert not app.exception
    assert len(app.metric) == 3
