"""Unit tests for PSI / drift helpers (hermetic; no Pushgateway)."""

import math

import pandas as pd
import pytest

from configs.configs import TelemetryConfig
from src.main_drift_calculation import (
    _binary_psi,
    _categorical_psi,
    equity_to_bin,
    get_psi,
)


def test_get_psi_identical_distributions_is_zero():
    percentages = (0.5, 0.5)
    assert get_psi(percentages, percentages) == pytest.approx(0.0)


def test_get_psi_known_shift():
    # Classic two-bin PSI: train (0.7, 0.3) vs live (0.4, 0.6)
    training = (0.7, 0.3)
    latest = (0.4, 0.6)
    expected = (0.4 - 0.7) * math.log(0.4 / 0.7) + (0.6 - 0.3) * math.log(0.6 / 0.3)
    assert get_psi(training, latest) == pytest.approx(expected)


def test_get_psi_with_epsilon_avoids_log_zero():
    epsilon = TelemetryConfig.epsilon
    training = (1.0 - epsilon, epsilon)
    latest = (epsilon, 1.0 - epsilon)
    # Should be finite and large when distributions barely overlap.
    psi = get_psi(training, latest)
    assert math.isfinite(psi)
    assert psi > 1.0


@pytest.mark.parametrize(
    "equity, expected_bin",
    [
        (0.0, "0.0-0.2"),
        (0.199, "0.0-0.2"),
        (0.2, "0.2-0.4"),
        (0.5, "0.4-0.6"),
        (0.8, "0.8-1.0"),
        (1.0, "0.8-1.0"),
        (-0.1, "0.8-1.0"),  # out of range falls through to last bin label
        (1.5, "0.8-1.0"),
    ],
)
def test_equity_to_bin(equity, expected_bin):
    assert equity_to_bin(equity) == expected_bin


def test_binary_psi_matched_live_near_zero():
    training = {"hand_is_pair": {"true": 50, "false": 50}}
    live = pd.DataFrame({"hand_is_pair": [True, False, True, False, True, False]})
    psi = _binary_psi(training, live, "hand_is_pair")
    assert psi == pytest.approx(0.0, abs=1e-9)


def test_binary_psi_full_drift_is_large():
    training = {"hand_is_pair": {"true": 10, "false": 90}}
    live = pd.DataFrame({"hand_is_pair": [True] * 10})
    psi = _binary_psi(training, live, "hand_is_pair")
    assert psi > 0.2


def test_binary_psi_missing_category_uses_epsilon():
    training = {"hand_is_pair": {"true": 100, "false": 0}}
    # Live has only False — training "false" share is 0, epsilon keeps log defined.
    live = pd.DataFrame({"hand_is_pair": [False, False, False, False, False]})
    psi = _binary_psi(training, live, "hand_is_pair")
    assert math.isfinite(psi)
    assert psi > 0.0


def test_categorical_psi_matched_live_near_zero():
    training = {
        "equity_bin": {
            "0.0-0.2": 1,
            "0.2-0.4": 1,
            "0.4-0.6": 1,
            "0.6-0.8": 1,
            "0.8-1.0": 1,
        }
    }
    live = pd.DataFrame(
        {
            "equity_bin": [
                "0.0-0.2",
                "0.2-0.4",
                "0.4-0.6",
                "0.6-0.8",
                "0.8-1.0",
            ]
        }
    )
    psi = _categorical_psi(training, live, "equity_bin")
    assert psi == pytest.approx(0.0, abs=1e-9)


def test_categorical_psi_short_window_missing_labels():
    training = {
        "equity_bin": {
            "0.0-0.2": 50,
            "0.2-0.4": 50,
            "0.4-0.6": 50,
            "0.6-0.8": 50,
            "0.8-1.0": 50,
        }
    }
    # Short live window only hits one bin — remaining labels get epsilon share.
    live = pd.DataFrame({"equity_bin": ["0.8-1.0", "0.8-1.0", "0.8-1.0"]})
    psi = _categorical_psi(training, live, "equity_bin")
    assert math.isfinite(psi)
    assert psi > 0.2


def test_categorical_psi_string_label_alignment():
    training = {"current_hand_class_value": {"0": 10, "9": 90}}
    live = pd.DataFrame({"current_hand_class_value": [9, 9, 9, 9, 0]})
    psi = _categorical_psi(training, live, "current_hand_class_value")
    assert math.isfinite(psi)
    # 4/5 vs 90/100 is a small shift — PSI should stay modest.
    assert 0.0 <= psi < 0.2
