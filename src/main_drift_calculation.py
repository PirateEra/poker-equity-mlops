"""
Entrypoint for calculating poker equity drift monitoring metrics for Prometheus.
"""

import json
import logging
import math
import typing

import pandas as pd
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

from configs.configs import ModelConfig, PathsConfig, TelemetryConfig

logger = logging.getLogger(__name__)


def get_psi(
    training_percentages: typing.Tuple[float, ...],
    latest_percentages: typing.Tuple[float, ...],
) -> float:
    psi = 0.0
    for training, latest in zip(training_percentages, latest_percentages):
        psi += (latest - training) * math.log(latest / training, math.e)
    return psi


def equity_to_bin(equity: float) -> str:
    edges = TelemetryConfig.equity_bin_edges
    for index in range(len(edges) - 1):
        low = edges[index]
        high = edges[index + 1]
        if low <= equity < high or (high == edges[-1] and equity == high):
            return f"{low}-{high}"
    return f"{edges[-2]}-{edges[-1]}"


def _binary_psi(
    training_telemetry_data: dict,
    latest_telemetry_data: pd.DataFrame,
    feature_name: str,
) -> float:
    training_count = (
        training_telemetry_data[feature_name]["true"]
        + training_telemetry_data[feature_name]["false"]
    )
    # Epsilon avoids division-by-zero / log(0) when a category is missing.
    training_percentages = (
        (training_telemetry_data[feature_name]["true"] / training_count) + TelemetryConfig.epsilon,
        (training_telemetry_data[feature_name]["false"] / training_count) + TelemetryConfig.epsilon,
    )

    latest_count = latest_telemetry_data[feature_name].shape[0]
    latest_true = latest_telemetry_data[feature_name].astype(bool).sum()
    latest_percentages = (
        (latest_true / latest_count) + TelemetryConfig.epsilon,
        ((latest_count - latest_true) / latest_count) + TelemetryConfig.epsilon,
    )
    return get_psi(training_percentages, latest_percentages)


def _categorical_psi(
    training_telemetry_data: dict,
    latest_telemetry_data: pd.DataFrame,
    feature_name: str,
) -> float:
    train_distribution = training_telemetry_data[feature_name]
    train_total = sum(train_distribution.values())

    live_counts = latest_telemetry_data[feature_name].astype(str).value_counts()
    live_total = len(latest_telemetry_data)

    train_percentage = []
    live_percentage = []
    for label in train_distribution.keys():
        train_percentage.append((train_distribution[label] / train_total) + TelemetryConfig.epsilon)
        live_percentage.append((live_counts.get(str(label), 0) / live_total) + TelemetryConfig.epsilon)

    return get_psi(tuple(train_percentage), tuple(live_percentage))


def main():  # pragma: no cover
    with open(PathsConfig.telemetry_training_data_path, "r") as file:
        training_telemetry_data = json.load(file)
    with open(PathsConfig.telemetry_live_data_path, "r") as file:
        live_telemetry_data = pd.DataFrame(json.load(file))

    if "equity" in live_telemetry_data.columns and "equity_bin" not in live_telemetry_data.columns:
        live_telemetry_data["equity_bin"] = live_telemetry_data["equity"].map(equity_to_bin)

    live_telemetry_data.sort_values("timestamp", inplace=True, ascending=False)
    latest_telemetry_data = live_telemetry_data.iloc[0 : TelemetryConfig.num_instances_for_live_dist]

    telemetry_data_count = latest_telemetry_data.shape[0]
    if telemetry_data_count < TelemetryConfig.num_instances_for_live_dist:
        logger.warning(
            "Telemetry calculation has %s rows which is less than required %s. Exiting.",
            telemetry_data_count,
            TelemetryConfig.num_instances_for_live_dist,
        )
        raise SystemExit(0)

    psi_values = {}
    for feature_name in TelemetryConfig.binary_features:
        psi_values[feature_name] = _binary_psi(
            training_telemetry_data, latest_telemetry_data, feature_name
        )

    for feature_name in TelemetryConfig.categorical_features:
        psi_values[feature_name] = _categorical_psi(
            training_telemetry_data, latest_telemetry_data, feature_name
        )

    registry = CollectorRegistry()
    psi_gauge = Gauge(
        f"{ModelConfig.model_name}_psi_s",
        "PSI calculations for poker feature and prediction distributions",
        labelnames=["target"],
        registry=registry,
    )
    for target, psi_value in psi_values.items():
        psi_gauge.labels(target=target).set(psi_value)

    push_to_gateway(TelemetryConfig.push_gateway_uri, job="telemetryBatch", registry=registry)


if __name__ == "__main__":
    main()
