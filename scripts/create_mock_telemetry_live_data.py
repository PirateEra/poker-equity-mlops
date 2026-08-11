"""
Generate mock live prediction telemetry for the poker PSI drift job.

By default writes a drifted live distribution so HighDataDrift can fire in demos.
Pass --matched to approximate the baseline instead.
"""

from datetime import datetime, timedelta
from pathlib import Path

import click
import pandas as pd

from configs.configs import PathsConfig, TelemetryConfig
from src.main_drift_calculation import equity_to_bin


def _build_matched_rows(row_count: int) -> pd.DataFrame:
    """Live rows roughly aligned with data/telemetry/data_dist.json."""
    rows = []
    # Mix of mid equity, unpaired / unsuited hands, high-card / pair classes.
    templates = [
        {"hand_is_pair": False, "hand_is_suited": False, "board_is_paired": False, "equity": 0.45, "current_hand_class_value": "9"},
        {"hand_is_pair": False, "hand_is_suited": True, "board_is_paired": False, "equity": 0.55, "current_hand_class_value": "8"},
        {"hand_is_pair": True, "hand_is_suited": False, "board_is_paired": False, "equity": 0.62, "current_hand_class_value": "8"},
        {"hand_is_pair": False, "hand_is_suited": False, "board_is_paired": True, "equity": 0.35, "current_hand_class_value": "7"},
        {"hand_is_pair": False, "hand_is_suited": False, "board_is_paired": False, "equity": 0.28, "current_hand_class_value": "9"},
        {"hand_is_pair": True, "hand_is_suited": True, "board_is_paired": False, "equity": 0.71, "current_hand_class_value": "6"},
        {"hand_is_pair": False, "hand_is_suited": True, "board_is_paired": False, "equity": 0.48, "current_hand_class_value": "9"},
        {"hand_is_pair": False, "hand_is_suited": False, "board_is_paired": False, "equity": 0.52, "current_hand_class_value": "8"},
    ]
    base_time = datetime(2026, 8, 11, 12, 0, 0)
    for index in range(row_count):
        row = dict(templates[index % len(templates)])
        row["timestamp"] = base_time + timedelta(minutes=index)
        row["equity_bin"] = equity_to_bin(row["equity"])
        rows.append(row)
    return pd.DataFrame(rows)


def _build_drifted_rows(row_count: int) -> pd.DataFrame:
    """
    Live rows skewed vs baseline: many pairs/suited boards, high equity bins,
    stronger hand classes — enough to push PSI above the 0.2 alert threshold.
    """
    rows = []
    templates = [
        {"hand_is_pair": True, "hand_is_suited": True, "board_is_paired": True, "equity": 0.92, "current_hand_class_value": "2"},
        {"hand_is_pair": True, "hand_is_suited": True, "board_is_paired": True, "equity": 0.88, "current_hand_class_value": "3"},
        {"hand_is_pair": True, "hand_is_suited": True, "board_is_paired": False, "equity": 0.95, "current_hand_class_value": "1"},
        {"hand_is_pair": True, "hand_is_suited": False, "board_is_paired": True, "equity": 0.85, "current_hand_class_value": "2"},
        {"hand_is_pair": True, "hand_is_suited": True, "board_is_paired": True, "equity": 0.91, "current_hand_class_value": "4"},
        {"hand_is_pair": True, "hand_is_suited": True, "board_is_paired": True, "equity": 0.97, "current_hand_class_value": "0"},
        {"hand_is_pair": True, "hand_is_suited": True, "board_is_paired": False, "equity": 0.86, "current_hand_class_value": "3"},
        {"hand_is_pair": True, "hand_is_suited": False, "board_is_paired": True, "equity": 0.89, "current_hand_class_value": "2"},
    ]
    base_time = datetime(2026, 8, 11, 18, 0, 0)
    for index in range(row_count):
        row = dict(templates[index % len(templates)])
        row["timestamp"] = base_time + timedelta(minutes=index)
        row["equity_bin"] = equity_to_bin(row["equity"])
        rows.append(row)
    return pd.DataFrame(rows)


@click.command()
@click.option(
    "--matched",
    is_flag=True,
    help="Write live rows aligned with the baseline (low PSI) instead of drifted rows.",
)
@click.option(
    "--rows",
    default=None,
    type=int,
    show_default=False,
    help="Number of live rows to write (default: TelemetryConfig.num_instances_for_live_dist).",
)
def main(matched: bool, rows: int | None):
    row_count = rows if rows is not None else TelemetryConfig.num_instances_for_live_dist
    predictions = _build_matched_rows(row_count) if matched else _build_drifted_rows(row_count)

    output_path = Path(PathsConfig.telemetry_live_data_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_json(output_path)
    print(f"Wrote {len(predictions)} live telemetry rows to {output_path}")


if __name__ == "__main__":
    main()
