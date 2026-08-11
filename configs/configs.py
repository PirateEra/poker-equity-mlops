import os
from dataclasses import dataclass, field

@dataclass(init=False, frozen=True)
class PathsConfig:
    intermediate_data_path: str = "data/intermediate/"
    features_data_path: str = "data/intermediate/features"
    stateful_features_data_path: str = "data/intermediate/stateful_features"
    all_features_artifact_data_path = "features/all_features"
    stateful_features_artifact_data_path = "features/stateful_features"

    telemetry_training_data_path: str = "data/telemetry/data_dist.json"
    telemetry_live_data_path: str = "data/telemetry/live_data_dist.json"

@dataclass(init=False, frozen=True)
class RunConfig:
    sample_rate: int = 0.2
    split_seed: int = 42
    test_size: float = 0.2


@dataclass(init=False, frozen=True)
class PreprocessingDataConfig:
    simulation_count: int = 10  # The amount of simulations to run for win equity

@dataclass(init=False, frozen=True)
class ModelConfig:
    model_name: str = "poker_equity"
    max_depth: int = 10
    n_estimators: int = 100
    n_jobs = -1


@dataclass(init=False, frozen=True)
class TelemetryConfig:
    # Latest N live rows used as the "recent" distribution for PSI.
    num_instances_for_live_dist: int = 5
    epsilon: float = 1 / 1e100
    push_gateway_uri: str = "http://prometheus_push_gateway:9091"
    # Binary poker hand / board flags (baseline uses true/false counts).
    binary_features: tuple = (
        "hand_is_pair",
        "hand_is_suited",
        "board_is_paired",
    )
    # Categorical distributions (baseline uses label -> count maps).
    categorical_features: tuple = (
        "equity_bin",
        "current_hand_class_value",
    )
    equity_bin_edges: tuple = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

@dataclass(init=False, frozen=True)
class MLFlowConfig:
    tracking_uri: str = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:8080")
    experiment_name: str = "PokerEquity"
    training_run_name: str = "Poker_Equity_training_run_1"
    feature_engineering_run_name: str = "Poker_Equity_feature_engineering_run_1"


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable {name}. "
            "Copy .env.example to .env and set local credentials."
        )
    return value


class _MYSQLConfig:
    raw_table = "raw_data"
    preprocessed_table = "preprocessed_data"

    @property
    def url(self) -> str:
        database = _require_env("MYSQL_DATABASE")
        host = os.environ.get("MYSQL_HOST", "mysql")
        port = os.environ.get("MYSQL_PORT", "3306")
        return f"jdbc:mysql://{host}:{port}/{database}"

    @property
    def properties(self) -> dict:
        return {
            "user": _require_env("MYSQL_USER"),
            "password": _require_env("MYSQL_PASSWORD"),
            "driver": "com.mysql.cj.jdbc.Driver",
        }


MYSQLConfig = _MYSQLConfig()