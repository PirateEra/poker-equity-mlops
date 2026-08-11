import mlflow
from pathlib import Path
from pyspark.sql import DataFrame

def create_mlflow_experiment_if_not_exists(experiment_name: str):
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        return experiment_id
    return experiment.experiment_id

def start_mlflow_run(run_name: str, experiment_name: str):
    create_mlflow_experiment_if_not_exists(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id
    run = mlflow.start_run(run_name=run_name, experiment_id=experiment_id)
    return run

def log_params(params: dict):
    for key, value in params.items():
        mlflow.log_param(key, value)

def log_metrics(metrics: dict):
    for key, value in metrics.items():
        mlflow.log_metric(key, value)

def get_latest_run_id(run_name: str, experiment_name: str) -> str:
    """
    get the latest run_id based on the run that is in run_config.experiment_name with the most recent end_time.
    """
    mlflow_experiment = mlflow.get_experiment_by_name(experiment_name)
    if mlflow_experiment is None:
        raise RuntimeError(f"Experiment {experiment_name} does not exist in MLFlow.")
    runs = mlflow.search_runs([mlflow_experiment.experiment_id], f"attributes.run_name = '{run_name}'")

    if runs.empty:
        raise RuntimeError(f"Run with name {run_name} is not found in MLFlow.")
    if len(runs[runs["end_time"].isna()]) > 1:
        raise RuntimeError("MLFlow has multiple unfinished runs. Expected one or none unfinished run.")
    if runs[runs["end_time"].isna()].empty:
        latest_run = runs.loc[runs["end_time"].idxmax()]
    else:
        latest_run = runs[runs["end_time"].isna()].iloc[0]

    return latest_run["run_id"]
