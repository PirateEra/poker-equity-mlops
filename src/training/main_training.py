import click
import mlflow
import pyspark.sql.functions as F
import pandas as pd
from src.models.model import PokerEquityModel
from configs.configs import PathsConfig, MLFlowConfig, RunConfig
from src.data_loading.load_data import load_raw_data, load_preprocessed_data, load_feature_data
from src.preprocessing.preprocess_data import preprocess_poker_data, save_preprocessed_data
from src.features.feature_engineering import feature_engineer_poker_data
from mlflow.entities import RunStatus
from src.utils.mlflow_utils import (
    create_mlflow_experiment_if_not_exists, 
    start_mlflow_run,
    log_params,
    log_metrics
)

@click.command()
@click.option("--preprocess", is_flag=True, help="If we need to first run the preprocessing step.")
@click.option("--feature_engineering", is_flag=True, help="If we need to run the feature engineering step on the preprocessed data.")
@click.option("--training", is_flag=True, help="If we need to run the training step.")
def main_training(preprocess: bool, feature_engineering: bool, training: bool):
    if preprocess:
        data = load_raw_data()
        data = preprocess_poker_data(data)
        save_preprocessed_data(data)

    if feature_engineering:
        data = load_preprocessed_data()
        data = feature_engineer_poker_data(data)
        spark = data.sparkSession
        data.write.parquet(PathsConfig.features_data_path, mode="overwrite")

        features_dataframe = pd.read_parquet(PathsConfig.features_data_path)
        stateful_features_dataframe = features_dataframe[[
            "current_hand_rank_value", 
            "relative_rank_z", 
            "relative_rank_cdf", 
            "global_rank_strenght"
        ]].drop_duplicates(subset=["current_hand_rank_value"])
        stateful_features_dataframe_spark = spark.createDataFrame(stateful_features_dataframe)
        stateful_features_dataframe_spark.write.parquet(PathsConfig.stateful_features_data_path, mode="overwrite")

        mlflow.set_tracking_uri(MLFlowConfig.tracking_uri)
        create_mlflow_experiment_if_not_exists(MLFlowConfig.experiment_name)
        with start_mlflow_run(MLFlowConfig.feature_engineering_run_name, MLFlowConfig.experiment_name):
            mlflow.log_artifacts(
                local_dir=PathsConfig.features_data_path, 
                artifact_path=PathsConfig.all_features_artifact_data_path
            )
            mlflow.log_artifacts(
                local_dir=PathsConfig.stateful_features_data_path, 
                artifact_path=PathsConfig.stateful_features_artifact_data_path
            )
            log_metrics({
                "num_rows": data.count(),
                "num_columns": len(data.columns)
            })
            log_params({
                "sample_rate": RunConfig.sample_rate
            })
            mlflow.end_run(RunStatus.to_string(RunStatus.FINISHED))

    if training:
        data = load_feature_data().toPandas()
        model = PokerEquityModel()
        model.train(data)
        model.log_model_to_mlflow()
        

if __name__ == "__main__":
    main_training()
