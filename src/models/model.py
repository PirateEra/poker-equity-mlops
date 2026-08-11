import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from configs.configs import ModelConfig, RunConfig, MLFlowConfig
from src.utils.mlflow_utils import create_mlflow_experiment_if_not_exists, start_mlflow_run
import tempfile
import pickle
import os
from pathlib import Path
from mlflow.entities import RunStatus

class PokerEquityModel:
    def __init__(self):
        self.features = [
            'S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5',
            'hand_is_pair', 'hand_is_suited', 'hand_is_connected', 'highest_hand_card',
            'current_hand_class_value', 'current_hand_rank_value',
            'is_flush_hand', 'is_straight_hand',
            'is_straight_draw', 'is_flush_draw',
            'board_is_paired', 'board_is_monotone',
        ]
        self.target = 'equity'
        self.mse = None
        self.r2 = None
    
    @staticmethod
    def _get_model_object() -> RandomForestRegressor:
        return RandomForestRegressor(
            max_depth=ModelConfig.max_depth,
            n_estimators=ModelConfig.n_estimators,
            n_jobs=ModelConfig.n_jobs
        )

    def train(self, dataframe: pd.DataFrame) -> None:
        self.regressor = self._get_model_object()

        X = dataframe[self.features]
        self.input_example = X.head(1)
        y = dataframe[self.target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=RunConfig.test_size, random_state=RunConfig.split_seed)
        self.regressor.fit(X_train, y_train)

        predictions = self.regressor.predict(X_test)
        self.mse = mean_squared_error(y_test, predictions)
        self.r2 = r2_score(y_test, predictions)
    
    def predict(self, predict_data: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "regressor"):
            raise ValueError("Model not trained yet!")
        return self.regressor.predict(predict_data)
    
    def load_model_from_mlflow(self, run_id: str):
        mlflow.set_tracking_uri(MLFlowConfig.tracking_uri)
        mlflow_experiment = mlflow.get_experiment_by_name(MLFlowConfig.experiment_name)
        if mlflow_experiment is None:
            raise RuntimeError(f"Experiment {MLFlowConfig.experiment_name} does not exist in MLFlow. Therefore loading the model has failed.")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model", dst_path=temp_dir, tracking_uri=MLFlowConfig.tracking_uri)
            model_file_name = f"{ModelConfig.model_name}.pkl"
            local_model_path = os.path.join(temp_dir, "model", model_file_name)
            if not os.path.exists(local_model_path):
                raise RuntimeError(f"Model {model_file_name} is not among MLFLow artifacts.")
            with open(local_model_path, "rb") as file:
                self.regressor = pickle.load(file)
    
    def log_model_to_mlflow(self):
        mlflow.set_tracking_uri(MLFlowConfig.tracking_uri)
        create_mlflow_experiment_if_not_exists(MLFlowConfig.experiment_name)
        with start_mlflow_run(MLFlowConfig.training_run_name, MLFlowConfig.experiment_name):
            assert self.mse is not None, "The model is not trained and there are no experiment results."
            mlflow.log_params({
                "features": self.features,
                "target": self.target
            })
            mlflow.log_metrics({
                "mse": self.mse,
                "r2": self.r2
            })
            model_file_name = f"{ModelConfig.model_name}.pkl"
            with tempfile.TemporaryDirectory() as tmp_dir:
                temp_directory = Path(tmp_dir)
                local_model_path = temp_directory / model_file_name
                with open(local_model_path, "wb") as f:
                    pickle.dump(self.regressor, f)
                mlflow.log_artifacts(local_dir=tmp_dir, artifact_path="model")
            mlflow.end_run(RunStatus.to_string(RunStatus.FINISHED))