from contextlib import asynccontextmanager
from functools import cache
from http import HTTPStatus
import os
import re
import tempfile

from fastapi import FastAPI, Request
from prometheus_client import Counter, Gauge, Histogram, Info, Summary, make_asgi_app
from pydantic import BaseModel, Field
from starlette.routing import Mount
import mlflow

from configs.configs import MLFlowConfig, ModelConfig, PathsConfig
from src.features.feature_engineering import feature_engineer_stateless_poker_data
from src.models.model import PokerEquityModel
from src.utils.mlflow_utils import get_latest_run_id
from src.utils.utils import get_spark_session

spark = get_spark_session()

APP_NAME = ModelConfig.model_name

pred_counter = Counter(
    f"{APP_NAME}_predictions_total",
    "Count of equity predictions served",
)
equity_histogram = Histogram(
    f"{APP_NAME}_equity",
    "Predicted poker equity values",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
model_load_latency = Histogram(
    f"{APP_NAME}_model_load_seconds",
    "Time spent loading the model and feature lookup from cache/MLflow",
)
model_inference_latency = Histogram(
    f"{APP_NAME}_model_inference_seconds",
    "Time spent on model.predict()",
)
api_errors_counter = Counter(
    f"{APP_NAME}_errors_total",
    "Total count of API errors",
    labelnames=["error_type"],
)
request_payload_size = Summary(
    f"{APP_NAME}_request_payload_size",
    "Size of incoming prediction requests in bytes",
)
current_hand_rank_gauge = Gauge(
    f"{APP_NAME}_current_hand_rank_value",
    "current_hand_rank_value from the last prediction request",
)
model_version_info = Info(
    f"{APP_NAME}_model_version",
    "Model version information",
)
model_version_info.info(
    {
        "experiment_name": MLFlowConfig.experiment_name,
        "training_run_name": MLFlowConfig.training_run_name,
        "model_name": ModelConfig.model_name,
    }
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(MLFlowConfig.tracking_uri)
    print("Getting the model...", flush=True)
    get_model()
    print("Getting the feature dataframe", flush=True)
    get_feature_dataframe()
    print("Startup complete: Model loaded.", flush=True)
    yield


app = FastAPI(title="Poker Equity API", lifespan=lifespan)

# Mount /metrics without the Starlette trailing-slash redirect so Prometheus can scrape /metrics.
metrics_route = Mount("/metrics", make_asgi_app())
metrics_route.path_regex = re.compile(r"^/metrics(?P<path>.*)$")
app.routes.append(metrics_route)


class PokerEquityResponse(BaseModel):
    equity: float


class PokerHandRequest(BaseModel):
    S1: int = Field(..., ge=1, le=4, description="Suit of Card 1")
    C1: int = Field(..., ge=1, le=13, description="Rank of Card 1")
    S2: int = Field(..., ge=1, le=4, description="Suit of Card 2")
    C2: int = Field(..., ge=1, le=13, description="Rank of Card 2")
    S3: int = Field(..., ge=1, le=4, description="Suit of Board Card 1")
    C3: int = Field(..., ge=1, le=13, description="Rank of Board Card  1")
    S4: int = Field(..., ge=1, le=4, description="Suit of Board Card  2")
    C4: int = Field(..., ge=1, le=13, description="Rank of Board Card  2")
    S5: int = Field(..., ge=1, le=4, description="Suit of Board Card  3")
    C5: int = Field(..., ge=1, le=13, description="Rank of Board Card  3")


@cache
def get_model():
    run_id = get_latest_run_id(MLFlowConfig.training_run_name, MLFlowConfig.experiment_name)
    print("Got the run_id...", flush=True)
    model = PokerEquityModel()
    print("Loading the model from mlflow...", flush=True)
    model.load_model_from_mlflow(run_id)
    return model


@cache
def get_feature_dataframe():
    run_id = get_latest_run_id(MLFlowConfig.feature_engineering_run_name, MLFlowConfig.experiment_name)
    print("Got the run_id...", flush=True)
    print("Loading the dataframe from mlflow...", flush=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=PathsConfig.stateful_features_artifact_data_path,
            dst_path=temp_dir,
            tracking_uri=MLFlowConfig.tracking_uri,
        )
        parquet_path = os.path.join(temp_dir, "stateful_features")
        lookup_dataframe = spark.read.parquet(parquet_path)
        lookup_dataframe = lookup_dataframe.toPandas()
        print("Successfully loaded Spark DataFrame.", flush=True)

    return lookup_dataframe


@app.middleware("http")
async def record_status_code_errors(request: Request, call_next):
    response = await call_next(request)
    if response.status_code >= 400:
        api_errors_counter.labels(error_type=str(response.status_code)).inc()
    return response


@app.get("/health")
def health_check():
    return {"status": "OK"}


@app.post("/reload")
def reload_model_endpoint():
    get_model.cache_clear()
    get_feature_dataframe.cache_clear()
    get_model()
    get_feature_dataframe()
    return "Model reload was a success", HTTPStatus.OK


@app.post("/predict", response_model=PokerEquityResponse)
def predict(request: PokerHandRequest, raw_request: Request):
    """
    Predict using the model through the use of raw cards it calculates features, and returns Equity.
    """
    content_length = raw_request.headers.get("content-length")
    if content_length:
        request_payload_size.observe(int(content_length))

    with model_load_latency.time():
        model = get_model()
        stateful_features = get_feature_dataframe()

    data_dict = request.model_dump()
    request_dataframe = spark.createDataFrame([data_dict])

    features_dataframe = feature_engineer_stateless_poker_data(request_dataframe)
    features_dataframe = features_dataframe.toPandas()
    current_hand_rank_gauge.set(float(features_dataframe["current_hand_rank_value"].iloc[0]))

    pandas_features_dataframe = features_dataframe.merge(
        stateful_features, on="current_hand_rank_value", how="left"
    ).fillna(0)
    pandas_features_dataframe = pandas_features_dataframe[model.regressor.feature_names_in_]

    with model_inference_latency.time():
        prediction = model.predict(pandas_features_dataframe)
    equity_value = float(prediction[0])

    pred_counter.inc()
    equity_histogram.observe(equity_value)
    return {"equity": equity_value}