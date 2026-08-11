"""API /health and /metrics tests with Spark / MLflow / model mocked."""

from contextlib import asynccontextmanager
import sys
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY


def _unregister_app_metrics(app_name: str = "poker_equity") -> None:
    """Drop previously registered collectors so re-import does not duplicate series."""
    for collector in list(REGISTRY._collector_to_names):
        names = REGISTRY._collector_to_names.get(collector, set())
        if any(name.startswith(app_name) for name in names):
            try:
                REGISTRY.unregister(collector)
            except KeyError:
                pass


@pytest.fixture(scope="module")
def api_client():
    mock_spark = MagicMock(name="spark")
    mock_mlflow = MagicMock(name="mlflow")

    _unregister_app_metrics()
    sys.modules.pop("src.api.main_api", None)

    # Patch heavy deps before importing the API (module-level spark + imports).
    with (
        patch("src.utils.utils.get_spark_session", return_value=mock_spark),
        patch.dict(
            "sys.modules",
            {
                "mlflow": mock_mlflow,
                "src.models.model": MagicMock(name="PokerEquityModelModule"),
                "src.features.feature_engineering": MagicMock(name="feature_engineering"),
                "src.utils.mlflow_utils": MagicMock(name="mlflow_utils"),
            },
        ),
    ):
        import src.api.main_api as main_api

        main_api.spark = mock_spark

        @asynccontextmanager
        async def _noop_lifespan(app):
            yield

        main_api.app.router.lifespan_context = _noop_lifespan

        with TestClient(main_api.app) as client:
            yield client, main_api


def test_health_returns_ok(api_client):
    client, _ = api_client
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}


def test_metrics_exposes_prometheus_text(api_client):
    client, main_api = api_client
    response = client.get("/metrics")
    assert response.status_code == 200
    body = response.text
    assert "text/plain" in response.headers.get("content-type", "")
    assert f"{main_api.APP_NAME}_predictions_total" in body
    assert f"{main_api.APP_NAME}_equity" in body
    assert f"{main_api.APP_NAME}_errors_total" in body
    assert f"{main_api.APP_NAME}_model_inference_seconds" in body


def test_metrics_reflects_prediction_counter_increment(api_client):
    client, main_api = api_client
    before = main_api.pred_counter._value.get()
    main_api.pred_counter.inc()
    response = client.get("/metrics")
    assert response.status_code == 200
    assert f"{main_api.APP_NAME}_predictions_total {before + 1.0}" in response.text


def test_health_does_not_require_model_load(api_client):
    """Lifespan is no-op in tests; /health must still succeed without MLflow."""
    client, main_api = api_client
    with patch.object(main_api, "get_model", side_effect=AssertionError("should not load")):
        response = client.get("/health")
    assert response.status_code == 200
