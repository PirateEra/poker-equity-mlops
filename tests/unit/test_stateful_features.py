import pytest
from pyspark.sql import SparkSession
from src.features.stateful_features import (
    get_class_relative_rank_z_score,
    get_class_relative_cdf,
    get_global_rank_stength
)

@pytest.fixture(scope="session")
def spark_session():
    return SparkSession.builder.master("local[2]").appName("PokerStatelessFeaturesTesting").getOrCreate() # local[2] to properly test multiprocessing

@pytest.fixture(scope="session")
def test_dataframe(spark_session):
    data = [
        # In this case the mean = 20 and std = 10
        (1, 10),  # Z score of -1.0
        (1, 20),  # Z score of 0.0
        (1, 30),  # Z score of 1.0
        (2, 500), # Different class to test grouping
        (2, 600)
    ]
    columns = ["current_hand_class_value", "current_hand_rank_value"]
    return spark_session.createDataFrame(data, columns)

def test_get_class_relative_rank_z_score(test_dataframe):
    dataframe = get_class_relative_rank_z_score(test_dataframe)
    dataframe = dataframe.collect()

    assert dataframe[0]["current_hand_rank_value"] == 10
    assert dataframe[0]["relative_rank_z"] == pytest.approx(-1.0, 0.01)

    assert dataframe[1]["current_hand_rank_value"] == 20
    assert dataframe[1]["relative_rank_z"] == pytest.approx(0.0, 0.01)

    assert dataframe[2]["current_hand_rank_value"] == 30
    assert dataframe[2]["relative_rank_z"] == pytest.approx(1.0, 0.01)


def test_get_class_relative_cdf(test_dataframe):
    dataframe = get_class_relative_cdf(test_dataframe)
    dataframe = dataframe.collect()

    assert dataframe[0]["relative_rank_cdf"] == pytest.approx(1/3, 0.001)
    assert dataframe[1]["relative_rank_cdf"] == pytest.approx(2/3, 0.001)
    assert dataframe[2]["relative_rank_cdf"] == pytest.approx(1.0, 0.001)


def test_get_global_rank_stength(test_dataframe):
    dataframe = get_global_rank_stength(test_dataframe)
    results = dataframe.collect()
    
    row_600 = [r for r in results if r["current_hand_rank_value"] == 600][0]
    assert row_600["global_rank_strenght"] == pytest.approx(0.0, 0.01)

    row_10 = [r for r in results if r["current_hand_rank_value"] == 10][0]
    assert row_10["global_rank_strenght"] == pytest.approx(1.0, 0.01)