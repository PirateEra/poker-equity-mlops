import pytest
from src.utils.PokerUtils import to_treys_str
from pyspark.sql import SparkSession

@pytest.fixture(scope="session")
def spark_session():
    return SparkSession.builder.master("local[2]").appName("PokerStatelessFeaturesTesting").getOrCreate() # local[2] to properly test multiprocessing

def test_conversion_logic():
    assert to_treys_str(1, 1) == "Ah" # test Ace of Hearts (1, 1)
    assert to_treys_str(2, 10) == "Ts" # test Ten of Spades (2, 10)
    assert to_treys_str(4, 9) == "9c" # test 9 of Clubs (4, 9)

def test_invalid_input():
    with pytest.raises(ValueError):
        to_treys_str(99, 1)  # The suit does not exist