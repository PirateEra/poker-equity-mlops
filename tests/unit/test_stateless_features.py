import pytest
from pyspark.sql import SparkSession
from src.features.stateless_features import (
    get_hand_is_pair,
    get_hand_is_suit,
    get_hand_is_connected,
    get_highest_hand_card,
    get_hand_rank_difference,
    get_current_hand_rank_value,
    get_current_hand_class_value,
    get_is_straight_hand,
    get_is_flush_hand,
    get_is_flush_draw,
    get_is_straight_draw,
    get_board_is_monotone,
    get_board_is_paired,
    get_win_equity
)

@pytest.fixture(scope="session")
def spark_session():
    return SparkSession.builder.master("local[2]").appName("PokerStatelessFeaturesTesting").getOrCreate() # local[2] to properly test multiprocessing

@pytest.fixture(scope="session")
def testing_dataframe(spark_session):
    data = [
        (1, 7, 2, 7),   # This is a pair
        (1, 2, 1, 5),   # This is a suit match (both hearts)
        (3, 9, 2, 10),  # This is a connecting hand
        (1, 1, 2, 13),  # This is a ace-king connecting hand (edge case)
        (1, 1, 4, 2)    # This is a ace-two connecting hand (edge case)
    ]
    columns = ["S1", "C1", "S2", "C2"]
    return spark_session.createDataFrame(data, columns)

@pytest.fixture(scope="session")
def advanced_testing_dataframe(spark_session):
    data = [
        (1, 10, 1, 11, 1, 12, 1, 13, 1, 1), # Royal flush hearts
        (2, 2, 2, 3, 2, 4, 2, 5, 1, 9), # Flush draw, 4 spades 1 heart
        (3, 5, 3, 6, 2, 7, 2, 8, 2, 2), # Straight draw (open ended, 5-6-7-8).
        (2, 2, 2, 3, 1, 5, 1, 5, 1, 9), # Monotone and paired board
        (1, 2, 2, 7, 3, 9, 4, 11, 1, 13) # High card garbage hand
    ]
    columns = ["S1", "C1", "S2", "C2", "S3", "C3", "S4", "C4", "S5", "C5"]
    return spark_session.createDataFrame(data, columns)

def test_get_hand_is_pair(testing_dataframe):
    dataframe = get_hand_is_pair(testing_dataframe)
    results = dataframe.collect()

    assert results[0]["hand_is_pair"] == 1
    assert results[1]["hand_is_pair"] == 0

def test_get_hand_is_suited(testing_dataframe):
    dataframe = get_hand_is_suit(testing_dataframe)
    results = dataframe.collect()
    
    assert results[1]["hand_is_suited"] == 1
    assert results[0]["hand_is_suited"] == 0

def test_get_hand_is_connected(testing_dataframe):
    dataframe = get_hand_is_connected(testing_dataframe)
    results = dataframe.collect()

    assert results[2]["hand_is_connected"] == 1
    assert results[3]["hand_is_connected"] == 1
    assert results[4]["hand_is_connected"] == 1
    assert results[0]["hand_is_connected"] == 0

def test_get_highest_hand_card(testing_dataframe):
    dataframe = get_highest_hand_card(testing_dataframe)
    results = dataframe.collect()

    assert results[0]["highest_hand_card"] == 7
    assert results[3]["highest_hand_card"] == 14

def test_get_hand_rank_difference(testing_dataframe):
    dataframe = get_hand_rank_difference(testing_dataframe)
    results = dataframe.collect()

    assert results[0]["hand_rank_diff"] == 0
    assert results[2]["hand_rank_diff"] == 1
    assert results[3]["hand_rank_diff"] == 12

def test_get_current_hand_rank_value(advanced_testing_dataframe):
    dataframe = get_current_hand_rank_value(advanced_testing_dataframe)
    results = dataframe.collect()
    assert results[0]["current_hand_rank_value"] == 1 # Check royal flush
    assert results[4]["current_hand_rank_value"] > 6000  # Check garbage hand

def test_get_current_hand_class_value(advanced_testing_dataframe):
    dataframe = get_current_hand_rank_value(advanced_testing_dataframe) 
    dataframe = get_current_hand_class_value(dataframe)
    results = dataframe.collect()
    assert results[0]["current_hand_class_value"] == 0 # Check royal flush
    assert results[4]["current_hand_class_value"] == 9 # Check high card

def test_get_is_straight_hand(advanced_testing_dataframe):
    dataframe = get_current_hand_rank_value(advanced_testing_dataframe)
    dataframe = get_current_hand_class_value(dataframe)
    dataframe = get_is_straight_hand(dataframe)
    results = dataframe.collect()

    assert results[0]["is_straight_hand"] == 1 # Check royal flush
    assert results[2]["is_straight_hand"] == 0 # Open ended, so not a straight

def test_get_is_flush_hand(advanced_testing_dataframe):
    dataframe = get_current_hand_rank_value(advanced_testing_dataframe)
    dataframe = get_current_hand_class_value(dataframe)
    dataframe = get_is_flush_hand(dataframe)
    results = dataframe.collect()

    assert results[0]["is_flush_hand"] == 1 # Royal flush
    assert results[1]["is_flush_hand"] == 0 # Not a flush yet

def test_get_is_flush_draw(advanced_testing_dataframe):
    dataframe = get_is_flush_draw(advanced_testing_dataframe)
    results = dataframe.collect()

    assert results[1]["is_flush_draw"] == 1 # 4 spades, 1 heart
    assert results[0]["is_flush_draw"] == 0 # 5 hearts, not a flush draw

def test_get_is_straight_draw(advanced_testing_dataframe):
    dataframe = get_is_straight_draw(advanced_testing_dataframe)
    results = dataframe.collect()

    assert results[2]["is_straight_draw"] == 1 # A draw hand
    assert results[4]["is_straight_draw"] == 0 # No draw, no connection

def test_board_features(advanced_testing_dataframe):
    dataframe = get_board_is_monotone(advanced_testing_dataframe)
    dataframe = get_board_is_paired(dataframe) # User specific naming "add_"
    results = dataframe.collect()

    assert results[3]["board_is_monotone"] == 1 # Monotone hearts
    assert results[3]["board_is_paired"] == 1 # Paired (pair of 5s)
    assert results[4]["board_is_monotone"] == 0 # mixed suits
    assert results[4]["board_is_paired"] == 0 # Not paired

def test_equity_calculation_range(advanced_testing_dataframe):
    dataframe = get_win_equity(advanced_testing_dataframe)
    equity = dataframe.first()["equity"] # Royal flush
    
    assert isinstance(equity, float)
    assert 0.0 <= equity <= 1.0
    # A royal flush hand should have high equity
    assert equity > 0.75