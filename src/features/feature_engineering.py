from pyspark.sql import DataFrame
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
    get_board_is_paired
)
from src.features.stateful_features import (
    get_class_relative_rank_z_score,
    get_class_relative_cdf,
    get_global_rank_stength
)

def feature_engineer_poker_data(dataframe : DataFrame) -> DataFrame:
    """
    This function, creates all features for a given preprocessed equity dataframe
    """
    dataframe = feature_engineer_stateless_poker_data(dataframe)
    dataframe = feature_engineer_statefull_data(dataframe)
    return dataframe

def feature_engineer_stateless_poker_data(dataframe : DataFrame) -> DataFrame:
    """
    This function, creates all stateless features for a given preprocessed equity dataframe
    """
    dataframe = get_hand_is_pair(dataframe)
    dataframe = get_hand_is_suit(dataframe)
    dataframe = get_hand_is_connected(dataframe)
    dataframe = get_highest_hand_card(dataframe)
    dataframe = get_hand_rank_difference(dataframe)
    dataframe = get_current_hand_rank_value(dataframe)
    dataframe = get_current_hand_class_value(dataframe)
    dataframe = get_is_straight_hand(dataframe)
    dataframe = get_is_flush_hand(dataframe)
    dataframe = get_is_flush_draw(dataframe)
    dataframe = get_is_straight_draw(dataframe)
    dataframe = get_board_is_monotone(dataframe)
    dataframe = get_board_is_paired(dataframe)
    return dataframe

def feature_engineer_statefull_data(dataframe : DataFrame) -> DataFrame:
    """
    This function, creates all stateful features for a given dataframe
    it assumes the dataframe contains current_hand_class_value and current_hand_rank_value columns
    """
    dataframe = get_class_relative_rank_z_score(dataframe)
    dataframe = get_class_relative_cdf(dataframe)
    dataframe = get_global_rank_stength(dataframe)
    return dataframe