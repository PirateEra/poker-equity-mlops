from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from src.utils.PokerUtils import (
    calculate_treys_hand_class_score,
    calculate_treys_hand_score,
    calculate_equity
)

def get_hand_is_pair(dataframe : DataFrame) -> DataFrame:
    """
    Checks if the hand Cards (1 and 2) form a pair.
    """
    return dataframe.withColumn("hand_is_pair", (F.col("C1") == F.col("C2")).cast("int"))
    
def get_hand_is_suit(dataframe: DataFrame) -> DataFrame:
    """
    Checks if the hand Cards (1 and 2) have the same suit.
    """
    return dataframe.withColumn("hand_is_suited", (F.col("S1") == F.col("S2")).cast("int"))

def get_hand_is_connected(dataframe: DataFrame) -> DataFrame:
    """
    Checks if the hand Cards (1 and 2) have the same suit.
    Idea: abs(C1-C2) == 1 implies the cards are connected. Ace and King are also connected so therefore
    an additional check is done to see if abs(C1-C2) == 12.
    """
    rank_difference = F.abs(F.col("C1") - F.col("C2"))
    return dataframe.withColumn("hand_is_connected", ((rank_difference == 1) | (rank_difference == 12)).cast("int"))

def get_highest_hand_card(dataframe: DataFrame) -> DataFrame:
    """
    Gets the highest card in the players hand, it turns the Ace (1) into 14 for this.
    """
    C1_value = F.when(F.col("C1") == 1, 14).otherwise(F.col("C1"))
    C2_value = F.when(F.col("C2") == 1, 14).otherwise(F.col("C2"))
    return dataframe.withColumn("highest_hand_card", F.greatest(C1_value, C2_value))

def get_hand_rank_difference(dataframe: DataFrame) -> DataFrame:
    """
    Gets the difference value between the players hands rank cards
    """
    return dataframe.withColumn("hand_rank_diff", F.abs(F.col("C1") - F.col("C2")))

def get_win_equity(dataframe: DataFrame) -> DataFrame:
    """
    Gets the equity for each row's hand given by the Treys library
    """
    dataframe = dataframe.withColumn("equity", calculate_equity(F.col("S1"), F.col("C1"),
                                                                F.col("S2"), F.col("C2"),
                                                                F.col("S3"), F.col("C3"),
                                                                F.col("S4"), F.col("C4"),
                                                                F.col("S5"), F.col("C5")))
    return dataframe

def get_current_hand_rank_value(dataframe: DataFrame) -> DataFrame:
    """
    Gets the ranking/rating of your current hand (2 cards and 3 cards on the board) given by the Treys library
    """
    dataframe = dataframe.withColumn("current_hand_rank_value", calculate_treys_hand_score(F.col("S1"), F.col("C1"),
                                                                                           F.col("S2"), F.col("C2"),
                                                                                           F.col("S3"), F.col("C3"),
                                                                                           F.col("S4"), F.col("C4"),
                                                                                           F.col("S5"), F.col("C5")))
    return dataframe

def get_current_hand_class_value(dataframe: DataFrame) -> DataFrame:
    """
    Gets the class of your current hand (2 cards and 3 cards on the board) given by the Treys library
    This should be preferred over the class of the original dataset, to stay consistent to the Treys library
    Additionally, make sure to have the 'current_hand_rank_value' column already setup
    """
    dataframe = dataframe.withColumn("current_hand_class_value", calculate_treys_hand_class_score(F.col("current_hand_rank_value")))
    return dataframe

def get_is_straight_hand(dataframe: DataFrame) -> DataFrame:
    """
    If the current hand is a straight, then it will be marked as 1. Otherwise 0
    For this the current_hand_class_value column has to be present, and is used.
    In the Treys library, 1 is a straight flush and 5 is a straight and 0 is a royal flush (a straight)
    """
    return dataframe.withColumn("is_straight_hand", ((F.col("current_hand_class_value") == 0) | (F.col("current_hand_class_value") == 1) | (F.col("current_hand_class_value") == 5)).cast("int"))

def get_is_flush_hand(dataframe: DataFrame) -> DataFrame:
    """
    If the current hand is a flush, then it will be marked as 1. Otherwise 0
    For this the current_hand_class_value column has to be present, and is used.
    In the Treys library, 1 is a straight flush and 0 is a royal flush and 4 is a flush
    """
    return dataframe.withColumn("is_flush_hand", ((F.col("current_hand_class_value") == 1) | (F.col("current_hand_class_value") == 0) | (F.col("current_hand_class_value") == 4)).cast("int"))

def get_is_flush_draw(dataframe: DataFrame) -> DataFrame:
    """
    If the current board status is a flush draw (4 cards out of 5 have the same suit)
    """
    all_suits = F.array("S1", "S2", "S3", "S4", "S5")
    is_straight_draw = F.lit(False)
    # Suit of cards are 1-2-3-4 (for hearts, spades diamonds, clubs)
    for suit_id in [1, 2, 3, 4]:
        count_of_suit_amount = F.size(F.filter(all_suits, lambda x: x == suit_id))
        is_straight_draw = is_straight_draw | (count_of_suit_amount == 4)
    return dataframe.withColumn("is_flush_draw", is_straight_draw.cast("int"))

def get_is_straight_draw(dataframe: DataFrame) -> DataFrame:
    """
    If the current board status is a straight draw (4 connected cards out of 5)
    This occurs in two cases, 5-6-7-8 (where you still need a 4 or 9)
    and, 5-6-8-9 and you still need a 7 to win
    For this we also treat the ace as 1 but also as 14 (high vs low straight)
    """
    # Create a deduplicated ranks array, that is sorted.
    dataframe = dataframe.withColumn("ranks", F.array_distinct(F.array(F.col("C1"), F.col("C2"), F.col("C3"), F.col("C4"), F.col("C5"))))
    dataframe = dataframe.withColumn("ranks", F.when(F.array_contains("ranks", 1), F.array_union(F.col("ranks"), F.array(F.lit(14)))).otherwise(F.col("ranks")))
    dataframe = dataframe.withColumn("ranks", F.sort_array(F.col("ranks"))) # Sort the array, for easier detection

    condition_a = (F.try_element_at(F.col("ranks"), F.lit(4)) - F.try_element_at(F.col("ranks"), F.lit(1))) <= 4
    condition_b = (F.try_element_at(F.col("ranks"), F.lit(5)) - F.try_element_at(F.col("ranks"), F.lit(2))) <= 4
    condition_c = (F.try_element_at(F.col("ranks"), F.lit(6)) - F.try_element_at(F.col("ranks"), F.lit(3))) <= 4

    combined_check = (condition_a | condition_b | condition_c)
    return dataframe.withColumn("is_straight_draw", F.coalesce(combined_check, F.lit(False)).cast("int")).drop("ranks")

def get_board_is_monotone(dataframe: DataFrame) -> DataFrame:
    """
    Checks if the board (cards 3, 4, 5) all have the same suits.
    """
    return dataframe.withColumn("board_is_monotone", ((F.col("S3") == F.col("S4")) & (F.col("S4") == F.col("S5"))).cast("int"))

def get_board_is_paired(dataframe: DataFrame) -> DataFrame:
    """
    Checks if the board (cards 3, 4, 5) contains a pair.
    """
    return dataframe.withColumn("board_is_paired", ((F.col("C3") == F.col("C4")) | (F.col("C4") == F.col("C5")) | (F.col("C3") == F.col("C5"))).cast("int"))