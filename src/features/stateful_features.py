from pyspark.sql.window import Window
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from src.utils.utils import get_free_cores

def get_class_relative_rank_z_score(dataframe: DataFrame) -> DataFrame:
    """
    Calculates the Z-score of the hand rank relative to other hands of the same class.
    It assumes the current_hand_class_value and current_hand_rank_value columns both exist.
    """
    class_windows = Window.partitionBy("current_hand_class_value")
    class_mean = F.mean("current_hand_rank_value").over(class_windows)
    class_std = F.stddev("current_hand_rank_value").over(class_windows)
    dataframe = dataframe.withColumn("relative_rank_z", (F.col("current_hand_rank_value") - class_mean) / (class_std + 1e-9))
    return dataframe

def get_class_relative_cdf(dataframe: DataFrame) -> DataFrame:
    """
    Calculates the cdf of the hand rank relative to its class. (0-1.0)
    """
    ordered_class_windows = Window.partitionBy("current_hand_class_value").orderBy("current_hand_rank_value")
    dataframe = dataframe.withColumn("relative_rank_cdf", F.cume_dist().over(ordered_class_windows))
    return dataframe

def get_global_rank_stength(dataframe: DataFrame) -> DataFrame:
    """
    Calculates the strength of a rows hand relative to the dataset
    """
    dataframe = dataframe.repartition(get_free_cores() * 4) # There is a warning implying Moving all data to a single partition. However this ensures good speed. Ignore the warning
    ordered_class_windows = Window.orderBy(F.desc("current_hand_rank_value"))
    dataframe = dataframe.withColumn("global_rank_strenght", F.percent_rank().over(ordered_class_windows))
    return dataframe