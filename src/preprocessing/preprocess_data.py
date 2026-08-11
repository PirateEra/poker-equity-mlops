from configs.configs import MYSQLConfig, PathsConfig
from src.data_loading.load_data import load_raw_data
from pyspark.sql import DataFrame
from src.features.stateless_features import (
    get_win_equity
)
from src.utils.utils import get_free_cores

def preprocess_poker_data(dataframe : DataFrame) -> DataFrame:
    """
    This function, creates the equity (new label) column and drops the old label (class)
    """
    dataframe = dataframe.repartition(get_free_cores() * 4)
    dataframe = dataframe.drop("CLASS") # Drop the old column for the class label since we changed the challenge
    dataframe = get_win_equity(dataframe)
    dataframe.persist() # Save to remove, added to see spark progress in terminal.
    spark = dataframe.sparkSession
    spark.sparkContext.setCheckpointDir(PathsConfig.intermediate_data_path) # Save a checkpoint of this, since it takes long to do
    dataframe = dataframe.checkpoint(eager=True)
    return dataframe

def save_preprocessed_data(dataframe : DataFrame) -> DataFrame:
    dataframe.write.jdbc(MYSQLConfig.url, MYSQLConfig.preprocessed_table, mode="overwrite", properties=MYSQLConfig.properties)

if __name__ == "__main__":
    print("------------------\nFetching the raw dataset from MySQL...\n------------------")
    spark_dataframe = load_raw_data()
    print("------------------\Preprocessing the raw dataset...\n------------------")
    spark_dataframe = preprocess_poker_data(spark_dataframe)
    print("------------------\nWriting back to MySQL...\n------------------")
    save_preprocessed_data(spark_dataframe)
    print("------------------\nDone!...\n------------------")

