from configs.configs import MYSQLConfig, RunConfig, PathsConfig
from src.utils.utils import get_spark_mysql_session, get_spark_session
from pyspark.sql import DataFrame

def load_raw_data() -> DataFrame:
    spark = get_spark_mysql_session()
    spark_dataframe = spark.read.jdbc(MYSQLConfig.url, MYSQLConfig.raw_table, properties=MYSQLConfig.properties)
    return spark_dataframe

def load_preprocessed_data() -> DataFrame:
    spark = get_spark_mysql_session()
    spark_dataframe = spark.read.jdbc(MYSQLConfig.url, MYSQLConfig.preprocessed_table, properties=MYSQLConfig.properties)
    spark_dataframe = spark_dataframe.sample(RunConfig.sample_rate)
    return spark_dataframe

def load_feature_data() -> DataFrame:
    spark = get_spark_session()
    spark_dataframe = spark.read.parquet(PathsConfig.features_data_path)
    return spark_dataframe