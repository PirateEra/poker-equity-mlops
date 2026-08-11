from pyspark.sql import SparkSession
import os

def get_free_cores() -> int:
    total_cores = os.cpu_count() or 2
    cores = max(total_cores - 1, 1) # Make sure we always keep 1 core free
    return cores

def get_spark_mysql_session() -> SparkSession:
    """
    Returns a Sparksession configured for the MySQL driver.
    """
    spark = SparkSession.builder \
        .master(f"local[{get_free_cores()}]") \
        .appName("PySpark MySQL Connection") \
        .config("spark.jars.packages", "com.mysql:mysql-connector-j:9.5.0") \
        .config("spark.driver.extraJavaOptions", "-Divy.message.logger.level=error") \
        .config("spark.executor.extraJavaOptions", "-Divy.message.logger.level=error") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.sql.ansi.enabled", "false") \
        .getOrCreate()
    return spark

def get_spark_session() -> SparkSession:
    """
    Returns a general purpose spark session
    """
    spark = SparkSession.builder \
        .master(f"local[{get_free_cores()}]") \
        .appName("PySpark general session") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.sql.ansi.enabled", "false") \
        .getOrCreate()
    return spark