from ucimlrepo import fetch_ucirepo 
from configs.configs import MYSQLConfig
import pandas as pd
import pyspark.pandas as ps
from src.utils.utils import get_spark_mysql_session

def upload_raw_poker_data():
    """
    This function, fetches the raw poker data, and uploads it to the mysql database using the tablename provided in the config
    """
    print("------------------\nSetting up spark session...\n------------------")
    spark = get_spark_mysql_session()
    print("------------------\nFetching the dataset...\n------------------")
    poker_hand = fetch_ucirepo(id=158) 
    dataframe = pd.concat([poker_hand.data.features, poker_hand.data.targets], axis=1)
    spark_dataframe = ps.from_pandas(dataframe).to_spark()
    print(f"------------------\nWriting {spark_dataframe.count()} rows to MySQL table {MYSQLConfig.raw_table}...\n------------------\n")
    spark_dataframe.write.jdbc(MYSQLConfig.url, MYSQLConfig.raw_table, mode="overwrite", properties=MYSQLConfig.properties)
    print(f"------------------\nDone!\n------------------\n")
    spark_dataframe = spark.read.jdbc(MYSQLConfig.url, MYSQLConfig.raw_table, properties=MYSQLConfig.properties)
    print(f"------------------\nTable {MYSQLConfig.raw_table} contains: {spark_dataframe.count()} rows\n------------------\n")
    print(f"------------------\nFirst row in MySQL table {MYSQLConfig.raw_table} contains: {spark_dataframe.first()}\n------------------\n")
    

if __name__ == "__main__":
    upload_raw_poker_data()