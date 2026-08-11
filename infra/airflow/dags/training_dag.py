from airflow import DAG
from docker.types import Mount
from airflow.operators.bash import BashOperator
from airflow.providers.docker.operators.docker import DockerOperator
import os

from datetime import timedelta

HOST_DATA_PATH = os.path.join(os.getcwd(), 'data')

data_mount = Mount(
    source=HOST_DATA_PATH,
    target="/poker/data",
    type="bind"
)

MYSQL_ENVIRONMENT = {
    "MYSQL_DATABASE": os.environ["MYSQL_DATABASE"],
    "MYSQL_USER": os.environ["MYSQL_USER"],
    "MYSQL_PASSWORD": os.environ["MYSQL_PASSWORD"],
}

default_args = {
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

dag = DAG("poker_equity_training", default_args=default_args, schedule="59 23 * * *")


preprocessing_task = DockerOperator(
    docker_url="unix://var/run/docker.sock",
    command="python src/training/main_training.py --preprocess",
    image="poker:latest",
    network_mode="poker_equity_network",
    environment=MYSQL_ENVIRONMENT,
    task_id="preprocessing",
    dag=dag,
)

feature_engineering_task = DockerOperator(
    docker_url="unix://var/run/docker.sock",
    command="python src/training/main_training.py --feature_engineering",
    image="poker:latest",
    network_mode="poker_equity_network",
    environment=MYSQL_ENVIRONMENT,
    task_id="feature_engineering",
    mounts=[data_mount],
    dag=dag,
)

model_training_task = DockerOperator(
    docker_url="unix://var/run/docker.sock",
    command="python src/training/main_training.py --training",
    image="poker:latest",
    network_mode="poker_equity_network",
    task_id="training",
    mounts=[data_mount],
    dag=dag,
)

reload_model_task = BashOperator(
    task_id="reload_model",
    bash_command="curl -X POST http://app:5001/reload",
    dag=dag
)

(
    preprocessing_task
    >> feature_engineering_task
    >> model_training_task
    >> reload_model_task
)
