FROM python:3.13-slim-bullseye

RUN apt-get update
RUN apt-get install openjdk-17-jdk --yes

WORKDIR /poker

COPY requirements.txt .
RUN pip install -r requirements.txt

ENV PYTHONPATH=/poker/

COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/
COPY data/ data/
COPY tests/ tests/
COPY Makefile .
COPY .coveragerc .