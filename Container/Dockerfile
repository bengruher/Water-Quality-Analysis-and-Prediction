FROM mcr.microsoft.com/azureml/sklearn-0.24.1-ubuntu18.04-py37-cpu-inference:latest

ENV PATH = "/opt/ml/code:${PATH}"

COPY /WaterQuality /opt/ml/code
WORKDIR /opt/ml/code