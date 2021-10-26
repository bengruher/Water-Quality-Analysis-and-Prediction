FROM ubuntu:latest

RUN apt-get update -y
RUN apt-get -y install python3-pip

COPY ./WaterQuality/* /opt/ml/code/WaterQuality/
WORKDIR /opt/ml/code/WaterQuality

RUN pip3 install -r requirements.txt

ENV PATH="/opt/ml/code/WaterQuality:${PATH}"