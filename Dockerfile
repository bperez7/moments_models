FROM python:3.6.14-buster
COPY ./
RUN pip install -r requirements.txt

