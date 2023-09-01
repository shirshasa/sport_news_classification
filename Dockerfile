FROM continuumio/miniconda3

RUN mkdir build
COPY ./requirements.txt /build/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /build/requirements.txt


COPY ./api build/api
COPY ./text_classifier build/text_classifier
COPY ./models build/models
COPY ./setup.py build/
RUN cd /build && pip install . --use-feature=in-tree-build


ENV CHECKPOINT_PATH=/build/models/baseline.pkl


WORKDIR build/api

ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
