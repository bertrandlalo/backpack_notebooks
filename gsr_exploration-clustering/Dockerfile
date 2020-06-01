FROM continuumio/miniconda:4.5.12

RUN mkdir /code
ADD environment.yaml /code/
RUN conda env create -f /code/environment.yaml

EXPOSE 8888
ADD entrypoint.sh /code/
ADD *.ipynb /code/
WORKDIR /code
ENTRYPOINT /code/entrypoint.sh
