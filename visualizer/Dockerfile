FROM python:3.8-slim-buster

WORKDIR /M36_Synthetic_Samples
RUN apt-get update && apt-get install -y coreutils findutils wget

COPY requirements.txt /M36_Synthetic_Samples/requirements.txt
RUN pip install -r requirements.txt

COPY vis_graph_salad.py /M36_Synthetic_Samples/
COPY gen_single_doc_graphs.py /M36_Synthetic_Samples/
COPY data /M36_Synthetic_Samples/data