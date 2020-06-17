FROM pytorch/pytorch:latest
WORKDIR /aida-utexas
RUN apk update && apk add --no-cache bash coreutils findutils wget openjdk8-jre
RUN wget -q http://mirrors.advancedhosters.com/apache/jena/binaries/apache-jena-3.14.0.tar.gz -O - | tar -xz -C /
COPY requirements.txt /aida-utexas/requirements.txt
RUN pip install -r requirements.txt
COPY aida_utexas /aida-utexas/aida_utexas
COPY pipeline /aida-utexas/pipeline
COPY neural_pipeline /aida-utexas/neural_pipeline
COPY scripts /aida-utexas/scripts
COPY duplicate_kb_id_mapping.json /aida-utexas/duplicate_kb_id_mapping.json
ENV PATH /apache-jena-3.14.0/bin:$PATH
