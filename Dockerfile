FROM pytorch/pytorch:latest
WORKDIR /aida-utexas
RUN apt-get update && apt-get install -y coreutils findutils wget
#RUN apt-get update && apt-get install -y coreutils findutils wget openjdk-8-jre
#RUN wget -q https://mirrors.koehn.com/apache/jena/binaries/apache-jena-3.16.0.tar.gz -O - | tar -xz -C /
#ENV PATH /apache-jena-3.16.0/bin:$PATH
COPY requirements.txt /aida-utexas/requirements.txt
RUN pip install -r requirements.txt
COPY aida_utexas /aida-utexas/aida_utexas
COPY pipeline /aida-utexas/pipeline
COPY scripts /aida-utexas/scripts
COPY resources /aida-utexas/resources

# For visualizer
COPY visualizer /aida_utexas/visualizer
RUN pip install -r /aida_utexas/visualizer/requirements.txt

ENTRYPOINT ["/aida-utexas/scripts/run_simple.sh"]
