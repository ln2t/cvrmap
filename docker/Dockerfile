FROM ubuntu:20.04

# Update packages and install necessary dependencies
RUN apt-get update && apt-get upgrade && apt-get install -y git
    
RUN apt-get install -y python3 pip

RUN git clone https://github.com/ln2t/cvrmap.git && \
    mv cvrmap /opt/cvrmap

RUN pip install -r /opt/cvrmap/requirements.txt

RUN cd /opt && rm -rf cvrmap && git clone https://github.com/ln2t/cvrmap.git && cd cvrmap && git pull

# Set entrypoint to entrypoint script
ENTRYPOINT ["/opt/cvrmap/docker/entrypoint.sh"]

