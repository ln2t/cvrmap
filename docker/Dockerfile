FROM ubuntu:20.04

# Update packages and install necessary dependencies
RUN apt-get update
RUN apt-get -y upgrade
RUN apt-get install -y git
RUN apt-get install -y python3 pip

RUN cd && cd && cd
RUN cd /opt && git clone https://github.com/ln2t/cvrmap.git
RUN cd /opt/cvrmap && git checkout v1.0.0.rc

RUN pip install -r /opt/cvrmap/requirements.txt

# Set entrypoint to entrypoint script
ENTRYPOINT ["/opt/cvrmap/docker/entrypoint.sh"]
