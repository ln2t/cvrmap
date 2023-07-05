FROM ubuntu:20.04

# Update packages and install necessary dependencies
RUN apt-get update
RUN apt-get -y upgrade
RUN apt-get update --fix-missing && apt-get -y upgrade
RUN apt-get install -y git
RUN apt-get install -y python3 pip

RUN cd /opt && git clone https://github.com/ln2t/cvrmap.git
RUN cd /opt/cvrmap && git checkout highpassfiltering

RUN pip install -r /opt/cvrmap/requirements.txt

# Set entrypoint to entrypoint script
ENTRYPOINT ["/opt/cvrmap/docker/entrypoint.sh"]
