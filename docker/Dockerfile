FROM ubuntu:20.04

# Update and install basic packages
RUN apt-get update
RUN apt-get -y upgrade
RUN apt-get update --fix-missing && apt-get -y upgrade
RUN apt-get install -y git
RUN apt-get install -y python3 pip

# Install cvrmap from github
RUN cd /root
RUN cd /opt && git clone https://github.com/ln2t/cvrmap.git
RUN cd /opt/cvrmap && git checkout main
RUN pip install -r /opt/cvrmap/requirements.txt

# Set entrypoint to entrypoint script
ENTRYPOINT ["/opt/cvrmap/docker/entrypoint.sh"]
