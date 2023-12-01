FROM ubuntu:20.04

# Update and install basic packages
RUN apt-get update
RUN apt-get -y upgrade
RUN apt-get update --fix-missing && apt-get -y upgrade
RUN apt-get install -y python3
RUN apt-get install -y pip
RUN pip install -U pip

# Install cvrmap from pypi
RUN pip install -U cvrmap==2.0.15

# Set entrypoint to entrypoint script
COPY entrypoint.sh /opt/cvrmap/entrypoint.sh
ENTRYPOINT ["/opt/cvrmap/entrypoint.sh"]
