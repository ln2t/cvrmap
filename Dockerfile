FROM ubuntu:20.04

# Update packages and install necessary dependencies
RUN apt-get update && \
    apt-get install -y git
    
RUN apt-get install -y python3 pip

RUN git clone https://github.com/ln2t/cvrmap.git && \
    mv cvrmap /opt/cvrmap

RUN pip install -r /opt/cvrmap/requirements.txt

# Set the default command to run when the container starts
CMD ["/opt/cvrmap/src/cvrmap/cvrmap -h"]

