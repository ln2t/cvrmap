FROM python:3.11

WORKDIR /code

COPY requirements.txt cvrmap/
RUN pip install --trusted-host files.pythonhosted.org --trusted-host pypi.org --trusted-host pypi.python.org -r cvrmap/requirements.txt

COPY cvrmap cvrmap/cvrmap
COPY setup.py cvrmap/
RUN cd /code/cvrmap && python setup.py sdist bdist_wheel
RUN python -m pip install /code/cvrmap/
COPY docker/entrypoint.sh .
ENTRYPOINT ["/code/entrypoint.sh"]
