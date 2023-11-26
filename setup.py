from setuptools import setup, find_packages

# Function to read the contents of the requirements file
def read_requirements():
    with open('requirements.txt') as req:
        return req.read().splitlines()

setup(
    name='cvrmap',
    version='2.0.13',
    url='https://github.com/ln2t/cvrmap',
    author='Antonin Rovai',
    author_email='antonin.rovai@hubruxelles.be',
    description='CVRmap is an opensource software to compute maps of Cerebro-Vascular Reactivity',
    packages=find_packages(),
    install_requires=read_requirements(),
    entry_points={
        'console_scripts': [
            'cvrmap = cvrmap.cvrmap:main',
        ]},
    include_package_data=True
)
