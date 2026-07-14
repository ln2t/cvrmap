from setuptools import setup, find_packages
import os

# Read version from cvrmap/__init__.py
version = {}
with open(os.path.join(os.path.dirname(__file__), 'cvrmap', '__init__.py')) as f:
    for line in f:
        if line.startswith('__version__'):
            exec(line, version)
            break

setup(
    name="cvrmap",
    version=version['__version__'],
    packages=find_packages(),
    package_data={
        'cvrmap': ['default_config.yaml', 'data/report_config.yaml', 'data/*.nii.gz'],
    },
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'cvrmap = cvrmap.cli:main',
        ],
    },
    install_requires=[
        # Core scientific computing
        'numpy>=1.20.0',
        'scipy>=1.7.0',
        'pandas>=1.3.0',
        
        # Neuroimaging
        'nibabel>=3.2.0',
        'nilearn>=0.8.0',
        
        # BIDS support
        'pybids>=0.15.0',
        
        # Signal processing and analysis
        'scikit-learn>=1.0.0',
        'peakutils>=1.3.0',
        'joblib>=1.0.0',  # For parallel processing
        
        # Visualization
        'matplotlib>=3.4.0',
        
        # Configuration and I/O
        'PyYAML>=5.4.0',
    ],
    author='CVRMap Development Team',
    description='A Python CLI application for cerebrovascular reactivity mapping using BIDS-compatible physiological and BOLD fMRI data.',
    long_description="""
    CVRMap is a comprehensive pipeline for processing physiological signals (CO2) and BOLD fMRI data 
    to generate cerebrovascular reactivity maps. The pipeline includes:
    
    - BIDS-compatible data handling
    - Physiological signal preprocessing with ETCO2 extraction
    - BOLD signal preprocessing with AROMA denoising
    - Cross-correlation analysis for delay mapping
    - Global signal analysis
    - Comprehensive output generation with visualizations
    
    The package follows BIDS derivatives standards for output organization.
    """,
    long_description_content_type='text/plain',
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    keywords='neuroimaging fmri bids cerebrovascular reactivity cvr',
    project_urls={
        'Source': 'https://github.com/yourusername/cvrmap',
        'Documentation': 'https://cvrmap.readthedocs.io/',
    },
)
