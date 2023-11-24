[![codecov](https://codecov.io/gh/ln2t/cvrmap/graph/badge.svg?token=VGQPJX5078)](https://codecov.io/gh/ln2t/cvrmap) ![License](https://img.shields.io/github/license/ln2t/cvrmap)

# About

CVRmap is an opensource (license AGPLv3) software to compute maps of Cerebro-Vascular Reactivity (CVR).

The software is compatible with the Brain Imagning Data Structure (BIDS) standard for applications.

The paper describing the toolbox will be pulished (hopefully!) soon.

A normative dataset can be downloaded from openneuro: [ds004604](https://openneuro.org/datasets/ds004604)

CVRmap is also distributed as a package, the full API documentation being [here](https://ln2t.github.io/cvrmap)

# Installation

# Note about dependencies

All dependencies are specified in `requirements.txt`.

# Option 1: docker (recommended option!)

The easiest way to install `cvrmap` is to use docker:

```
docker pull arovai/cvrmap:VERSION
```

where of course `VERSION` must be replaced by any other available version (e.g. 2.0.0).
Check out the [`cvrmap` docker hub page](https://hub.docker.com/repository/docker/arovai/cvrmap/general) to find more info on the available images.

To test your installation, you can use for instance

```
docker run arovai/cvrmap:VERSION --version
```

This will output the version and exit. You can also type

```
docker run arovai/cvrmap:VERSION --help
```

for help.

For the reader not familiar with docker, you can pass some data to the docker image with

```
docker run -v /path/to/your/bids/folder:/rawdata -v /path/to/your/derivatives:/derivatives arovai/cvrmap:VERSION /rawdata /derivatives/cvrmap participant
```

For more information about the command line options, see the **Usage** section.

# Option 2: Singularity (good for HPC)

You can also build a Singularity image file using

```
singularity build arovai.cvrmap.VERSION.sif docker://arovai/cvrmap:VERSION
```

You can for instance run this command inside your home directory, and then you'll have a singularity image file named `arovai.cvrmap.VERSION.sif` in there. To run it, still in the folder where the image is located, run

```
singularity run -B /path/to/your/bids/folder:/rawdata -B /path/to/your/derivatives:/derivatives arovai.cvrmap.VERSION.sif /rawdata /derivatives/cvrmap participant
```

Make sure that the folder `/path/to/your/derivatives` exists before launching this command.

Of course, you are free to place the image where ever suits you; you'll simply have to adapt the path when calling `singularity`.

# Option 3: python environment (using pip)

`cvrmap` is also available on [pypi](https://pypi.org/manage/project/cvrmap/releases). We strongly recommend that you install it in a virtual environement.
First, install `virtualenv`:

```
pip install virtualenv
```

Then create a virtual env. To deal with potential conflicts in versions of the required packages withing `cvrmap`, we recommend you create one environment for each `cvrmap` version:

```
export VERSION="2.0.0"
virtualenv cvrmap-env-$VERSION
```

Activate the environment and install `cvrmap`:
```
source cvrmap-env-$VERSION/bin/activate && pip install cvrmap==$VERSION
```

Warning: make sure you are using Python version 3.8 (or more recent)!

This will add a command in your `PATH` so that you can directly launch `cvrmap`:

```
cvrmap -h
```

Note that the docker image is essentially build using this procedure, as you can see in the `Dockerfile` located in the `docker` folder of this repo.

# Usage

To run CVRmap, you must first have data to crunch. If you don't have data, and you want to test CVRmap, you can download the publicly available dataset on openneuro [ds004604](https://openneuro.org/datasets/ds004604) which include compatible rawdata, fmriprep derivatives, as well as `cvrmap` (v1.0) outputs.
If you have your own data that you want to analyze with CVRmap, make sure to observe the following:
first of all, the data are supposed to a [BIDS](https://bids-specification.readthedocs.io/en/stable/) dataset. For each subject, you must have a T1w image, a BOLD image and a physiological file containing the breathing data (CO2) recorded during the fMRI scan. For instance, for `sub-01`, the data should look like this:

```
sub-01/anat/sub-01_T1w.nii.gz
sub-01/anat/sub-01_T1w.json
sub-01/func/sub-01_task-gas_bold.nii.gz
sub-01/func/sub-01_task-gas_bold.json
sub-01/func/sub-01_task-gas_physio.tsv.gz
sub-01/func/sub-01_task-gas_physio.json
```

In this example, the taskname BIDS entity is `gas`. If yours differs, that's not a problem, and you'll be able to run `cvrmap` provided you add the option `--taskname your_task_name` when launching the software.

Note that the `sub-01/func/sub-01_task-gas_physio.json` file must contain a `SamplingFrequency` field as well as a `co2` field to specify the units of measurement of CO2 levels. An example of `sub-01/func/sub-01_task-gas_physio.json` would be:

```
{
    "SamplingFrequency": 100,
    "StartTime": 0,
    "Columns": [
        "co2"
    ],
    "co2": {
        "Units": "mmHg"
    }
}
```

In this example, the `sub-01/func/sub-01_task-gas_physio.tsv.gz` must have only one column, giving the CO2 readings at a sampling frequency of 100 Hz, starting at time 0 with respect to the first valid fMRI volume, in the units of mmHg.
Note though that the `StartTime` field is not used at all by `cvrmap`, as it explores various time lags by itself.
If the CO2 readings are in percentage of co2 concentration (which is also often used), the "Units" field must be "%", and in that case `cvrmap` will convert percentages to mmHg automatically. Finally, the total duration of the CO2 recording must not necessarily match the duration of the BOLD acquisition: depending on the case, CVRmap trims or uses a baseline extrapolation automatically.

The rawdata must also have been processed using [fMRIPrep](https://fmriprep.org/en/stable). A minimalistic fMRIPrep call compatible with CVRmap is:

```
fmriprep /path/to/bids_dir /path/to/derivatives/fmriprep participant --fs-license-file /path/to/fslicense --use-aroma
```

*Warning* the versions of fMRIPrep above 23.1.0 don't support the `--use-aroma` option anymore. For this reason, we *must* use older versions. We will provide a patch to `cvrmap` to handle newer versions of fMRIPrep in the future.

We are now good to go and launch CVRmap with

```
cvrmap /path/to/bids_dir /path/to/derivatives/cvrmap participant --fmriprep_dir /path/to/derivatives/fmriprep
```

Notes:
- the exact `cvrmap` command might depend on the installation option you choose (see above, options 1, 2 and 3)
- the `--fmriprep_dir` option can be omitted if the fMRIPrep derivatives are located in `/path/to/bids_dir/derivatives/fmriprep`.
- if the BOLD taskname is not `gas`, you must add `--taskname your_task_name`.
- if you want the outputs in another space, and if this space was included in the fMRIPrep call, you must add `--space your_custom_space`. The default space is `MNI152NLin2009cAsym`, which is also the default space of fMRIPrep.
- if you want to use the ICA-AROMA classification for signal confounds, then add `--use-aroma`. Otherwise, when the flag is omitted, CVRmap will perform non-aggressive denoising itself using a refinement of the ICA-AROMA classification of noise sources (see paper for more details).
- more info and options can be found when asking politely for help with `cvrmap --help`.

`cvrmap` will run for about 3 hours per participant on recent computers. The results are stored in `/path/to/derivatives/cvrmap` following BIDS standard for derivatives.  More specifically, the outputs are as follows:

```
sub-01/extras/sub-01_desc-etco2_timecourse.tsv.gz
sub-01/extras/sub-01_desc-etco2_timecourse.json
sub-01/extras/sub-01_space-MNI152NLin2009cAsym_denoised.nii.gz
sub-01/extras/sub-01_space-MNI152NLin2009cAsym_denoised.json
sub-01/figures/sub-01_boldmean.svg
sub-01/figures/sub-01_breathing.svg
sub-01/figures/sub-001_denoising.html
sub-01/figures/sub-001_summary.html
sub-01/figures/sub-01_space-MNI152NLin2009cAsym_cvr.svg
sub-01/figures/sub-01_space-MNI152NLin2009cAsym_delay.svg
sub-01/sub-01_space-MNI152NLin2009cAsym_delay.nii.gz
sub-01/sub-01_space-MNI152NLin2009cAsym_delay.json
sub-01/sub-01_space-MNI152NLin2009cAsym_cvr.nii.gz
sub-01/sub-01_space-MNI152NLin2009cAsym_cvr.json
sub-01.html
```

The `extras` folder contains the `etco2` file with the end-tidal timecourse extracted from the original CO2 readings as well as the non-aggressively denoised BOLD series (the series are also high-pass filtered and smoothed at 5mm FWHM). Some pictures for the report are stored within `figures`. The delay map contains the computed time delays (or time lags) in seconds, and normalized to the global signal delay. The main map of interest is of course the CVR map! For a quick analysis, the html report is also included.

# Bugs or questions

Should you encounter any bug, weird behavior or if you have questions, do not hesitate to open an issue, and we'll happily try to answer!
