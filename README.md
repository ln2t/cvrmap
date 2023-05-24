# Important Notice:

Still in active development!
More info to come soon (expected Q2 2023).

# About

CVRmap is an opensource (license AGPLv3) software to compute maps of Cerebro-Vascular Reactivity (CVR).

The software is compatible with the Brain Imagning Data Structure standard for applications.

The paper describing the toolbox will be pulished soon, together with more documentation about the pipeline.

# Installations

# Option 1: python environment (using git)

The most basic way to install `cvrmap` is to clone this git repo, make `cvrmap` executable and add it to your path. Make sure you have all python3 dependencies installed:

```
pip install -r requirements.txt
```

In that case, `cvrmap` can be launched by typing `cvrmap` in a terminal, e.g.

```
cvrmap --version
```

# Option 2: python environment (using pip) - work in progress

`cvrmap` is also distributed as a `pip` package. It can be installed using

```
pip install cvrmap
```

pip doesn't add the executable to your path so you have to do it manually. This is mostly useful if you want to use some internal tools of the software without using the full software.

# Option 3: docker (recommended option!)

The easiest way to install `cvrmap` is to use docker:

```
docker pull arovai/cvrmap:VERSION
```

where `VERSION` is the version you wish to use. Check out the docker hub page (https://hub.docker.com/repository/docker/arovai/cvrmap/general) to find more info on the available images.

To launch `cvrmap`, you can use for instance

```
docker run arovai/cvrmap:VERSION --version
```

This will output the version of `cvrmap`. You can also type

```
docker run arovai/cvrmap:VERSION --help
```

for help.

Finally, to pass some data to the docker image, you can use something like:

```
docker run -v /path/to/your/bids/folder:/rawdata -v /path/to/your/derivatives:/derivatives arovai/cvrmap:VERSION /rawdata /derivatives participant
```

For more information about the command line options, see the **Usage** section.

# Option 4: Singularity (good for HPC) - work in progress

You can also build a Singularity image file using

```
singularity build arovai.cvrmap.VERSION.sif docker://arovai/cvrmap:VERSION
```

# Usage

To run CVRmap, you must first have data to crunch. The data are supposed to a BIDS (https://bids-specification.readthedocs.io/en/stable/) dataset. For each subject, you must have a T1w image, a BOLD image and a physiological file containing the breathing data (CO2) recorded during the fMRI scan. For instance, for `sub-01`, the data should look like this:

```
sub-01/anat/sub-01_T1w.nii.gz
sub-01/anat/sub-01_T1w.json
sub-01/func/sub-01_task-gas_bold.nii.gz
sub-01/func/sub-01_task-gas_bold.json
sub-01/func/sub-01_task-gas_physio.tsv.gz
sub-01/func/sub-01_task-gas_physio.json
```

In this example, the taskname BIDS entity is `gas`. If yours differs, that's not a problem and you'll be able to run CVRmap provided you add the option `--taskname your_task_name` when launching the software.

Note that the `sub-01/func/sub-01_task-gas_physio.json` file must contain a `SamplingFrequency` field as well as a `co2` field to specify the units of measurement of CO2 levels. An example of `sub-01/func/sub-01_task-gas_physio.json` would be:

```
{
    "SamplingFrequency": 100,
    "StartTime": 0,
    "Columns": [
        "co2"
    ],
    "co2": {
        "Units": "%"
    }
}
```

In this example, the `sub-01/func/sub-01_task-gas_physio.tsv.gz` must have only one colunm, giving the CO2 readings at a sampling frequency of 100 Hz, in the units of `%` (which means 'percentage of co2 concentration'). If the CO2 readings are in mmHg (which is a common unit), the "Units" field must be "mmHg". CVRmap converts percentages to mmHg automatically. Finally, the total duration of the CO2 recording must not necessarily match the duration of the BOLD acquisition: depending on the case, CVRmap trims or uses a baseline extrapolation automatically.

The rawdata must also have been processed using fMRIPrep (https://fmriprep.org/en/stable). A minimalistic fMRIPrep call compatible with CVRmap is:

```
fmriprep /path/to/bids_dir /path/to/derivatives/fmriprep participant --fs-license-file /path/to/fslicense --use-aroma --output-spaces MNI152NLin6Asym
```

Note that this includes the AROMA flag of fMRIPrep. If more output spaces are required for the computation of CVRmaps, say in original, participant's space, then the fMRIPrep call must be adapted by modifying the `--spaces` options accordingly. For instance:

```
fmriprep /path/to/bids_dir /path/to/derivatives/fmriprep participant --fs-license-file /path/to/fslicense --use-aroma --output-spaces MNI152NLin6Asym T1w
```

We are now good to go and launch CVRmap with

```
cvrmap /path/to/bids_dir /path/to/derivatives/cvrmap participant --fmriprep_dir /path/to/derivatives/fmriprep
```

Notes:
- the exact `cvrmap` command might depend on the installation option you choose.
- the `--fmriprep_dir` option can be ommitted if the fMRIPrep derivatives are located in `/path/to/bids_dir/derivatives/fmriprep`.
- if the BOLD taskname is not `gas`, you must add `--taskname your_task_name`.
- if you want the outputs in another space, and if this space was included in the fMRIPrep call, you must add `--space your_custom_space`. The default space is `MNI152NLin6Asym`.
- if you want to use the ICA-AROMA classification for signal confounds, then add `--use-aroma` and CVRmap will fetch the non-aggressively denoised BOLD series as procuded by fMRIPrep. Otherwise, when the flag is omitted, CVRmap will perform non-aggressive denoising itself using a refinement of the ICA-AROMA classification of noise sources (see paper for more details).
- more info and options can be found when asking politely for help with `cvrmap --help`.

CVRmap will run for about 2 hours on recent computers. The results are stored in `/path/to/derivatives/cvrmap` following BIDS standard for derivatives. The outputs will typically look as follows:

```
sub-01/sub-01_desc-etco2_timecourse.tsv.gz
sub-01/sub-01_desc-etco2_timecourse.json
sub-01/sub-01_space-MNI152NLin6Asym_delay.nii.gz
sub-01/sub-01_space-MNI152NLin6Asym_delay.json
sub-01/sub-01_space-MNI152NLin6Asym_cvr.nii.gz
sub-01/sub-01_space-MNI152NLin6Asym_cvr.json
sub-01/sub-01_space-MNI152NLin6Asym_report.html
```

The `etco2` file contains the end-tidal timecourse extracted from the original CO2 readings. The delay map contains the computed time delays (or time lags) in seconds, and normalized to the global signal delay. The main map of interest is of course the CVR map! For a quick analysis, an html report is also included.

# Bugs or questions

Should you encounter any bug, weird behavior or if you have questions, do not hesitate to open an issue and we'll happily try to answer!

# Complementary information

We recommend to use a python3 virtual environment to install CVRmap. We can do this using e.g. `conda`:

```
# create empty environment named "cvrmap"
conda create -n cvrmap python
# activate
conda activate -n cvrmap
# install the packages
pip install -r requirements.txt
```
