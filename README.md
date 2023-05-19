# Important Notice:

Still in active development!
More info to come soon (expected Q2 2023).

# About

CVRmap is an opensource (license AGPLv3) software to compute maps of Cerebro-Vascular Reactivity (CVR).

The software is compatible with the Brain Imagning Data Structure standard for applications.

The paper describing the toolbox will be pulished soon, together with more documentation about the pipeline.

# Installation

As of now, the only supported installation procedure is to clone this git repo and make `run.py` executable.

# Usage

To run CVRmap, you must first have data to crunch. The data are supposed to a BIDS (https://bids-specification.readthedocs.io/en/stable/) dataset. For each subject, you must have a T1w image, a BOLD image and a physiological file contained the breathing data (CO2) recorded during the fMRI scan. For instance, for `sub-01`, the data should look like this:

```
sub-01/anat/sub-01_T1w.nii.gz
sub-01/anat/sub-01_T1w.json
sub-01/func/sub-01_task-gas_bold.nii.gz
sub-01/func/sub-01_task-gas_bold.json
sub-01/func/sub-01_task-gas_physio.tsv.gz
sub-01/func/sub-01_task-gas_physio.json
```

In this example, the taskname BIDS entity is `gas`. If yours differs, that's not a problem and you'll be able to run CVRmap provided you add the option `--taskname your_task_name` when launching the software.

Note that the `sub-01/func/sub-01_task-gas_physio.json` file must containt a `SamplingFrequency` field as well as a `co2` field to specify the units of measurement of CO2 levels. A example of `sub-01/func/sub-01_task-gas_physio.json` would be:

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

In this example, the `sub-01/func/sub-01_task-gas_physio.tsv.gz` must have only one colunm, giving the CO2 readings at a sampling frequency of 100 Hz, in the units of `%` (which means 'percentage of co2 concentration'). If the CO2 readings are in mmHg (which is also a common units), the the "Units" field must be "mmHg". CVRmap converts percentages to mmHg automatically. Finally, the total duration of the CO2 reccording must not necessarily match the duration of the BOLD acquisition: if needed, CVRmap trims or uses a baseline interpolation automatically.



