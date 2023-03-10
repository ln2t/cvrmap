#!/usr/bin/env python3
"""BIDS app to compute Cerebrovascular Reactivity maps
This app complies with the Brain Imaging Data Structure (BIDS) App
standard. It assumes the dataset is organised according to BIDS.
Intended to run in command line with arguments as described in the help
(callable with -h)

Author: Antonin Rovai

Created: May 2022
"""

# todo: loop over all spaces automatically
# todo: option to override or keep existing stuff
# todo: run from spm preproc bold series (requires the inclusion of motion parameters in the analysis...)
# todo: smooth delaymaps
# todo: group analysis
# todo: check if fmriprep as run for each subject, with appropriate spaces, and with aroma
# todo: read from config file:
#  - sloppy flag
#  - threshold for vessel density
#  - time span for delay map computation
# todo: register results to patient space; include PET image in report
# todo: make histogram of delays
# todo: normalize delay shift by computing delay of ROI of reference
# todo: ROI delay analysis
# todo: signal of grey matter compared to etCO2, recording of delay, include this in the report
# todo: write sidecar json files
# todo: read CO2 col header in tsv

# imports
import argparse  # to deal with all the app arguments
import os  # to interact with dirs (join apths etc)
from os.path import join  # to build input/tmp/output paths
import subprocess  # to call bin outside of python
from pathlib import Path  # to create dirs
import core  # custom utilities from the package
from bids import BIDSLayout as bidslayout  # to handle BIDS data
from core.utils import *  # custom utilities from the package
from datetime import datetime  # for logging purposes
import pandas as pd  # to deal with .tsv files
import sys  # to exit execution
import numpy as np
import glob # to list files in folders

# modules
def run(command, env={}):
    """Execute command as in a terminal

    Also prints any output of the command into the python shell
    Inputs:
        command: string
            the command (including arguments and options) to be executed
        env: to add stuff in the environment before running the command

    Returns:
        nothing
    """

    # Update env
    merged_env = os.environ
    merged_env.update(env)

    # Run command
    process = subprocess.Popen(command, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, shell=True,
                               env=merged_env)

    # Read whatever is printed by the command and print it into the
    # python console
    while True:
        line = process.stdout.readline()
        line = str(line, 'utf-8')[:-1]
        print(line)
        if line == '' and process.poll() != None:
            break
    if process.returncode != 0:
        raise Exception("Non zero return code: %d" % process.returncode)

def main():

    #  Step 0: get current version and print
    __version__ = open(join(os.path.dirname(os.path.realpath(__file__)), '.git',
                            'HEAD')).read()
    msg_info("Version: %s"%__version__)

    #  Step 1: arguments
    #  setup the arguments
    # Note: action='store_true' means that if option is used, the arg is set to true. Used for flags (not options with arguments).
    parser = argparse.ArgumentParser(description = 'Entrypoint script.')
    parser.add_argument('bids_dir', help='The directory with the input '
                                         'dataset formatted according to '
                                         'the BIDS standard.')
    parser.add_argument('output_dir', help='The directory where the output '
                                           'files will be stored.')
    parser.add_argument('analysis_level', help='Level of the analysis that '
                                               'will be performed. '
                                               'Multiple participant level '
                                               'analyses can be run '
                                               'independently (in parallel)'
                                               ' using the same '
                                               'output_dir.',
                        choices = ['participant',
                                   'group'])
    parser.add_argument('--participant_label', help='The label(s) of the participant(s) that should be analyzed. The label corresponds to sub-<participant_label> from the BIDS spec (so it does not include "sub-"). If this parameter is not provided all subjects should be analyzed. Multiple participants can be specified with a space separated list.', nargs = "+")
    parser.add_argument('--skip_bids_validator', help='Whether or not to perform BIDS dataset validation', action='store_true')
    parser.add_argument('--fmriprep_dir', help='Path of the fmriprep derivatives. If ommited, set to bids_dir/derivatives/fmriprep')
    parser.add_argument('--task', help='Name of the task to be used. If omitted, will search for \'gas\'.')
    parser.add_argument('--space', help='Name of the space to be used. Must be associated with fmriprep output. Default: \'MNI152NLin6Asym\'.')
    parser.add_argument('--work_dir', help='Work dir for temporary files. If omitted, set to \'output_dir/work\'')
    parser.add_argument('--sloppy', help='Only for testing, computes a small part of the maps to save time. Off by default.', action='store_true')
    parser.add_argument('--use_aroma', help='If set, the noise regressors will be those as determined by aroma.', action='store_true')
    parser.add_argument('--overwrite', help='If set, existing results will be overwritten if they exist.', action='store_true')
    parser.add_argument('--label', help='If set, labels the output with custom label.')
    parser.add_argument('-v', '--version', action='version', version='BIDS-App example version {}'.format(__version__))
    #  parse
    args = parser.parse_args()

    # check 1: bids_dir exists?
    if not os.path.isdir(args.bids_dir):
        msg_error("Bids directory %s not found." % args.bids_dir)
        sys.exit(1)
    else:
        #  fall back to default if argument not provided
        bids_dir = args.bids_dir

    # check 2: space given?
    if not args.space:
        args.space = 'MNI152NLin6Asym'
        msg_info('Defaulting to space %s' % args.space)

    # check 3: bids_dir is bids-valid?
    if not args.skip_bids_validator:
        # todo: check if bids-validator is in path. Alternative: directly
        #  call the appropriate pip package - see example in fmriprep
        run('bids-validator --ignoreWarnings %s' % args.bids_dir)
    else:
        msg_info("Skipping bids-validation")

    # initiate BIDS layout
    msg_info("Indexing BIDS dataset...")
    layout = bidslayout(args.bids_dir)

    # check 4: valid subjects?
    subjects_to_analyze = []

    if args.participant_label:  # only for a subset of subjects
        subjects_to_analyze = args.participant_label
        # in that case we must ensure the subjects exist in the layout:
        for subj in subjects_to_analyze:
            if subj not in layout.get_subjects():
                msg_error("Subject %s in not included in the "
                          "bids database." % subj)
                sys.exit(1)
    else:  # for all subjects
        subjects_to_analyze = sorted(layout.get_subjects())

    #  fmriprep dir definition
    if args.fmriprep_dir:
        fmriprep_dir = args.fmriprep_dir
    else:
        fmriprep_dir = join(args.bids_dir, "derivatives", "fmriprep")

    # check 5: fmriprep dir exists?
    if not os.path.isdir(fmriprep_dir):
        msg_error("fmriprep dir %s not found." % fmriprep_dir)

    # sloppiness level
    if args.sloppy:
        sloppy_flag = 1
    else:
        sloppy_flag = 0

    # load the fmriprep derivatives in the BIDS layout
    # note that fmriprep dir should contain a valid dataset_description.json
    # file. Example: {"Name": "Example dataset", "BIDSVersion": "1.0.2",
    # "GeneratedBy": [{"Name": "fmriprep"}]}
    layout.add_derivatives(fmriprep_dir)

    # check 6: valid task?
    if args.task:
        if args.task not in layout.get_tasks():
            msg_error("Selected task %s is not in the BIDS dataset. "
                      "Available tasks are %s." % (args.task,
                                                   layout.get_tasks()))
            sys.exit(1)
        task = args.task
    else:
        # fall back to default value
        task = "gas"

    # check 7: space in fmriprep output?
    spaces = layout.get_spaces(scope='derivatives')  # this selects only
    # space present in the derivative folder
    if args.space not in spaces:
         msg_error("Selected space %s is invalid. Valid spaces are %s"
                   % (args.space, spaces))
         sys.exit(1)
    space = args.space

    # overwrite?
    if args.overwrite:
        overwrite_flag = True
    else:
        overwrite_flag = False

    # label?
    if args.label:
        custom_label = '_label-' + args.label
        msg_info('Outputs will be labeled using %s' % custom_label)
    else:
        custom_label = ''

    # create output dir
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # initiate dataset_description file for outputs
    dataset_description = os.path.join(args.output_dir,
                                       'dataset_description.json')
    with open(dataset_description, 'w') as ds_desc:
        # todo: get the correct BIDS version
        ds_desc.write(('{"Name": "cvrmap", "BIDSVersion": "x.x.x", '
                       '"DatasetType": "derivative", "GeneratedBy": '
                       '[{"Name": "cvrmap"}, {"Version": "%s"}]}')
                      % __version__.rstrip('\n'))
        ds_desc.close()

    # add output dir as BIDS derivatives in layout
    layout.add_derivatives(args.output_dir)

    # get absolute path for our outputs
    output_dir = layout.derivatives['cvrmap'].root

    # workdir
    if args.work_dir:
        work_dir = args.work_dir
    else:
        #  fall back to default value
        work_dir = os.path.join(output_dir, "work")

    # create work dir
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    # print some summary before running
    msg_info("Bids directory: %s" % bids_dir)
    msg_info("Fmriprep directory: %s" % fmriprep_dir)
    msg_info("Subject(s) to analyse: %s" % subjects_to_analyze)
    msg_info("Task to analyse: %s" % task)
    msg_info("Selected space: %s" % space)
    msg_info("Work directory: %s" % work_dir)
    if args.use_aroma:
        msg_info("All the noise regressors found by aroma will be used.")

    # main section

    # running participant level
    if args.analysis_level == "participant":

        # loop over subject to analyze
        for subject_label in subjects_to_analyze:

            # init a basic filter for layout.get()
            basic_filter = {}

            # add participant
            basic_filter.update(subject=subject_label)

            # return_type will always be set to filename
            basic_filter.update(return_type='filename')

            # add space to filter
            basic_filter.update(space=space)

            # add resolution to filter if working with template space
            if not space == 'T1w':
                basic_filter.update(res=2)

            msg_info("Running for participant %s" % subject_label)

            # create output_dir/sub-XXX directory
            subject_output_dir = os.path.join(output_dir,
                                              "sub-" + subject_label)
            Path(subject_output_dir).mkdir(parents=True, exist_ok=True)
            # set paths for various outputs
            outputs = {}
            if args.use_aroma:
               denoise_label = "_denoising-" +'AROMA'
            else:
                denoise_label = ''
            subject_prefix = os.path.join(subject_output_dir,
                                  "sub-" + subject_label)
            prefix = subject_prefix + "_space-" + space + denoise_label + custom_label
            nifti_extension = '.nii.gz'
            report_extension = '.html'
            outputs['cvr'] = prefix + '_cvr' + nifti_extension
            outputs['delay'] = prefix + '_delay' + nifti_extension
            outputs['report'] = prefix + '_report' + report_extension
            outputs['etco2'] = subject_prefix + '_desc-etco2_timecourse'

            # add subject to filter
            basic_filter.update(subject=subject_label)

            # create subject workdir
            subject_work_dir = os.path.join(work_dir,
                                            "sub-" + subject_label)
            Path(subject_work_dir).mkdir(parents=True, exist_ok=True)

            # clean previous tmp files, if any
            files = glob.glob(subject_work_dir + '/*')
            if not len(files) == 0:
                msg_info('Removing temporary files found in %s' % subject_work_dir)
                for f in files:
                    os.remove(f)

            # add task to filter
            basic_filter.update(task=task)

            # get physio data
            physio_filter = basic_filter.copy()
            physio_filter.update({'suffix': "physio"})
            if not space == 'T1w':
                physio_filter.pop('res')
            physio_filter.pop('space')
            physio = DataObj(label=r'$\text{Raw CO}_2\text{ signal}$')
            physio.bids_load(layout, physio_filter, 'timecourse', **{'col': 1})

            # extract upper envelope and baseline
            probe, baseline = core.endtidalextract.run(physio)

            # save the result
            probe.timecourse_save(outputs['etco2'])

            # has this subject been preprocessed with fmriprep?
            # todo: check if fmriprep output exists, with appropriate
            #  options (ica-aroma, res:2).

            # get preprocessed bold dataobj
            basic_filter.update({'desc': 'preproc', 'suffix': 'bold'})
            preproc = DataObj(label='Preprocessed BOLD images from fMRIPrep')
            preproc.bids_load(layout, basic_filter, 'bold')

            # get bold data filtered using aroma
            aroma_filter = basic_filter.copy()

            # aroma denoised data are only create in MNI152NLin6Asym with fMRIPrep

            aroma_filter.update({'space': 'MNI152NLin6Asym'})
            if not space == 'T1w':
                aroma_filter.pop('res')
            aroma_filter.update({'desc': 'smoothAROMAnonaggr'})
            aroma = DataObj(label='Denoised data from AROMA')
            aroma.bids_load(layout, aroma_filter, 'bold')

            # get brain mask
            mask_filter = basic_filter.copy()
            mask_filter.update({'suffix': 'mask', 'desc': 'brain'})
            mask = DataObj(label='Brain mask from fMRIPrep', measurement_type='binary')
            mask.bids_load(layout=layout, filters=mask_filter, data_type='map')

            # get T1w
            t1w_filter = basic_filter.copy()
            t1w_filter.pop('task')
            t1w_filter.update({'suffix': 'T1w', 'desc': 'preproc'})
            if space == 'T1w':
                t1w_filter.pop('space')
            t1w = DataObj(label='Preprocessed T1w from fMRIPrep')
            t1w.bids_load(layout=layout, filters=t1w_filter, data_type='map')

            if not args.use_aroma:
                # find and remove IC's that correlates with probe regressor
                aroma_noise_ic_filter = aroma_filter.copy()
                aroma_noise_ic_filter.pop('desc')
                aroma_noise_ic_filter.pop('space')
                aroma_noise_ic_filter.update({'suffix': 'AROMAnoiseICs', 'extension': '.csv'})
                aroma_noise_ic_list = open(layout.get(**aroma_noise_ic_filter)[0]).read().split(sep=',')

                melodic_mixing_filter = aroma_noise_ic_filter.copy()
                melodic_mixing_filter.update({'desc': 'MELODIC', 'suffix': 'mixing', 'extension': '.tsv'})
                melodic_mixing = layout.get(**melodic_mixing_filter)[0]
                melodic_mixing_df = pd.read_csv(melodic_mixing, sep='\t')

                # todo: put this value in some config file
                noise_ic_pearson_r_threshold = 0.6

                corrected_noise = []
                for noise_idx in aroma_noise_ic_list:
                    ic = DataObj(data=melodic_mixing_df.values[:, int(noise_idx) - 1], sampling_frequency=preproc.sampling_frequency, data_type='timecourse', path=None)
                    if core.tccorr.run(ic, probe) < noise_ic_pearson_r_threshold:
                        corrected_noise.append(noise_idx)

                tmp_output = os.path.join(subject_work_dir, 'tmp.nii.gz')

                msg_info("Data filtering: non-aggressive denoising and highpass")
                fsl_regfilt_cmd = "fsl_regfilt --in=%s --filter=%s --design=%s --out=%s" % (preproc.path, ','.join(corrected_noise), melodic_mixing, tmp_output)
                run(fsl_regfilt_cmd)

                tmp_output_smooth = os.path.join(subject_work_dir, 'tmp_smooth.nii.gz')
                # todo: check 3dmerge is in PATH/find a better solution than hardcoding the path to abin
                # todo: get fwhm from a config file
                fwhm = 5
                smoothing_cmd = "$HOME/abin/3dmerge -1blur_fwhm %s -doall -prefix %s %s" % (fwhm, tmp_output_smooth, tmp_output)
                run(smoothing_cmd)
                tmp_output_mean = os.path.join(subject_work_dir, 'tmp_mean.nii.gz')
                tmp_output_preproc = os.path.join(subject_work_dir, 'tmp_preproc.nii.gz')
                compute_mean_cmd = "fslmaths %s -Tmean %s" % (tmp_output_smooth, tmp_output_mean)

                # todo: compute high-pass filter by taking 2*center frequency of etCO2 curve
                hpsigma = preproc.sampling_frequency*128/2 # 128/TR/2
                highpass_cmd = "fslmaths %s -bptf %s -1 -add %s %s" % (tmp_output_smooth, hpsigma, tmp_output_mean, tmp_output_preproc)
                run(compute_mean_cmd)
                run(highpass_cmd)

                denoised = DataObj(label='Denoised data build out of custom classification of noise IC')
                denoised.nifti_load(path=tmp_output_preproc)

            else:
                denoised = aroma
                if not space == 'MNI152NLin6Asym':
                    # todo: make this verification earlier in the script
                    # todo: perform denoising in T1w space if space=T1w use_aroma is true
                    msg_warning('You are using aroma denoised maps with space=T1w, which has NOT been tested.')

            # compute global signal
            global_signal = DataObj(label='Whole brain BOLD signal')
            global_signal.data = denoised.data.mean(axis=0).mean(axis=0).mean(axis=0)
            global_signal.data = global_signal.data/global_signal.data.mean()  # convenient step that we can do since anyway BOLD signal has no units
            global_signal.sampling_frequency = denoised.sampling_frequency
            global_signal.data_type = 'timecourse'

            # define time 0 by shifting probe to global_signal
            absolute_shift_list = np.arange(-30, 30, 1)
            global_signal_shift = compute_delays(global_signal, probe, absolute_shift_list)[0]

            relative_shift_list = np.arange(-30, 30, 1)

            if not os.path.exists(outputs['delay']) or overwrite_flag:
                # compute delays and fit parameters (this is the most time-consuming step)
                if sloppy_flag:
                    msg_warning('Working in sloppy mode, only for quick testing!')
                    zmask = mask.data
                    zmask[:, :, :59] = np.zeros(zmask[:, :, :59].shape)
                    zmask[:, :, 61:] = np.zeros(zmask[:, :, 61:].shape)
                    denoised.mask = zmask
                else:
                    denoised.mask = mask.data

                # build the shift data needed to compute delays
                shift_options = dict()
                shift_options['origin'] = global_signal_shift
                shift_options['relative_values'] = relative_shift_list

                # compute delays for all voxels
                results = compute_delays(denoised, probe, shift_options)
                # save the obtained map
                results['delay'].nifti_save(denoised.path, outputs['delay'])
                # compute and save response maps
                response = compute_response(results['intercept'], results['slope'], probe.baseline, numpy.mean(probe.data))
                response.nifti_save(denoised.path, outputs['cvr'])
            else:
                results = dict()
                results['delay'] = DataObj(measurement_type='delay')
                results['delay'].nifti_load(outputs['delay'])
                response = DataObj()
                response.nifti_load(outputs['cvr'])

            response.measurement_type = 'CVR'

            msg_info("Building report...")

            # Report
            # init
            report = Report(
                outputs['report'])
            report.init(subject=subject_label,
                        date_and_time=datetime.now(),
                        version=__version__, cmd=args.__dict__)
            # Physio data
            # create figures for report
            physio.make_fig(fig_type='plot',
                            **{'title': r'$\text{Raw CO}_2$',
                               'xlabel': r'$\text{Time (s)}$',
                               'ylabel': r'$\text{CO}_2\text{ '
                                         r'concentration (%)}$'})
            probe.make_fig(fig_type='plot',
                           **{'title': r'$\text{Raw CO}_2$',
                              'xlabel': r'$\text{Time (s)}$',
                              'ylabel': r'$\text{CO}_2\text{ '
                                        r'concentration (%)}$'})
            baseline.make_fig(fig_type='plot',
                              **{'title': r'$\text{Raw CO}_2$',
                                 'xlabel': r'$\text{Time (s)}$',
                                 'ylabel': r'$\text{CO}_2\text{ concentration (%)}$'})
            report.add_subsection(title='Physiological data')
            report.add_sentence(
                sentence="Physiological data, with reconstructed "
                         "upper envelope and baseline:")
            report.add_image(
                gather_figures([probe, baseline, physio]))
            # global signal and etco2
            report.add_section(title='Global Signal and etCO2')
            report.add_sentence(sentence="The computed shift is %s seconds" % global_signal_shift)
            global_signal.make_fig(fig_type='plot', **{
                'title': r'Whole-brain mean BOLD signal',
                'xlabel': r'$\text{Time (s)}$',
                'ylabel': r'BOLD signal (arbitrary units)'})

            # report.add_image(global_signal.figs['plot'])
            report.add_image(gather_figures([global_signal, probe]))

            # info on denoising
            report.add_section(title='Info on denoising')
            if not args.use_aroma:
                report.add_sentence(sentence="The noise regressors as classified by AROMA are (total: %s): %s" % (len(aroma_noise_ic_list), ','.join(aroma_noise_ic_list)))
                report.add_sentence(sentence="Keeping only those that do not correlate too much with the probe regressor gives the following list (total: %s): %s" % (len( corrected_noise),','.join(corrected_noise)))
                report.add_sentence(sentence="Data are smoothed with a FWHM of %s mm" % fwhm)
                report.add_sentence(sentence="Highpass filter cut-off set to %s Hz" % hpsigma)
            else:
                report.add_sentence(sentence="Denoised data from the AROMA classification of noise regressors")

            report.add_section(title='Results')
            # Delay map
            report.add_subsection(title='Delay map')
            results['delay'].make_fig(fig_type='lightbox', **{'background': t1w})
            report.add_image(results['delay'].figs['lightbox'])
            report.add_sentence(
                sentence="Delay map histogram")
            # todo: add some stats (mean and std) of delay map
            results['delay'].make_fig(fig_type='histogram')
            report.add_image(results['delay'].figs['histogram'])
            # CVR
            report.add_subsection(title="CVR map")
            response.make_fig(fig_type='lightbox')
            report.add_image(response.figs['lightbox'])
            # finish the report
            report.finish()

    # running group level
    elif args.analysis_level == "group":
        print('No group level analysis is implemented yet.')

    msg_info("The End!")


if __name__ == '__main__':
    main()
