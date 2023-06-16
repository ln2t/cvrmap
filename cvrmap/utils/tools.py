"""
Various tools and wrappers for basic tasks or simple computations
"""

import numpy as np
import scipy
from scipy.signal import resample
from scipy.stats import pearsonr

def match_timecourses(y1, y2, delay):
    delay = int(delay)
    if delay >= 0:
        y1_matched = y1[:len(y2) - 2*delay]
        y2_matched = y2[delay:len(y2) - delay]
    else:
        delay = -delay
        if len(y2) + delay > len(y1):
            y1_matched = y1[delay:len(y1)]
            y2_matched = y2[:len(y1) - delay]
        else:
            y1_matched = y1[delay:len(y2) + delay]
            y2_matched = y2[:len(y2)]
    return y1_matched, y2_matched


def tccorr(data_object1, data_object2):

    tc1 = data_object1.data
    sf1 = data_object1.sampling_frequency
    tc2 = data_object2.data
    sf2 = data_object2.sampling_frequency

    delay_upper_range = 30  # in seconds
    delay_lower_range = -delay_upper_range

    tc1_n = len(tc1)
    tc2_n = len(tc2)

    span1 = np.arange(tc1_n)/sf1
    span2 = np.arange(tc2_n)/sf2

    # data downsampling

    if sf1 != sf2:
        if sf1 > sf2:
            downsampling_factor = sf1 / sf2
            tc1 = resample(tc1, int(tc1_n/downsampling_factor))
            samplingfreq = sf2
        if sf1 < sf2:
            downsampling_factor = sf2 / sf1
            tc2 = resample(tc2, int(tc2_n / downsampling_factor))
            samplingfreq = sf1
    else:
        samplingfreq = sf1

    tc1_n = len(tc1)
    tc2_n = len(tc2)

    span1 = np.arange(tc1_n)/samplingfreq
    span2 = np.arange(tc2_n)/samplingfreq

    # normalize...

    tc1_norm = tc1 - np.mean(tc1)
    tc1 = tc1_norm/np.std(tc1_norm)
    tc2_norm = tc2 - np.mean(tc2)
    tc2 = tc2_norm / np.std(tc2_norm)

    # delay analysis...

    pearson_r = []
    pearson_p = []

    # delay_range = np.arange(delay_lower_range*samplingfreq, (delay_upper_range + 1)*samplingfreq)
    delay_range = np.arange(delay_lower_range*samplingfreq, samplingfreq*(delay_upper_range + 1), samplingfreq)

    for delay in delay_range:
        if tc1_n >= tc2_n:
            tc1_matched, tc2_matched = match_timecourses(tc1, tc2, delay)
        else:
            tc2_matched, tc1_matched = match_timecourses(tc2, tc1, delay)
        r, p = pearsonr(tc1_matched, tc2_matched)
        pearson_r.append(r)
        pearson_p.append(p)

    pearson_r_max_loc = np.argmax(pearson_r)
    pearson_r_max = pearson_r[pearson_r_max_loc]
    pearson_r_max_p = pearson_p[pearson_r_max_loc]
    delay_max = delay_range[pearson_r_max_loc]/samplingfreq  # in seconds

    # result_summary_text = "Max Pearson R score: " + str(round(pearson_r_max, 3)) + ". Corresponding delay: " + str(round(delay_max, 1)) + " seconds. The associated p-value is " + str(pearson_r_max_p)
    # msg_info("Sign convention: a POSITIVE delay means that the FIRST timecourse must be shifted RIGHT")

    return pearson_r_max


def build_shifted_signal(probe, target, delta_t):
    """
    Shifts the probe signal by the amount delta_t, putting baseline values when extrapolation is needed.
    The baseline is read from probe.baseline; if undefined, the function first calls probe.build_baseline() to ensure it exists.
    Target is a signal used as a reference for the length and sampling frequency of the output.

    Arguments:
    __________

        probe: a DataObj with probe.data_type = 'timecourse'
        target: a DataObj with probe.data_type = 'timecourse'.
        delta_t: an integer (positive, negative or zero)

    Return:
    _______
        DataObj with self.data with shifted points according to the given delta_t. Moreover, it is resampled to the sampling frequency in target, and it's length is also adjusted to the one of the target.
    """

    from .processing import DataObj  # to avoid circular import

    # compute baseline if necessary
    if probe.baseline is None:
        probe.build_baseline()

    # copy length and fill with baseline
    n = len(probe.data)
    data = probe.baseline * np.ones(n)
    probe_sf = probe.sampling_frequency

    # number of points corresponding to delta_t
    delta_n = int(delta_t * probe_sf)

    # shift and copy the original data
    if delta_n == 0:
        data[:] = probe.data

    # this is where the sign convention for delay is made. We choose that a POSITIVE delay means that the data signal occurs AFTER the probe signal.
    if delta_n < 0:
        data[:n+delta_n] = probe.data[-delta_n:]
    if delta_n > 0:
        data[delta_n:] = probe.data[:n-delta_n]
    # we keep here the code used for the opposite signe convention, just in case.
    # if delta_n > 0:
    #     data[:n-delta_n] = probe.data[delta_n:]
    # if delta_n < 0:
    #     data[-delta_n:] = probe.data[:n+delta_n]

    # cut or add points to fit size of target
    target_n = len(target.data)
    target_sf = target.sampling_frequency
    target_t = target_n/target_sf
    probe_t = n/probe_sf

    if probe_t > target_t:
        data = data[:int(target_t*probe_sf)]
    elif probe_t < target_t:
        baseline_padding = probe.baseline * np.ones(int(target_t*probe_sf - n))
        data = [*data, *baseline_padding]

    # resample
    data = scipy.signal.resample(data, target_n)

    return DataObj(data=data, label=('probe timecourse shifted by %s seconds' % delta_t), data_type='timecourse',
                             sampling_frequency=target_sf)


def compute_global_signal(data):
    """
    Inputs: data, a DataObj for some fMRI data

    Returns:

    DataObj of timecourse type with computed global signal
    """

    from .processing import DataObj

    global_signal = DataObj(label='Whole brain BOLD signal')
    global_signal.data = data.data.mean(axis=0).mean(axis=0).mean(axis=0)
    global_signal.data = global_signal.data / global_signal.data.mean()  # convenient step that we can do since anyway BOLD signal has no units
    global_signal.sampling_frequency = data.sampling_frequency
    global_signal.data_type = 'timecourse'
    return global_signal


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

    import os
    import subprocess  # to call stuff outside of python

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


def arguments_manager(version):
    """
    Wrapper to define and read arguments for main function call
    Args:
        version: version to output when calling with -h

    Returns:
        args as read from command line call
    """
    import argparse
    #  Deal with arguments
    parser = argparse.ArgumentParser(description='Entrypoint script.')
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
                        choices=['participant', 'group'])
    parser.add_argument('--participant_label',
                        help='The label(s) of the participant(s) that should be analyzed. The label corresponds to sub-<participant_label> from the BIDS spec (so it does not include "sub-"). If this parameter is not provided all subjects should be analyzed. Multiple participants can be specified with a space separated list.',
                        nargs="+")
    parser.add_argument('--skip_bids_validator', help='Whether or not to perform BIDS dataset validation',
                        action='store_true')
    parser.add_argument('--fmriprep_dir',
                        help='Path of the fmriprep derivatives. If ommited, set to bids_dir/derivatives/fmriprep')
    parser.add_argument('--task', help='Name of the task to be used. If omitted, will search for \'gas\'.')
    parser.add_argument('--space',
                        help='Name of the space to be used. Must be associated with fmriprep output. Default: \'MNI152NLin6Asym\'.')
    parser.add_argument('--work_dir', help='Work dir for temporary files. If omitted, set to \'output_dir/work\'')
    parser.add_argument('--sloppy',
                        help='Only for testing, computes a small part of the maps to save time. Off by default.',
                        action='store_true')
    parser.add_argument('--use_aroma', help='If set, the noise regressors will be those as determined by aroma.',
                        action='store_true')
    parser.add_argument('--overwrite', help='If set, existing results will be overwritten if they exist.',
                        action='store_true')
    parser.add_argument('--label', help='If set, labels the output with custom label.')
    parser.add_argument('-v', '--version', action='version', version='BIDS-App example version {}'.format(version))
    return parser.parse_args()


def get_subjects_to_analyze(args, layout):
    """
    Generate list of subjects to analyze given the options and available subjects
    Args:
        args: return from arguments_manager
        layout: BIDS layout
    Returns:
        list of subjects to loop over
    """
    from .shellprints import msg_error
    import sys
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
    return subjects_to_analyze


def get_fmriprep_dir(args):
    """
    Get and check existence of fmriprep dir from options or default
    Args:
        args: return from arguments_manager

    Returns:
        path to fmriprep dir
    """
    from os.path import join, isdir
    from .shellprints import msg_error
    #  fmriprep dir definition
    if args.fmriprep_dir:
        fmriprep_dir = args.fmriprep_dir
    else:
        fmriprep_dir = join(args.bids_dir, "derivatives", "fmriprep")
    # exists?
    if not isdir(fmriprep_dir):
        msg_error("fmriprep dir %s not found." % fmriprep_dir)

    return fmriprep_dir


def get_task(args, layout):
    """
    Get and check task option or set to default
    Args:
        args: return from arguments_manager
        layout: BIDS layout

    Returns:
        string for task entity
    """
    from .shellprints import msg_error
    import sys
    if args.task:
        task = args.task
    else:
        # fall back to default value
        task = "gas"

    if task not in layout.get_tasks():
        msg_error("Selected task %s is not in the BIDS dataset. "
                  "Available tasks are %s." % (args.task,
                                               layout.get_tasks()))
        sys.exit(1)

    return task


def get_custom_label(args):
    """
    Create custom label if specified
    Args:
        args: return from arguments_manager

    Returns:
        string for task entity
    """
    from .shellprints import msg_info
    if args.label:
        custom_label = '_label-' + args.label
        msg_info('Outputs will be labeled using %s' % custom_label)
    else:
        custom_label = ''

    return custom_label


def get_space(args, layout):
    """
    Get space and checks if present in layout (rawdata and derivatives)
    Args:
        args: return from arguments_manager
        layout: BIDS layout

    Returns:
        string for space entity
    """
    from .shellprints import msg_info, msg_error
    import sys
    if args.space:
        space = args.space
    else:
        space = 'MNI152NLin6Asym'
        msg_info('Defaulting to space %s' % space)

    # space in fmriprep output?
    spaces = layout.get_spaces(scope='derivatives')
    if space not in spaces:
        msg_error("Selected space %s is invalid. Valid spaces are %s" % (args.space, spaces))
        sys.exit(1)

    return space


def setup_output_dir(args, version, layout):
    import os
    from pathlib import Path  # to create dirs
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
                      % version.rstrip('\n'))
        ds_desc.close()

    layout.add_derivatives(args.output_dir)  # add output dir as BIDS derivatives in layout

    return layout


def set_flags(args):
    flags = dict()
    # sloppiness
    flags['sloppy'] = args.sloppy
    flags['overwrite'] = args.overwrite
    flags['ica_aroma'] = args.use_aroma
    if flags['ica_aroma']:
        from shellprints import msg_info
        msg_info("All the noise regressors found by aroma will be used.")

    return flags


def setup_subject_output_paths(output_dir, subject_label, space, args, custom_label):
    """
    Setup various paths for subject output. Also creates subject output dir.
    Args:
        output_dir: str, cvrmap output dir
        subject_label: str, subject label
        space: str, space entity
        args: output of arguments_manager
        custom_label: str, custom label for outputs

    Returns:
        dict with various output paths (str)

    """
    from pathlib import Path  # to create dirs
    import os
    # create output_dir/sub-XXX directory
    subject_output_dir = os.path.join(output_dir,
                                      "sub-" + subject_label)
    Path(subject_output_dir).mkdir(parents=True, exist_ok=True)

    # directory for figures
    figures_dir = os.path.join(subject_output_dir, 'figures')
    Path(figures_dir).mkdir(parents=True, exist_ok=True)

    # directory for extras
    extras_dir = os.path.join(subject_output_dir, 'extras')
    Path(extras_dir).mkdir(parents=True, exist_ok=True)

    # set paths for various outputs
    outputs = {}
    if args.use_aroma:
        denoise_label = "_denoising-AROMA"
    else:
        denoise_label = ''
    subject_prefix = os.path.join(subject_output_dir,
                                  "sub-" + subject_label)
    prefix = subject_prefix + "_space-" + space + denoise_label + custom_label
    nifti_extension = '.nii.gz'
    report_extension = '.html'
    figures_extension = '.svg'

    # report
    # report is in root of derivatives (fmriprep-style), not in subject-specific directory
    outputs['report'] = os.path.join(output_dir,
                                     "sub-" + subject_label + '_report' + report_extension)

    # principal outputs (CVR and Delay map)
    outputs['cvr'] = prefix + '_cvr' + nifti_extension
    outputs['delay'] = prefix + '_delay' + nifti_extension

    # supplementary data (extras)
    outputs['denoised'] = os.path.join(extras_dir, 'sub-' + subject_label + "_space-" + space + denoise_label
                 + custom_label + '_denoised' + nifti_extension)
    outputs['etco2'] = os.path.join(extras_dir, 'sub-' + subject_label + '_desc-etco2_timecourse')

    # figures (for the report)
    outputs['breathing_figure'] = os.path.join(figures_dir, 'sub-' + subject_label + '_breathing' + '.png')
    outputs['boldmean_figure'] = os.path.join(figures_dir, 'sub-' + subject_label + '_boldmean' + '.png')
    outputs['cvr_figure'] = os.path.join(figures_dir, 'sub-' + subject_label + "_space-" + space + denoise_label
                                         + custom_label + '_cvr' + figures_extension)
    outputs['delay_figure'] = os.path.join(figures_dir, 'sub-' + subject_label + "_space-" + space + denoise_label
                                         + custom_label + '_delay' + figures_extension)

    return outputs


def save_figs(results, outputs, mask):
    """
    Generate and saves cvr and delay figures.
    Args:
        results: dict, containing the results to save
        outputs: dict, witht the paths where to save the figures
        mask: DataObj, to mask the delaymap image
    Returns:
        0 if successful
    """
    # Breathing data

    results['physio'].make_fig(fig_type='plot',
                    **{'title': r'$\text{Raw CO}_2$',
                       'xlabel': r'$\text{Time (s)}$',
                       'ylabel': r'$\text{CO}_2\text{ '
                                 r'concentration (%)}$'})
    results['probe'].make_fig(fig_type='plot',
                   **{'title': r'$\text{Raw CO}_2$',
                      'xlabel': r'$\text{Time (s)}$',
                      'ylabel': r'$\text{CO}_2\text{ '
                                r'concentration (%)}$'})
    results['baseline'].make_fig(fig_type='plot',
                      **{'title': r'$\text{Raw CO}_2$',
                         'xlabel': r'$\text{Time (s)}$',
                         'ylabel': r'$\text{CO}_2\text{ concentration (%)}$'})

    results['global_signal'].make_fig(fig_type='plot', **{
        'title': r'Whole-brain mean BOLD signal',
        'xlabel': r'$\text{Time (s)}$',
        'ylabel': r'BOLD signal (arbitrary units)'})

    from .viz import gather_figures
    breathing_fig = gather_figures([results['probe'], results['baseline'], results['physio']])
    breathing_fig.write_image(outputs['breathing_figure'])

    # BOLD mean and etCO2
    boldmean_fig = gather_figures([results['global_signal'], results['probe']])
    boldmean_fig.write_image(outputs['boldmean_figure'])

    # CVR and delay map
    import nilearn.plotting as plotting
    import nilearn.image as image

    cut_coords = int(image.load_img(results['cvr'].path).shape[-1]/10)


    _ = plotting.plot_img(results['cvr'].path, cmap='hot', display_mode='z', black_bg=True, cbar_tick_format="%0.2g", annotate=False,
                      colorbar=True, vmin=0, vmax=0.8, cut_coords=cut_coords).savefig(outputs['cvr_figure'])

    vmax = 1
    vmin = -2.5

    _img = image.math_img('np.where(mask, img, %s)' % vmax,
                          img=image.load_img(results['delay'].path), mask=image.load_img(mask.path))
    _ = plotting.plot_img(_img, cmap='black_purple_r', display_mode='z', black_bg=True, cbar_tick_format="%0.2g",
                          annotate=False, colorbar=True, vmin=vmin,
                          vmax=vmax, cut_coords=cut_coords).savefig(outputs['delay_figure'])

    return 0


def get_physio_data(bids_filter, layout):
    """
    Fetch raw physiological data
    Args:
        bids_filter: dict, basic BIDS filter
        layout: BIDS layout

    Returns:
        DataObj for raw physiolocical data

    """
    from .processing import DataObj
    # get physio data
    physio_filter = bids_filter.copy()
    physio_filter.update({'suffix': "physio"})
    physio_filter.pop('space')
    physio = DataObj(label=r'$\text{Raw CO}_2\text{ signal}$')
    physio.bids_load(layout, physio_filter, 'timecourse', **{'col': 1})
    return physio


def get_aroma_noise_ic_list(bids_filter, layout):
    """
    Get the list of Independent Components that were classified as noise by ICA-AROMA, from fmriprep outputs
    Args:
        bids_filter: dict, BIDS filter
        layout: BIDSlayout

    Returns:
        list, list of noise ICs
    """
    # find and remove IC's that correlates with probe regressor
    bids_filter.pop('desc')
    bids_filter.pop('space')
    if 'res' in bids_filter.keys():
        bids_filter.pop('res')
    bids_filter.update({'suffix': 'AROMAnoiseICs', 'extension': '.csv'})
    return open(layout.get(**bids_filter)[0]).read().split(sep=',')


def get_melodic_mixing(bids_filter, layout):
    """
    Get all IC's as found by MELODIC
    Args:
        bids_filter: dict, BIDS filter
        layout: BIDSlayout

    Returns:
        panda dataframe, all MELODIC mixing matrix from fmriprep outputs
    """
    import pandas as pd
    bids_filter.update({'desc': 'MELODIC', 'suffix': 'mixing', 'extension': '.tsv'})
    melodic_mixing = layout.get(**bids_filter)[0]

    return pd.read_csv(melodic_mixing, sep='\t', header=None)


def get_corrected_noiselist(probe, aroma_noise_ic_list, melodic_mixing_df, sf, noise_ic_pearson_r_threshold, aroma_flag):
    """

    Args:
        probe: DataObj, the probe data
        aroma_noise_ic_list: list, noise IC from AROMA
        melodic_mixing_df: pandas dataframe, all MELODIC components
        sf: float, BOLD sampling frequency
        noise_ic_pearson_r_threshold: float, threshold for correlation to correct noise classification
        aroma_flag: bool, flag to use orignal aroma classification or not

    Returns:
        list, refined classification of noise regressors
    """
    from .preprocessing import DataObj
    corrected_noise = []
    for noise_idx in aroma_noise_ic_list:
        ic = DataObj(data=melodic_mixing_df.values[:, int(noise_idx) - 1],
                     sampling_frequency=sf, data_type='timecourse', path=None)
        if tccorr(ic, probe) < noise_ic_pearson_r_threshold or aroma_flag:
            corrected_noise.append(int(noise_idx) - 1)
    return corrected_noise


def get_mask(basic_filter, layout):
    """
    Load brain mask for subject
    Args:
        basic_filter: dict, BIDS filter
        layout: BIDS layout

    Returns:
        DataObj, brain mask
    """
    from .processing import DataObj
    mask_filter = basic_filter.copy()
    mask_filter.update({'suffix': 'mask', 'desc': 'brain'})
    mask = DataObj(label='Brain mask from fMRIPrep', measurement_type='binary')
    mask.bids_load(layout=layout, filters=mask_filter, data_type='map')
    return mask


def get_t1w(basic_filter, space, layout):
    """
    Load t1w image for subject
    Args:
        basic_filter: dict, BIDS filter
        space: str
        layout: BIDS layout

    Returns:
        DataObj, t1w image of the subject
    """
    from .preprocessing import DataObj
    t1w_filter = basic_filter.copy()
    t1w_filter.pop('task')
    t1w_filter.update({'suffix': 'T1w', 'desc': 'preproc'})
    if space == 'T1w':
        t1w_filter.pop('space')
    t1w = DataObj(label='Preprocessed T1w from fMRIPrep')
    t1w.bids_load(layout=layout, filters=t1w_filter, data_type='map')
    return t1w


def get_preproc(basic_filter, layout):
    """
    Load preprocessed image for subject
    Args:
        basic_filter: dict, BIDS filter
        layout: BIDS layout

    Returns:
        DataObj, BOLD preprocessed image of the subject
    """
    from .preprocessing import DataObj
    basic_filter.update({'desc': 'preproc', 'suffix': 'bold'})
    preproc = DataObj(label='Preprocessed BOLD images from fMRIPrep')
    preproc.bids_load(layout, basic_filter, 'bold')
    return preproc


def read_config_file(flags):
    """
    Define all parameters in this function. In the future this will be done by reading a config file (if provided).
    Args:
        floags, dict with boolean values
    Returns:
        dict, with various values/np.arrays for the parameters used in main script
    """
    import numpy as np
    params = {}
    # todo: read these values from a config file
    params['fwhm'] = 5  # in mm
    params['ic_threshold'] = 0.6  # threshold for correlation coefficient (r) to classify ic as noise or not
    params['absolute_shift_list'] = np.arange(-30, 30, 1)  # this is used only for the global delay shift
    params['relative_shift_list'] = np.arange(-30, 30, 1)  # this is used for the voxel-by-voxel shifts
    if flags['no-shift']:
        params['relative_shift_list'] = np.array([0])
    return params