"""
Various input/output tools
"""

import numpy as np
import scipy
from scipy.signal import resample
from scipy.stats import pearsonr

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
    parser = argparse.ArgumentParser()
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
                        help='The label(s) of the participant(s) that should be analyzed. '
                             'The label corresponds to sub-<participant_label> from the BIDS spec '
                             '(so it does not include "sub-"). If this parameter is not provided all subjects '
                             'should be analyzed. Multiple participants can be specified with a space separated list.',
                        nargs="+")
    parser.add_argument('--skip_bids_validator', help='Whether or not to perform BIDS dataset validation',
                        action='store_true')
    parser.add_argument('--fmriprep_dir',
                        help='Path of the fmriprep derivatives. If ommited, set to bids_dir/derivatives/fmriprep')
    parser.add_argument('--task', help='Name of the task to be used. If omitted, will search for \'gas\'.')
    parser.add_argument('--space',
                        help='Name of the space to be used. Must be associated with fmriprep output. '
                             'Default: \'MNI152NLin2009cAsym\'.'
                             'Also accepts resolution modifier '
                             '(e.g. \'MNI152NLin2009cAsym:res-2\') as in fmriprep options.')
    parser.add_argument('--sloppy',
                        help='Only for testing, computes a small part of the maps to save time. Off by default.',
                        action='store_true')
    parser.add_argument('--use_aroma', help='If set, the noise regressors will be those as determined by aroma.',
                        action='store_true')
    parser.add_argument('--overwrite', help='If set, existing results will be overwritten if they exist.',
                        action='store_true')
    parser.add_argument('--label', help='If set, labels the output with custom label.')
    parser.add_argument('-v', '--version', action='version', version='cvrmap version {}'.format(version))
    parser.add_argument('--vesselsignal', help='If set, will extract BOLD signal from vessels as a '
                                               'surrogate for CO2 partial pressure. '
                                               'Results in measures of relative CVR.',
                        action='store_true')
    parser.add_argument('--globalsignal',
                        help='If set, will extract global BOLD signal as a surrogate for CO2 partial pressure. '
                             'Results in measures of relative CVR.',
                        action='store_true')
    parser.add_argument('--config',
                        help='Path to json file fixing the pipeline parameters. '
                             'If omitted, default values will be used.')
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
    from .shell_tools import msg_error
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
    from .shell_tools import msg_error
    import sys
    #  fmriprep dir definition
    if args.fmriprep_dir:
        fmriprep_dir = args.fmriprep_dir
    else:
        fmriprep_dir = join(args.bids_dir, "derivatives", "fmriprep")
    # exists?
    if not isdir(fmriprep_dir):
        msg_error("fmriprep dir %s not found." % fmriprep_dir)
        sys.exit(1)

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
    from .shell_tools import msg_error
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
    from .shell_tools import msg_info
    if args.label:
        custom_label = '_label-' + args.label
        msg_info('Outputs will be labeled using %s' % custom_label)
    else:
        custom_label = ''

    return custom_label


def get_space(args, layout):
    """
        Get space (and res, if any) and checks if present in layout (rawdata and derivatives)

    Args:
        args: return from arguments_manager
        layout: BIDS layout

    Returns:
        string for space entity
    """
    from .shell_tools import msg_info, msg_error
    import sys
    if args.space:
        space = args.space
    else:
        space = 'MNI152NLin2009cAsym'
        msg_info('Defaulting to space %s' % space)

    # check if space arg has res modifier
    res = None
    if ":" in space:
        res = space.split(":")[1].split('-')[1]
        space = space.split(":")[0]

    # space in fmriprep output?
    spaces = layout.get_spaces(scope='derivatives')
    if space not in spaces:
        msg_error("Selected space %s is invalid. Valid spaces are %s" % (args.space, spaces))
        sys.exit(1)

    if args.vesselsignal and space == 'T1w':
        msg_error("The vesselsignal option is not supported in T1w space (yet)")
        sys.exit(1)

    #todo: check if combination space+res is in fmriprep output

    return space, res


def setup_output_dir(args, version, layout):
    """
        Create output dir if it does not exist, together with dataset_description.json file in it with CVRmap version.
        Update BIDS layout to contain this folder as BIDS derivatives.

    Args:
        args: dict, arguments of the script
        version: str, version of the software
        layout: BIDS layout

    Returns:
        layout: BIDS layout, updated with output dir
    """
    import os
    from pathlib import Path  # to create dirs
    # create output dir
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # initiate dataset_description file for outputs
    dataset_description = os.path.join(args.output_dir,
                                       'dataset_description.json')
    with open(dataset_description, 'w') as ds_desc:
        ds_desc.write(('{"Name": "cvrmap", "BIDSVersion": "v1.8.0", '
                       '"DatasetType": "derivative", "GeneratedBy": '
                       '[{"Name": "cvrmap"}, {"Version": "%s"}]}')
                      % version.rstrip('\n'))
        ds_desc.close()

    # warning: adding cvrmap derivatives while running cvrmap can lead to inconsistencies when running jobs in
    # parallel. If error occur, simply re-launch interrupted jobs.
    # todo to solve this:
    #  don't add derivatives for cvrmap, simply build absolute output path (only thing that is needed!)
    layout.add_derivatives(args.output_dir)  # add output dir as BIDS derivatives in layout

    return layout


def set_flags(args):
    """
        Set various flags for options in the main script

    Args:
        args: NameSpace, argument of the script
    Returns:
        dict, with values for the flags as set by the options
    """
    flags = dict()
    # sloppiness
    flags['sloppy'] = args.sloppy
    flags['overwrite'] = args.overwrite
    flags['ica_aroma'] = args.use_aroma
    flags['vesselsignal'] = args.vesselsignal
    flags['globalsignal'] = args.globalsignal

    if flags['ica_aroma']:
        from .shell_tools import msg_info
        msg_info("All the noise regressors found by aroma will be used.")

    flags['no-shift'] = False
    flags['no-denoising'] = False

    if args.sloppy:
        flags['no-shift'] = True  # no shifts used in voxel-wise regression / saves a lot of time / set to True for testing
        flags['no-denoising'] = True  # skip denoising, use preproc from fmriprep / saves time / set to True  for testing

    return flags


def setup_subject_output_paths(output_dir, subject_label, space, res, args, custom_label):
    """
        Setup various paths for subject output. Also creates subject output dir.

    Args:
        output_dir: str, cvrmap output dir
        subject_label: str, subject label
        space: str, space entity
        res: int, resolution entity
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

    # directory for figures in work dir
    figures_dir = os.path.join(subject_output_dir, 'figures')
    Path(figures_dir).mkdir(parents=True, exist_ok=True)

    # directory for extras in the derivatives
    extras_dir = os.path.join(subject_output_dir, 'extras')
    Path(extras_dir).mkdir(parents=True, exist_ok=True)

    # set paths for various outputs
    outputs = {}

    # add denoise label
    if args.use_aroma:
        denoise_label = "_denoising-AROMA"
    else:
        denoise_label = ''

    subject_prefix = os.path.join(subject_output_dir,
                                  "sub-" + subject_label)
    if res is None:
        prefix = subject_prefix + "_space-" + space + denoise_label + custom_label
    else:
        prefix = subject_prefix + "_space-" + space + '_res-' + res + denoise_label + custom_label
    nifti_extension = '.nii.gz'
    report_extension = '.html'
    figures_extension = '.svg'

    # report is in root of derivatives (fmriprep-style), not in subject-specific directory
    outputs['report'] = os.path.join(output_dir,
                                     "sub-" + subject_label + '_report' + report_extension)

    # principal outputs (CVR and Delay map)
    outputs['cvr'] = prefix + '_cvr' + nifti_extension
    outputs['delay'] = prefix + '_delay' + nifti_extension

    # supplementary data (extras)
    if res is None:
        outputs['denoised'] = os.path.join(extras_dir, 'sub-' + subject_label + "_space-" + space + denoise_label
                                           + custom_label + '_denoised' + nifti_extension)
    else:
        outputs['denoised'] = os.path.join(extras_dir, 'sub-' + subject_label + "_space-" + space + '_res-' + res + denoise_label
                                           + custom_label + '_denoised' + nifti_extension)
    outputs['etco2'] = os.path.join(extras_dir, 'sub-' + subject_label + '_desc-etco2_timecourse')
    outputs['vesselsignal'] = os.path.join(extras_dir, 'sub-' + subject_label + '_desc-vesselsignal_timecourse')
    outputs['globalsignal'] = os.path.join(extras_dir, 'sub-' + subject_label + '_desc-globalsignal_timecourse')

    # figures (for the report)
    outputs['breathing_figure'] = os.path.join(figures_dir, 'sub-' + subject_label + '_breathing' + '.svg')
    outputs['boldmean_figure'] = os.path.join(figures_dir, 'sub-' + subject_label + '_boldmean' + '.svg')
    outputs['vesselsignal_figure'] = os.path.join(figures_dir, 'sub-' + subject_label + '_vesselsignal' + '.svg')
    outputs['globalsignal_figure'] = os.path.join(figures_dir, 'sub-' + subject_label + '_globalsignal' + '.svg')

    if res is None:
        outputs['cvr_figure'] = os.path.join(figures_dir, 'sub-' + subject_label + "_space-" + space + denoise_label
                                             + custom_label + '_cvr' + figures_extension)
        outputs['delay_figure'] = os.path.join(figures_dir, 'sub-' + subject_label + "_space-" + space + denoise_label
                                             + custom_label + '_delay' + figures_extension)
        outputs['vesselmask_figure'] = os.path.join(figures_dir, 'sub-' + subject_label + "_space-" + space
                                             + custom_label + '_vesselmask' + figures_extension)
        outputs['globalmask_figure'] = os.path.join(figures_dir, 'sub-' + subject_label + "_space-" + space
                                             + custom_label + '_globalmask' + figures_extension)
    else:
        outputs['cvr_figure'] = os.path.join(figures_dir, 'sub-' + subject_label + "_space-" + space + '_res-' + res + denoise_label
                                             + custom_label + '_cvr' + figures_extension)
        outputs['delay_figure'] = os.path.join(figures_dir, 'sub-' + subject_label + "_space-" + space + '_res-' + res + denoise_label
                                               + custom_label + '_delay' + figures_extension)
        outputs['vesselmask_figure'] = os.path.join(figures_dir, 'sub-' + subject_label + "_space-" + space + '_res-' + res
                                                    + custom_label + '_vesselmask' + figures_extension)
        outputs['globalmask_figure'] = os.path.join(figures_dir, 'sub-' + subject_label + "_space-" + space + '_res-' + res
                                                    + custom_label + '_globalmask' + figures_extension)

    # html reportlets
    outputs['summary_reportlet'] = os.path.join(figures_dir, 'sub-' + subject_label + '_summary' + '.html')
    outputs['denoising_reportlet'] = os.path.join(figures_dir, 'sub-' + subject_label + '_denoising' + '.html')

    return outputs


def get_physio_data(bids_filter, layout):
    """
        Fetch raw physiological data

    Args:
        bids_filter: dict, basic BIDS filter
        layout: BIDS layout

    Returns:
        DataObj for raw physiological data

    """
    from .processing import DataObj
    # get physio data
    physio_filter = bids_filter.copy()
    physio_filter.update({'suffix': "physio"})
    physio_filter.pop('space')
    if 'res' in physio_filter.keys():
        physio_filter.pop('res')
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
    _bids_filter = bids_filter.copy()
    _bids_filter.pop('space')
    if 'res' in _bids_filter.keys():
        _bids_filter.pop('res')
    _bids_filter.update({'suffix': 'AROMAnoiseICs', 'extension': '.csv'})
    return open(layout.get(**_bids_filter)[0]).read().split(sep=',')


def get_melodic_mixing(bids_filter, layout):
    """
        Get all IC's as found by MELODIC

    Args:
        bids_filter: dict, BIDS filter
        layout: BIDSlayout

    Returns:
        panda dataframe, all MELODIC mixing matrix from fmriprep outputs
        str, path to melodic mixing matrix file
    """
    import pandas as pd
    _bids_filter = bids_filter.copy()
    _bids_filter.update({'desc': 'MELODIC', 'suffix': 'mixing', 'extension': '.tsv'})
    _bids_filter.pop('space')
    melodic_mixing = layout.get(**_bids_filter)[0]

    return pd.read_csv(melodic_mixing, sep='\t', header=None), melodic_mixing


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
        if 'res' in t1w_filter.keys():
            t1w_filter.pop('res')
    t1w = DataObj(label='Preprocessed T1w from fMRIPrep')
    t1w.bids_load(layout=layout, filters=t1w_filter, data_type='map')
    return t1w


def get_vesselmask(preproc, threshold):
    """
        Get the vessel density atlas and binarize it to build the vessel mask

    Args:
        preproc: DataObj, used only to get the properties of the fMRI data to have the mask on same grid
        threshold: str, threshold value such as "95%", passed to nilearn.image.binarize_img

    Returns:
        niimg, mask for vessels in MNI space
    """
    from os.path import join, dirname
    from nilearn.image import binarize_img, resample_to_img
    vesselatlas = join(dirname(__file__), '..', 'data', 'VesselDensityLR.nii.gz')
    _vesselatlas = resample_to_img(source_img=vesselatlas, target_img=preproc.path)
    vessel_mask = binarize_img(img=_vesselatlas, threshold=threshold)
    return vessel_mask


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
    _basic_filter = basic_filter.copy()
    _basic_filter.update({'desc': 'preproc', 'suffix': 'bold'})
    preproc = DataObj(label='Preprocessed BOLD images from fMRIPrep')
    preproc.bids_load(layout, _basic_filter, 'bold')
    return preproc


def read_config_file(file=None):
    """
        All processing parameters are set here to their default values.
        If file is provided, then the parameters 'fwhm', 'ic_threshold' and 'vesseldensity_threshold' are read from the file (if unspecified, the default value will be kept).

    Args:
        file, path to json file
    Returns:
        dict, with various values/np.arrays for the parameters used in main script
    """
    import numpy as np
    params = {}

    # Default values

    params['fwhm'] = 5  # in mm
    params['ic_threshold'] = 0.6  # threshold for correlation coefficient (r) to classify ic as noise or not
    params['absolute_shift_list'] = np.arange(-30, 30, 1)  # this is used only for the global delay shift
    params['relative_shift_list'] = np.arange(-30, 30, 1)  # this is used for the voxel-by-voxel shifts
    params['vesseldensity_threshold'] = "99.5%"  # threshold to binarize the vessel density atlas
    params['highpass_frequency'] = 1/120  # in Hz

    if file:
        import json
        from .shell_tools import msg_info

        msg_info('Reading parameters from user-provided configuration file %s' % file)

        with open(file, 'r') as f:
            config = json.load(f)

        keys = ['fwhm', 'ic_threshold', 'vesseldensity_threshold', 'highpass_frequency']

        for key in keys:
            if key in config.keys():
                params[key] = config[key]

    return params


def get_report_config():
    import importlib.resources
    with importlib.resources.path('cvrmap.data', 'reports_config.yml') as file_p:
        return file_p

def save_figs(results, outputs, mask):
    """
        Generate and saves cvr and delay figures.

    Args:
        results: dict, containing the results to save
        outputs: dict, with the paths where to save the figures
        mask: DataObj, to mask the delaymap image
    Returns:
        0
    """
    # Breathing data

    if results['physio'] is not None:
        results['physio'].make_fig(fig_type='plot',
                        **{'title': r'$\text{Raw CO}_2$',
                           'xlabel': r'$\text{Time (s)}$',
                           'ylabel': r'$\text{CO}_2\text{ '
                                     r'concentration (%)}$'})
    if results['vesselsignal']:
        _probe_title = 'Vessel signal'
        _probe_yaxis = 'Arbitrary units'
    else:
        _probe_title = r'$\text{Raw CO}_2$'
        _probe_yaxis = r'$\text{CO}_2\text{ 'r'concentration (%)}$'

    if results['globalsignal']:
        _probe_title = 'Global signal'
        _probe_yaxis = 'Arbitrary units'
    else:
        _probe_title = r'$\text{Raw CO}_2$'
        _probe_yaxis = r'$\text{CO}_2\text{ 'r'concentration (%)}$'

    results['probe'].make_fig(fig_type='plot',
                              **{'title': _probe_title,
                                 'xlabel': r'$\text{Time (s)}$',
                                 'ylabel': _probe_yaxis})

    results['baseline'].make_fig(fig_type='plot',
                      **{'title': r'$\text{Raw CO}_2$',
                         'xlabel': r'$\text{Time (s)}$',
                         'ylabel': r'$\text{CO}_2\text{ concentration (%)}$'})

    results['global_signal'].make_fig(fig_type='plot', **{
        'title': r'Whole-brain mean BOLD signal',
        'xlabel': r'$\text{Time (s)}$',
        'ylabel': r'BOLD signal (arbitrary units)'})

    from .viz import gather_figures
    if results['physio'] is not None:
        breathing_fig = gather_figures([results['probe'], results['baseline'], results['physio']])
        plotly_formatted_svg_write_image(breathing_fig, outputs['breathing_figure'])

    # BOLD mean and etCO2
    boldmean_fig = gather_figures([results['global_signal'], results['probe']])
    plotly_formatted_svg_write_image(boldmean_fig, outputs['boldmean_figure'])

    # CVR and delay map
    import nilearn.plotting as plotting
    import nilearn.image as image

    cut_coords = int(image.load_img(results['cvr'].path).shape[-1]/10)


    _cvr_img = image.math_img('np.nan_to_num(img)', img = results['cvr'].path)  # set nans to 0 for viz
    _ = plotting.plot_img(_cvr_img, cmap='hot', display_mode='z', black_bg=True, cbar_tick_format="%0.2g", annotate=False,
                      colorbar=True, vmin=0, vmax=0.8, cut_coords=cut_coords).savefig(outputs['cvr_figure'])

    vmax = 1
    vmin = -2.5

    _delay_img = image.math_img('np.nan_to_num(img)', img=results['delay'].path)  # set nans to 0 for viz
    _img = image.math_img('np.where(mask, img, %s)' % vmax,
                          img=_delay_img, mask=image.load_img(mask.path))
    _ = plotting.plot_img(_img, cmap='black_purple_r', display_mode='z', black_bg=True, cbar_tick_format="%0.2g",
                          annotate=False, colorbar=True, vmin=vmin,
                          vmax=vmax, cut_coords=cut_coords).savefig(outputs['delay_figure'])
    # binarize_img(img=_vesselatlas, threshold="99%")
    if results['vesselsignal']:
        _ = plotting.plot_roi(results['vesselmask'], bg_img=results['meanepi'], cmap='cool', vmin=0,
                          vmax=1, draw_cross=False).savefig(outputs['vesselmask_figure'])
        plotly_formatted_svg_write_image(results['probe'].figs['plot'], outputs['vesselsignal_figure'])

    if results['globalsignal']:
        _ = plotting.plot_roi(results['globalmask'], bg_img=results['meanepi'], cmap='cool', vmin=0,
                          vmax=1, draw_cross=False).savefig(outputs['globalmask_figure'])
        plotly_formatted_svg_write_image(results['probe'].figs['plot'], outputs['globalsignal_figure'])
    return 0


def save_html_reportlets(subject_label, task, space, args, version, aroma_noise_ic_list,
                                 global_signal_shift, corrected_noise, parameters, outputs):
    """
        Save html reporlets with various data from execution of the pipeline. The reportles will be collected using
        nireports.

    Args:
        subject_label: str
        task: str
        space: str
        args: dict
        version: str
        aroma_noise_ic_list: list
        global_signal_shift: float
        corrected_noise: list
        parameters: dict
        outputs: dict

    Returns:
        0
    """

    SUMMARY_TEMPLATE = """\
    \t<h3 class="elem-title">Summary</h3>
	\t\t<ul class="elem-desc">
    \t\t\t<li>Subject ID: {subject_label}</li>
    \t\t\t<li>Functional Task: {task}</li>
    \t\t\t<li>Output space(s): {spaces}</li>
    \t\t\t<li>CVRmap version: {version}</li>
    \t\t\t<li>Command line options: {cmd_line_options}</li>
    \t\t</ul>
    """

    DENOISING_TEMPLATE = """\
    \t<h3 class="elem-title">Informations on denoising</h3>
    \t\t<ul class="elem-desc">
    \t\t\t<li>Noise regressors as classified by AROMA: {aroma_noise}</li>
    \t\t\t<li>Noise regressors after discarding regressors correlated with probe: {refined_noise}</li>
    \t\t\t<li>CVR maps where smoothed with fwhm of (0=no smoothing): {fwhm} mm</li>
    \t\t</ul>
    """

    # remove uninteresting or empty arguments

    args_dict = dict()
    exclude_list = ['bids_dir', 'output_dir', 'fmriprep_dir', 'config']
    for key in args.__dict__.keys():
        if key not in exclude_list:
            if not args.__dict__[key] is None:
                args_dict[key] = args.__dict__[key]

    with open(outputs['summary_reportlet'], 'w') as file:
        file.write(SUMMARY_TEMPLATE.format(subject_label=subject_label, task=task, spaces=space,
                            version=version, cmd_line_options=args_dict))

    if parameters['fwhm'] is None:
        fwhm = 0
    else:
        fwhm = parameters['fwhm']

    corrected_noise_one_based = []
    for xx in corrected_noise:
        corrected_noise_one_based.append(str(xx + 1))

    with open(outputs['denoising_reportlet'], 'w') as file:
        file.write(DENOISING_TEMPLATE.format(aroma_noise=aroma_noise_ic_list, refined_noise=corrected_noise_one_based, fwhm=fwhm))

    return 0


def plotly_formatted_svg_write_image(fig, output_path):
    """
        The plotly.io.write_image function write svg files without proper xml indentation (svg files are xml files,
        by the way). This causes an issue when using nireports run_reports function, that corrupt the svg files if
        indentation is not standard.
        To solve this bug, we must ensure the save svg files have the correct indentation, and we do this using the
        xmlformatter library.
        The original svg file is replaced in-place after being create by plotly.

    Args:
        fig: a plotly Figure
        output_path: path to where the svg must be saved

    Returns:
        0
    """

    # plotly save image
    fig.write_image(output_path)

    # format svg file
    from xmlformatter import Formatter
    formatter = Formatter(indent="1", indent_char='\t', encoding_output="ISO-8859-1", preserve=["literal"])
    formatted_bytes = formatter.format_file(output_path)

    # replace original file with formatted content
    with open(output_path, 'wb') as file:
        file.write(formatted_bytes)

    return 0