#!/usr/bin/env python3
"""BIDS app to compute Cerebrovascular Reactivity maps
This app complies with the Brain Imaging Data Structure (BIDS) App
standard. It assumes the dataset is organised according to BIDS.
Intended to run in command line with arguments as described in the help
(callable with -h)

Author: Antonin Rovai

Created: May 2022
"""

# imports
import os  # to interact with dirs
from bids import BIDSLayout as bidslayout  # to handle BIDS data
from .utils import *  # custom utilities

def main():
    """
        This is the main function to run cvrmap and is called at the end of this script.

    Returns:
        None
    """
    # Get and print version
    __version__ = get_version()

    msg_info("Version: %s" % __version__)

    args = arguments_manager(__version__)
    fmriprep_dir = get_fmriprep_dir(args)

    msg_info("Indexing BIDS dataset...")

    layout = bidslayout(args.bids_dir, validate=not args.skip_bids_validator)
    layout.add_derivatives(fmriprep_dir)
    subjects_to_analyze = get_subjects_to_analyze(args, layout)
    task = get_task(args, layout)
    custom_label = get_custom_label(args)
    space, res = get_space(args, layout)
    layout = setup_output_dir(args, __version__, layout)
    output_dir = layout.derivatives['cvrmap'].root
    flags = set_flags(args)
    parameters = read_config_file(args.config)

    if flags['no-shift']:
        parameters['relative_shift_list'] = np.array([0])

    # print some summary before running
    msg_info("Bids directory: %s" % layout.root)
    msg_info("Fmriprep directory: %s" % fmriprep_dir)
    msg_info("Subject(s) to analyse: %s" % subjects_to_analyze)
    msg_info("Task to analyse: %s" % task)
    msg_info("Selected space: %s" % space)

    if res is not None:
        msg_info("Selected resolution: %s" % res)

    if args.analysis_level == "participant":

        for subject_label in subjects_to_analyze:
            msg_info("Running for participant %s" % subject_label)

            basic_filter = dict(subject=subject_label, return_type='filename', space=space, res=res, task=task)

            outputs = setup_subject_output_paths(output_dir, subject_label, space, res, args, custom_label)

            preproc = get_preproc(basic_filter, layout)
            mask = get_mask(basic_filter, layout)
            t1w = get_t1w(basic_filter, space, layout)
            aroma_noise_ic_list = get_aroma_noise_ic_list(basic_filter, layout)
            melodic_mixing_df, melodic_mixing_path = get_melodic_mixing(basic_filter, layout)

            if flags['vesselsignal'] or flags['globalsignal']:
                if flags['vesselsignal']:
                    mask_img = get_vesselmask(preproc, parameters['vesseldensity_threshold'])
                    mask_output = outputs['vesselsignal']
                    signal_label = r'$vesselsignal timecourse$'
                if flags['globalsignal']:
                    mask_img = mask.img
                    mask_output = outputs['globalsignal']
                    signal_label = r'$globalsignal timecourse$'
                probe, baseline = masksignalextract(preproc, mask_img)
                probe.label = signal_label
                baseline.label = signal_label
                probe.save(mask_output)
                physio = None
            else:
                physio = get_physio_data(basic_filter, layout)
                probe, baseline = endtidalextract(physio)
                probe.save(outputs['etco2'])

            corrected_noise = get_corrected_noiselist(probe, aroma_noise_ic_list, melodic_mixing_df,
                                                      preproc.sampling_frequency,
                                                      parameters['ic_threshold'], args.use_aroma)

            if flags['sloppy'] or flags['no-denoising']:
                msg_warning('Working in sloppy mode, only for quick testing!')
                msg_info("Skipping data denoising")
                denoised = DataObj()
                denoised.data = preproc.data
                denoised.data_type = preproc.data_type
            else:
                msg_info("Data denoising in progress")
                denoised = bold_denoising(preproc.path, mask.path, melodic_mixing_df, corrected_noise, parameters)


            denoised.measurement_type = 'denoised BOLD'
            denoised.units = 'arbitrary'
            denoised.sampling_frequency = preproc.sampling_frequency
            denoised.save(outputs['denoised'], preproc.path)

            # get global signal
            global_signal = compute_global_signal(denoised)

            global_signal_shift = compute_delays(global_signal, probe, parameters['absolute_shift_list'])[0]

            if not os.path.exists(outputs['delay']) or flags['overwrite']:

                # if flags['sloppy']:
                    # msg_warning('Working in sloppy mode, only for quick testing!')
                    # zmask = mask.data
                    # zmask[:, :, :59] = np.zeros(zmask[:, :, :59].shape)
                    # zmask[:, :, 61:] = np.zeros(zmask[:, :, 61:].shape)
                    # denoised.mask = zmask

                denoised.mask = mask.data

                # build the shifted data needed to compute delays
                shift_options = dict()
                shift_options['origin'] = global_signal_shift
                shift_options['relative_values'] = parameters['relative_shift_list']

                # compute delays for all voxels
                msg_info("Computing delays")
                results = compute_delays(denoised, probe, shift_options)
                results['delay'].units = 'seconds'
                results['delay'].measurement_type = 'delay'

                # save the obtained map
                results['delay'].save(outputs['delay'], denoised.path)

                # compute and save response maps
                results['cvr'] = compute_response(results['intercept'], results['slope'], probe.baseline, np.mean(probe.data))
                if flags['vesselsignal']:
                    results['cvr'].units = "Arbitrary units"
                    results['cvr'].measurement_type = 'relative-CVR'
                    results['cvr'].data = 10*results['cvr'].data/np.nanstd(results['cvr'].data)  # this rescaling is mostly for visual purposes
                else:
                    results['cvr'].units = "Percentage of BOLD variation/%s" % probe.units
                    results['cvr'].measurement_type = 'CVR'
                results['cvr'].save(outputs['cvr'], denoised.path)
            else:
                results = dict()
                results['delay'] = DataObj(measurement_type='delay')
                results['delay'].nifti_load(outputs['delay'])
                results['cvr'] = DataObj()
                results['cvr'].nifti_load(outputs['cvr'])
                results['cvr'].measurement_type = 'CVR'

            results['physio'] = physio
            if flags['vesselsignal']:
                results['vesselmask'] = mask_img
                results['meanepi'] = get_meanepi(preproc)
                results['vesselsignal'] = True
            else:
                results['vesselsignal'] = False

            if flags['globalsignal']:
                results['globalmask'] = mask_img
                results['meanepi'] = get_meanepi(preproc)
                results['globalsignal'] = True
            else:
                results['globalsignal'] = False

            results['probe'] = probe
            results['baseline'] = baseline
            results['global_signal'] = global_signal

            save_figs(results, outputs, mask)

            msg_info("Building report...")

            save_html_reportlets(subject_label, task, space, args, __version__, aroma_noise_ic_list,
                                 global_signal_shift, corrected_noise, parameters, outputs)

            from nireports.assembler.tools import run_reports
            run_reports(output_dir, subject_label, 'madeoutuuid', bootstrap_file=get_report_config(),
                        reportlets_dir=output_dir)

    # running group level
    elif args.analysis_level == "group":
        print('No group level analysis is implemented yet.')

    msg_info("The End!")

if __name__ == '__main__':
    main()
