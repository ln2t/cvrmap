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
import sys

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

    layout = get_bidslayout(args)
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

            outputs = setup_subject_output_paths(output_dir, subject_label, space, res, task, args, custom_label)

            preproc, mni_preproc = get_preproc(basic_filter, layout)
            mask = get_mask(basic_filter, layout)
            aroma_noise_ic_list = get_aroma_noise_ic_list(basic_filter, layout)
            melodic_mixing_df, melodic_mixing_path = get_melodic_mixing(basic_filter, layout)

            if flags['vesselsignal'] or flags['globalsignal']:
                if flags['vesselsignal']:
                    # vesselsignal extraction must be done in MNI space
                    mask_img = get_vesselmask(mni_preproc, parameters['vesseldensity_threshold'])
                    mask_output = outputs['vesselsignal']
                    signal_label = r'$vesselsignal timecourse$'
                if flags['globalsignal']:
                    mask_img = mask.img
                    mask_output = outputs['globalsignal']
                    signal_label = r'$globalsignal timecourse$'

                args.use_aroma = True  # when using vessel or global signal,
                                       # we don't have an external probe to refine ICA-AROMA
                                       # classification of noise sources
                probe = None
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

            if flags['vesselsignal'] or flags['globalsignal']:
                probe, baseline = masksignalextract(mni_preproc, mask_img)
                probe.label = signal_label
                baseline.label = signal_label
                probe.save(mask_output)
                physio = None

            if task == 'restingstate':
                probe.baseline = np.mean(probe.data)

            # get global signal
            global_signal = compute_global_signal(denoised)

            # build once and for all the shifted versions of the probe
            probe.shift_ref = DataObj(sampling_frequency=denoised.sampling_frequency,
                                      data_type='timecourse', data=denoised.data[0, 0, 0, :])
            probe.shift_array = parameters['absolute_shift_list']
            probe.shifted_dataobjects = None
            probe.build_shifted()

            global_signal_shift = compute_delays(global_signal, probe, parameters['absolute_shift_list'])[0]

            denoised.mask = mask.data

            # build the shifted data needed to compute delays
            shift_options = dict()
            shift_options['origin'] = global_signal_shift
            shift_options['relative_values'] = parameters['relative_shift_list']

            # compute delays for all voxels
            msg_info("Computing delays")

            # build once and for all the shifted versions of the probe
            probe.shift_ref = DataObj(sampling_frequency=denoised.sampling_frequency,
                                      data_type='timecourse', data=denoised.data[0, 0, 0, :])
            probe.shift_array = shift_options['origin'] + shift_options['relative_values']
            probe.shifted_dataobjects = None
            probe.build_shifted()

            results = compute_delays(denoised, probe, shift_options)
            results['delay'].units = 'seconds'
            results['delay'].measurement_type = 'delay'

            # save the obtained map
            results['delay'].save(outputs['delay'], denoised.path)

            # compute and save response maps
            results['cvr'] = compute_response(results['intercept'], results['slope'], probe.baseline,
                                              np.mean(probe.data))
            if flags['vesselsignal']:
                results['cvr'].units = "Arbitrary units"
                results['cvr'].measurement_type = 'relative-CVR'
                results['cvr'].data = 10*results['cvr'].data/np.nanstd(results['cvr'].data)
                # this rescaling is mostly for visual purposes
            else:
                results['cvr'].units = "Percentage of BOLD variation/%s" % probe.units
                results['cvr'].measurement_type = 'CVR'
            results['cvr'].save(outputs['cvr'], denoised.path)

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
            results['shifted_probe'] = probe.shifted_dataobjects[global_signal_shift]
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

        msg_info("Starting group analysis!")
        msg_warning("Work in progress!")
        subjects = get_processed_subjects(layout)
        msg_info("Number of subjects detected: %s" % len(subjects))

        import pandas as pd

        participants_fn = layout.get(return_type='filename', extension='.tsv', scope='raw')[0]
        participants_df = pd.read_csv(participants_fn, sep='\t')

        tissue_list = ['GM', 'WM']

        for tissue in tissue_list:
            participants_df[tissue] = ""

        for _sub in subjects:
            _cvr = layout.derivatives['cvrmap'].get(subject=_sub, return_type='filename', suffix='cvr',
                                                    extension='.nii.gz', space=space)
            if len(_cvr) == 1:
                _cvr = _cvr[0]
            else:
                msg_error('Several CVR maps found for subject %s' % _sub)
                sys.exit()

            _masks = dict()
            _roi_mean = dict()

            for _tissue in tissue_list:
                _masks[_tissue] = layout.derivatives['fMRIPrep'].get(subject=_sub, return_type='filename',
                                                                     suffix='probseg', extension='.nii.gz',
                                                                     space=space, label=_tissue)[0]

                _roi_mean[_tissue] = compute_roi_mean(map=_cvr, mask=_masks[_tissue])

                participants_df.loc[participants_df['participant_id'] == 'sub-' + _sub, _tissue] = _roi_mean[_tissue]

    msg_info("The End!")


if __name__ == '__main__':
    main()
