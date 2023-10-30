"""
Classes and Functions for actual data processing.
"""

# imports

import numpy as np
import scipy.stats

from .math_tools import *
from .shell_tools import *

# classes


class DataObj:
    """
        Class to keep together data with sampling freq and other useful information on the data

        For most applications, we need the probe or fMRI data together with
        the corresponding sampling frequency. data_type must be "timecourse"
        or "map" or "bold". Sampling frequency in Hz.
        label is a field used to put some human-readable struff to describe what the object is containing, e.g. 'this is the denoised data'
        fig is a figure object from plotly
        measurement_type is the type of the actual values, like 'CVR' or 'delay', or 'concentration'.
        mask is a binary map, can be used to mask data in the case of 'map' or 'bold'data_type.

    """
    def __init__(self, data=None, sampling_frequency=None, data_type=None, path=None, label=None, figs=None, measurement_type=None, mask=None, baseline=None, units=None, img=None):
        self.sampling_frequency = sampling_frequency
        self.data = data
        self.data_type = data_type
        self.path = path
        self.label = label
        self.figs = figs
        self.measurement_type = measurement_type
        self.mask = mask
        self.baseline = baseline
        self.units = units
        self.img = img

    def bids_load(self, layout, filters, data_type, **kwargs):
        """
            Load DataObj from a BIDS layout and filters

            data_type must be 'timecourse' or "map" or "bold". The filter is used to extract both json and the actual data.
            kwargs is used to specify some options for the 'timecourse' type:
            if kwargs = {'col': 0}, then the first column of the .tsv file is extracted.

        """
        import nibabel
        import numpy
        import json
        json_path = layout.get(**filters, extension='json')[0]
        with open(json_path) as f:
            lf = json.load(f)
            if data_type == 'timecourse':
                data_path = layout.get(**filters, extension='tsv.gz')[0]
                data = numpy.loadtxt(data_path, usecols=kwargs['col'])
                sampling_frequency = lf['SamplingFrequency']
                units = lf['co2']['Units']
                if units == '%':
                    # convert % of co2 partial pressure to mmHg
                    msg_info('Converting CO2 units from percentage of partial pressure to mmHg (1 mmHg = 7.6%p.p.)')
                    data = data*7.6
                    units = 'mmHg'
                else:
                    if not units == 'mmHg':
                        msg_warning('The units read from json file, %s, are unknown. This affects the units of CVR.' % str(units))
                self.units = units
            else:
                data_path = layout.get(**filters, extension='nii.gz')[0]
                self.img = nibabel.load(data_path)
                data = self.img.get_fdata()
                if data_type == 'map':
                    sampling_frequency = 0
                elif data_type == 'bold':
                    sampling_frequency = 1 / lf['RepetitionTime']

        self.sampling_frequency = sampling_frequency
        self.data = data
        self.data_type = data_type
        self.path = data_path

    def nifti_load(self, path):
        """
            Load a nifti image from path and stores it in a DataObj

        """
        import nibabel
        img = nibabel.load(path)
        data = img.get_fdata()
        repetition_time = img.header.get_zooms()[-1]
        sf = 1 / repetition_time
        d_type = 'unknown'
        if len(data.shape) == 3:
            d_type = 'map'
        elif len(data.shape) == 4:
            d_type = 'bold'
        self.sampling_frequency = sf
        self.data = data
        self.data_type = d_type
        self.path = path

    def save(self, path_to_save, path_to_ref=None):
        """
            For self.data_type == 'map' or self.data_type == 'bold':
                Save data to path in nifti format using affine from ref
                path_to_ref: path to nifti file to extract affine data
                path_to_save: path where to save the nifti image

            For self.data_type == 'timecourse':
                Save timecourse data to path in .tsv.gz format - path_to_save must be something live /my/file/is/here and this function will write /my/file/is/here.tsv.gz and /my/file/is/here.json for timecourses
                path_to_save: path where to save the data
                path_to_ref: not needed in that case

        """
        import json

        # add path to class to keep track of things
        if self.path is None:
            self.path = path_to_save

        if self.data_type == 'map' or self.data_type == 'bold':
            import nibabel
            img = nibabel.Nifti1Image(self.data, nibabel.load(path_to_ref).affine)
            nibabel.save(img, path_to_save)
            if self.path is None:
                self.path = path_to_save

            json_path = []
            # create json file with sampling freq
            if path_to_save.split('.')[-1] == 'gz':
                json_path = '.'.join(path_to_save.split('.')[:-2]) + '.json'
            if path_to_save.split('.')[-1] == 'nii':
                json_path = '.'.join(path_to_save.split('.')[:-1]) + '.json'

            json_data = {
                'MeasurementType': self.measurement_type,
                'Units': self.units
            }
            with open(json_path, 'w') as outfile:
                json.dump(json_data, outfile)

        if self.data_type == 'timecourse':
            import numpy
            import gzip
            import os
            data_path = path_to_save + '.tsv'

            # save to temporary .tsv file
            numpy.savetxt(data_path, numpy.round(self.data, 4), '%1.4f')

            # compress as .tsv.gz
            with open(data_path, 'rb') as f_in, gzip.open(data_path + '.gz', 'wb') as f_out:
                f_out.writelines(f_in)

            os.remove(data_path)

            # create json file with sampling freq
            json_path = path_to_save + '.json'
            json_data = {
                'SamplingFrequency': self.sampling_frequency
            }
            with open(json_path, 'w') as outfile:
                json.dump(json_data, outfile)

    def make_fig(self, fig_type, **kwargs):
        """
            Create figure object from the data in self

            The figure is save in self.figs[fig_type].
            fig_type: 'plot', 'histogram' or 'lightbox'

             If self.data_type is "timecourse": fig_type must be 'plot' and plot the timecourse
                kwargs in that case must have values for 'title', 'xlabel' and 'ylabel'
             If self.data_type is "map":
                If fig_type is histogram:
                    create a histogram. kwargs can be empty
                If fig_type is lightbox:
                    create a lightbox. kwargs can be empty, but if not, it must have a single field called 'background'.
                    The corresponding value must be a DataObj of data_type = 'map' and will be used as a background image.
                    Dimensions must match those of self.data.
                    NB: the color scales are build automatically with hardcoded values, different for CVR and delays.

        """
        import numpy as np
        import plotly.graph_objects as go
        import plotly.express as px

        if self.figs is None:
            self.figs = {}

        if self.data_type == "timecourse":
            sampling_freq = self.sampling_frequency
            n_samples = len(self.data)
            time_step = 1 / sampling_freq
            total_time = time_step * n_samples
            time_span = np.arange(0, total_time, time_step)
            self.figs[fig_type] = go.Figure()
            self.figs[fig_type].add_trace(go.Scatter(x=time_span, y=self.data))
            self.figs[fig_type].update_layout(
                title=kwargs['title'],
                xaxis_title=kwargs['xlabel'],
                yaxis_title=kwargs['ylabel']
            )

        if self.data_type == "map":
            if fig_type == "histogram":
                self.figs[fig_type] = px.histogram(self.data.flatten())

            if fig_type == "lightbox":

                # get sizes
                nx, ny, nz = self.data.shape

                # flip y-axis for more convenient visualisation
                flipped_data = self.data[:,::-1,:]

                # crop data to zoom into interesting region (always the same numbers since we are in MNI)
                percentage_lower = 0.1
                percentage_higher = 0.9
                cropped_data = flipped_data[int(percentage_lower*nx):int(percentage_higher*nx), int(percentage_lower*ny):int(percentage_higher*ny), int(percentage_lower*nz):int(percentage_higher*nz)]

                nx_cropped, ny_cropped, nz_cropped = cropped_data.shape

                zmax = 10000
                zmin = -10000
                measurement_label = None

                if self.measurement_type == 'delay':
                    zmax = 15
                    zmin = -15
                    measurement_label = 'Delays (s)'
                elif self.measurement_type == 'CVR':
                    zmax = 5
                    zmin = -5
                    measurement_label = '%relative bold/CO2(%)'
                elif self.measurement_type == 'binary':
                    zmax = 1
                    zmin = -1
                    measurement_label = 'binary'
                    # todo: fix issue with colorscale in the case of binary map (mask)
                positive_colorscale = px.colors.sample_colorscale(
                    px.colors.sequential.Hot, abs(zmax))
                negative_colorscale = px.colors.sample_colorscale(
                    px.colors.sequential.Blues, abs(zmin))
                custom_color_scale = [*negative_colorscale,
                                      *positive_colorscale]

                img_sequence = [cropped_data[:, :, z].transpose() for z in
                                np.arange(nz_cropped)]

                self.figs[fig_type] = px.imshow(np.array(img_sequence),
                                                facet_col=0,
                                                range_color=[zmin, zmax],
                                                facet_col_wrap=5,
                                                facet_row_spacing=0.001,
                                                labels={'facet_col': 'z', 'color': measurement_label},
                                                color_continuous_scale=custom_color_scale)
                self.figs[fig_type].update_layout(width=1800, height=round(400 * nz / 5),
                                                  margin=dict(r=0, l=0, t=0,
                                                              b=0))

                # todo: check that kwargs exist and contain required field 'background'.
                # get associated T1w image
                # if not kwargs['background'] is None:
                #     t1w = kwargs['background']
                #     # todo: add check that dimensions of t1w.data match those of self.data.
                #     flipped_t1w = t1w.data[:, ::-1, :]
                #     cropped_t1w = flipped_t1w[int(percentage_lower * nx):int(percentage_higher * nx),
                #                   int(percentage_lower * ny):int(percentage_higher * ny),
                #                   int(percentage_lower * nz):int(percentage_higher * nz)]
                #     t1w_sequence = [cropped_t1w[:, :, z].transpose() for z in
                #                     np.arange(nz_cropped)]

        self.figs[fig_type].data[-1].name = self.label

    def build_baseline(self):
        """
            Compute baseline of data and stores it in self.baseline. Only for self.data_type='timecourse'

        """

        import peakutils

        if self.data_type == 'timecourse':
            self.baseline = round(np.mean(peakutils.baseline(self.data)), 6)

# functions


def compute_delays(reference, probe, shifts_option):
    """
        This function optimize linear correlation between the reference signal and shifted versions of the probe signal.
        This script involve potentially many recursion. It has several levels of usage, each with its own inputs/outputs.

    ----------------
        Level 1:
    ----------------

        - Arguments:

            * reference: a DataObj with data_type='timecourse' and data of length N
            * probe: a DataObj with data_type='timecourse' and data of length N
            * shifts_option: must be set to None

        - Outputs:

            * slope, intercept, r, p, std: as in scipy.stats.linregress

        - Description:

            At this level the function simply calls scipy.stats.linregress(probe.data, reference.data)

    ----------------
        Level 2:
    ----------------

        - Arguments:

            * reference: a DataObj with data_type='timecourse'
            * probe: a DataObj with data_type='timecourse'
            * shifts_option: an integer

        - Outputs:

            * slope, intercept, r, p, std: as in scipy.signal.linregress

        - Description:

            At this level the function first computes the shifted version of the probe timecourse, using shifts_option as the shift value.
            Note that reference.data and probe.data must not be of same lengths or sampling frequencies: the script takes care of harmonzing the sampling frequencies and array length.

    ----------------
        Level 3:
    ----------------

        - Arguments:

            * reference: a DataObj with data_type='timecourse' and data of length N
            * probe: a DataObj with data_type='timecourse'
            * shifts_option: a numpy array of any length

        - Outputs: delay, slope, intercept, correlation with:

            * delay: the time-shift that gave the best correlation coefficient for the shifted probe signal versus reference.data
            * slope, intercept, correlation: results of the corresponding fit (thus at optimal delay)

        - Description:

            Iterates over elements of shifts_option and calls each time itself at level 2 to extract all data from the fits at various shifts
            Then the shift giving the highest r-correlation score is selected and is returned, alongside the corresponding fit results (slope, intercept, r-score)

    ----------------
        Level 4:
    ----------------

        - Arguments:

            * reference: a DataObj with data_type='bold'. reference.mask must be set to some mask on which the computation will be done. reference.mask must have the same shape as reference.data[:,:,:,t]
            * probe: a DataObj with data_type='timecourse'
            * shifts_option: dict with two keys:
                ** shifts_option['origin']: an integer, used as the origin of times for the delay maps
                ** shifts_option['relative_values']: a numpy array of any length, used to compute the delays

        - Outputs: a dict() with the following keys/values:

            * key: delay, value: a DataObj with data_type='map', measurement_type='delay', and delay.data[x,y,z] = delay optimizing the correlation between probe['signal].data and reference.data[x,y,z,:]
            * keys: intercept, slope, correlation, values: DataObjs with data_type='map', with data corresponding to the fit at optimum shift.

        - Description:

            This is the top-level usage of this function. It takes a bold series specified in reference.data, and loop over all it's voxels provided reference.mask > 0.
            For each voxel, say (x,y,z), the corresponding time series reference.data[x,y,z,:] is then extracted and compute_delay is called at level 3 to find optimum time delay bewteen reference.data[x,y,z,:] and probe.data.
            The  array shifts_option['relative_values'] gives the values on which the optimum search must be run over. The origin of time, given by shifts_option['origin'], will appear as 0 in the delay map.

            Helper on delay and origin of time:
                Say the probe signal has a delay of 8s with the global brain bold signal and we set shifts_option['origin'] = 0.
                Then if shifts_option['relative_values'] = [-10 -5 0 5 10], a voxel having a delay of 0s
                is to be thought as synced with the probe signal.
                So in that case, it might be more relevant to set shifts_option['origin'] to 8, so that a voxel with delay = 0
                is now synced with the global signal instead.
    """

    shifted_probes = dict()
    script_progress_sentence = "Computing delays....."

    if reference.data_type == 'bold':
        # level 4
        # prepare shift data
        shift_origin = shifts_option['origin']
        absolute_shift_list = shift_origin + shifts_option['relative_values']

        # prepare parsing over whole volumes
        n_x, n_y, n_z, n_t = reference.data.shape
        n_x_range = range(n_x)
        n_y_range = range(n_y)
        n_z_range = range(n_z)
        total_voxels = n_x * n_y * n_z

        # quick check: is the mask of the same size as the spatial part of the bold data?
        if not ([n_x, n_y, n_z] == np.array(reference.mask.shape)).all():
            msg_error('Mask has not the correct dimension (got %s instead of %s)' % (np.array(reference.mask.shape), [n_x, n_y, n_z]))
            return 1

        # prepare outputs
        absolute_delay_data = np.zeros([n_x, n_y, n_z])
        slope_data = np.zeros([n_x, n_y, n_z])
        intercept_data = np.zeros([n_x, n_y, n_z])
        correlation_data = np.zeros([n_x, n_y, n_z])

        absolute_delay_data[:] = np.nan
        slope_data[:] = np.nan
        intercept_data[:] = np.nan
        correlation_data[:] = np.nan

        # Parsing
        loop_counter = 0
        printProgressBar(0, total_voxels, prefix=script_progress_sentence)
        for i_z in n_z_range:
            for i_x in n_x_range:
                for i_y in n_y_range:
                    loop_counter += 1
                    if reference.mask[i_x, i_y, i_z] > 0:
                        bold = DataObj(data=reference.data[i_x, i_y, i_z, :], data_type='timecourse', sampling_frequency=reference.sampling_frequency)
                        absolute_delay_data[i_x, i_y, i_z], slope_data[i_x, i_y, i_z], intercept_data[i_x, i_y, i_z], correlation_data[i_x, i_y, i_z] = compute_delays(bold, probe, absolute_shift_list)
            printProgressBar(loop_counter, total_voxels, prefix=script_progress_sentence)

        delay_data = absolute_delay_data.copy() - shift_origin

        delay_data = np.where(correlation_data > 0.6, delay_data, float('nan'))
        # delay_data = np.where(correlation_data > 0.6, delay_data, 0.)
        # slope_data = np.where(correlation_data > 0.6, slope_data, float('nan'))
        # intercept_data = np.where(correlation_data > 0.6, intercept_data, float('nan'))

        delay = DataObj(data=delay_data, data_type='map', measurement_type='delay')
        slope = DataObj(data=slope_data, data_type='map')
        intercept = DataObj(data=intercept_data, data_type='map')
        correlation = DataObj(data=correlation_data, data_type='map')

        return dict(delay=delay, intercept=intercept, slope=slope, correlation=correlation)

    if reference.data_type == 'timecourse':

        if shifts_option is not None:

            # shifts_option is then either a np array or int
            if not type(shifts_option) == int:
                # level 3
                # prepare arrays to search maximum correlation
                number_of_shifts = len(shifts_option)
                slope = dict()
                intercept = dict()
                correlation = dict()

                # iterate over all values of the array
                for shift in shifts_option:
                    shift = int(shift)
                    slope[shift], intercept[shift], correlation[shift], p, std = compute_delays(reference, probe, shift)

                # extract maximum and return corresponding values

                max_loc = max(correlation, key=correlation.get)

                return max_loc, slope[max_loc], intercept[max_loc], correlation[max_loc]
            else:
                # level 2
                # shift the probe signal and then proceed with the fit
                # here shifts_option is an int used to build keys in the dict shifted_probes
                if shifts_option not in shifted_probes:
                    shifted_probes[shifts_option] = build_shifted_signal(probe, reference, shifts_option)

                return compute_delays(reference, shifted_probes[shifts_option], None)
        else:
            # level 1
            return scipy.stats.linregress(probe.data, reference.data)


def compute_response(intercept, slope, regressorbaseline, regressormean):
    """
        This function computes slope/(intercept + (regressorbaseline)*slope)

    """
    intercept_data = intercept.data
    slope_data = slope.data
    n_x, n_y, n_z = intercept_data.shape
    response_data = np.zeros([n_x, n_y, n_z])

    # script_progress_sentence = "Computing response..."

    total_voxels = n_x*n_y*n_z
    loop_counter = 0
    # printProgressBar(0, total_voxels, prefix=script_progress_sentence)

    for i_z in range(n_z):
        for i_x in range(n_x):
            for i_y in range(n_y):
                loop_counter += 1
                if np.isnan(intercept_data[i_x, i_y, i_z]):
                    response_data[i_x, i_y, i_z] = float('nan')
                    # response_data[i_x, i_y, i_z] = 0.
                else:
                    denominator = intercept_data[i_x, i_y, i_z] + regressorbaseline*slope_data[i_x, i_y, i_z]
                    if not denominator == 0:
                        response_data[i_x, i_y, i_z] = 100*slope_data[i_x, i_y, i_z]/denominator
                    else:
                        response_data[i_x, i_y, i_z] = float('nan')
                        # response_data[i_x, i_y, i_z] = 0.

        # printProgressBar(loop_counter, total_voxels, prefix=script_progress_sentence)

    response = DataObj(data=response_data, data_type='map')

    return response
