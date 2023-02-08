#!/usr/bin/env python3

# imports

from datetime import datetime
import math
import matplotlib.pyplot as plt
import numpy
import numpy as np
import scipy.signal
import scipy.stats

# classes

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    ERROR = BOLD + RED
    INFO = BOLD + GREEN
    WARNING = BOLD + YELLOW


class DataObj:
    """Class to keep together data with sampling freq and other useful information on the data

    For most applications, we need the probe or fMRI data together with
    the corresponding sampling frequency. data_type must be "timecourse" or
    or "map" or "bold". Sampling frequency in Hz.
    label is a field used to put some human-readable struff to describe what the object is containing, e.g. 'this is the denoised data'
    fig is a figure object from plotly
    measurement_type is the type of the actual values, like 'CVR' or 'delay', or 'concentration'.
    mask is a binary map, can be used to mask data in the case of 'map' or 'bold'data_type.
    """
    def __init__(self, data=None, sampling_frequency=None, data_type=None, path=None, label=None, figs=None, measurement_type=None, mask=None, baseline=None):
        self.sampling_frequency = sampling_frequency
        self.data = data
        self.data_type = data_type
        self.path = path
        self.label = label
        self.figs = figs
        self.measurement_type = measurement_type
        self.mask = mask
        self.baseline = baseline

    def bids_load(self, layout, filters, data_type, **kwargs):
        """Load DataObj from a BIDS layout and filters

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
            else:
                data_path = layout.get(**filters, extension='nii.gz')[0]
                data = nibabel.load(data_path).get_fdata()
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

    def nifti_save(self, path_to_ref, path_to_save):
        """
        Save data to path in nifti format using affine from ref
        path_to_ref: path to nifti file to extract affine data
        path_to_save: path where to save the nifti image
        """
        import nibabel
        if self.data_type == 'map' or self.data_type == 'bold':
            img = nibabel.Nifti1Image(self.data, nibabel.load(path_to_ref).affine)
            nibabel.save(img, path_to_save)
            if self.path == None:
                self.path = path_to_save

    def timecourse_save(self, path_to_save):
        """
        Save timecourse data to path in .tsv.gz format - path_to_save must be something live /my/file/is/here and this function will write /my/file/is/here.tsv.gz and /my/file/is/here.json for timecourses
        path_to_save: path where to save the data
        """
        import numpy
        import gzip
        import os
        import json
        if self.data_type == 'timecourse':
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

            # add path to class to keep track of things
            if self.path == None:
                self.path = path_to_save

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
            self.baseline = round(numpy.mean(peakutils.baseline(self.data)), 6)


class Report:
    """
    A class to create, update, modify and save reports with pretty stuff
    """
    def __init__(self, path=None, string=""):
        """
        path is the place where the report is written/updated
        """
        self.path = path
        self.string = string

    def init(self, subject, date_and_time, version,
             cmd):
        """
        Init the report with html headers and various information on the report
        """
        intro_part1 = """<!DOCTYPE html>
            <html>
            <body>
            <h1>CVRmap: individual report</h1>
            <h2>Summary</h2>
            <div>
            <ul>
            <li>BIDS participant label: sub-001</li>
            <li>Date and time: 2022-06-01 13:54:41.301278</li>
            <li>CVRmap version: dev
            </li>
            <li>Command line options:
            <pre>
            <code>
            """
        intro_part2 = """</code>
            </pre>
            </li>
            </ul>
            </div>
        """
        self.string.join(intro_part1 + str(cmd) + '\n' + intro_part2)
        with open(self.path, "w") as f:
            f.write('<!DOCTYPE html>\n')
            f.write('<html>\n')
            f.write('<body>\n')
            f.write('<h1>CVRmap: individual report</h1>\n')
            f.write('<h2>Summary</h2>\n')
            f.write('<div>\n')
            f.write('<ul>\n')
            f.write(
                '<li>BIDS participant label: sub-%s</li>\n' % subject)
            f.write('<li>Date and time: %s</li>\n' % date_and_time)
            f.write('<li>CVRmap version: %s</li>\n' % version)
            f.write('<li>Command line options:\n')
            f.write('<pre>\n')
            f.write('<code>\n')
            f.write(str(cmd) + '\n')
            f.write('</code>\n')
            f.write('</pre>\n')
            f.write('</li>\n')
            f.write('</ul>\n')
            f.write('</div>\n')

    def add_section(self, title):
        """
        Add a section to the report
        """
        with open(self.path, "a") as f:
            f.write('<h2>%s</h2>\n' % title)

    def add_subsection(self, title):
        """
        Add a subsection to the report
        """
        with open(self.path, "a") as f:
            f.write('<h3>%s</h3>\n' % title)

    def add_sentence(self, sentence):
        """
        Add a sentence to the report
        """
        with open(self.path, "a") as f:
            f.write('%s<br>\n' % sentence)

    def add_image(self, fig):
        """
        Add an image path to the report

        The image is embedded in the hmtl report using base64
        fig is a Figure object from plotly
        """
        import base64
        # with open(self.path, "ab") as f:
        #   f.write(b'<img src = "data:image/png;base64, ')
        #   f.write(base64.b64encode(fig.to_image('png')))
        #   f.write(fig.to_html(full_html=False))
        #   f.write(b'" >')
        #   f.write(b'<br>')

        with open(self.path, "a") as f:
            f.write(fig.to_html(full_html=False))
            f.write('<br>')


    def finish(self):
        """
        Writes the last lines of the report to finish it.
        """
        with open(self.path, "a") as f:
            f.write('</body>\n')
            f.write('</html>\n')


# functions

def msg_info(msg):
    now = datetime.now()
    time_stamp = now.strftime("(%Y-%m-%d-%H-%M-%S) ")
    # print('(py) ' + f"{bcolors.INFO}INFO{bcolors.ENDC} " + time_stamp + msg, flush=True)
    print(f"{bcolors.INFO}INFO{bcolors.ENDC} " + time_stamp + msg, flush=True)


def msg_error(msg):
    now = datetime.now()
    time_stamp = now.strftime("(%Y-%m-%d-%H-%M-%S) ")
    # print('(py) ' + f"{bcolors.ERROR}ERROR{bcolors.ENDC} " + time_stamp + msg, flush=True)
    print(f"{bcolors.ERROR}ERROR{bcolors.ENDC} " + time_stamp + msg, flush=True)


def msg_warning(msg):
    now = datetime.now()
    time_stamp = now.strftime("(%Y-%m-%d-%H-%M-%S) ")
    # print('(py) ' + f"{bcolors.WARNING}WARNING{bcolors.ENDC} " + time_stamp + msg, flush=True)
    print(f"{bcolors.WARNING}WARNING{bcolors.ENDC} " + time_stamp + msg, flush=True)


def printProgressBar (iteration, total, prefix = '', suffix = 'Complete', decimals = 4, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    complete_prefix = f"{bcolors.OKCYAN}Progress {bcolors.ENDC}" + prefix
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{complete_prefix} |{bar}| {percent}% {suffix}', end = printEnd, flush="True")
    # Print New Line on Complete
    if iteration == total:
        print()


def save_lightbox(data, path, **options):
    # OBSELETE
    axis = 'z'
    numberOfSlices = 16

    for key in options.keys():
        if key == 'type':
            if options[key] == 'cvr':
                colormaps = ['autumn', 'winter']
                v_range = [0, 3]
            elif options[key] == 'delay':
                colormaps = ['autumn', 'winter']
                v_range = [0, 16]

    fig = plt.figure(figsize=(20, 20))

    # todo: discard the first and the last couple of slices as they are uninteresting
    # todo: add anatomical scan with transparency

    axis_dict = {'x' : 0, 'y' : 1, 'z' : 2}
    n_slices = data.shape[axis_dict[axis]]
    if numberOfSlices >= n_slices:
        span = numpy.arange(n_slices)
    else:
        step = math.floor(n_slices/(numberOfSlices + 1))
        span = numpy.arange(step, n_slices - step, step)

    for i in range(len(span)-1):
        plt.subplot(math.ceil(len(span)/4), 4, i+1)
        if axis == 'x':
            rawdata = data.get_fdata()[span[i], :, ::-1]
        elif axis == 'y':
            rawdata = data.get_fdata()[:, span[i], ::-1]
        elif axis == 'z':
            rawdata = data.get_fdata()[:, ::-1, span[i]]

        # todo: fix some display issues with the colour bars

        plt.title(axis + ' = ' + str(span[i]))
        plt.imshow(numpy.transpose(numpy.where(rawdata>=0, rawdata, float("nan"))), colormaps[0], interpolation='none', vmin=v_range[0], vmax=v_range[1])
        plt.colorbar().set_label('positive')
        plt.imshow(numpy.transpose(numpy.where(rawdata<0, -rawdata, float("nan"))), colormaps[1], interpolation='none', vmin=v_range[0], vmax=v_range[1])
        plt.colorbar().set_label('negative')

    plt.savefig(path, dpi=fig.dpi, bbox_inches='tight')
    plt.close()


def gather_figures(objs):
    """
    Create fig from several DataObj of 'timecourse' type by superimposing their individual plots
    If two objects are given, the first one will have it's y-axis on the left and the second on the right.
    """

    if len(objs) == 2:
        from plotly.subplots import make_subplots
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        counter = 0
        for obj in objs:
            if counter == 0:
                secondary_y = True
            else:
                secondary_y = False
            fig.add_trace(obj.figs['plot'].data[0], secondary_y=secondary_y)
            plot_title = obj.figs['plot'].layout.title.text
            plot_units = obj.figs['plot'].layout.yaxis.title.text
            title_text = plot_title + ', ' + plot_units
            fig.update_yaxes(title_text=title_text, secondary_y=secondary_y)
            counter = counter + 1
    else:
        import plotly.graph_objects as go
        fig = go.Figure()
        for obj in objs:
            fig.add_trace(obj.figs['plot'].data[0])
    return fig


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
        # before we finish, we discard points in the delay map having correlation < 0.6, but we keep them for intercept and slope:
        delay_data = np.where(correlation_data > 0.6, delay_data, float('nan'))
        #slope_data = np.where(correlation_data > 0.6, slope_data, float('nan'))
        #intercept_data = np.where(correlation_data > 0.6, intercept_data, float('nan'))

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


def compute_response(intercept, slope, regressorbaseline, regressormean):
    """
    This function computes slope/(intercept + (regressorbaseline-regressormean)*slope)
    """
    intercept_data = intercept.data
    slope_data = slope.data
    n_x, n_y, n_z = intercept_data.shape
    response_data = np.zeros([n_x, n_y, n_z])

    script_progress_sentence = "Computing response..."

    total_voxels = n_x*n_y*n_z
    loop_counter = 0
    printProgressBar(0, total_voxels, prefix=script_progress_sentence)

    for i_z in range(n_z):
        for i_x in range(n_x):
            for i_y in range(n_y):
                loop_counter += 1
                if np.isnan(intercept_data[i_x, i_y, i_z]):
                    response_data[i_x, i_y, i_z] = float('nan')
                else:
                    denominator = intercept_data[i_x, i_y, i_z] + (regressorbaseline - regressormean)*slope_data[i_x, i_y, i_z] # CORRECT FORMULA IS HERE
                    #denominator = intercept_data[i_x, i_y, i_z] + (regressorbaseline - 0*regressormean) * slope_data[
                    #    i_x, i_y, i_z]  # REMOVED regressormen IN DENOMINATOR
                    if not denominator == 0:
                        response_data[i_x, i_y, i_z] = 100*slope_data[i_x, i_y, i_z]/denominator
                    else:
                        response_data[i_x, i_y, i_z] = float('nan')

        printProgressBar(loop_counter, total_voxels, prefix=script_progress_sentence)

    # msg_info("Output values are PERCENTAGES")
    response = DataObj(data=response_data, data_type='map')

    return response
