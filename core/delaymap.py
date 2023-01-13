#!/usr/bin/env python3

# IMPORTS

from .utils import *
import numpy as np
from scipy.signal import resample
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families.family import Gaussian as GaussianFamily
from statsmodels.tools import add_constant

# ARGUMENTS

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

def simple_glm_estimation(data, design):
    design = design.copy() - np.mean(design)
    design = add_constant(design)  # without the prepend=False option, this will put the constant term in the 0th position in the param array
    glm = GLM(data, design, family=GaussianFamily())
    res = glm.fit()
    estimated_parameters = res.params
    t_values = res.tvalues
    return estimated_parameters, t_values

def tcglm(tc1, tc2, search_range, samplingfreq):
    tc1_n = len(tc1)
    tc2_n = len(tc2)
    glm_intercept = []
    glm_slope = []
    glm_intercept_tvalues= []
    glm_slope_tvalues = []
    for delay in search_range:
        if tc1_n >= tc2_n:
            tc1_matched, tc2_matched = match_timecourses(tc1, tc2, delay)
        else:
            tc2_matched, tc1_matched = match_timecourses(tc2, tc1, delay)
        param, tval = simple_glm_estimation(tc1_matched, tc2_matched)  # tc1 is considered as the DATA while tc2 is considered as the DESIGN - be careful
        glm_intercept.append(param[0])
        glm_slope.append(param[1])
        glm_intercept_tvalues.append(tval[0])
        glm_slope_tvalues.append(tval[1])
    max_loc = np.argmax(glm_slope_tvalues)
    max_slope = glm_slope[max_loc]
    max_intercept = glm_intercept[max_loc]
    max_slope_tval = glm_slope_tvalues[max_loc]
    max_intercept_tval = glm_intercept_tvalues[max_loc]
    delay_max = search_range[max_loc] / samplingfreq  # in seconds
    return max_intercept, max_slope, max_intercept_tval, max_slope_tval, delay_max


def run(probe, fmri, mask, **options):

    regressor = probe.data
    regressor_samplingfreq = probe.sampling_frequency
    fmri_samplingfreq = fmri.sampling_frequency
    sloppy_flag = 0
    for key in options.keys():
        if key == 'sloppy' and options[key]:
            sloppy_flag = options[key]
            msg_warning("Working in sloppy mode, only for testing!")
            msg_info("Sloppy flag: %s" % sloppy_flag)

    # todo: explore ranges beyond -15,+15
    delay_upper_range = 30 # in seconds
    # delay_lower_range = -delay_upper_range
    delay_lower_range = -5

    script_progress_sentence = "Computing delays....."

    # regressor_data = []
    #
    # with open(regressor_path) as file:
    #     regressor_data = np.asarray([float(x) for x in file.readlines()])
    #
    #     regressor_n = len(regressor_data)
    #
    #     regressor_span = np.arange(regressor_n)/regressor_samplingfreq

    regressor_data = regressor
    regressor_n = len(regressor_data)
    regressor_span = np.arange(regressor_n)/regressor_samplingfreq

    # fmri data

    # load the nifti files - nii.gz is supported
    # fmri_img = nb.load(fmri_path)
    # mask_img = nb.load(mask_path)

    # load the data as np arrays
    fmri_data = fmri.data
    mask_data = mask.data

    # initialize the output maps
    n_x, n_y, n_z, n_t = fmri_data.shape
    delay_map_data = np.zeros([n_x, n_y, n_z])
    # pearson_r_map_data = np.zeros([n_x, n_y, n_z])
    # pearson_p_map_data = np.zeros([n_x, n_y, n_z])

    intercept_map_data = np.zeros([n_x, n_y, n_z])
    slope_map_data = np.zeros([n_x, n_y, n_z])
    intercept_tval_map_data = np.zeros([n_x, n_y, n_z])
    slope_tval_map_data = np.zeros([n_x, n_y, n_z])

    delay_map_data[:] = np.nan
    intercept_map_data[:] = np.nan
    slope_map_data[:] = np.nan
    intercept_tval_map_data[:] = np.nan
    slope_tval_map_data[:] = np.nan

    # regressor resampling...

    # up- or downsample the regressor data to fit fmri sampling frequency

    if regressor_samplingfreq != fmri_samplingfreq:
        sampling_factor = regressor_samplingfreq / fmri_samplingfreq
        regressor_data = resample(regressor_data, int(regressor_n/sampling_factor))

    samplingfreq = fmri_samplingfreq

    regressor_n = len(regressor_data)
    regressor_span = np.arange(regressor_n)/regressor_samplingfreq

    # parse whole volume...

    delay_range = np.arange(delay_lower_range*samplingfreq, (delay_upper_range + 1)*samplingfreq)

    n_x_range = range(n_x)
    n_y_range = range(n_y)
    n_z_range = range(n_z)

    if sloppy_flag == 1:  # compute the full middle slice
        n_x_range = np.arange(int(0.9*n_x/2), int(1.1*n_x/2))
        n_y_range = np.arange(int(0.9*n_y/2), int(1.1*n_y/2))
        n_z_range = np.arange(int(n_z/2), int(n_z/2) + 1)
        n_x = len(n_x_range)
        n_y = len(n_y_range)
        n_z = len(n_z_range)

    if sloppy_flag == 2:  # compute only a few slices
        # n_x_range = np.arange(int(0.9*n_x/2), int(1.1*n_x/2))
        # n_y_range = np.arange(int(0.9*n_y/2), int(1.1*n_y/2))
        n_x_range = np.arange(1, n_x, int(n_x / 5))
        n_y_range = np.arange(1, n_y, int(n_y / 5))
        n_z_range = np.arange(1, n_z, int(n_z/5))
        n_x = len(n_x_range)
        n_y = len(n_y_range)
        n_z = len(n_z_range)

    if sloppy_flag == 3:  # compute only a small part of the middle slice
        n_x_range = np.arange(int(0.9*n_x/2), int(1.1*n_x/2))
        n_y_range = np.arange(int(0.9*n_y/2), int(1.1*n_y/2))
        n_z_range = np.arange(int(n_z/2), int(n_z/2) + 1)
        n_x = len(n_x_range)
        n_y = len(n_y_range)
        n_z = len(n_z_range)

    total_voxels = n_x*n_y*n_z
    loop_counter = 0
    printProgressBar(0, total_voxels, prefix=script_progress_sentence)

    for i_z in n_z_range:
        for i_x in n_x_range:
            for i_y in n_y_range:
                loop_counter += 1
                printProgressBar(loop_counter, total_voxels, prefix=script_progress_sentence)
                if mask_data[i_x, i_y, i_z] > 0:
                    intercept_map_data[i_x, i_y, i_z], slope_map_data[i_x, i_y, i_z], intercept_tval_map_data[i_x, i_y, i_z], slope_tval_map_data[i_x, i_y, i_z], delay_map_data[i_x, i_y, i_z] = tcglm(fmri_data[i_x, i_y, i_z, :], regressor_data, delay_range, samplingfreq)

    delay = DataObj(data=delay_map_data, data_type='map')
    intercept = DataObj(data=intercept_map_data, data_type='map')
    slope = DataObj(data=slope_map_data, data_type='map')

    return delay, intercept, slope
