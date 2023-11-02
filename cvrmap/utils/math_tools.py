"""
Various math tools
"""

import numpy as np
import scipy
from scipy.signal import resample
from scipy.stats import pearsonr

def match_timecourses(y1, y2, delay):
    """
        The goal of this function is to take two time courses and apply a time shift between the two.
        This is done carefully, in particular ensuring that the returned time courses are of equal length.

    Args:
        y1: np array of length n1
        y2: np array of length n2
        delay: int
    Returns:
        y1_matched, y2_matched: arrays of same length
    """
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
    """
        Computes cross-correlation between two time courses, by exploring variour time shifts. The inputs are DataObj
        of type "timecourse", and come with their own sampling frequency. If the sampling frequencies are different
        (typical in the case of BOLD signal compared to capnograph data), then the signal with higher sf is
        downsampled to the lower one. Delays are explored to find maximum correlation, and r-coefficient at maximum
        returned.

    Args:
        data_object1: DataObj of type "timecourse"
        data_object2: DataObj of type "timecourse"

    Returns:
        float, the maximum r-correlation over the explored delays
    """
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
        The baseline is read from probe.baseline; if undefined, the function first calls probe.build_baseline()
        to ensure it exists.
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
        Computes whole-brain signal

    Args:
        data, a DataObj for some fMRI data

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


def get_meanepi(img):
    """
        Compute temporal mean for fMRI data (calls nilearn.image.mean_img)

    Args:
        img: DataObj for the input fMRI data
    Returns:
        niimg, the mean of fMRI data
    """
    from nilearn.image import mean_img
    return mean_img(img.path)


def get_corrected_noiselist(probe, aroma_noise_ic_list, melodic_mixing_df, sf, noise_ic_pearson_r_threshold, aroma_flag):
    """
        Finds the noise independent components that do not correlate with the probe signal

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