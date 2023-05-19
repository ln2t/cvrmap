"""
Preprocessing tools for physiological and fMRI data
"""

# IMPORTS

from .processing import DataObj
import numpy as np
from scipy.ndimage import gaussian_filter
import peakutils
from scipy.interpolate import interp1d

def endtidalextract(physio):
    """Analyse physiological breathing data to extract etco2 curve

    Inputs:
        physio is DataObj
    Returns:
        etco2: DataObj with basically the upper enveloppe of physio.data
        baseline: DataObj with the baseline of the etco2 curve
    """

    sampling_freq = physio.sampling_frequency
    physio_raw = physio.data
    n_samples = len(physio.data)
    time_step = 1/sampling_freq
    total_time = time_step*n_samples
    time_span = np.arange(0, total_time, time_step)

    sigma1 = 0.06  # to take from config file - in seconds todo
    sigma1_sample = sigma1*sampling_freq
    physio_gaussian_smooth1 = gaussian_filter(physio_raw, sigma1_sample)

    # light smoothing of the raw signal: gaussian filtering

    sigma = 0.8  # to take from config file - in seconds
    sigma_sample = sigma*sampling_freq
    physio_gaussian_smooth = gaussian_filter(physio_gaussian_smooth1, sigma_sample)

    # extract upper envelope by finding peaks and interpolating
    # to find the peaks, we use the location of the peaks of the gaussian filtered curve,
    # and then we interpolate the value of the raw signal at these locations

    peak_locs = peakutils.indexes(physio_gaussian_smooth)
    peak_times = time_span[peak_locs]
    etco2_fit = interp1d(peak_times, physio_gaussian_smooth1[peak_locs], 3, bounds_error = False, fill_value=0.0) #kind = "cubic"
    first_peak = peak_locs[0]
    last_peak = peak_locs[-1]
    etco2_time_span = time_span[first_peak:last_peak]
    etco2 = etco2_fit(etco2_time_span)

    baseline_data = peakutils.baseline(etco2)

    probe = DataObj(data=etco2, sampling_frequency=sampling_freq, data_type='timecourse', label=r'$\text{etCO}_2\text{timecourse}$')
    baseline = DataObj(data=np.mean(baseline_data)*np.ones(len(baseline_data)), sampling_frequency=sampling_freq, data_type='timecourse', label=r'$\text{etCO}_2\text{ baseline}$')

    return probe, baseline