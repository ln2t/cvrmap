# IMPORTS

from .utils import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import peakutils
from scipy.interpolate import interp1d
import json
import os

# MAIN
# todo: remove this once the new one works well
def run_old(physio_file, sampling_freq, output_prefix):

    load_file = np.loadtxt(physio_file, usecols=1)
    physio_raw = load_file
    n_samples = len(physio_raw)
    time_step = 1/sampling_freq
    total_time = time_step*n_samples
    time_span = np.arange(0, total_time, time_step)

    # msg_info("The total time for the recording is " + str(round(total_time, 2)) + " seconds")

    # the signal needs to be very slightly smoothed, because sometimes the capnograph did weird measurements (see e.g. 005 or 008).

    sigma1 = 0.06  # to take from config file - in seconds
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
    baseline = np.mean(baseline_data)

    fig, ax = plt.subplots()
    plt.figure(1)
    plt.plot(time_span, physio_raw)
    # plt.plot(time_span, physio_gaussian_smooth)
    plt.plot(peak_times, physio_gaussian_smooth1[peak_locs], 'r+')
    plt.plot(etco2_time_span, etco2)
    plt.plot(time_span, baseline*np.ones(n_samples), color='r')

    # save the graph for visual inspection
    output_graph = output_prefix + "_physio.png"
    plt.savefig(output_graph)

    # add computed baseline and mean to json file
    output_json = output_prefix + ".json"

    json_data = {}
    json_data["SamplingFrequency"] = sampling_freq
    json_data["StartTime"] = "-10"
    json_data["Columns"] = "Respiratory"
    json_data["Baseline"] = round(baseline, 3)
    json_data["Mean"] = round(np.mean(etco2), 3)

    with open(output_json, "w") as json_file:
        json.dump(json_data, json_file)

    # save the etco2 data in a tsv
    output_tsv = output_prefix + ".tsv"
    np.savetxt(output_tsv, np.round(etco2, 4), '%1.4f')  # we round at 3 decimals and print at corresponding level

    plt.close()

    return np.round(etco2, 4), json_data["Baseline"], json_data["Mean"], output_graph

def run(physio):
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