#!/usr/bin/env python3

# IMPORTS

import numpy as np
from scipy.signal import resample
from scipy.stats import pearsonr

help_message = "This is a python tool to compute correlation between two time courses and find optimal delay"

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

# MAIN

# loading data...

def run(data_object1, data_object2):

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

