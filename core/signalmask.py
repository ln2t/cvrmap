#!/usr/bin/env python3

# IMPORTS

from .utils import msg_info, msg_error
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import nibabel as nb
import peakutils
import argparse
import json

# ARGUMENTS

# initiate the parser
help_message = "This is a python program to extract signal from a given masked fmri timeseries"

def run(input, mask, target):

    # MAIN

    # input data

    input_data = []
    input_img = nb.load(input)
    input_data = input_img.get_fdata()

    n_x, n_y, n_z, n_t = input_img.shape

    mask_data = []
    mask_img = nb.load(mask)
    mask_data = mask_img.get_fdata()

    m_x, m_y, m_z = mask_img.shape

    # check consistency of datasize

    if not([n_x, n_y, n_z] == [m_x, m_y, m_z]):
        msg_error('Data and mask do not have consistent spatial dimensions')
        exit(1)

    # parse whole volume...

    signal = np.zeros(n_t)

    for i_t in range(n_t):
        masked_data = ma.masked_array(input_data[:,:,:,i_t], mask=mask_data)
        signal[i_t] = masked_data.mean()

    # # save the mean signal data in a tsv
    # output_tsv = target + ".tsv"
    # np.savetxt(output_tsv, np.round(signal, 4), '%1.4f') #
    #
    # # save graph
    # fig, ax = plt.subplots()
    # plt.plot(signal)
    # output_graph = target + ".png"
    # plt.savefig(output_graph)

    # msg_info('Graph saved at ' + output_graph)
    # msg_info('Mean signal data saved at ' + output_graph)

    # add computed baseline and mean to json file
    # output_json = output_prefix + ".json"
    #
    # with open(output_json, "r") as json_file:
    #     json_data = json.load(json_file)

    # json_data['Baseline'] = round(np.mean(baseline_data), 3)
    # json_data['Mean'] = round(np.mean(signal), 3)
    #
    # with open(output_json, "w") as json_file:
    #     json.dump(json_data, json_file)

    return signal, np.mean(peakutils.baseline(signal)), np.mean(signal)
