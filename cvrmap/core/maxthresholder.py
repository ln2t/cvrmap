#!/usr/bin/env python3

# IMPORTS

from .utils import msg_info, msg_error
import os
import numpy as np
import nibabel as nb

# ARGUMENTS

# initiate the parser
help_message = "This is a python program to create a mask obtained by thresholding a map from its maximum value"

def run(input, target, threshold_pc):

    # input data

    input_data = []
    input_img = nb.load(input)
    input_data = input_img.get_fdata()

    # initialize the output map

    n_x, n_y, n_z = input_img.shape
    output_data = np.zeros([n_x, n_y, n_z])

    # parse whole volume...

    global_min = np.nanmin(input_data)
    global_max = np.nanmax(input_data)

    threshold_val = threshold_pc*(global_min-global_max)+global_max
    # 
    # msg_info('Global maximum is at ' + str(global_max))
    # msg_info('Global minimum is at ' + str(global_min))
    # msg_info('Threshold is then at ' + str(threshold_val))

    for i_z in range(n_z):
        for i_x in range(n_x):
            for i_y in range(n_y):
                if input_data[i_x, i_y, i_z] > threshold_val:
                    output_data[i_x, i_y, i_z] = 1

    output_img = nb.Nifti1Image(output_data, input_img.affine)
    nb.save(output_img, target)

    return 0
