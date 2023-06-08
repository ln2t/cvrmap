"""
Preprocessing tools for physiological and fMRI data
"""

# IMPORTS

from .processing import DataObj
import numpy as np
from scipy.ndimage import gaussian_filter
import peakutils
from scipy.interpolate import interp1d
from statsmodels.regression.linear_model import OLS

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

    probe = DataObj(data=etco2, sampling_frequency=sampling_freq, data_type='timecourse', label=r'$\text{etCO}_2\text{timecourse}$', units=physio.units)
    baseline = DataObj(data=np.mean(baseline_data)*np.ones(len(baseline_data)), sampling_frequency=sampling_freq, data_type='timecourse', label=r'$\text{etCO}_2\text{ baseline}$')

    return probe, baseline


def denoise(bold_fn, mask_fn, melodic_mixing_df, noise_indexes, fwhm=None):
    """
    Loops over all voxel not in mask for non_agg_denoise, which does what fsl_regfilt does.
    Args:
        bold_fn: path to 4D nii to denoise
        mask_fn: path to mask
        melodic_mixing_df: pandas df with IC's
        noise_indexes: list of int labelling the noise, 0-based
        fwhm: smoothing parameter. If None, no smoothing is done.

    Returns:
        DataObj containing denoised data. Voxels outside mask as set to 0.
    """
    from nilearn.image import smooth_img
    import nibabel as nb
    bold_img = nb.load(bold_fn)
    bold_data = bold_img.get_fdata()
    mask = nb.load(mask_fn).get_fdata()

    denoised_data = bold_data.copy()

    nx, ny, nz = bold_data.shape[:3]
    for x in np.arange(nx):
        for y in np.arange(ny):
            for z in np.arange(nz):
                if mask[x, y, z]:
                    # extract time series
                    Y = bold_data[x, y, z, :]
                    # denoise
                    denoised_data[x, y, z, :] = non_agg_denoise(Y, melodic_mixing_df, noise_indexes)
                else:
                    denoised_data[x, y, z, :] = int(0)

    denoised = DataObj(data_type='bold')
    denoised.label = 'Non-aggressively denoised data'
    denoised.data = smooth_img(nb.Nifti1Image(denoised_data, bold_img.affine, bold_img.header), fwhm).get_fdata()

    return denoised


def non_agg_denoise(signal, design_matrix_df, noise_indexes):
    """
    This is the in-house implementation of non-aggressive denoising. It is doing the same thing as fsl_regfilt.
    """
    # define model with ALL regressors
    model = OLS(signal, design_matrix_df.values)
    # fit the model
    results = model.fit()
    # get the fitted parameters for the noise components only
    noise_params = results.params[noise_indexes]
    # get the noise part of the design matrix
    noise_values = design_matrix_df[noise_indexes].values
    # remove noise from full signal
    return signal - np.dot(noise_values, noise_params)