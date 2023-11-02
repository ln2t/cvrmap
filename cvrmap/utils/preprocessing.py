"""
Preprocessing tools for physiological and fMRI data
"""

from .processing import DataObj
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d


def endtidalextract(physio):
    """
        Analyse physiological breathing data to extract etco2 curve

    Inputs:
        physio is DataObj
    Returns:
        etco2: DataObj with basically the upper enveloppe of physio.data
        baseline: DataObj with the baseline of the etco2 curve
    """

    import peakutils

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


def masksignalextract(preproc, mask):
    """
        Return time series from masked fMRI data

    Args:
        preproc: DataObj, fMRI data
        mask: str, path to mask

    Returns:
        probe: DataObj, containing the extracted time course
        baseline: DataObj, baseline of probe

    """
    from nilearn.masking import apply_mask
    import peakutils

    sampling_freq = preproc.sampling_frequency
    data = apply_mask(imgs=preproc.path, mask_img=mask)
    masksignal = np.mean(data, axis=-1)
    masksignal = masksignal/np.std(masksignal)
    baseline_data = peakutils.baseline(masksignal)
    probe = DataObj(data=masksignal, sampling_frequency=sampling_freq, data_type='timecourse', units='BOLD')
    baseline = DataObj(data=np.mean(baseline_data)*np.ones(len(baseline_data)), sampling_frequency=sampling_freq, data_type='timecourse', units='BOLD')
    return probe, baseline


def bold_denoising(bold_fn, mask_fn, melodic_mixing_df, noise_indexes, parameters):
    """
        BOLD signal denoising based on non-aggressive denoising.

    Steps:
        - non-aggressive denoising (similar to what fsl_regfilt do)
        - highpass filtering (similar to fslmaths -bptf), using parameters['highpass_frequency'] (value in Hz)
        - re-add mean BOLD signal to filtered data
        - spatial smoothing using parameters['fwhm']

    Note: this function does not rely on FSL.

    Args:
        bold_fn: path to 4D nii to denoise
        mask_fn: path to BOLD mask
        melodic_mixing_df: pandas df with IC's
        noise_indexes: list of int labelling the noise, 0-based
        parameters: dict, whose elements 'fwhm' and 'highpass_frequency' only are used.
        If parameters['fwhm'] is None, no smoothing is applied. parameters['highpass_frequency'] is passed to
        nilean.image.clean_img.

    Returns:
        DataObj, containing denoised data. Voxels outside mask as set to 0.
    """
    from nilearn.image import clean_img, smooth_img, mean_img
    import nibabel as nb
    bold_img = nb.load(bold_fn)
    bold_data = bold_img.get_fdata()
    mask = nb.load(mask_fn).get_fdata()

    _img_data = bold_data.copy()

    # non-aggressive denoising

    nx, ny, nz = bold_data.shape[:3]
    for x in np.arange(nx):
        for y in np.arange(ny):
            for z in np.arange(nz):
                if mask[x, y, z]:
                    # extract time series
                    Y = bold_data[x, y, z, :]
                    # denoise
                    _img_data[x, y, z, :] = non_agg_denoise(Y, melodic_mixing_df, noise_indexes)
                else:
                    _img_data[x, y, z, :] = int(0)

    _img = nb.Nifti1Image(_img_data, bold_img.affine, bold_img.header)

    # get mean for later use

    _mean_img = mean_img(_img)

    # highpass filter

    t_r = _img.header.get_zooms()[-1]
    _img = clean_img(imgs=_img, standardize=False, detrend=False, high_pass=parameters['highpass_frequency'], t_r=t_r)

    # re-add constant term

    _img = add_cst_img_to_series(_img, _mean_img)

    # spatial smoothing

    _img = smooth_img(_img, parameters['fwhm'])

    # finish

    denoised = DataObj(data_type='bold')
    denoised.label = 'Non-aggressively denoised data'
    denoised.data = _img.get_fdata()
    denoised.img = _img

    return denoised


def add_cst_img_to_series(img, cst_img):
    """
        Take a 4D niimg and a 3D niimg and voxel-wise adds the value of the 3D img to the timeseries of the 4D img.

    Args:
        img: niimg, representing a 4D nifti
        cst_img: niimg, representing a 3D nifti

    Returns:
        niimg, sum of the 3D and the 4D img as described above.

    """
    import nibabel as nb
    _img_data = img.get_fdata().copy()
    _cst_data = cst_img.get_fdata()

    nx, ny, nz, nt = _img_data.shape
    for x in np.arange(nx):
        for y in np.arange(ny):
            for z in np.arange(nz):
                for t in np.arange(nt):
                    _img_data[x, y, z, t] = _cst_data[x, y, z] + _img_data[x, y, z, t]

    return nb.Nifti1Image(_img_data, img.affine, img.header)


def non_agg_denoise(signal, design_matrix_df, noise_indexes):
    """
        This is the in-house implementation of non-aggressive denoising. It is doing the same thing as fsl_regfilt.
        This is work in progress and is not used in the main script.

    """

    from statsmodels.regression.linear_model import OLS

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
