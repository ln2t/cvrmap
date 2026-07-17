"""
ROI-based Probe Extraction for CVRmap

This module provides functionality to extract probe signals from brain ROIs as an alternative 
to physiological recordings for CVR analysis.

The ROI-based approach allows CVR analysis when:
1. Physiological recordings are not available or of poor quality
2. Alternative probe regions are of interest (e.g., specific vascular territories)
3. Validation studies comparing different probe approaches are needed

Key Features:
- ROI-based probe extraction from BOLD data
- Integration with existing CVRmap pipeline
- Support for various ROI definition methods (coordinates, masks, atlases)
- Automatic signal processing and normalization
- Optional bandpass filtering for isolating CVR-relevant frequency bands
"""

import numpy as np
import nibabel as nib
import os
import glob
from scipy.signal import butter, filtfilt


def apply_bandpass_filter(signal, sampling_frequency, highpass=None, lowpass=None, order=5, logger=None):
    """
    Apply bandpass filter to a probe signal using a Butterworth filter.
    
    This function applies a zero-phase bandpass filter (using filtfilt) to isolate
    frequency components of interest in the ROI probe signal. It's particularly
    useful for extracting low-frequency fluctuations relevant to CVR analysis.
    
    After filtering, the DC component (mean) of the original signal is restored
    to ensure compatibility with baseline computation methods.
    
    Parameters:
    -----------
    signal : numpy.ndarray
        1D array containing the probe signal to filter
    sampling_frequency : float
        Sampling frequency of the signal in Hz
    highpass : float, optional
        Highpass cutoff frequency in Hz. Frequencies below this are attenuated.
        If None, no highpass filtering is applied.
    lowpass : float, optional
        Lowpass cutoff frequency in Hz. Frequencies above this are attenuated.
        If None, no lowpass filtering is applied.
    order : int, optional
        Order of the Butterworth filter (default: 5). Higher orders give sharper cutoffs.
    logger : Logger, optional
        Logger instance for debug messages
        
    Returns:
    --------
    numpy.ndarray
        Filtered signal with same shape as input, with DC component restored
        
    Raises:
    -------
    ValueError
        If both highpass and lowpass are None, or if filter parameters are invalid
        
    Notes:
    ------
    - Uses scipy.signal.filtfilt for zero-phase filtering (no phase distortion)
    - Recommended cutoff frequencies for CVR analysis: highpass=0.02 Hz, lowpass=0.04 Hz
    - The Nyquist frequency (sampling_frequency/2) limits the valid cutoff range
    - The DC component is restored after filtering to maintain compatibility with
      baseline computation methods (mean or peakutils)
    """
    import numpy as np
    
    if highpass is None and lowpass is None:
        raise ValueError("At least one of highpass or lowpass must be specified for bandpass filtering")
    
    nyquist = sampling_frequency / 2.0
    
    # Validate cutoff frequencies
    if highpass is not None and highpass >= nyquist:
        raise ValueError(f"Highpass frequency ({highpass} Hz) must be less than Nyquist frequency ({nyquist} Hz)")
    if lowpass is not None and lowpass >= nyquist:
        raise ValueError(f"Lowpass frequency ({lowpass} Hz) must be less than Nyquist frequency ({nyquist} Hz)")
    if highpass is not None and lowpass is not None and highpass >= lowpass:
        raise ValueError(f"Highpass frequency ({highpass} Hz) must be less than lowpass frequency ({lowpass} Hz)")
    
    # Store the original DC component (mean) before filtering
    dc_component = np.mean(signal)
    
    # Determine filter type and cutoffs
    if highpass is not None and lowpass is not None:
        # Bandpass filter
        btype = 'bandpass'
        Wn = [highpass / nyquist, lowpass / nyquist]
        filter_desc = f"bandpass [{highpass}-{lowpass}] Hz"
    elif highpass is not None:
        # Highpass only
        btype = 'highpass'
        Wn = highpass / nyquist
        filter_desc = f"highpass {highpass} Hz"
    else:
        # Lowpass only
        btype = 'lowpass'
        Wn = lowpass / nyquist
        filter_desc = f"lowpass {lowpass} Hz"
    
    if logger:
        logger.debug(f"Applying {filter_desc} Butterworth filter (order={order})")
    
    # Design Butterworth filter
    b, a = butter(order, Wn, btype=btype)
    
    # Apply zero-phase filter
    filtered_signal = filtfilt(b, a, signal)
    
    # Restore the DC component (mean) after filtering
    # This is important because bandpass/highpass filters remove the DC component,
    # which would cause issues with baseline computation methods
    filtered_signal = filtered_signal + dc_component
    
    if logger:
        logger.info(f"Applied {filter_desc} filter to probe signal (DC component restored)")
    
    return filtered_signal


def resolve_roi_mask_pattern(pattern, fmriprep_dir, participant, task, space, logger=None):
    """
    Resolve ROI mask pattern to actual file path by searching in fMRIPrep derivatives.
    
    Parameters:
    -----------
    pattern : str
        Pattern with wildcards/regex (e.g., "sub-*_task-*_brain*")
    fmriprep_dir : str
        Path to fMRIPrep derivatives directory
    participant : str
        Participant ID (without 'sub-' prefix)
    task : str
        Task name
    space : str
        Space entity
    logger : logging.Logger, optional
        Logger for messages
        
    Returns:
    --------
    str
        Resolved file path
        
    Raises:
    ------
    FileNotFoundError
        If no matching file is found or multiple files match
    """
    # Check if pattern contains wildcards (*, ?, [])
    if not any(char in pattern for char in ['*', '?', '[']):
        # No wildcards, return as-is (existing behavior)
        return pattern
    
    # Import bids here to avoid circular imports
    from bids import BIDSLayout
    import re
    
    # Create BIDSLayout for fMRIPrep derivatives
    try:
        layout = BIDSLayout(fmriprep_dir, validate=False)
    except Exception as e:
        raise FileNotFoundError(f"Could not create BIDSLayout for fMRIPrep directory '{fmriprep_dir}': {e}")
    
    # Get all files for this subject, task, and space from fMRIPrep derivatives
    try:
        candidate_files = layout.get(
            subject=participant,
            task=task,
            space=space,
            return_type='filename',
            extension=['.nii', '.nii.gz']
        )
    except Exception as e:
        if logger:
            logger.warning(f"Error querying BIDSLayout: {e}")
        candidate_files = []
    
    if logger:
        logger.debug(f"Found {len(candidate_files)} candidate files for subject={participant}, task={task}, space={space}")
        logger.debug(f"Candidate files: {candidate_files}")
    
    if not candidate_files:
        raise FileNotFoundError(
            f"No files found in fMRIPrep derivatives for "
            f"participant {participant}, task {task}, space {space}"
        )
    
    # Convert pattern to regex
    # Replace BIDS-style wildcards with regex equivalents
    regex_pattern = pattern
    regex_pattern = regex_pattern.replace('*', '.*')  # * becomes .*
    regex_pattern = regex_pattern.replace('?', '.')   # ? becomes .
    
    # Ensure pattern matches the full filename (not just part of it)
    if not regex_pattern.startswith('^'):
        regex_pattern = '^.*' + regex_pattern
    if not regex_pattern.endswith('$'):
        regex_pattern = regex_pattern + '.*$'
    
    if logger:
        logger.debug(f"Original pattern: {pattern}")
        logger.debug(f"Regex pattern: {regex_pattern}")
    
    # Filter candidate files using regex pattern
    matched_files = []
    try:
        regex_compiled = re.compile(regex_pattern, re.IGNORECASE)
        for file_path in candidate_files:
            # Extract just the filename for pattern matching
            filename = os.path.basename(file_path)
            if regex_compiled.match(filename):
                matched_files.append(file_path)
                if logger:
                    logger.debug(f"Pattern matched: {filename}")
            else:
                if logger:
                    logger.debug(f"Pattern did not match: {filename}")
    except re.error as e:
        raise ValueError(f"Invalid regex pattern '{regex_pattern}' derived from '{pattern}': {e}")
    
    if not matched_files:
        raise FileNotFoundError(
            f"No files matching pattern '{pattern}' found among {len(candidate_files)} candidate files "
            f"for participant {participant}, task {task}, space {space}"
        )
    
    if len(matched_files) > 1:
        if logger:
            logger.warning(f"Multiple files found matching pattern '{pattern}': {matched_files}")
            logger.warning(f"Using the first match: {matched_files[0]}")
        # Sort to ensure consistent selection
        matched_files.sort()
    
    resolved_path = matched_files[0]
    
    if logger:
        logger.info(f"Resolved ROI mask pattern '{pattern}' to: {resolved_path}")
    
    return resolved_path
from .data_container import ProbeContainer


class ROIProbeExtractor:
    """
    Extracts probe signals from brain ROIs for CVR analysis.
    
    This class provides methods to extract time-series signals from specified brain regions
    that can be used as probes for CVR analysis, replacing physiological recordings.
    """
    
    def __init__(self, bold_container, logger=None, config=None):
        """
        Initialize ROI probe extractor.
        
        Parameters:
        -----------
        bold_container : BoldContainer
            Container with preprocessed BOLD data
        logger : Logger, optional
            Logger instance for debugging and info messages
        config : dict, optional
            Configuration dictionary for ROI extraction parameters
        """
        self.bold_container = bold_container
        self.logger = logger
        self.config = config if config is not None else {}
        
        # Default ROI extraction parameters
        self.roi_method = self.config.get('roi_probe', {}).get('method', 'coordinates')
        self.roi_radius = self.config.get('roi_probe', {}).get('radius_mm', 6.0)
        
        if self.logger:
            self.logger.info(f"ROI probe extractor initialized with method: {self.roi_method}")
    
    def extract_probe_from_coordinates(self, coordinates_mm, radius_mm=None):
        """
        Extract probe signal from spherical ROI around specified coordinates.
        
        Parameters:
        -----------
        coordinates_mm : tuple or list
            (x, y, z) coordinates in mm (world space)
        radius_mm : float, optional
            Radius of spherical ROI in mm. If None, uses config default.
            
        Returns:
        --------
        ProbeContainer
            Container with extracted ROI probe signal
        """
        if radius_mm is None:
            radius_mm = self.roi_radius
            
        if self.logger:
            self.logger.info(f"Extracting ROI probe from coordinates {coordinates_mm} with radius {radius_mm}mm")
        
        # Convert world coordinates to voxel coordinates
        affine = self.bold_container.affine
        inv_affine = np.linalg.inv(affine)
        coords_vox = nib.affines.apply_affine(inv_affine, coordinates_mm)
        coords_vox = np.round(coords_vox).astype(int)
        
        if self.logger:
            self.logger.debug(f"World coordinates {coordinates_mm} → voxel coordinates {coords_vox}")
        
        # Create spherical ROI mask
        roi_mask = self._create_spherical_mask(coords_vox, radius_mm)
        
        # Extract probe signal from ROI
        probe_signal = self._extract_signal_from_mask(roi_mask)
        
        # Create ProbeContainer
        probe_container = ProbeContainer(
            participant=self.bold_container.participant,
            task=self.bold_container.task,
            data=probe_signal,
            physio_metadata=None,
            path=None,  # No physical file for ROI-derived signal
            sampling_frequency=self.bold_container.sampling_frequency,
            units="BOLD_signal",
            layout=self.bold_container.layout,
            logger=self.logger
        )
        
        # Set probe type to indicate ROI source
        probe_container.probe_type = "roi_probe"
        
        # Compute baseline for CVR calculations using configured method
        baseline_method = self.config.get('physio', {}).get('baseline_method', 'peakutils')
        
        if self.logger:
            self.logger.info(f"Computing ROI probe baseline using method: '{baseline_method}'")
        
        if baseline_method == 'mean':
            # Use the mean of the signal as baseline (recommended for resting-state)
            probe_container.baseline = float(np.mean(probe_signal))
            if self.logger:
                self.logger.info(f"Computed ROI probe baseline using MEAN method: {probe_container.baseline:.6f} BOLD units")
        else:
            # Use peakutils to detect baseline from signal troughs (default, recommended for gas challenge)
            import peakutils
            probe_baseline_array = peakutils.baseline(probe_signal)
            probe_container.baseline = float(np.mean(probe_baseline_array))
            if self.logger:
                self.logger.info(f"Computed ROI probe baseline using PEAKUTILS method: {probe_container.baseline:.6f} BOLD units")
        
        if self.logger:
            n_voxels = np.sum(roi_mask)
            self.logger.info(f"ROI probe extracted from {n_voxels} voxels, "
                           f"signal length: {len(probe_signal)} timepoints, "
                           f"baseline: {probe_container.baseline:.3f} BOLD units")
        
        return probe_container
    
    def extract_probe_from_mask(self, mask_path):
        """
        Extract probe signal from binary mask file.
        
        Parameters:
        -----------
        mask_path : str
            Path to binary mask NIfTI file
            
        Returns:
        --------
        ProbeContainer
            Container with extracted ROI probe signal
        """
        if self.logger:
            self.logger.info(f"Extracting ROI probe from mask: {mask_path}")
        
        # Load mask
        mask_img = nib.load(mask_path)
        mask_data = mask_img.get_fdata()
        
        # Check if mask needs resampling to match BOLD data
        if mask_data.shape[:3] != self.bold_container.data.shape[:3]:
            if self.logger:
                self.logger.info(f"Mask shape {mask_data.shape[:3]} differs from BOLD shape {self.bold_container.data.shape[:3]}. Resampling mask to BOLD space.")
            
            # Create a reference NIfTI image from BOLD data for resampling
            bold_ref_img = nib.Nifti1Image(self.bold_container.data[:,:,:,0], self.bold_container.affine)
            
            # Resample mask to BOLD space using nilearn
            try:
                from nilearn.image import resample_to_img
                resampled_mask_img = resample_to_img(mask_img, bold_ref_img, interpolation='nearest')
                mask_data = resampled_mask_img.get_fdata()
                
                if self.logger:
                    self.logger.info(f"Successfully resampled mask from {mask_img.shape[:3]} to {mask_data.shape[:3]}")
            except ImportError:
                raise ImportError("nilearn is required for mask resampling. Please install nilearn: pip install nilearn")
            except Exception as e:
                raise ValueError(f"Failed to resample mask to BOLD space: {e}")
        
        # Convert to boolean mask
        roi_mask = mask_data > 0
        
        # Extract probe signal from ROI
        probe_signal = self._extract_signal_from_mask(roi_mask)
        
        # Create ProbeContainer
        probe_container = ProbeContainer(
            participant=self.bold_container.participant,
            task=self.bold_container.task,
            data=probe_signal,
            physio_metadata=None,
            path=mask_path,
            sampling_frequency=self.bold_container.sampling_frequency,
            units="BOLD_signal",
            layout=self.bold_container.layout,
            logger=self.logger
        )
        
        probe_container.probe_type = "roi_probe"
        
        # Compute baseline using configured method
        baseline_method = self.config.get('physio', {}).get('baseline_method', 'peakutils')
        
        if self.logger:
            self.logger.info(f"Computing ROI probe baseline using method: '{baseline_method}'")
        
        if baseline_method == 'mean':
            # Use the mean of the signal as baseline (recommended for resting-state)
            probe_container.baseline = float(np.mean(probe_signal))
            if self.logger:
                self.logger.info(f"Computed ROI probe baseline using MEAN method: {probe_container.baseline:.6f} BOLD units")
        else:
            # Use peakutils to detect baseline from signal troughs (default, recommended for gas challenge)
            import peakutils
            probe_baseline_array = peakutils.baseline(probe_signal)
            probe_container.baseline = float(np.mean(probe_baseline_array))
            if self.logger:
                self.logger.info(f"Computed ROI probe baseline using PEAKUTILS method: {probe_container.baseline:.6f} BOLD units")
        
        if self.logger:
            n_voxels = np.sum(roi_mask)
            self.logger.info(f"ROI probe extracted from {n_voxels} voxels using mask {mask_path}")
        
        return probe_container
    
    def extract_probe_from_atlas(self, atlas_path, region_id):
        """
        Extract probe signal from atlas region.
        
        Parameters:
        -----------
        atlas_path : str
            Path to atlas NIfTI file
        region_id : int
            Region ID/label in the atlas
            
        Returns:
        --------
        ProbeContainer
            Container with extracted ROI probe signal
        """
        if self.logger:
            self.logger.info(f"Extracting ROI probe from atlas {atlas_path}, region {region_id}")
        
        # Load atlas
        atlas_img = nib.load(atlas_path)
        atlas_data = atlas_img.get_fdata()
        
        # Check if atlas needs resampling to match BOLD data
        if atlas_data.shape != self.bold_container.data.shape[:3]:
            if self.logger:
                self.logger.info(f"Atlas shape {atlas_data.shape} differs from BOLD shape {self.bold_container.data.shape[:3]}, resampling atlas")
            
            # Create a reference NIfTI image from BOLD data for resampling
            bold_ref_img = nib.Nifti1Image(self.bold_container.data[:,:,:,0], self.bold_container.affine)
            
            # Resample atlas to BOLD space using nilearn
            try:
                from nilearn.image import resample_to_img
                atlas_resampled = resample_to_img(atlas_img, bold_ref_img, interpolation='nearest')
                atlas_data = atlas_resampled.get_fdata()
                if self.logger:
                    self.logger.info(f"Atlas resampled to shape {atlas_data.shape}")
            except ImportError:
                raise ImportError("nilearn is required for atlas resampling. Please install it with: pip install nilearn")
        
        # Create mask for specific region
        roi_mask = atlas_data == region_id
        
        if np.sum(roi_mask) == 0:
            raise ValueError(f"No voxels found for region {region_id} in atlas {atlas_path}")
        
        # Extract probe signal from ROI
        probe_signal = self._extract_signal_from_mask(roi_mask)
        
        # Create ProbeContainer
        probe_container = ProbeContainer(
            participant=self.bold_container.participant,
            task=self.bold_container.task,
            data=probe_signal,
            physio_metadata=None,
            path=atlas_path,
            sampling_frequency=self.bold_container.sampling_frequency,
            units="BOLD_signal",
            layout=self.bold_container.layout,
            logger=self.logger
        )
        
        probe_container.probe_type = "roi_probe"
        
        # Compute baseline using configured method
        baseline_method = self.config.get('physio', {}).get('baseline_method', 'peakutils')
        
        if self.logger:
            self.logger.info(f"Computing ROI probe baseline using method: '{baseline_method}'")
        
        if baseline_method == 'mean':
            # Use the mean of the signal as baseline (recommended for resting-state)
            probe_container.baseline = float(np.mean(probe_signal))
            if self.logger:
                self.logger.info(f"Computed ROI probe baseline using MEAN method: {probe_container.baseline:.6f} BOLD units")
        else:
            # Use peakutils to detect baseline from signal troughs (default, recommended for gas challenge)
            import peakutils
            probe_baseline_array = peakutils.baseline(probe_signal)
            probe_container.baseline = float(np.mean(probe_baseline_array))
            if self.logger:
                self.logger.info(f"Computed ROI probe baseline using PEAKUTILS method: {probe_container.baseline:.6f} BOLD units")
        
        if self.logger:
            n_voxels = np.sum(roi_mask)
            self.logger.info(f"ROI probe extracted from {n_voxels} voxels in atlas region {region_id}")
        
        return probe_container
    
    def _create_spherical_mask(self, center_vox, radius_mm):
        """
        Create spherical ROI mask around center coordinates.
        
        Parameters:
        -----------
        center_vox : array-like
            Center coordinates in voxel space
        radius_mm : float
            Radius in millimeters
            
        Returns:
        --------
        numpy.ndarray
            Boolean mask of spherical ROI
        """
        # Get voxel size from affine matrix
        affine = self.bold_container.affine
        voxel_size = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))
        
        # Convert radius from mm to voxels
        radius_vox = radius_mm / np.mean(voxel_size)  # Use mean voxel size
        
        # Get BOLD data dimensions
        x_size, y_size, z_size = self.bold_container.data.shape[:3]
        
        # Create coordinate grids
        x, y, z = np.ogrid[:x_size, :y_size, :z_size]
        
        # Calculate distance from center
        distance = np.sqrt((x - center_vox[0])**2 + 
                          (y - center_vox[1])**2 + 
                          (z - center_vox[2])**2)
        
        # Create mask
        mask = distance <= radius_vox
        
        if self.logger:
            n_voxels = np.sum(mask)
            self.logger.debug(f"Created spherical mask: radius {radius_mm}mm ({radius_vox:.1f} voxels), "
                            f"{n_voxels} voxels")
        
        return mask
    
    def _extract_signal_from_mask(self, roi_mask):
        """
        Extract average signal from ROI mask.
        
        Parameters:
        -----------
        roi_mask : numpy.ndarray
            Boolean mask defining ROI
            
        Returns:
        --------
        numpy.ndarray
            1D time-series signal averaged across ROI
        """
        # Get BOLD data
        bold_data = self.bold_container.data
        
        # Get brain mask if available to ensure we only use brain voxels
        if hasattr(self.bold_container, 'mask') and self.bold_container.mask is not None:
            brain_mask = self.bold_container.mask > 0
            roi_mask = roi_mask & brain_mask
        
        # Check that ROI contains brain voxels
        roi_voxels = np.sum(roi_mask)
        if roi_voxels == 0:
            raise ValueError("ROI contains no brain voxels")
        
        # Extract time-series for all ROI voxels
        roi_timeseries = bold_data[roi_mask, :]  # Shape: (n_voxels, n_timepoints)
        
        # Compute mean across voxels (ignore NaN values)
        probe_signal = np.nanmean(roi_timeseries, axis=0)
        
        # Check for NaN values in final signal
        if np.any(np.isnan(probe_signal)):
            n_nan = np.sum(np.isnan(probe_signal))
            if self.logger:
                self.logger.warning(f"ROI probe signal contains {n_nan} NaN values out of {len(probe_signal)} timepoints")
            
            # Interpolate NaN values if they exist
            if n_nan < len(probe_signal):  # Don't interpolate if all values are NaN
                probe_signal = self._interpolate_nan_values(probe_signal)
        
        return probe_signal
    
    def _interpolate_nan_values(self, signal):
        """
        Interpolate NaN values in signal using linear interpolation.
        
        Parameters:
        -----------
        signal : numpy.ndarray
            1D signal with potential NaN values
            
        Returns:
        --------
        numpy.ndarray
            Signal with NaN values interpolated
        """
        # Find valid (non-NaN) indices
        valid_indices = ~np.isnan(signal)
        
        if np.sum(valid_indices) < 2:
            raise ValueError("Cannot interpolate: fewer than 2 valid signal values")
        
        # Create time indices
        time_indices = np.arange(len(signal))
        
        # Interpolate NaN values
        signal_interpolated = np.interp(time_indices, 
                                      time_indices[valid_indices], 
                                      signal[valid_indices])
        
        if self.logger:
            n_interpolated = np.sum(np.isnan(signal))
            self.logger.debug(f"Interpolated {n_interpolated} NaN values in ROI probe signal")
        
        return signal_interpolated


def create_roi_probe_from_config(bold_container, config, logger=None, participant=None, task=None, space=None, fmriprep_dir=None):
    """
    Factory function to create ROI probe based on configuration.
    
    Parameters:
    -----------
    bold_container : BoldContainer
        Container with preprocessed BOLD data
    config : dict
        Configuration dictionary with ROI probe settings
    logger : Logger, optional
        Logger instance
    participant : str, optional
        Participant ID (without 'sub-' prefix) for pattern resolution
    task : str, optional
        Task name for pattern resolution  
    space : str, optional
        Space entity for pattern resolution
    fmriprep_dir : str, optional
        Path to fMRIPrep derivatives directory for pattern resolution
        
    Returns:
    --------
    ProbeContainer
        Container with extracted ROI probe signal
        
    Raises:
    -------
    ValueError
        If ROI configuration is invalid or incomplete
    """
    roi_config = config.get('roi_probe', {})
    
    if not roi_config.get('enabled', False):
        raise ValueError("ROI probe extraction is not enabled in configuration")
    
    extractor = ROIProbeExtractor(bold_container, logger=logger, config=config)
    
    method = roi_config.get('method')
    
    if method == 'coordinates':
        coordinates = roi_config.get('coordinates_mm')
        radius = roi_config.get('radius_mm', 6.0)
        
        if coordinates is None:
            raise ValueError("ROI coordinates not specified in configuration")
        
        probe_container = extractor.extract_probe_from_coordinates(coordinates, radius)
    
    elif method == 'mask':
        mask_path = roi_config.get('mask_path')

        if mask_path is None:
            raise ValueError("ROI mask path not specified in configuration")

        # Resolve pattern if necessary
        if participant and task and space and fmriprep_dir:
            try:
                resolved_mask_path = resolve_roi_mask_pattern(
                    mask_path, fmriprep_dir, participant, task, space, logger
                )
                mask_path = resolved_mask_path
                # Update config with resolved path so other functions can use it
                roi_config['mask_path'] = resolved_mask_path
                roi_config['mask_path_resolved'] = resolved_mask_path
            except Exception as e:
                if logger:
                    logger.warning(f"Failed to resolve ROI mask pattern '{mask_path}': {e}")
                raise

        probe_container = extractor.extract_probe_from_mask(mask_path)
    
    elif method == 'atlas':
        atlas_path = roi_config.get('atlas_path')
        region_id = roi_config.get('region_id')
        
        if atlas_path is None or region_id is None:
            raise ValueError("Atlas path or region ID not specified in configuration")
        
        probe_container = extractor.extract_probe_from_atlas(atlas_path, region_id)
    
    else:
        raise ValueError(f"Unknown ROI probe method: {method}")
    
    # Apply bandpass filter if enabled
    bandpass_config = roi_config.get('bandpass_filter', {})
    if bandpass_config.get('enabled', False):
        highpass = bandpass_config.get('highpass')
        lowpass = bandpass_config.get('lowpass')
        
        if highpass is not None or lowpass is not None:
            if logger:
                filter_desc = []
                if highpass is not None:
                    filter_desc.append(f"highpass={highpass} Hz")
                if lowpass is not None:
                    filter_desc.append(f"lowpass={lowpass} Hz")
                logger.info(f"Applying bandpass filter to ROI probe: {', '.join(filter_desc)}")
            
            # Apply bandpass filter
            probe_container.data = apply_bandpass_filter(
                probe_container.data,
                probe_container.sampling_frequency,
                highpass=highpass,
                lowpass=lowpass,
                logger=logger
            )
            
            # Store filter info in the container for later use (JSON sidecars, reports)
            probe_container.bandpass_filter = {
                'enabled': True,
                'highpass': highpass,
                'lowpass': lowpass
            }
            
            if logger:
                logger.info("Bandpass filter applied successfully to ROI probe signal")
        else:
            if logger:
                logger.warning("Bandpass filter enabled but no cutoff frequencies specified. Skipping filter.")
    
    return probe_container
