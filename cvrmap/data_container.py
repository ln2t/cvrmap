class DataContainer:
    """
    Container for holding data for a participant/task/space.
    Extend this class with actual data fields as needed.
    """
    def __init__(self, participant=None, task=None, space=None, data=None, path=None, sampling_frequency=None, dtype=None, units=None, logger=None):
        self.participant = participant
        self.task = task
        self.space = space
        self.data = data  # Should be a numpy array
        self.path = path  # Path to the data file
        self.sampling_frequency = sampling_frequency  # In Hz
        self.type = dtype  # Type of data (e.g., 'physio', 'bold', etc.)
        self.units = units  # Units of the data
        self.logger = logger  # Logger instance for consistent logging

    def load(self, source=None):
        """
        Placeholder for loading data into the container.
        Implement actual loading logic as needed.
        """
        # Example: self.data = load_from_source(source)
        _ = source  # Acknowledge parameter to avoid unused warning
        pass

    def get_normalized_signals(self):
        """
        Normalize signals across time by removing mean and dividing by standard deviation.
        
        For ProbeContainer: Normalizes main signal and time-shifted signals if they exist.
        For BoldContainer: Normalizes BOLD data across time for each voxel within brain mask.
        For other containers: Normalizes the main data signal.
        
        Returns:
        --------
        tuple
            A 2-tuple where:
            - First element: normalized DataContainer of the same type as the original
            - Second element: dict of normalized shifted signals (empty for BOLD or if no shifted signals exist)
        """
        import numpy as np
        
        if self.data is None:
            raise ValueError("No data available for normalization")
        
        # First, always normalize the original signal
        normalized_main_container = self._normalize_main_signal()
        
        # Second, handle shifted signals if they exist (ProbeContainer only)
        normalized_shifted_signals = None
        if hasattr(self, 'shifted_signals') and self.shifted_signals is not None:
            n_delays, n_timepoints = self.shifted_signals.shape
            normalized_shifted_signals = np.zeros_like(self.shifted_signals)
            
            for i in range(n_delays):
                signal_data = self.shifted_signals[i, :].copy()
                signal_mean = np.mean(signal_data)
                signal_std = np.std(signal_data)
                
                if signal_std > 0:
                    normalized_shifted_signals[i, :] = (signal_data - signal_mean) / signal_std
                else:
                    # Handle constant signals
                    normalized_shifted_signals[i, :] = np.zeros_like(signal_data)
                    if self.logger and hasattr(self, 'time_delays_seconds'):
                        self.logger.warning(f"Zero standard deviation in shifted signal at delay {self.time_delays_seconds[i]:.3f}s. Setting to zeros.")
        
        return normalized_main_container, normalized_shifted_signals
    
    def _normalize_main_signal(self):
        """
        Private helper method to normalize the main signal of this container.
        
        Returns:
        --------
        DataContainer
            Normalized container of the same type as the original.
        """
        import numpy as np
        
        # Handle BOLD data (4D: x, y, z, t)
        if self.type == "bold" and len(self.data.shape) == 4:
            x, y, z, t = self.data.shape
            normalized_data = np.full_like(self.data, np.nan)
            
            # Use brain mask if available
            if hasattr(self, 'mask') and self.mask is not None:
                brain_mask = self.mask > 0
            else:
                # Create a simple mask excluding NaN voxels
                brain_mask = ~np.isnan(self.data).any(axis=3)
            
            # Normalize each voxel's time series
            for i in range(x):
                for j in range(y):
                    for k in range(z):
                        if brain_mask[i, j, k]:
                            voxel_timeseries = self.data[i, j, k, :]
                            if not np.isnan(voxel_timeseries).any():
                                voxel_mean = np.mean(voxel_timeseries)
                                voxel_std = np.std(voxel_timeseries)
                                
                                if voxel_std > 0:
                                    normalized_data[i, j, k, :] = (voxel_timeseries - voxel_mean) / voxel_std
                                else:
                                    normalized_data[i, j, k, :] = np.zeros(t)
            
            # Create normalized BoldContainer
            normalized_container = type(self)(
                participant=self.participant,
                task=self.task,
                space=getattr(self, 'space', None),
                data=normalized_data,
                path=self.path,
                layout=getattr(self, 'layout', None),
                logger=self.logger
            )
            
            # Copy BOLD-specific attributes
            if hasattr(self, 'affine'):
                normalized_container.affine = self.affine
                normalized_container.header = self.header
                normalized_container.tr = self.tr
                normalized_container.sampling_frequency = self.sampling_frequency
                normalized_container.mask = self.mask
                normalized_container.mask_path = self.mask_path
            
            return normalized_container
        
        # Handle 1D signals (probe data, etc.)
        else:
            signal_data = self.data.copy()
            signal_mean = np.mean(signal_data)
            signal_std = np.std(signal_data)
            
            if signal_std > 0:
                normalized_data = (signal_data - signal_mean) / signal_std
            else:
                normalized_data = np.zeros_like(signal_data)
                if self.logger:
                    self.logger.warning("Zero standard deviation in signal. Setting to zeros.")
            
            # Create normalized container of the same type
            normalized_container = type(self)(
                participant=self.participant,
                task=self.task,
                data=normalized_data,
                path=self.path,
                sampling_frequency=self.sampling_frequency,
                units="normalized",
                logger=self.logger
            )
            
            # Copy type-specific attributes
            if hasattr(self, 'physio_metadata'):
                normalized_container.physio_metadata = self.physio_metadata
                normalized_container.layout = self.layout
                if hasattr(self, 'probe_type'):
                    normalized_container.probe_type = f"{self.probe_type}_normalized"
            
            return normalized_container


# Specialized DataContainer for probe data (e.g., ETCO2)
class ProbeContainer(DataContainer):
    """
    Specialized DataContainer for probe data such as ETCO2.
    Used for physiological signals that serve as regressors or probes in analysis.
    """
    def __init__(self, participant=None, task=None, data=None, physio_metadata=None, path=None, sampling_frequency=None, units=None, layout=None, logger=None):
        super().__init__(participant=participant, task=task, space=None, data=data, path=path, sampling_frequency=sampling_frequency, dtype="probe", units=units, logger=logger)
        self.physio_metadata = physio_metadata
        self.layout = layout
        self.probe_type = "etco2"  # Default probe type, can be extended for other probes
        
        # Properties for time-shifted signals
        self.shifted_signals = None  # 2D numpy array: (n_delays, n_timepoints)
        self.time_delays_seconds = None  # 1D numpy array of time delays in seconds
        self.baseline = None  # Baseline value computed using peakutils

    def _compute_shifted_signals(self, time_delays_seconds):
        """
        Compute time-shifted versions of the probe signal using direct index shifting.
        
        Parameters:
        -----------
        time_delays_seconds : np.ndarray
            Array of time delays in seconds to compute shifted signals for.
            
        Returns:
        --------
        None
            Populates self.shifted_signals (2D array) and self.time_delays_seconds (1D array)
        """
        import numpy as np
        
        if self.data is None or self.sampling_frequency is None:
            raise ValueError("ProbeContainer must have data and sampling_frequency to compute shifted signals")
        
        # Store the time delays array
        self.time_delays_seconds = np.array(time_delays_seconds)
        n_delays = len(time_delays_seconds)
        n_timepoints = len(self.data)
        
        # Initialize shifted signals array
        self.shifted_signals = np.zeros((n_delays, n_timepoints))
        
        # Use pre-computed baseline value if available, otherwise compute it
        if self.baseline is not None:
            probe_baseline = self.baseline
        else:
            # Fallback to computing baseline if not already set
            import peakutils
            probe_baseline_array = peakutils.baseline(self.data)
            probe_baseline = np.mean(probe_baseline_array)
            self.baseline = probe_baseline
            
            if self.logger:
                self.logger.warning("Baseline was not pre-computed. Computing baseline using peakutils as fallback.")
        
        for i, delta_t in enumerate(time_delays_seconds):
            # Copy length and fill with baseline
            n = len(self.data)
            data = probe_baseline * np.ones(n)
            probe_sf = self.sampling_frequency
            
            # Number of points corresponding to delta_t
            delta_n = int(delta_t * probe_sf)
            
            # Shift and copy the original data
            if delta_n == 0:
                data[:] = self.data
            
            # Sign convention: POSITIVE delay means that the data signal occurs AFTER the probe signal
            elif delta_n < 0:
                # Negative delay: probe signal is delayed relative to data
                data[:n+delta_n] = self.data[-delta_n:]
            elif delta_n > 0:
                # Positive delay: probe signal is advanced relative to data
                data[delta_n:] = self.data[:n-delta_n]
            
            # Store shifted signal in array
            self.shifted_signals[i, :] = data

    def get_resampled_shifted_signals(self, time_delays_seconds, target_sampling_frequency, target_duration_seconds):
        """
        Compute time-shifted signals at a target sampling frequency.

        CRITICAL: For TR-agnostic analysis, we must resample BEFORE shifting to avoid
        quantization of delays to the original TR. This method:
        1. Resamples the probe signal to target_sampling_frequency
        2. Computes shifts on the resampled signal

        Parameters:
        -----------
        time_delays_seconds : np.ndarray
            Array of time delays in seconds to compute shifted signals for.
        target_sampling_frequency : float
            Target sampling frequency in Hz to resample shifted signals to.
        target_duration_seconds : float
            Target duration in seconds to truncate the resampled signals to.

        Returns:
        --------
        tuple
            - resampled_shifted_signals: 2D numpy array (n_delays, n_target_timepoints)
            - time_delays_seconds: 1D numpy array of time delays in seconds
        """
        import numpy as np
        from scipy.signal import resample

        if not isinstance(time_delays_seconds, np.ndarray):
            time_delays_seconds = np.array(time_delays_seconds)

        if len(time_delays_seconds) == 0:
            raise ValueError("time_delays_seconds array cannot be empty")

        if target_duration_seconds <= 0:
            raise ValueError("target_duration_seconds must be positive")

        # STEP 1: Resample the original probe signal to target sampling frequency
        original_n_samples = len(self.data)
        original_duration = original_n_samples / self.sampling_frequency

        # Determine target number of samples
        target_n_samples = int(np.round(min(original_duration, target_duration_seconds) * target_sampling_frequency))

        # Resample the probe signal
        resampled_probe_data = resample(self.data, target_n_samples)

        if self.logger:
            self.logger.debug(f"Resampled probe for shifting: {self.sampling_frequency:.3f} Hz ({original_n_samples} samples) "
                            f"→ {target_sampling_frequency:.3f} Hz ({target_n_samples} samples)")

        # STEP 2: Compute shifted signals using the RESAMPLED probe data at target sampling frequency
        n_delays = len(time_delays_seconds)
        shifted_signals = np.zeros((n_delays, target_n_samples))

        # Get baseline value
        if self.baseline is not None:
            probe_baseline = self.baseline
        else:
            import peakutils
            probe_baseline_array = peakutils.baseline(self.data)
            probe_baseline = np.mean(probe_baseline_array)
            self.baseline = probe_baseline
            if self.logger:
                self.logger.warning("Baseline was not pre-computed. Computing baseline using peakutils as fallback.")

        # Compute shifts on the resampled data
        for i, delta_t in enumerate(time_delays_seconds):
            data = probe_baseline * np.ones(target_n_samples)

            # Number of points corresponding to delta_t at TARGET sampling frequency
            delta_n = int(delta_t * target_sampling_frequency)

            # Shift and copy the resampled data
            if delta_n == 0:
                data[:] = resampled_probe_data
            elif delta_n < 0:
                # Negative delay: probe signal is delayed relative to data
                data[:target_n_samples+delta_n] = resampled_probe_data[-delta_n:]
            elif delta_n > 0:
                # Positive delay: probe signal is advanced relative to data
                data[delta_n:] = resampled_probe_data[:target_n_samples-delta_n]

            shifted_signals[i, :] = data

        return shifted_signals, time_delays_seconds

    def extrapolate(self, target_duration_seconds, type="baseline"):
        """
        Extrapolate the probe signal to a target duration if current duration is shorter.
        
        Parameters:
        -----------
        target_duration_seconds : float
            Target duration in seconds to extrapolate the signal to.
        type : str, optional
            Type of extrapolation to use. Default is "baseline".
            Currently only "baseline" extrapolation is supported.
            
        Returns:
        --------
        None
            Modifies self.data in place by appending extrapolated values.
        """
        import numpy as np
        import peakutils
        import warnings
        
        if self.data is None or self.sampling_frequency is None:
            raise ValueError("ProbeContainer must have data and sampling_frequency to extrapolate")
        
        if target_duration_seconds <= 0:
            raise ValueError("target_duration_seconds must be positive")
        
        if type != "baseline":
            raise ValueError(f"Extrapolation type '{type}' not supported. Currently only 'baseline' is supported.")
        
        # Calculate current duration
        current_duration = len(self.data) / self.sampling_frequency
        
        if current_duration >= target_duration_seconds:
            # No extrapolation needed
            return
        
        # Calculate how many samples we need to add
        target_n_samples = int(np.round(target_duration_seconds * self.sampling_frequency))
        current_n_samples = len(self.data)
        samples_to_add = target_n_samples - current_n_samples
        
        # Use pre-computed baseline value if available, otherwise compute it
        if self.baseline is not None:
            baseline_value = self.baseline
        else:
            # Fallback to computing baseline if not already set
            import peakutils
            probe_baseline_array = peakutils.baseline(self.data)
            baseline_value = np.mean(probe_baseline_array)
            self.baseline = baseline_value
            
            if self.logger:
                self.logger.warning("Baseline was not pre-computed. Computing baseline using peakutils as fallback.")
        
        # Create extrapolated data
        extrapolated_samples = np.full(samples_to_add, baseline_value)
        
        # Append to existing data
        self.data = np.concatenate([self.data, extrapolated_samples])
        
        # Log the extrapolation information
        message = (
            f"ProbeContainer for participant {self.participant}: "
            f"Extrapolated signal from {current_duration:.2f}s to {target_duration_seconds:.2f}s "
            f"using baseline value {baseline_value:.4f} ({self.units}). "
            f"Added {samples_to_add} samples."
        )
        
        if self.logger is not None:
            self.logger.warning(message)
        else:
            warnings.warn(message, UserWarning)

    def get_resampled_normalized_shifted_signals(self, time_delays_seconds, target_sampling_frequency, target_duration_seconds):
        """
        Convenience method to get resampled and normalized shifted signals in one go.
        
        This method combines the functionality of get_resampled_shifted_signals() and 
        normalization to provide a streamlined interface for preparing 
        shifted probe signals that are both resampled to target specifications and 
        normalized for optimal cross-correlation analysis.
        
        Parameters
        ----------
        time_delays_seconds : np.ndarray
            Array of time delays in seconds to compute shifted signals for.
        target_sampling_frequency : float
            Target sampling frequency in Hz for resampling.
        target_duration_seconds : float
            Target duration in seconds for the signals.
            
        Returns
        -------
        tuple
            - normalized_shifted_signals: 2D numpy array (n_delays, n_timepoints) with normalized signals
            - time_delays_seconds: 1D numpy array of time delays in seconds
            
        Raises
        ------
        ValueError
            If required data or sampling_frequency is missing.
        """
        import numpy as np
        
        # First get the resampled shifted signals
        resampled_shifted_signals, delays = self.get_resampled_shifted_signals(
            time_delays_seconds, target_sampling_frequency, target_duration_seconds
        )
        
        if resampled_shifted_signals is None:
            return None, None
        
        # Store the resampled signals temporarily in the container
        original_shifted_signals = self.shifted_signals
        original_time_delays = self.time_delays_seconds
        
        # Temporarily set the resampled signals and delays
        self.shifted_signals = resampled_shifted_signals
        self.time_delays_seconds = delays
        
        # Use the existing get_normalized_signals method
        _, normalized_shifted_signals = self.get_normalized_signals()
        
        # Restore the original shifted signals
        self.shifted_signals = original_shifted_signals
        self.time_delays_seconds = original_time_delays
        
        # Log the operation
        if normalized_shifted_signals is not None:
            n_delays = normalized_shifted_signals.shape[0]
            min_delay = np.min(delays)
            max_delay = np.max(delays)
            message = (
                f"ProbeContainer for participant {self.participant}: "
                f"Generated {n_delays} resampled and normalized shifted signals "
                f"with delays from {min_delay:.3f}s to {max_delay:.3f}s, resampled to {target_sampling_frequency}Hz "
                f"and duration {target_duration_seconds}s."
            )
            
            if self.logger is not None:
                self.logger.info(message)
        
        return normalized_shifted_signals, delays




# Specialized DataContainer for physiological data
class PhysioDataContainer(DataContainer):
    """
    Specialized DataContainer for physiological data.
    Extend with physio-specific fields and methods as needed.
    """
    def __init__(self, participant=None, task=None, data=None, physio_metadata=None, path=None, sampling_frequency=None, units=None, layout=None, logger=None):
        super().__init__(participant=participant, task=task, space=None, data=data, path=path, sampling_frequency=sampling_frequency, dtype="physio", units=units, logger=logger)
        self.physio_metadata = physio_metadata
        self.layout = layout

    def load(self):
        """
        Use self.layout to find and load physio data and metadata for this participant/task/space.
        """
        import json
        import pandas as pd
        import numpy as np
        if self.layout is None:
            raise ValueError("A BIDSLayout instance must be provided to PhysioDataContainer at initialization.")
        # Find physio file
        physio_files = self.layout.get(
            subject=self.participant,
            task=self.task,
            suffix="physio",
            extension=[".tsv", ".tsv.gz"],
            return_type="file"
        )
        if not physio_files:
            raise FileNotFoundError(f"No physio file found for participant {self.participant}, task {self.task}.")
        physio_path = physio_files[0]
        # Find corresponding JSON file
        json_files = self.layout.get(
            subject=self.participant,
            task=self.task,
            suffix="physio",
            extension=".json",
            return_type="file"
        )
        if not json_files:
            raise FileNotFoundError(f"No physio JSON file found for participant {self.participant}, task {self.task}.")
        json_path = json_files[0]
        self.path = physio_path
        # Load JSON metadata
        with open(json_path, 'r') as f:
            meta = json.load(f)
        self.physio_metadata = meta
        self.sampling_frequency = meta.get('SamplingFrequency', None)
        columns = meta.get('Columns', None)
        if not columns or 'co2' not in [c.lower() for c in columns]:
            raise ValueError(f"No 'co2' column found in physio data for participant {self.participant}.")
        # Find the actual column name for CO2 (case-insensitive)
        co2_col = next(c for c in columns if c.lower() == 'co2')
        # Get units for CO2
        co2_info = meta.get('co2', None)
        if not co2_info or 'Units' not in co2_info:
            raise ValueError(f"No 'Units' found for 'co2' in physio metadata for participant {self.participant}.")
        self.units = co2_info['Units']
        # Load data with pandas
        df = pd.read_csv(physio_path, sep=' ', names=columns)
        self.data = df.to_numpy()


# Specialized DataContainer for BOLD data
class BoldContainer(DataContainer):
    """
    Specialized DataContainer for BOLD fMRI data.
    Extend with BOLD-specific fields and methods as needed.
    """
    def __init__(self, participant=None, task=None, space=None, data=None, path=None, layout=None, logger=None):
        super().__init__(participant=participant, task=task, space=space, data=data, path=path, dtype="bold", logger=logger)
        self.layout = layout
        self.affine = None  # Affine transformation matrix
        self.header = None  # NIfTI header information
        self.tr = None  # Repetition time
        self.mask = None  # Brain mask data
        self.mask_path = None  # Path to the mask file

    def load(self):
        """
        Use self.layout to find and load BOLD data for this participant/task/space.
        """
        import nibabel as nib
        if self.layout is None:
            raise ValueError("A BIDSLayout instance must be provided to BoldContainer at initialization.")
        
        # Find BOLD file in derivatives (fmriprep)
        bold_files = self.layout.derivatives["fMRIPrep"].get(
            subject=self.participant,
            task=self.task,
            space=self.space,
            suffix="bold",
            extension=".nii.gz",
            return_type="filename"
        )
        if not bold_files:
            raise FileNotFoundError(f"No BOLD file found for participant {self.participant}, task {self.task}, space {self.space}.")
        
        bold_path = bold_files[0]
        self.path = bold_path
        
        # Load BOLD data with nibabel
        img = nib.load(bold_path)
        self.data = img.get_fdata()
        self.affine = img.affine
        self.header = img.header
        self.tr = img.header.get_zooms()[-1]  # TR is the last dimension
        self.sampling_frequency = 1.0 / self.tr  # Sampling frequency is 1/TR
        
        # Find and load brain mask
        mask_files = self.layout.derivatives["fMRIPrep"].get(
            subject=self.participant,
            task=self.task,
            space=self.space,
            suffix="mask",
            extension=".nii.gz",
            return_type="file"
        )
        if not mask_files:
            raise FileNotFoundError(f"No brain mask found for participant {self.participant}, task {self.task}, space {self.space}.")
        
        self.mask_path = mask_files[0]
        mask_img = nib.load(self.mask_path)
        self.mask = mask_img.get_fdata()

    def get_global_signal(self):
        """
        Compute the BOLD global signal across all brain voxels and return it as a ProbeContainer.
        
        The global signal is computed as the mean BOLD signal across all voxels within the brain mask
        at each timepoint. This can be used as a probe signal for analysis or quality control.
        
        Returns:
        --------
        ProbeContainer
            ProbeContainer containing the global signal timecourse with appropriate metadata.
            
        Raises:
        ------
        ValueError
            If BOLD data or brain mask is not available.
        """
        import numpy as np
        
        if self.data is None:
            raise ValueError("BOLD data must be loaded before computing global signal")
        
        if self.mask is None:
            raise ValueError("Brain mask must be loaded before computing global signal")
        
        # Get BOLD data and mask
        bold_data = self.data  # Shape: (x, y, z, t)
        mask_data = self.mask  # Shape: (x, y, z)
        
        x, y, z, t = bold_data.shape
        
        # Create brain mask boolean array
        brain_voxels = mask_data > 0
        n_brain_voxels = np.sum(brain_voxels)
        
        if n_brain_voxels == 0:
            raise ValueError("No brain voxels found in mask")
        
        # Reshape BOLD data to 2D: (n_voxels, time)
        bold_2d = bold_data.reshape(-1, t)
        
        # Extract only brain voxels: (n_brain_voxels, time)
        brain_voxel_indices = brain_voxels.ravel()
        bold_brain = bold_2d[brain_voxel_indices, :]
        
        # Compute global signal as mean across all brain voxels at each timepoint
        global_signal = np.mean(bold_brain, axis=0)  # Shape: (t,)
        
        # Create ProbeContainer with global signal
        global_signal_container = ProbeContainer(
            participant=self.participant,
            task=self.task,
            data=global_signal,
            physio_metadata=None,  # No physio metadata for BOLD-derived signal
            path=self.path,  # Reference to original BOLD file
            sampling_frequency=self.sampling_frequency,
            units="BOLD_signal",  # Units are BOLD signal intensity
            layout=self.layout,
            logger=self.logger
        )
        
        # Set probe type to indicate this is a global signal
        global_signal_container.probe_type = "bold_global_signal"
        
        # Log the operation
        if self.logger:
            self.logger.info(f"Computed BOLD global signal for participant {self.participant}: "
                           f"{len(global_signal)} timepoints from {n_brain_voxels:,} brain voxels")
        
        return global_signal_container

    def save(self, output_dir, label=None):
        """
        Save the BOLD container data to a NIfTI file.

        Parameters:
        -----------
        output_dir : str
            Base output directory for BIDS derivatives.
        label : str, optional
            ROI label for BIDS naming (e.g., 'SSS'). If provided, adds '_label-{label}' entity.

        Returns:
        --------
        str : Path where the data was saved
        """
        import os
        import nibabel as nib

        # Build BIDS path: output_dir/sub-{participant}/func/sub-{participant}_task-{task}_label-{label}_space-{space}_desc-denoised_bold.nii.gz
        participant_dir = f"sub-{self.participant}"
        func_dir = "func"

        # Build filename components
        filename_parts = [f"sub-{self.participant}"]

        if self.task:
            filename_parts.append(f"task-{self.task}")

        if label:
            filename_parts.append(f"label-{label}")

        if self.space:
            filename_parts.append(f"space-{self.space}")

        filename_parts.append("desc-denoised")
        filename_parts.append("bold.nii.gz")
        
        filename = "_".join(filename_parts)
        
        # Construct full path
        output_path = os.path.join(output_dir, participant_dir, func_dir, filename)
        
        # Create directories if they don't exist
        output_directory = os.path.dirname(output_path)
        os.makedirs(output_directory, exist_ok=True)
        
        # Save BOLD data as NIfTI file
        if self.data is not None and self.affine is not None:
            # Create NIfTI image from data
            img = nib.Nifti1Image(self.data, self.affine, self.header)
            
            # Save to file
            nib.save(img, output_path)
            print(f"Saved BOLD data for participant {self.participant} to: {output_path}")
        else:
            raise ValueError("Cannot save BOLD data: data or affine matrix is None")
        
        return output_path

    def resample_to_frequency(self, target_sampling_frequency):
        """
        Resample BOLD data to a target sampling frequency.

        This is essential for TR-agnostic cross-correlation analysis. All signals
        must be resampled to match the delay step (target_sf = 1/delay_step) to
        ensure each delay increment actually shifts the signal.

        Parameters:
        -----------
        target_sampling_frequency : float
            Target sampling frequency in Hz (e.g., 1.0 Hz for 1-second delay steps)

        Returns:
        --------
        None
            Modifies self.data, self.sampling_frequency, and self.tr in place
        """
        import numpy as np
        from scipy.signal import resample

        if self.data is None:
            raise ValueError("BOLD data must be loaded before resampling")

        if self.sampling_frequency is None:
            raise ValueError("BOLD sampling frequency must be set before resampling")

        if target_sampling_frequency <= 0:
            raise ValueError("Target sampling frequency must be positive")

        # Calculate current duration and target number of samples
        current_n_timepoints = self.data.shape[-1]
        current_duration = current_n_timepoints / self.sampling_frequency
        target_n_timepoints = int(np.round(current_duration * target_sampling_frequency))

        if self.logger:
            self.logger.info(f"Resampling BOLD data: {self.sampling_frequency:.3f} Hz ({current_n_timepoints} samples) → "
                           f"{target_sampling_frequency:.3f} Hz ({target_n_timepoints} samples)")

        # Resample along the time axis (last dimension)
        # BOLD data shape: (x, y, z, time)
        resampled_data = resample(self.data, target_n_timepoints, axis=-1)

        # Update container
        self.data = resampled_data
        self.sampling_frequency = target_sampling_frequency
        self.tr = 1.0 / target_sampling_frequency

        if self.logger:
            self.logger.info(f"BOLD data resampled successfully to {target_sampling_frequency} Hz (TR={self.tr:.3f}s)")
