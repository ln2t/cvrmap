class PhysioPreprocessor:
    def __init__(self, args, logger, layout, participant, config):
        self.args = args
        self.logger = logger
        self.layout = layout
        self.participant = participant
        self.config = config
        self.upper_envelope = None  # Will store the ETCO2 container
        self.physio_container = None  # Will store the original physio container

    def run(self):
        self.logger.info(f"Running physio preprocessing for participant {self.participant}")
        from .data_container import PhysioDataContainer
        container = PhysioDataContainer(participant=self.participant, task=self.args.task, layout=self.layout, logger=self.logger)
        container.load()
        self._process_units(container)
        self.physio_container = container  # Store the original container
        self.upper_envelope = self._extract_etco2(container)
    
    def get_upper_envelope(self):
        """
        Return the upper envelope (ETCO2) container.
        
        Returns:
        --------
        ProbeContainer
            Container with the extracted ETCO2 signal (upper envelope of CO2).
        """
        if self.upper_envelope is None:
            raise ValueError("Upper envelope not computed. Run physio preprocessing first.")
        return self.upper_envelope
    
    def get_physio_container(self):
        """
        Return the original physio container.
        
        Returns:
        --------
        PhysioDataContainer
            Container with the original physiological data.
        """
        if self.physio_container is None:
            raise ValueError("Physio container not available. Run physio preprocessing first.")
        return self.physio_container

    def _extract_etco2(self, container):
        """
        Extract ETCO2 from the CO2 signal using smoothing, peak detection, and interpolation.
        """
        import numpy as np
        from scipy.ndimage import gaussian_filter
        import peakutils
        from scipy.interpolate import interp1d

        # Get sampling rate and CO2 column
        sampling_rate = container.sampling_frequency
        columns = container.physio_metadata.get('Columns', [])
        co2_col = next((i for i, c in enumerate(columns) if c.lower() == 'co2'), None)
        if co2_col is None:
            raise ValueError("CO2 column not found in physio data.")
        co2_signal = container.data[:, co2_col]
        co2_units = container.units

        # Apply smoothing
        window_sigma = self.config['physio']['raw_co2_light_smoothing'] * sampling_rate
        smoothed = gaussian_filter(co2_signal, sigma=window_sigma)

        # Prepare smoother signal for peak detection
        window_sigma_for_peaks_detection = self.config['physio']['peak_detection_smoothing'] * sampling_rate
        smoothed_for_peaks_detection = gaussian_filter(smoothed, sigma=window_sigma_for_peaks_detection)

        self.logger.debug(f"CO2 signal range: {co2_signal.min():.2f} - {co2_signal.max():.2f} {co2_units}")

        # Find peaks
        peaks = peakutils.indexes(smoothed_for_peaks_detection)
        self.logger.debug(f"Detected {len(peaks)} peaks using peakutils")

        # If no peaks found or too few, raise an error
        if len(peaks) < 2:
            raise ValueError("Insufficient peaks detected in CO2 signal")

        # Create time vectors
        co2_t = np.arange(0, len(co2_signal), 1 / sampling_rate)
        etco2_t = co2_t[peaks[0]:peaks[-1]]
        peak_times = co2_t[peaks]
        peak_values = smoothed[peaks]  # Use the lightly smoothed version for peak values

        # Interpolate between peaks
        etco2_interpolator = interp1d(
            peak_times,
            peak_values,
            kind='cubic',
            bounds_error=False,
            fill_value='extrapolate'
        )
        etco2 = etco2_interpolator(etco2_t)
        self.logger.info(f"ETCO2 extraction complete for participant {self.participant}")

        from .data_container import ProbeContainer
        etco2_container = ProbeContainer(
            participant=container.participant,
            task=container.task,
            data=etco2,
            physio_metadata=container.physio_metadata,
            path=container.path,
            sampling_frequency=container.sampling_frequency,
            units=container.units,
            layout=container.layout
        )
        
        # Compute and store baseline value using the configured method
        baseline_method = self.config.get('physio', {}).get('baseline_method', 'peakutils')
        
        if baseline_method == 'mean':
            # Use the mean of the signal as baseline (recommended for resting-state)
            etco2_container.baseline = np.mean(etco2)
            if self.logger:
                self.logger.info(f"Computed baseline using mean method: {etco2_container.baseline:.3f} {etco2_container.units or 'units'}")
        else:
            # Use peakutils to detect baseline from signal troughs (default, recommended for gas challenge)
            import peakutils
            probe_baseline_array = peakutils.baseline(etco2)
            etco2_container.baseline = np.mean(probe_baseline_array)  # Store baseline in container
            if self.logger:
                self.logger.info(f"Computed baseline using peakutils method: {etco2_container.baseline:.3f} {etco2_container.units or 'units'}")
        
        return etco2_container

    def _process_units(self, container):
        """
        Convert CO2 data from % to mmHg if needed.
        """
        if container.units == "mmHg":
            self.logger.info("Units are already mmHg, no conversion needed.")
            return
        elif container.units == "%":
            self.logger.info("Converting CO2 data from % to mmHg using 7.6 mmHg = 1%.")
            import numpy as np
            # Find the CO2 column index
            columns = container.physio_metadata.get('Columns', [])
            co2_col = next((i for i, c in enumerate(columns) if c.lower() == 'co2'), None)
            if co2_col is None:
                self.logger.warning("CO2 column not found in columns metadata during unit conversion.")
                return
            # Convert the CO2 column
            container.data[:, co2_col] = container.data[:, co2_col] * 7.6
            container.units = "mmHg"
            self.logger.info("CO2 data converted to mmHg.")
        else:
            self.logger.warning(f"Unknown units '{container.units}' for CO2. No conversion performed.")
