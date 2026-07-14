class BoldPreprocessor:
    def __init__(self, args, logger, layout, participant, config, etco2_container, time_delays_seconds):
        self.args = args
        self.logger = logger
        self.layout = layout
        self.participant = participant
        self.config = config
        self.etco2_container = etco2_container  # ETCO2 container for computing shifted probes
        self.time_delays_seconds = time_delays_seconds  # Array of time delays to consider in processing

    def _get_roi_label_entity(self):
        """
        Get the ROI label entity string for BIDS naming.

        Returns '_label-{label}' if ROI probe is enabled and has a label configured,
        otherwise returns an empty string.
        """
        roi_config = self.config.get('roi_probe', {}) if self.config else {}
        if roi_config.get('enabled') and roi_config.get('label'):
            return f"_label-{roi_config['label']}"
        return ''

    def run(self, bold_container):
        self.logger.info(f"Running BOLD preprocessing for participant {self.participant}")
        
        # Use the provided BOLD container
        self.bold_container = bold_container
        self.logger.info(f"Using provided BOLD container for participant {self.participant}")
        
        # Compute relative shifted probe signals based on global delay
        self._compute_relative_shifted_probes()
        
        if self.shifted_probes is not None and self.shifted_probes[0] is not None:
            n_probes = self.shifted_probes[0].shape[0]
            self.logger.debug(f"Using {n_probes} shifted probe signals")
        else:
            self.logger.debug("No shifted probe signals available")
        
        # Check probes before processing
        self._check_probes_are_normalized()
        
        # Apply preprocessing steps
        self._refine_aroma_components()
        self._nonaggressive_denoising()
        self._temporal_filtering()
        self._spatial_smooth()
        
        # Save the processed data with ROI label if applicable
        roi_label = self.config.get('roi_probe', {}).get('label') if self.config else None
        output_path = self.bold_denoised.save(self.args.output_dir, label=roi_label)
        self.logger.info(f"Processed BOLD data saved to: {output_path}")
        
        self.logger.info(f"BOLD preprocessing completed for participant {self.participant}")
        return self.bold_denoised
    
    def _compute_relative_shifted_probes(self):
        """
        Compute shifted probe signals using the provided time delays array.
        
        This method uses the time_delays_seconds array provided during initialization
        to compute shifted ETCO2 probe signals for BOLD preprocessing.
        """
        import numpy as np
        
        # Get BOLD container properties for resampling
        target_sampling_frequency = self.bold_container.sampling_frequency
        target_duration_seconds = self.bold_container.data.shape[-1] / self.bold_container.sampling_frequency
        
        self.logger.info(f"Computing resampled and normalized shifted ETCO2 probe with delays {self.time_delays_seconds[0]:.1f}s to {self.time_delays_seconds[-1]:.1f}s")
        
        # Compute the shifted probe signals using the provided time delays
        shifted_signals, time_delays = self.etco2_container.get_resampled_normalized_shifted_signals(
            time_delays_seconds=self.time_delays_seconds,
            target_sampling_frequency=target_sampling_frequency,
            target_duration_seconds=target_duration_seconds
        )
        
        # Store as tuple for consistency with new API
        self.shifted_probes = (shifted_signals, time_delays)
        
        if shifted_signals is not None:
            self.logger.info(f"Generated {shifted_signals.shape[0]} resampled and normalized shifted ETCO2 signals for BOLD preprocessing")
        else:
            self.logger.warning("No shifted signals generated")
    
    def _check_probes_are_normalized(self):
        """
        Check and validate the shifted probe signals before processing.
        Validates sampling frequency consistency and normalization status.
        """
        import numpy as np
        
        self.logger.debug("Checking probe signals...")
        
        if self.shifted_probes is None or self.shifted_probes[0] is None:
            self.logger.warning("No shifted probe signals provided for processing.")
            return
        
        shifted_signals, time_delays = self.shifted_probes
        bold_sampling_frequency = self.bold_container.sampling_frequency
        self.logger.debug(f"BOLD container sampling frequency: {bold_sampling_frequency} Hz")
        
        # Check that we have valid data
        n_delays, n_timepoints = shifted_signals.shape
        
        # Check normalization status by checking if signals have approximately zero mean and unit variance
        validation_errors = []
        normalized_count = 0
        
        for i in range(n_delays):
            signal = shifted_signals[i, :]
            signal_mean = np.mean(signal)
            signal_std = np.std(signal)
            delay = time_delays[i]
            
            # Check if signal is approximately normalized (mean ~0, std ~1)
            if abs(signal_mean) > 1e-10 or abs(signal_std - 1.0) > 1e-6:
                error_msg = f"Probe at delay {delay:.3f}s is not normalized: mean={signal_mean:.6f}, std={signal_std:.6f}"
                validation_errors.append(error_msg)
                self.logger.warning(error_msg)
            else:
                normalized_count += 1
        
        # Report validation results
        total_probes = n_delays
        
        if validation_errors:
            self.logger.warning(f"Probe validation found {len(validation_errors)} issues:")
            for error in validation_errors:
                self.logger.warning(f"  - {error}")
        else:
            self.logger.info("All probe validation checks passed successfully.")
        
        self.logger.info(f"Probe check completed:")
        self.logger.info(f"  Total shifted probe signals: {total_probes}")
        self.logger.info(f"  Normalized probes: {normalized_count}/{total_probes}")
        self.logger.info(f"  Sampling frequency matches: {total_probes - len([e for e in validation_errors if 'sampling frequency' in e])}/{total_probes}")
        
        # Raise exception if critical validation fails
        if validation_errors:
            raise ValueError(f"Probe validation failed with {len(validation_errors)} errors. See log for details.")
    
    def _refine_aroma_components(self):
        """
        Refine AROMA components for denoising.
        Load AROMA noise ICs and MELODIC mixing matrix from fmriprep outputs.
        """
        self.logger.debug("Refining AROMA components...")
        
        # Load AROMA noise ICs file (comma-separated)
        aroma_files = self.layout.derivatives["fMRIPrep"].get(
            subject=self.participant,
            task=self.args.task,
            suffix='AROMAnoiseICs',
            extension='.csv'
        )
        
        # Load MELODIC mixing matrix (tab-separated)
        melodic_files = self.layout.derivatives["fMRIPrep"].get(
            subject=self.participant,
            task=self.args.task,
            desc='MELODIC',
            suffix='mixing',
            extension='.tsv'
        )
        
        if not aroma_files:
            self.logger.warning(f"No AROMA noise ICs file found for participant {self.participant}, task {self.args.task}")
            return
            
        if not melodic_files:
            self.logger.warning(f"No MELODIC mixing matrix file found for participant {self.participant}, task {self.args.task}")
            return
            
        aroma_path = aroma_files[0]
        melodic_path = melodic_files[0]
        
        self.logger.debug(f"Found AROMA noise ICs: {aroma_path}")
        self.logger.debug(f"Found MELODIC mixing matrix: {melodic_path}")
        
        # Read AROMA IC indices
        import pandas as pd
        import numpy as np
        aroma_df = pd.read_csv(aroma_path, header=None)
        ic_indices_1based = aroma_df.iloc[0].values.astype(int)  # AROMA indices are 1-based
        ic_indices = ic_indices_1based - 1  # Convert to 0-based indexing for array access
        self.logger.debug(f"AROMA noise IC indices (1-based): {ic_indices_1based}")
        self.logger.debug(f"AROMA noise IC indices (0-based): {ic_indices}")
        
        # Read MELODIC mixing matrix
        melodic_df = pd.read_csv(melodic_path, sep='\t', header=None)
        melodic_timecourses = melodic_df.values  # Shape: (timepoints, components)
        
        # Store MELODIC data for later use in denoising
        self.all_melodic_components = melodic_timecourses
        
        self.logger.debug(f"MELODIC mixing matrix shape: {melodic_timecourses.shape}")
        
        # Extract timecourses for AROMA noise ICs and create ProbeContainers
        from .data_container import ProbeContainer
        
        # Get correlation threshold from config
        aroma_correlation_threshold = self.config['bold']['denoising']['aroma_correlation_threshold']
        self.logger.debug(f"Using AROMA correlation threshold: {aroma_correlation_threshold}")
        
        # List to store refined noise IC indices (components to keep as noise)
        refined_noise_ics = []
        
        # Store classification results for plotting
        classification_correlations = []
        classification_delays = []
        
        for i, ic_idx in enumerate(ic_indices):
            # Validate 0-based index for array access
            if ic_idx < 0 or ic_idx >= melodic_timecourses.shape[1]:
                self.logger.warning(f"IC index {ic_indices_1based[i]} (1-based) -> {ic_idx} (0-based) exceeds available components ({melodic_timecourses.shape[1]})")
                continue
                
            ic_timecourse = melodic_timecourses[:, ic_idx]
            
            # Create ProbeContainer for this IC timecourse
            melodic_ic_container = ProbeContainer(
                participant=self.participant,
                task=self.args.task,
                data=ic_timecourse,
                sampling_frequency=self.bold_container.sampling_frequency,
                units="arbitrary",
                layout=self.layout,
                logger=self.logger
            )
            melodic_ic_container.probe_type = "melodic_ic"
            
            # Normalize the IC timecourse before cross-correlation
            normalized_ic_container, _ = melodic_ic_container.get_normalized_signals()
            
            self.logger.debug(f"Processing IC {ic_indices_1based[i]} (1-based) with {len(ic_timecourse)} timepoints (normalized)")
            
            # Cross-correlate this normalized IC with all shifted ETCO2 probes to find best correlation
            from .cross_correlation import cross_correlate
            best_correlation, best_delay = cross_correlate(normalized_ic_container, self.shifted_probes, self.logger, self.config)
            
            # Store results for plotting
            classification_correlations.append(best_correlation)
            classification_delays.append(best_delay)
            
            self.logger.debug(f"IC {ic_indices_1based[i]} best correlation: {best_correlation:.3f} at delay {best_delay:.3f}s")
            
            # Check if best correlation exceeds threshold
            if best_correlation > aroma_correlation_threshold:
                self.logger.debug(f"IC {ic_indices_1based[i]} correlation ({best_correlation:.3f}) > threshold ({aroma_correlation_threshold}), removing from noise list")
                # Component is correlated with ETCO2, so it's likely signal, not noise
                # Do not add to refined_noise_ics
            else:
                self.logger.debug(f"IC {ic_indices_1based[i]} correlation ({best_correlation:.3f}) <= threshold ({aroma_correlation_threshold}), keeping as noise")
                # Component is not correlated with ETCO2, keep it as noise
                refined_noise_ics.append(int(ic_indices_1based[i]))  # Store 1-based index
        
        # Store refined noise ICs for later use in denoising
        self.refined_noise_ics = refined_noise_ics
        
        # Store classification results for plotting
        self.ic_classification_results = {
            'original_noise_ics': ic_indices_1based.tolist(),
            'correlations': classification_correlations,
            'delays': classification_delays,
            'refined_noise_ics': refined_noise_ics,
            'aroma_threshold': aroma_correlation_threshold
        }
        
        # Generate IC classification plots for the report
        self._generate_ic_classification_plots()
        
        self.logger.info(f"AROMA component refinement complete:")
        self.logger.info(f"  Original noise ICs: {ic_indices_1based.tolist()}")
        self.logger.info(f"  Refined noise ICs: {refined_noise_ics}")
        self.logger.info(f"  Removed {len(ic_indices_1based) - len(refined_noise_ics)} components from noise list")
    
    def _get_etco2_signal_at_delay(self, delay_seconds):
        """
        Get the ETCO₂ signal shifted to a specific delay and resampled to BOLD sampling frequency.
        
        Parameters:
        -----------
        delay_seconds : float
            The delay in seconds to shift the ETCO₂ signal
            
        Returns:
        --------
        np.ndarray or None
            The shifted and resampled ETCO₂ signal, or None if not available
        """
        try:
            import numpy as np
            
            # Get BOLD container properties for resampling
            target_sampling_frequency = self.bold_container.sampling_frequency
            target_duration_seconds = self.bold_container.data.shape[-1] / self.bold_container.sampling_frequency
            
            # Get the ETCO₂ signal shifted to the specific delay
            shifted_signal, _ = self.etco2_container.get_resampled_normalized_shifted_signals(
                time_delays_seconds=np.array([delay_seconds]),
                target_sampling_frequency=target_sampling_frequency,
                target_duration_seconds=target_duration_seconds
            )
            
            if shifted_signal is not None and shifted_signal.shape[0] > 0:
                return shifted_signal[0, :]  # Return the first (and only) signal
            else:
                return None
                
        except Exception as e:
            self.logger.warning(f"Could not get ETCO₂ signal at delay {delay_seconds}s: {e}")
            return None
    
    def _generate_ic_classification_plots(self):
        """
        Generate plots showing MELODIC IC timecourses and their classification status.
        Creates a figure showing all AROMA noise ICs with their cross-correlation results
        and whether they were kept as noise or reclassified as signal.
        """
        self.logger.debug("Generating IC classification plots...")
        
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        from pathlib import Path
        
        # Check if we have the necessary data
        if not hasattr(self, 'ic_classification_results'):
            self.logger.warning("No IC classification results available for plotting")
            return
        
        # Create figures directory
        figures_dir = Path(self.args.output_dir) / f"sub-{self.participant}" / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Get classification results
        results = self.ic_classification_results
        original_noise_ics = results['original_noise_ics']
        correlations = results['correlations']
        delays = results['delays']
        refined_noise_ics = results['refined_noise_ics']
        aroma_threshold = results['aroma_threshold']
        
        # Calculate grid dimensions for subplots
        n_ics = len(original_noise_ics)
        if n_ics == 0:
            self.logger.warning("No AROMA noise ICs to plot")
            return
        
        # Use a reasonable grid layout
        n_cols = min(4, n_ics)
        n_rows = (n_ics + n_cols - 1) // n_cols
        
        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        if n_ics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        # Plot each IC
        for i, ic_1based in enumerate(original_noise_ics):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Get IC timecourse (0-based indexing for array access)
            ic_0based = ic_1based - 1
            ic_timecourse = self.all_melodic_components[:, ic_0based]
            
            # Get classification info
            correlation = correlations[i]
            delay = delays[i]
            is_noise = ic_1based in refined_noise_ics
            
            # Get the ETCO₂ signal shifted to the optimal delay for this IC
            etco2_shifted = self._get_etco2_signal_at_delay(delay)
            
            # Plot timecourse
            time_points = np.arange(len(ic_timecourse)) / self.bold_container.sampling_frequency
            
            # Plot IC timecourse
            ic_color = 'red' if is_noise else 'green'
            ax.plot(time_points, ic_timecourse, linewidth=1.2, 
                   color=ic_color, label=f'IC {ic_1based}', alpha=0.8)
            
            # Plot ETCO₂ signal if available
            if etco2_shifted is not None:
                # Normalize ETCO₂ to have similar scale as IC for visualization
                etco2_normalized = (etco2_shifted - np.mean(etco2_shifted)) / np.std(etco2_shifted)
                ic_std = np.std(ic_timecourse)
                etco2_scaled = etco2_normalized * ic_std * 0.7  # Scale to 70% of IC amplitude
                
                ax.plot(time_points, etco2_scaled, linewidth=1, 
                       color='blue', linestyle='--', alpha=0.6, label='ETCO₂')
            
            # Set title with classification info
            status = "NOISE" if is_noise else "SIGNAL"
            ax.set_title(f"IC {ic_1based} - {status}\nr={correlation:.3f} @ {delay:.1f}s", 
                        fontsize=10, color=ic_color, weight='bold')
            
            # Add threshold line reference
            ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
            
            # Formatting
            ax.set_xlabel('Time (s)', fontsize=8)
            ax.set_ylabel('Amplitude', fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)
            
            # Add legend if ETCO₂ is plotted
            if etco2_shifted is not None:
                ax.legend(fontsize=6, loc='upper right')
            
            # Add correlation threshold info in corner
            ax.text(0.02, 0.98, f"thresh={aroma_threshold}", 
                   transform=ax.transAxes, fontsize=7, 
                   verticalalignment='top', alpha=0.7)
        
        # Hide unused subplots
        for i in range(n_ics, len(axes)):
            axes[i].set_visible(False)
        
        # Add overall title with statistics
        total_melodic_ics = self.all_melodic_components.shape[1]
        original_noise_count = len(original_noise_ics)
        refined_noise_count = len(refined_noise_ics)
        restored_count = original_noise_count - refined_noise_count
        
        fig.suptitle(f'MELODIC IC Classification Results\n'
                    f'Total ICs: {total_melodic_ics} | '
                    f'AROMA Noise: {original_noise_count} | '
                    f'Refined Noise: {refined_noise_count} | '
                    f'Restored as Signal: {restored_count}',
                    fontsize=12, weight='bold')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)

        # Save the figure with label entity
        label_entity = self._get_roi_label_entity()
        output_path = figures_dir / f"sub-{self.participant}_task-{self.args.task}{label_entity}_desc-icclassification.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.logger.info(f"IC classification plot saved to: {output_path}")
        
        # Store classification stats for the report
        self.ic_classification_stats = {
            'total_melodic_ics': total_melodic_ics,
            'original_noise_count': original_noise_count,
            'refined_noise_count': refined_noise_count,
            'restored_count': restored_count,
            'aroma_threshold': aroma_threshold
        }
    
    def _nonaggressive_denoising(self):
        """
        Apply non-aggressive denoising to BOLD data using refined AROMA components.
        This removes the contribution of noise ICs while preserving signal components.
        """
        self.logger.debug("Applying non-aggressive denoising...")
        
        import numpy as np
        import pandas as pd
        from sklearn.linear_model import LinearRegression
        
        # Get BOLD data and mask
        bold_data = self.bold_container.data  # Shape: (x, y, z, t)
        mask_data = self.bold_container.mask   # Shape: (x, y, z)
        
        # Use MELODIC mixing matrix stored from _refine_aroma_components
        all_melodic_components = self.all_melodic_components  # Shape: (time, n_all_components)
        
        x, y, z, t = bold_data.shape
        total_voxels = x * y * z
        
        brain_voxels = mask_data > 0
        n_brain_voxels = np.sum(brain_voxels)
        self.logger.debug(f"  Using brain mask: {n_brain_voxels:,} brain voxels out of {total_voxels:,} total voxels")
        
        # Reshape BOLD data to 2D: (n_voxels, time)
        bold_2d = bold_data.reshape(-1, t)
        
        # Initialize output with NaN
        bold_denoised_2d = np.full_like(bold_2d, np.nan)
        
        # Only process brain voxels
        brain_voxel_indices = brain_voxels.ravel()
        bold_brain = bold_2d[brain_voxel_indices, :]  # (n_brain_voxels, time)
        
        noise_component_indices = [int(idx) for idx in self.refined_noise_ics]  # 1-based indices
        
        self.logger.debug(f"  Using ALL {all_melodic_components.shape[1]} MELODIC components in design matrix")
        self.logger.debug(f"  Will remove contribution from {len(noise_component_indices)} noise components: {noise_component_indices}")
        
        # Find the column indices for noise components in the full MELODIC matrix
        # Since AROMA indices are 1-based and MELODIC matrix is 0-based, subtract 1
        noise_column_indices = [idx - 1 for idx in noise_component_indices]
        
        X_all = all_melodic_components  # (time, n_all_components)
        
        # Prepare data for vectorized regression: (time, n_voxels)
        Y = bold_brain.T  # (time, n_voxels)
        
        # Fit LinearRegression model with ALL MELODIC components
        self.logger.debug(f"  Fitting LinearRegression model with {X_all.shape[1]} components on {Y.shape[1]} voxels")
        reg = LinearRegression(fit_intercept=False)  # No intercept needed as MELODIC components include mean
        reg.fit(X_all, Y)
        
        # Get fitted coefficients: (n_all_components, n_voxels)
        all_params = reg.coef_.T  # (n_all_components, n_voxels)
        
        # Extract parameters for noise components only: (n_noise_components, n_voxels)
        noise_params = all_params[noise_column_indices, :]  # (n_noise_components, n_voxels)
        
        # Get noise component timeseries: (time, n_noise_components)
        noise_values = all_melodic_components[:, noise_column_indices]  # (time, n_noise_components)
        
        # Compute noise contribution: (time, n_voxels)
        noise_contrib = noise_values @ noise_params  # (time, n_voxels)
        
        # Remove noise contribution from signal: (time, n_voxels)
        denoised = Y - noise_contrib
        
        # Put denoised brain voxels back into the full array: (n_voxels, time)
        bold_brain[:, :] = denoised.T
        bold_denoised_2d[brain_voxel_indices, :] = bold_brain
        
        # Reshape back to 4D
        bold_denoised = bold_denoised_2d.reshape(x, y, z, t)
        
        # Update the container with denoised data
        self.bold_container.data = bold_denoised
        
        # Create a new BoldContainer for the denoised data
        from .data_container import BoldContainer
        self.bold_denoised = BoldContainer(
            participant=self.bold_container.participant,
            task=self.bold_container.task,
            space=self.bold_container.space,
            data=bold_denoised,
            path=self.bold_container.path,
            layout=self.bold_container.layout
        )
        # Copy additional attributes from original container
        self.bold_denoised.affine = self.bold_container.affine
        self.bold_denoised.header = self.bold_container.header
        self.bold_denoised.tr = self.bold_container.tr
        self.bold_denoised.sampling_frequency = self.bold_container.sampling_frequency
        self.bold_denoised.mask = self.bold_container.mask
        self.bold_denoised.mask_path = self.bold_container.mask_path
        
        self.logger.info(f"Non-aggressive denoising completed. Removed contribution from {len(noise_component_indices)} noise components.")
    
    def _temporal_filtering(self):
        """
        Apply temporal filtering to denoised BOLD data with DC component restoration.
        This applies a high-pass filter by subtracting a low-pass filtered version,
        then restores the DC component from the original denoised data.
        """
        self.logger.debug("Applying temporal filtering with DC restoration...")
        
        import numpy as np
        from scipy.ndimage import gaussian_filter
        
        # Get temporal filtering parameters from config
        sigma_seconds = self.config['bold']['temporal_filtering']['sigma']
        self.logger.debug(f"Using temporal filtering sigma: {sigma_seconds} seconds")
        
        # Use the denoised BOLD data
        bold_data = self.bold_denoised.data  # Shape: (x, y, z, t)
        mask_data = self.bold_denoised.mask  # Shape: (x, y, z)
        sampling_frequency = self.bold_denoised.sampling_frequency  # Hz
        
        # Convert sigma from seconds to samples
        sigma_samples = sigma_seconds * sampling_frequency
        self.logger.debug(f"Sigma in samples: {sigma_samples:.2f}")
        
        x, y, z, t = bold_data.shape
        
        brain_voxels = mask_data > 0
        n_brain_voxels = np.sum(brain_voxels)
        self.logger.debug(f"  Applying temporal filtering to {n_brain_voxels:,} brain voxels")
        
        # Reshape BOLD data to 2D: (n_voxels, time)
        bold_2d = bold_data.reshape(-1, t)
        
        # Initialize output with NaN
        bold_filtered_2d = np.full_like(bold_2d, np.nan)
        
        # Only process brain voxels
        brain_voxel_indices = brain_voxels.ravel()
        bold_brain = bold_2d[brain_voxel_indices, :]  # (n_brain_voxels, time)
        
        # Extract DC component (temporal average) from original denoised data BEFORE filtering
        dc_component = np.mean(bold_brain, axis=1, keepdims=True)  # (n_brain_voxels, 1)
        self.logger.debug(f"  Extracted DC component for {n_brain_voxels:,} brain voxels")
        
        # Apply temporal filtering to each voxel
        bold_brain_filtered = np.zeros_like(bold_brain)
        
        for voxel_idx in range(bold_brain.shape[0]):
            signal = bold_brain[voxel_idx, :]
            
            # Apply Gaussian low-pass filter
            lowpass = gaussian_filter(signal, sigma=sigma_samples)
            
            # High-pass filter by subtracting low-pass
            filtered_signal = signal - lowpass
            
            # Remove mean
            filtered_signal = filtered_signal - np.mean(filtered_signal)
            
            bold_brain_filtered[voxel_idx, :] = filtered_signal
        
        # Restore DC component to filtered data
        bold_brain_with_dc = bold_brain_filtered + dc_component  # (n_brain_voxels, time)
        self.logger.debug(f"  Restored DC component to filtered data")
        
        # Put filtered brain voxels back into the full array
        bold_filtered_2d[brain_voxel_indices, :] = bold_brain_with_dc
        
        # Reshape back to 4D
        bold_filtered = bold_filtered_2d.reshape(x, y, z, t)
        
        # Update the bold_denoised container with filtered data
        self.bold_denoised.data = bold_filtered
        
        self.logger.info(f"Temporal filtering with DC restoration completed. Applied {sigma_seconds}s sigma filter to {n_brain_voxels:,} brain voxels.")
        self.logger.debug(f"Updated self.bold_denoised container with filtered and DC-restored data")
    
    def _spatial_smooth(self):
        """
        Apply spatial smoothing to BOLD data using nilearn.
        This applies Gaussian spatial smoothing with the specified FWHM.
        """
        self.logger.debug("Applying spatial smoothing...")
        
        import numpy as np
        import nibabel as nib
        from nilearn.image import smooth_img
        
        # Get spatial smoothing parameters from config
        fwhm_mm = self.config['bold']['spatial_smoothing']['fwhm']
        self.logger.debug(f"Using spatial smoothing FWHM: {fwhm_mm} mm")
        
        # Use the denoised and filtered BOLD data
        bold_data = self.bold_denoised.data  # Shape: (x, y, z, t)
        mask_data = self.bold_denoised.mask  # Shape: (x, y, z)
        affine = self.bold_denoised.affine
        header = self.bold_denoised.header
        
        x, y, z, t = bold_data.shape
        
        brain_voxels = mask_data > 0
        n_brain_voxels = np.sum(brain_voxels)
        self.logger.debug(f"  Applying spatial smoothing to {n_brain_voxels:,} brain voxels")
        
        # Create NIfTI image from BOLD data
        bold_img = nib.Nifti1Image(bold_data, affine, header)
        
        # Apply spatial smoothing using nilearn
        self.logger.debug(f"  Smoothing with {fwhm_mm}mm FWHM...")
        smoothed_img = smooth_img(bold_img, fwhm=fwhm_mm)
        
        # Extract smoothed data
        smoothed_data = smoothed_img.get_fdata()
        
        # Apply mask to ensure we only keep brain voxels
        # Set non-brain voxels back to NaN
        smoothed_data_masked = smoothed_data.copy()
        for t_idx in range(t):
            smoothed_data_masked[~brain_voxels, t_idx] = np.nan
        
        # Update the bold_denoised container with smoothed data
        self.bold_denoised.data = smoothed_data_masked
        
        self.logger.info(f"Spatial smoothing completed. Applied {fwhm_mm}mm FWHM smoothing to {n_brain_voxels:,} brain voxels.")
        self.logger.debug(f"Updated self.bold_denoised container with spatially smoothed data")
