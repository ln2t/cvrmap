"""
Cross-correlation utilities for CVR analysis.

This module provides functions for cross-correlating physiological probe signals
with BOLD timecourses and other signals for CVR analysis.
"""
import numpy as np  # Keep this here to improve performance

def cross_correlate(reference_container, shifted_probes_data, logger=None, config=None):
    """
    Cross-correlate a reference signal with multiple shifted probe signals.
    
    Parameters:
    -----------
    reference_container : DataContainer
        Container with the reference signal (e.g., normalized IC timecourse)
    shifted_probes_data : tuple
        Tuple containing:
        - shifted_signals: 2D numpy array (n_delays, n_timepoints)
        - time_delays_seconds: 1D numpy array of time delays in seconds
    logger : Logger, optional
        Logger instance for debugging
    config : dict, optional
        Configuration dictionary (not used in current implementation)
    
    Returns:
    --------
    tuple
        - best_correlation: float, highest correlation found
        - best_delay_seconds: float, delay in seconds corresponding to best correlation
    """
    
    best_correlation = 0.0
    best_delay_seconds = None
    
    reference_signal = reference_container.data

    # Unpack the shifted probes data
    shifted_signals, time_delays_seconds = shifted_probes_data

    if shifted_signals is None or time_delays_seconds is None:
        return 0.0, 0.0

    n_delays = shifted_signals.shape[0]

    for i in range(n_delays):
        # Both signals are normalized and of same length, so correlation is simply dot product / n_samples
        probe_signal = shifted_signals[i, :]

        # Calculate correlation for normalized signals: dot product / n_samples
        correlation = np.dot(reference_signal, probe_signal) / len(reference_signal)

        if correlation > best_correlation:
            best_correlation = correlation
            best_delay_seconds = time_delays_seconds[i]
    
    if best_delay_seconds is None:
        best_delay_seconds = 0.0
    
    return best_correlation, best_delay_seconds