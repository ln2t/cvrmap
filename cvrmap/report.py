"""
HTML Report Generation for CVR Analysis

This module handles the creation of comprehensive HTML reports summarizing
CVR analysis results for individual subjects and studies.
"""

import os
from pathlib import Path
from datetime import datetime

from . import __version__


class CVRReportGenerator:
    """
    Generator for comprehensive HTML reports of CVR analysis results.
    
    This class creates detailed HTML reports that summarize the CVR analysis
    results for individual subjects, including figures, statistics, and 
    metadata from the processing pipeline.
    """
    
    def __init__(self, participant_id, task, output_dir, logger=None, config=None):
        """
        Initialize the CVR Report Generator.
        
        Parameters:
        -----------
        participant_id : str
            Subject identifier (e.g., '001')
        task : str
            Task name (e.g., 'gas')
        output_dir : str
            Base output directory for BIDS derivatives
        logger : Logger, optional
            Logger instance for debugging and progress tracking
        config : dict, optional
            Configuration dictionary containing pipeline parameters
        """
        self.participant_id = participant_id
        self.task = task
        self.output_dir = output_dir
        self.logger = logger
        self.config = config

        # Set up report paths
        self.participant_dir = os.path.join(output_dir, f"sub-{participant_id}")

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
    
    def generate_report(self, **kwargs):
        """
        Generate a comprehensive HTML report for CVR analysis results.
        
        Parameters:
        -----------
        **kwargs : dict
            Additional data and results to include in the report.
            Expected keys might include:
            - global_delay: Global delay value
            - physio_results: Physiological processing results
            - bold_results: BOLD processing results
            - delay_results: Delay analysis results
            - cvr_results: CVR analysis results
            - figures_info: Information about generated figures
        
        Returns:
        --------
        str
            Path to the generated HTML report file
        """
        if self.logger:
            self.logger.info(f"Generating CVR analysis report for participant {self.participant_id}")
        
        # TODO: Implement custom HTML report generation
        # Placeholder for now - this is where we'll build the complete HTML report
        
        # Create the main HTML report file
        report_html = self._create_main_report_html(**kwargs)
        
        # Write the report to file with label entity
        label_entity = self._get_roi_label_entity()
        report_filename = f"sub-{self.participant_id}_task-{self.task}{label_entity}_report.html"
        report_path = os.path.join(self.participant_dir, report_filename)
        
        with open(report_path, 'w') as f:
            f.write(report_html)
        
        if self.logger:
            self.logger.info(f"Generated HTML report at: {report_path}")
        
        return report_path
    
    def _create_main_report_html(self, **kwargs):
        """
        Create the main HTML report content.
        
        Parameters:
        -----------
        **kwargs : dict
            Analysis results and metadata for the report
            
        Returns:
        --------
        str
            Complete HTML content for the report
        """
        # Extract information from kwargs
        subject_label = self.participant_id
        task = self.task
        spaces = kwargs.get('space', 'MNI152NLin2009cAsym')
        version = __version__
        global_delay = kwargs.get('global_delay', 0.0)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Extract denoising parameters from config (with fallback defaults if config not available)
        if self.config:
            # Get parameters from actual config
            aroma_threshold = self.config.get('bold', {}).get('denoising', {}).get('aroma_correlation_threshold', 0.3)
            temporal_sigma = self.config.get('bold', {}).get('temporal_filtering', {}).get('sigma', 100)
            spatial_fwhm = self.config.get('bold', {}).get('spatial_smoothing', {}).get('fwhm', 6)
        else:
            # Fallback to kwargs or defaults if config not available
            aroma_threshold = kwargs.get('aroma_correlation_threshold', 0.3)
            temporal_sigma = kwargs.get('temporal_filtering_sigma', 100)
            spatial_fwhm = kwargs.get('spatial_smoothing_fwhm', 6)
        
        # Get figure paths - check for both physio and ROI probe figures
        figures_dir = os.path.join(self.participant_dir, 'figures')
        label_entity = self._get_roi_label_entity()
        physio_figure = f"sub-{subject_label}_task-{task}_desc-physio.png"
        roi_probe_figure = f"sub-{subject_label}_task-{task}{label_entity}_desc-roiprobe.png"
        roi_visualization_figure = f"sub-{subject_label}_task-{task}{label_entity}_desc-roivisualization.png"
        global_figure = f"sub-{subject_label}_task-{task}{label_entity}_space-{spaces}_desc-globalcorr.png"
        delay_figure = f"sub-{subject_label}_task-{task}{label_entity}_space-{spaces}_desc-delaymasked.png"
        cvr_figure = f"sub-{subject_label}_task-{task}{label_entity}_space-{spaces}_desc-cvr.png"
        ic_classification_figure = f"sub-{subject_label}_task-{task}{label_entity}_desc-icclassification.png"
        delay_histogram_figure = f"sub-{subject_label}_task-{task}{label_entity}_desc-delayhist.png"
        cvr_histogram_figure = f"sub-{subject_label}_task-{task}{label_entity}_desc-cvrhist.png"
        
        # Check which figures exist
        physio_exists = os.path.exists(os.path.join(figures_dir, physio_figure))
        roi_probe_exists = os.path.exists(os.path.join(figures_dir, roi_probe_figure))
        roi_visualization_exists = os.path.exists(os.path.join(figures_dir, roi_visualization_figure))
        global_exists = os.path.exists(os.path.join(figures_dir, global_figure))
        delay_exists = os.path.exists(os.path.join(figures_dir, delay_figure))
        cvr_exists = os.path.exists(os.path.join(figures_dir, cvr_figure))
        ic_classification_exists = os.path.exists(os.path.join(figures_dir, ic_classification_figure))
        delay_histogram_exists = os.path.exists(os.path.join(figures_dir, delay_histogram_figure))
        cvr_histogram_exists = os.path.exists(os.path.join(figures_dir, cvr_histogram_figure))
        
        # Determine probe mode for appropriate labeling
        roi_probe_enabled = self.config.get('roi_probe', {}).get('enabled', False) if self.config else False
        probe_mode = "ROI" if roi_probe_enabled else "Physiological"

        # Get ROI configuration details for the report
        roi_config = self.config.get('roi_probe', {}) if self.config else {}
        roi_method = roi_config.get('method', 'Unknown')
        roi_label = roi_config.get('label', '')
        roi_mask_path = roi_config.get('mask_path', '')

        # Build ROI info HTML for mask method
        roi_mask_info_html = ""
        if roi_probe_enabled and roi_method == 'mask' and roi_mask_path:
            roi_mask_info_html = f"""
                <div class="summary-card" style="margin-bottom: 1.5rem; background: #e8f4f8; border-left: 4px solid #17a2b8;">
                    <h4 style="margin-bottom: 0.5rem; color: #0c5460;">ROI Mask Information</h4>
                    <p style="margin-bottom: 0.5rem;"><strong>Label:</strong> {roi_label}</p>
                    <p style="margin-bottom: 0; word-break: break-all;"><strong>Mask Path:</strong> <code style="background: #fff; padding: 2px 6px; border-radius: 3px; font-size: 0.85em;">{roi_mask_path}</code></p>
                </div>
            """

        # Get IC classification stats
        ic_stats = kwargs.get('bold_results', {}).get('ic_classification_stats', None)
        
        # Get histogram statistics
        histogram_stats = kwargs.get('histogram_stats', {})
        
        # Build delay statistics HTML section
        delay_stats_html = ""
        if histogram_stats.get('delay_stats'):
            ds = histogram_stats['delay_stats']
            delay_stats_html = f"""
                        <div style="margin-top: 1rem;">
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                                <div style="text-align: center; padding: 0.75rem; background: #e8f4f8; border-radius: 6px;">
                                    <div style="font-size: 1.2em; font-weight: bold; color: #17a2b8;">{ds['mean']:.2f}s</div>
                                    <div style="font-size: 0.85em; color: #666;">Mean Delay</div>
                                </div>
                                <div style="text-align: center; padding: 0.75rem; background: #f8f9fa; border-radius: 6px;">
                                    <div style="font-size: 1.2em; font-weight: bold; color: #6c757d;">{ds['std']:.2f}s</div>
                                    <div style="font-size: 0.85em; color: #666;">Standard Dev</div>
                                </div>
                                <div style="text-align: center; padding: 0.75rem; background: #fff3cd; border-radius: 6px;">
                                    <div style="font-size: 1.2em; font-weight: bold; color: #856404;">{ds['median']:.2f}s</div>
                                    <div style="font-size: 0.85em; color: #666;">Median</div>
                                </div>
                                <div style="text-align: center; padding: 0.75rem; background: #d4edda; border-radius: 6px;">
                                    <div style="font-size: 1.2em; font-weight: bold; color: #155724;">{ds['n_voxels']:,}</div>
                                    <div style="font-size: 0.85em; color: #666;">Brain Voxels</div>
                                </div>
                            </div>
                            <div style="margin-top: 1rem; padding: 0.75rem; background: #f8f9fa; border-radius: 6px; font-size: 0.9em;">
                                <strong>Range:</strong> [{ds['min']:.2f}, {ds['max']:.2f}] seconds<br>
                                <strong>IQR:</strong> [{ds['q25']:.2f}, {ds['q75']:.2f}] seconds
                            </div>
                        </div>
                        """
        else:
            delay_stats_html = "<p style='color: #666; font-style: italic;'>Delay statistics not available</p>"
        
        # Build CVR statistics HTML section
        cvr_stats_html = ""
        if histogram_stats.get('cvr_stats'):
            cs = histogram_stats['cvr_stats']
            cvr_units = "arbitrary units" if roi_probe_enabled else "%BOLD/mmHg"
            cvr_stats_html = f"""
                        <div style="margin-top: 1rem;">
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                                <div style="text-align: center; padding: 0.75rem; background: #d4edda; border-radius: 6px;">
                                    <div style="font-size: 1.2em; font-weight: bold; color: #155724;">{cs['mean']:.4f}</div>
                                    <div style="font-size: 0.85em; color: #666;">Mean CVR</div>
                                </div>
                                <div style="text-align: center; padding: 0.75rem; background: #f8f9fa; border-radius: 6px;">
                                    <div style="font-size: 1.2em; font-weight: bold; color: #6c757d;">{cs['std']:.4f}</div>
                                    <div style="font-size: 0.85em; color: #666;">Standard Dev</div>
                                </div>
                                <div style="text-align: center; padding: 0.75rem; background: #fff3cd; border-radius: 6px;">
                                    <div style="font-size: 1.2em; font-weight: bold; color: #856404;">{cs['median']:.4f}</div>
                                    <div style="font-size: 0.85em; color: #666;">Median</div>
                                </div>
                                <div style="text-align: center; padding: 0.75rem; background: #e8f4f8; border-radius: 6px;">
                                    <div style="font-size: 1.2em; font-weight: bold; color: #17a2b8;">{cs['n_voxels']:,}</div>
                                    <div style="font-size: 0.85em; color: #666;">Brain Voxels</div>
                                </div>
                            </div>
                            <div style="margin-top: 1rem; padding: 0.75rem; background: #f8f9fa; border-radius: 6px; font-size: 0.9em;">
                                <strong>Range:</strong> [{cs['min']:.4f}, {cs['max']:.4f}] {cvr_units}<br>
                                <strong>IQR:</strong> [{cs['q25']:.4f}, {cs['q75']:.4f}] {cvr_units}
                            </div>
                        </div>
                        """
        else:
            cvr_stats_html = "<p style='color: #666; font-style: italic;'>CVR statistics not available</p>"
        
        # Build ROI visualization nav link (can't have backslashes in f-string expressions)
        roi_vis_link = '<a href="#roi-visualization">ROI Visualization</a>' if roi_probe_enabled else ""
        
        html_template = f"""<!DOCTYPE html>>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CVR Analysis Report - Subject {subject_label}</title>
    <style>
        /* Modern scientific report styling */
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f8f9fa;
        }}
        
        /* Top navigation bar */
        .navbar {{
            background: linear-gradient(135deg, #2c3e50, #3498db);
            color: white;
            padding: 1rem 0;
            position: sticky;
            top: 0;
            z-index: 1000;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .nav-container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .nav-title {{
            font-size: 1.5rem;
            font-weight: bold;
        }}
        
        .nav-links {{
            display: flex;
            gap: 2rem;
        }}
        
        .nav-links a {{
            color: white;
            text-decoration: none;
            padding: 0.5rem 1rem;
            border-radius: 5px;
            transition: background-color 0.3s;
        }}
        
        .nav-links a:hover {{
            background-color: rgba(255,255,255,0.2);
        }}
        
        /* Main content container */
        .container {{
            max-width: 1200px;
            margin: 2rem auto;
            padding: 0 2rem;
        }}
        
        /* Section styling */
        .section {{
            background: white;
            margin-bottom: 2rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        
        .section-header {{
            background: linear-gradient(135deg, #34495e, #2c3e50);
            color: white;
            padding: 1.5rem 2rem;
            border-bottom: 3px solid #3498db;
        }}
        
        .section-title {{
            font-size: 1.8rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }}
        
        .section-subtitle {{
            font-size: 1rem;
            opacity: 0.9;
        }}
        
        .section-content {{
            padding: 2rem;
        }}
        
        /* Summary section styling */
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}
        
        .summary-card {{
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }}
        
        .summary-card h4 {{
            color: #2c3e50;
            margin-bottom: 0.5rem;
            font-size: 1.1rem;
        }}
        
        .summary-card p {{
            color: #555;
            font-size: 1.1rem;
            font-weight: 500;
        }}
        
        /* Figure styling */
        .figure-container {{
            text-align: center;
            margin: 2rem 0;
        }}
        
        .figure-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        
        .figure-caption {{
            margin-top: 1rem;
            font-style: italic;
            color: #666;
            font-size: 0.95rem;
            line-height: 1.5;
            max-width: 800px;
            margin-left: auto;
            margin-right: auto;
        }}
        
        /* Processing details styling */
        .details-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
        }}
        
        .detail-item {{
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 6px;
            border: 1px solid #e9ecef;
        }}
        
        .detail-item strong {{
            color: #2c3e50;
        }}
        
        /* Error/warning messages */
        .warning {{
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
            padding: 1rem;
            border-radius: 6px;
            margin: 1rem 0;
        }}
        
        /* Responsive design */
        @media (max-width: 768px) {{
            .nav-container {{
                flex-direction: column;
                gap: 1rem;
            }}
            
            .nav-links {{
                gap: 1rem;
            }}
            
            .container {{
                padding: 0 1rem;
            }}
            
            .section-content {{
                padding: 1rem;
            }}
        }}
        
        /* Smooth scrolling */
        html {{
            scroll-behavior: smooth;
        }}
        
        /* Section targets for navigation */
        .section {{
            scroll-margin-top: 80px;
        }}
    </style>
</head>
<body>
    <!-- Navigation Bar -->
    <nav class="navbar">
        <div class="nav-container">
            <div class="nav-title">CVR Analysis Report</div>
            <div class="nav-links">
                <a href="#summary">Summary</a>
                <a href="#physiological">{"ROI Probe" if roi_probe_enabled else "Physiological"}</a>
                {roi_vis_link}
                <a href="#denoising">Denoising</a>
                <a href="#global-delay">Global Delay</a>
                <a href="#delay-maps">Delay Maps</a>
                <a href="#cvr-maps">CVR Maps</a>
                <a href="#statistics">Statistics</a>
                <a href="#references">References</a>
            </div>
        </div>
    </nav>
    
    <!-- Main Content -->
    <div class="container">
        <!-- Summary Section -->
        <section id="summary" class="section">
            <div class="section-header">
                <h2 class="section-title">Analysis Summary</h2>
                <p class="section-subtitle">Overview of CVR analysis parameters and results</p>
            </div>
            <div class="section-content">
                <div class="summary-grid">
                    <div class="summary-card">
                        <h4>Subject Information</h4>
                        <p>Subject {subject_label}</p>
                    </div>
                    <div class="summary-card">
                        <h4>Functional Task</h4>
                        <p>{task}</p>
                    </div>
                    <div class="summary-card">
                        <h4>Output Space</h4>
                        <p>{spaces}</p>
                    </div>
                    <div class="summary-card">
                        <h4>CVRmap Version</h4>
                        <p>{version}</p>
                    </div>
                    <div class="summary-card">
                        <h4>Global Delay</h4>
                        <p>{global_delay:.3f} seconds</p>
                    </div>
                    <div class="summary-card">
                        <h4>Report Generated</h4>
                        <p>{timestamp}</p>
                    </div>
                </div>
                <div class="details-grid">
                    <div class="detail-item">
                        <strong>Analysis Type:</strong> Cerebrovascular Reactivity Mapping
                    </div>
                    <div class="detail-item">
                        <strong>Preprocessing:</strong> BIDS-compatible pipeline with AROMA denoising
                    </div>
                    <div class="detail-item">
                        <strong>Signal Processing:</strong> {"ROI-based probe extraction and optimization" if roi_probe_enabled else "End-tidal CO₂ extraction and optimization"}
                    </div>
                    <div class="detail-item">
                        <strong>Spatial Normalization:</strong> {spaces} standard space
                    </div>
                </div>
            </div>
        </section>
        
        <!-- Probe Data Section -->
        <section id="physiological" class="section">
            <div class="section-header">
                <h2 class="section-title">{probe_mode} Probe Analysis</h2>
                <p class="section-subtitle">{"ROI-based probe signal extraction and processing" if roi_probe_enabled else "Breathing signal processing and CO₂ trace extraction"}</p>
            </div>
            <div class="section-content">
                {roi_mask_info_html}
                {("<div class='figure-container'>" +
                f"<img src='figures/{roi_probe_figure}' alt='ROI Probe Analysis' />" +
                "<div class='figure-caption'>" +
                "This graph shows the ROI-based probe signal extracted from the specified brain region. " +
                "The signal is averaged across all voxels within the ROI and serves as an alternative to physiological recordings " +
                "for CVR analysis. The baseline represents the mean signal level used for CVR computations." +
                "</div></div>") if roi_probe_enabled and roi_probe_exists else
                ("<div class='figure-container'>" +
                f"<img src='figures/{physio_figure}' alt='Physiological Data Analysis' />" +
                "<div class='figure-caption'>" +
                "This graph shows the original breathing data with the reconstructed upper envelope and corresponding baseline. " +
                "The end-tidal CO₂ (ETCO₂) trace is extracted from the breathing signal peaks and represents the CO₂ concentration " +
                "at the end of each expiration, which serves as the regressor for CVR analysis." +
                "</div></div>" if physio_exists else
                "<div class='warning'>Physiological data figure not found. Please ensure the analysis completed successfully.</div>")}
            </div>
        </section>
        
        <!-- ROI Visualization Section (only shown for ROI probe mode) -->
        {f'''
        <section id="roi-visualization" class="section">
            <div class="section-header">
                <h2 class="section-title">ROI Visualization</h2>
                <p class="section-subtitle">Visual confirmation of the selected region of interest</p>
            </div>
            <div class="section-content">
                {("<div class='figure-container'>" + 
                f"<img src='figures/{roi_visualization_figure}' alt='ROI Visualization' />" +
                "<div class='figure-caption'>" +
                "This figure shows the selected region of interest (ROI) overlaid on the mean preprocessed BOLD image. " +
                "The ROI is displayed in red across axial, sagittal, and coronal views to confirm proper localization. " +
                "ROI statistics including volume and number of voxels are provided along with the extracted signal time course." +
                "</div></div>") if roi_visualization_exists else 
                "<div class='warning'>ROI visualization figure not found. Please ensure the ROI was properly defined.</div>"}
            </div>
        </section>
        ''' if roi_probe_enabled else ''}
        
        <!-- Denoising Section -->
        <section id="denoising" class="section">
            <div class="section-header">
                <h2 class="section-title">BOLD Signal Denoising</h2>
                <p class="section-subtitle">Multi-step preprocessing pipeline for noise removal</p>
            </div>
            <div class="section-content">
                <div class="summary-card" style="margin-bottom: 2rem;">
                    <h4>Denoising Pipeline Summary</h4>
                    <p>The BOLD signal undergoes a comprehensive 4-step denoising process specifically designed for CVR analysis.
                    The pipeline intelligently preserves {"probe-correlated" if roi_probe_enabled else "CO₂-related"} signals while removing motion artifacts, physiological noise,
                    and scanner-related artifacts. This approach ensures optimal signal quality for reliable CVR quantification
                    while maintaining the temporal dynamics essential for hemodynamic delay estimation.</p>
                </div>
                
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 2rem;">
                    <div style="padding: 1rem; background: #f8f9fa; border-radius: 6px; border-left: 4px solid #28a745;">
                        <strong>1. AROMA Component Refinement</strong><br>
                        <span style="font-size: 0.9em; color: #666;">Cross-correlation analysis between MELODIC ICs and {"probe" if roi_probe_enabled else "ETCO₂"} signals. Components with correlation > {aroma_threshold} are preserved as signal.</span>
                    </div>
                    <div style="padding: 1rem; background: #f8f9fa; border-radius: 6px; border-left: 4px solid #ffc107;">
                        <strong>2. Non-Aggressive Denoising</strong><br>
                        <span style="font-size: 0.9em; color: #666;">Selective removal of refined noise components using linear regression while preserving {"probe-correlated" if roi_probe_enabled else "ETCO₂-correlated"} components.</span>
                    </div>
                    <div style="padding: 1rem; background: #f8f9fa; border-radius: 6px; border-left: 4px solid #17a2b8;">
                        <strong>3. Temporal Filtering</strong><br>
                        <span style="font-size: 0.9em; color: #666;">High-pass filtering with DC restoration using Gaussian filter (σ = {temporal_sigma}s) to remove slow drifts.</span>
                    </div>
                    <div style="padding: 1rem; background: #f8f9fa; border-radius: 6px; border-left: 4px solid #6f42c1;">
                        <strong>4. Spatial Smoothing</strong><br>
                        <span style="font-size: 0.9em; color: #666;">Gaussian spatial smoothing with {spatial_fwhm}mm FWHM kernel for improved signal-to-noise ratio.</span>
                    </div>
                </div>
                
                {f'''
                <div class="summary-card" style="margin-bottom: 2rem;">
                    <h4>AROMA Component Classification Statistics</h4>
                    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-top: 1rem;">
                        <div style="text-align: center; padding: 0.75rem; background: #e8f4f8; border-radius: 6px;">
                            <div style="font-size: 1.5em; font-weight: bold; color: #17a2b8;">{ic_stats['total_melodic_ics']}</div>
                            <div style="font-size: 0.85em; color: #666;">Total MELODIC ICs</div>
                        </div>
                        <div style="text-align: center; padding: 0.75rem; background: #fff3cd; border-radius: 6px;">
                            <div style="font-size: 1.5em; font-weight: bold; color: #856404;">{ic_stats['original_noise_count']}</div>
                            <div style="font-size: 0.85em; color: #666;">AROMA Noise ICs</div>
                        </div>
                        <div style="text-align: center; padding: 0.75rem; background: #f8d7da; border-radius: 6px;">
                            <div style="font-size: 1.5em; font-weight: bold; color: #721c24;">{ic_stats['refined_noise_count']}</div>
                            <div style="font-size: 0.85em; color: #666;">Final Noise ICs</div>
                        </div>
                        <div style="text-align: center; padding: 0.75rem; background: #d4edda; border-radius: 6px;">
                            <div style="font-size: 1.5em; font-weight: bold; color: #155724;">{ic_stats['restored_count']}</div>
                            <div style="font-size: 0.85em; color: #666;">Restored as Signal</div>
                        </div>
                    </div>
                    <p style="margin-top: 1rem; font-size: 0.9em; color: #666;">
                        AROMA initially classified {ic_stats['original_noise_count']} components as noise.
                        After cross-correlation analysis with {"probe" if roi_probe_enabled else "ETCO₂"} signals using threshold {ic_stats['aroma_threshold']},
                        {ic_stats['restored_count']} components were reclassified as signal and preserved.
                    </p>
                </div>
                ''' if ic_stats else ''}
                
                {f'''
                <div class="figure-container">
                    <img src="figures/{ic_classification_figure}" alt="IC Classification Results" />
                    <div class="figure-caption">
                        Individual MELODIC independent component timecourses showing their classification status.
                        Green traces indicate components reclassified as signal (correlation > {aroma_threshold}),
                        while red traces show components retained as noise. Each plot shows the maximum cross-correlation
                        with {"probe" if roi_probe_enabled else "ETCO₂"} signals and the corresponding delay.
                    </div>
                </div>
                ''' if ic_classification_exists else ''}
            </div>
        </section>
        
        <!-- Global Delay Analysis Section -->
        <section id="global-delay" class="section">
            <div class="section-header">
                <h2 class="section-title">Global Signal Analysis</h2>
                <p class="section-subtitle">BOLD signal correlation with {"probe" if roi_probe_enabled else "CO₂"} and optimal delay estimation</p>
            </div>
            <div class="section-content">
                {"<div class='figure-container'>" +
                f"<img src='figures/{global_figure}' alt='Global Signal Analysis' />" +
                "<div class='figure-caption'>" +
                f"This graph shows the normalized global BOLD signal and the optimally shifted {'probe' if roi_probe_enabled else 'ETCO₂'} signal with their cross-correlation. " +
                f"The optimal delay of {global_delay:.3f} seconds represents the hemodynamic lag between {'probe' if roi_probe_enabled else 'CO₂'} changes and BOLD signal response. " +
                "This global delay is used as the starting point for voxel-wise delay optimization." +
                "</div></div>" if global_exists else
                "<div class='warning'>Global signal analysis figure not found. Please ensure the analysis completed successfully.</div>"}
            </div>
        </section>
        
        <!-- Delay Mapping Section -->
        <section id="delay-maps" class="section">
            <div class="section-header">
                <h2 class="section-title">Hemodynamic Delay Mapping</h2>
                <p class="section-subtitle">Voxel-wise optimal delay estimation</p>
            </div>
            <div class="section-content">
                {"<div class='figure-container'>" + 
                f"<img src='figures/{delay_figure}' alt='Delay Mapping Results' />" +
                "<div class='figure-caption'>" +
                "This brain map shows the optimal hemodynamic delay at each voxel, masked by correlation threshold. " +
                f"Delays represent the time lag between {'probe' if roi_probe_enabled else 'CO₂ stimulus'} and peak BOLD response. " +
                "Brain slices are displayed in neurological convention (left on figure = left hemisphere). " +
                "Warmer colors indicate longer delays, while cooler colors represent shorter delays." +
                "</div></div>" if delay_exists else 
                "<div class='warning'>Delay mapping figure not found. Please ensure the analysis completed successfully.</div>"}
            </div>
        </section>
        
        <!-- CVR Mapping Section -->
        <section id="cvr-maps" class="section">
            <div class="section-header">
                <h2 class="section-title">Cerebrovascular Reactivity Maps</h2>
                <p class="section-subtitle">Quantitative CVR analysis results</p>
            </div>
            <div class="section-content">
                {"<div class='figure-container'>" + 
                f"<img src='figures/{cvr_figure}' alt='CVR Analysis Results' />" +
                "<div class='figure-caption'>" +
                ("This brain map shows the cerebrovascular reactivity (CVR) values in arbitrary units. " +
                "CVR quantifies the ability of cerebral blood vessels to respond to changes in the ROI probe signal. " +
                "Higher values (warmer colors) indicate greater vascular reactivity, while lower values (cooler colors) " +
                "suggest reduced vascular responsiveness. Note: CVR values are in arbitrary units since the ROI probe " +
                "signal lacks physiological calibration." if roi_probe_enabled else
                "This brain map shows the cerebrovascular reactivity (CVR) values in percentage change per mmHg CO₂. " +
                "CVR quantifies the ability of cerebral blood vessels to respond to CO₂ changes. " +
                "Higher values (warmer colors) indicate greater vascular reactivity, while lower values (cooler colors) " +
                "suggest reduced vascular responsiveness.") +
                " Brain slices are displayed in neurological convention " +
                "(left on figure = left hemisphere)." +
                "</div></div>" if cvr_exists else 
                "<div class='warning'>CVR mapping figure not found. Please ensure the analysis completed successfully.</div>"}
            </div>
        </section>
        
        <!-- Statistics Section -->
        <section id="statistics" class="section">
            <div class="section-header">
                <h2 class="section-title">Statistical Summary</h2>
                <p class="section-subtitle">Quantitative analysis of delay and CVR map distributions</p>
            </div>
            <div class="section-content">
                {f'''
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; margin-bottom: 2rem;">
                    <!-- Delay Statistics -->
                    <div class="summary-card">
                        <h4><i class="fas fa-clock"></i> Hemodynamic Delay Statistics</h4>
                        {delay_stats_html}
                    </div>
                    
                    <!-- CVR Statistics -->
                    <div class="summary-card">
                        <h4><i class="fas fa-brain"></i> CVR Statistics</h4>
                        {cvr_stats_html}
                    </div>
                </div>
                ''' if histogram_stats else ''}
                
                <!-- Histogram Figures -->
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem;">
                    {f'''
                    <div class="figure-container">
                        <img src="figures/{delay_histogram_figure}" alt="Delay Distribution Histogram" />
                        <div class="figure-caption">
                            Distribution of hemodynamic delays across all brain voxels. The histogram shows the 
                            frequency of different delay values, with vertical lines indicating the mean (red) and 
                            median (orange) delays. Statistics box provides summary measures of the distribution.
                        </div>
                    </div>
                    ''' if delay_histogram_exists else ''}
                    
                    {f'''
                    <div class="figure-container">
                        <img src="figures/{cvr_histogram_figure}" alt="CVR Distribution Histogram" />
                        <div class="figure-caption">
                            Distribution of cerebrovascular reactivity (CVR) values across all brain voxels. 
                            The histogram illustrates the range and frequency of CVR responses, with summary 
                            statistics showing the central tendency and variability of vascular reactivity.
                        </div>
                    </div>
                    ''' if cvr_histogram_exists else ''}
                </div>
                
                <div style="margin-top: 2rem; padding: 1rem; background: #e8f4f8; border-radius: 6px; border-left: 4px solid #17a2b8;">
                    <strong>Interpretation Notes:</strong>
                    <ul style="margin-top: 0.5rem; margin-bottom: 0; padding-left: 1.5rem;">
                        <li><strong>Delay values</strong> represent the optimal temporal shift between {"ROI probe" if roi_probe_enabled else "ETCO₂"} and BOLD signals, reflecting hemodynamic response timing</li>
                        <li><strong>CVR values</strong> quantify vascular reactivity {"in arbitrary units relative to ROI probe signal changes" if roi_probe_enabled else "as %BOLD signal change per mmHg CO₂ change"}</li>
                        <li><strong>Normal ranges</strong> vary by brain region, age, and individual physiology</li>
                        <li><strong>Outlier values</strong> may indicate data quality issues or pathological conditions</li>
                    </ul>
                </div>
            </div>
        </section>
        
        <!-- References Section -->
        <section id="references" class="section">
            <div class="section-header">
                <h2 class="section-title">References & Software Information</h2>
                <p class="section-subtitle">Citation information and source code repository</p>
            </div>
            <div class="section-content">
                <div class="summary-card" style="margin-bottom: 2rem;">
                    <h4><i class="fab fa-github"></i> Source Code Repository</h4>
                    <p style="margin-top: 1rem;">
                        <strong>CVRmap Toolbox:</strong> 
                        <a href="https://github.com/ln2t/cvrmap" target="_blank" style="color: #0366d6; text-decoration: none;">
                            https://github.com/ln2t/cvrmap
                        </a>
                    </p>
                    <p style="color: #666; font-size: 0.9em; margin-top: 0.5rem;">
                        Complete cerebrovascular reactivity mapping post-processing BIDS toolbox
                    </p>
                </div>
                
                <div class="summary-card">
                    <h4><i class="fas fa-file-alt"></i> Scientific Publication</h4>
                    <div style="margin-top: 1rem; padding: 1rem; background: #f8f9fa; border-radius: 6px; border-left: 4px solid #28a745;">
                        <p style="font-weight: bold; margin-bottom: 0.5rem;">
                            CVRmap—a complete cerebrovascular reactivity mapping post-processing BIDS toolbox
                        </p>
                        <p style="margin-bottom: 0.5rem;">
                            <strong>Authors:</strong> Rovai, A., Lolli, V., Trotta, N. et al.
                        </p>
                        <p style="margin-bottom: 0.5rem;">
                            <strong>Journal:</strong> Scientific Reports 14, 7252 (2024)
                        </p>
                        <p style="margin-bottom: 0.5rem;">
                            <strong>Published:</strong> 27 March 2024
                        </p>
                        <p style="margin-bottom: 1rem;">
                            <strong>DOI:</strong> 
                            <a href="https://doi.org/10.1038/s41598-024-57572-3" target="_blank" style="color: #0366d6; text-decoration: none;">
                                https://doi.org/10.1038/s41598-024-57572-3
                            </a>
                        </p>
                        <div style="background: #e8f4f8; padding: 0.75rem; border-radius: 4px; font-family: monospace; font-size: 0.85em;">
                            <strong>Citation:</strong><br>
                            Rovai, A., Lolli, V., Trotta, N. et al. CVRmap—a complete cerebrovascular reactivity mapping post-processing BIDS toolbox. <em>Sci Rep</em> <strong>14</strong>, 7252 (2024). https://doi.org/10.1038/s41598-024-57572-3
                        </div>
                    </div>
                    
                    <p style="margin-top: 1.5rem; color: #666; font-size: 0.9em;">
                        <strong>Please cite this paper when using CVRmap in your research.</strong> 
                        The publication describes the methodology, validation, and implementation details of the cerebrovascular reactivity mapping pipeline used to generate this report.
                    </p>
                </div>
            </div>
        </section>
    </div>
    
    <script>
        // Smooth scrolling for navigation links
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {{
            anchor.addEventListener('click', function (e) {{
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {{
                    target.scrollIntoView({{
                        behavior: 'smooth',
                        block: 'start'
                    }});
                }}
            }});
        }});
        
        // Highlight current section in navigation
        window.addEventListener('scroll', function() {{
            const sections = document.querySelectorAll('.section');
            const navLinks = document.querySelectorAll('.nav-links a');
            
            let current = '';
            sections.forEach(section => {{
                const sectionTop = section.offsetTop - 100;
                if (pageYOffset >= sectionTop) {{
                    current = section.getAttribute('id');
                }}
            }});
            
            navLinks.forEach(link => {{
                link.style.backgroundColor = '';
                if (link.getAttribute('href') === '#' + current) {{
                    link.style.backgroundColor = 'rgba(255,255,255,0.2)';
                }}
            }});
        }});
    </script>
</body>
</html>"""
        
        return html_template
    




