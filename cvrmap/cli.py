import argparse
import os
from . import __version__

# Set matplotlib backend to non-GUI to avoid tkinter issues in parallel processing
import matplotlib
matplotlib.use('Agg')

def main():
    def check_derivatives(derivatives, logger):
        import os
        if not derivatives:
            return
        for item in derivatives:
            if '=' not in item:
                logger.warning(f"Invalid derivatives format: '{item}'. Expected 'name=/path/to/derivatives'.")
                continue
            name, path = item.split('=', 1)
            if not name or not path:
                logger.warning(f"Invalid derivatives format: '{item}'. Name or path missing.")
                continue
            if not os.path.isdir(path):
                logger.warning(f"Derivatives path does not exist or is not a directory: '{path}' for pipeline '{name}'.")

    import os
    parser = argparse.ArgumentParser(
        description="cvrmap command line interface.\n\nExample for --derivatives: --derivatives fmriprep=/path/to/fmriprep/derivatives anotherpipe=/path/to/other/derivatives"
    )
    parser.add_argument('bids_dir', type=str, help='Path to the BIDS directory')
    parser.add_argument('output_dir', type=str, help='Path to the output directory')
    parser.add_argument('analysis_level', choices=['participant', 'group'], help='Level of the analysis that will be performed')
    parser.add_argument('--participant-label', '--participant_label', nargs='+', help='Space separated list of participant labels to process (e.g. "001 002")')
    parser.add_argument('--task', type=str, help='Task name')
    parser.add_argument('--space', type=str, help='Output space to use (default: fmriprep default space)')
    parser.add_argument('--config', type=str, help='Path to a configuration file (optional)')
    parser.add_argument('--debug-level', '--debug_level', type=int, choices=[0,1], help='Set debug level: 0=info/warnings, 1=debug')
    parser.add_argument(
        '--derivatives',
        nargs='+',
        metavar='PIPELINE=PATH',
        help='Other pipeline derivatives in the form name=/path/to/derivatives. Example: --derivatives fmriprep=/path/to/fmriprep/derivatives'
    )
    parser.add_argument('--version', action='version', version=f'cvrmap {__version__}')
    
    # Parallel processing option
    parser.add_argument('--n-jobs', type=int, default=-1,
                       help='Number of parallel jobs for voxel processing. -1 uses all available CPUs, 1 disables parallelization (default: -1)')
    
    # Baseline computation method option
    parser.add_argument('--baseline-method', type=str, choices=['peakutils', 'mean'], default=None,
                       help='Method for computing probe baseline: "peakutils" (default, detects baseline from signal troughs) or "mean" (uses signal mean). Recommended: "mean" for resting-state, "peakutils" for gas challenge tasks.')
    
    # ROI probe options
    parser.add_argument('--roi-probe', action='store_true', help='Enable ROI-based probe instead of physiological recordings')
    parser.add_argument('--roi-coordinates', nargs=3, type=float, metavar=('X', 'Y', 'Z'), 
                       help='ROI coordinates in millimeters (mm) in world space for spherical ROI extraction (e.g., --roi-coordinates 0 -52 26)')
    parser.add_argument('--roi-radius', type=float, default=6.0, 
                       help='Radius in millimeters (mm) for spherical ROI (default: 6.0)')
    parser.add_argument('--roi-mask', type=str,
                       help='Path to binary mask NIfTI file for ROI extraction, pattern with wildcards, or "SSS" to use the built-in Superior Sagittal Sinus mask')
    parser.add_argument('--roi-atlas', type=str,
                       help='Path to atlas NIfTI file for ROI extraction')
    parser.add_argument('--roi-region-id', type=int,
                       help='Region ID/label in atlas for ROI extraction')
    parser.add_argument('--roi-label', type=str,
                       help='Label for ROI-based outputs in BIDS naming (e.g., --roi-label myROI). Required when using --roi-mask with a custom path/pattern. Automatically set to "SSS" when using --roi-mask SSS.')
    
    args = parser.parse_args()

    # Check if bids_dir exists
    if not os.path.isdir(args.bids_dir):
        parser.error(f"bids_dir '{args.bids_dir}' does not exist or is not a directory.")

    from .logger import Logger
    logger = Logger(module_name="cli", debug_level=args.debug_level if args.debug_level is not None else 0)

    logger.info(f"bids_dir: {args.bids_dir}")
    logger.info(f"output_dir: {args.output_dir}")
    logger.info(f"analysis_level: {args.analysis_level}")
    logger.info(f"participant_label: {args.participant_label}")
    logger.info(f"task: {args.task}")
    logger.info(f"derivatives: {args.derivatives}")
    check_derivatives(args.derivatives, logger)

    # Determine fmriprep_dir
    import os
    fmriprep_dir = None
    if args.derivatives:
        for item in args.derivatives:
            if item.startswith('fmriprep='):
                _, fmriprep_dir_candidate = item.split('=', 1)
                if os.path.isdir(fmriprep_dir_candidate):
                    fmriprep_dir = fmriprep_dir_candidate
                else:
                    logger.warning(f"fmriprep path specified but does not exist: {fmriprep_dir_candidate}")
                break
    if not fmriprep_dir:
        fmriprep_dir = os.path.join(args.output_dir, 'fmriprep')
        if not os.path.isdir(fmriprep_dir):
            logger.warning(f"fmriprep derivatives not specified and default path does not exist: {fmriprep_dir}")
            parser.error(f"fmriprep derivatives not found. Please specify with --derivatives fmriprep=/path/to/fmriprep/derivatives or ensure {fmriprep_dir} exists.")
    logger.info(f"fmriprep_dir: {fmriprep_dir}")

    from .io import process_config
    config = process_config(user_config_path=args.config)
    
    # Add parallel processing configuration
    config['n_jobs'] = args.n_jobs
    
    # Add baseline method configuration
    if args.baseline_method:
        if 'physio' not in config:
            config['physio'] = {}
        config['physio']['baseline_method'] = args.baseline_method
        logger.info(f"Baseline method set to: {args.baseline_method}")
    
    # Add task-specific recommendations for baseline method
    if args.task and (args.task.lower() == 'restingstate' or args.task.lower() == 'resting-state' or args.task.lower() == 'rest'):
        current_baseline_method = config.get('physio', {}).get('baseline_method', 'peakutils')
        if current_baseline_method == 'peakutils':
            logger.warning("WARNING: Task appears to be resting-state data. Consider using --baseline-method mean instead of peakutils for better baseline estimation in resting-state data without gas challenge.")
    
    # Handle ROI probe command line arguments
    if args.roi_probe:
        logger.info("ROI probe mode enabled via command line")
        
        # Update config with ROI probe settings
        if 'roi_probe' not in config:
            config['roi_probe'] = {}
        
        config['roi_probe']['enabled'] = True
        
        # Determine ROI method and parameters
        if args.roi_coordinates:
            config['roi_probe']['method'] = 'coordinates'
            config['roi_probe']['coordinates_mm'] = list(args.roi_coordinates)
            config['roi_probe']['radius_mm'] = args.roi_radius
            logger.info(f"ROI coordinates: {args.roi_coordinates}, radius: {args.roi_radius}mm")
            
        elif args.roi_mask:
            config['roi_probe']['method'] = 'mask'
            # Check for built-in mask keywords
            if args.roi_mask.upper() == 'SSS':
                # Use bundled Superior Sagittal Sinus mask
                import pkg_resources
                sss_mask_path = pkg_resources.resource_filename('cvrmap', 'data/SuperiorSagittalSinus_mask.nii.gz')
                config['roi_probe']['mask_path'] = sss_mask_path
                config['roi_probe']['label'] = 'SSS'
                logger.info(f"Using built-in Superior Sagittal Sinus (SSS) mask: {sss_mask_path}")
            else:
                # Custom mask path/pattern requires --roi-label
                if not args.roi_label:
                    parser.error("When using --roi-mask with a custom path or pattern, --roi-label is required to identify outputs in BIDS naming.")
                config['roi_probe']['mask_path'] = args.roi_mask
                config['roi_probe']['label'] = args.roi_label
                logger.info(f"ROI mask: {args.roi_mask}, label: {args.roi_label}")
            
        elif args.roi_atlas and args.roi_region_id is not None:
            config['roi_probe']['method'] = 'atlas'
            config['roi_probe']['atlas_path'] = args.roi_atlas
            config['roi_probe']['region_id'] = args.roi_region_id
            logger.info(f"ROI atlas: {args.roi_atlas}, region: {args.roi_region_id}")
            
        else:
            parser.error("When using --roi-probe, you must specify either:\n"
                        "  --roi-coordinates X Y Z (and optionally --roi-radius)\n"
                        "  --roi-mask PATH\n"
                        "  --roi-atlas PATH --roi-region-id ID")
    
    from .pipeline import Pipeline
    pipeline = Pipeline(args, logger, fmriprep_dir, config=config)
    pipeline.run()

if __name__ == "__main__":
    main()
