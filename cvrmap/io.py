# Placeholder for later usage

import os
import yaml
import numpy as np
import json

from . import __version__

def convert_numpy_types(obj):
	"""
	Recursively convert numpy types to native Python types for JSON serialization.
	"""
	if isinstance(obj, dict):
		return {key: convert_numpy_types(value) for key, value in obj.items()}
	elif isinstance(obj, list):
		return [convert_numpy_types(item) for item in obj]
	elif isinstance(obj, np.integer):
		return int(obj)
	elif isinstance(obj, np.floating):
		return float(obj)
	elif isinstance(obj, np.ndarray):
		return obj.tolist()
	else:
		return obj

def safe_json_dump(obj, file_handle, **kwargs):
	"""
	Safely dump an object to JSON, converting numpy types as needed.
	"""
	converted_obj = convert_numpy_types(obj)
	return json.dump(converted_obj, file_handle, **kwargs)

def process_config(user_config_path=None, default_config_path=None):
	"""
	Load and merge user and default YAML config files. User config overrides defaults.
	"""
	if default_config_path is None:
		default_config_path = os.path.join(os.path.dirname(__file__), 'default_config.yaml')
	# Load default config
	with open(default_config_path, 'r') as f:
		default_config = yaml.safe_load(f) or {}
	config = default_config.copy()
	# If user config provided, load and update
	if user_config_path:
		with open(user_config_path, 'r') as f:
			user_config = yaml.safe_load(f) or {}
		config.update(user_config)
	return config


def _process_regressor_voxel(voxel_coord, delay_maps, time_delays_seconds, shifted_signals):
	"""Standalone worker function for single voxel regressor processing"""
	import numpy as np
	import os
	
	# Set environment variables to prevent GUI issues in workers
	os.environ['MPLBACKEND'] = 'Agg'
	os.environ['QT_QPA_PLATFORM'] = 'offscreen'
	
	i, j, k = voxel_coord
	
	# Get optimal delay for this voxel
	optimal_delay = delay_maps[i, j, k]
	
	# Skip if delay is NaN (masked voxel)
	if np.isnan(optimal_delay):
		return i, j, k, None
	
	# Find the closest delay in our time_delays_seconds array
	delay_idx = np.argmin(np.abs(time_delays_seconds - optimal_delay))
	
	# Return the shifted probe signal for this voxel
	return i, j, k, shifted_signals[delay_idx, :]


def _process_regressor_voxel_chunk(voxel_chunk, delay_maps, time_delays_seconds, shifted_signals):
	"""Standalone worker function for chunked regressor processing"""
	import numpy as np
	import os
	
	# Set environment variables to prevent GUI issues in workers
	os.environ['MPLBACKEND'] = 'Agg'
	os.environ['QT_QPA_PLATFORM'] = 'offscreen'
	
	results = []
	for voxel_coord in voxel_chunk:
		i, j, k = voxel_coord
		
		# Get optimal delay for this voxel
		optimal_delay = delay_maps[i, j, k]
		
		# Skip if delay is NaN (masked voxel)
		if np.isnan(optimal_delay):
			results.append((i, j, k, None))
			continue
		
		# Find the closest delay in our time_delays_seconds array
		delay_idx = np.argmin(np.abs(time_delays_seconds - optimal_delay))
		
		# Return the shifted probe signal for this voxel
		results.append((i, j, k, shifted_signals[delay_idx, :]))
	
	return results


class OutputGenerator:
	"""
	Handle how data will be saved to the output directory.
	
	This class manages the generation and organization of output files
	in a BIDS-compatible derivatives structure.
	"""
	
	def __init__(self, output_dir, logger=None, config=None):
		"""
		Initialize the OutputGenerator.
		
		Parameters:
		-----------
		output_dir : str
			Base output directory for saving results.
		logger : logging.Logger, optional
			Logger instance for consistent logging.
		config : dict, optional
			Configuration dictionary containing processing parameters.
		"""
		self.output_dir = output_dir
		self.logger = logger
		self.config = config or {}
		self._ensure_dataset_description()
	
	def _create_bids_paths(self, participant, data_type, filename_base):
		"""
		Create BIDS-compliant directory structure and file paths.
		
		Parameters:
		-----------
		participant : str
			Participant ID.
		data_type : str
			Data type (e.g., 'func', 'physio', 'figures').
		filename_base : str
			Base filename without extension.
			
		Returns:
		--------
		tuple
			(output_dir, nii_path, json_path, tsv_path) where applicable paths are None if not needed.
		"""
		participant_dir = f"sub-{participant}"
		output_data_dir = os.path.join(self.output_dir, participant_dir, data_type)
		os.makedirs(output_data_dir, exist_ok=True)
		
		nii_path = os.path.join(output_data_dir, f"{filename_base}.nii.gz")
		json_path = os.path.join(output_data_dir, f"{filename_base}.json")
		tsv_path = os.path.join(output_data_dir, f"{filename_base}.tsv.gz")
		png_path = os.path.join(output_data_dir, f"{filename_base}.png")
		
		return output_data_dir, nii_path, json_path, tsv_path, png_path

	def _get_roi_label_entity(self):
		"""
		Get the ROI label entity string for BIDS naming.

		Returns '_label-{label}' if ROI probe is enabled and has a label configured,
		otherwise returns an empty string.

		Returns:
		--------
		str
			The label entity string (e.g., '_label-SSS') or empty string.
		"""
		roi_config = self.config.get('roi_probe', {})
		if roi_config.get('enabled') and roi_config.get('label'):
			return f"_label-{roi_config['label']}"
		return ''

	def _create_standard_sidecar_base(self, description, units, space, task, data_type):
		"""
		Create a standard base JSON sidecar with common fields.
		
		Parameters:
		-----------
		description : str
			Description of the data.
		units : str
			Units of the data.
		space : str
			Space name.
		task : str
			Task name.
		data_type : str
			Type of data.
			
		Returns:
		--------
		dict
			Base sidecar dictionary.
		"""
		return {
			"Description": description,
			"Units": units,
			"Space": space,
			"DataType": data_type,
			"TaskName": task
		}
	
	def _setup_lightbox_figure(self, title, figsize=(16, 12)):
		"""
		Setup a standard lightbox figure with grid layout.
		
		Parameters:
		-----------
		title : str
			Figure title.
		figsize : tuple
			Figure size.
			
		Returns:
		--------
		tuple
			(fig, gs_inner, ax_cbar, n_rows, n_cols)
		"""
		import matplotlib.pyplot as plt
		from matplotlib.gridspec import GridSpec
		
		n_rows, n_cols = 4, 5
		
		# Create figure with custom layout: main plot area + colorbar
		fig = plt.figure(figsize=figsize, facecolor='black')
		gs = GridSpec(1, 2, width_ratios=[0.95, 0.05], wspace=0.02)
		
		# Main plot area for lightbox
		ax_main = fig.add_subplot(gs[0])
		ax_main.set_facecolor('black')
		ax_main.axis('off')
		
		# Colorbar area
		ax_cbar = fig.add_subplot(gs[1])
		
		# Create subplot grid within the main area
		gs_inner = GridSpec(n_rows, n_cols, figure=fig, 
		                   left=gs[0].get_position(fig).x0,
		                   right=gs[0].get_position(fig).x1,
		                   bottom=gs[0].get_position(fig).y0,
		                   top=gs[0].get_position(fig).y1 - 0.05,  # Leave space for title
		                   hspace=0.05, wspace=0.05)
		
		# Add title
		fig.suptitle(title, fontsize=16, color='white', y=0.95)
		
		return fig, gs_inner, ax_cbar, n_rows, n_cols
	
	def _add_baseline_to_sidecar(self, sidecar, probe_container, is_roi_probe=False):
		"""
		Add baseline information to sidecar if available.
		
		Parameters:
		-----------
		sidecar : dict
			Sidecar dictionary to modify.
		probe_container : ProbeContainer
			Container with potential baseline information.
		is_roi_probe : bool
			Whether this is an ROI probe.
		"""
		if hasattr(probe_container, 'baseline') and probe_container.baseline is not None:
			baseline_units = probe_container.units if probe_container.units and not is_roi_probe else ("BOLD units" if is_roi_probe else "mmHg")
			
			# Determine baseline method from config
			baseline_method = self.config.get('physio', {}).get('baseline_method', 'peakutils')
			if baseline_method == 'mean':
				description = "Baseline probe value computed as the mean of the signal"
				method = "mean"
			else:
				description = "Baseline probe value computed using peakutils baseline estimation"
				method = "peakutils.baseline"
			
			sidecar["BaselineValue"] = {
				"Value": float(probe_container.baseline),
				"Units": baseline_units,
				"Description": description,
				"Method": method
			}
	
	def _ensure_dataset_description(self):
		"""
		Create dataset_description.json in output_dir if it doesn't exist.
		"""
		import json
		
		dataset_desc_path = os.path.join(self.output_dir, 'dataset_description.json')
		
		if not os.path.exists(dataset_desc_path):
			os.makedirs(self.output_dir, exist_ok=True)
			
			dataset_description = {
				"Name": "CVRMap Analysis Results",
				"BIDSVersion": "1.8.0",
				"DatasetType": "derivative",
				"GeneratedBy": [
					{
						"Name": "cvrmap",
						"Version": __version__,
						"Description": "Cerebrovascular reactivity mapping pipeline"
					}
				],
				"SourceDatasets": [],
				"HowToAcknowledge": "Please cite the cvrmap software when using these results."
			}
			
			with open(dataset_desc_path, 'w') as f:
				safe_json_dump(dataset_description, f, indent=2)
			
			if self.logger:
				self.logger.info(f"Created dataset_description.json at {dataset_desc_path}")
	
	def save_etco2_data(self, etco2_container, participant, task):
		"""
		Save probe data (ETCO2 or ROI probe) to a .tsv.gz file with BIDS naming and JSON sidecar.
		
		Parameters:
		-----------
		etco2_container : ProbeContainer
			Container with probe data to save (ETCO2 or ROI-based).
		participant : str
			Participant ID.
		task : str
			Task name.
		"""
		import json
		import pandas as pd
		import numpy as np
		
		# Determine probe type and adjust naming accordingly
		probe_type = getattr(etco2_container, 'probe_type', 'etco2')
		is_roi_probe = 'roi_probe' in probe_type  # Handle both 'roi_probe' and 'roi_probe_normalized'
		
		# Create BIDS paths
		data_dir = "physio" if not is_roi_probe else "func"  # ROI probes go in func directory
		label_entity = self._get_roi_label_entity()
		if is_roi_probe:
			filename_base = f"sub-{participant}_task-{task}{label_entity}_desc-roiprobe_bold"
		else:
			filename_base = f"sub-{participant}_task-{task}_desc-etco2_physio"
		
		_, _, json_path, tsv_path, _ = self._create_bids_paths(participant, data_dir, filename_base)
		
		# Create time vector and DataFrame
		time_vector = np.arange(len(etco2_container.data)) / etco2_container.sampling_frequency
		probe_column_name = 'roiprobe' if is_roi_probe else 'etco2'
		df = pd.DataFrame({
			'time': time_vector,
			probe_column_name: etco2_container.data
		})
		df.to_csv(tsv_path, sep='\t', index=False, compression='gzip')
		
		# Create JSON sidecar
		if is_roi_probe:
			sidecar = {
				"Description": "ROI-based probe signal extracted from brain region for CVR analysis",
				"SamplingFrequency": float(etco2_container.sampling_frequency),
				"StartTime": 0.0,
				"Columns": ["time", "roiprobe"],
				"time": {
					"Description": "Time in seconds from start of recording",
					"Units": "s"
				},
				"roiprobe": {
					"Description": "ROI-averaged BOLD signal used as probe for CVR analysis",
					"Units": "BOLD signal intensity"
				},
				"ProcessingDescription": "Signal extracted by averaging BOLD timeseries across specified ROI voxels"
			}
		else:
			sidecar = {
				"Description": "End-tidal CO2 (ETCO2) signal extracted from physiological recordings",
				"SamplingFrequency": float(etco2_container.sampling_frequency),
				"StartTime": 0.0,
				"Columns": ["time", "etco2"],
				"time": {
					"Description": "Time in seconds from start of recording",
					"Units": "s"
				},
				"etco2": {
					"Description": "End-tidal CO2 concentration",
					"Units": etco2_container.units if etco2_container.units else "mmHg"
				},
				"ProcessingDescription": "ETCO2 extracted using peak detection and cubic interpolation"
			}
		
		# Add baseline value if available
		self._add_baseline_to_sidecar(sidecar, etco2_container, is_roi_probe)
		
		with open(json_path, 'w') as f:
			safe_json_dump(sidecar, f, indent=2)
		
		if self.logger:
			probe_desc = "ROI probe" if is_roi_probe else "ETCO2"
			self.logger.info(f"Saved {probe_desc} data to {tsv_path}")
		
		return tsv_path, json_path
	
	def create_physio_figure(self, physio_container, etco2_container, participant, task):
		"""
		Create a figure showing physio data and ETCO2 data together.
		
		Parameters:
		-----------
		physio_container : PhysioDataContainer
			Container with raw physiological data.
		etco2_container : ProbeContainer
			Container with ETCO2 data.
		participant : str
			Participant ID.
		task : str
			Task name.
		"""
		import matplotlib.pyplot as plt
		import numpy as np
		
		# Create BIDS directory structure
		participant_dir = f"sub-{participant}"
		figures_dir = "figures"
		output_figures_dir = os.path.join(self.output_dir, participant_dir, figures_dir)
		os.makedirs(output_figures_dir, exist_ok=True)
		
		# Create BIDS filename
		filename = f"sub-{participant}_task-{task}_desc-physio.png"
		fig_path = os.path.join(output_figures_dir, filename)
		
		# Get CO2 column from physio data
		columns = physio_container.physio_metadata.get('Columns', [])
		co2_col = next((i for i, c in enumerate(columns) if c.lower() == 'co2'), None)
		
		if co2_col is not None:
			co2_signal = physio_container.data[:, co2_col]
			physio_time = np.arange(len(co2_signal)) / physio_container.sampling_frequency
		
		etco2_time = np.arange(len(etco2_container.data)) / etco2_container.sampling_frequency
		
		# Create figure
		fig, ax = plt.subplots(figsize=(12, 6))
		
		if co2_col is not None:
			ax.plot(physio_time, co2_signal, label='Raw CO2', alpha=0.7, color='lightblue')
		ax.plot(etco2_time, etco2_container.data, label='ETCO2', color='darkblue', linewidth=2)
		
		# Add baseline line if available
		if hasattr(etco2_container, 'baseline') and etco2_container.baseline is not None:
			ax.axhline(y=etco2_container.baseline, color='red', linestyle='--', linewidth=2, 
			          label=f'Baseline ({etco2_container.baseline:.1f} {etco2_container.units if etco2_container.units else "mmHg"})')
		
		ax.set_xlabel('Time (s)')
		ax.set_ylabel(f'CO2 ({etco2_container.units if etco2_container.units else "mmHg"})')
		ax.set_title(f'Physiological Data - Subject {participant}, Task {task}')
		ax.legend()
		ax.grid(True, alpha=0.3)
		
		plt.tight_layout()
		plt.savefig(fig_path, dpi=300, bbox_inches='tight')
		plt.close()
		
		if self.logger:
			self.logger.info(f"Created physio figure at {fig_path}")
		
		return fig_path
	
	def create_roi_probe_figure(self, roi_probe_container, participant, task, config):
		"""
		Create a figure showing ROI-based probe signal with ROI information.
		
		Parameters:
		-----------
		roi_probe_container : ProbeContainer
			Container with ROI-extracted probe signal.
		participant : str
			Participant ID.
		task : str
			Task name.
		config : dict
			Configuration dictionary with ROI settings.
		"""
		import matplotlib.pyplot as plt
		import numpy as np
		
		# Create BIDS directory structure
		participant_dir = f"sub-{participant}"
		figures_dir = "figures"
		output_figures_dir = os.path.join(self.output_dir, participant_dir, figures_dir)
		os.makedirs(output_figures_dir, exist_ok=True)

		# Create BIDS filename with label entity
		label_entity = self._get_roi_label_entity()
		filename = f"sub-{participant}_task-{task}{label_entity}_desc-roiprobe.png"
		fig_path = os.path.join(output_figures_dir, filename)

		# Get ROI configuration details
		roi_config = config.get('roi_probe', {})
		roi_method = roi_config.get('method', 'Unknown')
		
		# Create time vector
		probe_time = np.arange(len(roi_probe_container.data)) / roi_probe_container.sampling_frequency
		
		# Create figure
		fig, ax = plt.subplots(figsize=(12, 6))
		
		# Plot ROI probe signal
		ax.plot(probe_time, roi_probe_container.data, label='ROI Probe Signal', color='green', linewidth=2)
		
		# Add baseline line if available
		if hasattr(roi_probe_container, 'baseline') and roi_probe_container.baseline is not None:
			ax.axhline(y=roi_probe_container.baseline, color='red', linestyle='--', linewidth=2, 
			          label=f'Baseline ({roi_probe_container.baseline:.3f} BOLD units)')
		
		# Set labels and title
		ax.set_xlabel('Time (s)')
		ax.set_ylabel('Signal Intensity (BOLD units)')
		
		# Create detailed title based on ROI method
		title_parts = [f'ROI Probe Signal - Subject {participant}, Task {task}']
		if roi_method == 'coordinates':
			coords = roi_config.get('coordinates_mm', 'Unknown')
			radius = roi_config.get('radius_mm', 'Unknown')
			title_parts.append(f'Coordinates: {coords}, Radius: {radius}mm')
		elif roi_method == 'mask':
			mask_path = roi_config.get('mask_path', 'Unknown')
			title_parts.append(f'Mask: {mask_path}')
		elif roi_method == 'atlas':
			atlas_path = roi_config.get('atlas_path', 'Unknown')
			region_id = roi_config.get('region_id', 'Unknown')
			title_parts.append(f'Atlas: {atlas_path}, Region: {region_id}')
		
		ax.set_title('\n'.join(title_parts))
		ax.legend()
		ax.grid(True, alpha=0.3)
		
		# Add information text box
		info_text = f'ROI Method: {roi_method}\nProbe Type: {getattr(roi_probe_container, "probe_type", "roi_probe")}'
		ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
		        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8), 
		        verticalalignment='top', fontsize=10)
		
		plt.tight_layout()
		plt.savefig(fig_path, dpi=300, bbox_inches='tight')
		plt.close()
		
		if self.logger:
			self.logger.info(f"Created ROI probe figure at {fig_path}")
		
		return fig_path
	
	def create_roi_visualization_figure(self, roi_probe_container, bold_container, participant, task, config):
		"""
		Create a figure showing the ROI overlay on the mean BOLD image.
		
		Parameters:
		-----------
		roi_probe_container : ProbeContainer
			Container with ROI-extracted probe signal.
		bold_container : BoldContainer
			Container with BOLD data for background image.
		participant : str
			Participant ID.
		task : str
			Task name.
		config : dict
			Configuration dictionary with ROI settings.
		"""
		import matplotlib.pyplot as plt
		import numpy as np
		import nibabel as nib
		
		# Create BIDS directory structure
		participant_dir = f"sub-{participant}"
		figures_dir = "figures"
		output_figures_dir = os.path.join(self.output_dir, participant_dir, figures_dir)
		os.makedirs(output_figures_dir, exist_ok=True)

		# Create BIDS filename with label entity
		label_entity = self._get_roi_label_entity()
		filename = f"sub-{participant}_task-{task}{label_entity}_desc-roivisualization.png"
		fig_path = os.path.join(output_figures_dir, filename)

		# Get ROI configuration details
		roi_config = config.get('roi_probe', {})
		roi_method = roi_config.get('method', 'Unknown')

		try:
			# Create mean BOLD image
			bold_img = nib.Nifti1Image(bold_container.data, bold_container.affine)
			
			# Create ROI mask based on method
			roi_mask = None
			roi_info = ""
			
			if roi_method == 'coordinates':
				coords = roi_config.get('coordinates_mm', [0, 0, 0])
				radius = roi_config.get('radius_mm', 5)
				roi_info = f"Spherical ROI: {coords} mm, radius: {radius}mm"
				
				# Create spherical mask
				x, y, z = np.meshgrid(
					np.arange(bold_container.data.shape[0]),
					np.arange(bold_container.data.shape[1]),
					np.arange(bold_container.data.shape[2]),
					indexing='ij'
				)
				
				# Convert voxel coordinates to mm
				voxel_coords = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
				mm_coords = nib.affines.apply_affine(bold_container.affine, voxel_coords)
				
				# Calculate distances to ROI center
				distances = np.sqrt(np.sum((mm_coords - coords) ** 2, axis=1))
				roi_mask = (distances <= radius).reshape(bold_container.data.shape[:3])
				
			elif roi_method == 'mask':
				mask_path = roi_config.get('mask_path', '')
				roi_info = f"Custom mask: {os.path.basename(mask_path)}"
				
				if mask_path and os.path.exists(mask_path):
					mask_img = nib.load(mask_path)
					mask_data = mask_img.get_fdata()
					
					# Handle resampling if needed (same logic as in roi_probe.py)
					if mask_data.shape[:3] != bold_container.data.shape[:3]:
						from nilearn.image import resample_to_img
						bold_ref_img = nib.Nifti1Image(bold_container.data[:,:,:,0], bold_container.affine)
						resampled_mask_img = resample_to_img(mask_img, bold_ref_img, interpolation='nearest')
						mask_data = resampled_mask_img.get_fdata()
					
					roi_mask = mask_data > 0
				else:
					roi_mask = np.zeros(bold_container.data.shape[:3], dtype=bool)
					
			elif roi_method == 'atlas':
				atlas_path = roi_config.get('atlas_path', '')
				region_id = roi_config.get('region_id', 0)
				roi_info = f"Atlas region: {os.path.basename(atlas_path)}, ID: {region_id}"
				
				if atlas_path and os.path.exists(atlas_path):
					atlas_img = nib.load(atlas_path)
					atlas_data = atlas_img.get_fdata()
					
					# Handle resampling if needed (same logic as in roi_probe.py)
					if atlas_data.shape != bold_container.data.shape[:3]:
						from nilearn.image import resample_to_img
						bold_ref_img = nib.Nifti1Image(bold_container.data[:,:,:,0], bold_container.affine)
						atlas_resampled = resample_to_img(atlas_img, bold_ref_img, interpolation='nearest')
						atlas_data = atlas_resampled.get_fdata()
					
					roi_mask = atlas_data == region_id
				else:
					roi_mask = np.zeros(bold_container.data.shape[:3], dtype=bool)
			
			if roi_mask is not None and np.any(roi_mask):
				# Create ROI mask image
				roi_mask_img = nib.Nifti1Image(roi_mask.astype(float), bold_container.affine)
				
				# Create the visualization using matplotlib directly (not nilearn plotting)
				fig, axes = plt.subplots(2, 3, figsize=(15, 10))
				fig.suptitle(f'ROI Visualization - Subject {participant}, Task {task}\n{roi_info}', fontsize=14)
				
				# Calculate central slices for each view
				mean_bold_data = np.mean(bold_container.data, axis=3)
				
				# Find slices with most ROI voxels
				roi_x_counts = np.sum(roi_mask, axis=(1, 2))
				roi_y_counts = np.sum(roi_mask, axis=(0, 2))
				roi_z_counts = np.sum(roi_mask, axis=(0, 1))
				
				best_x = np.argmax(roi_x_counts) if np.max(roi_x_counts) > 0 else roi_mask.shape[0]//2
				best_y = np.argmax(roi_y_counts) if np.max(roi_y_counts) > 0 else roi_mask.shape[1]//2
				best_z = np.argmax(roi_z_counts) if np.max(roi_z_counts) > 0 else roi_mask.shape[2]//2
				
				# Sagittal view (x slice)
				axes[0, 0].imshow(mean_bold_data[best_x, :, :].T, cmap='gray', origin='lower', aspect='auto')
				axes[0, 0].imshow(roi_mask[best_x, :, :].T, cmap='Reds', origin='lower', alpha=0.7, aspect='auto')
				axes[0, 0].set_title(f'Sagittal (x={best_x})')
				axes[0, 0].set_xlabel('Y (posterior-anterior)')
				axes[0, 0].set_ylabel('Z (inferior-superior)')
				
				# Coronal view (y slice)
				axes[0, 1].imshow(mean_bold_data[:, best_y, :].T, cmap='gray', origin='lower', aspect='auto')
				axes[0, 1].imshow(roi_mask[:, best_y, :].T, cmap='Reds', origin='lower', alpha=0.7, aspect='auto')
				axes[0, 1].set_title(f'Coronal (y={best_y})')
				axes[0, 1].set_xlabel('X (left-right)')
				axes[0, 1].set_ylabel('Z (inferior-superior)')
				
				# Axial view (z slice)
				axes[0, 2].imshow(mean_bold_data[:, :, best_z].T, cmap='gray', origin='lower', aspect='auto')
				axes[0, 2].imshow(roi_mask[:, :, best_z].T, cmap='Reds', origin='lower', alpha=0.7, aspect='auto')
				axes[0, 2].set_title(f'Axial (z={best_z})')
				axes[0, 2].set_xlabel('X (left-right)')
				axes[0, 2].set_ylabel('Y (posterior-anterior)')
				
				# Add ROI statistics
				n_voxels = np.sum(roi_mask)
				roi_volume_mm3 = n_voxels * np.abs(np.linalg.det(bold_container.affine[:3, :3]))
				
				axes[1, 0].text(0.1, 0.8, f'ROI Statistics:', fontsize=12, fontweight='bold')
				axes[1, 0].text(0.1, 0.6, f'Voxels: {n_voxels:,}', fontsize=11)
				axes[1, 0].text(0.1, 0.4, f'Volume: {roi_volume_mm3:.1f} mm³', fontsize=11)
				axes[1, 0].text(0.1, 0.2, f'Method: {roi_method}', fontsize=11)
				axes[1, 0].set_xlim(0, 1)
				axes[1, 0].set_ylim(0, 1)
				axes[1, 0].axis('off')
				
				# Plot ROI signal summary
				if hasattr(roi_probe_container, 'data') and roi_probe_container.data is not None:
					time_vec = np.arange(len(roi_probe_container.data)) / roi_probe_container.sampling_frequency
					axes[1, 1].plot(time_vec, roi_probe_container.data, 'g-', linewidth=1.5)
					axes[1, 1].set_xlabel('Time (s)')
					axes[1, 1].set_ylabel('ROI Signal')
					axes[1, 1].set_title('Extracted ROI Signal')
					axes[1, 1].grid(True, alpha=0.3)
				else:
					axes[1, 1].text(0.5, 0.5, 'ROI signal not available', 
								  ha='center', va='center', transform=axes[1, 1].transAxes)
					axes[1, 1].axis('off')
				
				# Hide the last subplot
				axes[1, 2].axis('off')
				
			else:
				# Fallback if no ROI found
				fig, ax = plt.subplots(figsize=(10, 6))
				ax.text(0.5, 0.5, f'ROI Visualization Not Available\nMethod: {roi_method}\nNo valid ROI found', 
						ha='center', va='center', fontsize=14)
				ax.axis('off')
			
			plt.tight_layout()
			plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
			plt.close()
			
			if self.logger:
				self.logger.info(f"Created ROI visualization figure at {fig_path}")
				
		except Exception as e:
			if self.logger:
				self.logger.warning(f"Failed to create ROI visualization: {e}")
			
			# Create a simple fallback figure
			fig, ax = plt.subplots(figsize=(10, 6))
			ax.text(0.5, 0.5, f'ROI Visualization Error\n{str(e)}', 
					ha='center', va='center', fontsize=12)
			ax.axis('off')
			plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
			plt.close()
		
		return fig_path
	
	def save_global_signal(self, global_signal_container, participant, task, space):
		"""
		Save global BOLD signal to a .tsv.gz file with BIDS naming and JSON sidecar.
		
		Parameters:
		-----------
		global_signal_container : ProbeContainer
			Container with global BOLD signal data.
		participant : str
			Participant ID.
		task : str
			Task name.
		space : str
			Space name (e.g., 'MNI152NLin2009cAsym').
		"""
		import json
		import pandas as pd
		import numpy as np

		# Create BIDS paths with label entity
		label_entity = self._get_roi_label_entity()
		filename_base = f"sub-{participant}_task-{task}{label_entity}_space-{space}_desc-global_bold"
		_, _, json_path, tsv_path, _ = self._create_bids_paths(participant, "func", filename_base)
		
		# Create time vector and DataFrame
		time_vector = np.arange(len(global_signal_container.data)) / global_signal_container.sampling_frequency
		df = pd.DataFrame({
			'time': time_vector,
			'global_signal': global_signal_container.data
		})
		df.to_csv(tsv_path, sep='\t', index=False, compression='gzip')
		
		# Create JSON sidecar
		sidecar = {
			"Description": "Global BOLD signal computed as the mean across all brain voxels",
			"SamplingFrequency": float(global_signal_container.sampling_frequency),
			"RepetitionTime": float(1.0 / global_signal_container.sampling_frequency),
			"StartTime": 0.0,
			"Columns": ["time", "global_signal"],
			"time": {
				"Description": "Time in seconds from start of acquisition",
				"Units": "s"
			},
			"global_signal": {
				"Description": "Mean BOLD signal across all voxels within brain mask",
				"Units": "arbitrary"
			},
			"Space": space,
			"ProcessingDescription": "Global signal computed as mean of all voxels within brain mask"
		}
		
		with open(json_path, 'w') as f:
			safe_json_dump(sidecar, f, indent=2)
		
		if self.logger:
			self.logger.info(f"Saved global signal to {tsv_path}")
		
		return tsv_path, json_path
	
	def create_global_signal_figure(self, normalized_global_signal, shifted_probe_container, global_delay, participant, task, space, unshifted_probe_container=None):
		"""
		Create a figure showing normalized global signal with time-shifted probe signal at global delay.
		
		Parameters:
		-----------
		normalized_global_signal : ProbeContainer
			Normalized global BOLD signal container.
		shifted_probe_container : ProbeContainer
			Time-shifted and normalized probe container at global delay.
		global_delay : float
			Global delay in seconds.
		participant : str
			Participant ID.
		task : str
			Task name.
		space : str
			Space name.
		unshifted_probe_container : ProbeContainer, optional
			Normalized but unshifted probe container (delay=0).
		"""
		import matplotlib.pyplot as plt
		import numpy as np
		
		# Determine probe type for appropriate labeling
		probe_type = getattr(shifted_probe_container, 'probe_type', 'etco2')
		is_roi_probe = 'roi_probe' in probe_type  # Handle both 'roi_probe' and 'roi_probe_normalized'
		probe_label = 'ROI Probe' if is_roi_probe else 'ETCO2'
		probe_units = 'BOLD units' if is_roi_probe else 'mmHg'
		
		# Create BIDS directory structure
		participant_dir = f"sub-{participant}"
		figures_dir = "figures"
		output_figures_dir = os.path.join(self.output_dir, participant_dir, figures_dir)
		os.makedirs(output_figures_dir, exist_ok=True)

		# Create BIDS filename with label entity
		label_entity = self._get_roi_label_entity()
		filename = f"sub-{participant}_task-{task}{label_entity}_space-{space}_desc-globalcorr.png"
		fig_path = os.path.join(output_figures_dir, filename)
		
		# Create time vectors
		global_time = np.arange(len(normalized_global_signal.data)) / normalized_global_signal.sampling_frequency
		probe_time = np.arange(len(shifted_probe_container.data)) / shifted_probe_container.sampling_frequency
		
		# Create figure
		fig, ax = plt.subplots(figsize=(12, 6))
		
		ax.plot(global_time, normalized_global_signal.data, label='Normalized Global BOLD', color='red', linewidth=2)
		ax.plot(probe_time, shifted_probe_container.data, label=f'Normalized {probe_label} (shift: {global_delay:.1f}s)', color='blue', linewidth=2)
		
		# Add unshifted probe signal if provided
		if unshifted_probe_container is not None:
			unshifted_time = np.arange(len(unshifted_probe_container.data)) / unshifted_probe_container.sampling_frequency
			ax.plot(unshifted_time, unshifted_probe_container.data, 
			       label=f'Normalized {probe_label} (unshifted)', color='blue', linestyle='--', linewidth=2, alpha=0.2)
		
		ax.set_xlabel('Time (s)')
		ax.set_ylabel('Normalized Signal (z-score)')
		ax.set_title(f'Global Signal Correlation - Subject {participant}, Task {task}\nOptimal Delay: {global_delay:.1f}s')
		ax.legend()
		ax.grid(True, alpha=0.3)
		
		# Add correlation text
		if len(normalized_global_signal.data) == len(shifted_probe_container.data):
			correlation = np.corrcoef(normalized_global_signal.data, shifted_probe_container.data)[0, 1]
			ax.text(0.02, 0.98, f'Correlation: {correlation:.3f}', transform=ax.transAxes, 
					bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), verticalalignment='top')
		
		plt.tight_layout()
		plt.savefig(fig_path, dpi=300, bbox_inches='tight')
		plt.close()
		
		if self.logger:
			self.logger.info(f"Created global signal correlation figure at {fig_path}")
		
		return fig_path

	def save_delay_maps(self, delay_results, normalized_bold_container, participant, task, space, global_delay=None, probe_container=None):
		"""
		Save delay and correlation maps to NIfTI files with BIDS naming and JSON sidecars.
		
		Parameters:
		-----------
		delay_results : dict
			Dictionary containing 'delay_maps', 'correlation_maps', and 'delay_range' from DelayProcessor
		normalized_bold_container : BoldContainer
			Normalized BOLD container with spatial information (affine, header)
		participant : str
			Participant ID.
		task : str
			Task name.
		space : str
			Space name (e.g., 'MNI152NLin2009cAsym').
		global_delay : float, optional
			Global delay value in seconds used as reference for delay computation
		probe_container : ProbeContainer, optional
			Container with probe information for metadata
			
		Returns:
		--------
		tuple
			Paths to saved delay map and correlation map files (delay_path, correlation_path)
		"""
		import json
		import numpy as np
		import nibabel as nib
		
		# Determine probe type for appropriate descriptions
		probe_type = getattr(probe_container, 'probe_type', 'etco2') if probe_container else 'etco2'
		is_roi_probe = 'roi_probe' in probe_type  # Handle both 'roi_probe' and 'roi_probe_normalized'
		probe_description = 'ROI probe' if is_roi_probe else 'ETCO2 probe'
		
		# Extract results
		delay_maps = delay_results['delay_maps']
		correlation_maps = delay_results['correlation_maps']
		masked_delay_maps = delay_results['masked_delay_maps']
		delay_range = delay_results['delay_range']
		correlation_threshold = delay_results.get('correlation_threshold', 0.6)
		
		if masked_delay_maps is None or correlation_maps is None:
			raise ValueError("Masked delay maps and correlation maps must not be None")

		# Create BIDS paths with label entity
		label_entity = self._get_roi_label_entity()
		delay_base = f"sub-{participant}_task-{task}{label_entity}_space-{space}_desc-delaymasked_bold"
		correlation_base = f"sub-{participant}_task-{task}{label_entity}_space-{space}_desc-correlation_bold"

		_, delay_nii_path, delay_json_path, _, _ = self._create_bids_paths(participant, "func", delay_base)
		_, correlation_nii_path, correlation_json_path, _, _ = self._create_bids_paths(participant, "func", correlation_base)
		
		# Save maps as NIfTI
		delay_img = nib.Nifti1Image(masked_delay_maps, normalized_bold_container.affine)
		nib.save(delay_img, delay_nii_path)
		
		correlation_img = nib.Nifti1Image(correlation_maps, normalized_bold_container.affine, normalized_bold_container.header)
		nib.save(correlation_img, correlation_nii_path)
		
		# Create JSON sidecars using helper function
		delay_description = f"Voxel-wise optimal delay map showing temporal delays that maximize correlation between BOLD signal and shifted {probe_description}, masked by correlation threshold (≥{correlation_threshold})"
		processing_description = f"Each voxel contains the delay (in seconds) that produced the maximum absolute correlation with the shifted {probe_description} signal. Only voxels with correlation ≥{correlation_threshold} are included; others are set to NaN"
		
		if global_delay is not None:
			delay_description += f", relative to global delay baseline of {global_delay:.3f}s"
			processing_description += f". Delays are expressed relative to the global delay ({global_delay:.3f}s), so positive values indicate delays longer than the global delay, and negative values indicate delays shorter than the global delay"
		
		delay_sidecar = self._create_standard_sidecar_base(delay_description, "s", space, task, "delay")
		delay_sidecar.update({
			"DelayRange": {
				"Minimum": float(np.min(delay_range)),
				"Maximum": float(np.max(delay_range)),
				"Description": "Range of delays tested in seconds"
			},
			"ProcessingDescription": processing_description,
			"NumberOfDelayConditions": len(delay_range),
			"DelayStep": float(delay_range[1] - delay_range[0]) if len(delay_range) > 1 else 1.0,
			"CorrelationMask": {
				"Threshold": float(correlation_threshold),
				"Description": f"Only voxels with absolute correlation ≥{correlation_threshold} are included in the delay map"
			}
		})
		
		if global_delay is not None:
			delay_sidecar["GlobalDelayReference"] = {
				"Value": float(global_delay),
				"Units": "s",
				"Description": "Global delay value used as reference point. Delay map values are relative to this baseline."
			}
		
		correlation_sidecar = self._create_standard_sidecar_base(
			f"Voxel-wise maximum correlation map showing the highest correlation between BOLD signal and shifted {probe_description} across all tested delays",
			"correlation coefficient", space, task, "correlation")
		correlation_sidecar.update({
			"CorrelationRange": {
				"Minimum": -1.0,
				"Maximum": 1.0,
				"Description": "Theoretical range of correlation coefficients"
			},
			"ProcessingDescription": f"Each voxel contains the maximum absolute correlation coefficient achieved across all tested delays with the {probe_description} signal",
			"NumberOfDelayConditions": len(delay_range),
			"DelayStep": float(delay_range[1] - delay_range[0]) if len(delay_range) > 1 else 1.0
		})
		
		# Save JSON sidecars
		with open(delay_json_path, 'w') as f:
			safe_json_dump(delay_sidecar, f, indent=2)
		
		with open(correlation_json_path, 'w') as f:
			safe_json_dump(correlation_sidecar, f, indent=2)
		
		if self.logger:
			self.logger.info(f"Saved delay map to {delay_nii_path}")
			self.logger.info(f"Saved correlation map to {correlation_nii_path}")
		
		# Create figure for the masked delay map
		try:
			fig_path = self.create_masked_delay_figure(delay_nii_path, participant, task, space)
		except Exception as e:
			if self.logger:
				self.logger.warning(f"Could not create masked delay map figure: {e}")
			fig_path = None
		
		return delay_nii_path, correlation_nii_path

	def create_masked_delay_figure(self, delay_nii_path, participant, task, space):
		"""
		Create a figure showing the masked delay map using custom lightbox plotting.
		
		Parameters:
		-----------
		delay_nii_path : str
			Path to the masked delay map NIfTI file
		participant : str
			Participant ID
		task : str
			Task name
		space : str
			Space name (e.g., 'MNI152NLin2009cAsym')
			
		Returns:
		--------
		str
			Path to the saved figure
		"""
		import matplotlib.pyplot as plt
		import matplotlib.colors as mcolors
		import nibabel as nib
		import numpy as np

		# Create figures directory with label entity
		label_entity = self._get_roi_label_entity()
		_, _, _, _, fig_path = self._create_bids_paths(participant, "figures",
		                                              f"sub-{participant}_task-{task}{label_entity}_space-{space}_desc-delaymasked")
		
		# Load the delay map
		delay_img = nib.load(delay_nii_path)
		delay_data = delay_img.get_fdata()
		
		# Calculate cut coordinates for lightbox display - select 20 slices across the brain
		z_min, z_max = 10, delay_img.shape[2] - 10  # Avoid empty slices at edges
		slice_indices = np.linspace(z_min, z_max, 20, dtype=int)
		
		# Setup lightbox figure
		title = f'Masked Delay Map - Subject {participant}, Task {task}'
		fig, gs_inner, ax_cbar, n_rows, n_cols = self._setup_lightbox_figure(title)
		
		# Set colormap and normalization
		cmap = plt.cm.get_cmap('coolwarm')
		vmin, vmax = -5, 5
		norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
		
		# Plot each slice
		for i, slice_idx in enumerate(slice_indices):
			row = i // n_cols
			col = i % n_cols
			
			ax = fig.add_subplot(gs_inner[row, col])
			ax.set_facecolor('black')
			
			# Extract and display the slice
			slice_data = delay_data[:, :, slice_idx]
			slice_data = np.rot90(slice_data, k=1)  # Rotate for proper orientation
			masked_slice = np.ma.masked_invalid(slice_data)  # Handle NaN values
			
			# Display the slice
			ax.imshow(masked_slice, cmap=cmap, norm=norm, 
			         interpolation='nearest', aspect='equal')
			
			# Remove axes and add slice number
			ax.set_xticks([])
			ax.set_yticks([])
			ax.set_title(f'z={slice_idx}', color='white', fontsize=8, pad=2)			
		
		# Add custom colorbar
		cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax_cbar)
		cbar.set_label('Delay (seconds)', rotation=270, labelpad=15, color='white', fontsize=11)
		cbar.ax.tick_params(colors='white', labelsize=9, width=0.5)
		cbar.outline.set_edgecolor('white')
		cbar.outline.set_linewidth(0.5)
		
		plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='black')
		plt.close(fig)
		
		if self.logger:
			self.logger.info(f"Created masked delay map figure at {fig_path}")
		
		return fig_path

	def save_cvr_maps(self, cvr_results, bold_container, participant, task, space, probe_container=None):
		"""
		Save CVR maps to NIfTI files with BIDS naming and JSON sidecars.
		
		Parameters:
		-----------
		cvr_results : dict
			Dictionary containing 'cvr_maps' from CVRProcessor
		bold_container : BoldContainer
			BOLD container with spatial information (affine, header)
		participant : str
			Participant ID
		task : str
			Task name
		space : str
			Space name (e.g., 'MNI152NLin2009cAsym')
			
		Returns:
		--------
		str
			Path to saved CVR map file
		"""
		import json
		import numpy as np
		import nibabel as nib
		
		# Extract CVR maps
		cvr_maps = cvr_results['cvr_maps']

		if cvr_maps is None:
			raise ValueError("CVR maps must not be None")

		# Create BIDS paths with label entity
		label_entity = self._get_roi_label_entity()
		cvr_base = f"sub-{participant}_task-{task}{label_entity}_space-{space}_desc-cvr_bold"
		_, cvr_nii_path, cvr_json_path, _, _ = self._create_bids_paths(participant, "func", cvr_base)
		
		# Save CVR map as NIfTI
		cvr_img = nib.Nifti1Image(cvr_maps, bold_container.affine)
		nib.save(cvr_img, cvr_nii_path)
		
		# Determine probe type and appropriate units/description
		probe_type = getattr(probe_container, 'probe_type', 'etco2') if probe_container else 'etco2'
		is_roi_probe = 'roi_probe' in probe_type  # Handle both 'roi_probe' and 'roi_probe_normalized'
		
		if is_roi_probe:
			description = "Cerebrovascular reactivity (CVR) map showing voxel-wise CVR values computed using GLM regression between BOLD signal and shifted ROI probe signal"
			units = "arbitrary units (unitless)"
			processing_desc = "CVR computed as b1/(b0 + probe_baseline*b1) where b0 and b1 are GLM coefficients from BOLD = b0 + b1*roi_probe_signal regression. Note: CVR values are in arbitrary units since ROI probe signal has no physiological calibration."
			probe_info = "ROI-based probe extracted from brain region signal"
		else:
			description = "Cerebrovascular reactivity (CVR) map showing voxel-wise CVR values computed using GLM regression between BOLD signal and shifted ETCO2 probe signal"
			units = "%BOLD/mmHg"
			processing_desc = "CVR computed as b1/(b0 + probe_baseline*b1) where b0 and b1 are GLM coefficients from BOLD = b0 + b1*etco2_signal regression"
			probe_info = "End-tidal CO2 (ETCO2) from physiological recordings"
		
		# Create JSON sidecar using helper function
		cvr_sidecar = self._create_standard_sidecar_base(description, units, space, task, "cvr")
		cvr_sidecar.update({
			"ProcessingDescription": processing_desc,
			"GLMFormula": "BOLD ~ intercept + probe_signal",
			"Method": "GeneralLinearModel",
			"ProbeType": getattr(probe_container, 'probe_type', 'etco2') if probe_container else 'etco2',
			"ProbeDescription": probe_info
		})
		
		# Save JSON sidecar
		with open(cvr_json_path, 'w') as f:
			safe_json_dump(cvr_sidecar, f, indent=2)
		
		if self.logger:
			self.logger.info(f"Saved CVR map to {cvr_nii_path}")
		
		# Create figure for the CVR map
		try:
			fig_path = self.create_cvr_figure(cvr_nii_path, participant, task, space, probe_container)
		except Exception as e:
			if self.logger:
				self.logger.warning(f"Could not create CVR map figure: {e}")
			fig_path = None
		
		return cvr_nii_path

	def save_coefficient_maps(self, cvr_results, bold_container, participant, task, space, probe_container=None):
		"""
		Save GLM coefficient maps (b0 and b1) to NIfTI files with BIDS naming and JSON sidecars.
		
		Parameters:
		-----------
		cvr_results : dict
			Dictionary containing 'b0_maps' and 'b1_maps' from CVRProcessor
		bold_container : BoldContainer
			BOLD container with spatial information (affine, header)
		participant : str
			Participant ID
		task : str
			Task name
		space : str
			Space name (e.g., 'MNI152NLin2009cAsym')
		probe_container : ProbeContainer, optional
			Container with probe information for metadata
			
		Returns:
		--------
		tuple
			Paths to saved b0 and b1 coefficient map files (b0_path, b1_path)
		"""
		import json
		import numpy as np
		import nibabel as nib
		
		# Determine probe type for appropriate descriptions
		probe_type = getattr(probe_container, 'probe_type', 'etco2') if probe_container else 'etco2'
		is_roi_probe = 'roi_probe' in probe_type  # Handle both 'roi_probe' and 'roi_probe_normalized'
		probe_description = 'ROI probe' if is_roi_probe else 'ETCO2 probe'
		
		# Extract coefficient maps
		b0_maps = cvr_results.get('b0_maps')
		b1_maps = cvr_results.get('b1_maps')

		if b0_maps is None or b1_maps is None:
			raise ValueError("Both b0_maps and b1_maps must be present in cvr_results")

		# Create BIDS paths with label entity
		label_entity = self._get_roi_label_entity()
		b0_base = f"sub-{participant}_task-{task}{label_entity}_space-{space}_desc-b0_bold"
		b1_base = f"sub-{participant}_task-{task}{label_entity}_space-{space}_desc-b1_bold"

		_, b0_nii_path, b0_json_path, _, _ = self._create_bids_paths(participant, "func", b0_base)
		_, b1_nii_path, b1_json_path, _, _ = self._create_bids_paths(participant, "func", b1_base)
		
		# Save coefficient maps as NIfTI
		b0_img = nib.Nifti1Image(b0_maps, bold_container.affine)
		nib.save(b0_img, b0_nii_path)
		
		b1_img = nib.Nifti1Image(b1_maps, bold_container.affine)
		nib.save(b1_img, b1_nii_path)
		
		# Create JSON sidecars using helper function
		b0_sidecar = self._create_standard_sidecar_base(
			"GLM intercept coefficient (b0) map from regression: BOLD = b0 + b1*probe_signal",
			"arbitrary", space, task, "coefficient")
		b0_sidecar.update({
			"ProcessingDescription": f"Intercept coefficient from voxel-wise GLM regression between BOLD signal and shifted {probe_description} signal",
			"GLMFormula": "BOLD ~ intercept + probe_signal",
			"Method": "GeneralLinearModel",
			"CoefficientType": "intercept"
		})
		
		b1_sidecar = self._create_standard_sidecar_base(
			"GLM slope coefficient (b1) map from regression: BOLD = b0 + b1*probe_signal",
			"signal_change_per_probe_unit", space, task, "coefficient")
		b1_sidecar.update({
			"ProcessingDescription": f"Slope coefficient from voxel-wise GLM regression between BOLD signal and shifted {probe_description} signal",
			"GLMFormula": "BOLD ~ intercept + probe_signal",
			"Method": "GeneralLinearModel",
			"CoefficientType": "slope"
		})
		
		# Save JSON sidecars
		with open(b0_json_path, 'w') as f:
			safe_json_dump(b0_sidecar, f, indent=2)
		
		with open(b1_json_path, 'w') as f:
			safe_json_dump(b1_sidecar, f, indent=2)
		
		if self.logger:
			self.logger.info(f"Saved b0 coefficient map to {b0_nii_path}")
			self.logger.info(f"Saved b1 coefficient map to {b1_nii_path}")
		
		return b0_nii_path, b1_nii_path

	def save_regressor_4d_map(self, delay_results, resampled_shifted_probes, bold_container, participant, task, space, probe_container=None):
		"""
		Create and save a 4D NIfTI file where each voxel contains the timecourse of the 
		resampled, optimally-shifted regressor based on the optimal delay for that voxel.
		
		Parameters:
		-----------
		delay_results : dict
			Dictionary containing 'delay_maps' from DelayProcessor
		resampled_shifted_probes : tuple
			Tuple containing (shifted_signals, time_delays_seconds) with non-normalized resampled probes
		bold_container : BoldContainer
			BOLD container with spatial information (affine, header)
		participant : str
			Participant ID
		task : str
			Task name
		space : str
			Space name (e.g., 'MNI152NLin2009cAsym')
		probe_container : ProbeContainer, optional
			Container with probe information for metadata
			
		Returns:
		--------
		str
			Path to saved 4D regressor map file
		"""
		import json
		import numpy as np
		import nibabel as nib
		
		# Extract delay maps and probe data
		delay_maps = delay_results['delay_maps']
		shifted_signals, time_delays_seconds = resampled_shifted_probes
		
		if delay_maps is None or shifted_signals is None:
			raise ValueError("Both delay_maps and resampled shifted signals must be provided")
		
		# Get BOLD data dimensions
		x, y, z, t = bold_container.data.shape
		n_delays, n_timepoints = shifted_signals.shape
		
		# Initialize 4D regressor map
		regressor_4d = np.full((x, y, z, n_timepoints), np.nan)
		
		# Get brain mask
		if hasattr(bold_container, 'mask') and bold_container.mask is not None:
			brain_mask = bold_container.mask > 0
		else:
			brain_mask = ~np.isnan(bold_container.data).any(axis=3)
		
		# Fill each voxel with its optimal regressor timecourse
		voxel_count = 0
		total_brain_voxels = np.sum(brain_mask)
		n_jobs = self.config.get('n_jobs', -1)
		
		if n_jobs == 1:
			# Sequential processing (original implementation)
			if self.logger:
				self.logger.info("Using sequential processing for 4D regressor map generation")
			voxel_count = self._fill_regressor_4d_sequential(
				regressor_4d, brain_mask, x, y, z, delay_maps, time_delays_seconds, 
				shifted_signals, total_brain_voxels
			)
		else:
			# Parallel processing
			if self.logger:
				actual_n_jobs = n_jobs if n_jobs > 0 else None  # None means all CPUs
				self.logger.info(f"Using parallel processing for 4D regressor map generation (n_jobs={actual_n_jobs})")
			voxel_count = self._fill_regressor_4d_parallel(
				regressor_4d, brain_mask, x, y, z, delay_maps, time_delays_seconds, 
				shifted_signals, total_brain_voxels, n_jobs
			)
		
		# Create BIDS paths with label entity
		label_entity = self._get_roi_label_entity()
		regressor_base = f"sub-{participant}_task-{task}{label_entity}_space-{space}_desc-regressor4d_bold"
		_, regressor_nii_path, regressor_json_path, _, _ = self._create_bids_paths(participant, "func", regressor_base)
		
		# Save 4D regressor map as NIfTI
		regressor_img = nib.Nifti1Image(regressor_4d, bold_container.affine, bold_container.header)
		nib.save(regressor_img, regressor_nii_path)
		
		# Create JSON sidecar using helper function
		regressor_sidecar = self._create_standard_sidecar_base(
			"4D regressor map where each voxel contains the timecourse of the resampled, optimally-shifted probe signal based on the optimal delay for that voxel",
			"probe_units", space, task, "regressor_timecourse")
		regressor_sidecar.update({
			"ProcessingDescription": "Each voxel contains the resampled probe signal shifted by the optimal delay determined from cross-correlation analysis. Non-brain voxels are set to NaN.",
			"NumberOfTimepoints": int(n_timepoints),
			"NumberOfDelayConditions": int(n_delays),
			"DelayRange": {
				"Minimum": float(np.min(time_delays_seconds)),
				"Maximum": float(np.max(time_delays_seconds)),
				"Description": "Range of delays tested in seconds"
			},
			"SpatialReference": "Each voxel uses its individually optimal delay from delay mapping analysis",
			"ProbeType": getattr(probe_container, 'probe_type', 'etco2') if probe_container else 'etco2'
		})
		
		# Save JSON sidecar
		with open(regressor_json_path, 'w') as f:
			safe_json_dump(regressor_sidecar, f, indent=2)
		
		if self.logger:
			self.logger.info(f"Saved 4D regressor map to {regressor_nii_path}")
			self.logger.info(f"Processed {voxel_count:,} brain voxels for 4D regressor map")
		
		return regressor_nii_path

	def _fill_regressor_4d_sequential(self, regressor_4d, brain_mask, x, y, z, delay_maps, 
									time_delays_seconds, shifted_signals, total_brain_voxels):
		"""Sequential 4D regressor filling (original implementation)"""
		import numpy as np
		
		voxel_count = 0
		
		for i in range(x):
			for j in range(y):
				for k in range(z):
					if brain_mask[i, j, k]:
						# Get optimal delay for this voxel
						optimal_delay = delay_maps[i, j, k]
						
						# Skip if delay is NaN (masked voxel)
						if np.isnan(optimal_delay):
							continue
						
						# Find the closest delay in our time_delays_seconds array
						delay_idx = np.argmin(np.abs(time_delays_seconds - optimal_delay))
						
						# Extract the corresponding shifted probe signal
						regressor_4d[i, j, k, :] = shifted_signals[delay_idx, :]
						
						voxel_count += 1
						
						# Progress logging
						if self.logger and voxel_count % 20000 == 0:
							progress = (voxel_count / total_brain_voxels) * 100
							self.logger.debug(f"Processed {voxel_count:,}/{total_brain_voxels:,} voxels ({progress:.1f}%) for 4D regressor map")
		
		return voxel_count

	def _fill_regressor_4d_parallel(self, regressor_4d, brain_mask, x, y, z, delay_maps, 
								   time_delays_seconds, shifted_signals, total_brain_voxels, n_jobs):
		"""Parallel 4D regressor filling using joblib"""
		from joblib import Parallel, delayed
		import numpy as np
		
		# Get brain voxel coordinates
		brain_coords = np.where(brain_mask)
		brain_voxels = list(zip(brain_coords[0], brain_coords[1], brain_coords[2]))
		
		if self.logger:
			self.logger.info(f"Processing {len(brain_voxels):,} brain voxels in parallel for 4D regressor map")
		
		# Process voxels in parallel
		results = Parallel(n_jobs=n_jobs, verbose=1 if self.logger else 0)(
			delayed(_process_regressor_voxel)(voxel_coord, delay_maps, time_delays_seconds, shifted_signals) 
			for voxel_coord in brain_voxels
		)
		
		# Assemble results
		voxel_count = 0
		
		for i, j, k, signal in results:
			if signal is not None:
				regressor_4d[i, j, k, :] = signal
				voxel_count += 1
		
		return voxel_count

	def create_cvr_figure(self, cvr_nii_path, participant, task, space, probe_container=None):
		"""
		Create a figure showing the CVR map using custom lightbox plotting.
		
		Parameters:
		-----------
		cvr_nii_path : str
			Path to the CVR map NIfTI file
		participant : str
			Participant ID
		task : str
			Task name
		space : str
			Space name (e.g., 'MNI152NLin2009cAsym')
			
		Returns:
		--------
		str
			Path to the saved figure
		"""
		import matplotlib.pyplot as plt
		import matplotlib.colors as mcolors
		import nibabel as nib
		import numpy as np

		# Create figures directory with label entity
		label_entity = self._get_roi_label_entity()
		_, _, _, _, fig_path = self._create_bids_paths(participant, "figures",
		                                              f"sub-{participant}_task-{task}{label_entity}_space-{space}_desc-cvr")
		
		# Load the CVR map
		cvr_img = nib.load(cvr_nii_path)
		cvr_data = cvr_img.get_fdata()
		
		# Calculate cut coordinates for lightbox display - select 20 slices across the brain
		z_min, z_max = 10, cvr_img.shape[2] - 10  # Avoid empty slices at edges
		slice_indices = np.linspace(z_min, z_max, 20, dtype=int)
		
		# Setup lightbox figure
		title = f'CVR Map - Subject {participant}, Task {task}'
		fig, gs_inner, ax_cbar, n_rows, n_cols = self._setup_lightbox_figure(title)
		
		# Determine probe type for appropriate units and scaling
		probe_type = getattr(probe_container, 'probe_type', 'etco2') if probe_container else 'etco2'
		is_roi_probe = 'roi_probe' in probe_type  # Handle both 'roi_probe' and 'roi_probe_normalized'
		
		# Set colormap and normalization
		cmap = plt.cm.get_cmap('hot').copy()
		cmap.set_bad(color='black')  # Set NaN/masked values to black
		
		if is_roi_probe:
			# For ROI probe mode, adjust vmin/vmax to capture central portion of histogram
			valid_data = cvr_data[np.isfinite(cvr_data)]
			if len(valid_data) > 0:
				vmin = np.percentile(valid_data, 5)   # 5th percentile
				vmax = np.percentile(valid_data, 95)  # 95th percentile
			else:
				vmin, vmax = 0, 1
			colorbar_label = 'CVR (arbitrary units)'
		else:
			# Traditional physio mode scaling
			vmin, vmax = 0, 0.8
			colorbar_label = 'CVR (%BOLD/mmHg)'
		
		norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
		
		# Plot each slice
		for i, slice_idx in enumerate(slice_indices):
			row = i // n_cols
			col = i % n_cols
			
			ax = fig.add_subplot(gs_inner[row, col])
			ax.set_facecolor('black')
			
			# Extract and display the slice
			slice_data = cvr_data[:, :, slice_idx]
			slice_data = np.rot90(slice_data, k=1)  # Rotate for proper orientation
			masked_slice = np.ma.masked_invalid(slice_data)  # Handle NaN values
			
			# Display the slice
			ax.imshow(masked_slice, cmap=cmap, norm=norm, 
			         interpolation='nearest', aspect='equal')
			
			# Remove axes and add slice number
			ax.set_xticks([])
			ax.set_yticks([])
			ax.set_title(f'z={slice_idx}', color='white', fontsize=8, pad=2)
		
		# Add custom colorbar
		cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax_cbar)
		cbar.set_label(colorbar_label, rotation=270, labelpad=15, color='white', fontsize=11)
		cbar.ax.tick_params(colors='white', labelsize=9, width=0.5)
		cbar.outline.set_edgecolor('white')
		cbar.outline.set_linewidth(0.5)
		
		plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='black')
		plt.close(fig)
		
		if self.logger:
			self.logger.info(f"Created CVR map figure at {fig_path}")
		
		return fig_path



	def _fill_regressor_4d_parallel(self, regressor_4d, brain_mask, x, y, z, delay_maps,
	                                time_delays_seconds, shifted_signals, total_brain_voxels, n_jobs):
		"""Parallel processing for 4D regressor map generation with chunked processing"""
		from joblib import Parallel, delayed
		import numpy as np
		
		# Set matplotlib backend to non-GUI to avoid tkinter issues in parallel processing
		import matplotlib
		matplotlib.use('Agg')
		
		# Get brain voxel coordinates
		brain_coords = np.where(brain_mask)
		brain_voxels = list(zip(brain_coords[0], brain_coords[1], brain_coords[2]))
		
		if self.logger:
			self.logger.info(f"Processing {len(brain_voxels):,} brain voxels in parallel for 4D regressor map using chunked processing")
		
		# Calculate optimal chunk size
		if n_jobs == -1:
			import multiprocessing
			actual_n_jobs = multiprocessing.cpu_count()
		else:
			actual_n_jobs = n_jobs
			
		chunk_size = max(1000, min(5000, len(brain_voxels) // (actual_n_jobs * 4)))
		
		if self.logger:
			self.logger.info(f"Using chunk size: {chunk_size} voxels per chunk for 4D regressor processing")
		
		# Split brain voxels into chunks
		voxel_chunks = [brain_voxels[i:i + chunk_size] for i in range(0, len(brain_voxels), chunk_size)]
		
		# Process chunks in parallel using multiprocessing backend for true parallelization
		chunk_results = Parallel(n_jobs=n_jobs, backend='multiprocessing', verbose=1 if self.logger else 0)(
			delayed(_process_regressor_voxel_chunk)(chunk, delay_maps, time_delays_seconds, shifted_signals) 
			for chunk in voxel_chunks
		)
		
		# Flatten results from all chunks
		results = []
		for chunk_result in chunk_results:
			results.extend(chunk_result)
		
		# Assemble results
		voxel_count = 0
		
		for i, j, k, signal in results:
			if signal is not None:
				regressor_4d[i, j, k, :] = signal
				voxel_count += 1
		
		return voxel_count