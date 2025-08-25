"""
FCS file processing functions adapted from the original codebase.
Handles FCS file reading, compensation, and transformation.
"""

import numpy as np
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import flowkit as fk

logger = logging.getLogger(__name__)


def read_fcs_file(
    fcs_filepath: Path,
    selected_indices: List[int],
    compensation_method: Optional[str] = None,
    max_cells: Optional[int] = None
) -> np.ndarray:
    """
    Read and process an FCS file with compensation and transformation.
    
    Args:
        fcs_filepath: Path to the FCS file
        selected_indices: Indices of channels to include in output
        compensation_method: Compensation method ('spill', 'none', or file path)
        max_cells: Maximum number of cells to return (takes first N)
        
    Returns:
        numpy array of cell data with shape (n_cells, n_selected_channels)
    """
    logger.debug(f"Processing FCS file: {fcs_filepath}")
    
    try:
        # Load the FCS file
        sample = fk.Sample(str(fcs_filepath))
        initial_cell_count = sample._raw_events.shape[0]
        logger.debug(f"Initial cell count: {initial_cell_count}")
        
        # Apply compensation if specified
        if compensation_method and compensation_method != 'none':
            try:
                if compensation_method == 'spill':
                    logger.debug("Using internal spill matrix for compensation")
                    if 'spill' in sample.metadata:
                        sample.apply_compensation(sample.metadata['spill'])
                    elif 'spillover' in sample.metadata:
                        sample.apply_compensation(sample.metadata['spillover'])
                    else:
                        logger.warning("No spill matrix found in FCS file metadata")
                else:
                    # Assume it's a file path
                    comp_file = Path(compensation_method)
                    if comp_file.exists():
                        logger.debug(f"Using compensation file: {comp_file}")
                        sample.apply_compensation(str(comp_file))
                    else:
                        logger.warning(f"Compensation file not found: {comp_file}")
            except Exception as e:
                logger.warning(f"Error applying compensation: {e}")
                logger.warning("Continuing without compensation")
        
        # Apply logicle transformation
        max_value = sample._raw_events.max()
        current_max_intensity = 262144
        
        if max_value > 262144:
            current_max_intensity = 4194304.0
        
        logicle_xform = fk.transforms.LogicleTransform(
            param_t=current_max_intensity,
            param_w=0.5,
            param_m=4.5,
            param_a=0
        )
        
        try:
            sample.apply_transform(logicle_xform)
            logger.debug("Applied logicle transformation")
        except Exception as e:
            logger.error(f"Error applying logicle transform: {e}")
            raise
        
        # Get all transformed fluorescent channel data
        available_channels = sample.pnn_labels
        df_all_fluoro = sample.as_dataframe(source='xform', subsample=False)[available_channels]
        
        # Convert to numpy array
        data_all_channels = np.array(df_all_fluoro)
        
        logger.debug(f"Data shape after transformation: {data_all_channels.shape}")
        logger.debug(f"Data max value: {data_all_channels.max():.3f}")
        
        # Select only the specified channels
        if max(selected_indices) >= data_all_channels.shape[1]:
            raise ValueError(f"Selected index {max(selected_indices)} exceeds available channels ({data_all_channels.shape[1]})")
        
        data_selected_channels = data_all_channels[:, selected_indices]
        
        # Limit number of cells if requested (take first N cells)
        if max_cells is not None and data_selected_channels.shape[0] > max_cells:
            data_selected_channels = data_selected_channels[:max_cells, :]
            logger.debug(f"Limited to first {max_cells} cells")
        
        logger.debug(f"Final data shape: {data_selected_channels.shape}")
        
        # Validate output
        if data_selected_channels.shape[0] == 0:
            logger.warning(f"No cells remaining after processing {fcs_filepath}")
            return np.zeros((0, len(selected_indices)))
        
        if data_selected_channels.shape[1] != len(selected_indices):
            raise ValueError(f"Output channel count mismatch: expected {len(selected_indices)}, got {data_selected_channels.shape[1]}")
        
        return data_selected_channels
        
    except Exception as e:
        logger.error(f"Error processing FCS file {fcs_filepath}: {e}")
        # Return empty array with correct number of channels on error
        return np.zeros((0, len(selected_indices)))


def build_reference_dataset(
    reference_batch_path: Path,
    selected_indices: List[int],
    compensation_method: Optional[str] = None,
    max_cells_per_file: Optional[int] = None,
    max_total_cells: Optional[int] = None
) -> np.ndarray:
    """
    Build a dataset from all FCS files in the reference batch.
    
    Args:
        reference_batch_path: Path to reference batch directory
        selected_indices: Indices of channels to include
        compensation_method: Compensation method
        max_cells_per_file: Maximum cells to take from each file
        max_total_cells: Maximum total cells in final dataset
        
    Returns:
        numpy array of combined cell data
    """
    logger.info(f"Building reference dataset from: {reference_batch_path}")
    
    # Find all FCS files
    fcs_files = sorted(list(reference_batch_path.glob("*.fcs")))
    if not fcs_files:
        raise ValueError(f"No FCS files found in reference batch: {reference_batch_path}")
    
    logger.info(f"Found {len(fcs_files)} FCS files in reference batch")
    
    sample_list = []
    processing_errors = []
    cell_threshold = 1000  # Minimum cells per file to include
    
    for i, fcs_file in enumerate(fcs_files):
        logger.debug(f"Processing file {i+1}/{len(fcs_files)}: {fcs_file.name}")
        
        try:
            sample_data = read_fcs_file(
                fcs_file,
                selected_indices,
                compensation_method,
                max_cells_per_file
            )
            
            if sample_data.shape[0] > cell_threshold:
                sample_list.append(sample_data)
                logger.debug(f"Added {sample_data.shape[0]} cells from {fcs_file.name}")
            else:
                error_msg = f"Sample {fcs_file.name} too small ({sample_data.shape[0]} cells < {cell_threshold})"
                processing_errors.append(error_msg)
                logger.warning(error_msg)
                
        except Exception as e:
            error_msg = f"Error processing {fcs_file.name}: {e}"
            processing_errors.append(error_msg)
            logger.error(error_msg)
    
    if not sample_list:
        error_details = "\n".join(processing_errors) if processing_errors else "Unknown error"
        raise ValueError(f"No valid samples found in reference batch. Errors:\n{error_details}")
    
    logger.info(f"Successfully processed {len(sample_list)} FCS files")
    
    # Combine all samples
    all_samples = np.vstack(sample_list)
    logger.info(f"Combined dataset: {all_samples.shape[0]} cells, {all_samples.shape[1]} channels")
    
    # Select subset of cells if needed
    if max_total_cells and all_samples.shape[0] > max_total_cells:
        # Randomly select cells
        indices = np.random.choice(all_samples.shape[0], max_total_cells, replace=False)
        selected_cells = all_samples[indices, :]
        logger.info(f"Randomly selected {max_total_cells} cells from {all_samples.shape[0]} total")
    else:
        selected_cells = all_samples
    
    return selected_cells


def process_batch_files_with_scatter(
    batch_path: Path,
    selected_indices: List[int],
    scatter_indices: List[int],
    compensation_method: Optional[str] = None,
    max_cells_per_file: Optional[int] = None
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Process all FCS files in a batch directory, returning both marker and scatter data.
    
    Args:
        batch_path: Path to batch directory
        selected_indices: Indices of marker channels to include
        scatter_indices: Indices of scatter channels to include
        compensation_method: Compensation method
        max_cells_per_file: Maximum cells per file
        
    Returns:
        Dictionary mapping filename to dict with 'markers' and 'scatter' data
    """
    logger.info(f"Processing batch with scatter channels: {batch_path}")
    
    fcs_files = sorted(list(batch_path.glob("*.fcs")))
    if not fcs_files:
        raise ValueError(f"No FCS files found in batch: {batch_path}")
    
    logger.info(f"Found {len(fcs_files)} FCS files in batch")
    
    processed_files = {}
    
    for i, fcs_file in enumerate(fcs_files):
        logger.info(f"Processing file {i+1}/{len(fcs_files)}: {fcs_file.name}")
        
        try:
            # Get marker data (compensated and transformed)
            marker_data = read_fcs_file(
                fcs_file,
                selected_indices,
                compensation_method,
                max_cells_per_file
            )
            
            # Get scatter data (raw, no compensation or transformation)
            scatter_data = read_fcs_scatter_data(
                fcs_file,
                scatter_indices,
                max_cells_per_file
            )
            
            # Ensure same number of events for both
            if marker_data.shape[0] > 0 and scatter_data.shape[0] > 0:
                min_events = min(marker_data.shape[0], scatter_data.shape[0])
                marker_data = marker_data[:min_events, :]
                scatter_data = scatter_data[:min_events, :]
                
                processed_files[fcs_file.name] = {
                    'markers': marker_data,
                    'scatter': scatter_data
                }
                logger.debug(f"Processed {min_events} cells from {fcs_file.name}")
            else:
                logger.warning(f"No cells remaining after processing {fcs_file.name}")
                
        except Exception as e:
            logger.error(f"Error processing {fcs_file.name}: {e}")
            # Continue with other files
    
    logger.info(f"Successfully processed {len(processed_files)} files from batch")
    return processed_files


def read_fcs_scatter_data(
    fcs_filepath: Path,
    scatter_indices: List[int],
    max_cells: Optional[int] = None
) -> np.ndarray:
    """
    Read scatter channel data directly from raw events (no compensation or transformation).
    
    Args:
        fcs_filepath: Path to the FCS file
        scatter_indices: Indices of scatter channels to extract
        max_cells: Maximum number of cells to return (takes first N)
        
    Returns:
        numpy array of scatter data with shape (n_cells, n_scatter_channels)
    """
    logger.debug(f"Reading scatter data from: {fcs_filepath}")
    
    try:
        import flowkit as fk
        
        # Load the FCS file
        sample = fk.Sample(str(fcs_filepath))
        
        # Get raw events (no processing)
        raw_events = sample._raw_events
        
        # Extract scatter channels
        if max(scatter_indices) >= raw_events.shape[1]:
            raise ValueError(f"Scatter index {max(scatter_indices)} exceeds available channels ({raw_events.shape[1]})")
        
        scatter_data = raw_events[:, scatter_indices]
        
        # Limit number of cells if requested (take first N cells)
        if max_cells is not None and scatter_data.shape[0] > max_cells:
            scatter_data = scatter_data[:max_cells, :]
            logger.debug(f"Limited scatter data to first {max_cells} cells")
        
        logger.debug(f"Extracted scatter data shape: {scatter_data.shape}")
        
        return scatter_data
        
    except Exception as e:
        logger.error(f"Error reading scatter data from {fcs_filepath}: {e}")
        # Return empty array with correct number of channels on error
        return np.zeros((0, len(scatter_indices)))


def validate_fcs_channels(fcs_file: Path, expected_markers: List[str]) -> bool:
    """
    Validate that an FCS file contains the expected markers.
    
    Args:
        fcs_file: Path to FCS file
        expected_markers: List of marker names to check for
        
    Returns:
        True if all markers are found, False otherwise
    """
    try:
        sample = fk.Sample(str(fcs_file))
        available_markers = sample.pns_labels
        
        missing_markers = [marker for marker in expected_markers if marker not in available_markers]
        
        if missing_markers:
            logger.error(f"Missing markers in {fcs_file.name}: {missing_markers}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating FCS file {fcs_file}: {e}")
        return False


def get_fcs_file_info(fcs_file: Path) -> Dict[str, Any]:
    """
    Get basic information about an FCS file.
    
    Args:
        fcs_file: Path to FCS file
        
    Returns:
        Dictionary with file information
    """
    try:
        sample = fk.Sample(str(fcs_file))
        
        info = {
            'filename': fcs_file.name,
            'path': str(fcs_file),
            'cell_count': sample._raw_events.shape[0],
            'channel_count': len(sample.pnn_labels),
            'pnn_labels': sample.pnn_labels,
            'pns_labels': sample.pns_labels,
            'file_size_bytes': fcs_file.stat().st_size,
            'has_spill_matrix': 'spill' in sample.metadata or 'spillover' in sample.metadata
        }
        
        return info
        
    except Exception as e:
        logger.error(f"Error getting info for FCS file {fcs_file}: {e}")
        return {
            'filename': fcs_file.name,
            'path': str(fcs_file),
            'error': str(e)
        }