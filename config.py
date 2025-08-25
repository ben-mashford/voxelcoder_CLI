"""
Configuration handling for the Batch Alignment CLI.
Loads and validates YAML configuration files.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import flowkit as fk

logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML syntax in {config_path}: {e}")
    except Exception as e:
        raise ValueError(f"Error loading configuration from {config_path}: {e}")


def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate the configuration and return list of errors.
    Performs comprehensive validation including file system checks and marker validation.
    """
    errors = []
    
    # Validate required top-level sections
    required_sections = ['experiment', 'data', 'channels']
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required section: {section}")
    
    if errors:  # If basic structure is wrong, return early
        return errors
    
    # Validate experiment section
    errors.extend(_validate_experiment_config(config.get('experiment', {})))
    
    # Validate data section
    errors.extend(_validate_data_config(config.get('data', {})))
    
    # Validate channels section
    errors.extend(_validate_channels_config(config.get('channels', {}), config.get('data', {})))
    
    # Validate optional sections
    if 'compensation' in config:
        errors.extend(_validate_compensation_config(config['compensation']))
    
    if 'processing' in config:
        errors.extend(_validate_processing_config(config['processing']))
    
    if 'model' in config:
        errors.extend(_validate_model_config(config['model']))
    
    if 'hardware' in config:
        errors.extend(_validate_hardware_config(config['hardware']))
    
    if 'export' in config:
        errors.extend(_validate_export_config(config['export']))
    
    return errors


def _validate_experiment_config(experiment: Dict[str, Any]) -> List[str]:
    """Validate experiment configuration section."""
    errors = []
    
    if not experiment.get('name'):
        errors.append("experiment.name is required")
    
    if not experiment.get('output_dir'):
        errors.append("experiment.output_dir is required")
    else:
        output_dir = Path(experiment['output_dir'])
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create output directory {output_dir}: {e}")
    
    return errors


def _validate_data_config(data: Dict[str, Any]) -> List[str]:
    """Validate data configuration section."""
    errors = []
    
    # Check reference batch
    if not data.get('reference_batch'):
        errors.append("data.reference_batch is required")
    else:
        ref_path = Path(data['reference_batch'])
        if not ref_path.exists():
            errors.append(f"Reference batch directory does not exist: {ref_path}")
        elif not ref_path.is_dir():
            errors.append(f"Reference batch path is not a directory: {ref_path}")
        else:
            # Check for FCS files in reference batch
            fcs_files = list(ref_path.glob("*.fcs"))
            if not fcs_files:
                errors.append(f"No FCS files found in reference batch: {ref_path}")
            else:
                logger.info(f"Found {len(fcs_files)} FCS files in reference batch")
    
    # Check target batches
    if not data.get('target_batches'):
        errors.append("data.target_batches is required (must be a list)")
    elif not isinstance(data['target_batches'], list):
        errors.append("data.target_batches must be a list")
    else:
        for i, target_batch in enumerate(data['target_batches']):
            target_path = Path(target_batch)
            if not target_path.exists():
                errors.append(f"Target batch {i+1} directory does not exist: {target_path}")
            elif not target_path.is_dir():
                errors.append(f"Target batch {i+1} path is not a directory: {target_path}")
            else:
                fcs_files = list(target_path.glob("*.fcs"))
                if not fcs_files:
                    errors.append(f"No FCS files found in target batch {i+1}: {target_path}")
                else:
                    logger.info(f"Found {len(fcs_files)} FCS files in target batch: {target_path.name}")
    
    return errors


def _validate_channels_config(channels: Dict[str, Any], data: Dict[str, Any]) -> List[str]:
    """Validate channels configuration and marker availability."""
    errors = []
    
    if not channels.get('selected_markers'):
        errors.append("channels.selected_markers is required")
        return errors
    
    if not isinstance(channels['selected_markers'], list):
        errors.append("channels.selected_markers must be a list")
        return errors
    
    if len(channels['selected_markers']) == 0:
        errors.append("channels.selected_markers cannot be empty")
        return errors
    
    # Validate scatter channels if provided
    if 'scatter_channels' in channels:
        if not isinstance(channels['scatter_channels'], list):
            errors.append("channels.scatter_channels must be a list")
        elif len(channels['scatter_channels']) == 0:
            errors.append("channels.scatter_channels cannot be empty if specified")
    
    # Validate markers and scatter channels exist in reference batch
    if data.get('reference_batch'):
        ref_path = Path(data['reference_batch'])
        if ref_path.exists():
            try:
                errors.extend(_validate_channels_in_batch(channels, ref_path))
            except Exception as e:
                errors.append(f"Error validating channels in reference batch: {e}")
    
    return errors


def _validate_channels_in_batch(channels_config: Dict[str, Any], batch_path: Path) -> List[str]:
    """Validate that all selected markers and scatter channels exist in the batch FCS files."""
    errors = []
    
    # Find first FCS file in the batch
    fcs_files = list(batch_path.glob("*.fcs"))
    if not fcs_files:
        return ["No FCS files found for channel validation"]
    
    try:
        # Load the first FCS file to check available channels
        sample = fk.Sample(str(fcs_files[0]))
        available_pnn_labels = sample.pnn_labels  # Channel names (PnN)
        available_pns_labels = sample.pns_labels  # Antibody names (PnS)
        
        logger.info(f"Available PnN labels in reference batch: {len(available_pnn_labels)} total")
        logger.debug(f"Available PnN labels: {sorted(available_pnn_labels)}")
        logger.debug(f"Available PnS labels: {sorted(available_pns_labels)}")
        
        # Validate selected markers (these are matched against PnS labels)
        selected_markers = channels_config.get('selected_markers', [])
        missing_markers = []
        for marker in selected_markers:
            if marker not in available_pns_labels:
                missing_markers.append(marker)
        
        if missing_markers:
            errors.append(f"Selected markers not found in reference batch: {missing_markers}")
            errors.append(f"Available markers (PnS): {sorted([s for s in available_pns_labels if s])}")
            
            # Suggest similar markers
            for missing in missing_markers:
                suggestions = _find_similar_markers(missing, available_pns_labels)
                if suggestions:
                    errors.append(f"Did you mean one of these for '{missing}': {suggestions}")
        else:
            logger.info(f"✅ All {len(selected_markers)} selected markers found in reference batch")
        
        # Validate scatter channels (these are matched against PnN labels)
        scatter_channels = channels_config.get('scatter_channels', [])
        if scatter_channels:
            missing_scatter = []
            for scatter_ch in scatter_channels:
                if scatter_ch not in available_pnn_labels:
                    missing_scatter.append(scatter_ch)
            
            if missing_scatter:
                errors.append(f"Scatter channels not found in reference batch: {missing_scatter}")
                errors.append(f"Available scatter-like channels (PnN): {[ch for ch in available_pnn_labels if 'FSC' in ch or 'SSC' in ch]}")
                
                # Suggest similar scatter channels
                for missing in missing_scatter:
                    suggestions = _find_similar_markers(missing, available_pnn_labels)
                    if suggestions:
                        errors.append(f"Did you mean one of these for '{missing}': {suggestions}")
            else:
                logger.info(f"✅ All {len(scatter_channels)} scatter channels found in reference batch")
    
    except Exception as e:
        errors.append(f"Error reading FCS file {fcs_files[0]} for channel validation: {e}")
    
    return errors


def _find_similar_markers(target: str, available: List[str], max_suggestions: int = 3) -> List[str]:
    """Find similar marker names using simple string matching."""
    target_lower = target.lower()
    suggestions = []
    
    # Look for partial matches
    for marker in available:
        marker_lower = marker.lower()
        if target_lower in marker_lower or marker_lower in target_lower:
            suggestions.append(marker)
    
    return suggestions[:max_suggestions]


def _validate_compensation_config(compensation: Dict[str, Any]) -> List[str]:
    """Validate compensation configuration."""
    errors = []
    
    method = compensation.get('method')
    if method not in ['spill', 'none', None]:
        # Check if it's a file path
        comp_path = Path(method)
        if not comp_path.exists():
            errors.append(f"Compensation file does not exist: {comp_path}")
        elif not comp_path.is_file():
            errors.append(f"Compensation path is not a file: {comp_path}")
    
    return errors


def _validate_processing_config(processing: Dict[str, Any]) -> List[str]:
    """Validate processing configuration."""
    errors = []
    
    max_cells_per_file = processing.get('max_cells_per_file')
    if max_cells_per_file is not None:
        if not isinstance(max_cells_per_file, int) or max_cells_per_file <= 0:
            errors.append("processing.max_cells_per_file must be a positive integer")
    
    max_cells_for_training = processing.get('max_cells_for_training')
    if max_cells_for_training is not None:
        if not isinstance(max_cells_for_training, int) or max_cells_for_training <= 0:
            errors.append("processing.max_cells_for_training must be a positive integer")
    
    return errors


def _validate_model_config(model: Dict[str, Any]) -> List[str]:
    """Validate model configuration."""
    errors = []
    
    encoding_dim = model.get('encoding_dim')
    if encoding_dim is not None:
        if not isinstance(encoding_dim, int) or encoding_dim <= 0:
            errors.append("model.encoding_dim must be a positive integer")
    
    epochs = model.get('epochs')
    if epochs is not None:
        if not isinstance(epochs, int) or epochs <= 0:
            errors.append("model.epochs must be a positive integer")
    
    learning_rate = model.get('learning_rate')
    if learning_rate is not None:
        if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
            errors.append("model.learning_rate must be a positive number")
    
    beta = model.get('beta')
    if beta is not None:
        if not isinstance(beta, (int, float)) or beta < 0:
            errors.append("model.beta must be a non-negative number")
    
    batch_size = model.get('batch_size')
    if batch_size is not None:
        if not isinstance(batch_size, int) or batch_size <= 0:
            errors.append("model.batch_size must be a positive integer")
    
    inference_batch_size = model.get('inference_batch_size')
    if inference_batch_size is not None:
        if not isinstance(inference_batch_size, int) or inference_batch_size <= 0:
            errors.append("model.inference_batch_size must be a positive integer")
    
    return errors


def _validate_hardware_config(hardware: Dict[str, Any]) -> List[str]:
    """Validate hardware configuration."""
    errors = []
    
    device = hardware.get('device')
    if device is not None:
        if device not in ['auto', 'cpu', 'cuda']:
            errors.append("hardware.device must be 'auto', 'cpu', or 'cuda'")
    
    return errors


def _validate_export_config(export: Dict[str, Any]) -> List[str]:
    """Validate export configuration."""
    errors = []
    
    fcs_files = export.get('fcs_files')
    if fcs_files is not None:
        if not isinstance(fcs_files, bool):
            errors.append("export.fcs_files must be true or false")
    
    return errors


def resolve_marker_indices(selected_markers: List[str], reference_fcs_file: Path) -> List[int]:
    """
    Convert marker names to indices based on reference FCS file.
    
    Args:
        selected_markers: List of marker names from YAML
        reference_fcs_file: Path to sample FCS file from reference batch
        
    Returns:
        List of indices corresponding to the markers
        
    Raises:
        ValueError: If any marker is not found
    """
    try:
        sample = fk.Sample(str(reference_fcs_file))
        antibody_channels = sample.pns_labels  # Antibody names
        
        indices = []
        missing_markers = []
        
        for marker in selected_markers:
            try:
                idx = antibody_channels.index(marker)
                indices.append(idx)
            except ValueError:
                missing_markers.append(marker)
        
        if missing_markers:
            available_markers = sorted(antibody_channels)
            raise ValueError(
                f"Markers not found in reference batch: {missing_markers}\n"
                f"Available markers: {available_markers}"
            )
        
        logger.info(f"Resolved {len(selected_markers)} markers to indices: {indices}")
        return indices
        
    except Exception as e:
        raise ValueError(f"Error resolving marker indices from {reference_fcs_file}: {e}")


def resolve_scatter_indices(scatter_channels: List[str], reference_fcs_file: Path) -> List[int]:
    """
    Convert scatter channel names to indices based on reference FCS file.
    
    Args:
        scatter_channels: List of scatter channel names from YAML
        reference_fcs_file: Path to sample FCS file from reference batch
        
    Returns:
        List of indices corresponding to the scatter channels
        
    Raises:
        ValueError: If any scatter channel is not found
    """
    try:
        sample = fk.Sample(str(reference_fcs_file))
        channel_labels = sample.pnn_labels  # Channel names (PnN)
        
        indices = []
        missing_channels = []
        
        for channel in scatter_channels:
            try:
                idx = channel_labels.index(channel)
                indices.append(idx)
            except ValueError:
                missing_channels.append(channel)
        
        if missing_channels:
            available_channels = sorted(channel_labels)
            raise ValueError(
                f"Scatter channels not found in reference batch: {missing_channels}\n"
                f"Available channels: {available_channels}"
            )
        
        logger.info(f"Resolved {len(scatter_channels)} scatter channels to indices: {indices}")
        return indices
        
    except Exception as e:
        raise ValueError(f"Error resolving scatter indices from {reference_fcs_file}: {e}")