#!/usr/bin/env python3
"""
Voxel Analysis CLI Tool
Processes batch-aligned FCS files to generate voxel occupancy analysis.
"""

import argparse
import sys
import logging
import time
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any

from config import load_config, validate_config, resolve_marker_indices
from utils import setup_logging, format_time, get_device

logger = logging.getLogger(__name__)

def read_aligned_fcs_file(
    fcs_filepath: Path,
    selected_indices: List[int],
    max_cells: int = None
) -> np.ndarray:
    """
    Read marker data from an aligned FCS file.
    
    Args:
        fcs_filepath: Path to aligned FCS file
        selected_indices: Indices of marker channels to extract
        max_cells: Maximum number of cells to return
        
    Returns:
        numpy array of marker data
    """
    import flowkit as fk
    
    logger = logging.getLogger(__name__)
    logger.debug(f"Reading aligned FCS file: {fcs_filepath}")
    
    try:
        # Load the FCS file (already aligned, no compensation needed)
        sample = fk.Sample(str(fcs_filepath))
        initial_cell_count = sample._raw_events.shape[0]
        
        # Get all channel data (assuming it's already in the correct scale)
        available_channels = sample.pnn_labels
        df_all_channels = sample.as_dataframe(source='raw', subsample=False)[available_channels]
        
        # Convert to numpy array
        data_all_channels = np.array(df_all_channels)
        
        # Select only the specified marker channels
        if max(selected_indices) >= data_all_channels.shape[1]:
            raise ValueError(f"Selected index {max(selected_indices)} exceeds available channels ({data_all_channels.shape[1]})")
        
        data_selected_markers = data_all_channels[:, selected_indices]
        
        # Limit number of cells if requested
        if max_cells is not None and data_selected_markers.shape[0] > max_cells:
            data_selected_markers = data_selected_markers[:max_cells, :]
            logger.debug(f"Limited to first {max_cells} cells")
        
        logger.debug(f"Extracted marker data shape: {data_selected_markers.shape}")
        return data_selected_markers
        
    except Exception as e:
        logger.error(f"Error reading aligned FCS file {fcs_filepath}: {e}")
        # Return empty array with correct number of channels on error
        return np.zeros((0, len(selected_indices)))


def process_aligned_batch_directory(
    batch_dir: Path,
    selected_indices: List[int],
    selected_markers: List[str],
    marker_combinations: List[str],
    device: str,
    batch_size: int,
    max_cells_per_file: int = None
) -> Dict[str, Dict[str, float]]:
    """
    Process all aligned FCS files in a batch directory.
    
    Args:
        batch_dir: Path to directory containing aligned FCS files
        selected_indices: Indices of marker channels
        selected_markers: Names of selected markers
        marker_combinations: List of all marker combinations
        device: Device for GPU processing
        batch_size: Batch size for GPU processing
        max_cells_per_file: Maximum cells per file
        
    Returns:
        Dictionary mapping filename to voxel occupancy results
    """
    logger = logging.getLogger(__name__)
    
    # Find all FCS files in the directory
    fcs_files = sorted(list(batch_dir.glob("*.fcs")))
    if not fcs_files:
        raise ValueError(f"No FCS files found in directory: {batch_dir}")
    
    logger.info(f"Found {len(fcs_files)} FCS files in {batch_dir.name}")
    
    batch_results = {}
    
    for i, fcs_file in enumerate(fcs_files):
        logger.info(f"Processing file {i+1}/{len(fcs_files)}: {fcs_file.name}")
        
        try:
            # Read marker data from aligned FCS file
            marker_data = read_aligned_fcs_file(
                fcs_file,
                selected_indices,
                max_cells_per_file
            )
            
            if marker_data.shape[0] == 0:
                logger.warning(f"No cells found in {fcs_file.name}, skipping")
                continue
            
            # Calculate voxel occupancies
            voxel_occupancies = count_cells_in_voxels_gpu_batched(
                marker_data,
                selected_markers,
                marker_combinations,
                device=device,
                batch_size=batch_size
            )
            
            # Store results using filename without extension
            filename_key = fcs_file.stem
            batch_results[filename_key] = voxel_occupancies
            
            logger.debug(f"Calculated {len(voxel_occupancies)} voxel occupancies for {fcs_file.name}")
            
        except Exception as e:
            logger.error(f"Error processing {fcs_file.name}: {e}")
            raise  # Stop on error as requested
    
    return batch_results


def create_voxel_dataframe(all_batch_results: Dict[str, Dict[str, Dict[str, float]]]) -> pd.DataFrame:
    """
    Create a pandas DataFrame from voxel analysis results.
    
    Args:
        all_batch_results: Nested dict {batch_name: {filename: {combination: occupancy}}}
        
    Returns:
        pandas DataFrame with samples as rows, voxel combinations as columns
    """
    logger = logging.getLogger(__name__)
    
    # Collect all data into rows
    rows = []
    
    for batch_name, batch_results in all_batch_results.items():
        for filename, voxel_occupancies in batch_results.items():
            row = {'filename': filename, 'batch': batch_name}
            row.update(voxel_occupancies)
            rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Reorder columns: filename, batch, then all voxel combinations
    voxel_columns = [col for col in df.columns if col not in ['filename', 'batch']]
    ordered_columns = ['filename', 'batch'] + sorted(voxel_columns)
    df = df[ordered_columns]
    
    logger.info(f"Created DataFrame with {len(df)} samples and {len(voxel_columns)} voxel combinations")
    
    return df

def generate_all_marker_combinations(antibody_channels):
    """
    Generate all unique 2 or 3 marker combinations with LOW, MED, HIGH intervals.
    Each marker appears only once in a combination.
    
    Args:
        antibody_channels (list): List of marker names
        
    Returns:
        list: All unique marker combinations formatted as 'M1:I1~M2:I2[~M3:I3]'
    """
    intervals = ['LOW', 'MED', 'HIGH']
    unique_combinations = set()
    
    # Generate combinations for 2 and 3 markers
    for n in (2, 3):
        # Get all possible n-marker combinations
        marker_combos = itertools.combinations(antibody_channels, n)
        
        for markers in marker_combos:
            # Get all possible interval combinations for these markers
            interval_combos = itertools.product(intervals, repeat=n)
            
            for intervals_combo in interval_combos:
                # Create the combination string
                combo_parts = [f"{marker}:{interval}" 
                             for marker, interval in zip(markers, intervals_combo)]
                combo = '~'.join(combo_parts)
                unique_combinations.add(combo)
    
    # Convert set to sorted list for consistent output
    all_combinations = sorted(list(unique_combinations))
    print(f'Generated {len(all_combinations)} unique combinations')
    return all_combinations


def count_cells_in_voxels_gpu_batched(data_array, antibody_channels, marker_combinations, device='cpu', batch_size=1000):
    """
    Count cells in each voxel combination using PyTorch with batched processing.
    
    Args:
        data_array: numpy array of shape (n_cells, n_channels)
        antibody_channels: list of channel names
        marker_combinations: list of marker combinations (e.g. "CD4:HIGH~CD8:LOW")
        device: 'cuda' for GPU or 'cpu' for CPU processing
        batch_size: number of combinations to process in each batch
        
    Returns:
        dict: Mapping of combinations to their cell proportions
    """
    import torch

    torch.set_num_threads(8)

    # Create channel name to index mapping
    channel_to_idx = {channel: idx for idx, channel in enumerate(antibody_channels)}
    n_cells = data_array.shape[0]
    counts = {}
    
    # Convert data to tensor and move to device
    data_tensor = torch.tensor(data_array, device=device, dtype=torch.float32)
    
    # Pre-compute masks for each level once (on device)
    # Using voxel intervals: LOW (<0.3), MED (0.3-0.6), HIGH (≥0.6)
    low_mask = data_tensor < 0.3
    med_mask = (data_tensor >= 0.3) & (data_tensor < 0.6)
    high_mask = data_tensor >= 0.6
    
    # Group combinations by the number of channels they use (for batch processing)
    combinations_by_length = {}
    for combo in marker_combinations:
        length = len(combo.split('~'))
        if length not in combinations_by_length:
            combinations_by_length[length] = []
        combinations_by_length[length].append(combo)
    
    # Process each group separately
    for length, combos in combinations_by_length.items():
        # Process combinations in batches
        for i in range(0, len(combos), batch_size):
            batch_combos = combos[i:i+batch_size]
            batch_masks = torch.ones((len(batch_combos), n_cells), dtype=torch.bool, device=device)
            
            combo_map = {}  # Map batch index to combo string
            for batch_idx, combo in enumerate(batch_combos):
                combo_map[batch_idx] = combo
                markers = combo.split('~')
                
                for marker in markers:
                    channel, level = marker.split(':')
                    channel_idx = channel_to_idx[channel]
                    
                    if level == 'LOW':
                        batch_masks[batch_idx] &= low_mask[:, channel_idx]
                    elif level == 'MED':
                        batch_masks[batch_idx] &= med_mask[:, channel_idx]
                    else:  # HIGH
                        batch_masks[batch_idx] &= high_mask[:, channel_idx]
            
            # Calculate proportions for all combinations in this batch at once
            cell_counts = torch.sum(batch_masks, dim=1).float() / n_cells
            
            # Store results
            for batch_idx, count in enumerate(cell_counts.cpu().numpy()):
                counts[combo_map[batch_idx]] = count
    
    return counts


class VoxelAnalysisProcessor:
    """Processor for voxel analysis of batch-aligned FCS files."""
    
    def __init__(self, config: Dict[str, Any], aligned_data_dir: Path):
        self.config = config
        self.aligned_data_dir = aligned_data_dir
        self.output_dir = Path(config['experiment']['output_dir'])
        
        # Get processing parameters from config
        self.selected_markers = config['channels']['selected_markers']
        self.max_cells_per_file = config.get('processing', {}).get('max_cells_per_file', 300000)
        
        # Get device settings
        device_config = config.get('hardware', {}).get('device', 'auto')
        self.device = get_device(device_config)
        
        # GPU batch processing settings
        self.batch_size = config.get('model', {}).get('inference_batch_size', 1000)
        
        # Initialize variables
        self.selected_indices = None
        self.marker_combinations = None
    
    def run(self) -> bool:
        """Run the complete voxel analysis process."""
        try:
            start_time = time.time()
            
            # Step 1: Resolve marker indices using first available FCS file
            logger.info("🔍 Step 1: Resolving marker indices...")
            self._resolve_marker_indices()
            
            # Step 2: Generate marker combinations
            logger.info("🧬 Step 2: Generating marker combinations...")
            self.marker_combinations = generate_all_marker_combinations(self.selected_markers)
            logger.info(f"Generated {len(self.marker_combinations)} marker combinations")
            
            # Step 3: Process all batch directories
            logger.info("⚡ Step 3: Processing aligned batches...")
            all_batch_results = self._process_all_aligned_batches()
            
            # Step 4: Create and save DataFrame
            logger.info("📊 Step 4: Creating output DataFrame...")
            df = create_voxel_dataframe(all_batch_results)
            self._save_results(df, time.time() - start_time)
            
            logger.info("✅ Voxel analysis completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Voxel analysis failed: {e}", exc_info=True)
            return False
    
    def _resolve_marker_indices(self) -> None:
        """Resolve marker names to indices using first available aligned FCS file."""
        # Find first FCS file in any batch directory
        first_fcs_file = None
        
        for batch_dir in self.aligned_data_dir.iterdir():
            if batch_dir.is_dir():
                fcs_files = list(batch_dir.glob("*.fcs"))
                if fcs_files:
                    first_fcs_file = fcs_files[0]
                    break
        
        if not first_fcs_file:
            raise ValueError(f"No FCS files found in aligned data directory: {self.aligned_data_dir}")
        
        self.selected_indices = resolve_marker_indices(self.selected_markers, first_fcs_file)
        logger.info(f"Resolved {len(self.selected_markers)} markers to indices: {self.selected_indices}")
    
    def _process_all_aligned_batches(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Process all batch directories in the aligned data folder."""
        all_batch_results = {}
        
        # Find all batch directories
        batch_dirs = [d for d in self.aligned_data_dir.iterdir() if d.is_dir()]
        
        if not batch_dirs:
            raise ValueError(f"No batch directories found in: {self.aligned_data_dir}")
        
        logger.info(f"Found {len(batch_dirs)} batch directories to process")
        
        for batch_dir in sorted(batch_dirs):
            logger.info(f"Processing batch: {batch_dir.name}")
            
            try:
                batch_results = process_aligned_batch_directory(
                    batch_dir,
                    self.selected_indices,
                    self.selected_markers,
                    self.marker_combinations,
                    self.device,
                    self.batch_size,
                    self.max_cells_per_file
                )
                
                all_batch_results[batch_dir.name] = batch_results
                logger.info(f"Completed batch {batch_dir.name}: {len(batch_results)} files processed")
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_dir.name}: {e}")
                raise  # Stop on error as requested
        
        return all_batch_results






    
    def _save_results(self, df: pd.DataFrame, total_time: float) -> None:
        """Save the voxel analysis results."""
        # Save CSV
        csv_path = self.output_dir / 'voxel_analysis_results.csv'
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved CSV results to: {csv_path}")
        
        # Save pickle
        pkl_path = self.output_dir / 'voxel_analysis_results.pkl'
        df.to_pickle(pkl_path)
        logger.info(f"Saved pickle results to: {pkl_path}")
        
        # Log summary
        self._log_summary(df, total_time)
    
    def _log_summary(self, df: pd.DataFrame, total_time: float) -> None:
        """Log summary statistics."""
        logger.info("=" * 60)
        logger.info("VOXEL ANALYSIS SUMMARY")
        logger.info("=" * 60)
        
        logger.info(f"Total runtime: {format_time(total_time)}")
        logger.info(f"Device used: {self.device}")
        logger.info("")
        
        logger.info("Dataset Summary:")
        logger.info(f"  • Total samples: {len(df)}")
        logger.info(f"  • Markers analyzed: {len(self.selected_markers)}")
        logger.info(f"  • Voxel combinations: {len(self.marker_combinations)}")
        
        if 'batch' in df.columns:
            batch_counts = df['batch'].value_counts()
            logger.info(f"  • Batches processed: {len(batch_counts)}")
            for batch_name, count in batch_counts.items():
                logger.info(f"    - {batch_name}: {count} samples")
        
        logger.info("")
        logger.info("Output Files:")
        logger.info(f"  • CSV: {self.output_dir}/voxel_analysis_results.csv")
        logger.info(f"  • Pickle: {self.output_dir}/voxel_analysis_results.pkl")
        
        logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Voxel Analysis CLI Tool - Generate voxel occupancy analysis from batch-aligned FCS files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python voxel_analysis.py --config config.yaml --aligned-data ./results/aligned_data
  python voxel_analysis.py --config config.yaml --aligned-data ./results/aligned_data --verbose
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        required=True,
        type=Path,
        help='Path to YAML configuration file (same as used for batch alignment)'
    )
    
    parser.add_argument(
        '--aligned-data',
        required=True,
        type=Path,
        help='Path to aligned_data directory from batch alignment output'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    try:
        # Validate inputs
        if not args.config.exists():
            print(f"❌ Configuration file not found: {args.config}")
            sys.exit(1)
        
        if not args.aligned_data.exists():
            print(f"❌ Aligned data directory not found: {args.aligned_data}")
            sys.exit(1)
        
        if not args.aligned_data.is_dir():
            print(f"❌ Aligned data path is not a directory: {args.aligned_data}")
            sys.exit(1)
        
        # Load configuration
        config = load_config(args.config)
        
        # Setup logging
        log_level = "DEBUG" if args.verbose else config.get('logging', {}).get('level', 'INFO')
        setup_logging(log_level)
        
        logger = logging.getLogger(__name__)
        logger.info("🔬 Starting Voxel Analysis CLI")
        logger.info(f"📁 Configuration file: {args.config}")
        logger.info(f"📂 Aligned data directory: {args.aligned_data}")
        logger.info(f"🧪 Experiment: {config['experiment']['name']}")
        
        # Validate that required config sections exist
        required_keys = ['channels.selected_markers', 'experiment.output_dir']
        for key in required_keys:
            keys = key.split('.')
            current = config
            for k in keys:
                if k not in current:
                    logger.error(f"❌ Required configuration key missing: {key}")
                    sys.exit(1)
                current = current[k]
        
        # Check that aligned data directory contains batch subdirectories
        batch_dirs = [d for d in args.aligned_data.iterdir() if d.is_dir()]
        if not batch_dirs:
            logger.error(f"❌ No batch directories found in: {args.aligned_data}")
            logger.error("Expected structure: aligned_data/batch_name/*.fcs")
            sys.exit(1)
        
        # Verify at least one batch has FCS files
        total_fcs_count = 0
        for batch_dir in batch_dirs:
            fcs_count = len(list(batch_dir.glob("*.fcs")))
            total_fcs_count += fcs_count
            logger.debug(f"Batch {batch_dir.name}: {fcs_count} FCS files")
        
        if total_fcs_count == 0:
            logger.error(f"❌ No FCS files found in any batch directory")
            sys.exit(1)
        
        logger.info(f"📊 Found {total_fcs_count} total FCS files across {len(batch_dirs)} batches")
        
        # Initialize processor
        processor = VoxelAnalysisProcessor(config, args.aligned_data)
        
        # Run the voxel analysis
        logger.info("🎯 Starting voxel analysis process...")
        success = processor.run()
        
        if success:
            logger.info("🎉 Voxel analysis completed successfully!")
            logger.info(f"📊 Results saved to: {config['experiment']['output_dir']}")
        else:
            logger.error("❌ Voxel analysis failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.getLogger(__name__).error(f"❌ Unexpected error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()