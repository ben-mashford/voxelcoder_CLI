"""
Core batch alignment processor.
Handles the complete workflow from loading data to training models and processing batches.
"""

import json
import time
import math
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

from config import resolve_marker_indices
from fcs_processing import build_reference_dataset, get_fcs_file_info
from models import create_autoencoder, histogram_loss, count_model_parameters
from utils import ProgressTracker, ensure_directory, get_device, format_time, summarize_batch_info

logger = logging.getLogger(__name__)


class BatchAlignmentProcessor:
    """Main processor for batch alignment workflow."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.experiment_name = config['experiment']['name']
        self.output_dir = Path(config['experiment']['output_dir'])
        
        # Setup output directories
        self.models_dir = self.output_dir / 'models'
        self.aligned_data_dir = self.output_dir / 'aligned_data'
        self.logs_dir = self.output_dir / 'logs'
        
        ensure_directory(self.models_dir)
        ensure_directory(self.aligned_data_dir)
        ensure_directory(self.logs_dir)
        
        # Get paths
        self.reference_batch_path = Path(config['data']['reference_batch'])
        self.target_batch_paths = [Path(p) for p in config['data']['target_batches']]
        
        # Get processing parameters
        self.selected_markers = config['channels']['selected_markers']
        self.scatter_channels = config['channels'].get('scatter_channels', [])
        self.compensation_method = config.get('compensation', {}).get('method', 'spill')
        self.max_cells_per_file = config.get('processing', {}).get('max_cells_per_file', 300000)
        self.max_cells_for_training = config.get('processing', {}).get('max_cells_for_training', 40000)
        
        # Get model parameters
        model_config = config.get('model', {})
        self.encoding_dim = model_config.get('encoding_dim', 16)
        self.epochs = model_config.get('epochs', 600)
        self.learning_rate = model_config.get('learning_rate', 0.003)
        self.beta = model_config.get('beta', 0.002)
        self.batch_size = model_config.get('batch_size', 1024)
        self.inference_batch_size = model_config.get('inference_batch_size', 8192)
        self.model_type = model_config.get('type', 'standard')
        
        # Get device
        device_config = config.get('hardware', {}).get('device', 'auto')
        self.device = get_device(device_config)
        
        # Initialize variables
        self.selected_indices = None
        self.scatter_indices = None
        self.model = None
        self.training_stats = {}
        self.processing_stats = {}
        
    def show_processing_plan(self) -> None:
        """Show what would be processed in dry run mode."""
        logger.info("=" * 60)
        logger.info("PROCESSING PLAN")
        logger.info("=" * 60)
        
        logger.info(f"Experiment: {self.experiment_name}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Device: {self.device}")
        logger.info("")
        
        # Reference batch info
        ref_info = summarize_batch_info(self.reference_batch_path)
        logger.info(f"Reference batch: {ref_info['name']}")
        logger.info(f"  • Path: {ref_info['path']}")
        logger.info(f"  • FCS files: {ref_info['fcs_count']}")
        logger.info(f"  • Total size: {ref_info['total_size_formatted']}")
        logger.info("")
        
        # Target batches info
        logger.info("Target batches:")
        for i, target_path in enumerate(self.target_batch_paths, 1):
            target_info = summarize_batch_info(target_path)
            logger.info(f"  {i}. {target_info['name']}")
            logger.info(f"     • Path: {target_info['path']}")
            logger.info(f"     • FCS files: {target_info['fcs_count']}")
            logger.info(f"     • Total size: {target_info['total_size_formatted']}")
        logger.info("")
        
        # Processing parameters
        logger.info("Processing parameters:")
        logger.info(f"  • Selected markers: {len(self.selected_markers)}")
        logger.info(f"  • Compensation: {self.compensation_method}")
        logger.info(f"  • Max cells per file: {self.max_cells_per_file:,}")
        logger.info(f"  • Max cells for training: {self.max_cells_for_training:,}")
        logger.info("")
        
        # Model parameters
        logger.info("Model parameters:")
        logger.info(f"  • Model type: {self.model_type}")
        logger.info(f"  • Encoding dimension: {self.encoding_dim}")
        logger.info(f"  • Epochs: {self.epochs}")
        logger.info(f"  • Learning rate: {self.learning_rate}")
        logger.info(f"  • Beta: {self.beta}")
        logger.info(f"  • Batch size: {self.batch_size}")
        logger.info(f"  • Inference batch size: {self.inference_batch_size}")
        
        logger.info("=" * 60)
    
    def run(self) -> bool:
        """Run the complete batch alignment process."""
        try:
            start_time = time.time()
            
            # Step 1: Resolve marker indices
            logger.info("🔍 Step 1: Resolving marker indices...")
            self._resolve_marker_indices()
            
            # Step 2: Build reference dataset
            logger.info("📊 Step 2: Building reference dataset...")
            reference_data = self._build_reference_dataset()
            
            # Step 3: Train autoencoder
            logger.info("🧠 Step 3: Training autoencoder...")
            self._train_autoencoder(reference_data)
            
            # Step 4: Process all batches
            logger.info("⚡ Step 4: Processing batches...")
            self._process_all_batches()
            
            # Step 5: Generate summary report
            logger.info("📋 Step 5: Generating summary report...")
            self._generate_summary_report(time.time() - start_time)
            
            logger.info("✅ Batch alignment completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Batch alignment failed: {e}", exc_info=True)
            return False
    
    def _resolve_marker_indices(self) -> None:
        """Resolve marker names and scatter channels to indices using reference batch."""
        from config import resolve_marker_indices, resolve_scatter_indices
        
        # Find first FCS file in reference batch
        fcs_files = list(self.reference_batch_path.glob("*.fcs"))
        if not fcs_files:
            raise ValueError(f"No FCS files found in reference batch: {self.reference_batch_path}")
        
        self.selected_indices = resolve_marker_indices(self.selected_markers, fcs_files[0])
        logger.info(f"Resolved {len(self.selected_markers)} markers to indices: {self.selected_indices}")
        
        if self.scatter_channels:
            self.scatter_indices = resolve_scatter_indices(self.scatter_channels, fcs_files[0])
            logger.info(f"Resolved {len(self.scatter_channels)} scatter channels to indices: {self.scatter_indices}")
        else:
            self.scatter_indices = []
            logger.info("No scatter channels specified")
    
    def _build_reference_dataset(self) -> np.ndarray:
        """Build the reference dataset for training."""
        reference_data = build_reference_dataset(
            self.reference_batch_path,
            self.selected_indices,
            self.compensation_method,
            self.max_cells_per_file,
            self.max_cells_for_training
        )
        
        # Save reference data
        ref_data_path = self.aligned_data_dir / 'reference_data.npy'
        np.save(ref_data_path, reference_data)
        logger.info(f"Saved reference data to: {ref_data_path}")
        
        return reference_data
    
    def _train_autoencoder(self, reference_data: np.ndarray) -> None:
        """Train the autoencoder model."""
        import torch
        
        input_dim = reference_data.shape[1]
        
        # Create model
        self.model = create_autoencoder(self.model_type, input_dim, self.encoding_dim)
        self.model.to(self.device)
        
        param_count = count_model_parameters(self.model)
        logger.info(f"Model has {param_count:,} trainable parameters")
        
        # Setup training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        mse_criterion = torch.nn.MSELoss()
        
        # Prepare data
        data_tensor = torch.tensor(reference_data, dtype=torch.float32).to(self.device)
        dataset = torch.utils.data.TensorDataset(data_tensor, data_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training loop
        self.model.train()
        training_start = time.time()
        progress = ProgressTracker(self.epochs, "Training autoencoder")
        
        epoch_losses = []
        
        for epoch in range(self.epochs):
            beta_scaler = 1.0 - (epoch / self.epochs)
            epoch_loss = 0.0
            batch_count = 0
            
            for batch_X, batch_y in dataloader:
                outputs = self.model(batch_X)
                mse_loss = mse_criterion(outputs, batch_y)
                hist_loss = histogram_loss(outputs, batch_y, num_bins=20)
                loss = mse_loss + self.beta * beta_scaler * hist_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
            
            avg_epoch_loss = epoch_loss / batch_count
            epoch_losses.append(avg_epoch_loss)
            
            # Update progress
            if epoch % 10 == 0 or epoch == self.epochs - 1:
                progress.update(10 if epoch < self.epochs - 1 else (self.epochs - epoch), 
                              f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_epoch_loss:.6f}")
        
        progress.finish()
        
        training_time = time.time() - training_start
        final_loss = epoch_losses[-1] if epoch_losses else 0.0
        
        # Save training statistics
        self.training_stats = {
            'epochs': self.epochs,
            'final_loss': final_loss,
            'training_time_seconds': training_time,
            'parameter_count': param_count,
            'device': self.device
        }
        
        # Save model
        model_path = self.models_dir / 'autoencoder.pt'
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"Model training completed in {format_time(training_time)}")
        logger.info(f"Final loss: {final_loss:.6f}")
        logger.info(f"Model saved to: {model_path}")
    
    def _process_all_batches(self) -> None:
        """Process all batches (reference + targets) using the trained model."""
        import torch
        import math
        
        self.model.eval()
        
        all_batch_paths = [self.reference_batch_path] + self.target_batch_paths
        all_batch_names = [self.reference_batch_path.name] + [p.name for p in self.target_batch_paths]
        batch_roles = ['reference'] + ['target'] * len(self.target_batch_paths)
        
        total_batches = len(all_batch_paths)
        
        # Check if FCS export is enabled
        export_fcs = self.config.get('export', {}).get('fcs_files', False)
        
        for batch_idx, (batch_path, batch_name, role) in enumerate(
            zip(all_batch_paths, all_batch_names, batch_roles)
        ):
            logger.info(f"Processing batch {batch_idx+1}/{total_batches}: {batch_name} ({role})")
            batch_start_time = time.time()
            
            # Create output directory for this batch
            batch_output_dir = self.aligned_data_dir / batch_name
            ensure_directory(batch_output_dir)
            
            # Process all files in the batch
            if self.scatter_channels:
                # Use the new function that handles both marker and scatter data
                from fcs_processing import process_batch_files_with_scatter
                processed_files_data = process_batch_files_with_scatter(
                    batch_path,
                    self.selected_indices,
                    self.scatter_indices,
                    self.compensation_method,
                    self.max_cells_per_file
                )
                # Extract just the marker data for autoencoder processing
                processed_files = {filename: data['markers'] for filename, data in processed_files_data.items()}
            else:
                # Use the original function for marker data only
                processed_files = process_batch_files(
                    batch_path,
                    self.selected_indices,
                    self.compensation_method,
                    self.max_cells_per_file
                )
            
            batch_stats = {
                'role': role,
                'files_processed': len(processed_files),
                'total_cells': 0,
                'processing_time_seconds': 0,
                'files': {}
            }
            
            # Process each file through the autoencoder
            file_progress = ProgressTracker(len(processed_files), f"Processing {batch_name} files")
            
            for file_idx, (filename, file_data) in enumerate(processed_files.items()):
                if file_data.shape[0] == 0:
                    logger.warning(f"Skipping empty file: {filename}")
                    continue
                
                # Process in chunks to manage memory
                cell_batch_count = math.ceil(file_data.shape[0] / self.inference_batch_size)
                processed_cells = []
                start_idx = 0
                
                for chunk_idx in range(cell_batch_count):
                    end_idx = min(start_idx + self.inference_batch_size, file_data.shape[0])
                    data_chunk = file_data[start_idx:end_idx, :]
                    
                    # Convert to tensor and process
                    x = torch.tensor(data_chunk).float().to(self.device)
                    with torch.no_grad():
                        pred_ae = self.model(x)
                    result = pred_ae.cpu().detach().numpy()
                    processed_cells.append(result)
                    start_idx = end_idx
                
                # Combine processed chunks
                file_data_corrected = np.vstack(processed_cells)
                
                # Save before and after data
                # before_filename = f"{filename}_before.npy"
                # after_filename = f"{filename}_after.npy"
                # 
                # before_path = batch_output_dir / before_filename
                # after_path = batch_output_dir / after_filename
                # 
                # np.save(before_path, file_data)
                # np.save(after_path, file_data_corrected)
                
                # Export corrected data as FCS if enabled
                if export_fcs:
                    try:
                        self._export_corrected_fcs(
                            batch_path / filename,
                            file_data_corrected,
                            batch_output_dir / f"{filename}"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to export corrected FCS for {filename}: {e}")
                
                # Update statistics
                cell_count = file_data.shape[0]
                batch_stats['total_cells'] += cell_count
                batch_stats['files'][filename] = {
                    'cell_count': cell_count,
                    # 'before_path': str(before_path),
                    # 'after_path': str(after_path)
                }
                
                if export_fcs:
                    batch_stats['files'][filename]['corrected_fcs_path'] = str(batch_output_dir / f"{filename}_corrected.fcs")
                
                file_progress.update(1, f"{filename}: {cell_count:,} cells")
            
            file_progress.finish()
            
            batch_processing_time = time.time() - batch_start_time
            batch_stats['processing_time_seconds'] = batch_processing_time
            
            # Store batch statistics
            self.processing_stats[batch_name] = batch_stats
            
            logger.info(f"Completed batch {batch_name}: {batch_stats['files_processed']} files, "
                       f"{batch_stats['total_cells']:,} cells in {format_time(batch_processing_time)}")
    
    def _generate_summary_report(self, total_runtime: float) -> None:
        """Generate a comprehensive summary report."""
        
        # Calculate overall statistics
        total_files = sum(stats['files_processed'] for stats in self.processing_stats.values())
        total_cells = sum(stats['total_cells'] for stats in self.processing_stats.values())
        
        # Create summary report
        summary = {
            'experiment_name': self.experiment_name,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'config': self.config,
            'training': self.training_stats,
            'processing': {
                'total_files_processed': total_files,
                'total_cells_processed': total_cells,
                'total_runtime_seconds': total_runtime,
                'device_used': self.device
            },
            'batches': self.processing_stats,
            'output_structure': {
                'models_dir': str(self.models_dir),
                'aligned_data_dir': str(self.aligned_data_dir),
                'logs_dir': str(self.logs_dir)
            }
        }
        
        # Save summary report
        summary_path = self.output_dir / 'summary_report.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Summary report saved to: {summary_path}")
        
        # Log summary to console
        self._log_summary_to_console(summary)
    
    def _log_summary_to_console(self, summary: Dict[str, Any]) -> None:
        """Log a human-readable summary to the console."""
        logger.info("=" * 60)
        logger.info("BATCH ALIGNMENT SUMMARY")
        logger.info("=" * 60)
        
        # Basic info
        logger.info(f"Experiment: {summary['experiment_name']}")
        logger.info(f"Completed: {summary['timestamp']}")
        logger.info(f"Total runtime: {format_time(summary['processing']['total_runtime_seconds'])}")
        logger.info(f"Device used: {summary['processing']['device_used']}")
        logger.info("")
        
        # Training info
        training = summary['training']
        logger.info("Training Results:")
        logger.info(f"  • Model parameters: {training.get('parameter_count', 'N/A'):,}")
        logger.info(f"  • Training epochs: {training.get('epochs', 'N/A')}")
        logger.info(f"  • Final loss: {training.get('final_loss', 'N/A'):.6f}")
        logger.info(f"  • Training time: {format_time(training.get('training_time_seconds', 0))}")
        logger.info("")
        
        # Processing info
        processing = summary['processing']
        logger.info("Processing Results:")
        logger.info(f"  • Total files processed: {processing['total_files_processed']:,}")
        logger.info(f"  • Total cells processed: {processing['total_cells_processed']:,}")
        logger.info(f"  • Processing time: {format_time(processing['total_runtime_seconds'] - training.get('training_time_seconds', 0))}")
        logger.info("")
        
        # Batch details
        logger.info("Batch Details:")
        for batch_name, stats in summary['batches'].items():
            avg_cells = stats['total_cells'] / stats['files_processed'] if stats['files_processed'] > 0 else 0
            logger.info(f"  • {batch_name} ({stats['role']}):")
            logger.info(f"    - Files: {stats['files_processed']:,}")
            logger.info(f"    - Cells: {stats['total_cells']:,}")
            logger.info(f"    - Avg cells/file: {avg_cells:,.0f}")
            logger.info(f"    - Processing time: {format_time(stats['processing_time_seconds'])}")
        logger.info("")
        
        # Output locations
        logger.info("Output Files:")
        logger.info(f"  • Models: {summary['output_structure']['models_dir']}")
        logger.info(f"  • Aligned data: {summary['output_structure']['aligned_data_dir']}")
        logger.info(f"  • Logs: {summary['output_structure']['logs_dir']}")
        logger.info(f"  • Summary: {self.output_dir}/summary_report.json")
        
        logger.info("=" * 60)


    def _export_corrected_fcs(self, original_fcs_path: Path, corrected_data: np.ndarray, output_path: Path) -> None:
        """
        Export corrected data as FCS file, combining scatter channels (untransformed) with corrected markers.
        
        Args:
            original_fcs_path: Path to original FCS file (for metadata and scatter data)
            corrected_data: Corrected marker data in logicle scale
            output_path: Path for output FCS file
        """
        import flowkit as fk
        import flowio
        
        try:
            # Load original sample to get metadata and raw scatter data
            original_sample = fk.Sample(str(original_fcs_path))
            
            # Get scatter data directly from raw events (no transformations)
            raw_events = original_sample._raw_events
            
            # Extract scatter channels if specified
            if self.scatter_indices:
                scatter_data = raw_events[:, self.scatter_indices]
                scatter_pnn_labels = [original_sample.pnn_labels[i] for i in self.scatter_indices]
                scatter_pns_labels = [original_sample.pns_labels[i] for i in self.scatter_indices]
            else:
                scatter_data = np.empty((raw_events.shape[0], 0))
                scatter_pnn_labels = []
                scatter_pns_labels = []
            
            # Get marker channel info
            marker_pnn_labels = [original_sample.pnn_labels[i] for i in self.selected_indices]
            marker_pns_labels = [original_sample.pns_labels[i] for i in self.selected_indices]
            
            # Transform corrected marker data back from logicle to linear scale
            linear_corrected_markers = self._logicle_to_linear(corrected_data, original_sample)
            
            # Combine data in the order specified in YAML: scatter channels first, then markers
            all_channel_labels_pnn = scatter_pnn_labels + marker_pnn_labels
            all_channel_labels_pns = scatter_pns_labels + marker_pns_labels
            
            # Ensure we have the same number of events for both scatter and marker data
            min_events = min(scatter_data.shape[0], linear_corrected_markers.shape[0])
            if scatter_data.shape[0] > 0:
                scatter_data = scatter_data[:min_events, :]
            linear_corrected_markers = linear_corrected_markers[:min_events, :]
            
            # Combine the data
            if scatter_data.shape[1] > 0:
                combined_data = np.hstack([scatter_data, linear_corrected_markers])
            else:
                combined_data = linear_corrected_markers
            
            # Create metadata for export
            metadata_dict = self._create_export_metadata(
                original_sample, 
                len(all_channel_labels_pnn),
                all_channel_labels_pnn,
                all_channel_labels_pns
            )
            
            # Export as FCS
            with open(output_path, 'wb') as fh:
                flowio.create_fcs(
                    fh,
                    combined_data.flatten().tolist(),
                    channel_names=all_channel_labels_pnn,
                    opt_channel_names=all_channel_labels_pns,
                    metadata_dict=metadata_dict
                )
            
            logger.debug(f"Exported corrected FCS with {len(all_channel_labels_pnn)} channels: {output_path}")
            
        except Exception as e:
            logger.error(f"Error exporting corrected FCS for {original_fcs_path.name}: {e}")
            raise
    
    def _logicle_to_linear(self, logicle_data: np.ndarray, original_sample) -> np.ndarray:
        """
        Transform marker data from logicle scale back to linear scale.
        
        Args:
            logicle_data: Marker data in logicle scale
            original_sample: Original FlowKit Sample for metadata
            
        Returns:
            Marker data transformed back to linear scale
        """
        import flowkit as fk
        
        # Determine the appropriate max intensity based on original data
        max_value = original_sample._raw_events.max()
        current_max_intensity = 262144
        
        if max_value > 262144:
            current_max_intensity = 4194304.0
        
        # Create inverse logicle transform
        inverse_logicle = fk.transforms.LogicleTransform(
            param_t=current_max_intensity,
            param_w=0.5,
            param_m=4.5,
            param_a=0
        )
        
        # Apply inverse transform to convert back to linear scale
        linear_data = inverse_logicle.inverse(logicle_data)
        
        return linear_data
    
    def _create_export_metadata(
        self, 
        original_sample, 
        total_channel_count: int,
        all_pnn_labels: List[str],
        all_pns_labels: List[str]
    ) -> dict:
        """
        Create metadata dictionary for FCS export with both scatter and marker channels.
        
        Args:
            original_sample: Original FlowKit Sample
            total_channel_count: Total number of channels (scatter + markers)
            all_pnn_labels: Combined list of PnN labels (scatter + markers)
            all_pns_labels: Combined list of PnS labels (scatter + markers)
            
        Returns:
            Metadata dictionary for FCS export
        """
        metadata = {
            'datatype': 'F',  # Float data type
            'mode': 'L',      # List mode
            'tot': str(len(original_sample._raw_events)),  # Total events
            'par': str(total_channel_count),  # Number of parameters
        }
        
        # Add channel-specific metadata
        for i, (pnn_label, pns_label) in enumerate(zip(all_pnn_labels, all_pns_labels), 1):
            metadata[f'p{i}n'] = pnn_label
            metadata[f'p{i}s'] = pns_label
            
            # Different metadata for scatter vs marker channels
            if pnn_label in self.scatter_channels:
                # Scatter channels: use original metadata if available
                original_idx = original_sample.pnn_labels.index(pnn_label)
                original_range = original_sample.channels.iloc[original_idx]['pnr']
                metadata[f'p{i}r'] = str(int(original_range))
                metadata[f'p{i}g'] = '1.0'  # Gain already applied
                metadata[f'p{i}e'] = '0,0'  # Linear scale
            else:
                # Marker channels: standard corrected data metadata
                metadata[f'p{i}r'] = '262144'  # Range
                metadata[f'p{i}g'] = '1.0'     # Gain already applied
                metadata[f'p{i}e'] = '0,0'     # Linear scale
        
        # Add some original metadata if available
        preserve_keys = ['fil', 'date', 'cyt', 'cytsn', 'op', 'sys']
        for key in preserve_keys:
            if key in original_sample.metadata:
                metadata[key] = original_sample.metadata[key]
        
        # Add processing note
        metadata['note'] = 'Batch aligned using autoencoder - scatter channels preserved, markers corrected'
        
        return metadata
    
    def _create_export_metadata(
        self, 
        original_sample, 
        total_channel_count: int,
        all_pnn_labels: List[str],
        all_pns_labels: List[str]
    ) -> dict:
        """
        Create metadata dictionary for FCS export with both scatter and marker channels.
        
        Args:
            original_sample: Original FlowKit Sample
            total_channel_count: Total number of channels (scatter + markers)
            all_pnn_labels: Combined list of PnN labels (scatter + markers)
            all_pns_labels: Combined list of PnS labels (scatter + markers)
            
        Returns:
            Metadata dictionary for FCS export
        """
        metadata = {
            'datatype': 'F',  # Float data type
            'mode': 'L',      # List mode
            'tot': str(len(original_sample._raw_events)),  # Total events
            'par': str(total_channel_count),  # Number of parameters
        }
        
        # Add channel-specific metadata
        for i, (pnn_label, pns_label) in enumerate(zip(all_pnn_labels, all_pns_labels), 1):
            metadata[f'p{i}n'] = pnn_label
            metadata[f'p{i}s'] = pns_label
            
            # Different metadata for scatter vs marker channels
            if pnn_label in self.scatter_channels:
                # Scatter channels: use original metadata if available
                original_idx = original_sample.pnn_labels.index(pnn_label)
                original_range = original_sample.channels.iloc[original_idx]['pnr']
                metadata[f'p{i}r'] = str(int(original_range))
                metadata[f'p{i}g'] = '1.0'  # Gain already applied
                metadata[f'p{i}e'] = '0,0'  # Linear scale
            else:
                # Marker channels: standard corrected data metadata
                metadata[f'p{i}r'] = '262144'  # Range
                metadata[f'p{i}g'] = '1.0'     # Gain already applied
                metadata[f'p{i}e'] = '0,0'     # Linear scale
        
        # Add some original metadata if available
        preserve_keys = ['fil', 'date', 'cyt', 'cytsn', 'op', 'sys']
        for key in preserve_keys:
            if key in original_sample.metadata:
                metadata[key] = original_sample.metadata[key]
        
        # Add processing note
        metadata['note'] = 'Batch aligned using autoencoder - scatter channels preserved, markers corrected'
        
        return metadata


class BatchValidationError(Exception):
    """Custom exception for batch validation errors."""
    pass