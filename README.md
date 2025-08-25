# Voxelcoder Flow Cytomery Batch Alignment: command-line interface

A command-line tool for aligning flow cytometry batches using autoencoder neural networks. This tool processes FCS files to correct for batch effects, enabling more robust cross-batch analysis.

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Your Data**
   ```
   data/
   ├── batch_A/          # Reference batch
   │   ├── file1.fcs
   │   ├── file2.fcs
   │   └── ...
   ├── batch_B/          # Target batch 1
   │   ├── file1.fcs
   │   └── ...
   └── batch_C/          # Target batch 2
       └── ...
   ```

3. **Create Configuration**
   ```bash
   cp config_example.yaml my_config.yaml
   # Edit my_config.yaml with your paths and markers
   ```

4. **Validate Configuration**
   ```bash
   python cli.py --config my_config.yaml --validate
   ```

5. **Run Alignment**
   ```bash
   python cli.py --config my_config.yaml
   ```

## Configuration

The tool uses YAML configuration files. Here's a minimal example:

```yaml
experiment:
  name: "my_experiment"
  output_dir: "./results"

data:
  reference_batch: "./data/batch_A"
  target_batches:
    - "./data/batch_B"
    - "./data/batch_C"

channels:
  selected_markers:
    - "CD4"
    - "CD8"
    - "CD3"
    # ... add your markers here

compensation:
  method: "spill"  # or "none" or "/path/to/comp.csv"
```

### Key Configuration Sections

- **experiment**: Basic experiment settings and output location
- **data**: Paths to reference and target batch directories
- **channels**: List of marker names to include in alignment
- **compensation**: Compensation method configuration
- **processing**: Cell count limits and processing parameters
- **model**: Neural network architecture and training parameters
- **hardware**: Device selection (GPU/CPU)
- **logging**: Logging configuration

## Command Line Options

```bash
python cli.py --config CONFIG [OPTIONS]

Options:
  --config, -c PATH     Path to YAML configuration file (required)
  --validate           Validate configuration without running
  --dry-run            Show processing plan without running
  --skip-validation    Skip pre-flight validation (not recommended)
  --verbose, -v        Enable verbose output
  --help              Show help message
```

## Output Structure

The tool creates the following output structure:

```
results/
├── models/
│   └── autoencoder.pt              # Trained model
├── aligned_data/
│   ├── reference_data.npy          # Reference dataset
│   ├── batch_A/                   # Reference batch results
│   │   ├── file1.fcs_before.npy
│   │   ├── file1.fcs_after.npy
│   │   ├── file1.fcs_corrected.fcs  # (if FCS export enabled)
│   │   └── ...
│   ├── batch_B/                   # Target batch results
│   │   └── ...
│   └── batch_C/
│       └── ...
├── logs/                          # Log files (if configured)
└── summary_report.json            # Comprehensive report
```

## FCS Export Feature

The tool can export corrected data back to FCS format with proper linear scaling. To enable FCS export, add this to your configuration:

```yaml
export:
  fcs_files: true  # Export corrected data as FCS files
```

When enabled, the tool will:
- Transform corrected data from logicle scale back to linear scale
- Preserve original metadata (channel names, acquisition info, etc.)
- Create new FCS files with suffix `_corrected.fcs`
- Only include the selected markers in the output files

## Validation

The tool performs extensive validation before processing:

1. **Configuration Syntax**: YAML format and required fields
2. **File System**: Directory existence and permissions
3. **FCS Files**: Presence of .fcs files in batch directories
4. **Marker Validation**: Ensures all selected markers exist in reference batch
5. **Hardware**: CUDA availability (if requested)

## Model Architecture

The tool supports three model types:

- **small**: Faster training, suitable for small datasets
- **standard**: Default balanced architecture
- **large**: Better reconstruction, requires more memory

All models use:
- Encoder-decoder architecture with batch normalization
- Adam optimizer
- Combined MSE and histogram loss
- Logicle transformation for flow cytometry data

## Examples

### Basic Usage
```bash
python cli.py --config config.yaml
```

### Validation Only
```bash
python cli.py --config config.yaml --validate
```

### Dry Run (Show Plan)
```bash
python cli.py --config config.yaml --dry-run
```

### Verbose Output
```bash
python cli.py --config config.yaml --verbose
```

## Troubleshooting

### Common Issues

1. **Marker Not Found**
   ```
   Error: Markers not found in reference batch: ['CD45RA']
   Available markers: ['CD45 RA', 'CD33', ...]
   ```
   **Solution**: Check marker names exactly match those in your FCS files

2. **CUDA Out of Memory**
   ```
   RuntimeError: CUDA out of memory
   ```
   **Solution**: Reduce `batch_size` or `inference_batch_size` in config

3. **No FCS Files Found**
   ```
   Error: No FCS files found in reference batch
   ```
   **Solution**: Verify directory paths and file extensions (.fcs)

### Performance Tips

- Use GPU for faster processing: `device: "cuda"`
- Adjust batch sizes based on available memory
- Use `max_cells_per_file` to limit memory usage
- Monitor GPU memory with `nvidia-smi`

## Requirements

- Python 3.7+
- PyTorch 1.9+
- FlowKit 1.0+
- NumPy, Pandas, PyYAML
- CUDA (optional, for GPU acceleration)

## License

This project is based on the VoxelCoder batch alignment methodology.