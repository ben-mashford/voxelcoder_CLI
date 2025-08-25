"""
Utilities for the Batch Alignment CLI.
Includes logging setup, dependency checks, and helper functions.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Optional, List
import importlib
import tempfile
import os


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Setup logging configuration."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup handlers
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    handlers.append(console_handler)
    
    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        handlers=handlers,
        force=True
    )
    
    # Reduce noise from other libraries
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('flowkit').setLevel(logging.WARNING)


def check_dependencies() -> None:
    """Check that required dependencies are available."""
    logger = logging.getLogger(__name__)
    
    required_packages = [
        'numpy',
        'pandas', 
        'flowkit',
        'torch',
        'yaml'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            logger.debug(f"✅ {package} available")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"❌ {package} not available")
    
    if missing_packages:
        logger.error(f"Missing required packages: {missing_packages}")
        logger.error("Install with: pip install " + " ".join(missing_packages))
        sys.exit(1)
    
    # Check CUDA availability if torch is available
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"🚀 CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("💻 CUDA not available, will use CPU")
    except Exception as e:
        logger.warning(f"Could not check CUDA availability: {e}")


def get_device(device_config: str = "auto") -> str:
    """Determine the appropriate device for PyTorch."""
    logger = logging.getLogger(__name__)
    
    if device_config == "cpu":
        logger.info("Using CPU as specified in configuration")
        return "cpu"
    elif device_config == "cuda":
        try:
            import torch
            if torch.cuda.is_available():
                logger.info("Using CUDA as specified in configuration")
                return "cuda"
            else:
                logger.warning("CUDA requested but not available, falling back to CPU")
                return "cpu"
        except ImportError:
            logger.error("PyTorch not available for CUDA")
            return "cpu"
    else:  # auto
        try:
            import torch
            if torch.cuda.is_available():
                logger.info("Auto-detected CUDA, using GPU")
                return "cuda"
            else:
                logger.info("Auto-detected CPU only")
                return "cpu"
        except ImportError:
            logger.error("PyTorch not available")
            return "cpu"


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_memory(bytes_val: int) -> str:
    """Format bytes into human-readable memory string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.1f}{unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.1f}TB"


def ensure_directory(path: Path) -> None:
    """Ensure directory exists, create if necessary."""
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise RuntimeError(f"Could not create directory {path}: {e}")


def count_fcs_files(directory: Path) -> int:
    """Count FCS files in a directory."""
    if not directory.exists() or not directory.is_dir():
        return 0
    return len(list(directory.glob("*.fcs")))


def get_fcs_files(directory: Path) -> List[Path]:
    """Get list of FCS files in a directory."""
    if not directory.exists() or not directory.is_dir():
        return []
    return sorted(list(directory.glob("*.fcs")))


class ProgressTracker:
    """Simple progress tracker for long-running operations."""
    
    def __init__(self, total_items: int, description: str = "Processing"):
        self.total_items = total_items
        self.current_item = 0
        self.description = description
        self.start_time = time.time()
        self.logger = logging.getLogger(__name__)
        
    def update(self, increment: int = 1, message: str = None) -> None:
        """Update progress."""
        self.current_item += increment
        progress_pct = (self.current_item / self.total_items * 100) if self.total_items > 0 else 0
        elapsed = time.time() - self.start_time
        
        if self.current_item > 0:
            estimated_total = elapsed * self.total_items / self.current_item
            remaining = estimated_total - elapsed
            eta_str = f" (ETA: {format_time(remaining)})" if remaining > 0 else ""
        else:
            eta_str = ""
        
        status_msg = f"{self.description}: {self.current_item}/{self.total_items} ({progress_pct:.1f}%){eta_str}"
        if message:
            status_msg += f" - {message}"
            
        self.logger.info(status_msg)
    
    def finish(self, message: str = None) -> None:
        """Mark progress as complete."""
        elapsed = time.time() - self.start_time
        final_msg = f"{self.description} completed in {format_time(elapsed)}"
        if message:
            final_msg += f" - {message}"
        self.logger.info(final_msg)


class TemporaryFile:
    """Context manager for temporary files."""
    
    def __init__(self, suffix: str = "", prefix: str = "batch_align_"):
        self.suffix = suffix
        self.prefix = prefix
        self.path = None
        
    def __enter__(self) -> Path:
        fd, self.path = tempfile.mkstemp(suffix=self.suffix, prefix=self.prefix)
        os.close(fd)  # Close the file descriptor, we just want the path
        return Path(self.path)
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.path and os.path.exists(self.path):
            try:
                os.unlink(self.path)
            except Exception:
                pass  # Ignore cleanup errors


def validate_fcs_file(fcs_path: Path) -> bool:
    """Quick validation that a file is a readable FCS file."""
    try:
        import flowkit as fk
        sample = fk.Sample(str(fcs_path))
        # Just try to access basic properties
        _ = sample.pnn_labels
        _ = sample.pns_labels
        return True
    except Exception:
        return False


def get_file_size(file_path: Path) -> int:
    """Get file size in bytes."""
    try:
        return file_path.stat().st_size
    except Exception:
        return 0


def summarize_batch_info(batch_path: Path) -> dict:
    """Get summary information about a batch directory."""
    fcs_files = get_fcs_files(batch_path)
    
    info = {
        'path': str(batch_path),
        'name': batch_path.name,
        'fcs_count': len(fcs_files),
        'total_size_bytes': sum(get_file_size(f) for f in fcs_files),
        'files': [f.name for f in fcs_files]
    }
    
    info['total_size_formatted'] = format_memory(info['total_size_bytes'])
    
    return info