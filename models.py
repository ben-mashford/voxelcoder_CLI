"""
Neural network models for batch alignment.
Adapted from the original models.py with lazy imports.
"""

import logging

logger = logging.getLogger(__name__)


def _get_torch_modules():
    """Lazy import torch modules when needed."""
    import torch
    import torch.nn as nn
    return torch, nn


class Autoencoder:
    """
    Standard autoencoder for batch alignment.
    Uses lazy loading to avoid importing PyTorch unless actually needed.
    """
    
    def __init__(self, input_dim: int, encoding_dim: int):
        torch, nn = _get_torch_modules()
        
        class _Autoencoder(nn.Module):
            def __init__(self, input_dim, encoding_dim):
                super(_Autoencoder, self).__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 64),
                    nn.BatchNorm1d(64, track_running_stats=False),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.BatchNorm1d(32, track_running_stats=False),
                    nn.ReLU(),
                    nn.Linear(32, encoding_dim),
                    nn.BatchNorm1d(encoding_dim, track_running_stats=False)
                )
                self.decoder = nn.Sequential(
                    nn.Linear(encoding_dim, 32),
                    nn.BatchNorm1d(32, track_running_stats=False),
                    nn.ReLU(),
                    nn.Linear(32, 64),
                    nn.BatchNorm1d(64, track_running_stats=False),
                    nn.ReLU(),
                    nn.Linear(64, input_dim)
                )
            
            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded
        
        self._model = _Autoencoder(input_dim, encoding_dim)
        logger.debug(f"Created autoencoder: {input_dim} -> {encoding_dim} -> {input_dim}")
        
    def __getattr__(self, name):
        """Delegate all attribute access to the internal model."""
        return getattr(self._model, name)
    
    def __call__(self, *args, **kwargs):
        return self._model(*args, **kwargs)


class AutoencoderSmall:
    """Smaller autoencoder variant for faster training."""
    
    def __init__(self, input_dim: int, encoding_dim: int):
        torch, nn = _get_torch_modules()
        
        class _AutoencoderSmall(nn.Module):
            def __init__(self, input_dim, encoding_dim):
                super(_AutoencoderSmall, self).__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 32),
                    nn.BatchNorm1d(32, track_running_stats=False),
                    nn.ReLU(),
                    nn.Linear(32, 16),
                    nn.BatchNorm1d(16, track_running_stats=False),
                    nn.ReLU(),
                    nn.Linear(16, encoding_dim),
                    nn.BatchNorm1d(encoding_dim, track_running_stats=False)
                )
                self.decoder = nn.Sequential(
                    nn.Linear(encoding_dim, 16),
                    nn.BatchNorm1d(16, track_running_stats=False),
                    nn.ReLU(),
                    nn.Linear(16, 32),
                    nn.BatchNorm1d(32, track_running_stats=False),
                    nn.ReLU(),
                    nn.Linear(32, input_dim)
                )
            
            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded
        
        self._model = _AutoencoderSmall(input_dim, encoding_dim)
        logger.debug(f"Created small autoencoder: {input_dim} -> {encoding_dim} -> {input_dim}")
        
    def __getattr__(self, name):
        return getattr(self._model, name)
    
    def __call__(self, *args, **kwargs):
        return self._model(*args, **kwargs)


class AutoencoderLarge:
    """Larger autoencoder variant for better reconstruction."""
    
    def __init__(self, input_dim: int, encoding_dim: int):
        torch, nn = _get_torch_modules()
        
        class _AutoencoderLarge(nn.Module):
            def __init__(self, input_dim, encoding_dim):
                super(_AutoencoderLarge, self).__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 128),
                    nn.BatchNorm1d(128, track_running_stats=False),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.BatchNorm1d(64, track_running_stats=False),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.BatchNorm1d(32, track_running_stats=False),
                    nn.ReLU(),
                    nn.Linear(32, encoding_dim),
                    nn.BatchNorm1d(encoding_dim, track_running_stats=False)
                )
                self.decoder = nn.Sequential(
                    nn.Linear(encoding_dim, 32),
                    nn.BatchNorm1d(32, track_running_stats=False),
                    nn.ReLU(),
                    nn.Linear(32, 64),
                    nn.BatchNorm1d(64, track_running_stats=False),
                    nn.ReLU(),
                    nn.Linear(64, 128),
                    nn.BatchNorm1d(128, track_running_stats=False),
                    nn.ReLU(),
                    nn.Linear(128, input_dim)
                )
            
            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded
        
        self._model = _AutoencoderLarge(input_dim, encoding_dim)
        logger.debug(f"Created large autoencoder: {input_dim} -> {encoding_dim} -> {input_dim}")
        
    def __getattr__(self, name):
        return getattr(self._model, name)
    
    def __call__(self, *args, **kwargs):
        return self._model(*args, **kwargs)


def wasserstein_distance(hist1, hist2):
    """Calculate Wasserstein distance between two histograms."""
    torch, _ = _get_torch_modules()
    return torch.mean(torch.abs(torch.cumsum(hist1, dim=1) - torch.cumsum(hist2, dim=1)))


def histogram_loss(output, target, num_bins: int = 10):
    """
    Calculate histogram loss between output and target.
    
    Args:
        output: Model output tensor
        target: Target tensor
        num_bins: Number of bins for histogram calculation
        
    Returns:
        Histogram loss value
    """
    torch, _ = _get_torch_modules()
    batch_size, num_channels = output.shape
    hist_loss = 0

    for c in range(num_channels):
        output_channel = output[:, c]
        target_channel = target[:, c]

        output_hist = torch.histc(output_channel, bins=num_bins, min=0, max=1)
        target_hist = torch.histc(target_channel, bins=num_bins, min=0, max=1)

        # Normalize histograms
        output_hist = output_hist / (output_hist.sum() + 1e-8)
        target_hist = target_hist / (target_hist.sum() + 1e-8)

        hist_loss += wasserstein_distance(output_hist.unsqueeze(0), target_hist.unsqueeze(0))

    return hist_loss / num_channels


def create_autoencoder(model_type: str, input_dim: int, encoding_dim: int):
    """
    Factory function to create autoencoders.
    
    Args:
        model_type: Type of model ('standard', 'small', 'large')
        input_dim: Input dimension (number of channels)
        encoding_dim: Encoding dimension (bottleneck size)
        
    Returns:
        Autoencoder instance
    """
    model_classes = {
        'standard': Autoencoder,
        'small': AutoencoderSmall,
        'large': AutoencoderLarge
    }
    
    if model_type not in model_classes:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(model_classes.keys())}")
    
    logger.info(f"Creating {model_type} autoencoder with {input_dim} inputs and {encoding_dim} encoding dimensions")
    return model_classes[model_type](input_dim, encoding_dim)


def count_model_parameters(model) -> int:
    """Count the number of trainable parameters in a model."""
    try:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    except:
        return 0