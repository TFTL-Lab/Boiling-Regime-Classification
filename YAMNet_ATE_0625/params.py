"""Hyperparameters for YAMNet."""
from dataclasses import dataclass

# Use the @dataclass decorator to automatically generate init, repr, etc.
# frozen=True makes the instances immutable (cannot change fields after creation).
@dataclass(frozen=True)
class Params:
  # Audio sample rate (in Hz) for input waveform
  sample_rate: float = 10000.0  

  # Duration (in seconds) of the window used for each STFT frame
  stft_window_seconds: float = 0.025

  # Duration (in seconds) between successive STFT frames (hop length)
  stft_hop_seconds: float = 0.010 

  # Number of Mel frequency bands for Mel-spectrogram computation
  mel_bands: int = 64

  # Minimum frequency (Hz) considered in Mel filterbank
  mel_min_hz: float = 0 

  # Maximum frequency (Hz) considered in Mel filterbank
  mel_max_hz: float = 2000  

  # Small constant added to log mel spectrogram to avoid log(0)
  log_offset: float = 0.001

  # Duration (in seconds) of each spectrogram patch passed to the model
  patch_window_seconds: float = 1.00

  # Step size (in seconds) between adjacent spectrogram patches
  patch_hop_seconds: float = 1.00 

  @property
  def patch_frames(self):
    # Number of STFT frames in one patch, derived from window and hop duration
    return int(round(self.patch_window_seconds / self.stft_hop_seconds))

  @property
  def patch_bands(self):
    # The number of frequency bins in the mel-spectrogram patch (same as mel_bands)
    return self.mel_bands

  # Number of output classes for the classifier (e.g., sound categories)
  num_classes: int = 3

  # Padding method used in convolutional layers ('same' retains input size)
  conv_padding: str = 'same'

  # Whether to use centering in batch normalization layers
  batchnorm_center: bool = True

  # Whether to use scaling in batch normalization layers
  batchnorm_scale: bool = False

  # Small constant added to variance in batch normalization for numerical stability
  batchnorm_epsilon: float = 1e-4

  # Activation function used in the final classifier layer
  classifier_activation: str = 'sigmoid'

  # If True, model will use only operations compatible with TensorFlow Lite
  tflite_compatible: bool = False
