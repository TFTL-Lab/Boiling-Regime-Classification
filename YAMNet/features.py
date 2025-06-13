import numpy as np
import tensorflow as tf
from scipy import signal
import scipy.io.wavfile as wav
from scipy.signal import spectrogram


# Audio Feature Extraction
fs = 10000
segment_duration = 1
frame_duration = 0.025
hop_duration = 0.010
fft_size= 512
segment_samples = int(segment_duration * fs)
frame_samples = int(frame_duration * fs)
hop_samples = int(hop_duration * fs)
min_frequency = 0
max_frequency = 2000
un_norm = 2 / (np.sum(signal.windows.hann(frame_samples, sym=False))**2)

# Function to extract features from an audio file

def extract_features(file):
    sample_rate, y = wav.read(file)
    y = np.int16(((y - y.min()) / (y.max() - y.min())) * 65535 - 32768)
    f, t, Zxx = spectrogram(y, fs=fs, nperseg=frame_samples, noverlap=frame_samples-hop_samples,nfft = 512)#window='hann'
    freq_mask = (f >= min_frequency) & (f <= max_frequency)
    linear_spectrum = Zxx.astype(float) 
    linear_spectrum = linear_spectrum[freq_mask, : ]
    linear_spectrum /= un_norm
    epsil=1e-6
    linear_spectrum = np.log10(linear_spectrum + epsil)
    return linear_spectrum[..., np.newaxis]