import librosa
import sounddevice as sd
import soundfile as sf
import numpy as np


import matplotlib.pyplot as plt

PLAY_SOUND = 0

x, Fs = librosa.load("problem2_6.wav")

if PLAY_SOUND:
    sd.play(x, Fs)
    sd.wait()


def spectrogram(signal, figure_name: str, sampling_rate: int, window_size: int, window_overlap: int, window_function = "hann"):
    # Calculate STFT (Short-Time Fourier Transform)
    spectrogram = librosa.stft(signal, n_fft=window_size, hop_length=window_overlap, center=True, window=window_function)
    
    # Convert amplitude to dB
    signal_db = librosa.amplitude_to_db(abs(spectrogram))
    
    # Create time axis for plotting
    # Number of frames in the spectrogram, divided by the sample rate and hop length
    time_axis = librosa.frames_to_time(np.arange(signal_db.shape[1]), sr=sampling_rate, hop_length=window_overlap)
    
    # Create figure
    plt.figure(figsize=(14, 6))
    plt.title(figure_name)
    
    # Display spectrogram with time on x-axis and frequency on y-axis (log scale)
    plt.pcolormesh(time_axis, librosa.fft_frequencies(sr=sampling_rate, n_fft=window_size), signal_db, shading='gouraud', cmap="jet")
    
    # Add color bar to reflect dB levels
    plt.colorbar(format='%+2.0f dB')
    
    # Labels for axes
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.tight_layout()

# Call the function
spectrogram(x, "Syntheziser spectrogram", Fs, 512, 16, "hann")
plt.show()