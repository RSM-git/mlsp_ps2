from scipy.io import loadmat
from scipy import signal

import librosa

import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt

from adaptive_filtering_algorithms import lms, nlms, rls

PLAY_SPEECH = 1
PLOT_SPECTROGRAMS = 1

fs = 8000

def spectrogram(signal, figure_name: str, sampling_rate: int = 8000):
    spectrogram = librosa.stft(signal, n_fft=512, hop_length=32, center=True)
    signal_db = librosa.amplitude_to_db(abs(spectrogram))
    plt.figure(figsize=(14, 4))
    plt.title(figure_name)
    librosa.display.specshow(signal_db, sr=sampling_rate)

def play_sound(signal, fs: int = 8000):
    sd.play(signal, fs)
    sd.wait()


speech = loadmat("problem2_7_speech.mat")["speech"].flatten()

lpir = loadmat("problem2_7_lpir.mat")["lpir"].flatten()
bpir = loadmat("problem2_7_bpir.mat")["bpir"].flatten()
hpir = loadmat("problem2_7_hpir.mat")["hpir"].flatten()

noise = np.random.normal(loc=0, scale=0.2, size=len(speech))

filtered_noise = signal.lfilter(lpir, 1, noise)

noisy_speech = speech + filtered_noise

if PLAY_SPEECH:
    play_sound(speech)

    play_sound(noisy_speech)

if PLOT_SPECTROGRAMS:
    # noisy speech
    spectrogram(noisy_speech, "Noisy speech")

    # clean speech
    spectrogram(speech, "Clean speech")

    plt.show()

# lms
mu_lms = 0.001
L = 10

noise_estimate, _ = lms(noisy_speech, noise, L, mu_lms)

play_sound(noisy_speech-noise_estimate)

spectrogram(noisy_speech-noise_estimate, "LMS filtered signal")
plt.show()

# nlms

mu_nlms = 0.005
delta = 1e-2
L = 10

noise_estimate_nlms, _ = nlms(noisy_speech, noise, L, mu_nlms, delta)

sd.play(noisy_speech-noise_estimate_nlms, fs)
sd.wait()

spectrogram(noisy_speech-noise_estimate_nlms, "NLMS filtered signal")
plt.show()

# rls

rls_beta = 0.9999
rls_lambda = 1e8
L = 10

noise_estimate_rls, _ = rls(noisy_speech, noise, L, rls_beta, rls_lambda)

sd.play(noisy_speech-noise_estimate_rls, fs)
sd.wait()

spectrogram(noisy_speech-noise_estimate_rls, "RLS filtered signal")
plt.show()
