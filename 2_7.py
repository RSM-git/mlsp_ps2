from scipy.io import loadmat
from scipy import signal

import librosa

import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt

from adaptive_filtering_algorithms import lms, nlms, rls, mse

PLAY_SPEECH = 1
PLOT_SPECTROGRAMS = 1

repeats = 5

fs = 8000

L = 64

mu_lms = 0.01

mu_nlms = 0.1
delta = 1e-4

rls_beta = 0.999
rls_lambda = 1

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

lms_mses, nlms_mses, rls_mses = [], [], []

for _ in range(repeats):
    noise = np.random.normal(loc=0, scale=0.8, size=len(speech))

    filtered_noise = signal.lfilter(lpir, 1, noise)

    noisy_speech = speech + filtered_noise

    # lms
    noise_estimate_lms, lms_error = lms(noise, noisy_speech, L, mu_lms)
    lms_denoised_speech = noisy_speech - noise_estimate_lms

    # nlms
    noise_estimate_nlms, nlms_error = nlms(noise, noisy_speech, L, mu_nlms, delta)
    nlms_denoised_speech = noisy_speech - noise_estimate_nlms

    # rls
    noise_estimate_rls, rls_error = rls(noise, noisy_speech, L, rls_beta, rls_lambda)
    rls_denoised_speech = noisy_speech - noise_estimate_rls

    lms_mse, nlms_mse, rls_mse = mse(lms_denoised_speech, speech), mse(nlms_denoised_speech, speech), mse(rls_denoised_speech, speech)

    lms_mses.append(lms_mse)
    nlms_mses.append(nlms_mse)
    rls_mses.append(rls_mse)

print(f"MSE of LMS signal {np.mean(lms_mses)}")
print(f"MSE of NLMS signal {np.mean(nlms_mses)}")
print(f"MSE of RLS signal {np.mean(rls_mses)}")


if PLAY_SPEECH:
    play_sound(speech)

    play_sound(noisy_speech)

    play_sound(lms_denoised_speech)

    play_sound(nlms_denoised_speech)

    play_sound(rls_denoised_speech)

if PLOT_SPECTROGRAMS:
    # noisy speech
    spectrogram(noisy_speech, "Noisy speech")

    # clean speech
    spectrogram(speech, "Clean speech")

    # lms
    spectrogram(lms_denoised_speech, "LMS filtered signal")

    # nlms
    spectrogram(nlms_denoised_speech, "NLMS filtered signal")

    # rls
    spectrogram(rls_denoised_speech, "RLS filtered signal")

    plt.show()
