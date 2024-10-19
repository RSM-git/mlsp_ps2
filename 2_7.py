from scipy.io import loadmat
from scipy import signal

import librosa

import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt

fs = 8000

speech = loadmat("problem2_7_speech.mat")["speech"].flatten()

lpir = loadmat("problem2_7_lpir.mat")["lpir"].flatten()
bpir = loadmat("problem2_7_bpir.mat")["bpir"].flatten()
hpir = loadmat("problem2_7_hpir.mat")["hpir"].flatten()

print(lpir, bpir, hpir)
print(len(lpir), len(bpir), len(hpir))


noise = np.random.normal(loc=0, scale=0.1, size=len(speech))

filtered_noise = signal.lfilter(lpir, 1, noise)

noisy_speech = speech + filtered_noise

# sd.play(speech, fs)
# sd.wait()

# sd.play(noisy_speech, fs)
# sd.wait()

# Plot the spectrograms of the signals
# noisy speech
spec_y = librosa.stft(noisy_speech, n_fft=512, hop_length=32, center=True)
y_db = librosa.amplitude_to_db(abs(spec_y))
plt.figure(figsize=(14, 3))
plt.title("NOISY SPEECH")
librosa.display.specshow(y_db, sr=fs)

# clean speech
spec_x = librosa.stft(speech, n_fft=512, hop_length=32, center=True)
x_db = librosa.amplitude_to_db(abs(spec_x))
plt.figure(figsize=(14, 3))
plt.title("Speech")
librosa.display.specshow(x_db, sr=fs)

plt.show()

# lms

# nlms

# rls