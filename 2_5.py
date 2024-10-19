import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat
from scipy import signal
from scipy.linalg import toeplitz

import sys

compute_mse = int(sys.argv[1])

def xcorr(x, y, k):
    N = min(len(x),len(y))
    r_xy = (1/N) * signal.correlate(x,y,'full') # reference implementation is unscaled
    return r_xy[N-k-1:N+k]


def compute_filter(signal, desired_signal, length):
    r_xx = xcorr(signal, signal, length-1)
    R_xx = toeplitz(r_xx[length-1:])
    r_dx = xcorr(desired_signal, signal, length-1)
    coefficients = np.linalg.solve(R_xx, r_dx[length-1:])
    return coefficients

s = loadmat("problem2_5_signal.mat")
w = loadmat("problem2_5_noise.mat")

s = s["signal"].flatten()
w = w["noise"].flatten()

x = s + w
d = s

if compute_mse:
    lengths = range(1, 500)  # Different filter lengths to test
    mse_values = []

    for length in lengths:
        # Compute autocorrelations and R_xx matrix
        theta = compute_filter(x, d, length)
        
        # Filter the signal
        estimated_signal = signal.lfilter(theta, 1, x)

        s_estimated = x - estimated_signal
        
        # Compute MSE between estimated and true signal
        mse = np.mean((s - s_estimated) ** 2)
        mse_values.append(mse)

    # Plot MSE vs Filter Length
    plt.plot(lengths, mse_values, 'o-')
    plt.title('MSE vs Filter Length')
    plt.xlabel('Filter Length')
    plt.ylabel('Mean Squared Error')
    plt.show()


filter_length = 115

theta = compute_filter(x, d, filter_length)

estimated_signal = signal.lfilter(theta, 1, x)

s_estimated = estimated_signal

# Sampling rate (8 kHz)
fs = 8000  # 8 kHz

# Plotting the frequency response of the Wiener filter
w, h = signal.freqz(theta, worN=8192)
frequencies_hz = w * fs / (2 * np.pi)  # Convert from rad/sample to Hz

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.title('Frequency Response of the FIR Wiener Filter')
plt.plot(frequencies_hz, 20 * np.log10(np.abs(h)), 'b')
plt.ylabel('Magnitude [dB]')
plt.xlabel('Frequency [Hz]')
plt.grid()

# Frequency response of the original signal (s), noisy signal (x), and estimated signal (s_estimated)

S = np.fft.fft(s)
X = np.fft.fft(x)
S_est = np.fft.fft(s_estimated)

# Compute corresponding frequency bins in Hz
freqs = np.fft.fftfreq(len(d), 1/fs)

plt.subplot(2, 1, 2)
plt.title('Frequency Response of Signals')
plt.plot(freqs[:len(freqs)//2], 20 * np.log10(np.abs(S[:len(freqs)//2])), label='Original Signal (s)')
plt.plot(freqs[:len(freqs)//2], 20 * np.log10(np.abs(X[:len(freqs)//2])), label='Noisy Signal (x)')
plt.plot(freqs[:len(freqs)//2], 20 * np.log10(np.abs(S_est[:len(freqs)//2])), label='Estimated Signal (s_estimated)')
plt.ylabel('Magnitude [dB]')
plt.xlabel('Frequency [Hz]')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

