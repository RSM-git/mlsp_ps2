from scipy.io import loadmat
from scipy import signal

import librosa

import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt

from scipy import linalg

def convmtx(h, n):
    '''
    Convolution matrix, same as convmtx does in matlab
    '''
    return linalg.toeplitz(
        np.hstack([h, np.zeros(n-1)]),
        np.hstack([h[0], np.zeros(n-1)]),
    )



PLAY_SPEECH = 1
PLOT_SPECTROGRAMS = 0

fs = 8000

speech = loadmat("problem2_7_speech.mat")["speech"].flatten()

lpir = loadmat("problem2_7_lpir.mat")["lpir"].flatten()
bpir = loadmat("problem2_7_bpir.mat")["bpir"].flatten()
hpir = loadmat("problem2_7_hpir.mat")["hpir"].flatten()

noise = np.random.normal(loc=0, scale=0.2, size=len(speech))

filtered_noise = signal.lfilter(lpir, 1, noise)

noisy_speech = speech + filtered_noise

if PLAY_SPEECH:
    sd.play(speech, fs)
    sd.wait()

    sd.play(noisy_speech, fs)
    sd.wait()

if PLOT_SPECTROGRAMS:
    # noisy speech
    spec_noisy = librosa.stft(noisy_speech, n_fft=512, hop_length=32, center=True)
    y_db = librosa.amplitude_to_db(abs(spec_noisy))
    plt.figure(figsize=(14, 3))
    plt.title("NOISY SPEECH")
    librosa.display.specshow(y_db, sr=fs)

    # clean speech
    spec_clean = librosa.stft(speech, n_fft=512, hop_length=32, center=True)
    x_db = librosa.amplitude_to_db(abs(spec_clean))
    plt.figure(figsize=(14, 3))
    plt.title("Speech")
    librosa.display.specshow(x_db, sr=fs)

    plt.show()

# lms
def lms(x, y, L, mu):
    N = y.shape[0]
    w = np.zeros(L,)
    yhat = np.zeros(N,)
    e = np.zeros(N,)

    # zero-pad input signal
    x = np.concatenate((np.zeros(L-1,), x), axis=0)

    for n in range(0, N):
        x_n = x[n:n+L]
        yhat[n] = w.T @ x_n
        e[n] = y[n] - yhat[n]
        w = w + mu*e[n]*x_n
        
    return yhat, e

mu_lms = 0.001
L = 128

noise_estimate, _ = lms(noisy_speech, noise, L, mu_lms)


if PLOT_SPECTROGRAMS:
    spec_yhat = librosa.stft(noisy_speech-noise_estimate, n_fft=512, hop_length=32, center=True)
    yhat_db = librosa.amplitude_to_db(abs(spec_yhat))
    plt.figure(figsize=(14, 3))
    plt.title("FILTER ERROR SIGNAL")
    librosa.display.specshow(yhat_db, sr=fs)
    plt.show()

sd.play(noisy_speech-noise_estimate, fs)
sd.wait()

# nlms
def nlms(x, y, L, mu, delta):
    N = y.shape[0]
    w = np.zeros(L,)
    yhat = np.zeros(N,)
    e = np.zeros(N,)

    if x.ndim == 1:
        X = convmtx(x, L).T
    else:
        X = x

    for n in range(0, N):
        x_n = X[:,n]
        yhat[n] = w.T @ x_n
        e[n] = y[n] - yhat[n]
        w = w + (mu / (delta+x_n.T@x_n)) * x_n * e[n]
        
    return yhat, e

mu_nlms = 0.005
delta = 1e-2
L = 64

noise_estimate_nlms, _ = nlms(noisy_speech, noise, L, mu_nlms, delta)


if PLOT_SPECTROGRAMS:
    spec_yhat = librosa.stft(noisy_speech-noise_estimate_nlms, n_fft=512, hop_length=32, center=True)
    yhat_db = librosa.amplitude_to_db(abs(spec_yhat))
    plt.figure(figsize=(14, 3))
    plt.title("FILTER ERROR SIGNAL")
    librosa.display.specshow(yhat_db, sr=fs)
    plt.show()

sd.play(noisy_speech-noise_estimate_nlms, fs)
sd.wait()

# rls

def rls(x: np.ndarray, y: np.ndarray, L: int, beta: float, lambda_: float):
    '''
    Input
        x: input signal
        y: desired signal
        L: filter length
        beta: forget factor
        lambda_: regularization

    Output
        yhat: filter output
    '''
    yhat = np.zeros(len(y))
    e = np.zeros(len(y))

    if x.ndim == 1:
        X = np.fliplr(convmtx(x,L)).T
    else:
        X = x

    # start RLS
    # initialize
    theta = np.zeros((L, 1))  # theta in the book
    P = 1/lambda_*np.eye(L)

    for n in range(len(y)):
        x_n = X[:,n, None]

        # get filter output
        yhat[n] = theta.T@x_n

        # update iteration
        e[n] = y[n] - yhat[n]
        z = P @ x_n
        denominator = beta + x_n.T @ z
        K_n = z/denominator
        theta = theta + K_n*e[n]
        P = (P - K_n @ z.T)/beta

    return yhat, e

rls_beta = 0.999
rls_lambda = 1e5
L = 128

noise_estimate_rls, _ = rls(noisy_speech, noise, L, rls_beta, rls_lambda)


if PLOT_SPECTROGRAMS:
    spec_yhat = librosa.stft(noisy_speech-noise_estimate_rls, n_fft=512, hop_length=32, center=True)
    yhat_db = librosa.amplitude_to_db(abs(spec_yhat))
    plt.figure(figsize=(14, 3))
    plt.title("FILTER ERROR SIGNAL")
    librosa.display.specshow(yhat_db, sr=fs)
    plt.show()

sd.play(noisy_speech-noise_estimate_rls, fs)
sd.wait()