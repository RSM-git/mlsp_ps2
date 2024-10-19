import librosa
import sounddevice as sd
import soundfile as sf


x, Fs = librosa.load("problem2_6.wav")

print("Playing...")
sd.play(x, Fs)
sd.wait()
