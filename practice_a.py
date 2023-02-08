from IPython.display import Audio
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from python_speech_features import mfcc
#this allows us to compute the MFCC 
import librosa 
import librosa.display
import os
import tqdm
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
import seaborn as sns
from pathlib import Path

import math, random
import torch
import torchaudio
from torchaudio import transforms

from IPython.display import Audio 
sample_path = "Ses01F_impro01.wav"
#when I try to visualize this audio file, it gives me orange and blue lines - why is this the case?
#returns a tuple with the waveform data
#how does it know the sample rate? does it change based on the file?
samplerate, signal = read(sample_path)
#what does the signal actually say.
print(samplerate, signal)
Audio(sample_path)
timestamps = [i/samplerate for i in range(len(signal))]
#convert the x axis to the time in seconds
plt.plot(timestamps, signal)
plt.ylabel("Signal Amplitude")
plt.xlabel("Time (in Seconds)")
plt.show()

#now we want to create the MFCC. this is the mel-freqeny cepstral coefficients
#this is a type of spectral representation, it shows colors and stuff
#it uses the discrete fourier transform of the signal and maps the frequency to the Mel scale
#then we take the log of power in the frequency band 
#take the discrete cosin transform of eacch band 
#then we take the amplitude of the result. 

#why were these numbers chosen?
mfcc_features = mfcc(signal, samplerate, winlen = 0.05, winstep = 0.01, numcep = 10, nfilt = 1024)
plt.title("MFCC of Sample")
plt.pcolor(mfcc_features.transpose(), cmap = "inferno")
plt.show()