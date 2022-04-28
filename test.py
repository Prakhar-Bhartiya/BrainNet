 #Libraries
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
xx = loadmat('Dataset1.mat') #dict_keys(['__header__', '__version__', '__globals__', 'Raw_Data', 'Sampling_Rate'])
yy = loadmat('sampleAttack.mat')

xx = xx['Raw_Data']
yy = yy['attackVectors']

def filter_band(data, duration):
        #high pass and low pass filter
        sampling_freq = 160.0
        time = np.arange(0.0, duration, 1/sampling_freq)
        low_freq = 0.5 #0.1 Hz
        high_freq = 2.0 #60 Hz

        filter = signal.firwin(401, [low_freq, high_freq], pass_zero=False,fs=sampling_freq)

        filtered_signal = signal.convolve(data, filter, mode='same')
        return filtered_signal
        # plt.plot(time, filtered_signal)
        # plt.show()

def power_spectrum_plot(data, duration):
    #https://www.adamsmith.haus/python/answers/how-to-plot-a-power-spectrum-in-python
    #Plotting a power spectrum of data will plot how much of the data exists at a range of frequencies.
    #The power spectrum is calculated as the square of the absolute value of the discrete Fourier transform
    #time_stop = 120sec
    sampling_freq = 160.0
    time = np.arange(0.0, duration, 1/160) #(start, stop, step)

    fourier_transform = np.fft.rfft(data)

    abs_fourier_transform = np.abs(fourier_transform)

    power_spectrum = np.square(abs_fourier_transform)

    frequency = np.linspace(0, 160/2, len(power_spectrum))

    plt.plot(frequency, power_spectrum)
    plt.show()

def fft(data, duration):
    fourier_transform = np.fft.rfft(data)
    time = np.arange(0,len(fourier_transform))

    plt.plot(fourier_transform)
    plt.show()


zz = filter_band(xx[0][0], 120)
power_spectrum_plot(zz, 120)
fft(zz, 120)
