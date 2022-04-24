#!/usr/bin/env python
# coding: utf-8

 #Libraries
from scipy.io import loadmat
import numpy as np

import sklearn as sk
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf

import warnings
warnings.filterwarnings('ignore')

class model:

  def logReg(X_train, X_test, y_train, y_test):
    logReg = LogisticRegression().fit(X_train,y_train)
    logTest = logReg.score(X_test, y_test)
    return logReg, logTest

  def kMeans(X_train, X_test, y_train, y_test):
    kmeans = KMeans(init="k-means++", n_clusters=2, n_init=10, max_iter=300)
    kmeans.fit(X_train)
    labels1 = kmeans.labels_
    kmeansTest = kmeans.predict(X_test)
    accuracy = accuracy_score(y_test, kmeansTest)
    return kmeans, accuracy

  def SVM(X_train, X_test, y_train, y_test):
    svm = LinearSVC(max_iter=30000).fit(X_train,y_train)
    svmTest = svm.score(X_test, y_test)
    return svm, svmTest


class base:
    def plot_data(data,duration):
        sampling_freq = 160.0
        time = np.arange(0.0, duration, 1/sampling_freq)
        plt.plot(time,data)

    def segment_data(input_data, seg_time=30):
        # 30 seconds
        segment_points = seg_time * 160 #sampling freq
        splited_data =np.asarray(np.split(input_data.flatten(), segment_points)).T

        return splited_data

    def form_data(input_data,attack_data):
        segment_time = 30 #window = 30seconds
        input_ = base.segment_data(input_data,segment_time)
        attack_ = base.segment_data(attack_data,segment_time)

        X = np.concatenate((input_,attack_))
        Y = np.concatenate((np.zeros(input_.shape[0]),np.ones(attack_.shape[0]))) #normal = 0, attack = 1

        return X,Y


class feature:

    sampling_freq = 160.0

    def power_spectrum_plot(data, duration):
        #https://www.adamsmith.haus/python/answers/how-to-plot-a-power-spectrum-in-python

        #time_stop = 120sec
        time = np.arange(0.0, duration, 1/sampling_freq) #(start, stop, step)

        fourier_transform = np.fft.rfft(data)

        abs_fourier_transform = np.abs(fourier_transform)

        power_spectrum = np.square(abs_fourier_transform)

        frequency = np.linspace(0, sampling_freq/2, len(power_spectrum))

        plt.plot(frequency, power_spectrum)

    def calcPCA(data):
        pca = PCA(n_components=20)
        X_pca = pca.fit_transform(data)
        return X_pca

class preprocess:

    sampling_freq = 160.0

    def filter_band(data, duration):
        #high pass and low pass filter
        sampling_freq = 160.0
        time = np.arange(0.0, duration, 1/sampling_freq)
        low_freq = 0.5 #0.1 Hz
        high_freq = 2.0 #60 Hz

        filter = signal.firwin(401, [low_freq, high_freq], pass_zero=False,fs=sampling_freq)

        filtered_signal = signal.convolve(data, filter, mode='same')

        plt.plot(time, filtered_signal)

def main():


    #Read data
    input_data = loadmat('Dataset1.mat') #dict_keys(['__header__', '__version__', '__globals__', 'Raw_Data', 'Sampling_Rate'])
    attack_data = loadmat('sampleAttack.mat')#dict_keys(['__header__', '__version__', '__globals__', 'attackVectors'])

    #loading data
    input_data = input_data['Raw_Data']
    attack_data = attack_data['attackVectors']

    #matrix of 106*3*19200 == > 106 subjects, 3 times of 2 min per subject,
    #160 Hz sampling rate. (19200 = 120 s * 160 hz) 160 samples per second
    print("Input data shape: ", input_data.shape)

    #matrix of 106*3*19200 == > 6 attack types | 106 subjects | 3 times | 30 sec per subject,
    #160 Hz sampling rate. (4800 = 30 s * 160 hz) 160 samples per second
    print("Attack data shape: ", attack_data.shape)

    #Combine all data
    X,Y = base.form_data(input_data,attack_data)

    print(X.shape)
    print(Y.shape)


    #Divide data into train and test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42)
    #Pass an int for reproducible output across multiple function calls

    # PCA on training and testdata
    PCA_train = feature.calcPCA(X_train)
    PCA_test = feature.calcPCA(X_test)

    # Log Reg
    logRegPCA, logTestPCA = model.logReg(PCA_train, PCA_test, y_train, y_test) #PCA
    print("Log Reg PCA: ", logTestPCA)

    #K-Means
    kMeansPCA, kTestPCA = model.kMeans(PCA_train, PCA_test, y_train, y_test) #PCA
    print("KMeans PCA: ", kTestPCA)

    # SVM
    svmPCA, svmTestPCA = model.SVM(PCA_train, PCA_test, y_train, y_test) #PCA
    print("SVM PCA: ", svmTestPCA)


if __name__ == "__main__":
    main()
