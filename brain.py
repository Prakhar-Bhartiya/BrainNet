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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy import signal
# import warnings
# warnings.filterwarnings('ignore')

class base:
    def plot_data(data,duration):
        sampling_freq = 165.0
        time = np.arange(0.0, duration, 1/sampling_freq)
        plt.plot(time,data)

    def apply_all(f, data):
        """Return applied to whole dataset"""
        return np.array(list(map(f, data)))

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

    def accuracy(y_pred, y_true):
        from sklearn.metrics import accuracy_score
        return accuracy_score(y_true, y_pred)

    def report(y_pred, y_true):
        from sklearn.metrics import confusion_matrix
        TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()

        #https://en.wikipedia.org/wiki/Confusion_matrix

        #senstivity | recall | hit_rate | True_positive_rate
        TPR = TP/(TP+FN)

        #specificity | selectivity | True_negative_rate
        TNR = TN/(TN+FP)

        #precision | Positive_predective_value
        precision = TP/(TP+FP)

        #Miss_rate | False_negative_rate | false_reject_rate
        FNR = FN/(FN+TP)

        #Fall_out | False_positive_rate | false_accept_Rate
        FPR = FP/(FP+TN)

        #accuracy
        ACC = (TP+TN)/(TP+TN+FP+FN)

        #error_rate
        ERROR = (FP+FN)/(TP+TN+FP+FN)

        #F1-score
        F1 = 2*TP / (2*TP + FP + FN)

        #http://publications.idiap.ch/downloads/reports/2005/bengio_2005_icml.pdf
        #half_total_error_rate
        HTER = (FPR+FNR)/2

        print("TPR: ",TPR)
        print("TNR: ",TNR)
        print("precision: ",precision)
        print("FNR: ",FNR)
        print("FPR: ",FPR)
        print("ACC: ",ACC)
        print("ERROR: ",ERROR)
        print("F1: ",F1)
        print("HTER: ",HTER)

class preprocess:

    sampling_freq = 165.0

    def filter_band(data):
        #high pass and low pass filter
        #https://www.daanmichiels.com/blog/2017/10/filtering-eeg-signals-using-scipy/
        #https://youtu.be/uNNNj9AZisM
        """frequency bands are delta band (0–4 Hz), theta band (3.5–7.5 Hz), alpha band (7.5–13 Hz), beta band (13–26 Hz), and gamma band (26–70 Hz)"""

        sampling_freq = 165.0
        # time = np.arange(0.0, duration, 1/sampling_freq)
        low_freq = 0.1 #0.1 Hz
        high_freq = 60.0 #60 Hz

        filter = signal.firwin(400, [low_freq, high_freq], pass_zero=False,fs=sampling_freq) #fs == fixed sampling frequency

        filtered_signal = signal.convolve(data, filter, mode='same')
        return filtered_signal
        # plt.plot(time, filtered_signal)

    def standard_scalar(data):
        scaler = StandardScaler()
        return scaler.fit_transform(data)

class feature:
    #5 features
    sampling_freq = 165.0

    """
        # Define EEG bands
        eeg_bands = {'Delta': (0, 4),
                     'Theta': (4, 8),
                     'Alpha': (8, 12),
                     'Beta': (12, 30),
                     'Gamma': (30, 45)}
    """

    def delta_band(data):
        #https://dsp.stackexchange.com/questions/45345/how-to-correctly-compute-the-eeg-frequency-bands-with-python
        fs = 165  # Sampling rate
        # Get frequencies for amplitudes in Hz
        fft_freq = np.fft.rfftfreq(len(data), 1.0 / fs)
        """Delta Band Values"""
        low_freq = 0
        high_freq = 4

        freqs = fft_freq[np.where((fft_freq >= low_freq) &   #np.where is like asking "tell me where in this array, entries satisfy a given condition".
                           (fft_freq <= high_freq))]
        return freqs

    def theta_band(data):
        #https://dsp.stackexchange.com/questions/45345/how-to-correctly-compute-the-eeg-frequency-bands-with-python
        fs = 165  # Sampling rate
        # Get frequencies for amplitudes in Hz
        fft_freq = np.fft.rfftfreq(len(data), 1.0 / fs)
        """Theta Band Values"""
        low_freq = 4
        high_freq = 8

        freqs = fft_freq[np.where((fft_freq >= low_freq) &   #np.where is like asking "tell me where in this array, entries satisfy a given condition".
                           (fft_freq <= high_freq))]
        return freqs

    def alpha_band(data):
        #https://dsp.stackexchange.com/questions/45345/how-to-correctly-compute-the-eeg-frequency-bands-with-python
        fs = 165  # Sampling rate
        # Get frequencies for amplitudes in Hz
        fft_freq = np.fft.rfftfreq(len(data), 1.0 / fs)
        """Alpha Band Values"""
        low_freq = 8
        high_freq = 12

        freqs = fft_freq[np.where((fft_freq >= low_freq) &   #np.where is like asking "tell me where in this array, entries satisfy a given condition".
                           (fft_freq <= high_freq))]
        return freqs

    def beta_band(data):
        #https://dsp.stackexchange.com/questions/45345/how-to-correctly-compute-the-eeg-frequency-bands-with-python
        fs = 165  # Sampling rate
        # Get frequencies for amplitudes in Hz
        fft_freq = np.fft.rfftfreq(len(data), 1.0 / fs)
        """Beta Band Values"""
        low_freq = 12
        high_freq = 30

        freqs = fft_freq[np.where((fft_freq >= low_freq) &   #np.where is like asking "tell me where in this array, entries satisfy a given condition".
                           (fft_freq <= high_freq))]
        return freqs

    def gamma_band(data):
        #https://dsp.stackexchange.com/questions/45345/how-to-correctly-compute-the-eeg-frequency-bands-with-python
        fs = 165  # Sampling rate
        # Get frequencies for amplitudes in Hz
        fft_freq = np.fft.rfftfreq(len(data), 1.0 / fs)
        """Gamma Band Values"""
        low_freq = 30
        high_freq = 45

        freqs = fft_freq[np.where((fft_freq >= low_freq) &   #np.where is like asking "tell me where in this array, entries satisfy a given condition".
                           (fft_freq <= high_freq))]
        return freqs


    def power_spectral_density(data):
        #https://www.adamsmith.haus/python/answers/how-to-plot-a-power-spectrum-in-python

        fourier_transform = np.fft.rfft(data)

        abs_fourier_transform = np.abs(fourier_transform)

        power_spectrum = np.square(abs_fourier_transform)

        return power_spectrum

    def calcPCA(data):
        pca = PCA(n_components=20) #top 20 features
        X_pca = pca.fit_transform(data)
        return X_pca

    # http://wavelets.pybytes.com/family/coif/

class model:
    def logReg(X_train, X_test, y_train, y_test):
        logReg = LogisticRegression().fit(X_train,y_train)
        y_pred = logReg.predict(X_test)
        #logTest = logReg.score(X_test, y_test)
        return y_pred
    def kMeans(X_train, X_test, y_train, y_test):
        kmeans = KMeans(init="k-means++", n_clusters=2, n_init=10, max_iter=300)
        kmeans.fit(X_train)
        #labels1 = kmeans.labels_
        y_pred = kmeans.predict(X_test)
        alter_y_pred = 1-y_pred
        if base.accuracy(y_test,y_pred) < base.accuracy(y_test, alter_y_pred):
            y_pred = alter_y_pred

        return y_pred
    def SVM(X_train, X_test, y_train, y_test):

        svm = LinearSVC(max_iter=30000).fit(X_train,y_train)
        y_pred = svm.predict(X_test)
        #svmTest = svm.score(X_test, y_test)
        return y_pred

    def KNN(X_train, X_test, y_train, y_test):
        knn = KNeighborsClassifier() #Euclidean distance
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        return y_pred


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

    """"Preprocessing"""
    #Filter data within 0.1 - 60Hz
    filtered_X = base.apply_all(preprocess.filter_band,X)

    #Scalar around means
    scaled_X = preprocess.standard_scalar(filtered_X)


    """Feature Extraction"""
    #For Bands and PSD
    #feature_X = X
    feature_X  = base.apply_all(feature.beta_band, scaled_X)

    #For PCA
    #feature_X = feature.calcPCA(X)


    print(feature_X.shape)


    """Model training"""
    #Divide data into train and test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(feature_X, Y, test_size=0.30, random_state=42)
    #Pass an int for reproducible output across multiple function calls


    # Log Reg
    print("Log Reg: ") #PCA
    print("==========================================================")
    y_pred = model.logReg(X_train, X_test, y_train, y_test)
    print("Accuracy: ",base.accuracy(y_pred, y_test))
    print("Report")
    print("----------------------------------------------------------")
    base.report(y_pred, y_test)
    print("----------------------------------------------------------")

    print("\n")

    #K-Means
    print("KMeans: ") #PCA
    print("==========================================================")
    y_pred =  model.kMeans(X_train, X_test, y_train, y_test)
    print("Accuracy: ",base.accuracy(y_pred, y_test))
    print("Report")
    print("----------------------------------------------------------")
    base.report(y_pred, y_test)
    print("----------------------------------------------------------")

    print("\n")

    # SVM
    print("SVM: ") #PCA
    print("==========================================================")
    y_pred = model.SVM(X_train, X_test, y_train, y_test)
    print("Accuracy: ",base.accuracy(y_pred, y_test))
    print("Report")
    print("----------------------------------------------------------")
    base.report(y_pred, y_test)
    print("----------------------------------------------------------")

    print("\n")

    # SVM
    print("KNN: ") #PCA
    print("==========================================================")
    y_pred = model.KNN(X_train, X_test, y_train, y_test)
    print("Accuracy: ",base.accuracy(y_pred, y_test))
    print("Report")
    print("----------------------------------------------------------")
    base.report(y_pred, y_test)
    print("----------------------------------------------------------")

if __name__ == "__main__":
    main()
