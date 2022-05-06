#!/usr/bin/env python
# coding: utf-8

 #Libraries
from __future__ import print_function, division
from scipy.io import loadmat, savemat
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
from sklearn.model_selection import train_test_split
from scipy import signal
import pickle
import pywt # pip insyall PyWavelets

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Conv2DTranspose, Layer, ReLU
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, save_model, load_model
from keras.optimizers import adam_v2
from keras.optimizers import rmsprop_v2
from keras.metrics import Mean
from tensorflow import GradientTape
from keras import losses
import tensorflow as tf

import warnings
import time

from torch import conv1d
warnings.filterwarnings('ignore')

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

        #print(X.shape)

        Y = np.concatenate((np.zeros(input_.shape[0]),np.ones(attack_.shape[0]))) #normal = 0, attack = 1

        return X,Y

    def get_subject(subject, attack):
        # subject is int 0-105
        # attack = -1 for real data, else 0-7
        # pulls one sample

        if(attack==-1):
            input_data = loadmat('Dataset1.mat') #dict_keys(['header', 'version', 'globals', 'Raw_Data', 'Sampling_Rate'])
            input_data = input_data['Raw_Data']
            data = input_data[subject, 0, :4800]
        elif(attack < 6):
            input_data = loadmat('sampleAttack.mat')#dict_keys(['header', 'version', 'globals', 'attackVectors'])
            input_data = input_data['attackVectors']
            input_data = input_data[attack, :, :, :]
            data = input_data[subject, 0, :4800]
        else:
            input_data = loadmat('GeneratedAttackVector.mat')
            input_data = input_data['attackVectors']
            data = input_data[attack - 6]

        return data

    def get_multiple(subLow, subHigh):
        #pulls range of samples, both real and attack
        input_data = loadmat('Dataset1.mat') #dict_keys(['header', 'version', 'globals', 'Raw_Data', 'Sampling_Rate'])
        input_data = input_data['Raw_Data']
        input_data = input_data[subLow:subHigh, :, :]

        attack_data = loadmat('sampleAttack.mat')#dict_keys(['header', 'version', 'globals', 'attackVectors'])
        attack_data = attack_data['attackVectors']
        attack_data = attack_data[:, subLow:subHigh, :, :]

        return base.form_data(input_data, attack_data)


    def accuracy(y_pred, y_true):
        from sklearn.metrics import accuracy_score
        return accuracy_score(y_true, y_pred)

    def report(y_pred, y_true):
        #aggregate results
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

        results = ""
        results += "TPR: " + str(TPR)
        results += "\nTNR: " + str(TNR)
        results += "\nprecison: " + str(precision)
        results += "\nFNR: " + str(FNR)
        results += "\nFPR: " + str(FPR)
        results += "\nACC: " + str(ACC)
        results += "\nERROR: " + str(ERROR)
        results += "\nF1: " + str(F1)
        results += "\nHTER: " + str(HTER)

        # print("TPR: ",TPR)
        # print("TNR: ",TNR)
        # print("precision: ",precision)
        # print("FNR: ",FNR)
        # print("FPR: ",FPR)
        # print("ACC: ",ACC)
        # print("ERROR: ",ERROR)
        # print("F1: ",F1)
        # print("HTER: ",HTER)
        return results

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
        scaled = scaler.fit_transform(data)
        pickle.dump(scaler, open('scaler.pkl','wb'))
        return scaled

class feature:
    #6 features
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
        fft_vals = np.absolute(np.fft.rfft(data))
        # Get frequencies for amplitudes in Hz
        fft_freq = np.fft.rfftfreq(len(data), 1.0 / fs)
        """Delta Band Values"""
        low_freq = 0
        high_freq = 4

        freqs = fft_vals[np.where((fft_freq >= low_freq) &   #np.where is like asking "tell me where in this array, entries satisfy a given condition".
                           (fft_freq <= high_freq))]
        return freqs

    def theta_band(data):
        #https://dsp.stackexchange.com/questions/45345/how-to-correctly-compute-the-eeg-frequency-bands-with-python
        fs = 165  # Sampling rate
        fft_vals = np.absolute(np.fft.rfft(data))
        # Get frequencies for amplitudes in Hz
        fft_freq = np.fft.rfftfreq(len(data), 1.0 / fs)
        """Theta Band Values"""
        low_freq = 4
        high_freq = 8

        freqs = fft_vals[np.where((fft_freq >= low_freq) &   #np.where is like asking "tell me where in this array, entries satisfy a given condition".
                           (fft_freq <= high_freq))]
        return freqs

    def alpha_band(data):
        #https://dsp.stackexchange.com/questions/45345/how-to-correctly-compute-the-eeg-frequency-bands-with-python
        fs = 165  # Sampling rate
        fft_vals = np.absolute(np.fft.rfft(data))
        # Get frequencies for amplitudes in Hz
        fft_freq = np.fft.rfftfreq(len(data), 1.0 / fs)
        """Alpha Band Values"""
        low_freq = 8
        high_freq = 12

        freqs = fft_vals[np.where((fft_freq >= low_freq) &   #np.where is like asking "tell me where in this array, entries satisfy a given condition".
                           (fft_freq <= high_freq))]
        return freqs

    def beta_band(data):
        #https://dsp.stackexchange.com/questions/45345/how-to-correctly-compute-the-eeg-frequency-bands-with-python
        fs = 165  # Sampling rate
        fft_vals = np.absolute(np.fft.rfft(data))
        # Get frequencies for amplitudes in Hz
        fft_freq = np.fft.rfftfreq(len(data), 1.0 / fs)
        """Beta Band Values"""
        low_freq = 12
        high_freq = 30

        freqs = fft_vals[np.where((fft_freq >= low_freq) &   #np.where is like asking "tell me where in this array, entries satisfy a given condition".
                           (fft_freq <= high_freq))]
        return freqs

    def gamma_band(data):
        #https://dsp.stackexchange.com/questions/45345/how-to-correctly-compute-the-eeg-frequency-bands-with-python
        fs = 165  # Sampling rate
        fft_vals = np.absolute(np.fft.rfft(data))
        # Get frequencies for amplitudes in Hz
        fft_freq = np.fft.rfftfreq(len(data), 1.0 / fs)
        """Gamma Band Values"""
        low_freq = 30
        high_freq = 45

        freqs = fft_vals[np.where((fft_freq >= low_freq) &   #np.where is like asking "tell me where in this array, entries satisfy a given condition".
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
        pickle.dump(pca, open('pca.pkl','wb'))
        return X_pca

    def coiflets(data):
        #https://pywavelets.readthedocs.io/en/0.2.2/ref/dwt-discrete-wavelet-transform.html
        #approximation (cA) and detail (cD) coefficients
        ca, cd = pywt.dwt(data, 'coif1')
        return ca

class model:
    def logReg(X_train, X_test, y_train, y_test):
        strain = time.time()
        logReg = LogisticRegression().fit(X_train,y_train)
        etrain = time.time()
        y_pred = logReg.predict(X_test)
        etest = time.time()

        print("Logistic Regression Training Time: ", etrain-strain)
        print("Logistic Regression Testing Time: ", etest-etrain)

        return logReg, y_pred

    def kMeans(X_train, X_test, y_train, y_test):
        strain = time.time()
        kmeans = KMeans(init="k-means++", n_clusters=2, n_init=10, max_iter=300)
        kmeans.fit(X_train)
        etrain = time.time()
        y_pred = kmeans.predict(X_test)
        alter_y_pred = 1-y_pred
        if base.accuracy(y_test,y_pred) < base.accuracy(y_test, alter_y_pred):
            y_pred = alter_y_pred
        etest = time.time()

        print("K-Means Training Time: ", etrain-strain)
        print("K-Means Testing Time: ", etest-etrain)

        return kmeans, y_pred

    def SVM(X_train, X_test, y_train, y_test):
        strain = time.time()
        svm = LinearSVC(max_iter=30000).fit(X_train,y_train)
        etrain = time.time()
        y_pred = svm.predict(X_test)
        etest = time.time()
        
        print("SVM Training Time: ", etrain-strain)
        print("SVM Testing Time: ", etest-etrain)

        return svm, y_pred

    def KNN(X_train, X_test, y_train, y_test):
        strain = time.time()
        knn = KNeighborsClassifier() #Euclidean distance
        knn.fit(X_train, y_train)
        etrain = time.time()
        y_pred = knn.predict(X_test)
        etest = time.time()
        print("KNN Training Time: ", etrain-strain)
        print("KNN Testing Time: ", etest-etrain)

        return knn, y_pred

class training:
    def trainModels(X, Y, feature, save=False):
        # feat: 'PCA', 'alpha', 'beta', 'delta', 'PD', 'coif'
        # Trains all four models on a given feature on given data
        print("\n***********************************   ", feature, "   ***********************************\n")
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42)

        print("Log Reg: ")
        logReg, y_pred = model.logReg(X_train, X_test, y_train, y_test)
        print("==========================================================")
        print("Accuracy: ",base.accuracy(y_pred, y_test))
        print("Report")
        print("----------------------------------------------------------")
        print(base.report(y_pred, y_test))
        print("----------------------------------------------------------")

        print("\n")

        print("K-Means: ")
        kmeans, y_pred =  model.kMeans(X_train, X_test, y_train, y_test)
        print("==========================================================")
        print("Accuracy: ",base.accuracy(y_pred, y_test))
        print("Report")
        print("----------------------------------------------------------")
        print(base.report(y_pred, y_test))
        print("----------------------------------------------------------")

        print("\n")

        print("SVM: ")
        svm, y_pred = model.SVM(X_train, X_test, y_train, y_test)
        print("==========================================================")
        print("Accuracy: ",base.accuracy(y_pred, y_test))
        print("Report")
        print("----------------------------------------------------------")
        print(base.report(y_pred, y_test))
        print("----------------------------------------------------------")

        print("\n")

        print("KNN: ")
        knn, y_pred = model.KNN(X_train, X_test, y_train, y_test)
        print("==========================================================")
        print("Accuracy: ",base.accuracy(y_pred, y_test))
        print("Report")
        print("----------------------------------------------------------")
        print(base.report(y_pred, y_test))
        print("----------------------------------------------------------")

        if(save):
            pickle.dump(logReg, open(feature + '_logReg.pkl', 'wb'))
            pickle.dump(kmeans, open(feature + '_kmeans.pkl', 'wb'))
            pickle.dump(svm, open(feature + '_svm.pkl', 'wb'))
            pickle.dump(knn, open(feature + '_knn.pkl', 'wb'))

    def trainFeature(feat):
        # feat: 'PCA', 'alpha', 'beta', 'delta', 'PD', 'coif'
        # Trains and saves all 4 models on a single feature using the provided data
        input_data = loadmat('Dataset1.mat') #dict_keys(['__header__', '__version__', '__globals__', 'Raw_Data', 'Sampling_Rate'])
        attack_data = loadmat('sampleAttack.mat')#dict_keys(['__header__', '__version__', '__globals__', 'attackVectors'])

        #loading data
        input_data = input_data['Raw_Data']
        attack_data = attack_data['attackVectors']

        #Combine all data
        X,Y = base.form_data(input_data,attack_data)

        if(feat == 'PCA'):
            X = feature.calcPCA(X)
        else:
            #Filter data within 0.1 - 60Hz
            filtered_X = base.apply_all(preprocess.filter_band,X)
            #Scalar around means
            scaled_X = preprocess.standard_scalar(filtered_X)
            if(feat == 'alpha'):
                X = base.apply_all(feature.alpha_band, scaled_X)
            if(feat == 'delta'):
                X  = base.apply_all(feature.beta_band, scaled_X)
            if(feat == 'beta'):
                X  = base.apply_all(feature.delta_band, scaled_X)
            if(feat == 'PD'):
                X  = base.apply_all(feature.power_spectral_density, scaled_X)
            if(feat == 'coif'):
                X = feature.coiflets(scaled_X)

        training.trainModels(X, Y, feat, True)
    
    def getModels(X, Y, save=False):
        # Trains all models on all features

        """"Preprocessing"""
        #Filter data within 0.1 - 60Hz
        filtered_X = base.apply_all(preprocess.filter_band,X)

        #Scalar around means
        scaled_X = preprocess.standard_scalar(filtered_X)


        """Feature Extraction"""
        #For Bands and PSD
        #feature_X = X

        #For Alpha
        alpha = base.apply_all(feature.alpha_band, scaled_X)
        training.trainModels(alpha, Y, "alpha", save)

        #For Beta
        beta  = base.apply_all(feature.beta_band, scaled_X)
        training.trainModels(beta, Y, "beta", save)

        #For Delta
        delta  = base.apply_all(feature.delta_band, scaled_X)
        training.trainModels(delta, Y, "delta", save)

        #For Gamma
        # gamma = base.apply_all(feature.gamma_band, scaled_X)
        # trainModels(gamma, Y, "gamma", save)

        #For Theta
        # theta  = base.apply_all(feature.theta_band, scaled_X)
        # trainModels(theta, Y, "theta", save)

        #For Power Density
        powerDensity  = base.apply_all(feature.power_spectral_density, scaled_X)
        training.trainModels(powerDensity, Y, "PD", save)

        #For PCA
        pca = feature.calcPCA(X)
        training.trainModels(pca, Y, "PCA", save)

        #For Coiflet Family
        coif = feature.coiflets(scaled_X)
        training.trainModels(coif, Y, "coif", save)

    def test(sample, name):
        # test a single sample on a single model
        loaded_model = pickle.load(open(name + '.pkl', 'rb'))
        pred = loaded_model.predict(sample)
        return pred

    def runSample(data, feat='PCA'):
        # run a single sample and get predicted output on all 4 models
        # subject 0-105
        # attack -1-5
        # feat: 'PCA', 'alpha', 'beta', 'delta', 'PD', 'coif'
        if(feat == 'PCA'):
            pca = pickle.load(open('pca.pkl','rb'))
            data = data.reshape(1,-1)
            sample = pca.transform(data)
        else:
            sc = pickle.load(open('scaler.pkl','rb'))
            # data = base.get_subject(subject,attack)
            filtered = preprocess.filter_band(data)
            scaled_X = sc.transform(filtered.reshape(1, -1)).reshape(4800, )
            if(feat == 'alpha'):
                sample = feature.alpha_band(scaled_X).reshape(1, -1)
            if(feat == 'delta'):
                sample = feature.delta_band(scaled_X).reshape(1, -1)
            if(feat == 'beta'):
                sample = feature.beta_band(scaled_X).reshape(1, -1)
            if(feat == 'PD'):
                sample = feature.power_spectral_density(scaled_X).reshape(1, -1)
            if(feat == 'coif'):
                sample = feature.coiflets(scaled_X).reshape(1, -1)

        y_pred1 = training.test(sample, feat+"_logReg")[0]
        y_pred2 = training.test(sample, feat+"_kmeans")[0]
        y_pred3 = training.test(sample, feat+"_svm")[0]
        y_pred4 = training.test(sample, feat+"_knn")[0]

        return y_pred1, y_pred2, y_pred3, y_pred4
    
    def runMultiple(data, feat, Y):
        # feat: 'PCA', 'alpha', 'beta', 'delta', 'PD', 'coif'
        # tests a set of data and returns aggregate results
        logReg = []
        kmeans = []
        svm = []
        knn = []
        for i in range(data.shape[0]):
            lr1, km1, svm1, knn1 = training.runSample(data[i], feat)
            logReg.append(lr1)
            kmeans.append(km1)
            svm.append(svm1)
            knn.append(knn1)

        #create voting model by taking majority vote
        voting = np.add(logReg, kmeans)
        voting = np.add(voting, svm)
        voting = np.add(voting, knn)

        voting = voting/4
        voting = np.round_(voting, decimals = 0)

        logRegResults = ""
        logRegResults += "Log Reg: \n"
        logRegResults += "==========================================================\n"
        logRegResults += "Accuracy: " + str(base.accuracy(logReg, Y)) + "\n"
        logRegResults += base.report(logReg, Y)
        logRegResults += "\n----------------------------------------------------------\n"

        kMResults = ""
        kMResults += "\nK-Means: \n"
        kMResults += "==========================================================\n"
        kMResults += "Accuracy: " + str(base.accuracy(kmeans, Y)) + "\n"
        kMResults += base.report(kmeans, Y)
        kMResults += "\n----------------------------------------------------------\n"

        svmResults = ""
        svmResults += "\nSVM: \n"
        svmResults += "==========================================================\n"
        svmResults += "Accuracy: " + str(base.accuracy(svm, Y)) + "\n"
        svmResults += base.report(svm, Y)
        svmResults += "\n----------------------------------------------------------\n"

        knnResults = ""
        knnResults += "\nKNN: \n"
        knnResults += "==========================================================\n"
        knnResults += "Accuracy: " + str(base.accuracy(knn, Y)) + "\n"
        knnResults += base.report(knn, Y)
        knnResults += "\n----------------------------------------------------------\n"

        votingResults = ""
        votingResults += "\nVoting: \n"
        votingResults += "==========================================================\n"
        votingResults += "Accuracy: " + str(base.accuracy(voting, Y)) + "\n"
        votingResults += base.report(voting, Y)
        votingResults += "\n----------------------------------------------------------\n"

        return logRegResults + kMResults + svmResults + knnResults + votingResults

    def getSubandRun(userID, attackID, feat):
        # single function to test a sample (for app)
        y_pred = training.runSample(base.get_subject(userID,attackID), feat)
        return y_pred

    def getMultandRun(userLow, userHigh, feat):
        # single function to test a range of data (for app)
        X, Y = base.get_multiple(userLow, userHigh)
        results = training.runMultiple(X, feat, Y)
        return results

def main():


    #Read data
    input_data = loadmat('Dataset1.mat') #dict_keys(['__header__', '__version__', '__globals__', 'Raw_Data', 'Sampling_Rate'])
    attack_data = loadmat('sampleAttack.mat')#dict_keys(['__header__', '__version__', '__globals__', 'attackVectors'])

    #loading data
    input_data = input_data['Raw_Data']
    attack_data = attack_data['attackVectors']

    #Combine all data
    X,Y = base.form_data(input_data,attack_data)

    """Model training"""

    # training.getModels(X, Y, True)

    # test examples
    print(training.getSubandRun(10, -1, "alpha"))
    print(training.getMultandRun(10, 20, "PD"))

    """GAN training"""

    start = time.time()
    gan = GAN()
    gan.train(epochs=128*2, adv_train=X, batch_size=5, sample_interval=200)
    end = time.time()
    print("Time elapsed for generating GAN: ", (end - start))

    """VAE training"""

    start = time.time()
    encoder = buildEncoder()
    decoder = buildDecoder(latent_dim = 2)
    # need to split train and test data: 70, 30
    x_train = np.concatenate((X[0:890], X[1272:2608]))
    y_train = np.concatenate((Y[0:890], Y[1272:2608]))
    x_test = np.concatenate((X[890:1272], X[2608:]))
    trainVAE(encoder, decoder, x_train, x_test, y_train)
    end = time.time()
    print("Time elapsed for generating VAE: ", (end - start))

    """Generate Signal with GAN and VAE"""

    gan = load_model("./GANSavedModel")
    vaeencoder = load_model("./VAEEncoderSavedModel")
    vaedecoder = load_model("./VAEDecoderSavedModel")
    start = time.time()
    noise = np.random.normal(0, 1, (1, 10000))
    attack_vector1 = gan.predict(noise)
    end = time.time()
    print("Time elapsed for generating attack vector with GAN: ", (end - start))

    #Generation range near x: see attachment
    start = time.time()
    attack_vector2 = vaedecoder.predict(np.array([[-10, -10]]))
    end = time.time()
    print("Time elapsed for generating attack vector with VAE: ", (end - start))

    attack_vector1 = attack_vector1[0,:,0]
    attack_vector2 = attack_vector2[0,:,0]
    obj_array = np.zeros((2,4800))

    for av in range(4800):
        obj_array[0][av] = attack_vector1[av]
        obj_array[1][av] = attack_vector2[av]
    savemat("./GeneratedAttackVector.mat", mdict={'attackVectors': obj_array})
    attack_data2 = loadmat('GeneratedAttackVector.mat')
    attack_data2 = attack_data2['attackVectors']
    print(attack_data2.shape)
    
# Adapted from https://towardsdatascience.com/gan-by-example-using-keras-on-tensorflow-backend-1a6d515a60d0
# and https://github.com/eriklindernoren/Keras-GAN/blob/master/gan/gan.py
class GAN():
    def __init__(self):
        self.img_rows = 1
        self.img_cols = 4800
        self.channels = 1
        self.img_shape = ( self.img_cols, self.channels)
        self.latent_dim = 10000

        #optimizer = rmsprop_v2.RMSProp(0.0002)
        optimizer = adam_v2.Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)


    def build_generator(self):

        model = Sequential()

        model.add(Dense(16, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(32))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(64))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        #model.add(BatchNormalization(momentum=0.9))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        #model.add(BatchNormalization(momentum=0.9))
        model.add(Dropout(0.4))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        #model.add(Dropout(0.4))
        #model.add(BatchNormalization(momentum=0.9))
        model.add(Dropout(0.4))
        model.add(Dense(np.prod(self.img_shape), activation='softsign'))
        model.add(Reshape(self.img_shape))

        #model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))
        #model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, adv_train, batch_size=5, sample_interval=50):

        # Load the dataset
        X_train = adv_train
        #print(X_train.shape)
        #print(X_train[0][0])
        #print(type(X_train[0][0]))

        # Rescale -1 to 1
        max = np.absolute(np.max(X_train))
        min = np.absolute(np.min(X_train))
        #print(max)
        #print(min)
        max = np.max([np.absolute(np.max(X_train)), np.absolute(np.min(X_train))])
        X_train = X_train / (max/2)
        #print(X_train.shape)
        #print(X_train[0][0])
        X_train = np.expand_dims(X_train, axis=2)
        #print(X_train.shape)

        # Adversarial ground truths
        valid = np.zeros((batch_size, 1))
        fake = np.ones((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]
            #print(idx)

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generate a batch of new images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            #print("real: %s",d_loss_real)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            #print("fake: %s", d_loss_fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            #print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

        #save_model(self.generator, "./GANSavedModel", overwrite= True)

# Adapted from https://keras.io/examples/generative/vae/
class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = Mean(name="total_loss")
        self.reconstruction_loss_tracker = Mean(name="reconstruction_loss")
        self.kl_loss_tracker = Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)

            reconstruction = self.decoder(z)

            reconstruction_loss = tf.reduce_mean(
                #tf.reduce_sum(
                    losses.binary_crossentropy(data, reconstruction)#, axis=(1, 2)
                #)
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def plot_label_clusters(vae, data, labels):
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = vae.encoder.predict(data)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()

def buildEncoder():
    latent_dim = 2

    encoder_inputs = Input(shape=(4800, 1))
    #x = Conv(32, 2, activation="relu", strides=2, padding="same")(encoder_inputs)
    #x = conv1d(64, 2, activation="relu", strides=2, padding="same")(x)
    x = Dense(32, activation="relu")(encoder_inputs)#ReLU()(encoder_inputs)
    x = Dense(64, activation="relu")(x)#ReLU()(x)
    x = Flatten()(x)
    x = Dense(16, activation="relu")(x)
    z_mean = Dense(latent_dim, name="z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    #encoder.summary()
    return encoder

def buildDecoder(latent_dim):
    latent_inputs = Input(shape=(latent_dim,))
    x = Flatten()(latent_inputs)
    x = Dense(64, activation="relu")(x)#Conv2DTranspose(64, 2, activation="relu", strides=2, padding="same")(x)
    x = Dense(32, activation="relu")(x)#Conv2DTranspose(32, 2, activation="relu", strides=2, padding="same")(x)
    x = Dense(np.prod((4800,1)), activation='sigmoid')(x)
    decoder_outputs = Reshape((4800,1))(x)#Dense(1, activation="sigmoid")(x)
    decoder = Model(latent_inputs, decoder_outputs, name="decoder")
    #decoder.summary()
    return decoder

def trainVAE(encoder, decoder, x_train, x_test, y_train):
    data_resized = np.concatenate([x_train, x_test], axis=0)
    data_resized = np.expand_dims(data_resized, -1).astype("float32") / 65535
    print("SHAPE OF DATA: %s", data_resized.shape)
    vae = VAE(encoder, decoder)
    vae.compile(optimizer=adam_v2.Adam())
    vae.fit(data_resized, epochs=1, batch_size=5)
    #plot_label_clusters(vae, x_train, y_train)
    #save_model(vae.encoder, "./VAEEncoderSavedModel", overwrite=True)
    #save_model(vae.decoder, "./VAEDecoderSavedModel", overwrite=True)
    # training.getModels(X, Y)

    """Testing on one sample"""    

    # sample from provided data
    # print(training.runSample(base.get_subject(0,1), 'alpha'))
    # sample from generated data
    # print(training.runSample(base.get_subject(0,6), 'alpha'))

    print(training.getSubandRun(0, -1, 'alpha'))

    training.runMultiple(X, "coif", Y)



if __name__ == "__main__":
    main()