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
from sklearn.model_selection import train_test_split
from scipy import signal
import pickle
import pywt # pip insyall PyWavelets

import warnings
warnings.filterwarnings('ignore')

def segment_data(input_data, seg_time=30):
  # 30 seconds
  segment_points = seg_time * 160 #sampling freq
  splited_data =np.asarray(np.split(input_data.flatten(), segment_points)).T

  return splited_data

def form_data(input_data,attack_data):
  segment_time = 30 #window = 30seconds
  input_ = segment_data(input_data,segment_time)
  attack_ = segment_data(attack_data,segment_time)

  X = np.concatenate((input_,attack_))

  #print(X.shape)

  Y = np.concatenate((np.zeros(input_.shape[0]),np.ones(attack_.shape[0]))) #normal = 0, attack = 1

  return X,Y

def get_multiple(subLow, subHigh):
  input_data = loadmat('Dataset1.mat') #dict_keys(['header', 'version', 'globals', 'Raw_Data', 'Sampling_Rate'])
  input_data = input_data['Raw_Data']
  input_data = input_data[subLow:subHigh, :, :]

  attack_data = loadmat('sampleAttack.mat')#dict_keys(['header', 'version', 'globals', 'attackVectors'])
  attack_data = attack_data['attackVectors']
  attack_data = attack_data[:, subLow:subHigh, :, :]

  return form_data(input_data, attack_data)


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

def get_subject(subject, attack):

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

#5 features
sampling_freq = 165.0

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

def test(sample, name):
  loaded_model = pickle.load(open(name + '.pkl', 'rb'))
  pred = loaded_model.predict(sample)
  return pred

def runSample(data, feat='PCA'):
  # subject 0-105
  # attack -1-5
  # feat: 'PCA', 'alpha', 'beta', 'delta', 'PD', 'coif'
  # if(data==[]):
  #     data = base.get_subject(subject,attack)
  if(feat == 'PCA'):
    pca = pickle.load(open('pca.pkl','rb'))
    # data = base.get_subject(subject,attack).reshape(1, -1)
    data = data.reshape(1,-1)
    sample = pca.transform(data)
  else:
    sc = pickle.load(open('scaler.pkl','rb'))
    # data = base.get_subject(subject,attack)
    filtered = filter_band(data)
    scaled_X = sc.transform(filtered.reshape(1, -1)).reshape(4800, )
    if(feat == 'alpha'):
        sample = alpha_band(scaled_X).reshape(1, -1)
    if(feat == 'delta'):
        sample = delta_band(scaled_X).reshape(1, -1)
    if(feat == 'beta'):
        sample = beta_band(scaled_X).reshape(1, -1)
    if(feat == 'PD'):
        sample = power_spectral_density(scaled_X).reshape(1, -1)
    if(feat == 'coif'):
        sample = coiflets(scaled_X).reshape(1, -1)

  y_pred1 = test(sample, feat+"_logReg")[0]
  y_pred2 = test(sample, feat+"_kmeans")[0]
  y_pred3 = test(sample, feat+"_svm")[0]
  y_pred4 = test(sample, feat+"_knn")[0]

  return y_pred1, y_pred2, y_pred3, y_pred4

def runMultiple(data, feat, Y):
  logReg = []
  kmeans = []
  svm = []
  knn = []
  for i in range(data.shape[0]):
      lr1, km1, svm1, knn1 = runSample(data[i], feat)
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
  logRegResults += "Accuracy: " + str(accuracy(logReg, Y)) + "\n"
  logRegResults += report(logReg, Y)
  logRegResults += "\n----------------------------------------------------------\n"

  kMResults = ""
  kMResults += "\nK-Means: \n"
  kMResults += "==========================================================\n"
  kMResults += "Accuracy: " + str(accuracy(kmeans, Y)) + "\n"
  kMResults += report(kmeans, Y)
  kMResults += "\n----------------------------------------------------------\n"

  svmResults = ""
  svmResults += "\nSVM: \n"
  svmResults += "==========================================================\n"
  svmResults += "Accuracy: " + str(accuracy(svm, Y)) + "\n"
  svmResults += report(svm, Y)
  svmResults += "\n----------------------------------------------------------\n"

  knnResults = ""
  knnResults += "\nKNN: \n"
  knnResults += "==========================================================\n"
  knnResults += "Accuracy: " + str(accuracy(knn, Y)) + "\n"
  knnResults += report(knn, Y)
  knnResults += "\n----------------------------------------------------------\n"

  votingResults = ""
  votingResults += "\nVoting: \n"
  votingResults += "==========================================================\n"
  votingResults += "Accuracy: " + str(accuracy(voting, Y)) + "\n"
  votingResults += report(voting, Y)
  votingResults += "\n----------------------------------------------------------\n"

  return logRegResults + kMResults + svmResults + knnResults + votingResults

def getSubandRun(userID, attackID, feat):
  y_pred = runSample(get_subject(userID,attackID), feat)
  return y_pred

def getMultandRun(userLow, userHigh, feat):
  X, Y = get_multiple(userLow, userHigh)
  results = runMultiple(X, feat, Y)
  return results
